# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np
import time
import os
import json
import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
import t5

#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt, cond=None):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(x, t_time, latent_z, cond=cond)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x


def sample_from_model_classifier_free_guidance(coefficients, generator, n_time, x_init, T, opt, text_encoder, cond=None, guidance_scale=0):
    x = x_init
    null = text_encoder([""] * len(x_init), return_only_pooled=False)
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            
            x_0_uncond = generator(x, t_time, latent_z, cond=null)
            x_0_cond = generator(x, t_time, latent_z, cond=cond)

            eps_uncond = (x - torch.sqrt(coefficients.alphas_cumprod[i]) * x_0_uncond) / torch.sqrt(1 - coefficients.alphas_cumprod[i])
            eps_cond = (x - torch.sqrt(coefficients.alphas_cumprod[i]) * x_0_cond) / torch.sqrt(1 - coefficients.alphas_cumprod[i])
            
            # eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            eps = eps_uncond * (1 - guidance_scale) + eps_cond * guidance_scale
            x_0 = (1/torch.sqrt(coefficients.alphas_cumprod[i])) * (x - torch.sqrt(1 - coefficients.alphas_cumprod[i]) * eps)
           

            # Dynamic thresholding
            q = args.dynamic_thresholding_percentile
            print("Before", x_0.min(), x_0.max())
            if q:
                shape = x_0.shape
                x_0_v = x_0.view(shape[0], -1)
                d = torch.quantile(torch.abs(x_0_v), q, dim=1, keepdim=True)
                d.clamp_(min=1)
                x_0_v = x_0_v.clamp(-d, d) / d
                x_0 = x_0_v.view(shape)
            print("After", x_0.min(), x_0.max())
            
            x_new = sample_posterior(coefficients, x_0, x, t)
            
            # Dynamic thresholding
            # q = args.dynamic_thresholding_percentile
            # shape = x_new.shape
            # x_new_v = x_new.view(shape[0], -1)
            # d = torch.quantile(torch.abs(x_new_v), q, dim=1, keepdim=True)
            # d = torch.maximum(d, torch.ones_like(d))
            # d.clamp_(min = 1.)
            # x_new_v = torch.clamp(x_new_v, -d, d) / d
            # x_new = x_new_v.view(shape)
            x = x_new.detach()
        
    return x


#%%
def sample_and_test(args):
    torch.manual_seed(args.seed)
    device = 'cuda:0'
    text_encoder = t5.T5Encoder(name=args.text_encoder, masked_mean=args.masked_mean).to(device)
    args.cond_size = text_encoder.output_size
    # cond = text_encoder([str(yi%10) for yi in range(args.batch_size)])

    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    
    netG = NCSNpp(args).to(device)
    ckpt = torch.load('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    
    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)
        
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.compute_fid:
        from torch.nn.functional import adaptive_avg_pool2d
        from pytorch_fid.fid_score import calculate_activation_statistics, calculate_fid_given_paths, ImagePathDataset, compute_statistics_of_path, calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3

        texts = open(args.cond_text).readlines()
        #iters_needed = len(texts) // args.batch_size
        #texts = list(map(lambda s:s.strip(), texts))
        #ntimes = max(30000 // len(texts), 1)
        #texts = texts * ntimes
        print("Text size:", len(texts))
        #print("Iters:", iters_needed)
        i = 0
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inceptionv3 = InceptionV3([block_idx]).to(device)

        if not args.real_img_dir.endswith("npz"):
            real_mu, real_sigma = compute_statistics_of_path(
                args.real_img_dir, inceptionv3, args.batch_size, dims, device, 
                resize=args.image_size,
            )
            np.savez("inception_statistics.npz", mu=real_mu, sigma=real_sigma)
        else:
            stats = np.load(args.real_img_dir)
            real_mu = stats['mu']
            real_sigma = stats['sigma']

        fake_features = []
        for b in range(0, len(texts), args.batch_size):
            text = texts[b:b+args.batch_size]
            with torch.no_grad():
                cond = text_encoder(text, return_only_pooled=False)
                bs = len(text)
                t0 = time.time()
                x_t_1 = torch.randn(bs, args.num_channels,args.image_size, args.image_size).to(device)
                if args.guidance_scale:
                    fake_sample = sample_from_model_classifier_free_guidance(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args, text_encoder, cond=cond, guidance_scale=args.guidance_scale)
                else:
                    fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args, cond=cond)
                fake_sample = to_range_0_1(fake_sample)
                """
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                """
                with torch.no_grad():
                    pred = inceptionv3(fake_sample)[0]
                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                fake_features.append(pred)
                if i % 10 == 0:
                    print('generating batch ', i, time.time() - t0)
                """
                if i % 10 == 0:
                    ff = np.concatenate(fake_features)
                    fake_mu = np.mean(ff, axis=0)
                    fake_sigma = np.cov(ff, rowvar=False)
                    fid =  calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
                    print("FID", fid)
                """
            i += 1

        fake_features = np.concatenate(fake_features)
        fake_mu = np.mean(fake_features, axis=0)
        fake_sigma = np.cov(fake_features, rowvar=False)
        fid =  calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        dest = './saved_info/dd_gan/{}/{}/fid_{}.json'.format(args.dataset, args.exp, args.epoch_id)
        results = {
            "fid": fid,
        }
        results.update(vars(args))
        with open(dest, "w") as fd:
            json.dump(results, fd)
        print('FID = {}'.format(fid))
    else:
        cond = text_encoder([args.cond_text] * args.batch_size, return_only_pooled=False)
        x_t_1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
        if args.guidance_scale:
            fake_sample = sample_from_model_classifier_free_guidance(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args, text_encoder, cond=cond, guidance_scale=args.guidance_scale)
        else:
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args, cond=cond)
        fake_sample = to_range_0_1(fake_sample)
        torchvision.utils.save_image(fake_sample, './samples_{}.jpg'.format(args.dataset))

    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--guidance_scale', type=float,default=0)
    parser.add_argument('--dynamic_thresholding_percentile', type=float,default=0)
    parser.add_argument('--cond_text', type=str,default="0")

    parser.add_argument('--cross_attention', action='store_true',default=False)

    
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    parser.add_argument('--text_encoder', type=str, default="google/t5-v1_1-base")
    parser.add_argument('--masked_mean', action='store_true',default=False)
        



   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                
