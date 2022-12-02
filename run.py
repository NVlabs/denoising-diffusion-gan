import os
from glob import glob
from subprocess import call
import json
def base():
    return {
        "slurm":{
            "t": 360,
            "N": 2,
            "n": 8,
        },
        "model":{
            "dataset" :"wds",
            "dataset_root": "/p/scratch/ccstdl/cherti1/CC12M/{00000..01099}.tar",
            "image_size": 256,
            "num_channels": 3,
            "num_channels_dae": 128,
            "ch_mult": "1 1 2 2 4 4",
            "num_timesteps": 4,
            "num_res_blocks": 2,
            "batch_size": 8,
            "num_epoch": 1000,
            "ngf": 64,
            "embedding_type": "positional",
            "use_ema": "",
            "ema_decay": 0.999,
            "r1_gamma": 1.0,
            "z_emb_dim": 256,
            "lr_d": 1e-4,
            "lr_g": 1.6e-4,
            "lazy_reg": 10,
            "save_content": "",
            "save_ckpt_every": 1,
            "masked_mean": "",
            "resume": "",
        },
    }
def ddgan_cc12m_v2():
    cfg =  base()
    cfg['slurm']['N'] = 2
    cfg['slurm']['n'] = 8
    return cfg

def ddgan_cc12m_v6():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"
    return cfg

def ddgan_cc12m_v7():
    cfg = base()
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['slurm']['N'] = 2
    cfg['slurm']['n'] = 8
    return cfg

def ddgan_cc12m_v8():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    return cfg

def ddgan_cc12m_v9():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['model']['num_channels_dae'] = 320
    cfg['model']['image_size'] = 64
    cfg['model']['batch_size'] = 1
    return cfg

def ddgan_cc12m_v11():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['model']['cross_attention'] = ""
    return cfg

def ddgan_cc12m_v12():
    cfg = ddgan_cc12m_v11()
    cfg['model']['text_encoder'] = "google/t5-v1_1-xl"    
    cfg['model']['preprocessing'] = 'random_resized_crop_v1'
    return cfg

def ddgan_cc12m_v13():
    cfg = ddgan_cc12m_v12()
    cfg['model']['discr_type'] = "large_cond_attn"
    return cfg

def ddgan_cc12m_v14():
    cfg = ddgan_cc12m_v12()
    cfg['model']['num_channels_dae'] = 192
    return cfg

def ddgan_cc12m_v15():
    cfg = ddgan_cc12m_v11()
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg

def ddgan_cifar10_cond17():
    cfg = base()
    cfg['model']['image_size'] = 32    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['model']['ch_mult'] = "1 2 2 2"
    cfg['model']['cross_attention'] = ""
    cfg['model']['dataset'] = "cifar10"
    cfg['model']['n_mlp'] = 4
    return cfg

def ddgan_cifar10_cond18():
    cfg = ddgan_cifar10_cond17()
    cfg['model']['text_encoder'] = "google/t5-v1_1-xl"    
    return cfg

def ddgan_cifar10_cond19():
    cfg = ddgan_cifar10_cond17()
    cfg['model']['discr_type'] = 'small_cond_attn'
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg

def ddgan_laion_aesthetic_v1():
    cfg = ddgan_cc12m_v11()
    cfg['model']['dataset_root'] = '"/p/scratch/ccstdl/cherti1/LAION-aesthetic/output/{00000..05038}.tar"'
    return cfg

def ddgan_laion_aesthetic_v2():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['discr_type'] = "large_cond_attn"
    return cfg

def ddgan_laion_aesthetic_v3():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['text_encoder'] = "google/t5-v1_1-xl" 
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg

def ddgan_laion_aesthetic_v4():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['text_encoder'] = "openclip/ViT-L-14-336/openai" 
    return cfg


def ddgan_laion_aesthetic_v5():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg



def ddgan_laion2b_v1():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    cfg['model']['num_channels_dae'] = 224
    cfg['model']['batch_size'] = 2
    cfg['model']['discr_type'] = "large_cond_attn"
    cfg['model']['preprocessing'] = 'random_resized_crop_v1'
    return cfg

def ddgan_laion_aesthetic_v6():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['no_lr_decay'] = ''
    return cfg



def ddgan_laion_aesthetic_v7():
    cfg = ddgan_laion_aesthetic_v6()
    cfg['model']['r1_gamma'] = 5
    return cfg


def ddgan_laion_aesthetic_v8():
    cfg = ddgan_laion_aesthetic_v6()
    cfg['model']['num_timesteps'] = 8
    return cfg

def ddgan_laion_aesthetic_v9():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['num_channels_dae'] = 384
    return cfg

def ddgan_sd_v1():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v2():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v3():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v4():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v5():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['num_timesteps'] = 8
    return cfg
def ddgan_sd_v6():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['num_channels_dae'] = 192
    return cfg
def ddgan_sd_v7():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v8():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['image_size'] = 512
    return cfg
def ddgan_laion_aesthetic_v12():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_laion_aesthetic_v13():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-H-14/laion2b_s32b_b79k" 
    return cfg

def ddgan_laion_aesthetic_v14():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-H-14/laion2b_s32b_b79k" 
    return cfg
def ddgan_sd_v9():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-H-14/laion2b_s32b_b79k" 
    return cfg

models = [
    ddgan_cifar10_cond17, # cifar10, cross attn for discr
    ddgan_cifar10_cond18, # cifar10, xl encoder
    ddgan_cifar10_cond19, # cifar10, xl encoder

    ddgan_cc12m_v2, # baseline (no large text encoder, no classifier guidance)
    ddgan_cc12m_v6, # like v2 but using large T5 text encoder
    ddgan_cc12m_v7, # like v2 but with classifier guidance
    ddgan_cc12m_v8, # like v6 but classifier guidance
    ddgan_cc12m_v9, # ~1B model but 64x64 resolution
    ddgan_cc12m_v11, # large text encoder + cross attention + classifier free guidance
    ddgan_cc12m_v12, # T5-XL + cross attention + classifier free guidance + random_resized_crop_v1
    ddgan_cc12m_v13, # T5-XL + cross attention + classifier free guidance + random_resized_crop_v1 + cond attn
    ddgan_cc12m_v14, # T5-XL + cross attention + classifier free guidance + random_resized_crop_v1 + 300M model
    ddgan_cc12m_v15, # fine-tune v11 with --mismatch_loss and --grad_penalty_cond
    ddgan_laion_aesthetic_v1, # like ddgan_cc12m_v11 but fine-tuned on laion aesthetic
    ddgan_laion_aesthetic_v2, # like ddgan_laion_aesthetic_v1 but trained from scratch with the new cross attn discr
    ddgan_laion_aesthetic_v3, # like ddgan_laion_aesthetic_v1 but trained from scratch with T5-XL (continue from 23aug with mismatch and grad penalty and random_resized_crop_v1)
    ddgan_laion_aesthetic_v4, # like ddgan_laion_aesthetic_v1 but trained from scratch with OpenAI's ClipEncoder 
    ddgan_laion_aesthetic_v5, # fine-tune ddgan_laion_aesthetic_v1 with mismatch and cond grad penalty  losses
    ddgan_laion_aesthetic_v6, # like v3 but without lr decay
    ddgan_laion_aesthetic_v7, # like v6 but  with r1 gamma of 5 instead of 1, trying to constrain the discr more.
    ddgan_laion_aesthetic_v8, # like v6 but with 8 timesteps
    ddgan_laion_aesthetic_v9,
    ddgan_laion_aesthetic_v12,
    ddgan_laion_aesthetic_v13,
    ddgan_laion_aesthetic_v14,
    ddgan_laion2b_v1,
    ddgan_sd_v1,
    ddgan_sd_v2,
    ddgan_sd_v3,
    ddgan_sd_v4,
    ddgan_sd_v5,
    ddgan_sd_v6,
    ddgan_sd_v7,
    ddgan_sd_v8,
    ddgan_sd_v9,
]

def get_model(model_name):
    for model in models:
        if model.__name__ == model_name:
            return model()


def test(model_name, *, cond_text="", batch_size:int=None, epoch:int=None, guidance_scale:float=0, fid=False, real_img_dir="", q=0.0, seed=0, nb_images_for_fid=0, scale_factor_h=1, scale_factor_w=1, compute_clip_score=False, eval_name="", scale_method="convolutional"):

    cfg = get_model(model_name)
    model = cfg['model']
    if epoch is None:
        paths = glob('./saved_info/dd_gan/{}/{}/netG_*.pth'.format(model["dataset"], model_name))
        epoch = max(
            [int(os.path.basename(path).replace(".pth", "").split("_")[1]) for path in paths]
        )
    args = {}
    args['exp'] = model_name
    args['image_size'] = model['image_size']
    args['seed'] = seed
    args['num_channels'] = model['num_channels']
    args['dataset'] = model['dataset']
    args['num_channels_dae'] = model['num_channels_dae']
    args['ch_mult'] = model['ch_mult']
    args['num_timesteps'] = model['num_timesteps']
    args['num_res_blocks'] = model['num_res_blocks']
    args['batch_size'] = model['batch_size'] if batch_size is None else batch_size
    args['epoch'] = epoch
    args['cond_text'] = f'"{cond_text}"'
    args['text_encoder'] = model.get("text_encoder")
    args['cross_attention'] = model.get("cross_attention")
    args['guidance_scale'] = guidance_scale
    args['masked_mean'] = model.get("masked_mean")
    args['dynamic_thresholding_quantile'] = q
    args['scale_factor_h'] = scale_factor_h
    args['scale_factor_w'] = scale_factor_w
    args['n_mlp'] = model.get("n_mlp")
    args['scale_method'] = scale_method
    if fid:
        args['compute_fid'] = ''
        args['real_img_dir'] = real_img_dir 
        args['nb_images_for_fid'] = nb_images_for_fid
    if compute_clip_score:
        args['compute_clip_score'] = ""
    if eval_name:
        args["eval_name"] = eval_name
    cmd = "python -u test_ddgan.py " + " ".join(f"--{k} {v}" for k, v in args.items() if v is not None)
    print(cmd)
    call(cmd, shell=True)

def eval_results(model_name):
    import pandas as pd
    rows = []
    cfg = get_model(model_name)
    model = cfg['model']
    paths = glob('./saved_info/dd_gan/{}/{}/fid*.json'.format(model["dataset"], model_name))
    for path in paths:
        with open(path, "r") as fd:
            data = json.load(fd)
        row = {}
        row['fid'] = data['fid']
        row['epoch'] = data['epoch_id']
        rows.append(row)
    out = './saved_info/dd_gan/{}/{}/fid.csv'.format(model["dataset"], model_name)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

if __name__ == "__main__":
    from clize import run
    run([test, eval_results])
