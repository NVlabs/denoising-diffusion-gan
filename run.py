import os
from clize import run
from glob import glob
from subprocess import call

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
        }
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

models = [
    ddgan_cc12m_v2,
    ddgan_cc12m_v6,
    ddgan_cc12m_v7,
    ddgan_cc12m_v8,
    ddgan_cc12m_v9,
    ddgan_cc12m_v11,

]
def get_model(model_name):
    for model in models:
        if model.__name__ == model_name:
            return model()


def test(model_name, *, cond_text="", batch_size:int=None, epoch:int=None, guidance_scale:float=0, fid=False, real_img_dir=""):

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

    if fid:
        args['compute_fid'] = ''
        args['real_img_dir'] = real_img_dir 
    cmd = "python test_ddgan.py " + " ".join(f"--{k} {v}" for k, v in args.items() if v is not None)
    print(cmd)
    call(cmd, shell=True)

run([test])