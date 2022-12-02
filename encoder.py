import t5
import clip_encoder

def build_encoder(name, **kwargs):
    if name.startswith("google"):
        return t5.T5Encoder(name=name, **kwargs)
    elif name.startswith("openclip"):
        _, model, pretrained = name.split("/")
        return clip_encoder.CLIPEncoder(model, pretrained)