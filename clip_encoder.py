import torch
import torch.nn as nn
import open_clip
from einops import rearrange


def exists(val):
    return val is not None

class CLIPEncoder(nn.Module):

    def __init__(self, model, pretrained):
        super().__init__()
        self.model = model
        self.pretrained = pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.output_size = self.model.transformer.width

    def forward(self, texts, return_only_pooled=True):
        device = next(self.parameters()).device
        toks = open_clip.tokenize(texts).to(device)
        x = self.model.token_embedding(toks)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        mask = (toks!=0)
        pooled = x[torch.arange(x.shape[0]), toks.argmax(dim=-1)] @ self.model.text_projection
        if return_only_pooled:
            return pooled
        else:
            return pooled, x, mask




class CLIPImageEncoder(nn.Module):

    def __init__(self, model_type="ViT-B/32"):
        super().__init__()
        import clip
        self.model, preprocess = clip.load(model_type, device="cpu", jit=False)
        CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.output_size = 512

    def forward_image(self, x):
        x = torch.nn.functional.interpolate(x, mode='bicubic', size=(224, 224))
        x = (x-self.mean)/self.std
        return self.model.encode_image(x)

    def forward_text(self, texts):
        import clip
        toks = clip.tokenize(texts, truncate=True).to(self.mean.device)
        return self.model.encode_text(toks)




