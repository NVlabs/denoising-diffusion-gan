import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

transformers.logging.set_verbosity_error()

def exists(val):
    return val is not None

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

class T5Encoder(torch.nn.Module):

    def __init__(self, name=DEFAULT_T5_NAME, max_length=MAX_LENGTH, padding='longest', masked_mean=False):
        super().__init__()
        self.name = name
        self.t5, self.tokenizer = get_model_and_tokenizer(name)
        self.max_length = max_length
        self.output_size = get_encoded_dim(name)
        self.padding = padding
        self.masked_mean = masked_mean

    def forward(self, x, return_only_pooled=True):
        encoded = self.tokenizer.batch_encode_plus(
            x,
            return_tensors = "pt",
            padding = self.padding,
            max_length = self.max_length,
            truncation = True
        )
        device = next(self.t5.parameters()).device
        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device).bool()
        output = self.t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()
        # return encoded_text[:, 0]
        # print(input_ids)
        # print(attn_mask)
        #if self.masked_mean:
        pooled =  masked_mean(encoded_text, dim=1, mask=attn_mask)
        if return_only_pooled:
            return pooled
        else:
            return pooled, encoded_text, attn_mask
        #else:
        #    return encoded_text.mean(dim=1)


from einops import rearrange
def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)
