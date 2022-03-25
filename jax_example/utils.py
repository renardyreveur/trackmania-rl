from jax import random


def init_params(key, size):
    key, subkey = random.split(key)
    return key, random.normal(subkey, shape=size)


# Encoder layers
def enc_params(key, feat_dim, qkv_dim, heads):
    key, proj1 = init_params(key, (feat_dim, qkv_dim))
    key, proj2 = init_params(key, (qkv_dim*heads, qkv_dim))
    key, proj3 = init_params(key, (qkv_dim, qkv_dim))
    return key,  {"proj_1": proj1, "proj_2": proj2, "proj_3": proj3}


# Decoder layers
def dec_params(key, feat_dim, qkv_dim, heads):
    key, dproj1 = init_params(key, (feat_dim, qkv_dim))
    key, dproj2 = init_params(key, (qkv_dim*heads, qkv_dim))
    key, dproj3 = init_params(key, (qkv_dim, qkv_dim))
    key, dproj4 = init_params(key, (qkv_dim, feat_dim))
    return key,  {"dproj_1": dproj1, "dproj_2": dproj2, "dproj_3": dproj3, "dproj_4": dproj4}


# Attention layers
def attn_params(key, qkv_dim):
    key, w_query = init_params(key, (qkv_dim, qkv_dim))
    key, w_key = init_params(key, (qkv_dim, qkv_dim))
    key, w_val = init_params(key, (qkv_dim, qkv_dim))
    return key,  {"w_query": w_query, "w_key": w_key, "w_val": w_val}


# Context params
def ctx_params(key, in_dim, feat_dim):
    key, key_proj = init_params(key, (in_dim, feat_dim))
    key, val_proj = init_params(key, (in_dim, feat_dim))
    return key,  {"key_proj": key_proj, "val_proj": val_proj}
