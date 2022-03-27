import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")

import numpy as np

import jax.numpy as jnp

from processes.agent.policies.neural_agent.act_layers import conv2d, linear, layer_norm, gelu
from processes.agent.policies.neural_agent.attention import attention


def get_params(params, idx):
    return None if params is None else params[idx]


def downsample(in_x, in_dim, out_dim, params=None):
    x, p0 = layer_norm(in_x, 1, get_params(params, 0))
    x, p1 = conv2d(x, in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2), params=get_params(params, 1))
    return x, [p0, p1]


def convnext_block(in_x, dim, params=None):
    x, p0 = conv2d(in_x, dim, dim, kernel_size=(7, 7), padding=((3, 3), (3, 3)), groups=dim,
                   params=get_params(params, 0))
    x = jnp.transpose(x, (0, 2, 3, 1))
    x, p1 = layer_norm(x, 3, get_params(params, 1))
    x, p2 = linear(x, dim * 2, get_params(params, 2))
    x = gelu(x)
    x, p3 = linear(x, dim, get_params(params, 3))
    x = jnp.transpose(x, (0, 3, 1, 2))
    # TODO: Add DropPath
    return x, [p0, p1, p2, p3]


def frame_feature_extractor(in_x, dims, params=None):
    new_params = []
    # Stem
    x, p0 = conv2d(in_x, 1, 16, kernel_size=(4, 4), stride=(4, 4), params=get_params(params, 0))
    x, p1 = layer_norm(x, 1, get_params(params, 1))
    new_params.append(p0)
    new_params.append(p1)

    for i in range(len(dims)):
        x, p2 = convnext_block(x, dim=dims[i], params=get_params(params, 2 + i * 2))
        new_params.append(p2)
        if i != len(dims) - 1:
            x, p3 = downsample(x, in_dim=dims[i], out_dim=dims[i + 1], params=get_params(params, 3 + i * 2))
            new_params.append(p3)

    features, p9 = layer_norm(jnp.mean(x, axis=(-2, -1)), dim=1, params=get_params(params, 9))
    new_params.append(p9)
    return features, new_params


def neural_model(in_x, feat_dims, heads, params=None):
    feature_seq = []
    new_params = []
    for frame_num in range(in_x.shape[1]):
        x, p0 = frame_feature_extractor(jnp.expand_dims(in_x[:, frame_num, :, :], 1), dims=feat_dims,
                                        params=get_params(params, 0))
        feature_seq.append(x)
        new_params.append(p0)

    feature_seq = jnp.stack(feature_seq, axis=1)

    # Multi-head attention, concat results
    attn_heads = []
    for i in range(heads):
        a_out, p1 = attention(feature_seq, dim=feat_dims[-1], params=get_params(params, i + 1))
        attn_heads.append(a_out)
        new_params.append(p1)

    mh_out = jnp.concatenate(attn_heads, axis=-1)

    # Residual skip-connection and layer norm
    reproj, p2 = linear(mh_out, feat_dims[-1], params=get_params(params, heads + 1))
    new_params.append(p2)
    reproj = gelu(reproj + feature_seq)
    nmres, p3 = layer_norm(reproj, dim=2, params=get_params(params, heads + 2))
    new_params.append(p3)

    # Flatten
    nmres = jnp.reshape(nmres, (nmres.shape[0], -1))

    # Output projection (2 branches)
    output = []
    for i, _ in enumerate(['mean', 'std']):
        mustd, p4 = linear(nmres, feat_dims[-1], params=get_params(params, heads + 3 + i * 3))
        mustd = gelu(mustd)
        mustd, p5 = linear(mustd, feat_dims[-1] // 2, params=get_params(params, heads + 4 + i * 3))
        mustd = gelu(mustd)
        mustd, p6 = linear(mustd, 3, params=get_params(params, heads + 5 + i * 3))
        output.append(mustd)
        new_params.append(p4)
        new_params.append(p5)
        new_params.append(p6)

    return output, new_params


def neural_policy(frames, **kwargs):
    frames = jnp.asarray(np.concatenate(frames, 1))
    output, _ = neural_model(frames, feat_dims=[16, 64, 128], heads=2, params=None)
    mu = output[0][0]
    std = output[0][1]

    action = {
        "rt_mu": mu[0], "lt_mu": mu[1], "ls_mu": mu[2],
        "rt_sigma": std[0], "lt_sigma": std[1], "ls_sigma": std[2]
    }
    print(action)
    return action
