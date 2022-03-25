from functools import partial

import jax.numpy as jnp
from jax import jit

from act_layers import hard_swish, layer_norm
from attention import attention


# @partial(jit, static_argnums=(1,))
# @jit
def encoder(params, heads, in_x):
    # Make input into model dim
    x = in_x @ params['proj_1']

    # Multi-head attention, concat results
    mh_out = jnp.concatenate([hard_swish(attention(params[f'attn_{i}'], x)) for i in range(heads)], axis=-1)

    # Residual skip-connection and layer norm
    res = hard_swish(mh_out @ params['proj_2'] + x)
    nmres = layer_norm(res, dim=-1)

    # Feed forward, residual skip-connection, layer norm
    return layer_norm(hard_swish(nmres @ params['proj_3'] + nmres), dim=-1)


# @jit
def decoder(params, heads, in_x, key, value):
    # Make input into model dim
    x = in_x @ params['dproj_1']

    # Multi-head attention, concat results
    mh_out = jnp.concatenate([hard_swish(attention(params[f'attn_{i}'], x, key, value)) for i in range(heads)], axis=-1)

    # Residual skip-connection and layer norm
    res = hard_swish(mh_out @ params['dproj_2'] + x)
    nmres = layer_norm(res, dim=-1)

    # Feed forward, residual skip-connection, layer norm
    return layer_norm(hard_swish(nmres @ params['dproj_3'] + nmres), dim=-1) @ params['dproj_4']


# @jit
def simple_transformer(params, in_x, heads=2, max_out=10):
    # Encoder forward pass
    encout = encoder(params, heads, in_x)

    # Re-project the 'context' for the decoder
    key = encout @ params['key_proj']
    val = encout @ params['val_proj']

    # Prime the decoder
    primer = jnp.expand_dims(jnp.zeros_like(in_x)[:, 0, :], axis=1)

    # Iterate the decoder until end condition
    output = []
    for i in range(max_out):
        primer = decoder(params, heads, primer, key, val)
        output.append(primer)

    return jnp.concatenate(output, axis=1)

