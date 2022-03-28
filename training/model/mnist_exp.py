

import jax.numpy as jnp
from jax import jit

from .act_layers import conv2d, layer_norm, linear, gelu, get_params


@jit
def mnist_model(in_x, params=None):
    x, p0 = conv2d(in_x, 1, 16, kernel_size=(4, 4), stride=(4, 4), bias=True, params=get_params(params, 0))
    x, p1 = layer_norm(x, 1, get_params(params, 1))
    x = gelu(x)
    x, p2 = conv2d(x, 16, 32, kernel_size=(7, 7), padding=((3, 3), (3, 3)), groups=16, params=get_params(params, 2))
    x = jnp.transpose(x, (0, 2, 3, 1))
    x, p3 = layer_norm(x, 3, get_params(params, 3))
    x, p4 = linear(x, 64, get_params(params, 4))
    x = gelu(x)
    x = jnp.transpose(x, (0, 3, 1, 2))
    x, p5 = layer_norm(x, 1, get_params(params, 5))
    x, p6 = conv2d(x, 64, 32, kernel_size=(2, 2), stride=(2, 2), params=get_params(params, 6))
    x = gelu(x)
    x = jnp.reshape(x, (x.shape[0], -1))
    x, p7 = linear(x, 10, get_params(params, 7))

    return x, [p0, p1, p2, p3, p4, p5, p6, p7]
