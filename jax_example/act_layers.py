import jax.numpy as jnp
from jax import jit


@jit
def hard_swish(in_x):
    return in_x * jnp.maximum(0, in_x + 3) / 6


@jit
def softmax(in_x):
    return jnp.divide(jnp.exp(in_x), jnp.sum(jnp.exp(in_x)))


def layer_norm(in_x, dim):
    eps = 0.00001
    # Without elementwise affine (which involved learnable parameters)
    mean = jnp.mean(in_x, axis=dim, keepdims=True)
    var = jnp.var(in_x, axis=dim, keepdims=True)
    return (in_x-mean) / jnp.sqrt(var + eps)
