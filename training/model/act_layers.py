import jax.lax as jlax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jit, random


def get_params(params, idx):
    return None if params is None else params[idx]


def init_params(key, size):
    key, subkey = random.split(key)
    return key, random.normal(subkey, shape=size)


@jit
def sigmoid(x):
    return jnp.where(x >= 0,
                     1 / (1 + jnp.exp(-x)),
                     jnp.exp(x) / (1 + jnp.exp(x)))


@jit
def hard_swish(in_x):
    return in_x * jnp.maximum(0, in_x + 3) / 6


@jit
def gelu(in_x):
    return in_x * jscipy.stats.norm.cdf(in_x)


@jit
def softmax(in_x):
    return jnp.divide(jnp.exp(in_x), jnp.sum(jnp.exp(in_x)))


def layer_norm(in_x, dim, params=None):
    if params is None:
        params = {"weight": jnp.ones(in_x.shape[dim]), "bias": jnp.zeros(in_x.shape[dim])}
    if in_x.shape[dim] != params['weight'].shape[0]:
        raise ValueError(f"Parameter Shape doesn't match layer! {in_x.shape[dim]} and {params['weight'].shape}")
    eps = 0.00001
    mean = jnp.mean(in_x, axis=dim, keepdims=True)
    var = jnp.var(in_x, axis=dim, keepdims=True)
    x = (in_x - mean) / jnp.sqrt(var + eps)
    w = jnp.reshape(params['weight'], (1,) * dim + (-1,) + (1,) * (len(in_x.shape) - 1 - dim))
    b = jnp.reshape(params['bias'], (1,) * dim + (-1,) + (1,) * (len(in_x.shape) - 1 - dim))
    return w * x + b, params


def linear(in_x, proj_dim, bias=True, seed=0, params=None):
    if params is None:
        k = random.PRNGKey(seed)
        params = {}
        for i in ['weight', 'bias'] if bias else ['weight']:
            k, subkey = random.split(k)
            shape = (in_x.shape[-1], proj_dim) if i == "weight" else (proj_dim,)
            p = random.normal(subkey, shape=shape)
            p *= jnp.sqrt(2 / proj_dim)
            params.update({i: p})

    if params['weight'].shape[-1] != proj_dim or params['weight'].shape[-2] != in_x.shape[-1]:
        raise ValueError(
            f"Parameter Shape doesn't match layer! {params['weight']} is not {in_x.shape[-1]} x {proj_dim}")

    return in_x @ params['weight'] + (params['bias'] if bias else 0), params


def conv2d(in_x, in_chns, out_chns, kernel_size, padding="VALID", groups=1, stride=(1,) * 2, bias=False,
           seed=0, params=None):
    if params is None:
        k = random.PRNGKey(seed)
        params = {}
        for i in (['weight', 'bias'] if bias else ['weight']):
            k, subkey = random.split(k)
            shape = (out_chns, in_chns // groups, *kernel_size) if i == "weight" else (out_chns,)
            p = random.uniform(subkey, minval=-jnp.sqrt(groups / (in_chns * jnp.prod(jnp.array(kernel_size)))),
                               maxval=jnp.sqrt(groups / (in_chns * jnp.prod(jnp.array(kernel_size)))),
                               shape=shape)
            params.update({i: p})

    conv = jlax.conv_general_dilated(in_x, params['weight'], stride, padding, feature_group_count=groups)

    if bias:
        conv += jnp.reshape(params['bias'], (1, out_chns, 1, 1))

    return conv, params
