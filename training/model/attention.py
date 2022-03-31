import jax.numpy as jnp
from jax import random
from .act_layers import softmax


def attention(in_x, dim, keys=None, values=None, seed=0, params=None):
    """
    Attention Layer Forward Pass Function
    :param params: List of 1 or 3 entries with Q, K, V weights of shape (Feat_dim, QKV_dim)
    :param in_x: Array of Shape (Batch, Seq_len, Feat_dim)
    :param dim: QKV_dim
    :param keys: Array of Shape (Seq_len, QKV_dim) if provided
    :param values: Array of Shape (Seq_len, QKV_dim) if provided
    :param seed: Random seed used in initializing the weights
    :return:
    """
    if params is None:
        k = random.PRNGKey(seed)
        params = {}
        for i in ['w_query', 'w_key', 'w_val']:
            k, subkey = random.split(k)
            shape = (in_x.shape[-1], dim)
            p = random.normal(subkey, shape=shape)
            p *= jnp.sqrt(2 / dim)
            params.update({i: p})

    # Generate Query, Key, Value from input
    queries = in_x @ params['w_query']
    if keys is None:
        keys = in_x @ params['w_key']
        values = in_x @ params['w_val']

    # Calculate attention scores (using query and key) - scaled dot product
    attn_scores = softmax(1 / jnp.sqrt(queries.shape[-1]) * (queries @ jnp.transpose(keys, axes=[0, 2, 1])))

    # Get sum of weighted values as output
    return attn_scores @ values, params
