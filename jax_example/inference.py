import timeit

from jax import random

from model import simple_transformer
from utils import init_params, enc_params, dec_params, attn_params, ctx_params

# Create initial PRNG key
k = random.PRNGKey(0)

# Dimensions
FEAT_DIM = 64
QKV_DIM = 32
HEADS = 2

# Initialize weights for model
params = {}
k, eparams = enc_params(k, FEAT_DIM, QKV_DIM, HEADS)
k, dparams = dec_params(k, FEAT_DIM, QKV_DIM, HEADS)
k, cparams = ctx_params(k, QKV_DIM, QKV_DIM)
params.update(eparams)
params.update(dparams)
params.update(cparams)
for i in range(HEADS):
    k, aparams = attn_params(k, QKV_DIM)
    params.update({f"attn_{i}": aparams})

# Dummy input
k, x = init_params(k, (5, 2, FEAT_DIM))

# Forward pass of the model
output = simple_transformer(params, x, heads=HEADS)
print(f"Output of shape: {output.shape} is produced!")

# Calculate Inference time
inf_times = timeit.repeat("simple_transformer(params, x, heads=HEADS).block_until_ready()",
                          "from __main__ import simple_transformer, params, x, HEADS", number=20)
print(f"Average full forward pass took {sum(inf_times)/len(inf_times)} seconds!")
