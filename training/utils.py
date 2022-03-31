import jax.numpy as jnp


# Training progress
def progress(data_loader, b_idx):
    base = '[{}/{} ({:.0f}%)]'
    total = len(data_loader)
    current = b_idx
    return base.format(current, total, 100.0 * current / total)


# Get model parameter count
def parameter_count(params):
    params = [parameter_count(x) if isinstance(x, list) else sum([jnp.prod(jnp.asarray(v.shape)) for _, v in x.items()])
              for x in params]
    return sum(params)
