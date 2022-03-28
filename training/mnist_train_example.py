import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import optim
from loss import batch_cross_entropy
from model.act_layers import init_params
from model.mnist_exp import mnist_model
from optim import update


# Helper Functions
def to_numpy(x):
    return np.expand_dims(np.asarray(x) / 255., 0)


def progress(data_loader, b_idx):
    base = '[{}/{} ({:.0f}%)]'
    total = len(data_loader)
    current = b_idx
    return base.format(current, total, 100.0 * current / total)


# Dataset and Dataloader
dataset = MNIST(root='./', transform=to_numpy)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

pkey = random.PRNGKey(0)
pkey, dummy = init_params(pkey, (2, 1, 28, 28))

# Initialize Model
_, mnist_params = mnist_model(dummy)
num_params = sum([sum([jnp.prod(jnp.asarray(v.shape)) for k, v in p.items()]) for p in mnist_params])
print(f"The model has {num_params} parameters!")


# Set Optimizer
OPTIM = "sgd"
optimizer = getattr(optim, OPTIM)
optimizer_params = optimizer(mnist_params, None)

# Training loop
for epoch in range(10):
    print("\n")
    start = time.time()
    for batch_idx, (data, label) in enumerate(dataloader):
        data = jnp.asarray(data.numpy())
        label = jnp.asarray(label.numpy())
        mnist_params, optim_params = update(jax.tree_util.Partial(batch_cross_entropy),
                                            jax.tree_util.Partial(mnist_model),
                                            mnist_params,
                                            data, label,
                                            jax.tree_util.Partial(optimizer), **{"optimizer_params": optimizer_params})
        if batch_idx % int(jnp.sqrt(dataloader.batch_size)) == 0:
            print(f"Epoch {epoch + 1} {progress(dataloader, batch_idx)}"
                  f" --- Loss:  {batch_cross_entropy(mnist_params, mnist_model, data, label)}")
    end = time.time()
    print(f"Epoch {epoch + 1} took {end - start:.2f} seconds to complete!")
