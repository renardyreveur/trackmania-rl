import jax.numpy as jnp
import numpy as np
from jax import random, vmap, grad
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from act_layers import conv2d, linear, layer_norm, gelu, init_params, softmax


def to_numpy(x):
    return np.expand_dims(np.asarray(x) / 255., 0)


# Dataset and Dataloader
dataset = MNIST(root='./', transform=to_numpy)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


def get_params(params, idx):
    return None if params is None else params[idx]


# Model
def mnist_model(in_x, params=None):
    x, p0 = conv2d(in_x, 1, 16, kernel_size=(4, 4), stride=(4, 4), params=get_params(params, 0))
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


k = random.PRNGKey(0)
k, dummy = init_params(k, (2, 1, 28, 28))

# Initialize Model
_, mnist_params = mnist_model(dummy)


# One hot encoding
def one_hot(target, len_pos):
    return jnp.zeros(len_pos).at[target].set(1)


# Loss function
def cross_entropy(output, target):
    output = softmax(output)
    target = one_hot(target, 10)
    return - jnp.sum(jnp.dot(target, jnp.log(output)))


def loss(params, x, y):
    preds, params = mnist_model(x, params)
    loss_fn = vmap(cross_entropy)
    return jnp.mean(loss_fn(preds, y))


# SGD
def update_sgd(params, x, y, lr=0.01):
    gradient = grad(loss)(params, x, y)

    new_params = []
    for p, g in zip(params, gradient):
        up_param = {'weight': p['weight'] - lr * g['weight']}
        if 'bias' in p.keys():
            up_param.update({"bias": p['bias'] - lr * g['bias']})
        new_params.append(up_param)
    return new_params


def progress(data_loader, b_idx):
    base = '[{}/{} ({:.0f}%)]'
    total = len(data_loader)
    current = b_idx
    return base.format(current, total, 100.0 * current / total)


# Training loop
for epoch in range(10):
    print("\n")
    for batch_idx, (data, label) in enumerate(dataloader):
        data = jnp.asarray(data.numpy())
        label = jnp.asarray(label.numpy())
        mnist_params = update_sgd(mnist_params, data, label)
        if batch_idx % int(jnp.sqrt(dataloader.batch_size)) == 0:
            print(f"Epoch {epoch+1} {progress(dataloader, batch_idx)} --- Loss:  {loss(mnist_params, data, label)}")
