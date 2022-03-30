import os
if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")

import jax.numpy as jnp
import numpy as np

from training.model.policy_model import neural_model


def neural_policy(frames, params, **kwargs):
    frames = jnp.asarray(np.concatenate(frames, 1))
    (action, _), _ = neural_model(frames, (16, 64, 128), 2, params=params)

    # Enforce Action bounds? tanh, sigmoid
    return {"rt": action[0].item(), "lt": action[1].item(), "ls": action[2].item()}
