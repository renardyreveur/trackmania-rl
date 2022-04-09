import jax.numpy as jnp
import numpy as np

from training.model.policy_model import neural_model


def neural_policy(frames, params, **kwargs):
    if len(frames) == 0:
        return {"rt": 0, "lt": 0, "ls": 0}, None
    frames = jnp.asarray(np.concatenate(frames, 1))
    (action, _), _ = neural_model(frames, kwargs['feat_dims'], 2, params=params, seed=kwargs['seed'])
    return {"rt": action[0].item(), "lt": action[1].item(), "ls": action[2].item()}, action
