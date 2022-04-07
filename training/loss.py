import jax.numpy as jnp
from jax import vmap

from model.act_layers import softmax


# One hot encoding
def one_hot(target, len_pos):
    return jnp.zeros(len_pos).at[target].set(1)


# Cross Entropy single example
def cross_entropy(output, target):
    output = softmax(output)
    target = one_hot(target, 10)
    return - jnp.sum(jnp.dot(target, jnp.log(output)))


# Batched Cross Entropy loss with 'mean' reduction
def batch_cross_entropy(params, model, x, y):
    preds, params = model(x, params)
    loss_fn = vmap(cross_entropy)
    return jnp.mean(loss_fn(preds, y))


# soft Q-function target
def soft_q_loss(parameters, models, trajectory_iteration, decay=0.99, entropy_temp=0.2):
    neural_model, q_func = models
    q1_p, q2_p, pi_p, q1_t_p, q2_t_p = parameters

    s_0, a_0 = trajectory_iteration[0], trajectory_iteration[1]
    reward, s_1 = trajectory_iteration[2], trajectory_iteration[3]

    # Calculate Q values for the given state and action pair
    q_val1, _ = q_func(s_0, a_0, params=q1_p)
    q_val2, _ = q_func(s_0, a_0, params=q2_p)

    # --- Soft Q function target ---
    # Sample action from 'current' policy with s_{t+1}
    (a_1, logpi_a_1), _ = neural_model(s_1, params=pi_p)
    a_1, logpi_a_1 = jnp.stack(a_1, axis=1), jnp.stack(logpi_a_1, axis=1)

    # Q values Target
    q_val1_t, _ = q_func(s_1, a_1, params=q1_t_p)
    q_val2_t, _ = q_func(s_1, a_1, params=q2_t_p)

    q_val_t = jnp.minimum(q_val1_t, q_val2_t)
    q_target = reward + decay * (q_val_t - entropy_temp * logpi_a_1)

    # MSE Loss against Bellman backup
    loss_q1 = jnp.mean((q_val1 - q_target)**2)
    loss_q2 = jnp.mean((q_val2 - q_target)**2)
    loss_q = loss_q1 + loss_q2

    return loss_q


# Policy Loss
def policy_loss(parameters, models, trajectory_iteration, entropy_temp=0.35):
    neural_model, q_func = models
    pi_p, q1_p, q2_p = parameters

    s_0 = trajectory_iteration[0]

    # Sample action from current time
    (a_0, logprob_a_0), _ = neural_model(s_0, params=pi_p)
    a_0, logprob_a_0 = jnp.stack(a_0, axis=1), jnp.stack(logprob_a_0, axis=1)

    # Get Q value of sampled action
    q1_val, _ = q_func(s_0, a_0, params=q1_p)
    q2_val, _ = q_func(s_0, a_0, params=q2_p)
    q_val = jnp.minimum(q1_val, q2_val)

    # Minimize expected KL-divergence
    loss_pi = jnp.mean(entropy_temp * logprob_a_0 - q_val)

    return loss_pi
