import os
import time
from functools import partial

if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")
import jax
import jax.numpy as jnp
import jax.random as jrandom

import optim
from dataloader import TrajectoryDataset, TrajectoryLoader
from loss import soft_q_loss, policy_loss
from model.act_layers import init_params
from model.policy_model import neural_model
from model.q_function import q_func
from utils import progress, parameter_count

# TRAIN PARAMETERS
EPOCH = 10
Q_FEAT_DIMS = (16, 128)
PI_FEAT_DIMS = (16, 64, 128)
T_SMOOTH = 0.005

# Initialize PRNG
k = jrandom.PRNGKey(123)

# Two soft Q-functions
print("** Creating Q Functions ...")
k, qs_dummy = init_params(k, size=(1, 5, 480, 320))
k, qa_dummy = init_params(k, size=(1, 3))
q_func = jax.tree_util.Partial(q_func, feat_dims=Q_FEAT_DIMS)  # Create partial function with static argument
_, q1_params = q_func(qs_dummy, qa_dummy)
_, q2_params = q_func(qs_dummy, qa_dummy)
q_num_params = parameter_count(q1_params)
print(f"Each Q-Value function has {q_num_params} parameters! \n")

# Exponentially moving average of soft Q-function weights (used in soft-q function target calculation)
q1_expmov_params = q1_params.copy()
q2_expmov_params = q2_params.copy()

# Policy network
print("** Creating Policy network ...")
k, pi_dummy = init_params(k, size=(1, 5, 480, 320))
neural_model = jax.tree_util.Partial(neural_model, feat_dims=PI_FEAT_DIMS)  # Create partial function with static argument
_, pi_params = neural_model(pi_dummy)
pi_num_params = parameter_count(pi_params)
print(f"The Policy network has {pi_num_params} parameters!\n")

# Set Optimizer
print("** Defining Optimizers to update the networks ...")
OPTIM = "adamw"
lr = 0.0003
q_optimizer = jax.tree_util.Partial(getattr(optim, OPTIM), lr=lr)
pi_optimizer = jax.tree_util.Partial(getattr(optim, OPTIM), lr=lr)

q1_optimizer_params = q_optimizer(q1_params, None)
q2_optimizer_params = q_optimizer(q2_params, None)
pi_optimizer_params = pi_optimizer(pi_params, None)
print("Done!\n")

# Get Dataloader
print("** Loading Dataset and creating DataLoader ...")
dataset = TrajectoryDataset("data")
dataloader = TrajectoryLoader(dataset, 5)
print(f"There are {len(dataset)} samples, and one epoch with a batch size of {20} has {len(dataset)//20} batches!\n")

# Training loop
print("** Start Training!\n")
print("Please be patient, it's probably JITing sth...")
for epoch in range(EPOCH):
    print("\n")
    start = time.time()
    for batch_idx, data in enumerate(dataloader):
        # Gradient Descent on q1 and q2
        q_loss, (q1_params, q2_params), (q1_optimizer_params, q2_optimizer_params) = optim.update(
            in_data=data,
            loss_fn=jax.tree_util.Partial(soft_q_loss),
            model=(neural_model, q_func),
            params=(q1_params, q2_params, pi_params, q1_expmov_params, q2_expmov_params),
            optimizer=q_optimizer,
            optimizer_params=(q1_optimizer_params, q2_optimizer_params),
            )

        # Gradient Descent on policy network
        pi_loss, pi_params, pi_optimizer_params = optim.update(
            in_data=data,
            loss_fn=jax.tree_util.Partial(policy_loss),
            model=(neural_model, q_func),
            params=(pi_params, q1_params, q2_params),
            optimizer=pi_optimizer,
            optimizer_params=(pi_optimizer_params,),
        )

        # Q Targets - Exponential moving average of Target Parameters update
        q1_expmov_params = optim.polyak(q1_params, q1_expmov_params, T_SMOOTH)
        q2_expmov_params = optim.polyak(q2_params, q2_expmov_params, T_SMOOTH)

        if batch_idx % int(jnp.sqrt(dataloader.batch_size)) == 0:
            print(f"Epoch {epoch + 1} {progress(dataloader, batch_idx)}"
                  f" --- Q Loss: {q_loss},   Policy Loss: {pi_loss}")
    end = time.time()
    print(f"Epoch {epoch + 1} took {end - start:.2f} seconds to complete!")
