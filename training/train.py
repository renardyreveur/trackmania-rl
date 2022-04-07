import os
import time

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


def train(trainer_conn, agent_params):
    # Init
    q_feat_dims = agent_params['q_feat_dims']
    pi_feat_dims = agent_params['p_feat_dims']
    k = jrandom.PRNGKey(42)

    # Two soft Q-functions
    print("\n---- Train Process Initializing... ----")
    print("** Creating Q Functions ...")
    k, qs_dummy = init_params(k, size=(1, agent_params['screenshot_maxlen'], *agent_params['screenshot_size']))
    k, qa_dummy = init_params(k, size=(1, 3))
    q_model = jax.tree_util.Partial(q_func, feat_dims=q_feat_dims)  # Create partial function with static argument
    _, q1_params = q_model(qs_dummy, qa_dummy)
    _, q2_params = q_model(qs_dummy, qa_dummy)
    q_num_params = parameter_count(q1_params)
    print(f"Each Q-Value function has {q_num_params} parameters! \n")

    # Exponentially moving average of soft Q-function weights (used in soft-q function target calculation)
    q1_expmov_params = q1_params.copy()
    q2_expmov_params = q2_params.copy()

    # Policy network
    print("** Creating Policy network ...")
    k, pi_dummy = init_params(k, size=(1, agent_params['screenshot_maxlen'], *agent_params['screenshot_size']))
    pi_model = jax.tree_util.Partial(neural_model,
                                     feat_dims=pi_feat_dims)  # Create partial function with static argument
    _, pi_params = pi_model(pi_dummy)
    pi_num_params = parameter_count(pi_params)
    print(f"The Policy network has {pi_num_params} parameters!\n")

    # Set Optimizer
    print("** Defining Optimizers to update the networks ...")
    optimizer = "adamw"
    lr = 0.0003
    q_optimizer = jax.tree_util.Partial(getattr(optim, optimizer), lr=lr)
    pi_optimizer = jax.tree_util.Partial(getattr(optim, optimizer), lr=lr)

    q1_optimizer_params = q_optimizer(q1_params, None)
    q2_optimizer_params = q_optimizer(q2_params, None)
    pi_optimizer_params = pi_optimizer(pi_params, None)
    print("Done!\n")

    q_feat_dims, pi_feat_dims = agent_params['q_feat_dims'], agent_params['p_feat_dims']
    q_model = jax.tree_util.Partial(q_func, feat_dims=q_feat_dims)  # Create partial function with static argument
    pi_model = jax.tree_util.Partial(neural_model, feat_dims=pi_feat_dims)

    # Send initial Policy to Agent
    trainer_conn.send(pi_params)

    while True:
        # Receive start training signal
        trajectory_list = trainer_conn.recv()
        print("Received Trajectory Buffer!")

        # Prepare dataset and data loader
        dataset = TrajectoryDataset(trajectory_list, online_training=True)
        dataloader = TrajectoryLoader(dataset, agent_params['batch_size'])
        print("** Start Training!\n")
        print("Please be patient, it's probably JITing sth...")

        # Training loop
        for epoch in range(agent_params['epochs']):
            print("\n")
            start = time.time()
            for batch_idx, data in enumerate(dataloader):
                # Gradient Descent on q1 and q2
                q_loss, (q1_params, q2_params), (q1_optimizer_params, q2_optimizer_params) = optim.update(
                    in_data=data,
                    loss_fn=jax.tree_util.Partial(soft_q_loss),
                    model=(pi_model, q_model),
                    params=(q1_params, q2_params, pi_params, q1_expmov_params, q2_expmov_params),
                    optimizer=q_optimizer,
                    optimizer_params=(q1_optimizer_params, q2_optimizer_params),
                )

                # Gradient Descent on policy network
                pi_loss, pi_params, pi_optimizer_params = optim.update(
                    in_data=data,
                    loss_fn=jax.tree_util.Partial(policy_loss),
                    model=(pi_model, q_model),
                    params=(pi_params, q1_params, q2_params),
                    optimizer=pi_optimizer,
                    optimizer_params=(pi_optimizer_params,),
                )

                # Q Targets - Exponential moving average of Target Parameters update
                q1_expmov_params = optim.polyak(q1_params, q1_expmov_params, 0.005)
                q2_expmov_params = optim.polyak(q2_params, q2_expmov_params, 0.005)

                if batch_idx % int(jnp.sqrt(dataloader.batch_size)) == 0:
                    print(f"Epoch {epoch + 1} {progress(dataloader, batch_idx)}"
                          f" --- Q Loss: {q_loss},   Policy Loss: {pi_loss}")
            end = time.time()
            print(f"Epoch {epoch + 1} took {end - start:.2f} seconds to complete!")

        # Send updated params to main process
        trainer_conn.send(pi_params)
        print("Sent data back to agent process!")
