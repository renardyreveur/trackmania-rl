import time
import random
import jax.numpy as jnp

from processes.agent import policies
from processes.agent.utils import init_game, reset_game, RaceManager


# Controller process
def game(met_que, img_que, policy_str, agent_conn, agent_params):
    # Instantiate a RaceManager object
    rm = RaceManager(met_que, img_que, agent_conn, agent_params, policy_str)

    # Get Policy
    policy = getattr(policies, policy_str)

    # Prime the policy (JIT)
    print("Priming the policy...")
    _ = policy(frames=[jnp.ones((1, 1, *agent_params['screenshot_size'])) for _ in range(agent_params['screenshot_maxlen'])],
               start=rm.start_timer, params=rm.policy_params,
               feat_dims=agent_params['p_feat_dims'], seed=random.randint(0, 2222222))

    # Let the virtual controller be recognised, wait for track selection
    gamepad = init_game()
    print("GO!\n")
    print(f"Trajectory buffer has length: {agent_params['trajectory_maxlen']}")
    time.sleep(1)
    reset_game(gamepad, rm)

    # Main control loop
    while True:
        # Get metrics and screenshots
        metrics, frames = rm.get_metrics_frames()
        if metrics == -1:
            continue

        sub_action, action = policy(frames=frames, start=rm.start_timer, params=rm.policy_params,
                                    feat_dims=agent_params['p_feat_dims'], seed=random.randint(0, 2222222))
        gamepad.right_trigger_float(value_float=sub_action['rt'])
        gamepad.left_trigger_float(value_float=sub_action['lt'])
        gamepad.left_joystick_float(x_value_float=sub_action['ls'], y_value_float=0.0)
        gamepad.update()

        # Check race state, and collect data for training
        state = rm.check_state()
        ret = rm.collect_data(state, gamepad)
        if ret:
            reset_game(gamepad, rm)
            continue

        if state in [-1, -2]:
            time.sleep(0.3)
            reset_game(gamepad, rm)

        # Save 'current' state action pair for trajectory
        rm.update_history(frames, action)

    # reset gamepad to default state
    gamepad.reset()
    gamepad.update()
