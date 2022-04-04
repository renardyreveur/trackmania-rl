import time

from processes.agent import policies
from processes.agent.utils import init_game, reset_game, RaceManager


# Controller process
def game(met_que, img_que, max_t_len, policy_str):
    # Instantiate a RaceManager object
    rm = RaceManager(met_que, img_que, max_t_len)

    # Let the virtual controller be recognised, wait for track selection
    gamepad = init_game()
    print("GO!\n")
    time.sleep(1)
    reset_game(gamepad, rm)

    # Main control loop
    while True:
        # Get metrics and screenshots
        metrics, frames = rm.get_metrics_frames()
        if metrics == -1:
            continue

        # Agent Action
        policy = getattr(policies, policy_str)
        sub_action, action = policy(frames=frames, start=rm.start_timer, params=rm.policy_params)
        gamepad.right_trigger_float(value_float=sub_action['rt'])
        gamepad.left_trigger_float(value_float=sub_action['lt'])
        gamepad.left_joystick_float(x_value_float=sub_action['ls'], y_value_float=0.0)
        gamepad.update()

        # Check race state, and collect data for training
        state = rm.check_state()
        ret = rm.collect_data(state, gamepad)
        if ret:
            continue

        if state in [-1, -2]:
            reset_game(gamepad, rm)

        # Save 'current' state action pair for trajectory
        rm.update_history(frames, action)

    # reset gamepad to default state
    gamepad.reset()
    gamepad.update()
