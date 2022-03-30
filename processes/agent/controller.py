from processes.agent import policies
from processes.agent.utils import init_game, reset_game, RaceManager


# Controller process
def game(met_que, img_que, policy_str, params=None):
    # Instantiate a RaceManager object
    rm = RaceManager(met_que, img_que)

    # Let the virtual controller be recognised, wait for track selection
    gamepad = init_game()
    print("GO!\n")
    reset_game(gamepad, rm)

    # Main control loop
    while True:
        # Get metrics and screenshots
        metrics, frames = rm.get_metrics_frames()
        if metrics == -1:
            continue

        # Agent Action
        policy = getattr(policies, policy_str)
        action = policy(frames=frames, start=rm.start_timer, params=params)
        gamepad.right_trigger_float(value_float=action['rt'])
        gamepad.left_trigger_float(value_float=action['lt'])
        gamepad.left_joystick_float(x_value_float=action['ls'], y_value_float=0.0)
        gamepad.update()

        # Check race state, and collect data for training
        state = rm.check_state()
        rm.collect_data(state)
        if state in [-1, -2]:
            reset_game(gamepad, rm)

        # Save 'current' state action pair for trajectory
        rm.update_history(frames, action)

    # reset gamepad to default state
    gamepad.reset()
    gamepad.update()
