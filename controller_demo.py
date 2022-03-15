import queue
import time

import vgamepad as vg

# Loop Parameters
stage = 1
start = time.time()
delta_distance = 0
delta_counter = 0


def reset_game(gp):
    global stage, start, delta_distance, delta_counter
    # Respawn
    gp.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
    gp.update()
    gp.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
    gp.update()
    time.sleep(0.1)

    # Release trigger
    gp.right_trigger_float(value_float=0)
    gp.update()
    time.sleep(0.3)

    # Reset Loop Parameters
    stage = 1
    start = time.time()
    delta_distance = 0
    delta_counter = 0


# Controller process
def game(work_queue):
    global stage, start, delta_distance, delta_counter

    # Virtual gamepad instance
    gamepad = vg.VDS4Gamepad()

    # press a button to wake the device up
    gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE)
    gamepad.update()
    time.sleep(0.5)
    gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE)
    gamepad.update()
    print("HERE WE GO! Change focus to the TrackMania window")
    time.sleep(3)

    # Trackmania Tutorial Track 1
    print("GO!\n")
    reset_game(gamepad)
    time.sleep(1)

    # Main control loop
    while True:
        # Get metrics from Trackmania
        try:
            result = work_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.02)
            continue

        # Accelerate
        if stage == 1:
            print("Accelerating!")
            stage = 2
        gamepad.right_trigger_float(value_float=1)
        dur = time.time() - start

        if 5.6 > dur > 4.3:
            # Turn right
            if stage == 2:
                print("Turning Right!")
                stage = 3
            gamepad.left_joystick_float(x_value_float=1, y_value_float=0.0)
        elif 7.15 > dur >= 5.85:
            # Turn left, reduce speed for a bit
            if stage == 3:
                print("Turning Left!")
                stage = 4
            gamepad.right_trigger_float(value_float=max(1.8 * dur - 5*1.8, 1))
            gamepad.left_joystick_float(x_value_float=-1, y_value_float=0.0)
        elif 8.55 > dur > 7.15:
            # Turn right
            if stage == 4:
                print("Turning Right!")
                stage = 5
            gamepad.left_joystick_float(x_value_float=1, y_value_float=0.0)
        else:
            # Straight ahead!
            if stage == 5:
                print("Straight Ahead!")
                stage = 6
            gamepad.left_joystick_float(x_value_float=0, y_value_float=0.0)

        # If vehicle is stuck
        if abs(delta_distance - result['distance']) < 0.1:
            if delta_counter == 0:
                timer = time.time()
                delta_counter += 1
            delta_counter += time.time() - timer
        else:
            delta_counter = 0
        delta_distance = result['distance']
        gamepad.update()

        # If vehicle stuck for more than 2000 units of continuous 'time', reset
        if delta_counter > 2000:
            print("RESULT -- Going nowhere!\n")
            reset_game(gamepad)

        # If race is finished, reset
        if result['race_finished']:
            print(f"RESULT -- Race Finished!! Record: {result['duration']/1000.} seconds\n")
            time.sleep(0.3)
            reset_game(gamepad)

    # reset gamepad to default state
    gamepad.reset()
    gamepad.update()
