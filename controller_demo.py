import queue
import time

import vgamepad as vg


def game(work_queue):
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

    # Restart
    gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
    gamepad.update()
    gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
    gamepad.update()
    time.sleep(1)

    # Log stage controller
    stage = 1

    # Main control loop
    start = time.time()
    while True:
        try:
            result = work_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.02)
            continue
        print(f"Front Speed: {result['front_speed']}")

        # Accelerate
        if stage == 1:
            print("Accelerating!\n")
            stage = 2
        gamepad.right_trigger_float(value_float=1)
        dur = time.time() - start

        if 4.85 > dur > 3.55:
            # Turn right
            if stage == 2:
                print("Turning Right!")
                stage = 3
            gamepad.left_joystick_float(x_value_float=1, y_value_float=0.0)
        elif 6.3 > dur >= 5:
            # Turn left, reduce speed for a bit
            if stage == 3:
                print("Turning Left!")
                stage = 4
            gamepad.right_trigger_float(value_float=max(1.8 * dur - 5*1.8, 1))
            gamepad.left_joystick_float(x_value_float=-1, y_value_float=0.0)
        elif 7.7 > dur > 6.3:
            # Turn right
            if stage == 4:
                print("Turning Right!")
                stage = 5
            gamepad.left_joystick_float(x_value_float=1, y_value_float=0.0)
        else:
            # Straight ahead!
            if stage == 5:
                print("Straight Ahead!\n")
                stage = 6
            gamepad.left_joystick_float(x_value_float=0, y_value_float=0.0)

        gamepad.update()

        if result['race_finished']:
            print(f"Race Finished!! Record: {result['duration']/1000.} seconds")
            break

    # reset gamepad to default state
    gamepad.reset()

    gamepad.update()

    time.sleep(1.0)
