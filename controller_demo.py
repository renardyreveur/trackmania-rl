import vgamepad as vg
import time

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

PRINT = 1

start = time.time()
while True:
    # Accelerate
    if PRINT == 1:
        print("Accelerating!\n")
        PRINT = 2
    gamepad.right_trigger_float(value_float=1)
    dur = time.time() - start

    if 4.85 > dur > 3.55:
        # Turn right
        if PRINT == 2:
            print("Turning Right!")
            PRINT = 3
        gamepad.left_joystick_float(x_value_float=1, y_value_float=0.0)
    elif 6.3 > dur >= 5:
        # Turn left, reduce speed for a bit
        if PRINT == 3:
            print("Turning Left!")
            PRINT = 4
        gamepad.right_trigger_float(value_float=max(1.8 * dur - 5*1.8, 1))
        gamepad.left_joystick_float(x_value_float=-1, y_value_float=0.0)
    elif 7.7 > dur > 6.3:
        # Turn right
        if PRINT == 4:
            print("Turning Right!")
            PRINT = 5
        gamepad.left_joystick_float(x_value_float=1, y_value_float=0.0)
    else:
        # Straight ahead!
        if PRINT == 5:
            print("Straight Ahead!\n")
            PRINT = 6
        gamepad.left_joystick_float(x_value_float=0, y_value_float=0.0)

    gamepad.update()

    if dur > 8.5:
        print("Race Finished!!")
        break

# reset gamepad to default state
gamepad.reset()

gamepad.update()

time.sleep(1.0)
