import time
import queue

import vgamepad as vg
import win32api as wapi


class RaceManager:
    def __init__(self, metric_que, image_que):
        # 'Stuck' parameters
        self.delta_distance, self.delta_counter = 0, 0

        # State and Reward parameters
        self.mque, self.ique = metric_que, image_que

        # Timers
        self.start_timer, self.stuck_timer = time.time(), 0

        # Observations
        self.metrics, self.frames = None, None

    def reset(self):
        self.delta_counter, self.delta_distance = 0, 0
        self.start_timer, self.stuck_timer = time.time(), 0

    def get_metrics_frames(self):
        try:
            self.metrics = self.mque.get_nowait()
            if self.metrics['race_finished']:
                return self.metrics, []
            elif self.ique.length() == self.ique.get_max_len():
                self.frames = self.ique.get_frames()
            else:
                raise queue.Empty
            return self.metrics, self.frames
        except queue.Empty:
            time.sleep(0.02)
            return -1, -1

    def check_state(self):
        # If vehicle is stuck
        if abs(self.delta_distance - self.metrics['distance']) < 0.2:
            if self.delta_counter == 0:
                self.stuck_timer = time.time()
                self.delta_counter += 1
            self.delta_counter += time.time() - self.stuck_timer
        else:
            self.delta_counter = 0
        self.delta_distance = self.metrics['distance']

        # If vehicle stuck for more than 1500 units of continuous 'time', reset
        if self.delta_counter > 50:
            print("RESULT -- Going nowhere!\n")
            return -1

        # If race is finished, reset
        if self.metrics['race_finished']:
            print(f"RESULT -- Race Finished!! Record: {self.metrics['duration'] / 1000.} seconds\n")
            time.sleep(0.3)
            return -2

        return 0

    def collect_data(self):
        pass


def init_game():
    # Virtual gamepad instance
    gamepad = vg.VDS4Gamepad()

    # press a button to wake the device up
    gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE)
    gamepad.update()
    time.sleep(0.5)
    gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE)
    gamepad.update()

    print("Press 'g' when ready")
    while True:
        time.sleep(1)
        if wapi.GetAsyncKeyState(ord("G")):
            break

    return gamepad


def reset_game(gp: vg.VDS4Gamepad, rm: RaceManager):
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

    # Reset RaceManager
    rm.reset()
