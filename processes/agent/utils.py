import time
import queue

import vgamepad as vg
import win32api as wapi
import pickle


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

        # History
        self.trajectory = []
        self.num_trajectory = 0
        self.state_history, self.act_history = None, None

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
        if abs(self.delta_distance - self.metrics['distance']) < 0.1:
            if self.delta_counter == 0:
                self.stuck_timer = time.time()
                self.delta_counter += 1
            self.delta_counter += time.time() - self.stuck_timer
        else:
            self.delta_counter = 0
        self.delta_distance = self.metrics['distance']

        # If vehicle stuck for more than 1500 units of continuous 'time', reset
        if self.delta_counter > 100:
            print("RESULT -- Going nowhere!\n")
            return -1

        # If race is finished, reset
        if self.metrics['race_finished']:
            print(f"RESULT -- Race Finished!! Record: {self.metrics['duration'] / 1000.} seconds\n")
            time.sleep(0.3)
            return -2

        return 0

    def update_history(self, state, action):
        self.state_history = state
        self.act_history = action

    def collect_data(self, state):
        # Initial state
        if self.state_history is None or self.act_history is None:
            return -1

        # Reward Calculation
        metrics = [self.metrics['distance'],
                   self.metrics['front_speed'],
                   int(self.metrics['checkpoint'][1:]) + 1,
                   self.metrics['duration']]
        weights = [-0.01, 0.2, 1000, -1]
        reward = sum([weights[i] * metrics[i] for i in range(len(metrics))])

        # Race finish gives the ultimate reward
        if state == -2:
            reward += 10000

        self.trajectory.append((self.state_history, self.act_history, reward, self.frames))

        # If a run is completed, reset replay buffer and save (as s_t and s_{t+1} won't match)
        if state != 0:
            with open(f"training/data/t_{self.num_trajectory}.traj", 'wb') as f:
                pickle.dump(self.trajectory, f)
            self.trajectory = []
            self.num_trajectory += 1


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
