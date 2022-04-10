import pickle
import queue
import time
from datetime import date

import vgamepad as vg
import win32api as wapi

from helpers import remove_focus, set_tm_window


class RaceManager:
    def __init__(self, metric_que, image_que, agent_conn, agent_params, policy_str):
        # 'Stuck' parameters
        self.delta_distance, self.delta_counter = 0, 0
        self.start_timer, self.stuck_timer = time.time(), 0

        # State and Reward parameters
        self.mque, self.ique = metric_que, image_que
        self.metrics, self.frames = None, None

        # History
        self.state_history, self.act_history = None, None
        self.trajectory = []
        self.position_history, self.max_pos_hist = [], 10

        # Online Training
        self.policy_str = policy_str
        self.agent_conn = agent_conn
        self.agent_params = agent_params
        self.policy_params = agent_conn.recv() if policy_str != "rule_based_policy_test" else None
        self.runs, self.save_period = 0, 2

    def reset(self):
        self.delta_counter, self.delta_distance = 0, 0
        self.start_timer, self.stuck_timer = time.time(), 0
        self.position_history = []

    def get_metrics_frames(self):
        try:
            # Handle metrics queue
            if self.mque.length() == self.mque.get_max_len():
                self.metrics = self.mque.get_frames()[0]
                # Handle race finished
                if self.metrics['race_finished']:
                    return self.metrics, []
            else:
                raise queue.Empty

            # Handle screenshot queue
            if self.ique.length() == self.ique.get_max_len():
                self.frames = self.ique.get_frames()
            else:
                raise queue.Empty

            # Initialize position history
            if len(self.position_history) == 0:
                self.position_history = [self.metrics['position']] * self.max_pos_hist

            return self.metrics, self.frames

        except queue.Empty:
            time.sleep(0.02)
            return -1, -1

    def check_state(self):
        # If vehicle is stuck; TODO: How do I check for donuts?
        if abs(self.metrics['front_speed']) < 8:
            if self.delta_counter == 0:
                self.stuck_timer = time.time()
                self.delta_counter += 1
            self.delta_counter += time.time() - self.stuck_timer
        else:
            self.delta_counter = 0
        self.delta_distance = self.metrics['distance']

        # If vehicle stuck for more than N units of continuous 'time', reset
        if self.delta_counter > 50:
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

    def collect_data(self, state, gamepad):
        # Initial state
        if self.state_history is None or self.act_history is None or self.metrics['duration'] < 0:
            return 0

        # Reward Calculation
        metrics = [self.metrics['distance'],
                   self.metrics['front_speed'],
                   int(self.metrics['checkpoint'][1:]) + 1 if self.metrics['checkpoint'][1:] != '' else 0,
                   self.metrics['duration'],
                   sum([(e1-e2)**2 for e1, e2 in zip(self.metrics['position'], self.position_history[0])])
                   ]
        weights = [-0.001, 0.02, 10, -0.001, 0.002]
        reward = sum([weights[i] * metrics[i] for i in range(len(metrics))])
        self.position_history.append(self.metrics['position'])
        self.position_history = self.position_history[-self.max_pos_hist:]
        if state == -1:
            reward += -1000

        # Race finish gives the ultimate reward
        if state == -2:
            reward += 100000

        # Create trajectory
        self.trajectory.append((self.state_history, self.act_history, reward, self.frames))
        if len(self.trajectory) % (self.agent_params['trajectory_maxlen'] // 4) == 0:
            print(f"Trajectory buffer {len(self.trajectory)*100/ self.agent_params['trajectory_maxlen']:.2f}% full!")

        # If a run is completed, reset replay buffer and save (as s_t and s_{t+1} won't match)
        if state != 0:
            self.state_history, self.act_history = None, None

        # Training
        if len(self.trajectory) == self.agent_params['trajectory_maxlen'] \
                and self.policy_str != "rule_based_policy_test":
            # Reset gamepad
            gamepad.reset()
            gamepad.update()

            # Minimize Trackmania to release GPU for training
            remove_focus()

            print("Trajectory full, pause and train!\n")
            self.runs += 1

            # Training Process
            self.agent_conn.send(self.trajectory)
            self.policy_params = self.agent_conn.recv()

            # Once we receive new policy params, open Trackmania again and try out the new brain!
            set_tm_window()
            print("Done! Resuming Agent Exploration with New Policy Weights!\n")
            self.trajectory, self.state_history, self.act_history = [], None, None

            # Save weights to file
            if self.runs % self.save_period == 0:
                today = date.today()
                with open(f'training/saved/{today.strftime("%Y%m%d")}_{self.runs}.params', 'wb') as f:
                    pickle.dump(self.policy_params, f)
            return 1
        return 0


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
    gp.reset()
    gp.update()
    time.sleep(0.1)

    # Respawn
    gp.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
    gp.update()
    time.sleep(0.1)
    gp.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
    gp.update()
    time.sleep(0.1)

    # Release trigger
    gp.right_trigger_float(value_float=0)
    gp.left_trigger_float(value_float=0)
    gp.update()
    time.sleep(0.3)

    # Reset RaceManager
    rm.reset()
