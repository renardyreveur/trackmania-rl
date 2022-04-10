import os

if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")

from multiprocessing import Process, Queue, Pipe
from multiprocessing.managers import BaseManager

from helpers import FrameList, set_tm_window
from processes.agent.controller import game
from processes.screengetter import screen_getter
from processes.tmdatagrabber import start_sever
from training.train import train

# ===== Parameters =====
POLICY = "neural_policy"
# POLICY = "rule_based_policy_test"

TM_HOST, TM_PORT = "127.0.0.1", 20222

# (H, W)
FRAME_SIZE = (320, 480)

SCREENSHOT_PARAMS = {
    "framerate": 20,
    "size": FRAME_SIZE
}

AGENT_PARAMS = {
    "pretrained": "training/saved/20220409_66.params",
    "trajectory_maxlen": 500,
    "screenshot_maxlen": 5,
    "q_feat_dims": (16, 64),
    "p_feat_dims": (16, 32, 64),
    "screenshot_size": FRAME_SIZE,
    "epochs": 10,
    "batch_size": 25
}
# =======================


if __name__ == "__main__":
    # Queue that holds Screenshots - 10 most recent when gathered at a speed of 40 fps
    BaseManager.register('FrameList', FrameList)
    BaseManager.register('MetricList', FrameList)
    manager = BaseManager()
    manager.start()
    image_queue = manager.FrameList(max_len=AGENT_PARAMS['screenshot_maxlen'])
    # Queue with maxsize 1 such that the most recent entry is always kept for processing
    metric_queue = manager.MetricList(max_len=1)

    # Pipe between agent and training
    agent_conn, trainer_conn = Pipe()

    # Controller Process (Consumer), Socket Server Process (Producer), Screenshot Taking Process (Producer)
    p1 = Process(target=game, args=(metric_queue, image_queue, POLICY, agent_conn, AGENT_PARAMS))
    p2 = Process(target=start_sever, args=(metric_queue, TM_HOST, TM_PORT))
    p3 = Process(target=screen_getter, args=(image_queue, SCREENSHOT_PARAMS))
    p4 = Process(target=train, args=(trainer_conn, AGENT_PARAMS))
    processes = [p1, p2, p3, p4]

    # Set Trackmania window
    set_tm_window()

    # Start the processes
    [p.start() for p in (processes if POLICY != "rule_based_policy_test" else processes[:3])]
    [p.join() for p in (processes if POLICY != "rule_based_policy_test" else processes[:3])]
