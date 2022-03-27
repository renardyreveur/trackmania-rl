from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

from helpers import FrameList, set_tm_window
from processes.agent.controller import game
from processes.screengetter import screen_getter
from processes.tmdatagrabber import start_sever

# Parameters
POLICY = "neural_policy"

TM_HOST, TM_PORT = "127.0.0.1", 20222

SCREENSHOT_FRAMERATE = 20
SCREENSHOT_MAXLEN = 5


# Queue with maxsize 1 such that the most recent entry is always kept for processing
metric_queue = Queue(maxsize=1)


if __name__ == "__main__":
    # Queue that holds Screenshots - 10 most recent when gathered at a speed of 40 fps
    BaseManager.register('FrameList', FrameList)
    manager = BaseManager()
    manager.start()
    image_queue = manager.FrameList(max_len=SCREENSHOT_MAXLEN)

    # Controller Process (Consumer), Socket Server Process (Producer), Screenshot Taking Process (Producer)
    p1 = Process(target=game, args=(metric_queue, image_queue, POLICY))
    p2 = Process(target=start_sever, args=(metric_queue, TM_HOST, TM_PORT))
    p3 = Process(target=screen_getter, args=(image_queue, SCREENSHOT_FRAMERATE))
    processes = [p1, p2, p3]

    # Set Trackmania window
    set_tm_window()

    # Start the processes
    [p.start() for p in processes]
    [p.join() for p in processes]
