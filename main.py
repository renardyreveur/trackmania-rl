from multiprocessing import Process, Queue

from TMDataGrabber import start_sever
from controller_demo import game

import pygetwindow as gw

# Queue with maxsize 1 such that the most recent entry is always kept for processing
work_queue = Queue(maxsize=1)


if __name__ == "__main__":
    # Controller Process (Consumer)
    p1 = Process(target=game, args=(work_queue,))

    # Socket Server Process (Producer)
    p2 = Process(target=start_sever, args=(work_queue,))

    # Activate Trackmania window
    tmwindow = [x for x in gw.getWindowsWithTitle("Trackmania") if x.title == "Trackmania"]
    if len(tmwindow) == 0:
        raise ValueError("Trackmania isn't on! Please turn on the game!")
    tmwindow[0].activate()

    p1.start()
    p2.start()
    p1.join()
    p2.join()
