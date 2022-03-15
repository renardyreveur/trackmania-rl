from multiprocessing import Process, Queue

from TMDataGrabber import start_sever
from controller_demo import game

# Queue with maxsize 1 such that the most recent entry is always kept for processing
work_queue = Queue(maxsize=1)


if __name__ == "__main__":
    # Controller Process (Consumer)
    p1 = Process(target=game, args=(work_queue,))

    # Socket Server Process (Producer)
    p2 = Process(target=start_sever, args=(work_queue,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
