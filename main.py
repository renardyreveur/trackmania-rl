import multiprocessing
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

import win32api as wapi
import win32gui as wgui
import win32process as wproc

from TMDataGrabber import start_sever
from controller_demo import game
from screengetter import screen_getter


class FrameList(object):
    def __init__(self, max_len):
        self.lock = multiprocessing.Lock()
        self.frames = []
        self.max_len = max_len

    def length(self):
        return len(self.frames)

    def add_frame(self, frame):
        with self.lock:
            self.frames.append(frame)
            self.frames = self.frames[max(0, len(self.frames) - self.max_len):]

    def get_frames(self):
        with self.lock:
            frames, self.frames = self.frames, []
        return frames


# Queue with maxsize 1 such that the most recent entry is always kept for processing
metric_queue = Queue(maxsize=1)


if __name__ == "__main__":
    BaseManager.register('FrameList', FrameList)
    manager = BaseManager()
    manager.start()

    # Queue that holds Screenshots - 10 most recent when gathered at a speed of 40 fps
    image_queue = manager.FrameList(max_len=20)

    # Controller Process (Consumer)
    p1 = Process(target=game, args=(metric_queue, image_queue))

    # Socket Server Process (Producer)
    p2 = Process(target=start_sever, args=(metric_queue,))

    # Screenshot Taking Process (Producer)
    p3 = Process(target=screen_getter, args=(image_queue,))

    # Set focus to the Trackmania screen for input
    handle = wgui.FindWindow(None, "Trackmania")

    if not handle:
        raise ValueError("Trackmania isn't on!")

    # Attach Thread Process to this Python Thread
    remote_thread, _ = wproc.GetWindowThreadProcessId(handle)
    wproc.AttachThreadInput(wapi.GetCurrentThreadId(), remote_thread, True)

    # Focus the window
    prev_handle = wgui.SetFocus(handle)

    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
