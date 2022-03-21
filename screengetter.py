import time

import numpy as np
import win32api as wapi
import win32gui as wgui
from PIL import ImageGrab

FRAMERATE = 20


def screen_getter(im_queue):
    # Get handle of the Trackmania window
    handle = wgui.FindWindow(None, "Trackmania")

    # Resize the window to a quarter of the screen size and place it top left
    swidth, sheight = wapi.GetSystemMetrics(0), wapi.GetSystemMetrics(1)
    wgui.MoveWindow(handle, -7, 0, swidth//2, sheight//2, True)

    while True:
        start = time.time()
        # Capture the screen segment and put it in a queue
        x0, y0, x1, y1 = wgui.GetWindowRect(handle)
        frame = ImageGrab.grab(bbox=(0, 32, x1-8, y1-7)).convert("RGB")
        np_frame = np.asarray(frame)[:, :, ::-1]
        end = time.time()

        elapsed = end - start
        time.sleep(1/FRAMERATE - elapsed)
        im_queue.add_frame(np_frame)
