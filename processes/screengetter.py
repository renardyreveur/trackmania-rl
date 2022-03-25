import time

import numpy as np
import win32gui as wgui
from PIL import ImageGrab


def screen_getter(im_queue, framerate):
    # Get handle of the Trackmania window
    handle = wgui.FindWindow(None, "Trackmania")

    while True:
        start = time.time()

        # Capture the screen segment and put it in a queue
        x0, y0, x1, y1 = wgui.GetWindowRect(handle)
        frame = ImageGrab.grab(bbox=(0, 32, x1-8, y1-7)).convert("RGB")
        np_frame = np.asarray(frame)[:, :, ::-1]
        im_queue.add_frame(np_frame)
        end = time.time()

        elapsed = end - start
        time.sleep(max(1/framerate - elapsed, 0))
