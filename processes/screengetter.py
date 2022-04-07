import time

import cv2
import numpy as np
import win32gui as wgui
from PIL import ImageGrab


def screen_getter(im_queue, screenshot_params):
    # Get handle of the Trackmania window
    handle = wgui.FindWindow(None, "Trackmania")

    while True:
        start = time.time()

        # Capture the screen segment and put it in a queue
        x0, y0, x1, y1 = wgui.GetWindowRect(handle)
        if x1 <= 0 or y1 <= 0:
            continue
        frame = ImageGrab.grab(bbox=(0, 32, x1 - 8, y1 - 7)).convert("RGB")
        np_frame = np.asarray(frame)[:, :, ::-1]
        np_frame = cv2.resize(np_frame, screenshot_params['size'][::-1])
        np_frame = np.expand_dims(cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY), 0)
        np_frame = np.expand_dims(np_frame, 0)
        np_frame = np_frame.astype('float32') / 255.
        im_queue.add_frame(np_frame)
        end = time.time()

        elapsed = end - start
        time.sleep(max(1 / screenshot_params['framerate'] - elapsed, 0))
