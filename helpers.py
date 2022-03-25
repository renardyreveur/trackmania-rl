import multiprocessing
import time

import win32api as wapi
import win32com.client as wclient
import win32gui as wgui
import win32process as wproc

wsh = wclient.Dispatch("WScript.Shell")


class FrameList(object):
    def __init__(self, max_len):
        self.lock = multiprocessing.Lock()
        self.frames = []
        self.max_len = max_len

    def length(self):
        return len(self.frames)

    def get_max_len(self):
        return self.max_len

    def add_frame(self, frame):
        with self.lock:
            self.frames.append(frame)
            self.frames = self.frames[max(0, len(self.frames) - self.max_len):]

    def get_frames(self):
        with self.lock:
            frames, self.frames = self.frames, []
        return frames


def set_tm_window():
    # Set focus to the Trackmania screen for input
    handle = wgui.FindWindow(None, "Trackmania")
    if not handle:
        raise ValueError("Trackmania isn't on!")

    # Attach Thread Process to this Python Thread
    remote_thread, _ = wproc.GetWindowThreadProcessId(handle)
    wproc.AttachThreadInput(wapi.GetCurrentThreadId(), remote_thread, True)

    # Focus the window
    wgui.ShowWindow(handle, 9)  # SW_RESTORE
    wgui.SetFocus(handle)
    rect = wgui.GetWindowRect(handle)

    # Resize the window to a quarter of the screen size and place it top left
    swidth, sheight = wapi.GetSystemMetrics(0), wapi.GetSystemMetrics(1)
    if (rect[2] == swidth) and (rect[3] == sheight):
        wsh.SendKeys("{F11}")
        time.sleep(1)

    wgui.MoveWindow(handle, -7, 0, swidth//2, sheight//2, True)

    return 0
