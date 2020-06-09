import cv2
from signals import *

cv2_width_magic = cv2.CAP_PROP_FRAME_WIDTH
cv2_height_magic = cv2.CAP_PROP_FRAME_HEIGHT
cv2_fps_magic = cv2.CAP_PROP_FPS

class VideoReader():
    def __init__(self, video, width=None, height=None):
        self.capture = cv2.VideoCapture(video)
        if not(width is None):
            try:
                self.capture.set(cv2_width_magic, int(width))
            except (ValueError, TypeError):
                raise ErrorSignal("")
            except:
                pass
        if not(height is None):
            try:
                self.capture.set(cv2_height_magic, int(height))
            except (ValueError, TypeError):
                raise ErrorSignal("")
            except:
                pass
        self.width = int(self.capture.get(cv2_width_magic))
        self.height = int(self.capture.get(cv2_height_magic))
        self.fps = float(self.capture.get(cv2_fps_magic))
    def get_width(self):
        return self.width
    def get_height(self):
        return self.height
    def get_fps(self):
        return self.fps
    def read(self):
        ret, frame = self.capture.read()
        if not(ret):
            raise StopIteration()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    def release(self):
        self.capture.release()
