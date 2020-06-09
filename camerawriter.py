from fcntl import ioctl
import v4l2
import os
import threading
from queue import Queue
import cv2
import numpy as np

class CameraWriter:
    def open_yuyv_camera(self):
        d = os.open(self.webcam, os.O_RDWR)
        cap = v4l2.v4l2_capability()
        ioctl(d, v4l2.VIDIOC_QUERYCAP, cap)
        vid_format = v4l2.v4l2_format()
        vid_format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        vid_format.fmt.pix.width = self.width
        vid_format.fmt.pix.height = self.height
        #vid_format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YUYV
        vid_format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_RGB24
        vid_format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        vid_format.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_SRGB
        ioctl(d, v4l2.VIDIOC_S_FMT, vid_format)
        return d

    def writer_thread(self):
        def cvt_format(image):
            if self.input_fmt == "RGB24":
                return image
            elif self.input_fmt == "YUY2":
                img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                y0 = np.expand_dims(img_yuv[...,0][::,::2], axis=2)
                u = np.expand_dims(img_yuv[...,1][::,::2], axis=2)
                y1 = np.expand_dims(img_yuv[...,0][::,1::2], axis=2)
                v = np.expand_dims(img_yuv[...,2][::,::2], axis=2)
                img_yuyv = np.concatenate((y0, u, y1, v), axis=2)
                img_yuyv_cvt = img_yuyv.reshape(img_yuyv.shape[0], img_yuyv.shape[1] * 2, int(img_yuyv.shape[2] / 2))
                return img_yuyv_cvt
            else:
                assert False

        while True:
            elem = self.queue.get()
            if elem is None:
                break
            image_data = cvt_format(cv2.resize(elem, (self.width, self.height))).tobytes()
            try:
                os.write(self.d, image_data)
            except Exception:
                break

    def __init__(self, webcam, width, height, input_fmt="RGB24"):
        self.input_fmt = input_fmt
        self.webcam = webcam
        self.width = width
        self.height = height
        self.d = self.open_yuyv_camera()
        self.queue = Queue(maxsize=1)
        self.thread = threading.Thread(target=self.writer_thread)
        self.thread.start()

    def write(self, image):
        self.queue.put(image)

    def close(self):
        self.queue.put(None)
        os.close(self.d)
