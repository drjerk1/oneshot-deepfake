from tensorflow.python.keras.layers import Conv2D, Input, ZeroPadding2D, Dense, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
import math
import numpy as np
import cv2

def load_mobilenetv2_224_075_detector(path):
    input_tensor = Input(shape=(224, 224, 3))
    output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=0.75).output
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(path)

    return model

def up(a, b):
    return (a + b - 1) // b

def r(x):
    return int(round(x))

def sum_lists(l):
    su = []
    for i in l:
        su += i
    return su

def transpose_shots(shots):
    return [(shot[1], shot[0], shot[3], shot[2], shot[4]) for shot in shots]

def simple_shot_scheme(w, h, min_intersection=0.025, min_w = 1.):
  if w > h:
    return transpose_shots(simple_shot_scheme(h, w, min_intersection, min_w))

  min_intersection = r(min_intersection * h)
  x = max(up(h, w), up((h + min_intersection), (w - min_intersection)))
  intersection = int(math.ceil((x * w - h) / (x - 1)))
  assert intersection >= min_intersection and x >= up(h, w)

  return [(0., (i * w - i * intersection) / h, 1., ((i + 1) * w - i * intersection) / h - (i * w - i * intersection) / h, min_w) for i in range(0, x)]

def smaller_shot_scheme(w, h, k, min_intersection=0.025, min_w = 1.):
  if k == 1:
    return simple_shot_scheme(w, h, min_intersection, min_w)

  if w > h:
    return transpose_shots(smaller_shot_scheme(h, w, k, min_intersection, min_w))

  w_scheme = simple_shot_scheme(w, int(math.ceil(w / k + min_intersection * h)), min_intersection, min_w)
  w_s = int(math.ceil(w_scheme[0][2] * w))
  h_scheme = simple_shot_scheme(w_s, h, min_intersection, min_w)

  return sum_lists([[(w[0], h[1], w[2], h[3], min_w) for w in w_scheme] for h in h_scheme])

def shot_scheme(w, h, k_l, min_w_l, min_intersection=0.025):
  if type(k_l) is not list:
    k_l = [k_l,]

  if type(min_w_l) is not list:
    min_w_l = [min_w_l,]

  assert len(k_l) == len(min_w_l)

  return sum_lists([smaller_shot_scheme(w, h, k, min_intersection, min_w) for k, min_w in zip(k_l, min_w_l)])

def sigmoid(x):
    return 1 / (np.exp(-x + 1e-9) + 1)

def non_max_suppression(boxes, p, iou_threshold):
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort(p)
    true_boxes_indexes = []

    while len(indexes) > 0:
        true_boxes_indexes.append(indexes[-1])

        intersection = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0) * np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        iou = intersection / ((x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]]) + (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]]) - intersection)

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, np.where(iou >= iou_threshold)[0])

    return boxes[true_boxes_indexes]

def union_suppression(boxes, threshold):
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort((x2 - x1) * (y2 - y1))
    result_boxes = []

    while len(indexes) > 0:
        intersection = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0) * np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        min_s = np.minimum((x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]]), (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]]))
        ioms = intersection / (min_s + 1e-9)
        neighbours = np.where(ioms >= threshold)[0]
        if len(neighbours) > 0:
            result_boxes.append([min(np.min(x1[indexes[neighbours]]), x1[indexes[-1]]), min(np.min(y1[indexes[neighbours]]), y1[indexes[-1]]), max(np.max(x2[indexes[neighbours]]), x2[indexes[-1]]), max(np.max(y2[indexes[neighbours]]), y2[indexes[-1]])])
        else:
            result_boxes.append([x1[indexes[-1]], y1[indexes[-1]], x2[indexes[-1]], y2[indexes[-1]]])

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, neighbours)

    return result_boxes

class FaceDetector():
    def __init__(self, model, shots_reduce_list = [1], shots_min_width_list = [1], min_intersection=0.025, image_size=224, grids=7, iou_threshold=0.1, union_threshold=0.1, prob_threshold=0.4, one_face=False):
        self.model = model
        self.image_size = image_size
        self.grids = grids
        self.iou_threshold = iou_threshold
        self.union_threshold = union_threshold
        self.prob_threshold = -1 if prob_threshold is None else prob_threshold
        self.min_intersection = min_intersection
        self.shots_reduce_list = shots_reduce_list
        self.shots_min_width_list = shots_min_width_list
        self.one_face = one_face

    def detect(self, frame):
        original_frame_shape = frame.shape
        shots = shot_scheme(frame.shape[1], frame.shape[0], self.shots_reduce_list, self.shots_min_width_list, self.min_intersection)

        aspect_ratio = frame.shape[1] / frame.shape[0]
        c = min(frame.shape[0], frame.shape[1] / aspect_ratio)
        slice_h_shift = r((frame.shape[0] - c) / 2)
        slice_w_shift = r((frame.shape[1] - c * aspect_ratio) / 2)
        if slice_w_shift != 0 and slice_h_shift == 0:
            frame = frame[:, slice_w_shift:-slice_w_shift]
        elif slice_w_shift == 0 and slice_h_shift != 0:
            frame = frame[slice_h_shift:-slice_h_shift, :]

        frames = []
        for s in shots:
            frames.append(cv2.resize(frame[r(s[1] * frame.shape[0]):r((s[1] + s[3]) * frame.shape[0]), r(s[0] * frame.shape[1]):r((s[0] + s[2]) * frame.shape[1])], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST))
        frames = np.array(frames)

        predictions = self.model.predict(frames, batch_size=len(frames), verbose=0)

        boxes = []
        for i in range(len(shots)):
            slice_boxes = []
            slice_prob = []
            for j in range(predictions.shape[1]):
                for k in range(predictions.shape[2]):
                    p = sigmoid(predictions[i][j][k][4])
                    if not(p is None) and p > self.prob_threshold:
                        px = sigmoid(predictions[i][j][k][0])
                        py = sigmoid(predictions[i][j][k][1])
                        pw = min(math.exp(predictions[i][j][k][2] / self.grids), self.grids)
                        ph = min(math.exp(predictions[i][j][k][3] / self.grids), self.grids)
                        if not(px is None) and not(py is None) and not(pw is None) and not(ph is None) and pw > 1e-9 and ph > 1e-9:
                            cx = (px + j) / self.grids
                            cy = (py + k) / self.grids
                            wx = pw / self.grids
                            wy = ph / self.grids
                            if wx <= shots[i][4] and wy <= shots[i][4]:
                                lx = min(max(cx - wx / 2, 0), 1)
                                ly = min(max(cy - wy / 2, 0), 1)
                                rx = min(max(cx + wx / 2, 0), 1)
                                ry = min(max(cy + wy / 2, 0), 1)

                                lx *= shots[i][2]
                                ly *= shots[i][3]
                                rx *= shots[i][2]
                                ry *= shots[i][3]

                                lx += shots[i][0]
                                ly += shots[i][1]
                                rx += shots[i][0]
                                ry += shots[i][1]

                                slice_boxes.append([lx, ly, rx, ry])
                                slice_prob.append(p)

            slice_boxes = np.array(slice_boxes)
            slice_prob = np.array(slice_prob)

            if self.iou_threshold is not None:
                slice_boxes = non_max_suppression(slice_boxes, slice_prob, self.iou_threshold)
            else:
                order = np.argsort(-1 * slice_prob)
                slice_boxes = slice_boxes[order]
                slice_prob = slice_prob[order]

            for ii in range(len(slice_boxes)):
                boxes.append(slice_boxes[ii])

        boxes = np.array(boxes)
        if self.iou_threshold is not None:
            boxes = union_suppression(boxes, self.union_threshold)

        for i in range(len(boxes)):
            boxes[i][0] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][1] /= original_frame_shape[0] / frame.shape[0]
            boxes[i][2] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][3] /= original_frame_shape[0] / frame.shape[0]

            boxes[i][0] += slice_w_shift / original_frame_shape[1]
            boxes[i][1] += slice_h_shift / original_frame_shape[0]
            boxes[i][2] += slice_w_shift / original_frame_shape[1]
            boxes[i][3] += slice_h_shift / original_frame_shape[0]

        return list(boxes)
