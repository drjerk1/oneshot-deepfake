import math
import numpy as np

def r(x):
    return int(round(x))

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1).reshape((x.shape[0], 1)))
    return e_x / np.sum(e_x, axis=-1).reshape((x.shape[0], 1))

def cosine_dist_norm(a, b):
    return 1 - np.sum(a * b, axis=-1)

def rel2abs(faces, frame):
    for i in range(len(faces)):
        lx = r(faces[i][0] * frame.shape[1])
        ly = r(faces[i][1] * frame.shape[0])
        rx = r(faces[i][2] * frame.shape[1])
        ry = r(faces[i][3] * frame.shape[0])
        faces[i] = (lx, ly, rx, ry)
    return faces
