from videoreader import VideoReader
from signals import QuitSignal
import cv2
from yolofacedetector import FaceDetector, load_mobilenetv2_224_075_detector
from utils import rel2abs
import numpy as np
import argparse
from faceswapper import FaceSwapper
from insightface import Backbone
import torch
import imageio
from vasyakeypoints import VasyaPointDetector
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

try:
    from camerawriter import CameraWriter
except Exception:
    print("V4L2 library not found, camerawriter feauture would not be avaliable")
    CameraWriter = None
    pass

try:
    import face_alignment
except Exception:
    print("face_alignment library not found, fallback keypointdetector would not be avaliable")
    face_alignment = None
    pass

def read_image(fn):
    try:
        if fn != '':
            image = imageio.imread(fn)
            return image
    except Exception:
        print("Failed to read image")
        exit(1)
    return None

parser = argparse.ArgumentParser(description='One shot face swapper')
parser.add_argument('--input-image', type=str, dest='input_image', default='', help='Input image', required=True)
parser.add_argument('--output-image', type=str, dest='output_image', default='', help='Output image', required=True)
parser.add_argument('--source-image', type=str, dest='source_image', default='', help='Source image', required=True)
parser.add_argument('--refl-coef', type=float, dest='refl_coef', default=3., help='Reflection coeficent')
parser.add_argument('--fallback-point-detector', dest='use_vasya_point_detector', action='store_false', default=True)

args = parser.parse_args()
output_image = args.output_image
input_image = read_image(args.input_image)
refl_coef = args.refl_coef
source_image = read_image(args.source_image)
use_vasya_point_detector = args.use_vasya_point_detector

def detect_single_face(image, detector, keypointdetector):
    faces = rel2abs(detector.detect(image), image)
    assert len(faces) == 1
    keypoints = keypointdetector.get_landmarks(image, detected_faces=faces)
    return faces, keypoints

def main():
    detector = FaceDetector(load_mobilenetv2_224_075_detector("weights/facedetection-mobilenetv2-size224-alpha0.75.h5"), shots_reduce_list = [1, 3], shots_min_width_list = [1, .3])
    if use_vasya_point_detector:
        keypointdetector = VasyaPointDetector()
    else:
        keypointdetector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    source_faces, source_landmarks = detect_single_face(source_image, detector, keypointdetector)
    input_faces, input_landmarks = detect_single_face(input_image, detector, keypointdetector)
    faceswapper = FaceSwapper({'image': source_image, 'keypoints': source_landmarks[0]}, refl_coef=refl_coef)
    result_image = faceswapper.get_image({'image': input_image, 'keypoints': input_landmarks[0]})
    imageio.imwrite(output_image, result_image)
    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except QuitSignal:
        exit(0)
