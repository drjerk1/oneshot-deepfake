from videoreader import VideoReader
from signals import QuitSignal
import cv2
from yolofacedetector import FaceDetector, load_mobilenetv2_224_075_detector
from utils import rel2abs
from faceid import FACEID
import numpy as np
import argparse
from skimage import io as imageio
from faceswapper import FaceSwapper
from insightface import Backbone
import torch
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


quit_buttons = set([ord('q'), 27, ord('c')])

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
parser.add_argument('--input-video', type=str, dest='input_video', default='', help='Input video')
parser.add_argument('--output-video', type=str, dest='output_video', default='', help='Output video')
parser.add_argument('--output-camera', type=str, dest='output_camera', default='', help='Output camera')
parser.add_argument('--camera-width', type=int, dest='camera_width', default=1280, help='Output camera width')
parser.add_argument('--camera-height', type=int, dest='camera_height', default=720, help='Output camera height')
parser.add_argument('--source-image', type=str, dest='source_image', default='', help='Source image for face swap', required=True)
parser.add_argument('--target-image', type=str, dest='target_image', default='', help='Target image for face swap')
parser.add_argument('--verbose', action='store_true', dest='verbose', default=False, help='View progress in window')
parser.add_argument('--refl-coef', type=float, dest='refl_coef', default=3., help='Reflection coeficent')
parser.add_argument('--fallback-point-detector', dest='use_vasya_point_detector', action='store_false', default=True)
parser.add_argument('--disable-faceid', dest='disable_faceid', action='store_true', default=False)

args = parser.parse_args()
verbose = args.verbose
output_video = args.output_video
output_camera = args.output_camera
camera_width = args.camera_width
camera_height = args.camera_height
input_video = args.input_video
refl_coef = args.refl_coef
if input_video == '':
    input_video = 0
elif input_video.isdigit():
    input_video = int(input_video)
source_image = read_image(args.source_image)
target_image = read_image(args.target_image)
ffmpeg_format = 'mp4v'
use_vasya_point_detector = args.use_vasya_point_detector
disable_faceid = args.disable_faceid

def detect_single_face(image, detector, keypointdetector):
    faces = rel2abs(detector.detect(image), image)
    assert len(faces) == 1
    keypoints = keypointdetector.get_landmarks(image, detected_faces=faces)
    return faces, keypoints

def main():
    vis_debug_flag = False
    if type(input_video) is int:
        videoreader = VideoReader(input_video, width=1920, height=1080)
    else:
        videoreader = VideoReader(input_video)
    if output_video != '':
        print("Warning writing output video...")
        videowriter = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*ffmpeg_format), videoreader.get_fps(), (videoreader.get_width(), videoreader.get_height()))
    else:
        videowriter = None
    if output_camera != '':
        camerawriter = CameraWriter(output_camera, camera_width, camera_height)
    else:
        camerawriter = None

    detector = FaceDetector(load_mobilenetv2_224_075_detector("weights/facedetection-mobilenetv2-size224-alpha0.75.h5"), shots_reduce_list = [1, 3], shots_min_width_list = [1, .3])
    if use_vasya_point_detector:
        keypointdetector = VasyaPointDetector()
    else:
        keypointdetector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    if not(disable_faceid):
        faceid_model = Backbone(50, 0.6, 'ir_se')
        faceid_model.load_state_dict(torch.load("weights/model_ir_se50.pth"))
        faceid = FACEID(faceid_model)
    source_faces, source_landmarks = detect_single_face(source_image, detector, keypointdetector)
    if not(disable_faceid):
        target_faces, target_landmarks = detect_single_face(target_image, detector, keypointdetector)
        faceid.create_face_dict(["target",], [target_image,], target_faces, target_landmarks)
    faceswapper = FaceSwapper({'image': source_image, 'keypoints': source_landmarks[0]}, refl_coef=refl_coef)
    while True:
        frame = videoreader.read()
        vis_image = np.copy(frame)
        faces = rel2abs(detector.detect(frame), frame)

        keypoints = keypointdetector.get_landmarks(frame, detected_faces=faces)
        if not(disable_faceid):
            faceids = faceid.search_faces(frame, faces, keypoints)
        if source_image != '' and target_image != '':
            for i in range(len(faces)):
                if disable_faceid or faceids[i] == 'target':
                    kpts = keypoints[i]
                    vis_image = faceswapper.get_image({'image': vis_image, 'keypoints': kpts})

        if vis_debug_flag:
            for i in range(len(faces)):
                if disable_faceid or faceids[i] == 'target':
                    color = (255,0,0)
                else:
                    color = (0,255,0)
                cv2.rectangle(vis_image, (faces[i][0], faces[i][1]), (faces[i][2], faces[i][3]), color, 1)
                for j in range(len(keypoints[i])):
                    cv2.circle(vis_image, (int(round(keypoints[i][j][0])), int(round(keypoints[i][j][1]))), 3, color)


        if videowriter is not None:
            videowriter.write(cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        if camerawriter is not None:
            camerawriter.write(vis_image)

        if verbose:
            cv2.imshow("faciallib", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            key = (cv2.waitKey(1) & 0xFF)
            if key in quit_buttons:
                if videowriter is not None:
                    videowriter.release()
                    camerawriter.close()
                raise QuitSignal(0)
            if key == ord('v'):
                vis_debug_flag = not(vis_debug_flag)
            if key == ord('n'):
                for i in range(100):
                    _ = videoreader.read()
            if key == ord('a'):
                faceswapper.blurred_debug_flag = False
                faceswapper.keypoints_debug_flag = False
                faceswapper.affine_debug_flag = not(faceswapper.affine_debug_flag)
                faceswapper.use_seamless_clone = not(faceswapper.affine_debug_flag)
            if key == ord('k'):
                faceswapper.blurred_debug_flag = False
                faceswapper.keypoints_debug_flag = not(faceswapper.keypoints_debug_flag)
                faceswapper.use_seamless_clone = not(faceswapper.keypoints_debug_flag)
                faceswapper.affine_debug_flag = False
            if key == ord('b'):
                faceswapper.blurred_debug_flag = not(faceswapper.blurred_debug_flag)
                faceswapper.use_seamless_clone = not(faceswapper.blurred_debug_flag)
                faceswapper.keypoints_debug_flag = False
                faceswapper.affine_debug_flag = False
            if key == ord('s'):
                faceswapper.use_seamless_clone = not(faceswapper.use_seamless_clone)
if __name__ == "__main__":
    try:
        main()
    except QuitSignal:
        exit(0)
