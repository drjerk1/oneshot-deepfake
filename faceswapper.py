from faceswapper_affine_fast_refl import FaceSwapper as FaceAligner
from biggan import BigGenerator
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from PIL import ImagePath
import time
import math
from copy import deepcopy

IMAGE_SIZE = 256
HIDDEN = 140
MEAN = 255 / 2
STD = 255 / 2
BLUR = 8
EXPAND = 1.2
LITTLE_BLUR = 64
SUBSET = None

MODEL_CURR = "/media/Notebooks/generator_512_deep_fin3_256.state_dict"
MODEL_BEST = "weights/model_best.state_dict"

class FaceSwapper(object):
    def load_biggan_model(self, path):
        generator = BigGenerator(num_images=3,
                              image_size=IMAGE_SIZE,
                              encoder_image_size=IMAGE_SIZE,
                              encoder_num_images=9,
                              ch=512).to(self.device)
        generator.load_state_dict(torch.load(path))
        generator.eval()
        return generator

    @staticmethod
    def cut(image, face):
        target_size = IMAGE_SIZE
        padding = 0
        w = face[2] - face[0] + 1
        h = face[3] - face[1] + 1
        w *= (1 + padding)
        h *= (1 + padding)
        w = int(round(w))
        h = int(round(h))
        cx = (face[0] + face[2]) // 2
        cy = (face[1] + face[3]) // 2
        s = max(w, h)
        cx -= min(cx - s//2, 0)
        cx += min(image.shape[1] - (cx + (s+1)//2), 0)
        cy -= min(cy - s//2, 0)
        cy += min(image.shape[0] - (cy + (s+1)//2), 0)
        s += 2 * min(min(cx - s//2, 0), min(image.shape[1] - (cx + (s+1)//2), 0),
                      min(cy - s//2, 0), min(image.shape[0] - (cy + (s+1)//2), 0))

        face = [cx - s//2, cy - s//2, cx + (s+1)//2, cy + (s+1)//2]
        assert face[0] >= 0 and face[1] >= 0 and face[2] <= image.shape[1] and face[3] <= image.shape[0]

        cut_face = image[face[1]:face[3], face[0]:face[2]]

        return cv2.resize(cut_face, (target_size, target_size)), face

    @staticmethod
    def recalc_keypoints(keypoints, box):
        keypoints = np.array(keypoints.copy(), dtype=np.float)
        size = box[3] - box[1]
        keypoints[:, 0] -= box[0]
        keypoints[:, 1] -= box[1]
        keypoints /= size
        keypoints *= IMAGE_SIZE

        return np.array(np.round(keypoints), dtype=np.int)

    @staticmethod
    def expand(pts, k=1):
        pts = np.array(pts.copy(), dtype=np.float)
        center = pts.mean(axis=0)
        pts -= center
        pts *= k
        pts += center
        return np.array(np.round(pts), dtype=np.int)

    def resize_blur(self, image, blur):
        size = image.shape[0]
        return cv2.resize(cv2.resize(image, (blur, blur)), (size, size))

    def face_polygon(self, keypoints, e=1):
        keypoints = np.array([(x, y) for x, y in keypoints[0:17]] +\
                             [(keypoints[i][0], keypoints[i][1]) for i in [26, 25, 24, 23, 22]] +\
                             [(keypoints[i][0], keypoints[i][1]) for i in [21, 20, 19, 18, 17]])
        return [(x, y) for x, y in self.expand(keypoints, e)]

    def face_polygon_left(self, keypoints, e=1):
        keypoints = np.array([(x, y) for x, y in keypoints[0:9]] +\
                             [(keypoints[i][0], keypoints[i][1]) for i in [57, 66, 62, 51, 33, 30, 29, 28, 27]] +\
                             [(keypoints[i][0], keypoints[i][1]) for i in [21, 20, 19, 18, 17]])
        return [(x, y) for x, y in self.expand(keypoints, e)]

    def face_polygon_right(self, keypoints, e=1):
        keypoints = np.array([(x, y) for x, y in keypoints[8:17]] +\
                             [(keypoints[i][0], keypoints[i][1]) for i in [26, 25, 24, 23, 22]] +\
                             [(keypoints[i][0], keypoints[i][1]) for i in [27, 28, 29, 30, 33, 51, 62, 66, 57]])
        return [(x, y) for x, y in self.expand(keypoints, e)]

    def mask_face(self, image, keypoints, box):
        def left_eye_polygon(keypoints):
            return [(x, y) for x, y in self.expand(keypoints[36:42], 1.5)]

        def right_eye_polygon(keypoints):
            return [(x, y) for x, y in self.expand(keypoints[42:48], 1.5)]

        def left_eyebrow_polygon(keypoints):
            return [(x, y) for x, y in self.expand(keypoints[17:22], 2)]

        def right_eyebrow_polygon(keypoints):
            return [(x, y) for x, y in self.expand(keypoints[22:27], 2)]

        def up_mouth(keypoints):
            keypoints = np.array([(x, y) for x, y in keypoints[48:55]] + \
                        [(keypoints[i][0], keypoints[i][1]) for i in [64, 63, 62, 61, 60]])
            return [(x, y) for x, y in self.expand(keypoints, 1)]

        def down_mouth(keypoints):
            keypoints = np.array([(x, y) for x, y in keypoints[54:60]] + \
                        [(keypoints[48][0], keypoints[48][1])] + \
                        [(keypoints[60][0], keypoints[60][1])] + \
                        [(keypoints[i][0], keypoints[i][1]) for i in [67, 66, 65, 64]])
            return [(x, y) for x, y in self.expand(keypoints, 1)]

        def mouth(keypoints):
            keypoints = np.array([(x, y) for x, y in keypoints[48:60]])
            return [(x, y) for x, y in self.expand(keypoints, 1)]

        def nose_line(keypoints):
            angle = math.atan2(keypoints[30][0] - keypoints[27][0], keypoints[30][1] - keypoints[27][1])
            keypoints = [
                (keypoints[27][0], keypoints[27][1]),
                (keypoints[27][0] + math.sin(angle - 7 / 180 * math.pi) * 70, keypoints[27][1] + math.cos(angle - 7 / 180 * math.pi) * 70),
                (keypoints[27][0] + math.sin(angle + 7 / 180 * math.pi) * 70, keypoints[27][1] + math.cos(angle + 7 / 180 * math.pi) * 70)
            ]
            return [(x, y) for x, y in self.expand(keypoints, 1)]

        oimg = np.copy(image)
        oimg = self.resize_blur(oimg, LITTLE_BLUR)
        oimg = np.array(oimg.mean(-1), dtype=np.uint8).reshape((IMAGE_SIZE, IMAGE_SIZE, 1)).repeat(3, axis=-1)
        pts = self.recalc_keypoints(keypoints, box)
        img = Image.fromarray(self.resize_blur(image, LITTLE_BLUR))
        draw = ImageDraw.Draw(img)
        draw.polygon(self.face_polygon(pts, e=1), fill ="#ff0000")
        draw.polygon(self.face_polygon_left(pts, e=1), fill ="#ff0000")
        draw.polygon(self.face_polygon_right(pts, e=1), fill ="#ff0000")
        draw.polygon(nose_line(pts), fill ="#ffff00") 
        img = np.asarray(img)
        trans_mask = Image.fromarray(np.zeros_like(oimg))
        draw = ImageDraw.Draw(trans_mask)
        left_eye = np.array(left_eye_polygon(pts))
        left_eye_left_part = np.array([left_eye[0], left_eye[1], left_eye[1] / 2. + left_eye[2] / 2.,\
                                       left_eye[4] / 2. + left_eye[5] / 2., left_eye[5]])
        left_eye_right_part = np.array([left_eye[1] / 2. + left_eye[2] / 2., left_eye[2], left_eye[3], left_eye[4],\
                                       left_eye[4] / 2. + left_eye[5] / 2.])
        draw.polygon([(x, y) for x, y in left_eye], fill ="#ffffff")
        draw.polygon([(x, y) for x, y in left_eye_left_part], fill ="#ffffff")
        draw.polygon([(x, y) for x, y in left_eye_right_part], fill ="#ffffff")

        right_eye = np.array(right_eye_polygon(pts))
        right_eye_left_part = np.array([right_eye[0], right_eye[1], right_eye[1] / 2. + right_eye[2] / 2.,\
                                       right_eye[4] / 2. + right_eye[5] / 2., right_eye[5]])
        right_eye_right_part = np.array([right_eye[1] / 2. + right_eye[2] / 2., right_eye[2], right_eye[3], right_eye[4],\
                                       right_eye[4] / 2. + right_eye[5] / 2.])
        draw.polygon([(x, y) for x, y in right_eye], fill ="#ffffff")
        draw.polygon([(x, y) for x, y in right_eye_left_part], fill ="#ffffff")
        draw.polygon([(x, y) for x, y in right_eye_right_part], fill ="#ffffff")

        mouth_poly = np.array(mouth(pts))
        mouth_left_part = mouth_poly[[0, 1, 2, 3, 9, 10, 11]]
        mouth_right_part = mouth_poly[[3, 4, 5, 6, 7, 8, 9]]
        draw.polygon([(x, y) for x, y in mouth_poly], fill ="#ffffff")
        draw.polygon([(x, y) for x, y in mouth_left_part], fill ="#ffffff")
        draw.polygon([(x, y) for x, y in mouth_right_part], fill ="#ffffff")
        trans_mask = np.asarray(trans_mask)
        trans_mask = (np.array(trans_mask, dtype=np.float) / 255.)[:, :, :1]
        img = oimg * trans_mask + img * (1 - trans_mask)
        img = np.array(img, dtype=np.uint8)
        return img

    def face_border(self, image, keypoints, box, e=EXPAND):
        pts = self.recalc_keypoints(keypoints, box)
        mask = Image.fromarray(np.zeros_like(image))
        draw = ImageDraw.Draw(mask)
        draw.polygon(self.face_polygon(pts, e=e), fill ="#ffffff")
        draw.polygon(self.face_polygon_left(pts, e=1), fill ="#ffffff")
        draw.polygon(self.face_polygon_right(pts, e=1), fill ="#ffffff")
        mask = np.asarray(mask)[:, :, :1]
        img = image * (np.array(mask, dtype=np.float) / 255.)
        img = np.array(img, dtype=np.uint8)
        return img, mask

    def blurred_face_border(self, image, keypoints, box, s=1, e=1.07, st=7):
        pts = self.recalc_keypoints(keypoints, box)
        mask = Image.fromarray(np.zeros_like(image))
        draw = ImageDraw.Draw(mask)
        for f in np.linspace(e, s, st):
            c = min(max(int((1-(f-s)/(e-s))*255), 0), 255)
            draw.polygon(self.face_polygon(pts, e=f), fill=c)
            draw.polygon(self.face_polygon_left(pts, e=f), fill=c)
            draw.polygon(self.face_polygon_right(pts, e=f), fill=c)
        mask = np.asarray(mask)[:, :, :1]
        img = image * (np.array(mask, dtype=np.float) / 255.)
        img = np.array(img, dtype=np.uint8)
        return img, mask

    @staticmethod
    def norm(image):
        return (image.float() - MEAN) / STD

    @staticmethod
    def denorm(image):
        return (image * STD + MEAN).int()

    @staticmethod
    def dnorm(image):
        image = np.array(image, dtype=np.float)
        image -= image.min((0, 1, 2)).reshape((1, 1, 1))
        image /= image.max((0, 1, 2)).reshape((1, 1, 1))
        return np.array(image * 255, dtype=np.int)

    def __init__(self, source, use_seamless_clone=True, device='cuda:0', path=MODEL_BEST, refl_coef=1.5):
        self.device = device
        self.affine_debug_flag = False
        self.blurred_debug_flag = False
        self.keypoints_debug_flag = False
        self.use_seamless_clone = use_seamless_clone
        self.model = self.load_biggan_model(path)
        self.random_vec = torch.distributions.Normal(torch.zeros(1, HIDDEN), torch.ones(1, HIDDEN)).sample().float().to(self.device)
        self.source = source
        self.affine_transformer = FaceAligner(source, norm_contrast=False, subset=SUBSET, refl_coef=refl_coef)

    def get_image(self, target):
        keypoints = np.array(target['keypoints'], dtype=np.int)
        image = np.copy(target["image"])
        face = ((np.min(keypoints[:, 0]),\
                 np.min(keypoints[:, 1]),\
                 np.max(keypoints[:, 0]),\
                 np.max(keypoints[:, 1])))

        target_align = deepcopy(target)
        target_align['image'] = np.zeros_like(target_align['image'])
        affine = self.affine_transformer.get_image(target_align)
        affine_condition, cut_face = self.cut(affine, face)
        cutted_face = self.cut(image, face)[0]

        cutted_face, _ = self.face_border(cutted_face, keypoints, cut_face)
        _, mask = self.blurred_face_border(cutted_face, keypoints, cut_face)

        keypoints_condition = self.mask_face(cutted_face, keypoints, cut_face)

        affine_condition = self.norm(torch.tensor(affine_condition).transpose(0, 2)).view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        blurred_condition = self.norm(torch.tensor(self.resize_blur(cutted_face, BLUR)).transpose(0, 2)).view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        keypoints_condition = self.norm(torch.tensor(keypoints_condition).transpose(0, 2)).view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        condition = torch.cat((keypoints_condition, blurred_condition, affine_condition), dim=1).to(self.device)
        with torch.no_grad():
            swapped = self.model(x=self.random_vec, z=condition)

        if self.affine_debug_flag:
            swapped = affine_condition
        elif self.blurred_debug_flag:
            swapped = blurred_condition
        elif self.keypoints_debug_flag:
            swapped = keypoints_condition
        swapped = self.denorm(swapped)[0].transpose(0, 2).cpu().float().numpy()
        swapped = np.array(swapped, dtype=np.uint8)

        if self.use_seamless_clone:
            swapped_image_cut = image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])]
            if swapped_image_cut.shape[0] != swapped.shape[0] or swapped_image_cut.shape[1] != swapped.shape[1]:
                swapped_image_cut = cv2.resize(swapped_image_cut, (swapped.shape[0], swapped.shape[1]))

            swapped = np.array(cv2.seamlessClone(swapped,\
                                                 swapped_image_cut,\
                                                 np.ones_like(swapped) * mask,
                                                 (swapped.shape[0]//2, swapped.shape[1]//2),\
                                                 cv2.NORMAL_CLONE), dtype=np.uint8)

            if swapped.shape[0] != int(cut_face[2]) - int(cut_face[0]) or swapped.shape[1] != int(cut_face[3]) - int(cut_face[1]):
                swapped = cv2.resize(swapped, (int(cut_face[2]) - int(cut_face[0]), int(cut_face[3]) - int(cut_face[1])), interpolation=cv2.INTER_LINEAR)

            image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])] = swapped
        else:
            if swapped.shape[0] != int(cut_face[2]) - int(cut_face[0]) or swapped.shape[1] != int(cut_face[3]) - int(cut_face[1]):
                swapped = cv2.resize(swapped, (int(cut_face[2]) - int(cut_face[0]), int(cut_face[3]) - int(cut_face[1])))

            if mask.shape[0] != int(cut_face[2]) - int(cut_face[0]) or mask.shape[1] != int(cut_face[3]) - int(cut_face[1]):
                mask = cv2.resize(mask, (int(cut_face[2]) - int(cut_face[0]), int(cut_face[3]) - int(cut_face[1])), interpolation=cv2.INTER_NEAREST)
            mask = np.array(mask.reshape((mask.shape[0], mask.shape[1], 1)), dtype=np.float) / 255.

            image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])] =\
                np.array(\
                        swapped * mask +\
                        image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])] * (1 - mask)\
                ,dtype=np.uint8)

        return image
