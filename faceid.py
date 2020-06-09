from utils import r
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import random
from skimage import transform as trans
import cv2
import math
from face_align import estimate_norm

class FACEID:
    @staticmethod
    def norm(batch, mean=127.5, std=127.5):
        return (batch.float() - mean) / std

    @staticmethod
    def read_image(fp):
        return np.asarray(Image.open(fp), dtype=np.uint8)

    @staticmethod
    def d(x, y):
      return 1 - (x * y).sum(-1) / (np.sqrt((x * x).sum(-1)) * np.sqrt((y * y).sum(-1)))

    @staticmethod
    def kpts_68_to_5(pts):
        return np.array([[pts[36:42, 0].mean(), pts[36:42, 1].mean()],
                         [pts[42:48, 0].mean(), pts[42:48, 1].mean()],
                         [pts[30, 0], pts[30, 1]],
                         [pts[48, 0], pts[48, 1]],
                         [pts[54, 0], pts[54, 1]]])

    def norm_crop(self, img, landmark, image_size):
      M, pose_index = estimate_norm(landmark)
      warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue = 0.0)
      return warped

    def align(self, image, face, target_size, landmarks):
        if landmarks.shape[0] == 68:
            landmarks = self.kpts_68_to_5(landmarks)
        elif landmarks.shape[0] != 5:
            assert False
        return self.norm_crop(image, landmarks, target_size)

    def embed_faces(self, image, faces, landmarks):
        if landmarks is None:
            landmarks = []
        assert len(faces) == len(landmarks)
        ids = []
        for i, face in enumerate(faces):
            face_image = self.align(image, face, self.image_size, landmarks[i])
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            face_image = self.norm(torch.tensor(face_image).permute(2, 0, 1).unsqueeze(0))
            with torch.no_grad():
                ids.append(self.model(face_image.to(self.device)).squeeze(0).detach().cpu().numpy())
        return np.array(ids)

    def search_faces(self, image, faces, landmarks):
        if self.faces is None:
            return [None,] * len(faces)
        if landmarks is None:
            landmarks = []
        assert len(faces) == len(landmarks)
        faceids = self.embed_faces(image, faces, landmarks)
        res = []
        for faceid in faceids:
            faceid = torch.tensor(faceid)
            faceid = faceid / torch.sqrt((faceid * faceid).sum(-1)).unsqueeze(0)
            if self.faces.shape[0] >self.cuda_find_thres:
                faceid = faceid.to(self.device)
            scores = (self.faces * faceid).sum(-1).cpu()
            most_similar = torch.argmax(scores).item()
            dist = 1. - scores[most_similar].item()
            if dist <= self.threshold:
                res.append(self.ids[most_similar])
            else:
                res.append(None)
        return res

    def create_face_dict(self, ids, images, faces, landmarks):
        if landmarks is None:
            landmarks = []
        assert len(ids) == len(images) and len(images) == len(faces) and len(faces) == len(landmarks)
        self.faces = torch.zeros((len(ids), self.dim)).float()
        self.ids = ids
        for i in range(len(ids)):
            faceid = self.embed_faces(images[i], [faces[i],], [landmarks[i],])[0]
            faceid = torch.tensor(faceid)
            self.faces[i] = faceid / torch.sqrt((faceid * faceid).sum(-1))
        if self.faces.shape[0] > self.cuda_find_thres:
            self.faces = self.faces.to(self.device)

    def __init__(self, model, device='cuda:0', dim=512, thres=0.6940869408694087, image_size=(112, 112), cuda_find_thres=10000):
        self.dim = dim
        self.image_size = image_size
        self.faces = None
        self.model = model.to(device)
        self.model.eval()
        self.threshold = thres
        self.device = device
        self.cuda_find_thres = cuda_find_thres
