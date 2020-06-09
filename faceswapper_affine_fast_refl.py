# Each 3 vertices and mirror index

import cv2
import numpy as np

TRIANGLE_FACE_MODEL = [
                       (22, 23, 28, 1),
                       (22, 40, 28, 3), (23, 28, 43, 2),
                       (40, 28, 29, 5), (43, 28, 29, 4),
                       (40, 29, 30, 7), (29, 30, 43, 6),
                       (40, 30, 32, 9), (43, 30, 36, 8),
                       (30, 31, 32, 11), (30, 31, 36, 10),
                       (31, 32, 33, 13), (31, 36, 35, 12),
                       (31, 33, 34, 15), (31, 35, 34, 14),
                       (34, 51, 52, 17), (34, 53, 52, 16),
                       (33, 34, 51, 19), (34, 35, 53, 18),
                       (32, 33, 51, 21), (35, 36, 53, 20),
                       (32, 49, 50, 23), (36, 55, 54, 22),
                       (32, 50, 51, 25), (36, 53, 54, 24),
                       (51, 52, 63, 27), (52, 53, 63, 26),
                       (51, 62, 63, 29), (53, 64, 63, 28),
                       (50, 51, 62, 31), (53, 54, 64, 30),
                       (50, 61, 62, 33), (64, 65, 54, 32),
                       (49, 50, 61, 35), (54, 55, 65, 34),
                       (61, 62, 68, 37), (64, 65, 66, 36),
                       (62, 68, 67, 39), (64, 66, 67, 38),
                       (62, 63, 67, 41), (63, 64, 67, 40),
                       (49, 61, 60, 43), (65, 55, 56, 42),
                       (61, 60, 68, 45), (65, 66, 56, 44),
                       (60, 68, 59, 47), (66, 56, 57, 46),
                       (68, 67, 59, 49), (67, 66, 57, 48),
                       (59, 67, 58, 51), (58, 67, 57, 50),
                       (59, 58, 9, 53), (58, 57, 9, 52),
                       (59, 8, 9, 55), (57, 10, 9, 54),
                       (59, 7, 8, 57), (57, 11, 10, 56),
                       (60, 59, 7, 59), (57, 56, 11, 58),
                       (49, 60, 7, 61), (55, 56, 11, 60),
                       (55, 11, 12, 63), (49, 6, 7, 62),
                       (49, 5, 6, 65), (55, 12, 13, 64),
                       (55, 13, 14, 67), (49, 4, 5, 66),
                       (3, 4, 49, 69), (14, 15, 55, 68),
                       (3, 32, 49, 71), (15, 55, 36, 70),
                       (42, 3, 32, 73), (47, 36, 15, 72),
                       (43, 48, 36, 75), (41, 40, 32, 74),
                       (41, 42, 32, 77), (47, 48, 36, 76),
                       (39, 40, 41, 79), (43, 44, 48, 78),
                       (38, 39, 41, 81), (44, 45, 48, 80),
                       (41, 42, 38, 83), (45, 47, 48, 82),
                       (37, 38, 42, 85), (45, 46, 47, 84),
                       (39, 40, 22, 87), (23, 43, 44, 86),
                       (23, 24, 44, 89), (21, 22, 39, 88),
                       (24, 25, 44, 91), (20, 21, 39, 90),
                       (20, 38, 39, 93), (25, 44, 45, 92),
                       (19, 20, 38, 95), (25, 26, 45, 94),
                       (19, 37, 38, 97), (26, 45, 46, 96),
                       (3, 37, 42, 99), (15, 46, 47, 98),
                       (2, 3, 37, 101), (15, 16, 46, 100),
                       (2, 18, 37, 103), (16, 46, 27, 102),
                       (1, 2, 18, 105), (27, 16, 17, 104),
                       (18, 19, 37, 107), (26, 27, 46, 106)
                      ]

MIRRORED_POINTS = [
    (1, 17),
    (2, 16),
    (3, 15),
    (4, 14),
    (5, 13),
    (6, 12),
    (7, 11),
    (8, 10),
    (9, 9),
    (32, 36),
    (33, 35),
    (34, 34),
    (28, 28),
    (29, 29),
    (30, 30),
    (31, 31),
    (18, 27),
    (19, 26),
    (20, 25),
    (21, 24),
    (22, 23),
    (37, 46),
    (38, 45),
    (39, 44),
    (40, 43),
    (42, 47),
    (41, 48),
    (49, 55),
    (50, 54),
    (51, 53),
    (52, 52),
    (60, 56),
    (59, 57),
    (58, 58),
    (61, 65),
    (62, 64),
    (63, 63),
    (68, 66),
    (67, 67)
]

class FaceSwapperTriangular():
    def draw_order(self, points):
        s = []
        for triangle in TRIANGLE_FACE_MODEL:
            v1 = points[triangle[1] - 1] - points[triangle[0] - 1]
            v2 = points[triangle[2] - 1] - points[triangle[0] - 1]
            s.append(abs(v1[0] * v2[1] - v1[1] * v2[0]))
        return np.argsort(s)

    def triangle_points(self, triangle, points):
        return np.array([
            points[triangle[0] - 1],
            points[triangle[1] - 1],
            points[triangle[2] - 1]
        ])

    def masked_mean(self, image, mask):
        return (image * mask).sum((0, 1)) / mask.sum((0, 1))

    def masked_std(self, image, mask, mean):
        return np.sqrt(((image * mask - mean)**2).sum((0, 1)) / mask.sum((0, 1)))

    def __init__(self, source, norm_contrast=False, subset=None, draw_lines=False, reflect_transform=True, mean_only=False, refl_coef=1.5, cut_pad=10):
        assert subset is None
        self.refl_coef = refl_coef
        self.cut_pad = cut_pad
        self.draw_lines = draw_lines
        self.mean_only = mean_only
        self.norm_contrast = norm_contrast
        self.reflect_transform = reflect_transform
        self.source_image = np.copy(source["image"])
        self.source_image_flipped = np.copy(cv2.flip(self.source_image, 1))
        self.source_points = np.copy(np.array(source["keypoints"], dtype=np.float32))
        self.source_points_flipped = np.copy(self.source_points)
        self.source_points_flipped[:, 0] = self.source_image.shape[1] - self.source_points_flipped[:, 0]
        for i, j in MIRRORED_POINTS:
            t = np.copy(self.source_points_flipped[i - 1])
            self.source_points_flipped[i - 1] = self.source_points_flipped[j - 1]
            self.source_points_flipped[j - 1] = t

    def get_image(self, target):
        def r(x):
            return int(round(x))

        def S(t):
            v1 = t[2] - t[0]
            v2 = t[1] - t[0]
            return abs(v1[0] * v2[1] - v1[1] * v2[0])

        target_image = np.copy(target["image"])
        target_points = np.copy(np.array(target["keypoints"], dtype=np.float32))

        for idx in self.draw_order(target_points):
            triangle = TRIANGLE_FACE_MODEL[idx]
            target_triangle = self.triangle_points(triangle, target_points)

            source_triangle = self.triangle_points(triangle, self.source_points)
            source_triangle_flipped = self.triangle_points(triangle, self.source_points_flipped)
            source_image = self.source_image

            if self.reflect_transform and S(source_triangle) * self.refl_coef < S(source_triangle_flipped):
                source_triangle = source_triangle_flipped
                source_image = self.source_image_flipped

            source_cut = [r(np.min(source_triangle[:, 0])),\
                          r(np.min(source_triangle[:, 1])),\
                          r(np.max(source_triangle[:, 0])),\
                          r(np.max(source_triangle[:, 1]))]
            if source_cut[2] - source_cut[0] <= 1 or source_cut[3] - source_cut[1] <= 1:
                continue
            source_cut = [min(max(source_cut[0]-self.cut_pad, 0), source_image.shape[1]),
                          min(max(source_cut[1]-self.cut_pad, 0), source_image.shape[0]),
                          min(max(source_cut[2]+self.cut_pad, 0), source_image.shape[1]),
                          min(max(source_cut[3]+self.cut_pad, 0), source_image.shape[0])]
            source_triangle[:, 0] -= source_cut[0]
            source_triangle[:, 1] -= source_cut[1]


            target_cut = [r(np.min(target_triangle[:, 0])),\
                          r(np.min(target_triangle[:, 1])),\
                          r(np.max(target_triangle[:, 0])),\
                          r(np.max(target_triangle[:, 1]))]
            if target_cut[2] - target_cut[0] <= 1 or target_cut[3] - target_cut[1] <= 1:
                continue
            target_cut = [min(max(target_cut[0]-self.cut_pad, 0), target_image.shape[1]),
                          min(max(target_cut[1]-self.cut_pad, 0), target_image.shape[0]),
                          min(max(target_cut[2]+self.cut_pad, 0), target_image.shape[1]),
                          min(max(target_cut[3]+self.cut_pad, 0), target_image.shape[0])]
            target_triangle[:, 0] -= target_cut[0]
            target_triangle[:, 1] -= target_cut[1]

            M = cv2.getAffineTransform(source_triangle,\
                                       target_triangle)
            s = source_image[source_cut[1]:source_cut[3],\
                             source_cut[0]:source_cut[2]]


            source_mask = np.zeros((s.shape[0], s.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(source_mask, source_triangle.astype(np.int32), 1, lineType=cv2.LINE_AA)
            source_mask = source_mask.reshape((source_mask.shape[0], source_mask.shape[1], 1))

            if source_mask.sum() == 0:
                continue

            if self.norm_contrast:
                source_mean = self.masked_mean(s, source_mask).reshape((1, 1, 3))
                source_std = self.masked_std(s, source_mask, source_mean).reshape((1, 1, 3))

            warped_triangle = cv2.warpAffine(s,\
                                             M,\
                                             (target_cut[2] - target_cut[0],\
                                              target_cut[3] - target_cut[1]))
            mask = np.zeros((warped_triangle.shape[0], warped_triangle.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, target_triangle.astype(np.int32), 1, lineType=cv2.LINE_AA)
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

            if mask.sum() == 0:
                continue

            if self.norm_contrast:
                target_mean = self.masked_mean(target_image[target_cut[1]:target_cut[3],\
                                                            target_cut[0]:target_cut[2]], mask).reshape((1, 1, 3))
                target_std = self.masked_std(target_image[target_cut[1]:target_cut[3],\
                                                          target_cut[0]:target_cut[2]], mask, target_mean).reshape((1, 1, 3))

                if self.mean_only or source_std.any() < 1e-6:
                    warped_triangle = warped_triangle - source_mean + target_mean
                else:
                    warped_triangle = (warped_triangle - source_mean) / source_std * target_std + target_mean

                warped_triangle = np.array(np.maximum(np.minimum(warped_triangle, 255), 0), dtype=np.uint8)

            if self.draw_lines:
                cv2.polylines(warped_triangle, [target_triangle.astype(np.int32),], False, (0, 255, 0), 1, lineType=cv2.LINE_4)

            target_image[target_cut[1]:target_cut[3],\
                         target_cut[0]:target_cut[2]] =\
            target_image[target_cut[1]:target_cut[3],\
                         target_cut[0]:target_cut[2]] * (1 - mask) + warped_triangle * mask

        return target_image

FaceSwapper = FaceSwapperTriangular
