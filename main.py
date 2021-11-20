#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from oct2py import octave

import os
import warnings
warnings.filterwarnings(action='ignore')

from image_to_sift import image_to_sift
from ransac import ransac
from utils import *
# from ransac import ransac
from simpleMatcher import SimpleMatcher

# Load camera information
camera_file = 'data/K.txt'
K = readCamera(camera_file)

# Set octave path
octave.addpath(os.path.abspath('Givenfunctions/'))
# Load images
filename = [
    'data/dataset_two-view/sfm01.jpg',
    'data/dataset_two-view/sfm02.jpg'
]
img = [cv2.imread(f) for f in filename]
gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

# create SIFT
kp, desc = image_to_sift(img)

# Draw keypoints on original images
# img_kp = [cv2.drawKeypoints(gray[i], kp[i], img[i], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for i in range(len(img))]
# img_kp = [cv2.drawKeypoints(gray[i], kp[i], img[i]) for i in range(len(img))]
# plt.imshow(img_kp[0])

# Match the keypoints between images
matcher_type = 'flann'
matcher = SimpleMatcher(matcher_type)
matches = matcher(desc)

# matched_image = matcher.draw(img, kp)
# plt.imshow(matched_image)
# plt.savefig('test.png', dpi=300)

# Estimate Essential Matrix
ratio = 0.75
good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
img0_pt = np.int32([kp[0][m.queryIdx].pt for m in good_matches])
img1_pt = np.int32([kp[1][m.trainIdx].pt for m in good_matches])

# Package code
F, mask = cv2.findEssentialMat(img0_pt, img1_pt)
E, in1, in2, out1, out2 = ransac(img0_pt, img1_pt, K, 0.8)
K_inv = np.linalg.inv(K)
in1 = [K_inv @ i for i in in1]
in2 = [K_inv @ i for i in in2]    

# Essential Matrix Decomposition
# Package code
# r1, r2, t = cv2.decomposeEssentialMat(F)
r1, r2, t = decomposeEssentialMat(E)
rot = r1
trans = t
if not in_front_of_both_cameras(in1, in2, rot, trans):
    trans = -t
    if not in_front_of_both_cameras(in1, in2, rot, trans):
        rot = r2
        trans = t
        if not in_front_of_both_cameras(in1, in2, rot, trans):
            trans = -t

# Triangulation
P = K @ np.concatenate([rot, trans[:, np.newaxis]], axis=1)
X = []
for i1, i2 in zip(in1, in2):
    A = np.array([
        i1[0] * P[2] - P[0],
        i1[1] * P[2] - P[1],
        i2[0] * P[2] - P[0],
        i2[1] * P[2] - P[1]
    ])
    vals, vecs = np.linalg.eig(A.T @ A)
    X.append(vecs[:, np.argmin(vals)])

# ransac(img0_pt, img1_pt)

print(matches)
# %%
