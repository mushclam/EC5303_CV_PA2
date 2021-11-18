#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibrated_fivepoints import calibrated_fivepoints

from image_to_sift import image_to_sift
from simpleMatcher import SimpleMatcher

# Load images
filename = [
    'data/dataset_two-view/sfm01.jpg',
    'data/dataset_two-view/sfm02.jpg'
]
img = [cv2.imread(f) for f in filename]
gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

# create SIFT
kp, desc = image_to_sift(img)

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

F, mask = cv2.findEssentialMat(img0_pt, img1_pt)
r1, r2, t = cv2.decomposeEssentialMat(F)

# ransac_iter = 10000
# for i in range(ransac_iter):
#     # choose 5 random points
#     random_index = np.random.choice(len(img0_pt), 5)
#     # normalize feature coordinates
#     fa_n = img0_pt[random_index]
#     fb_n = img1_pt[random_index]
#     # calc the probable essential matrix
#     e = calibrated_fivepoints(fa_n, fb_n)

print(matches)
# %%
