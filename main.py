import cv2
import numpy as np

from oct2py import octave

import os
import warnings
warnings.filterwarnings(action='ignore')
import argparse

from ransac import ransac
from utils import *
# from ransac import ransac
from simpleMatcher import SimpleMatcher


def initiateArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsic', '-K', type=str, default='data/K.txt',
                        help='File path of intrinsic paramter')
    parser.add_argument('--input_file1', type=str, default='data/dataset_two-view/sfm01.jpg',
                        help='Path to reference image file')
    parser.add_argument('--input_file2', type=str, default='data/dataset_two-view/sfm02.jpg',
                        help='Path to input image file')
    parser.add_argument('--matcher', type=str, default='flann',
                        choices=['normal', 'knn', 'flann'])
    parser.add_argument('--match_ratio', type=float, default=0.75,
                        help='Ratio to select matching points')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Threshold for RANSAC')
    parser.add_argument('--confidence', type=float, default=0.9999,
                        help='Confidence for RANSAC')
    parser.add_argument('--output_file', type=str, default='result/structure_3D.ply',
                        help='Path to output 3D point file')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = initiateArgument()

    # Load camera information
    camera_file = args.intrinsic
    K = readCamera(camera_file)
    K_inv = np.linalg.inv(K)

    # Set octave path
    octave.addpath(os.path.abspath('Givenfunctions/'))
    # Load images
    filename = [
        args.input_file1,
        args.input_file2
    ]
    img = [cv2.imread(f) for f in filename]
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

    # create SIFT
    kp, desc = image_to_sift(img)

    # Visualization code
    # Draw keypoints on original images
    # img_kp = [cv2.drawKeypoints(gray[i], kp[i], img[i], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for i in range(len(img))]
    # img_kp = [cv2.drawKeypoints(gray[i], kp[i], img[i]) for i in range(len(img))]
    # plt.figure(figsize=(12,4))
    # plt.imshow(img_kp[0])
    # plt.show()

    # Match the keypoints between images
    matcher_type = args.matcher
    matcher = SimpleMatcher(matcher_type)
    matches = matcher(desc)

    # Visualization code
    # matched_image = matcher.draw(img, kp)
    # plt.figure(figsize=(12,4))
    # plt.imshow(matched_image)
    # plt.show()

    # Estimate Essential Matrix
    ratio = args.match_ratio
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    img0_pt = np.int32([kp[0][m.queryIdx].pt for m in good_matches])
    img1_pt = np.int32([kp[1][m.trainIdx].pt for m in good_matches])

    E, in1, in2, out1, out2 = ransac(img0_pt, img1_pt, K,
                                     args.threshold, args.confidence)  

    # Essential Matrix Decomposition
    r1, r2, t = decomposeEssentialMat(E)
    P1 = K @ RT(np.identity(3), np.zeros((3,)))

    # normalize inliers
    in1, in2 = [normalizeK(pt, K) for pt in [in1, in2]]
    # Get candidate Extrinsic
    e_set = [(r1, t), (r2, t), (r1, -t), (r2, -t)]
    Ps = [K @ RT(rot, trans) for rot, trans in e_set]
    # Get camera center
    cs = [get_center(rot, trans) for rot, trans in e_set]
    # Get 3D point by triangulation
    rs = [triangulation(P1, P2, in1, in2) for P2 in Ps]
    rs = [r[:,:-1]/r[:,3,np.newaxis] for r in rs]
    # Get best index for camera matrix
    best_ind = np.argmax(np.sum([(r-c)@e[0][2].T >= 0 for r, c, e in zip(rs, cs, e_set)], axis=1))
    best_P = Ps[best_ind]
    best_c = cs[best_ind]
    best_r = rs[best_ind]
    # best_r[np.where(best_r[:,2]<0)] = -best_r[np.where(best_r[:,2]<0)]

    # 3D point reprojection
    # r_pt = (best_P @ homogeneous(best_r).T).T
    # r_pt = r_pt[:,:-1]/r_pt[:,2,np.newaxis]
    # r_pt = r_pt - np.min(r_pt, axis=0)
    # r_pt = r_pt / np.max(r_pt, axis=0)
    # r_pt = np.array([img[0].shape[0], img[0].shape[1]]) * r_pt

    # Add color to 3D points
    img = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in img]
    color = img[0][in1[:,1].astype(np.int64),in1[:,0].astype(np.int64)]
    best_r = np.concatenate([best_r, color], axis=1)

    # Save PLY file
    # octave.SavePLY(args.output_file, best_r.T)