import numpy as np
import cv2

def readCamera(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.rstrip().replace('[', '').replace(']', '').split(';')
    line = [list(map(float, l.split())) for l in line]
    return np.array(line)

def image_to_sift(img):
    # create SIFT
    sift = cv2.SIFT_create()
    # Detect keypoints
    kp = []
    desc = []
    for i in range(len(img)):
        tmp_kp, tmp_desc = sift.detectAndCompute(img[i], None)
        kp.append(tmp_kp)
        desc.append(tmp_desc)

    return kp, desc

def nTrials(inlierRatio, nSamples, desiredConfidence):
    return np.log(1-desiredConfidence) / np.log(1-(inlierRatio)**nSamples)

def distSampson(x1, f, x2):
    x1 = x1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    f_x1 = f @ x1
    x2t_f_x1 = x2.T @ f_x1
    ft_x2 = f.T @ x2

    return x2t_f_x1**2 / (f_x1[0]**2 + f_x1[1]**2 + ft_x2[0]**2 + ft_x2[1]**2)

def decomposeEssentialMat(E):
    u, _, vt = np.linalg.svd(E)
    w = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    r1 = u @ w.T @ vt
    r2 = u @ w @ vt
    r1 = r1 if np.linalg.det(r1) > 0 else -r1
    r2 = r2 if np.linalg.det(r2) > 0 else -r2
    t = u[:, 2]

    return r1, r2, t

def triangulation(P1, P2, in1, in2):
    X = []
    for i1, i2 in zip(in1, in2):
        A = np.array([
            i1[0] * P1[2] - P1[0],
            i1[1] * P1[2] - P1[1],
            i2[0] * P2[2] - P2[0],
            i2[1] * P2[2] - P2[1]
        ])
        vals, vecs = np.linalg.eig(A.T @ A)
        X.append(vecs[:, np.argmin(vals)])
    return np.array(X)

def homogeneous(v):
    return np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)

def normalizeK(pt, K):
    return (np.linalg.inv(K) @ pt.T).T

def skew(mat):
    return np.array([[0, -mat[2], mat[1]], [mat[2], 0, -mat[0]], [-mat[1], mat[0], 0]])

def EtoF(E, K):
    K_inv = np.linalg.inv(K)
    return K_inv.T@E@K_inv

def RT(R, T):
    return np.concatenate([R, T[:, np.newaxis]], axis=1)

def get_center(R, T):
    return -R.T@T