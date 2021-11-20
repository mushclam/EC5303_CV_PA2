import numpy as np

def readCamera(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.rstrip().replace('[', '').replace(']', '').split(';')
    line = [list(map(float, l.split())) for l in line]
    return np.array(line)

def nTrials(inlierRatio, nSamples, desiredConfidence):
    return np.log(1-desiredConfidence) / np.log(1-(inlierRatio)**nSamples)

def distSampson(x1, f, x2):
    x1 = x1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    f_x1 = f @ x1
    x2t_f_x1 = x2.T @ f_x1
    ft_x2 = f.T @ x2

    return x2t_f_x1**2 / (f_x1[0, :]**2 + f_x1[1, :]**2 + ft_x2[0, :]**2 + ft_x2[1, :]**2)

def decomposeEssentialMat(E):
    u, s, vt = np.linalg.svd(E)
    w = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    # z = np.array([
    #     [0, 1, 0],
    #     [-1, 0 ,0],
    #     [0, 0, 0]
    # ])
    # S = u @ z @ u.T
    r1 = u @ w @ vt
    r2 = u @ w.T @ vt
    t = u[:, 2]

    return r1, r2, t

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, first[1] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True