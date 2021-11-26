import numpy as np
from oct2py import octave
from utils import *

def ransac(pt1, pt2, K=None, threshold=1.0, desiredConfidence=0.9999):
    # make points homogeneous
    pt1 = homogeneous(pt1)
    pt2 = homogeneous(pt2)
    # normalize points
    if K.any():
        pt1_n = normalizeK(pt1, K)
        pt2_n = normalizeK(pt2, K)

    # best values
    best_inlier_count = 0
    best_inliers = []
    best_err = np.inf
    best_E = []

    nSamples = 5

    # initial iteration count
    max_iter, trials = 10000, 1
    while trials < max_iter:
        # choose 5 random points
        random_index = np.random.choice(len(pt1), nSamples)
        # normalize feature coordinates
        q1 = pt1_n[random_index].T
        q2 = pt2_n[random_index].T
        
        # calculate essential matrix for randomly chosen 5-pt
        Evec = octave.calibrated_fivepoint(q1, q2)
        E = np.transpose(Evec.reshape(3, 3, -1), (2,0,1)) if len(Evec) else np.array([])
        for e in E:
            # calculate fundamental matrix for E
            F = EtoF(e, K)
            # calculate sampson distances for all match points
            d = np.array([distSampson(pt1[j], F, pt2[j]) for j in range(len(pt1))]).squeeze()

            # find inlier indxes
            inlier_indexes = np.nonzero(d < threshold)[0]
            inlier_count = len(inlier_indexes)
            total_count = len(pt1)
            Err = (sum(d[inlier_indexes]) + (total_count - inlier_count)*threshold) / total_count

            # update the best values
            if ( inlier_count > best_inlier_count or (inlier_count == best_inlier_count and Err < best_err ) ):
                best_inlier_count = inlier_count
                best_err = Err
                best_inliers = inlier_indexes
                best_E = e

                # update max iteration count
                inlier_ratio = inlier_count / total_count
                max_iter = nTrials(inlier_ratio, nSamples, desiredConfidence)
        trials = trials+1

    # final inlier points
    in1 = pt1[best_inliers]
    in2 = pt2[best_inliers]
    # final outlier points
    out1 = pt1[np.setdiff1d(range(len(pt1)), best_inliers)]
    out2 = pt2[np.setdiff1d(range(len(pt2)), best_inliers)]
    # final Essential Matrix
    E = best_E

    return E, in1, in2, out1, out2