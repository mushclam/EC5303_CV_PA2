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
    bestInlierCount = 0
    bestInliers = []
    bestErr = np.inf
    bestE = []

    nSamples = 5

    # initial iteration count
    m, t = 10000, 1
    while t < m:
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
            inlierCount = len(inlier_indexes)
            total_count = len(pt1)
            Err = (sum(d[inlier_indexes]) + (total_count - inlierCount)*threshold) / total_count

            # update the best values
            if ( inlierCount > bestInlierCount or (inlierCount == bestInlierCount and Err < bestErr ) ):
                bestInlierCount = inlierCount
                bestErr = Err
                bestInliers = inlier_indexes
                bestE = e

                # update max iteration count
                inlierRatio = inlierCount / total_count
                m = nTrials(inlierRatio, nSamples, desiredConfidence)
        t = t+1

    # final inlier points
    in1 = pt1[bestInliers]
    in2 = pt2[bestInliers]
    # final outlier points
    out1 = pt1[np.setdiff1d(range(len(pt1)), bestInliers)]
    out2 = pt2[np.setdiff1d(range(len(pt2)), bestInliers)]
    # final Essential Matrix
    E = bestE

    return E, in1, in2, out1, out2