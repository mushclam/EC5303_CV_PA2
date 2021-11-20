import numpy as np
from oct2py import octave
from utils import *

def ransac(pt1, pt2, K, threshold):
    # make points homogeneous
    pt1 = np.concatenate((pt1, np.ones((pt1.shape[0], 1))), axis=1)
    pt2 = np.concatenate((pt2, np.ones((pt2.shape[0], 1))), axis=1)

    # best values
    bestInlierCount = 0
    bestInliers = []
    bestErr = np.inf
    bestE = []

    desiredConfidence = 0.99
    nSamples = 5

    # initial iteration count (enything greater than 1 would work)
    m, t = 3, 1
    while t < m:
        # choose 5 random points
        random_index = np.random.choice(len(pt1), nSamples)
        # normalize feature coordinates
        fa_n = np.linalg.lstsq(K, pt1[random_index].T)[0]
        fb_n = np.linalg.lstsq(K, pt2[random_index].T)[0]
        
        # calc the probable essential matrix
        Evector = octave.calibrated_fivepoint(fa_n, fb_n)
        E = Evector.reshape(3, 3, -1) if len(Evector) else []
        # now E is a 3x3xN matrix containing N essential matrices
        for ec in range(E.shape[2]):
            # calc fundamental matrix
            F = np.linalg.lstsq(K.T, (np.linalg.inv(K).T @ E[:,:,ec]).T)[0].T

            # calc sampson distances    
            d = np.array([distSampson(pt1[j], F,  pt2[j]) for j in range(len(pt1))]).squeeze()

            # find inlier indxes
            inlier_indexes = np.nonzero(d < threshold)[0]
            # compute the sum error of inliers and outliers for this F matrix
            inlierCount = len(inlier_indexes)
            total_count = len(pt1)
            Err = (sum(d[inlier_indexes]) + (total_count - inlierCount)*threshold) / total_count

            # update the best values if needed
            if ( inlierCount > bestInlierCount or (inlierCount == bestInlierCount and Err < bestErr ) ):
                bestInlierCount = inlierCount
                bestErr = Err
                bestInliers = inlier_indexes
                bestE = E[:,:,ec]

                # update max iteration count
                inlierRatio = inlierCount / len(pt1)
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