import numpy as np
def calibrated_fivepoints(q1, q2):
    q1 = np.transpose(q1)
    q2 = np.transpose(q2)

    q = [
        q1[:, 0]*q2[:, 0],
        q1[:, 1]*q2[:, 0],
        q1[:, 2]*q2[:, 0],
        q1[:, 0]*q2[:, 1],
        q1[:, 1]*q2[:, 1],
        q1[:, 2]*q2[:, 1],
        q1[:, 0]*q2[:, 2],
        q1[:, 1]*q2[:, 2],
        q1[:, 2]*q2[:, 2],
    ]

    u, s, v = np.linalg.svd(q)
    ee = v[:, 6:10]

    a = calibrated_fivepoints_helper(ee)
    a = a[:,0:11]/a[:,11:21]
    m = -a[[1, 2, 3, 5, 6, 8], :]

    m[6, 0] = 1
    m[7, 1] = 1
    m[8, 3] = 1
    m[9, 6] = 1

    v, d = np.linalg.eig(m)
    sols = v[6:10, :] / (np.ones(2, 0) * v[9, :])

    evec = ee * [[sols], [np.ones(0, 11)]]
    evec = evec / (np.ones(8, 0) * np.sqrt(np.sum(evec**2)))

    i = find(not(imag(evec[0, :])))
    evec = evec[:, i]