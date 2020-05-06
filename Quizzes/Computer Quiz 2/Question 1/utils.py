import numpy as np


def dataGenerator(seed, n):
    '''
        Randomly generating two data distrbution
    '''

    n1 = n
    mu1 = np.array([2, 3])
    sigma1 = 2.5 * np.matrix([[1, -1], [-1, 3]])

    np.random.seed(seed)
    x1 = np.random.multivariate_normal(mu1, sigma1, n1).T
    y1 = np.ones([1, n1], dtype=np.int64)

    n2 = n
    mu2 = np.array([6, 6])
    sigma2 = 2.5 * np.matrix([[1, -1], [-1, 3]])
    np.random.seed(seed + 100)
    x2 = np.random.multivariate_normal(mu2, sigma2, n2).T
    y2 = -1 * np.ones([1, n2], dtype=np.int64)

    x = np.concatenate((x1, x2), axis=1)
    y = np.concatenate((y1, y2), axis=1)

    return x, y
