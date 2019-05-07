"""
Spectral Clustering Demo - mySpectralClustering

ECE 510

python version: Python 3.7.2

Spring 2019
"""

import numpy as np
import scipy as sp
import math
from numpy import linalg as lin
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linear_sum_assignment


def mySpectralClustering(W, K, normalized):
    r"""
    Customized version of Spectral Clustering

    Inputs:
    -------
        W: weighted adjacency matrix of size N x N
        K: number of output clusters
        normalized: 1 for normalized Laplacian, 0 for unnormalized
    
    Outputs:
    -------
        estLabels: estimated cluster labels
        Y: transformed data matrix of size K x N
    """

    degMat = np.diag(np.sum(W, axis=0))
    L = degMat - W

    if normalized == 0:
        D, V = lin.eig(L)
        V_real = V.real
        inds = np.argsort(D)
        Y = V_real[:, inds[0:K]].T

        k_means = KMeans(n_clusters=K, max_iter=100).fit(Y.T)
        estLabels = k_means.labels_
    else:
        # Invert degree matrix
        degInv = np.diag(1.0 / np.diag(degMat))
        Ln = degInv @ L

        # Eigen decomposition
        D, V = lin.eig(Ln)
        V_real = V.real
        inds = np.argsort(D)
        Y = V_real[:, inds[0:K]].T

        k_means = KMeans(n_clusters=K, max_iter=100).fit(Y.T)
        estLabels = k_means.labels_

    return estLabels, Y
