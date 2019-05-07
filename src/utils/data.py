import numpy as np
from scipy import linalg
from torch.utils.data import Dataset


def generate_uos_data(D, d, K, N_k, normalized=True, noisy=False, varn=0.01):
    U_full = np.empty((K, D, d))
    X = np.empty((D, 0))
    for kk in range(K):
        U_full[kk, :, :] = linalg.orth(np.random.randn(D, d))
        X_k = U_full[kk, :, :] @ np.random.randn(d, N_k)
        X = np.append(X, X_k, axis=1)
    if normalized:
        X = X / linalg.norm(X, axis=0)
    if noisy:
        noise = np.sqrt(varn) * np.random.randn(X.shape[0], X.shape[1])
        X = X + noise
    y = np.empty((K * N_k, 1))
    for i in range(K):
        y[i * N_k:i * N_k + N_k, :] = i * np.ones((N_k, 1))
    y = y.reshape((N_k * K, 1))
    return X, y


def generate_cc(n=1200, noise_sigma=0.1, train_set_fraction=1.):
    '''
    Generates and returns the nested 'C' example dataset (as seen in the leftmost
    graph in Fig. 1)
    '''
    pts_per_cluster = int(n / 2)
    r = 1

    # generate clusters
    theta1 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)
    theta2 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)

    cluster1 = np.concatenate((np.cos(theta1) * r, np.sin(theta1) * r), axis=1)
    cluster2 = np.concatenate((np.cos(theta2) * r, np.sin(theta2) * r), axis=1)

    # shift and reverse cluster 2
    cluster2[:, 0] = -cluster2[:, 0] + 0.5
    cluster2[:, 1] = -cluster2[:, 1] - 1

    # combine clusters
    x = np.concatenate((cluster1, cluster2), axis=0)

    # add noise to x
    x = x + np.random.randn(x.shape[0], 2) * noise_sigma

    # generate labels
    y = np.concatenate((np.zeros(shape=(pts_per_cluster, 1)), np.ones(shape=(pts_per_cluster, 1))), axis=0)

    # shuffle
    p = np.random.permutation(n)
    y = y[p]
    x = x[p]

    # make train and test splits
    n_train = int(n * train_set_fraction)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train].flatten(), y[n_train:].flatten()

    return x_train, x_test, y_train, y_test


class CustomDataSet(Dataset):

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.x[idx, :]
        label = self.y[idx]
        sample = {'sample': sample, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
