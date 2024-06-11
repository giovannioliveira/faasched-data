import numpy as np
from sklearn.neighbors import KernelDensity


def pdf_from_samples(xs: np.ndarray, xl: float, xr: float, n_samples: int, kernel='tophat', bandwidth='silverman',
                     random_state=None) -> (np.ndarray, np.ndarray, KernelDensity):
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(xs.reshape(-1, 1))
    X_d = np.array([])
    while len(X_d) < n_samples:
        x_d: np.ndarray = kde.sample(n_samples, random_state=random_state).reshape(n_samples)
        x_d = x_d[xl <= x_d]
        x_d = x_d[xr >= x_d]
        X_d = np.concat([X_d, x_d])
    X_d.sort()
    Y_d = np.exp(kde.score_samples(X_d.reshape(-1, 1))).reshape(len(X_d))
    return X_d, Y_d, kde
