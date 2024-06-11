import numpy as np
from sklearn.neighbors import KernelDensity


def pdf_from_samples(xs: np.ndarray, xl: float = None, xr: float = None, n_samples: int = 100, kernel='gaussian',
                     bandwidth='silverman', random_state=None) -> (np.ndarray, np.ndarray, KernelDensity):
    if xl is None:
        xl = min(xs)
    if xr is None:
        xr = max(xs)
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


def cdf_from_pdf(xs: np.ndarray, ys: np.ndarray):
    ret = np.cumsum(np.append(0,np.vectorize(lambda i: (xs[i+1] - xs[i])*(ys[i+1] + ys[i])/2)(range(len(xs)-1))))
    return ret/ret[-1]
