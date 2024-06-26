import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt


def pdf_from_samples(xs: np.ndarray, xl: float = None, xr: float = None, n_samples: int = 100, kernel='gaussian',
                     bandwidth='silverman', random_state=None) -> (np.ndarray, np.ndarray, KernelDensity):
    if xl is None:
        xl = min(xs)
    if xr is None:
        xr = max(xs)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(xs.reshape(-1, 1))
    X_d = np.array([])
    while len(X_d) < n_samples:
        x_d: np.ndarray = xs.choose(n_samples)
        x_d = x_d[xl <= x_d]
        x_d = x_d[xr >= x_d]
        X_d = np.append(X_d, x_d)
    X_d.sort()
    Y_d = np.exp(kde.score_samples(X_d.reshape(-1, 1))).reshape(len(X_d))
    return X_d, Y_d, kde


def cdf_from_pdf(xs: np.ndarray, ys: np.ndarray):
    ret = np.cumsum(
        np.append(0, np.vectorize(lambda i: (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2)(range(len(xs) - 1))))
    return ret / ret[-1]


def probs_from_samples(samples: np.ndarray, n_size: int, xl: float = None, xr: float = None, plot: bool = True,
                       plot_xlog: bool = False, random_state: int = None, kernel: str = 'tophat'):
    if xl is None:
        xl = min(samples)
    if xr is None:
        xr = max(samples)
    X_d, Y_d, kde = pdf_from_samples(samples, xl=xl, xr=xr, n_samples=n_size, random_state=random_state, kernel=kernel,
                                     bandwidth='silverman')
    Y_c = cdf_from_pdf(X_d, Y_d)
    if plot:
        plot_area(X_d, Y_d, plot_xlog)
        plot_area(X_d, Y_c, plot_xlog)
    return X_d, Y_d, Y_c


def plot_area(xs, ys, xlog=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if xlog:
        plt.xscale('log', base=10)
    else:
        plt.xscale('linear')
    ax.fill_between(xs, ys)
    plt.show(fig)
