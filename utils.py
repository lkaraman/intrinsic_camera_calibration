from functools import wraps
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numba import jit
from sklearn.neighbors import KDTree


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r: %f took: %2.4f sec' % \
              (f.__name__, result, te - ts))
        return result

    return wrap


def convert_pointcloud_from_cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    num_points = len(points)
    r = np.empty(num_points)
    theta = np.empty(num_points)
    phi = np.empty(num_points)

    # Calculate and store the spherical coordinates
    np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2, out=r)
    np.arccos(points[:, 2] / r, out=theta)
    np.arctan2(points[:, 1], points[:, 0], out=phi)

    return np.vstack((r, phi, theta)).T


def calculate_convolution(distances: np.ndarray, gaussian: np.ndarray):
    m = len(gaussian)
    s = int(np.floor(m / 2))

    res = []

    # place 0 for distances for first and last s elements
    distances_dec = np.hstack((np.zeros(s), distances, np.zeros(s)))

    for i in range(len(distances)):

        m, n = 0, 0
        for jj in range(i, 2 * s + i):
            m += distances_dec[jj] * gaussian[jj - i]
            n += distances_dec[jj] ** 2

        n = np.sqrt(n)

        r = np.abs(m / n)
        res.append(r)

    return np.asarray(res)


def calculate_convolution_fft(distances: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    distances_sq = distances ** 2
    up = scipy.signal.correlate(distances, gaussian, mode='full', method='fft')
    down = scipy.signal.correlate(distances_sq, np.ones_like(gaussian), mode='full', method='fft')
    down = np.sqrt(down)

    s = int(len(gaussian) / 2)

    return np.abs(up / down)[s: -s]


@jit(nopython=True)
def non_max_suppression(input: np.ndarray, window_size: int) -> np.ndarray:
    k = int(np.floor(window_size / 2))
    input_dec = np.hstack((np.zeros(k), input, np.zeros(k)))

    result = []

    for i in range(len(input)):
        left = input_dec[i:i + k]
        right = input_dec[i + k + 1:i + 2 * k]
        temp = input[i] if input[i] > np.max(left) and input[i] > np.max(right) else 0

        result.append(temp)

    return np.asarray(result)


@jit(nopython=True)
def threshold_values(input: np.ndarray, T=1) -> np.ndarray:
    return input[np.where(input > T)], np.where(input > T)


def azimuth_change(input: np.ndarray, Taz=0.01) -> np.ndarray:
    d = np.diff(input[:, 1])

    indexes = np.where(np.abs(d) > Taz)

    return indexes


def gaussian_derivative_eval(sigma_c: float = 0.1) -> np.ndarray:
    def fnc(x):
        return -x / (sigma_c ** 3 * np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * sigma_c ** 2))

    points_to_eval = np.arange(-2, 2.05, 0.1)
    return fnc(points_to_eval)


@timing
def Lf_full(image_edges, pointcloud_corners):
    n = len(pointcloud_corners)
    m = len(image_edges)

    tau = 2
    sigma = 1

    outer_sum = 0
    for j in range(n):
        inner_sum = 0
        for i in range(m):
            inner_sum += np.exp(-1 / (2 * sigma) * np.linalg.norm(image_edges[i] - pointcloud_corners[j]) ** 2)

        outer_sum += np.log(tau + inner_sum / m)

    res = -1 / n * outer_sum
    return res


# @timing
def Lf(image_edges, pointcloud_corners):
    n = len(pointcloud_corners)
    m = len(image_edges)

    tree = KDTree(image_edges, leaf_size=2)
    distances, _ = tree.query(pointcloud_corners, k=20)

    if n == 0:
        return 10

    tau = 0.0000001
    sigma = 3

    return calculate_sum(n=n, m=m, sigma=sigma, tau=tau, distances=distances)


@jit(nopython=True)
def calculate_sum(n, m, sigma, tau, distances):
    outer_sum = 0
    for j in range(n):
        inner_sum = 0
        for i in range(len(distances[0])):
            inner_sum += np.exp(-1 / (2 * sigma ** 2) * distances[j][i] ** 2)

        outer_sum += np.log(tau + inner_sum / m)

    res = -1 / n * outer_sum
    # print(f'OUTER SUM: {outer_sum}')
    return res


if __name__ == '__main__':
    # for i in range(1000):
    #     a = np.asarray([1, 2, 3, 4, 1, 8, -5, 4, 3, 1, 0]) * np.random.rand()
    #     ns = non_max_suppression_new(a, window_size=4)

    a = [1, 2, 3, 4, 5]
    c = [1, 4, 9, 16, 25]
    b = [1, 1]
    w = scipy.signal.correlate(c, b, mode='full')
    print(w)

    points = gaussian_derivative_eval(sigma_c=0.1)

    plt.plot(np.arange(-2, 2.05, 0.1), points)
    plt.show()
