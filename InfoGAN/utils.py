import numpy as np


def sample_uniform(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def to_categorical(y, num_classes=None):

    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
