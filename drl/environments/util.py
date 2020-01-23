import numpy as np


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]
