import numpy as np

def chen_buggy(v):
    x, y, z = v
    return np.array([
        35.0 * (y - x),
        -7.0 * x - x * z + 28.0 * y,
        x * y - (8.0 / 3.0) * z,
    ])