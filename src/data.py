"""Shared data-generation utilities for the reproduction scripts.

Produces a chaotic drive trajectory, discards the transient, splits it into
a training segment and a testing segment, and provides random on-attractor
initial conditions for the response.
"""

import numpy as np

from .systems import rk4_system, discard_transient


def generate_drive_trajectories(f, dt=0.001, t_trans=200.0, t_train=100.0,
                                t_test=20.0, v0=None, seed=None):
    """Return (tr, ts) dicts with x/y/z arrays for training and testing.

    v0      : arbitrary starting state (a transient of t_trans is discarded).
    t_train : length (s) of the training segment.
    t_test  : length (s) of the testing segment.
    """
    rng = np.random.default_rng(seed)
    if v0 is None:
        v0 = rng.uniform(-5, 5, size=3)

    # discard transient to land on the attractor
    v_on_attr = discard_transient(f, v0, t_trans, dt)

    n_train = int(round(t_train / dt))
    n_test = int(round(t_test / dt))

    # training segment
    v_tr = rk4_system(f, v_on_attr, 0.0, t_train, dt)[:, :n_train + 1]
    # testing segment continues from the end of training
    v_ts = rk4_system(f, v_tr[:, -1], 0.0, t_test, dt)[:, :n_test + 1]

    tr = {k: v_tr[i].copy() for i, k in enumerate("xyz")}
    ts = {k: v_ts[i].copy() for i, k in enumerate("xyz")}
    return tr, ts


def random_attractor_point(f, dt=0.001, t_trans=200.0, seed=None):
    """A random point on the chaotic attractor (for response init)."""
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(-5, 5, size=3)
    return discard_transient(f, v0, t_trans, dt)
