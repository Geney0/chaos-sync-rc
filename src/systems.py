"""Dynamical systems used in the paper (Sec. III A / III B).

The drive and response are identical oscillators.  For the Chen system the
only state transmitted to the response is x_D(t); the reservoir estimates
y_D(t) -> yhat_D(t).  For the Roessler system the same logic applies.

All formulas follow Eq. (10) [Chen] and Eq. (13) [Roessler] of the paper.

NOTE: the Chen second component printed in the paper, ``(-7-z(t)) + 28 y(t)``,
is a typesetting contraction of the standard Chen field
    dy/dt = -7 x - x z + 28 y.
We use the standard form.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Vector fields (autonomous, used to generate the drive trajectory)
# --------------------------------------------------------------------------- #

def chen(v):
    """Chen oscillator, Eq. (10): (a, b, c) = (35, 8/3, 28)."""
    x, y, z = v[0], v[1], v[2]
    return np.array([
        35.0 * (y - x),
        -7.0 * x - x * z + 28.0 * y,
        x * y - (8.0 / 3.0) * z,
    ])


def roessler(v):
    """Roessler oscillator, Eq. (13)."""
    x, y, z = v[0], v[1], v[2]
    return np.array([
        -y - z,
        x + 0.2 * y,
        0.2 + (x - 9.0) * z,
    ])


SYSTEMS = {
    "chen":    chen,
    "roessler": roessler,
}


# --------------------------------------------------------------------------- #
# Coupling helpers.
#
# We write the coupled response as
#     dv_R/dt = F(v_R) + kappa * cpl(v_R, v_D_used)
# where v_D_used is the information about the drive actually available at the
# response (possibly the reservoir estimate).  Different coupling "schemes"
# select different components of the coupling vector.
# --------------------------------------------------------------------------- #

def coupling_vector(v_R, x_D, y_D):
    """Return [0, (y_D - y_R) + (x_D - x_R), 0] for the Chen scheme of Eq. (12).

    This is the combination used in Fig. 2(c) / Fig. 3: x_D->y_R and yhat_D->y_R.
    Passing y_D = None turns off the y-coupling (x_D->y_R only).
    """
    x_R, y_R = v_R[0], v_R[1]
    term_y = (x_D - x_R) + (y_D - y_R) if y_D is not None else (x_D - x_R)
    return np.array([0.0, term_y, 0.0])


def coupling_rossler_x(v_R, x_D):
    """x_D -> x_R coupling only (Fig. 8a, Fig. 9a)."""
    return np.array([x_D - v_R[0], 0.0, 0.0])


def coupling_rossler_y(v_R, y_D):
    """yhat_D -> y_R coupling only (Fig. 8b)."""
    return np.array([0.0, y_D - v_R[1], 0.0])


def coupling_rossler_xy(v_R, x_D, y_D):
    """x_D -> x_R AND yhat_D -> y_R (Fig. 8c, Fig. 9b)."""
    return np.array([x_D - v_R[0], y_D - v_R[1], 0.0])


# --------------------------------------------------------------------------- #
# Standard RK4 integrator for a single (drive) oscillator.
# --------------------------------------------------------------------------- #

def rk4_system(f, v0, t_start, t_end, dt):
    """Integrate dv/dt = f(v) on [t_start, t_end] with fixed step dt.

    Returns arrays v[:, j] at j=0..floor((t_end-t_start)/dt).
    """
    n = int(round((t_end - t_start) / dt))
    vs = np.empty((3, n + 1))
    vs[:, 0] = v0
    v = v0.copy()
    for j in range(n):
        k1 = f(v)
        k2 = f(v + 0.5 * dt * k1)
        k3 = f(v + 0.5 * dt * k2)
        k4 = f(v + dt * k3)
        v = v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        vs[:, j + 1] = v
    return vs


def discard_transient(f, v0, t_trans, dt):
    """Integrate f for t_trans seconds and return the final state (on attractor)."""
    vs = rk4_system(f, v0, 0.0, t_trans, dt)
    return vs[:, -1]
