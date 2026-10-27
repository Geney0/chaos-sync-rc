"""High-level observer workflow: train -> test -> closed-loop control.

Reproduces the scheme of Fig. 1:
  (a) training: drive signal x_D is fed to the reservoir, target is y_D;
  (b) control : the response is coupled to x_D and to the reservoir estimate
      yhat_D (plus optionally the true states for the "ideal" baselines).

Synchronization error follows Eq. (11), averaged over t in [500, 1000].
"""

import numpy as np

from .reservoir import ReservoirComputer
from .integration import integrate_coupled


# --------------------------------------------------------------------------- #
# Reservoir setup
# --------------------------------------------------------------------------- #

def train_reservoir(xD_tr, yD_tr, **rc_kwargs):
    """Train a reservoir to estimate y_D from x_D. Returns (rc, Delta_tr)."""
    rc = ReservoirComputer(**rc_kwargs)
    _, Delta_tr = rc.train(xD_tr, yD_tr)
    return rc, Delta_tr


def test_reservoir(rc, xD_ts, yD_ts):
    """Compute testing error and record reservoir states for later seeding.

    Returns (Delta_ts, states, yhat_ts).
    """
    states = rc.evolve(xD_ts, steps=len(xD_ts))
    yhat_ts = states @ rc.w_out[:-1] + rc.w_out[-1]
    Delta_ts = np.std(yhat_ts - yD_ts) / np.std(yD_ts)
    return Delta_ts, states, yhat_ts


# --------------------------------------------------------------------------- #
# Coupling helpers (Chen, Eq. 12 / Fig. 2, 3 and Roessler, Fig. 8, 9)
# --------------------------------------------------------------------------- #

def chen_coupling(vR, xD, yD):
    """Eq. (12): coupling x_D->y_R and yhat_D->y_R simultaneously."""
    return np.array([0.0, (yD - vR[1]) + (xD - vR[0]), 0.0])


def chen_coupling_xonly(vR, xD, yD=None):
    """x_D -> y_R only (Fig. 2a)."""
    return np.array([0.0, xD - vR[1], 0.0])


def chen_coupling_yonly(vR, xD, yD):
    """yhat_D -> y_R only (Fig. 2b)."""
    return np.array([0.0, yD - vR[1], 0.0])


def chen_coupling_xy(vR, xD, yD):
    """x_D -> x_R and yhat_D -> y_R (Fig. 6 hyperparameter scans)."""
    return np.array([xD - vR[0], yD - vR[1], 0.0])


def roessler_coupling_xy(vR, xD, yD):
    """x_D -> x_R and yhat_D -> y_R (Fig. 8c, 9b)."""
    return np.array([xD - vR[0], yD - vR[1], 0.0])


def roessler_coupling_x(vR, xD, yD=None):
    """x_D -> x_R only (Fig. 8a, 9a)."""
    return np.array([xD - vR[0], 0.0, 0.0])


def roessler_coupling_y(vR, xD, yD):
    """yhat_D -> y_R only (Fig. 8b)."""
    return np.array([0.0, yD - vR[1], 0.0])


# --------------------------------------------------------------------------- #
# Closed-loop control
# --------------------------------------------------------------------------- #

def run_control(rc, f, coupling_fn, vD_ts, states_ts, vR0, kappa, dt, Tf,
                use_reservoir=True, t_s=None, vD_ref=None, noise_eps=0.0,
                noise_T=6.0, noise_seed=None):
    """Closed-loop synchronization run (Fig. 1b).

    The drive is seeded from a random point of the testing trajectory, the
    reservoir from the corresponding reservoir state.  If use_reservoir is
    False, yD is taken to be the *true* drive value (ideal coupling).
    If vD_ref is given (pinned UPO signal), the drive is not integrated and
    the control starts at vD_ref[:, 0].

    Returns (vD_series, vR_series).
    """
    if vD_ref is not None:
        vD0 = vD_ref[:, 0]
    else:
        if t_s is None:
            t_s = int(len(vD_ts) * 0.5) + np.random.randint(0, len(vD_ts) // 2)
        vD0 = vD_ts[:, t_s]
    if use_reservoir and rc is not None:
        rc.reset(states_ts[t_s] if states_ts is not None else None)
        the_rc = rc
    else:
        the_rc = None
    return integrate_coupled(f, coupling_fn, vD0, vR0, dt, Tf,
                             kappa=kappa, rc=the_rc, vD_ref=vD_ref,
                             noise_eps=noise_eps, noise_T=noise_T,
                             noise_seed=noise_seed)


# --------------------------------------------------------------------------- #
# Synchronization error, Eq. (11)
# --------------------------------------------------------------------------- #

def sync_error(vD_s, vR_s, dt, t_start=500.0, t_end=1000.0):
    """Instantaneous E(t) and averaged Ebar over [t_start, t_end] (seconds)."""
    a = int(round(t_start / dt))
    b = int(round(t_end / dt))
    vD = vD_s[:, a:b + 1]
    vR = vR_s[:, a:b + 1]
    stds = vD.std(axis=1)
    E = np.zeros(b - a + 1)
    for i in range(3):
        E = E + ((vD[i] - vR[i]) / stds[i]) ** 2
    E = np.sqrt(E)
    return E, E.mean()


def sync_error_full(vD_s, vR_s, dt):
    """E(t) over the whole run (used for Fig. 4-style plots)."""
    stds = vD_s.std(axis=1)
    E = np.zeros(vD_s.shape[1])
    for i in range(3):
        E = E + ((vD_s[i] - vR_s[i]) / stds[i]) ** 2
    return np.sqrt(E)
