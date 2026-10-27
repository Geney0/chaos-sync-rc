"""Coupled drive-response integration (Algorithm 1 of the paper).

The drive and response evolve simultaneously with fixed-step 4th-order
Runge-Kutta.  The reservoir (if used) is advanced once per macro step with the
current drive signal x_D(t_j); its readout yhat = r^T w_out is held constant
across the four RK substeps of the response (zero-order hold).  The drive
system is unidirectional and therefore independent of the response.

See Algorithm 1 and Sec. Appendix of the paper.  With K = 1 (h = dt = 0.001)
the reservoir is updated every RK step.
"""

import numpy as np


def integrate_coupled(f, coupling_fn, vD0, vR0, dt, t_end, kappa=1.0,
                      rc=None, vD_ref=None, noise_eps=0.0, noise_T=6.0,
                      noise_seed=None):
    """Integrate the unidirectionally coupled pair for t_end seconds.

    Parameters
    ----------
    f          : callable v -> dv, the (shared) vector field.
    coupling_fn: callable (vR, xD, yD) -> coupling vector (3,).
                 yD is whatever y-information is available: the true value
                 (ideal coupling) or the reservoir estimate.
    vD0, vR0   : initial states of drive / response.
    kappa      : coupling strength, multiplies the coupling vector (Eq. 1).
    rc         : ReservoirComputer or None.  If given, its state is advanced
                 one step per macro step with the (possibly noisy) drive x
                 and yD is taken from its readout; otherwise yD is the true
                 (possibly noisy) drive y.
    vD_ref     : optional precomputed drive trajectory (3, n+1).  When given
                 (e.g. a pinned UPO signal, Fig. 9) the drive is *not*
                 integrated; its state is read from this reference at each
                 step.
    noise_eps  : measurement-noise magnitude epsilon in Eq. (14).
    noise_T    : approximate oscillation period T used in the noise scaling.
    noise_seed : RNG seed for the measurement noise (Eq. 14).
    dt, t_end  : integration step and horizon.

    Returns (vD_series, vR_series), both (3, n+1).
    """
    rng = np.random.default_rng(noise_seed)
    n = int(round(t_end / dt))
    vD = np.asarray(vD0, dtype=float).copy()
    vR = np.asarray(vR0, dtype=float).copy()
    vD_series = np.empty((3, n + 1))
    vR_series = np.empty((3, n + 1))
    vD_series[:, 0] = vD if vD_ref is None else vD_ref[:, 0]
    vR_series[:, 0] = vR

    scale = noise_eps * np.sqrt(dt / noise_T) if noise_eps > 0 else 0.0

    for j in range(n):
        if vD_ref is not None:
            vD = vD_ref[:, j]
        xD_true = vD[0]
        yD_true = vD[1]

        # --- noisy measurements (Eq. 14) ----------------------------------
        if scale > 0:
            xD = xD_true + scale * rng.normal()
            yD_meas = yD_true + scale * rng.normal()
        else:
            xD = xD_true
            yD_meas = yD_true

        # --- reservoir readout + one-step update (Algorithm 1, lines 4, 11)
        if rc is not None:
            yD = rc.estimate_next(xD)      # reservoir fed with (noisy) x_D
        else:
            yD = yD_meas                    # true (noisy) value

        if vD_ref is None:
            # --- drive (unidirectional, autonomous RK4)
            k1d = f(vD)
            k2d = f(vD + 0.5 * dt * k1d)
            k3d = f(vD + 0.5 * dt * k2d)
            k4d = f(vD + dt * k3d)
            vD = vD + (dt / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)

        # --- response (coupling uses xD and yD fixed over the substeps)
        def rhs(v):
            return f(v) + kappa * coupling_fn(v, xD, yD)
        k1 = rhs(vR)
        k2 = rhs(vR + 0.5 * dt * k1)
        k3 = rhs(vR + 0.5 * dt * k2)
        k4 = rhs(vR + dt * k3)
        vR = vR + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        vD_series[:, j + 1] = vD if vD_ref is None else vD_ref[:, j + 1]
        vR_series[:, j + 1] = vR

    return vD_series, vR_series
