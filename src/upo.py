"""Locate an unstable periodic orbit (UPO) of a chaotic flow.

The paper (Sec. III C) finds the period-1 orbit of the Roessler system with
the MatCont toolbox.  Here we reproduce it numerically with a Poincare-section
Newton (shooting) method:

  * integrate the flow, collect upward crossings of the section x = 0;
  * define the return map P over the 2-D section (y, z);
  * solve P(s*) = s* with Newton (scipy.optimize.fsolve); the fixed point
    corresponds to a period-1 orbit of period T = flight time.

The orbit is unstable, so we return its state and period; a drive trajectory
can be generated starting on it.
"""

import numpy as np
from scipy.optimize import fsolve


def _rk4_step(f, v, dt):
    k1 = f(v)
    k2 = f(v + 0.5 * dt * k1)
    k3 = f(v + 0.5 * dt * k2)
    k4 = f(v + dt * k3)
    return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def advance_to_upcross(f, v0, dt=0.001, max_steps=600000):
    """Integrate from v0 until x crosses 0 upward (dx/dt > 0).

    v0 is taken to be on the section (x ~ 0).  Returns (v_at_cross, T) with
    bisection refinement, or None if no crossing within max_steps.
    """
    t = 0.0
    v = np.asarray(v0, float).copy()
    prev_x = v[0]
    for _ in range(max_steps):
        v_next = _rk4_step(f, v, dt)
        x_new = v_next[0]
        # upward crossing: prev_x < 0 and x_new >= 0
        if prev_x < 0.0 and x_new >= 0.0:
            # bisection refine the crossing time within [t, t+dt]
            v_lo = v.copy(); v_hi = v_next.copy()
            t_lo = t; t_hi = t + dt
            for _b in range(8):
                tm = 0.5 * (t_lo + t_hi)
                # midpoint via RK4 from t_lo
                vm = _rk4_step(f, v_lo, tm - t_lo)
                if vm[0] < 0.0:
                    v_lo, t_lo = vm, tm
                else:
                    v_hi, t_hi = vm, tm
            vm = _rk4_step(f, v_lo, t_hi - t_lo)  # refine value at t_hi
            return vm, t_hi
        v = v_next
        prev_x = x_new
        t += dt
    return None


def collect_upcrossings(f, v0, t_trans=200.0, n_cross=300, dt=0.001, seed=0):
    """Burn in, then collect n_cross upward x=0 section points (as y,z)."""
    from .systems import rk4_system, discard_transient
    v = discard_transient(f, v0, t_trans, dt)
    # force onto the upward half-crossing: advance until an upcross then use it
    r = advance_to_upcross(f, v, dt)
    if r is None:
        raise RuntimeError("no upcross found")
    v, _ = r
    pts = []
    while len(pts) < n_cross:
        r = advance_to_upcross(f, v, dt)
        if r is None:
            break
        v, _ = r
        pts.append((v[1], v[2]))   # (y, z) on section x=0, upward
    return np.array(pts)


def _return_map(s, f, dt=0.001):
    """From section point s=(y,z), return next upward-crossing section point."""
    v0 = np.array([0.0, s[0], s[1]])
    r = advance_to_upcross(f, v0, dt)
    if r is None:
        raise RuntimeError("return map failed")
    vnext, _ = r
    return np.array([vnext[1], vnext[2]])


def find_period1_upo(f, v0=None, t_trans=200.0, n_cross=300, seed=0,
                     dt=0.001):
    """Find a period-1 UPO on the section x = 0 (upward).

    Returns dict(state=(x,y,z), period=T, section=(y,z)).
    """
    rng = np.random.default_rng(seed)
    if v0 is None:
        v0 = rng.uniform(-2.0, 2.0, size=3)
    pts = collect_upcrossings(f, v0, t_trans, n_cross, dt, seed)

    # Initial guess: crossing that returns closest to itself in one return
    best = None
    best_err = np.inf
    for i in range(len(pts) - 1):
        err = np.linalg.norm(pts[i + 1] - pts[i])
        if err < best_err:
            best_err = err
            best = pts[i]

    def fvec(s):
        return _return_map(s, f, dt) - s

    sol, info, ier, msg = fsolve(fvec, best, full_output=True)
    if ier != 1:
        raise RuntimeError(f"fsolve failed to converge: {msg}")

    y, z = sol
    r = advance_to_upcross(f, np.array([0.0, y, z]), dt)
    period = r[1]
    return {"state": np.array([0.0, y, z]), "period": period, "section": (y, z)}


def upo_trajectory(f, upo, t_len, dt=0.001, n_periods=6):
    """Integrate from the UPO state for t_len seconds (drive on the UPO).

    The orbit is unstable; the trajectory stays near the UPO for a while then
    leaves it.  We return the full trajectory.
    """
    from .systems import rk4_system
    return rk4_system(f, upo["state"], 0.0, t_len, dt)
