"""Reproduce Fig. 9: synchronization onto an unstable periodic orbit (Roessler).

A period-1 UPO is located with a Poincare-section Newton method (see
src/upo.py; the paper uses MatCont).  The drive is pinned onto the UPO signal.
The reservoir (N=100, alpha=9.7e-4, rho=0.9) is trained on the UPO signal to
estimate y_D from x_D.

  (a) coupling x_D -> x_R only, kappa = 0.3  -> does NOT synchronize;
  (b) coupling x_D -> x_R and yhat_D -> y_R  -> synchronizes onto the UPO.
Time window [1000, 1300] s as in the paper.

Usage:  python scripts/fig9_rossler_upo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.systems import roessler
from src.upo import find_period1_upo, advance_to_upcross
from src.data import random_attractor_point
from src.reservoir import ReservoirComputer
from src.observer import run_control, sync_error_full, \
    roessler_coupling_x, roessler_coupling_xy

DT = 0.001
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KAPPA = 0.3
TF = 1300.0


def build_upo_signal(f, upo, t_end, dt):
    """Pinned periodic drive signal: repeat one UPO period to t_end seconds."""
    T = upo["period"]
    n_per = int(np.ceil(t_end / T))
    # integrate exactly one period from the UPO state
    v = upo["state"].copy()
    frames = [v]
    t = 0.0
    while t < T - 0.5 * dt:
        v = _rk4_roessler(f, v, dt)
        t += dt
        frames.append(v)
    one_period = np.array(frames).T          # (3, nT)
    nT = one_period.shape[1]
    # tile to required length
    n = int(round(t_end / dt))
    reps = int(np.ceil((n + 1) / nT))
    sig = np.tile(one_period, reps)[:, :n + 1]
    return sig


def _rk4_roessler(f, v, dt):
    k1 = f(v); k2 = f(v + 0.5*dt*k1); k3 = f(v + 0.5*dt*k2); k4 = f(v + dt*k3)
    return v + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def main():
    upo = find_period1_upo(roessler, seed=0)
    print(f"UPO state={upo['state']}  period={upo['period']:.3f} s")

    # ---- pinned periodic drive signal (Fig. 9 setup) ----------------------
    sig = build_upo_signal(roessler, upo, TF, DT)

    # ---- reservoir trained on the UPO signal (N=100) ---------------------
    xD = sig[0]; yD = sig[1]
    rc = ReservoirComputer(N=100, alpha=9.7e-4, rho=0.9, seed=7)
    _, Dtr = rc.train(xD, yD)
    print(f"UPO training estimation error  Delta_tr = {Dtr:.3e}")

    # reserve reservoir state after training to seed the control phase
    r_train_end = rc.r.copy()

    # response starts on the chaotic attractor (different from UPO)
    vR0 = random_attractor_point(roessler, dt=DT, seed=3)

    # ---- (a) x_D -> x_R only ----------------------------------------------
    vD_a, vR_a = run_control(None, roessler, roessler_coupling_x, sig, None,
                             vR0, kappa=KAPPA, dt=DT, Tf=TF,
                             use_reservoir=False, t_s=0, vD_ref=sig)
    # ---- (b) x_D -> x_R and yhat_D -> y_R ---------------------------------
    rc.reset(r_train_end)
    vD_b, vR_b = run_control(rc, roessler, roessler_coupling_xy, sig, None,
                             vR0, kappa=KAPPA, dt=DT, Tf=TF,
                             use_reservoir=True, t_s=0, vD_ref=sig)

    t_all = np.arange(vD_a.shape[1]) * DT
    sel = (t_all >= 1000.0) & (t_all <= 1300.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_all[sel], vD_a[0, sel], color="#1f77b4", lw=0.8,
                 label=r"$x_D(t)$")
    axes[0].plot(t_all[sel], vR_a[0, sel], color="#ff7f0e", lw=0.8,
                 label=r"$x_R(t)$")
    axes[0].set_title("(a) " + r"$x_D \rightarrow x_R$, " + r"$\kappa=0.3$")
    axes[1].plot(t_all[sel], vD_b[0, sel], color="#1f77b4", lw=0.8,
                 label=r"$x_D(t)$")
    axes[1].plot(t_all[sel], vR_b[0, sel], color="#ff7f0e", lw=0.8,
                 label=r"$x_R(t)$")
    axes[1].set_title("(b) " + r"$x_D \rightarrow x_R$, $\hat{y}_D \rightarrow y_R$, "
                      r"$\kappa=0.3$")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8); ax.set_ylabel(r"$x$")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel(r"$t$")
    fig.suptitle("Roessler: synchronization onto the period-1 UPO")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "fig9_rossler_upo.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", os.path.abspath(out))

    # quantitative check
    E_a = sync_error_full(vD_a, vR_a, DT); E_b = sync_error_full(vD_b, vR_b, DT)
    print(f"Ebar(a) x-only over [1000,1300] = {E_a[sel].mean():.3e}")
    print(f"Ebar(b) with RC over [1000,1300] = {E_b[sel].mean():.3e}")


if __name__ == "__main__":
    main()
