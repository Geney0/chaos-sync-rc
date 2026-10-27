"""Reproduce Figs. 3, 4, 5: Chen system, kappa = 3.1, Tf = 1000 s.

  Fig. 3 : x_D(t) and x_R(t) on t in [970, 1000] -- coupled through the RC
           (top) and coupled ideally (bottom).
  Fig. 4 : instantaneous sync errors E_RC(t) and E_ideal(t) on t in [500,1000]
           (log-y).
  Fig. 5 : |y_D(t) - yhat_D(t)| on t in [970, 1000].

Usage:  python scripts/fig3_4_5_chen.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.systems import chen
from src.data import generate_drive_trajectories, random_attractor_point
from src.observer import (train_reservoir, test_reservoir, run_control,
                          sync_error_full, chen_coupling)

DT = 0.001
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KAPPA = 3.1
TF = 1000.0


def main():
    tr, ts = generate_drive_trajectories(chen, dt=DT, seed=1)
    vD_full = np.stack([ts["x"], ts["y"], ts["z"]])

    rc, _ = train_reservoir(tr["x"], tr["y"], N=500, alpha=0.61, rho=0.9,
                            seed=7)
    _, states_ts, _ = test_reservoir(rc, ts["x"], ts["y"])

    vR0 = random_attractor_point(chen, dt=DT, seed=3)
    t_s = 3000

    # coupled through the RC
    vD_rc, vR_rc = run_control(rc, chen, chen_coupling, vD_full, states_ts,
                               vR0, kappa=KAPPA, dt=DT, Tf=TF, t_s=t_s)
    E_rc = sync_error_full(vD_rc, vR_rc, DT)

    # coupled ideally
    vD_id, vR_id = run_control(None, chen, chen_coupling, vD_full, None, vR0,
                               kappa=KAPPA, dt=DT, Tf=TF,
                               use_reservoir=False, t_s=t_s)
    E_id = sync_error_full(vD_id, vR_id, DT)

    t_all = np.arange(vD_rc.shape[1]) * DT

    # ---------------- Fig 3: time series --------------------------------
    sel = (t_all >= 970.0) & (t_all <= 1000.0)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_all[sel], vD_rc[0, sel], color="#1f77b4", lw=0.8,
                 label=r"$x_D(t)$")
    axes[0].plot(t_all[sel], vR_rc[0, sel], color="#ff7f0e", lw=0.8,
                 label=r"$x_R(t)$")
    axes[0].set_title("Coupled through RC")
    axes[1].plot(t_all[sel], vD_id[0, sel], color="#1f77b4", lw=0.8,
                 label=r"$x_D(t)$")
    axes[1].plot(t_all[sel], vR_id[0, sel], color="#ff7f0e", lw=0.8,
                 label=r"$x_R(t)$")
    axes[1].set_title("Coupled ideally")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8); ax.set_ylabel(r"$x$")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel(r"$t$")
    fig.tight_layout()
    out1 = os.path.join(OUT, "fig3_chen_timeseries.png")
    fig.savefig(out1, dpi=150); plt.close(fig)

    # ---------------- Fig 4: sync error -------------------------------
    sel4 = (t_all >= 500.0) & (t_all <= 1000.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(t_all[sel4], E_rc[sel4], color="#1f77b4", lw=1.0,
                label=r"$E_{RC}(t)$")
    ax.semilogy(t_all[sel4], E_id[sel4], color="#ff7f0e", lw=1.0,
                label=r"$E_{ideal}(t)$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$E(t)$")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    ax.set_title(r"Chen: synchronization error, $\kappa=3.1$")
    fig.tight_layout()
    out2 = os.path.join(OUT, "fig4_chen_syncerr.png")
    fig.savefig(out2, dpi=150); plt.close(fig)

    # ---------------- Fig 5: estimation error --------------------------
    # replay the reservoir over the control horizon to get yhat_D(t)
    rc.reset(states_ts[t_s])
    yhat = np.array([rc.estimate_next(vD_rc[0, j])
                     for j in range(vD_rc.shape[1])])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_all[sel], np.abs(vD_rc[1, sel] - yhat[sel]), color="#1f77b4",
            lw=0.8)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$|y_D(t)-\hat{y}_D(t)|$")
    ax.grid(alpha=0.3)
    ax.set_title("Chen: reservoir estimation error")
    fig.tight_layout()
    out3 = os.path.join(OUT, "fig5_chen_esterr.png")
    fig.savefig(out3, dpi=150); plt.close(fig)

    for o in (out1, out2, out3):
        print("saved:", os.path.abspath(o))


if __name__ == "__main__":
    main()
