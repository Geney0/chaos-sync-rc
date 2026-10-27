"""Reproduce Fig. 8: Roessler system synchronization error vs coupling strength.

alpha = 0.002, rho = 0.9 (see text before Fig. 8).  Three schemes:
  (a) x_D -> x_R          (measured x only)
  (b) yhat_D -> y_R       (reservoir estimate of y)
  (c) x_D -> x_R and yhat_D -> y_R
Averaged over 20 realizations, Ebar over t in [500, 1000].

Usage:
    python scripts/fig8_rossler_kappa.py          # quick
    python scripts/fig8_rossler_kappa.py --full   # 20 realizations
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.systems import roessler
from src.data import generate_drive_trajectories, random_attractor_point
from src.observer import (train_reservoir, test_reservoir, run_control,
                          sync_error, roessler_coupling_x, roessler_coupling_y,
                          roessler_coupling_xy)

DT = 0.001
OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def ebar_grid(coupling_fn, vD_full, states_ts, kappas, vR0s, rc,
              use_reservoir, Tf, t_start, t_end):
    means, stds = [], []
    for k in kappas:
        ebars = []
        for i, vR0 in enumerate(vR0s):
            vD, vR = run_control(rc, roessler, coupling_fn, vD_full,
                                 states_ts, vR0, kappa=k, dt=DT, Tf=Tf,
                                 use_reservoir=use_reservoir, t_s=3000 + i)
            _, Ebar = sync_error(vD, vR, DT, t_start, t_end)
            ebars.append(Ebar)
        means.append(np.mean(ebars)); stds.append(np.std(ebars))
    return np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    n_rep = 20 if args.full else 3
    Tf = 1000.0 if args.full else 220.0
    t_start, t_end = (500.0, 1000.0) if args.full else (60.0, 220.0)

    tr, ts = generate_drive_trajectories(roessler, dt=DT, seed=2)
    vD_full = np.stack([ts["x"], ts["y"], ts["z"]])

    # Roessler reservoir: alpha = 0.002, rho = 0.9 (Fig. 8 settings)
    rc, _ = train_reservoir(tr["x"], tr["y"], N=500, alpha=0.002, rho=0.9,
                            seed=7)
    _, states_ts, _ = test_reservoir(rc, ts["x"], ts["y"])

    vR0s = [random_attractor_point(roessler, dt=DT, seed=30 + i) for i in range(n_rep)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    # (a) x_D -> x_R
    kap = np.linspace(0, 0.4, 9)
    m_a, s_a = ebar_grid(roessler_coupling_x, vD_full, None, kap, vR0s, None,
                         False, Tf, t_start, t_end)
    axes[0].errorbar(kap, m_a**2, yerr=2*m_a*s_a, fmt="-o", capsize=3,
                     color="#1f77b4")
    axes[0].set_xlabel(r"$\kappa$"); axes[0].set_ylabel(r"$\bar{E}^2$")
    axes[0].set_title("(a) " + r"$x_D \rightarrow x_R$")

    # (b) yhat_D -> y_R
    m_b, s_b = ebar_grid(roessler_coupling_y, vD_full, states_ts, kap, vR0s,
                         rc, True, Tf, t_start, t_end)
    axes[1].errorbar(kap, m_b**2, yerr=2*m_b*s_b, fmt="-s", capsize=3,
                     color="#ff7f0e")
    axes[1].set_xlabel(r"$\kappa$")
    axes[1].set_title("(b) " + r"$\hat{y}_D \rightarrow y_R$")

    # (c) both
    m_c, s_c = ebar_grid(roessler_coupling_xy, vD_full, states_ts, kap, vR0s,
                         rc, True, Tf, t_start, t_end)
    axes[2].errorbar(kap, m_c**2, yerr=2*m_c*s_c, fmt="-^", capsize=3,
                     color="#2ca02c")
    axes[2].set_xlabel(r"$\kappa$")
    axes[2].set_title("(c) both")

    for ax in axes:
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    fig.suptitle("Roessler system — synchronization error vs coupling strength "
                 f"({n_rep} realizations)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "fig8_rossler_kappa.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
