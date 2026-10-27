"""Reproduce Fig. 6: Chen system hyperparameter scans.

Coupling is x_D -> x_R and yhat_D -> y_R with kappa = 10 (see caption).
  (a) leakage alpha varied, rho = 0.9;
  (b) spectral radius rho varied, alpha = 0.5.
Each point plots training (Delta_tr), testing (Delta_ts) and synchronization
(Ebar) errors with error bars over realizations.

Usage:
    python scripts/fig6_chen_hyperparams.py          # quick
    python scripts/fig6_chen_hyperparams.py --full   # denser scans
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.systems import chen
from src.data import generate_drive_trajectories, random_attractor_point
from src.observer import (train_reservoir, test_reservoir, run_control,
                          sync_error, chen_coupling_xy)

DT = 0.001
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KAPPA = 10.0


def run_point(alpha, rho, tr, ts, vD_full, vR0s, Tf, t_start, t_end):
    """Train at (alpha, rho), return (Delta_tr, Delta_ts, Ebar_mean, Ebar_std)."""
    rc, Dtr = train_reservoir(tr["x"], tr["y"], N=500, alpha=alpha, rho=rho,
                              seed=7)
    Dts, states_ts, _ = test_reservoir(rc, ts["x"], ts["y"])
    ebars = []
    for i, vR0 in enumerate(vR0s):
        vD, vR = run_control(rc, chen, chen_coupling_xy, vD_full, states_ts,
                             vR0, kappa=KAPPA, dt=DT, Tf=Tf, t_s=3000 + i)
        _, Ebar = sync_error(vD, vR, DT, t_start, t_end)
        ebars.append(Ebar)
    return Dtr, Dts, np.mean(ebars), np.std(ebars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    n_rep = 4 if args.full else 2
    Tf = 300.0 if args.full else 150.0
    t_start, t_end = (200.0, 300.0) if args.full else (50.0, 150.0)

    tr, ts = generate_drive_trajectories(chen, dt=DT, seed=1)
    vD_full = np.stack([ts["x"], ts["y"], ts["z"]])
    vR0s = [random_attractor_point(chen, dt=DT, seed=20 + i) for i in range(n_rep)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) alpha scan, rho = 0.9
    alphas = np.linspace(0.05, 1.0, 10) if args.full else np.linspace(0.1, 1.0, 6)
    Dtr_a, Dts_a, Eb_a, Es_a = [], [], [], []
    for a in alphas:
        dtr, dts, e, es = run_point(a, 0.9, tr, ts, vD_full, vR0s, Tf,
                                    t_start, t_end)
        Dtr_a.append(dtr); Dts_a.append(dts); Eb_a.append(e); Es_a.append(es)
    ax = axes[0]
    ax.semilogy(alphas, Dtr_a, "-o", color="#1f77b4", ms=3, label=r"$\Delta_{tr}$")
    ax.semilogy(alphas, Dts_a, "-s", color="#d62728", ms=3, label=r"$\Delta_{ts}$")
    ax.errorbar(alphas, Eb_a, yerr=Es_a, fmt="-^", ms=3, capsize=3,
                color="#ffbf00", label=r"$\bar{E}$")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("error")
    ax.set_title("(a) " + r"$\rho=0.9$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (b) rho scan, alpha = 0.5
    rhos = np.linspace(0.05, 1.0, 10) if args.full else np.linspace(0.1, 1.0, 6)
    Dtr_b, Dts_b, Eb_b, Es_b = [], [], [], []
    for r in rhos:
        dtr, dts, e, es = run_point(0.5, r, tr, ts, vD_full, vR0s, Tf,
                                    t_start, t_end)
        Dtr_b.append(dtr); Dts_b.append(dts); Eb_b.append(e); Es_b.append(es)
    ax = axes[1]
    ax.semilogy(rhos, Dtr_b, "-o", color="#1f77b4", ms=3, label=r"$\Delta_{tr}$")
    ax.semilogy(rhos, Dts_b, "-s", color="#d62728", ms=3, label=r"$\Delta_{ts}$")
    ax.errorbar(rhos, Eb_b, yerr=Es_b, fmt="-^", ms=3, capsize=3,
                color="#ffbf00", label=r"$\bar{E}$")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("(b) " + r"$\alpha=0.5$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle(r"Chen system, $x_D \to x_R$, $\hat{y}_D \to y_R$, "
                 r"$\kappa=10$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "fig6_chen_hyperparams.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
