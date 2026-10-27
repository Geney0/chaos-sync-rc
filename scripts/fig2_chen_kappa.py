"""Reproduce Fig. 2: Chen system synchronization error vs coupling strength.

Three coupling schemes (random on-attractor initial conditions, averaged over
N_REALIZ realizations):
  (a) x_D -> y_R                 (only the measured x_D is used)
  (b) yhat_D -> y_R              (reservoir estimate only)
  (c) x_D -> y_R and yhat_D->y_R (both, Eq. 12)

The paper reports kappa_min ~ 10.78 / 3.96 / 3.00 respectively and averages
over 20 realizations with E averaged on t in [500, 1000].

Usage:
    python scripts/fig2_chen_kappa.py            # quick check (3 realizations)
    python scripts/fig2_chen_kappa.py --full     # paper settings (20 realizations)
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
                          sync_error, chen_coupling, chen_coupling_xonly,
                          chen_coupling_yonly)

DT = 0.001
OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def ebar_grid(coupling_fn, vD_ts, states_ts, yD_true, kappas, vR0s, rc,
              use_reservoir, Tf, t_start, t_end, seed):
    """Return mean and std of Ebar across realizations for each kappa."""
    means, stds = [], []
    for k in kappas:
        ebars = []
        for i, vR0 in enumerate(vR0s):
            vD, vR = run_control(rc, chen, coupling_fn, vD_ts, states_ts,
                                 vR0, kappa=k, dt=DT, Tf=Tf,
                                 use_reservoir=use_reservoir,
                                 t_s=3000 + i)
            _, Ebar = sync_error(vD, vR, DT, t_start, t_end)
            ebars.append(Ebar)
        means.append(np.mean(ebars)); stds.append(np.std(ebars))
    return np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="paper settings: 20 realizations, Tf=1000 s")
    args = ap.parse_args()
    N_REALIZ = 20 if args.full else 3
    Tf = 1000.0 if args.full else 220.0
    t_start, t_end = (500.0, 1000.0) if args.full else (60.0, 220.0)

    # ---- generate data ----------------------------------------------------
    tr, ts = generate_drive_trajectories(chen, dt=DT, seed=1)
    vD_full = np.stack([ts["x"], ts["y"], ts["z"]])

    # ---- train one reservoir (estimate y_D from x_D) ----------------------
    rc, _ = train_reservoir(tr["x"], tr["y"], N=500, alpha=0.61, rho=0.9,
                            seed=7)
    _, states_ts, _ = test_reservoir(rc, ts["x"], ts["y"])

    # ---- response initial conditions (20 independent attractor points) ----
    vR0s = [random_attractor_point(chen, dt=DT, seed=10 + i) for i in range(N_REALIZ)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    # (a) x_D -> y_R : no reservoir, no y-coupling
    kap_a = np.linspace(0, 20, 9)
    m_a, s_a = ebar_grid(chen_coupling_xonly, vD_full, None, None, kap_a,
                         vR0s, None, use_reservoir=False, Tf=Tf,
                         t_start=t_start, t_end=t_end, seed=0)
    ax = axes[0]
    ax.errorbar(kap_a, m_a**2, yerr=2 * m_a * s_a, fmt="-o", capsize=3,
                color="#1f77b4")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\bar{E}^2$")
    ax.set_title("(a) " + r"$x_D \rightarrow y_R$")
    ax.axvline(10.78, color="k", ls="--", lw=0.8)
    ax.annotate(r"$\kappa_{min}\!\approx\!10.78$", (10.78, 0.35 * ax.get_ylim()[1]),
                ha="center", fontsize=8)

    # (b) yhat_D -> y_R (via reservoir)
    kap_b = np.linspace(0, 8, 9)
    m_b, s_b = ebar_grid(chen_coupling_yonly, vD_full, states_ts, None, kap_b,
                         vR0s, rc, use_reservoir=True, Tf=Tf,
                         t_start=t_start, t_end=t_end, seed=0)
    ax = axes[1]
    ax.errorbar(kap_b, m_b**2, yerr=2 * m_b * s_b, fmt="-s", capsize=3,
                color="#ff7f0e")
    ax.set_xlabel(r"$\kappa$")
    ax.set_title("(b) " + r"$\hat{y}_D \rightarrow y_R$")

    # (c) both x_D -> y_R and yhat_D -> y_R
    kap_c = np.linspace(0, 8, 9)
    m_c, s_c = ebar_grid(chen_coupling, vD_full, states_ts, None, kap_c,
                         vR0s, rc, use_reservoir=True, Tf=Tf,
                         t_start=t_start, t_end=t_end, seed=0)
    ax = axes[2]
    ax.errorbar(kap_c, m_c**2, yerr=2 * m_c * s_c, fmt="-^", capsize=3,
                color="#2ca02c")
    ax.set_xlabel(r"$\kappa$")
    ax.set_title("(c) " + r"$x_D \rightarrow y_R$, $\hat{y}_D \rightarrow y_R$")

    for ax in axes:
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    fig.suptitle("Chen system — synchronization error vs coupling strength "
                 f"({N_REALIZ} realizations)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "fig2_chen_kappa.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
