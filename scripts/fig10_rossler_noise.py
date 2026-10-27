"""Reproduce Fig. 10: Roessler system, robustness to measurement noise.

Adopts the reservoir trained in Fig. 8 (alpha=0.002, rho=0.9), kappa = 0.2.
Measurement noise follows Eq. (14):  nu~ = nu + eps*sqrt(dt/T)*zeta, with
T the approximate oscillation period (~6 s for Roessler).

  (a) y-coupling only:  y~_D -> y_R   vs   yhat_D -> y_R (via RC)
  (b) x- and y-coupling: x~_D->x_R + y~_D->y_R   vs   x~_D->x_R + yhat_D->y_R
Ebar is plotted as a function of the noise magnitude epsilon.

Usage:
    python scripts/fig10_rossler_noise.py          # quick
    python scripts/fig10_rossler_noise.py --full   # more epsilon points
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
                          sync_error, roessler_coupling_y,
                          roessler_coupling_xy)

DT = 0.001
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KAPPA = 0.2
T_OSC = 6.0          # approximate Roessler period (seconds)
N_REAL = 6           # independent noise realizations per epsilon (full: 12)


def ebar_at(coupling_fn, vD_full, states_ts, vR0s, rc, use_reservoir,
            eps, Tf, t_start, t_end):
    ebars = []
    for i, vR0 in enumerate(vR0s):
        vD, vR = run_control(rc, roessler, coupling_fn, vD_full, states_ts,
                             vR0, kappa=KAPPA, dt=DT, Tf=Tf,
                             use_reservoir=use_reservoir, t_s=3000 + i,
                             noise_eps=eps, noise_T=T_OSC,
                             noise_seed=1000 + i)
        _, Ebar = sync_error(vD, vR, DT, t_start, t_end)
        ebars.append(Ebar)
    return np.mean(ebars), np.std(ebars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.full:
        epsilons = np.linspace(0, 15, 12)
        n_real = 12
        Tf = 400.0; t_start, t_end = 200.0, 400.0
    else:
        epsilons = np.linspace(0, 15, 6)
        n_real = 6
        Tf = 120.0; t_start, t_end = 40.0, 120.0

    tr, ts = generate_drive_trajectories(roessler, dt=DT, seed=2)
    vD_full = np.stack([ts["x"], ts["y"], ts["z"]])
    rc, _ = train_reservoir(tr["x"], tr["y"], N=500, alpha=0.002, rho=0.9,
                            seed=7)
    _, states_ts, _ = test_reservoir(rc, ts["x"], ts["y"])
    vR0s = [random_attractor_point(roessler, dt=DT, seed=50 + i)
            for i in range(n_real)]

    # (a) y-coupling only
    m_a_rc, s_a_rc, m_a_nc, s_a_nc = [], [], [], []
    for eps in epsilons:
        m, s = ebar_at(roessler_coupling_y, vD_full, states_ts, vR0s, rc,
                       True, eps, Tf, t_start, t_end)
        m_a_rc.append(m); s_a_rc.append(s)
        m, s = ebar_at(roessler_coupling_y, vD_full, None, vR0s, None,
                       False, eps, Tf, t_start, t_end)
        m_a_nc.append(m); s_a_nc.append(s)

    # (b) x- and y-coupling
    m_b_rc, s_b_rc, m_b_nc, s_b_nc = [], [], [], []
    for eps in epsilons:
        m, s = ebar_at(roessler_coupling_xy, vD_full, states_ts, vR0s, rc,
                       True, eps, Tf, t_start, t_end)
        m_b_rc.append(m); s_b_rc.append(s)
        m, s = ebar_at(roessler_coupling_xy, vD_full, None, vR0s, None,
                       False, eps, Tf, t_start, t_end)
        m_b_nc.append(m); s_b_nc.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(epsilons, m_a_rc, yerr=s_a_rc, fmt="-o", capsize=3,
                     color="#1f77b4", label="Noise Corrupted + RC")
    axes[0].errorbar(epsilons, m_a_nc, yerr=s_a_nc, fmt="-s", capsize=3,
                     color="#ff7f0e", label="Noise Corrupted")
    axes[0].set_xlabel(r"$\epsilon$"); axes[0].set_ylabel(r"$\bar{E}$")
    axes[0].set_title("(a) " + r"$\tilde{y}_D \rightarrow y_R$ / "
                      r"$\hat{y}_D \rightarrow y_R$")
    axes[0].legend(fontsize=8)

    axes[1].errorbar(epsilons, m_b_rc, yerr=s_b_rc, fmt="-o", capsize=3,
                     color="#1f77b4", label="Noise Corrupted + RC")
    axes[1].errorbar(epsilons, m_b_nc, yerr=s_b_nc, fmt="-s", capsize=3,
                     color="#ff7f0e", label="Noise Corrupted")
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_title("(b) " + r"$\tilde{x}_D\to x_R,\tilde{y}_D\to y_R$ / "
                      r"$\tilde{x}_D\to x_R,\hat{y}_D\to y_R$")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    fig.suptitle(r"Roessler: synchronization error vs measurement noise, "
                 r"$\kappa=0.2$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "fig10_rossler_noise.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
