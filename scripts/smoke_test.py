"""Smoke test: verify the train -> estimate -> synchronize pipeline works.

Runs a short Chen-system reproduction: trains the reservoir to estimate y_D
from x_D, then drives a response with kappa = 3.1 (as in Fig. 3) and prints
the training/testing estimation errors and the final synchronization error.

Usage:  python scripts/smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.systems import chen
from src.data import generate_drive_trajectories, random_attractor_point
from src.observer import (train_reservoir, test_reservoir, run_control,
                          sync_error, chen_coupling, chen_coupling_xonly)

DT = 0.001


def main():
    print("== Reservoir synchronization of the Chen system (smoke test) ==")

    tr, ts = generate_drive_trajectories(chen, dt=DT, seed=1)
    print("training points:", len(tr["x"]), " testing points:", len(ts["x"]))

    # Train reservoir: input x_D, target y_D (Chen params of Fig. 3)
    rc, Delta_tr = train_reservoir(tr["x"], tr["y"],
                                   N=500, alpha=0.61, rho=0.9, seed=7)
    print(f"training estimation error  Delta_tr = {Delta_tr:.3e}")

    Delta_ts, states_ts, yhat_ts = test_reservoir(rc, ts["x"], ts["y"])
    print(f"testing  estimation error  Delta_ts = {Delta_ts:.3e}")

    # Estimation quality on testing data
    err = np.abs(yhat_ts - ts["y"]).mean()
    print(f"mean |y_D - yhat_D| on testing data = {err:.3e}")

    # Response initial condition (random point on attractor)
    vR0 = random_attractor_point(chen, dt=DT, seed=3)

    # Coupled through the reservoir: x_D->y_R and yhat_D->y_R, kappa = 3.1
    t_s = 3000
    vD_rc, vR_rc = run_control(rc, chen, chen_coupling,
                               np.stack([ts["x"], ts["y"], ts["z"]]),
                               states_ts, vR0, kappa=3.1, dt=DT, Tf=200.0,
                               t_s=t_s)
    _, Ebar_rc = sync_error(vD_rc, vR_rc, DT, 50.0, 200.0)
    print(f"sync error via reservoir     Ebar = {Ebar_rc:.3e}")

    # Ideal coupling: use true y_D (kappa = 3.1)
    vD_ideal, vR_ideal = run_control(None, chen, chen_coupling,
                                     np.stack([ts["x"], ts["y"], ts["z"]]),
                                     None, vR0, kappa=3.1, dt=DT, Tf=200.0,
                                     use_reservoir=False, t_s=t_s)
    _, Ebar_ideal = sync_error(vD_ideal, vR_ideal, DT, 50.0, 200.0)
    print(f"sync error ideal coupling    Ebar = {Ebar_ideal:.3e}")

    # x_D->y_R only (no y estimate): should NOT synchronize at kappa=3.1
    vD_x, vR_x = run_control(None, chen, chen_coupling_xonly,
                             np.stack([ts["x"], ts["y"], ts["z"]]),
                             None, vR0, kappa=3.1, dt=DT, Tf=200.0,
                             use_reservoir=False, t_s=t_s)
    _, Ebar_xonly = sync_error(vD_x, vR_x, DT, 50.0, 200.0)
    print(f"sync error x_D->y_R only     Ebar = {Ebar_xonly:.3e}")

    print("\nExpectation:")
    print("  ideal << reservoir << x-only (x-only stays unsynchronized at 3.1)")


if __name__ == "__main__":
    main()
