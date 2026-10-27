"""Discrete-time reservoir computer (Eq. 2-9 of the paper).

State update (Eq. 2):
    r(t+dt) = (1 - alpha) r(t) + alpha * tanh( A r(t) + s(t) w_in )

A is the adjacency matrix of a directed Erdos-Renyi graph (connectivity
p = 0.1) rescaled to spectral radius rho.  The scalar drive signal s(t) is
fed through the input vector w_in (default: uniform in [-0.5, 0.5]).

Training (Eq. 3-8):
    Omega (rows are reservoir states at each time step + a bias column of 1)
    w_out = (Omega^T Omega + beta I)^{-1} Omega^T g
    Delta_tr = std(Omega w_out - g) / std(g)

Testing (Eq. 9) reuses w_out on fresh drive data.
"""

import numpy as np


class ReservoirComputer:
    """Trainable reservoir computer that estimates an unmeasured state."""

    def __init__(self, N=500, alpha=0.61, rho=0.9, p=0.1, beta=1e-9,
                 win_lo=-0.5, win_hi=0.5, dt=0.001, seed=None):
        self.N = N
        self.alpha = alpha
        self.rho = rho
        self.p = p
        self.beta = beta
        self.dt = dt

        rng = np.random.default_rng(seed)
        # Directed Erdos-Renyi adjacency matrix (elements in {0,1}).
        A = (rng.random((N, N)) < p).astype(float)
        # Rescale to spectral radius rho: A <- A * rho / lambda_max.
        eig = np.linalg.eigvals(A)
        lambda_max = np.max(np.real(eig))
        if abs(lambda_max) < 1e-12:
            raise ValueError("Reservoir matrix has a zero largest eigenvalue.")
        self.A = A * (rho / lambda_max)
        # Input weights.
        self.w_in = rng.uniform(win_lo, win_hi, size=N)

        self.w_out = None          # readout weights (N+1,)
        self.r = None              # current reservoir state (N,)

    # ------------------------------------------------------------------ #
    # evolution
    # ------------------------------------------------------------------ #
    def evolve(self, s, r0=None, steps=None):
        """Advance the reservoir for `steps` steps with scalar input series `s`.

        s : (steps,) array of drive-signal values.
        Returns reservoir states (steps, N), one per time step.
        """
        steps = steps if steps is not None else len(s)
        r = np.zeros(self.N) if r0 is None else r0.copy()
        states = np.empty((steps, self.N))
        for t in range(steps):
            r = (1 - self.alpha) * r + self.alpha * np.tanh(
                self.A @ r + s[t] * self.w_in)
            states[t] = r
        self.r = r
        return states

    # ------------------------------------------------------------------ #
    # training / testing
    # ------------------------------------------------------------------ #
    def train(self, s, g, return_estimate=False):
        """Train on drive signal `s` with training target `g` (both length T1).

        Returns (w_out, Delta_tr [, h]) where h = Omega w_out is the fit.
        """
        T1 = len(s)
        states = self.evolve(s, steps=T1)
        Omega = np.hstack([states, np.ones((T1, 1))])          # (T1, N+1)
        # Ridge regression closed form, Eq. (6).
        Id = np.eye(self.N + 1)
        Omega_pinv = np.linalg.solve(Omega.T @ Omega + self.beta * Id,
                                     Omega.T)
        self.w_out = Omega_pinv @ g                             # (N+1,)
        h = Omega @ self.w_out
        Delta_tr = np.std(h - g) / np.std(g)
        if return_estimate:
            return self.w_out, Delta_tr, h
        return self.w_out, Delta_tr

    def test(self, s, g):
        """Testing phase (Eq. 9): reuse w_out on fresh drive data."""
        states = self.evolve(s, steps=len(s))
        Omega = np.hstack([states, np.ones((len(s), 1))])
        h_ts = Omega @ self.w_out
        Delta_ts = np.std(h_ts - g) / np.std(g)
        return Delta_ts, h_ts

    # ------------------------------------------------------------------ #
    # closed-loop estimation (used inside the coupled integrator)
    # ------------------------------------------------------------------ #
    def estimate_next(self, s_t):
        """Update reservoir one step and return the scalar estimate yhat.

        Implements the "R/O" readout used at each RK substep: the estimate is
        r(t)^T w_out computed from the state *before* the update, then the
        reservoir advances one step (Algorithm 1 lines 4 & 11).
        """
        yhat = self.r @ self.w_out[:-1] + self.w_out[-1]
        self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(
            self.A @ self.r + s_t * self.w_in)
        return yhat

    def reset(self, r0=None):
        """Reset reservoir state (optionally to a provided state)."""
        if r0 is None:
            self.r = np.zeros(self.N)
        else:
            self.r = r0.copy()
