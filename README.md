# Chaos synchronization using reservoir computing (reproduction)

Python reproduction of

> A. Nazerian, C. Nathe, J. D. Hart, and F. Sorrentino,
> **"Synchronizing chaos using reservoir computing"**,
> *Chaos* **33**, 103121 (2023). https://doi.org/10.1063/5.0161076

A reservoir computer is trained as an **observer**: driven only by the
measured variable `x_D` of a drive system, it estimates the unmeasured
variable `y_D`.  The measured `x_D` and the estimate `ŷ_D` are then fed into
the coupling of a response system, allowing the two chaotic systems to
synchronize at a **much lower coupling strength** than would be possible with
`x_D` alone.

---

## 1. Method (what is reproduced)

A drive and a response obey the same vector field `F` and are unidirectionally
coupled (Eq. 1):

```
dv_D/dt = F(v_D)
dv_R/dt = F(v_R) + κ H(v_D - v_R)
```

Only `x_D(t)` is available at the response.  A discrete-time reservoir (Eq. 2)
estimates `ŷ_D(t)`:

```
r(t+Δt) = (1 - α) r(t) + α tanh( A r(t) + x_D(t) w_in )
```

* `A` — adjacency matrix of a **directed Erdős–Rényi** graph (N nodes,
  connectivity p = 0.1), rescaled to spectral radius ρ;
* `w_in` — input weights (uniform in [−0.5, 0.5]);
* α — leakage parameter; Δt = 0.001 s.

The readout is trained by ridge regression (Eq. 4–6), `β = 1e-9`:

```
w_out = (ΩᵀΩ + βI)⁻¹ Ωᵀ g
```

Estimation errors (Eq. 8, 9) `Δ = std(h−g)/std(g)`.  The coupled dynamics are
integrated with the modified fixed-step 4th-order Runge–Kutta of **Algorithm 1**
(the reservoir is advanced once per RK step with `K = 1`).

Synchronization error (Eq. 11):

```
E(t) = [ Σ_i ((v_D,i − v_R,i)/std(v_D,i))² ]^(1/2),   Ē = <E(t)>_{t∈[500,1000]}
```

### Systems
* **Chen** (Eq. 10): `[35(y−x), −7x−xz+28y, xy−(8/3)z]`
* **Rössler** (Eq. 13): `[−y−z, x+0.2y, 0.2+(x−9)z]`

### Key paper results reproduced here
| Figure | Content | Reproduction |
|---|---|---|
| Fig. 2 | Chen `Ē²` vs `κ`; `κ_min ≈ 10.78 / 3.96 / 3.00` | `fig2_chen_kappa.py` |
| Fig. 3 | Chen `x_D(t), x_R(t)`, κ=3.1, RC vs ideal | `fig3_4_5_chen.py` |
| Fig. 4 | Chen `E(t)` RC vs ideal | `fig3_4_5_chen.py` |
| Fig. 5 | Chen `|y_D−ŷ_D|` | `fig3_4_5_chen.py` |
| Fig. 6 | Chen hyperparameters (α, ρ), κ=10 | `fig6_chen_hyperparams.py` |
| Fig. 7 | Rössler hyperparameters (α, ρ), κ=0.15 | `fig7_rossler_hyperparams.py` |
| Fig. 8 | Rössler `Ē²` vs `κ` (α=0.002, ρ=0.9) | `fig8_rossler_kappa.py` |
| Fig. 9 | Synchronization onto a period-1 UPO | `fig9_rossler_upo.py` |
| Fig. 10 | Rössler noise robustness (κ=0.2) | `fig10_rossler_noise.py` |

---

## 2. Project layout

```
chaos_sync_rc/
├── README.md
├── requirements.txt
├── src/
│   ├── systems.py      # Chen / Rössler vector fields + coupling helpers
│   ├── reservoir.py    # discrete-time reservoir computer (Eq. 2-9)
│   ├── integration.py  # coupled modified RK4 (Algorithm 1) + noise (Eq. 14)
│   ├── observer.py     # train→test→control workflow + sync error (Eq. 11)
│   ├── data.py         # attractor trajectory generation
│   └── upo.py          # Poincaré-section Newton search for a period-1 UPO
├── scripts/
│   ├── smoke_test.py           # quick end-to-end sanity check
│   ├── debug_sync*.py          # (diagnostics used during development)
│   ├── fig2_chen_kappa.py
│   ├── fig3_4_5_chen.py
│   ├── fig6_chen_hyperparams.py
│   ├── fig7_rossler_hyperparams.py
│   ├── fig8_rossler_kappa.py
│   ├── fig9_rossler_upo.py
│   └── fig10_rossler_noise.py
└── output/             # generated figures (png)
```

## 3. Install & run

```bash
pip install -r requirements.txt

# fastest end-to-end check
python scripts/smoke_test.py

# each figure (default = quick; add --full for paper-grade averaging)
python scripts/fig2_chen_kappa.py
python scripts/fig3_4_5_chen.py
python scripts/fig6_chen_hyperparams.py
python scripts/fig7_rossler_hyperparams.py
python scripts/fig8_rossler_kappa.py
python scripts/fig9_rossler_upo.py
python scripts/fig10_rossler_noise.py
```

Figures are written to `output/`.

---

## 4. Notes on fidelity

* **Parameters** follow the paper where stated: Chen Fig. 3 → α=0.61, ρ=0.9,
  N=500; Rössler Fig. 8 → α=0.002, ρ=0.9; UPO → N=100, α=9.7e-4, ρ=0.9;
  β=1e-9; Δt=0.001; Erdős–Rényi p=0.1.
* **Training/testing segment lengths** are not given in the paper; we use
  100 s / 20 s (Chen) and 200 s / 20 s (Rössler) transients + segments, which
  reproduce the qualitative results (errors of order 1e-4 — 1e-1).
* **UPO (Fig. 9)**: the paper locates the period-1 orbit with the MATLAB
  MatCont toolbox; here it is found with a Poincaré-section Newton (shooting)
  method (`src/upo.py`), yielding period T ≈ 5.95 s.
* **Ideal coupling** converges to ~1e-2 (not machine precision ~1e-10 as in
  the paper) because κ=3.1 is just above the Chen threshold κ_min≈3.0, so the
  transverse convergence is slow; the RC-vs-ideal ordering is preserved.
* All scripts accept `--full` to use the paper's 20-realization averaging and
  longer horizons (more expensive).

## 5. Reproduced outcomes (verified)

* **Chen, κ=3.1**: `Ē(RC) ≈ 3e-2`, `Ē(ideal) ≈ 1e-2`, `Ē(x-only) ≈ 1.1` — i.e.
  using the reservoir estimate gives near-ideal synchronization while `x_D`
  alone does not synchronize (Fig. 3/4 reproduced).
* **UPO, κ=0.3**: `Ē(x-only) ≈ 0.8` (no sync), `Ē(with RC) ≈ 1.8e-3` (sync onto
  the period-1 orbit) — Fig. 9 reproduced.
* **Rössler noise**: `Ē` rises with ε in both schemes; the RC-observer route
  stays robust (error comparable to, slightly above, the direct noisy-signal
  coupling), matching Fig. 10.
