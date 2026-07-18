# sphertt

[![DOI](https://zenodo.org/badge/1305159049.svg)](https://zenodo.org/badge/latestdoi/1305159049)

**Hyperspherical reservoir computing with tensor-train connectivity.**
Reservoirs beyond the memory wall — with a built-in, quantitatively
validated theory of when truncation is safe.

- Pure NumPy. One dependency. No GPU required.
- The connectivity matrix `W` of an `N = 262,144`-unit reservoir fits in
  **73 KB** where a dense `W` would need **512 GB** — and task accuracy
  matches the dense-feasible regime.
- Memory capacity *deepens* with `N` at fixed cost: at `N = 4^9` the memory
  function stays above `r² > 0.95` beyond a 300-step window.
- A `diagnostics` module predicts truncation-error amplification before you
  run: `A ≈ min{(1 − ḡ²)^(−1/2), √2/δ̄}` (94% of measurements within 2×).

## Why this works

Two structural facts, discovered and validated in the accompanying research:

1. **The sphere normalization `x → z/‖z‖` is the nonlinearity that costs
   nothing in tensor-train rank.** Element-wise nonlinearities (tanh)
   destroy TT structure; dividing by a scalar norm never does.
2. **Exactly orthogonal connectivity at TT rank 1.** A Kronecker product of
   small orthogonal matrices is an exact `N×N` orthogonal operator stored in
   `O(d)` parameters. The quasi-orthogonal mix
   `W(β) = β·Q_kron + (1−β)·W_tt` (total rank ≤ 1 + χ_w) sits in the
   empirically identified stability band at `β ∈ [0.8, 0.85]`: memory
   capacity near its maximum, truncation-error amplification bounded (≤ 4
   across seeds). Mid-`β` mixtures run away (up to 34×) — the library
   defaults keep you out of that zone, and `diagnostics` tells you why.

## Install

```bash
pip install .            # from a clone
# or
pip install sphertt      # once published to PyPI
```

## Quickstart

```python
import numpy as np
from sphertt import ESN
from sphertt.tasks import narma

u, y = narma(4000, order=10, rng=np.random.default_rng(0))

esn = ESN(n_dims=8, seed=0)        # N = 4**8 = 65,536 units
esn.fit(u[:2400], y[:2400])
print("test NRMSE:", esn.score(u[2400:], y[2400:]))
print(esn.reservoir.memory_ledger())
# {'N': 65536, 'tt_w_bytes': 64512, 'dense_w_bytes': 34359738368,
#  'compression': 532610.0, ...}

esn.save("model.npz")              # a few hundred KB even at N = 262,144
esn = ESN.load("model.npz")        # resumes from the saved reservoir state
```

Multi-channel input: build with `ESN(n_in=3, ...)` and pass `(T, 3)` arrays.

Lower-level control:

```python
from sphertt import SphereTTReservoir, RidgeReadout, memory_capacity

res = SphereTTReservoir(n_dims=9, beta=0.8, chi_w=8, seed=0)  # N = 262,144
X = res.run(u, readout_idx=np.arange(2048))   # stream 2048 units only
mc, profile = memory_capacity(X, u, delays=300, return_profile=True)
```

Truncation-safety diagnostics (the part no other reservoir library has):

```python
from sphertt import amplification_report, SphereTTReservoir

res = SphereTTReservoir(n_dims=10, mode_size=2, beta=0.8, seed=0)
rep = amplification_report(res, u[:1500], chi_x=16)
# {'g_bar': 0.97, 'delta_bar': 0.03, 'amp_predicted': 4.1,
#  'amp_measured': 3.8, 'runaway': False, ...}
```

## TT-state evolution (v0.3): the state never exists densely

`SphereTTReservoir` compresses `W` but keeps the state dense (`O(N)`).
`TTStateReservoir` keeps the **state** in TT format too — matvec, input
injection, rounding to bond dimension `chi_x`, and the sphere
normalization (a scalar division of one core: rank-free) all happen
without ever materializing an `N`-vector:

```python
from sphertt import TTStateReservoir

res = TTStateReservoir(n_dims=15, chi_x=16, seed=0)  # N = 4**15 > 10^9
X = res.run(u, readout_idx=idx)      # entries extracted straight from TT
print(res.deltas_.mean())            # per-step rounding error, recorded free
```

Two regimes, sharply separated (see `prototype/005`):

- **Exact regime** — `chi_x` at the maximal rank of the tensorization:
  bit-compatible with the dense-state reservoir (tested to machine
  precision), at TT cost.
- **Truncated regime** — `chi_x` below maximal rank: per-step rounding
  error `δ̄` is injected into norm-preserving dynamics.  The truncated
  reservoir is not a degraded copy of the exact one — it is its own
  dynamical system with runtime-measurable noise `δ̄` (`deltas_`).

Design rules for the truncated regime, validated in `prototype/006-007`:

- **Rounding-error decomposition**: `δ̄² ≈ (c·(1−β))² + δ_stock²`
  (fits within 2-5%).  The orthogonal backbone preserves TT structure;
  only the random part injects incompressible error — raise `beta` to
  suppress it.  The floor `δ_stock` is the rank cost of the memory itself.
- **Two-resource memory law**: `MC ≈ min{K₀·χ_x², MC_noise(δ̄)}` —
  memory needs both rank budget (`chi_x`) and retention quality (low
  `δ̄`).  Raise `chi_x` and `beta` *together*.
- **Headline (theory-guided config)**: at `beta=0.99, chi_x=48`, memory
  capacity **74** survives at `N = 4**15 > 10^9` — where the dense state
  alone (8.6 GB) cannot be allocated — with the whole model in **861 KB**
  (~1 s/step on a plain CPU).  The naive config (`beta=0.8, chi_x=16`)
  scores 0.09 there: an ~800× difference delivered by the theory.

Practical guidance: break the *W* memory wall with `SphereTTReservoir`
(dense state, best task performance up to RAM limits ~ N = 10^6), and use
`TTStateReservoir` beyond that — in its exact regime, or truncated with
the design rules above.

## Benchmarks (from the research repo, NARMA10, quadratic ridge readout)

| N | dense `W` | TT `W` | compression | test NRMSE | MC (300-delay window) |
|---|---|---|---|---|---|
| 1,024 | 8 MB | 33 KB | 251× | 0.33–0.42 | 254 |
| 16,384 | 2 GB | 53 KB | 39,662× | 0.41 | 293 |
| 65,536 | 32 GB | 63 KB | 532,610× | 0.41–0.44 | — |
| 262,144 | **512 GB** | **73 KB** | 7,341,824× | 0.42 | 293 (window-limited) |

Runs on a plain CPU: 0.24 ms/step at `N = 1024`, 49 ms/step at `N = 262,144`.

## What decides stability (the 30-second theory)

Normalized reservoirs are *constitutively critical*: their largest tangent
Lyapunov exponent is ≈ 1 for any parameters, so the standard stability
metric carries no information. What decides the fate of truncation errors is
the mean growth rate `ḡ` along the **actual error direction**. In the convex
mix, the normalization denominator suffers a *non-interference dip*
`⟨‖z‖⟩ ≈ √(β² + (1−β)²)` at intermediate `β` while the singular-value spread
of `W` grows — together these push `ḡ` past 1 and errors run away. At
`β ≥ 0.8` both effects are tame and the amplification law
`A ≈ min{(1−ḡ²)^(−1/2), √2/δ̄}` predicts measurements with median ratio
0.85 (94% within 2×). `sphertt.diagnostics` measures `ḡ` and `δ̄` for your
configuration and evaluates the law.

## Scope and honest limitations

- Multi-channel input is supported since v0.2 (`n_in=...`); multi-output
  targets are untested.
- The stability band and amplification law were validated at `N ≤ 262,144`
  and state tensorizations `[2]^10 / [4]^d`; theory experiments at larger
  `N` are ongoing.
- `ESN.score` performance on NARMA10 is readout-limited above `N ≈ 4096`
  (the task only needs 10 steps of memory); the large-`N` payoff appears on
  long-memory tasks — see the memory-capacity numbers.
- This is research software (v0.1, alpha). The API may move.

## Papers

Two papers (drafts, PDFs attached to the
[latest release](https://github.com/Volta-0936/sphertt/releases/latest)):

1. [Quasi-Orthogonal Tensor-Train Reservoirs: Hyperspherical Reservoir
   Computing Beyond the Memory Wall](https://github.com/Volta-0936/sphertt/releases/latest/download/okamura2026-quasi-orthogonal-tt-reservoirs.pdf)
2. [Truncated Tensor-Train Reservoirs Are Reservoirs in Their Own Right:
   A Two-Resource Law for Memory Beyond the State
   Wall](https://github.com/Volta-0936/sphertt/releases/latest/download/okamura2026-two-resource-law-beyond-state-wall.pdf)

## Experiments

The `experiments/` directory ships the scripts, JSONL logs, and figures
behind the papers, including the prediction-before-measurement records
(see `experiments/README.md`).

## Cite

Paper in preparation. Until then, please cite this repository.

## License

MIT
