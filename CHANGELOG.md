# Changelog

## 0.5.0 (2026-07-19)

- **GPU backend (optional)**: `reservoir.to("cupy", "float32")` moves the
  state evolution to the GPU.  Construction stays NumPy float64 (seeds
  and W are bit-identical regardless of backend); `pip install
  sphertt[gpu]`.  All TT primitives are now backend/dtype-agnostic.
- Measured on an RTX 3070 Ti: 3.1x at the heavy end (n_dims=15,
  chi_x=48: 1060 -> 342 ms/step, i.e. a 70-min MC run in 23 min).
  float32 is physics-validated (MC 65.5 vs 64.8 fp64 reference,
  delta_bar identical to 4 decimals) — the truncation noise is orders of
  magnitude above float32 roundoff.  Guidance: use GPU+float32 for
  chi_x >= 32; below that the CPU is faster (small-matrix
  QR/SVD kernels don't amortize).
- `tt_round` is now synchronization-free on GPU: kept ranks are decided
  from shapes only, and the truncation error is returned as a device
  scalar on CuPy inputs (`deltas_` is still a NumPy array after `run`).
- 4 new tests (44 total; GPU tests skip automatically without CuPy).

## 0.4.1 (2026-07-19)

- Ship `experiments/`: scripts, JSONL logs, figures, and
  prediction-before-measurement records behind the papers.
- README: link the paper PDFs; citation metadata (CITATION.cff,
  .zenodo.json); Zenodo archival enabled from this release.

## 0.4.0 (2026-07-18)

- **`ESN.save` / `ESN.load` now support `TTStateReservoir`** (format 2:
  serializes W cores, per-channel TT input vectors, and the TT state —
  under 1 MB even at N > 10^9).
- README/design guidance updated with the validated results from the
  research notebooks (prototype/006-007): the rounding-error decomposition
  `δ̄² = (c·(1−β))² + δ_stock²`, the two-resource memory law
  `MC ≈ min{K₀·χ_x², MC_noise(δ̄)}`, and the theory-guided configuration
  `β=0.99, χ_x=48` that keeps MC ≈ 74 alive at N = 4^15 > 10^9
  (dense state 8.6 GB — unallocatable; TT model 861 KB).
- 1 new unit test (36 total).

## 0.3.0 (2026-07-18)

- **`TTStateReservoir` — native TT state evolution.**  The reservoir state
  itself now lives in tensor-train format and is never densified:
  `z = W x (TT matvec) + u·w_in (TT add) → tt_round(χ_x) → normalize`.
  The sphere normalization is a scalar division of one core (rank-free),
  and the orthogonal backbone Q_kron (rank 1) does not inflate state rank.
  At `chi_x` = the maximal rank of the tensorization the evolution is
  *exact* (machine-precision equal to the dense-state reservoir, tested).
  Per-step rounding errors are recorded in `deltas_` — the δ̄ of the
  amplification law comes built in.
- **Structured input coupling**: `tt_rank1_kron_sum` — w_in as a sum of
  `chi_in` random Kronecker rank-1 vectors, norm-matched to the dense
  heuristic.  Measured: `chi_in=4` matches dense-w_in task performance
  (delay-3: 0.017 vs 0.017; NARMA10 at N=1024: 0.42 vs 0.52).
- New TT primitives: `tt_zeros`, `tt_add`, `tt_norm`, `tt_round`
  (QR + truncated-SVD with error tracking), `tt_entries` (O(K·d·χ²)
  readout without densifying), `ttm_ttv` (TT matrix × TT vector),
  `power_iteration_norm_tt` (spectral estimate with a TT iterate — used
  automatically for N > 4^9, where the dense iterate would cost minutes
  and gigabytes).
- `ESN` readout-index sampling no longer materializes `arange(N)`
  (works at N = 4^15 > 10^9).
- `ESN.save` raises a clear error for `TTStateReservoir` (support planned).
- 6 new unit tests (35 total).

## 0.2.0 (2026-07-18)

- **Multi-channel input**: `SphereTTReservoir(n_in=...)` / `ESN(n_in=...)`
  accept (T, n_in) input sequences; auto input scaling extends to
  `0.02·sqrt(1024/N)/sqrt(n_in)`.  With `n_in=1`, scalars / (T,) arrays keep
  working and reproduce v0.1 trajectories bit-for-bit.
- **Model persistence**: `ESN.save(path)` / `ESN.load(path)` (`.npz`,
  includes TT cores, input weights, reservoir state, and readout — a few
  hundred KB even at N = 262,144).
- **Fix — `memory_capacity` with `delays > washout`**: delayed targets were
  built with `np.roll`, so for k > washout a few wrapped-around (end of
  sequence) values leaked into the training targets.  Fitting now starts at
  `max(washout, delays)` and no wrapped targets are ever used.  Results with
  the default `delays <= washout` are unchanged.
- `diagnostics.error_growth_rate` accepts (T, n_in) input.
- 6 new unit tests (29 total).

## 0.1.0 (2026-07-16)

Initial release.

- `SphereTTReservoir`: quasi-orthogonal hyperspherical reservoir with
  tensor-train connectivity `W(β) = β·Q_kron + (1−β)·W_tt` (rank ≤ 1+χ_w),
  auto input scaling ∝ 1/√N, streaming readout for very large N.
- `RidgeReadout` / `ESN`: quadratic ridge readout with validated α,
  sklearn-flavored fit/predict/score.
- `tt`: TT vector/matrix primitives (TT-SVD, random TT, Kronecker
  orthogonal, weighted TT sum, matvec that never densifies, power-iteration
  spectral estimate, memory guard on densification).
- `diagnostics`: error-direction growth rate ḡ, one-shot truncation error
  δ̄, and the amplification law A ≈ min{(1−ḡ²)^(−1/2), √2/δ̄} with
  runaway detection.
- `tasks`: NARMA-k, Mackey-Glass, NRMSE, linear memory capacity.
- 23 unit tests; NumPy is the only runtime dependency.
