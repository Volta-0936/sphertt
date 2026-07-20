"""017b — uncensor the d8 measurements (GPU, ~5 h).

013-016 measured every d8 MC through a 300-delay window that the d8
profiles hit at r^2 ~ 0.98 (016's own data): all d8 values, K0(d8), the
"93% of dense ceiling" record, and the x2.1 depth factor are ratios
against censored numbers.  Re-measure with DELAYS=1200, T=24000.

Configs (fp32 GPU as in 015; 017d cross-checks fp32 at d15):
  d8  random beta=0.90 K=2048  (the matched-noise point of the x2.1)
  d8  random beta=0.98 K=2048  (noise-curve top)
  d8  kron   beta=0.99 K=2048  (the "277 = 93% of ceiling" record)
  d15 random beta=0.99 K=8192  (uncensored d15 side of the ratio)
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\experiments\017")
sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from exp017_common import log, mc_validated                # noqa: E402
from sphertt import TTStateReservoir                       # noqa: E402
from sphertt.model import _sample_idx                      # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\017\results017b.jsonl"
T, WASHOUT, DELAYS = 24000, 1200, 1200
U = np.random.default_rng(7).uniform(0, 0.5, T)


def run_config(n_dims, w_kind, beta, K, seed=0):
    res = TTStateReservoir(n_dims=n_dims, beta=beta, chi_w=8, chi_x=48,
                           chi_in=4, w_kind=w_kind, seed=seed)
    res.to("cupy", "float32")
    idx = _sample_idx(np.random.default_rng(0), res.N, K)
    t0 = time.time()
    X = res.reset().run(U, readout_idx=idx).astype(np.float64)
    ms = (time.time() - t0) / T * 1e3
    mc, a_star, prof = mc_validated(X, U, T, WASHOUT, DELAYS)
    log(OUT, {"part": "window1200", "n_dims": n_dims, "w_kind": w_kind,
              "beta": beta, "chi_x": 48, "K_readout": K, "seed": seed,
              "T": T, "runner": "gpu-fp32",
              "mc1200_test": round(mc, 2),
              "mc300_test": round(float(np.sum(prof[:300])), 2),
              "alpha_star": a_star,
              "delta_bar": round(float(res.deltas_[WASHOUT:].mean()), 5),
              "ms_per_step": round(ms, 1),
              "r2_profile": np.round(prof, 4)})


run_config(8, "random-tt", 0.90, 2048)
run_config(8, "random-tt", 0.98, 2048)
run_config(8, "kron-sum", 0.99, 2048)
run_config(15, "random-tt", 0.99, 8192)
print("017b done.")
