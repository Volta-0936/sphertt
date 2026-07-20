"""017d — fp32/fp64 MC equivalence at d15 (CPU fp64, ~5 h).

011 validated the fp32 GPU backend at d8 only, yet all decisive d15
numbers ran fp32 (bond condition numbers at d15 reach 1e6-1e10).  Run
the d15 random seed-0 definitive config in CPU float64 and compare MC
against the fp32 GPU value (015: 103.72).
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\experiments\017")
sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from exp017_common import log, mc_validated                # noqa: E402
from sphertt import TTStateReservoir                       # noqa: E402
from sphertt.model import _sample_idx                      # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\017\results017d.jsonl"
T, WASHOUT, DELAYS = 16000, 300, 300
U = np.random.default_rng(7).uniform(0, 0.5, T)

res = TTStateReservoir(n_dims=15, beta=0.99, chi_w=8, chi_x=48,
                       chi_in=4, w_kind="random-tt", seed=0)
idx = _sample_idx(np.random.default_rng(0), res.N, 8192)
t0 = time.time()
X = res.reset().run(U, readout_idx=idx)
ms = (time.time() - t0) / T * 1e3
mc, a_star, prof = mc_validated(X, U, T, WASHOUT, DELAYS)
log(OUT, {"part": "fp64check", "n_dims": 15, "w_kind": "random-tt",
          "beta": 0.99, "chi_x": 48, "K_readout": 8192, "seed": 0,
          "T": T, "runner": "cpu-fp64",
          "mc300_test": round(mc, 2), "alpha_star": a_star,
          "delta_bar": round(float(res.deltas_[WASHOUT:].mean()), 5),
          "ms_per_step": round(ms, 1),
          "r2_profile": np.round(prof, 4)})
print("017d done.")
