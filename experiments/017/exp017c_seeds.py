"""017c — seed replication for the d15 definitive-protocol claims (GPU, ~8 h).

Every d15 number from 014 onward is seed 0, while the project's own v1
data shows a 2.4x seed spread at d15.  Replicate seeds 1/2 for both
structures under the exact 014/015 definitive protocol
(T=16000, DELAYS=300, K=8192, alpha grid extended at both ends).

Judged claims:
  * "x2.1 depth penalty"        — does it survive the seed spread?
  * "structure reversal at d15" — random > kron on how many seeds?
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\experiments\017")
sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from exp017_common import log, mc_validated                # noqa: E402
from sphertt import TTStateReservoir                       # noqa: E402
from sphertt.model import _sample_idx                      # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\017\results017c.jsonl"
T, WASHOUT, DELAYS = 16000, 300, 300
U = np.random.default_rng(7).uniform(0, 0.5, T)


def run_config(w_kind, seed):
    res = TTStateReservoir(n_dims=15, beta=0.99, chi_w=8, chi_x=48,
                           chi_in=4, w_kind=w_kind, seed=seed)
    res.to("cupy", "float32")
    idx = _sample_idx(np.random.default_rng(0), res.N, 8192)
    t0 = time.time()
    X = res.reset().run(U, readout_idx=idx).astype(np.float64)
    ms = (time.time() - t0) / T * 1e3
    mc, a_star, prof = mc_validated(X, U, T, WASHOUT, DELAYS)
    log(OUT, {"part": "d15-seeds", "n_dims": 15, "w_kind": w_kind,
              "beta": 0.99, "chi_x": 48, "K_readout": 8192, "seed": seed,
              "T": T, "runner": "gpu-fp32",
              "mc300_test": round(mc, 2), "alpha_star": a_star,
              "delta_bar": round(float(res.deltas_[WASHOUT:].mean()), 5),
              "ms_per_step": round(ms, 1),
              "r2_profile": np.round(prof, 4)})


for w_kind in ("random-tt", "kron-sum"):
    for seed in (1, 2):
        run_config(w_kind, seed)
print("017c done.")
