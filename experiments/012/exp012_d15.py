"""The decisive follow-up: kron-sum at the beyond-the-wall headline config
(n_dims=15, beta=0.99, chi_x=48).  Reference (random-tt): MC 73.7/30.4/41.4
(seeds 0/1/2)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\012\results012.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)

for seed in (0, 1):
    res = TTStateReservoir(n_dims=15, beta=0.99, chi_w=8, chi_x=48,
                           chi_in=4, w_kind="kron-sum", seed=seed)
    res.to("cupy", "float32")
    idx = _sample_idx(np.random.default_rng(seed), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc = memory_capacity(X.astype(np.float64), u_mc, washout=300,
                         delays=300)
    rec = {"part": "d15", "w_kind": "kron-sum", "seed": seed,
           "n_dims": 15, "N": res.N, "beta": 0.99, "chi_x": 48,
           "mc300": round(mc, 2),
           "delta_bar": round(float(res.deltas_[300:].mean()), 5),
           "ms_per_step": round((time.time() - t0) / 4000 * 1e3, 1),
           "reference_random_tt": [73.7, 30.4, 41.4]}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
