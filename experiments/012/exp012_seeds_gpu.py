"""Seed check (GPU lane): the key comparison at seeds 1-2, MC only."""
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
CONFIGS = [("kron-sum", 1), ("kron-sum", 2)]        # (w_kind, seed)

for w_kind, seed in CONFIGS:
    res = TTStateReservoir(n_dims=8, beta=0.95, chi_w=8, chi_x=32,
                           chi_in=4, w_kind=w_kind, seed=seed)
    res.to("cupy", "float32")
    idx = _sample_idx(np.random.default_rng(seed), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc = memory_capacity(X.astype(np.float64), u_mc, washout=300,
                         delays=300)
    rec = {"part": "seeds", "w_kind": w_kind, "chi_w": 8, "seed": seed,
           "runner": "gpu", "beta": 0.95, "chi_x": 32,
           "mc300": round(mc, 2),
           "delta_bar": round(float(res.deltas_[300:].mean()), 5),
           "ms_per_step": round((time.time() - t0) / 4000 * 1e3, 1)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
