"""C2 — testing MC ~= K(beta) * chi_x^2 at beta=0.95 (predictions logged
in REPORT.md before this run): chi_x in {8, 24, 48}."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\006\results006d.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)
idx = _sample_idx(np.random.default_rng(0), 65536, 2048)

for chi_x in (8, 24, 48):
    res = TTStateReservoir(n_dims=8, beta=0.95, chi_x=chi_x, chi_in=4,
                           seed=0)
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc, prof = memory_capacity(X, u_mc, washout=300, delays=300,
                               return_profile=True)
    rec = {"part": "C2", "beta": 0.95, "chi_x": chi_x, "mc300": mc,
           "delta_bar": float(res.deltas_[300:].mean()),
           "ms_per_step": (time.time() - t0) / 4000 * 1e3,
           "mc_profile": list(prof)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "mc_profile"}),
          flush=True)
print("done.")
