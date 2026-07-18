"""Part C rerun (ASCII prints only): MC at n_dims 12 / 15, beta=0.99,
chi_x=48 - memory beyond the state wall."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\007\results007.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)

for n_dims in (12, 15):
    res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_x=48, chi_in=4,
                           seed=0)
    idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc, prof = memory_capacity(X, u_mc, washout=300, delays=300,
                               return_profile=True)
    rec = {"part": "C", "n_dims": n_dims, "N": res.N, "beta": 0.99,
           "chi_x": 48, "mc300": mc,
           "delta_bar": float(res.deltas_[300:].mean()),
           "ms_per_step": (time.time() - t0) / 4000 * 1e3,
           "mc_profile": list(prof)}
    rec.update(res.memory_ledger())
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "mc_profile"}),
          flush=True)
print("done.")
