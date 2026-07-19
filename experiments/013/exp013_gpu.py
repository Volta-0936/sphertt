"""GPU lane: B) delta_bar fills (kron d10/d12), D) kron MC at d10/d12
for the stock-limited-curve test (P3)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\013\results013.jsonl"
u_short = np.random.default_rng(7).uniform(0, 0.5, 300)
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)


def log(rec):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


print("=== B: delta_bar fills (beta=0.99, chi_x=48) ===")
for n_dims in (10, 12):
    for w_kind in ("random-tt", "kron-sum"):
        res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_w=8,
                               chi_x=48, chi_in=4, w_kind=w_kind, seed=0)
        res.to("cupy", "float32")
        res.run(u_short, readout_idx=np.arange(8))
        log({"part": "B", "n_dims": n_dims, "w_kind": w_kind,
             "beta": 0.99, "chi_x": 48,
             "delta_bar": round(float(res.deltas_[100:].mean()), 5)})

print("=== D: kron MC at d10 / d12 (beta=0.99, chi_x=48) ===")
for n_dims in (10, 12):
    res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_w=8, chi_x=48,
                           chi_in=4, w_kind="kron-sum", seed=0)
    res.to("cupy", "float32")
    idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc = memory_capacity(X.astype(np.float64), u_mc, washout=300,
                         delays=300)
    log({"part": "D", "n_dims": n_dims, "w_kind": "kron-sum",
         "beta": 0.99, "chi_x": 48, "mc300": round(mc, 2),
         "delta_bar": round(float(res.deltas_[300:].mean()), 5),
         "ms_per_step": round((time.time() - t0) / 4000 * 1e3, 1)})
print("gpu lane done.")
