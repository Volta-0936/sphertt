"""H4 probe: what controls the per-step rounding error delta at N=65536?

Sweep tensorization granularity (mode_size), state rank chi_x, random-part
rank chi_w, and orthogonal weight beta.  delta_bar is the one-shot
truncation error of the amplification law — if it cannot be pushed down,
truncated task performance cannot be rescued (A*delta ~ O(1)).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir  # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\005\results005b.jsonl"
u = np.random.default_rng(7).uniform(0, 0.5, 150)

CONFIGS = [
    # (mode_size, n_dims, chi_x, chi_w, beta)
    (4, 8, 16, 8, 0.8),      # H3 reference point
    (4, 8, 24, 8, 0.8),
    (4, 8, 32, 8, 0.8),
    (2, 16, 16, 8, 0.8),     # finer tensorization
    (2, 16, 24, 8, 0.8),
    (2, 16, 32, 8, 0.8),
    (4, 8, 16, 4, 0.8),      # smaller random rank
    (4, 8, 16, 2, 0.8),
    (4, 8, 16, 8, 0.9),      # more orthogonal
    (4, 8, 16, 8, 0.95),
    (4, 8, 16, 2, 0.95),     # combined
]

for mode, nd, chi_x, chi_w, beta in CONFIGS:
    res = TTStateReservoir(n_dims=nd, mode_size=mode, chi_x=chi_x,
                           chi_w=chi_w, beta=beta, chi_in=4, seed=0)
    t0 = time.time()
    res.run(u, readout_idx=np.arange(8))
    rec = {"mode": mode, "n_dims": nd, "chi_x": chi_x, "chi_w": chi_w,
           "beta": beta, "N": res.N,
           "delta_mean": float(res.deltas_[30:].mean()),
           "ms_per_step": (time.time() - t0) / len(u) * 1e3}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
