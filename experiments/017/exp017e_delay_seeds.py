"""017e — seed replication of Paper I's delay-task table (CPU, ~30 min).

Paper I Table 1 (delay-k reconstruction vs N) is seed 0 only.  Replicate
seeds 1/2 with the exact exp009_tasks2 protocol (T=4000, washout 300,
linear ridge readout, 60/40 split, 2048 sampled units).
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import RidgeReadout, SphereTTReservoir        # noqa: E402
from sphertt.model import _sample_idx                       # noqa: E402
from sphertt.tasks import nrmse                             # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\017\results017e.jsonl"
T = 4000
WASHOUT = 300
u = np.random.default_rng(7).uniform(0, 0.5, T)

TARGETS = {}
for k in (100, 200, 300, 400):
    y = np.zeros(T)
    y[k:] = u[:-k]
    TARGETS[f"delay{k}"] = (y, max(WASHOUT, k))


def evaluate(X, **meta):
    n_tr = int(T * 0.6)
    for name, (y, start) in TARGETS.items():
        ro = RidgeReadout(quadratic=False)
        ro.fit(X[start:n_tr], y[start:n_tr])
        pred = ro.predict(X[n_tr:])
        c = np.corrcoef(pred, y[n_tr:])[0, 1]
        rec = {"task": name, "nrmse": float(nrmse(y[n_tr:], pred)),
               "r2": float(max(c, 0) ** 2 if np.isfinite(c) else 0.0),
               **meta}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


for seed in (1, 2):
    for n_dims in (5, 7, 9):
        res = SphereTTReservoir(n_dims=n_dims, seed=seed)
        idx = (_sample_idx(np.random.default_rng(0), res.N, 2048)
               if res.N > 2048 else None)
        X = res.reset().run(u, readout_idx=idx)
        evaluate(X, tag="dense-state", n_dims=n_dims, N=res.N, beta=0.8,
                 seed=seed)
print("017e done.")
