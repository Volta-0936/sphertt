"""017e completion — the single config missing when the batch was stopped
(seed 2, n_dims=9).  Same protocol as exp017e_delay_seeds.py."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import RidgeReadout, SphereTTReservoir        # noqa: E402
from sphertt.model import _sample_idx                       # noqa: E402
from sphertt.tasks import nrmse                             # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\017\results017e.jsonl"
T, WASHOUT = 4000, 300
u = np.random.default_rng(7).uniform(0, 0.5, T)

TARGETS = {}
for k in (100, 200, 300, 400):
    y = np.zeros(T)
    y[k:] = u[:-k]
    TARGETS[f"delay{k}"] = (y, max(WASHOUT, k))

res = SphereTTReservoir(n_dims=9, seed=2)
idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
X = res.reset().run(u, readout_idx=idx)
n_tr = int(T * 0.6)
for name, (y, start) in TARGETS.items():
    ro = RidgeReadout(quadratic=False)
    ro.fit(X[start:n_tr], y[start:n_tr])
    pred = ro.predict(X[n_tr:])
    c = np.corrcoef(pred, y[n_tr:])[0, 1]
    rec = {"task": name, "nrmse": float(nrmse(y[n_tr:], pred)),
           "r2": float(max(c, 0) ** 2 if np.isfinite(c) else 0.0),
           "tag": "dense-state", "n_dims": 9, "N": res.N, "beta": 0.8,
           "seed": 2}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("017e fill done.")
