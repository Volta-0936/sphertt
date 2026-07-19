"""Free checks on saved X: (a) is the d8 K=2048 drop a rows~features
ridge artifact (stronger alpha should recover it)? (b) alpha sensitivity
at d15."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\014\results014.jsonl"

for n_dims in (8, 15):
    data = np.load(rf"D:\sphertt-0.1.0\prototype\014\X_d{n_dims}_kron.npz")
    X, u = data["X"].astype(np.float64), data["u"]
    for K in (1024, 2048):
        cols = np.random.default_rng(0).choice(2048, K, replace=False)
        for alpha in (1e-6, 1e-4, 1e-2, 1.0):
            mc = memory_capacity(X[:, cols], u, washout=300, delays=300,
                                 alpha=alpha)
            rec = {"part": "alpha-check", "w_kind": "kron-sum",
                   "n_dims": n_dims, "K_readout": K, "alpha": alpha,
                   "mc300": round(mc, 2)}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
print("done.")
