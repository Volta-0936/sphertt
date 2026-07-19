"""Readout-subset analysis on the SAVED d15 trajectory (exp009,
random-tt, beta=0.99, chi_x=48, T=4000, K=2048) - no new reservoir run."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\014\results014.jsonl"
data = np.load(r"D:\sphertt-0.1.0\prototype\009\X_d15_guided.npz")
X, u = data["X"], data["u"]

for K in (256, 512, 1024, 2048):
    cols = np.random.default_rng(0).choice(X.shape[1], K, replace=False)
    mc = memory_capacity(X[:, cols], u, washout=300, delays=300)
    rec = {"part": "subsets", "source": "exp009 X_d15_guided",
           "w_kind": "random-tt", "n_dims": 15, "K_readout": K,
           "mc300": round(mc, 2)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
