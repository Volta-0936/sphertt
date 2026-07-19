"""GPU lane: d8 and d15 kron-sum trajectories with X saved, then the same
readout-subset analysis on both."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\014\results014.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)

for n_dims in (8, 15):
    res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_w=8, chi_x=48,
                           chi_in=4, w_kind="kron-sum", seed=0)
    res.to("cupy", "float32")
    idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
    X = res.reset().run(u_mc, readout_idx=idx)
    np.savez_compressed(
        rf"D:\sphertt-0.1.0\prototype\014\X_d{n_dims}_kron.npz",
        X=X, u=u_mc, idx=idx)
    for K in (256, 512, 1024, 2048):
        cols = np.random.default_rng(0).choice(2048, K, replace=False)
        mc = memory_capacity(X[:, cols].astype(np.float64), u_mc,
                             washout=300, delays=300)
        rec = {"part": "subsets", "w_kind": "kron-sum", "n_dims": n_dims,
               "K_readout": K, "mc300": round(mc, 2)}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("done.")
