"""Free analysis: effective rank (participation ratio) of readout columns
at d8 vs d15 from saved trajectories - the entry-redundancy mechanism."""
import json

import numpy as np

OUT = r"D:\sphertt-0.1.0\prototype\015\results015.jsonl"
FILES = [
    ("d8_kron_T4k", r"D:\sphertt-0.1.0\prototype\014\X_d8_kron.npz"),
    ("d15_kron_T4k", r"D:\sphertt-0.1.0\prototype\014\X_d15_kron.npz"),
    ("d15_kron_T16k_K8192",
     r"D:\sphertt-0.1.0\prototype\014\X_d15_kron_T16k_K8192.npz"),
    ("d15_random_T4k",
     r"D:\sphertt-0.1.0\prototype\009\X_d15_guided.npz"),
]
for tag, path in FILES:
    X = np.load(path)["X"].astype(np.float64)[300:]
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    pr = float(1.0 / np.sum(p ** 2))
    cum = np.cumsum(p)
    rec = {"part": "effrank", "source": tag, "shape": list(X.shape),
           "participation_ratio": round(pr, 1),
           "rank95": int(np.searchsorted(cum, 0.95) + 1),
           "rank99": int(np.searchsorted(cum, 0.99) + 1)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
