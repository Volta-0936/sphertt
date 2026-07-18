"""Direct check of the chi^2 mechanism: the state trajectory of a rank-chi
TT reservoir should span a linear subspace of dimension ~ chi^2 (if the
Schmidt frames are quasi-stationary).

Measure: singular spectrum of the (T x 2048) readout trajectory ->
effective rank (participation ratio and 99%-variance count) vs chi_x^2.
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\007\results007b.jsonl"
u = np.random.default_rng(7).uniform(0, 0.5, 4000)
idx = _sample_idx(np.random.default_rng(0), 65536, 2048)

for chi_x in (8, 16, 24):
    res = TTStateReservoir(n_dims=8, beta=0.95, chi_x=chi_x, chi_in=4,
                           seed=0)
    X = res.reset().run(u, readout_idx=idx)[300:]
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    part_ratio = float(1.0 / np.sum(p ** 2))
    cum = np.cumsum(p)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    rec = {"chi_x": chi_x, "chi2": chi_x ** 2,
           "participation_ratio": part_ratio, "rank99": r99,
           "rank999": int(np.searchsorted(cum, 0.999) + 1)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
