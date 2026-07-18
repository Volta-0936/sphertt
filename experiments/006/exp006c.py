"""Part C — theory-driven verification: MC at configs chosen AFTER fitting
the delta decomposition (predictions logged in REPORT before running this).

Configs: chi_x=32 at beta 0.9 / 0.95 (does lower delta_bar buy memory?),
plus the stock-only reservoir (beta=1) in TT and dense form.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import SphereTTReservoir, TTStateReservoir  # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\006\results006c.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)
idx = _sample_idx(np.random.default_rng(0), 65536, 2048)


def log(rec):
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in rec.items()}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "mc_profile"}),
          flush=True)


def mc_of(res, tag, **meta):
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc, prof = memory_capacity(X, u_mc, washout=300, delays=300,
                               return_profile=True)
    rec = {"part": "C", "engine": tag, "mc300": mc,
           "ms_per_step": (time.time() - t0) / 4000 * 1e3,
           "mc_profile": np.asarray(prof), **meta}
    if hasattr(res, "deltas_"):
        rec["delta_bar"] = float(res.deltas_[300:].mean())
    log(rec)


for beta in (0.9, 0.95):
    res = TTStateReservoir(n_dims=8, beta=beta, chi_x=32, chi_in=4, seed=0)
    mc_of(res, "tt-state", beta=beta, chi_x=32)

res = TTStateReservoir(n_dims=8, beta=1.0, chi_x=16, chi_in=4, seed=0)
mc_of(res, "tt-state", beta=1.0, chi_x=16)

res = SphereTTReservoir(n_dims=8, beta=1.0, seed=0)
mc_of(res, "dense-state", beta=1.0)

print("done.")
