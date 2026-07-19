"""Shared evaluation for experiment 012 (imported by gpu/cpu runners)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import ESN, TTStateReservoir                 # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity, narma          # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\012\results012.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)
u_n, y_n = narma(2500, order=10, rng=np.random.default_rng(0))


def evaluate(w_kind, chi_w, backend, dtype, runner):
    res = TTStateReservoir(n_dims=8, beta=0.95, chi_w=chi_w, chi_x=32,
                           chi_in=4, w_kind=w_kind, seed=0)
    if backend != "numpy":
        res.to(backend, dtype)
    idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    ms = (time.time() - t0) / 4000 * 1e3
    mc = memory_capacity(X.astype(np.float64), u_mc, washout=300,
                         delays=300)
    esn = ESN(reservoir=res, washout=200)
    esn.fit(u_n[:1500], y_n[:1500])
    narma10 = esn.score(u_n[1500:], y_n[1500:])
    rec = {"w_kind": w_kind, "chi_w": chi_w, "backend": backend,
           "dtype": dtype, "runner": runner, "beta": 0.95, "chi_x": 32,
           "mc300": round(mc, 2), "narma10": round(float(narma10), 4),
           "delta_bar": round(float(res.deltas_[300:].mean()), 5),
           "ms_per_step": round(ms, 1)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
