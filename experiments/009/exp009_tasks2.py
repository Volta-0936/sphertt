"""Experiment 009 (fixed protocol).

Fixes vs exp009_tasks.py:
  * linear readout for delay tasks (they are linear; quadratic features
    only add overfitting at low SNR), quadratic kept for NARMA
  * delays extended to 300/400 where the N-advantage should appear
  * NARMA-10 (stable) instead of NARMA-20/30 (diverged for this input)
  * the d=15 guided state trajectory is SAVED for later re-analysis
Results -> results009b.jsonl
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import RidgeReadout, SphereTTReservoir, TTStateReservoir  # noqa: E402
from sphertt.model import _sample_idx                                   # noqa: E402
from sphertt.tasks import nrmse                                         # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\009\results009b.jsonl"
T = 4000
WASHOUT = 300
u = np.random.default_rng(7).uniform(0, 0.5, T)

y_narma = np.zeros(T)
for t in range(9, T - 1):
    y_narma[t + 1] = (0.3 * y_narma[t]
                      + 0.05 * y_narma[t] * np.sum(y_narma[t - 9:t + 1])
                      + 1.5 * u[t - 9] * u[t] + 0.1)

TARGETS = {}
for k in (50, 100, 200, 300, 400):
    y = np.zeros(T)
    y[k:] = u[:-k]
    TARGETS[f"delay{k}"] = (y, max(WASHOUT, k), False)   # linear readout
TARGETS["narma10"] = (y_narma, WASHOUT, True)            # quadratic readout


def evaluate(X, tag, **meta):
    n_tr = int(T * 0.6)
    for name, (y, start, quad) in TARGETS.items():
        ro = RidgeReadout(quadratic=quad)
        ro.fit(X[start:n_tr], y[start:n_tr])
        pred = ro.predict(X[n_tr:])
        c = np.corrcoef(pred, y[n_tr:])[0, 1]
        rec = {"tag": tag, "task": name,
               "nrmse": float(nrmse(y[n_tr:], pred)),
               "r2": float(max(c, 0) ** 2 if np.isfinite(c) else 0.0),
               **meta}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


print("=== 009a2: dense-state, N sweep ===")
for n_dims in (5, 7, 9):
    res = SphereTTReservoir(n_dims=n_dims, seed=0)
    idx = (_sample_idx(np.random.default_rng(0), res.N, 2048)
           if res.N > 2048 else None)
    X = res.reset().run(u, readout_idx=idx)
    evaluate(X, "dense-state", n_dims=n_dims, N=res.N, beta=0.8)

print("=== 009b2: TT-state guided at n_dims=15 ===")
res = TTStateReservoir(n_dims=15, beta=0.99, chi_x=48, chi_in=4, seed=0)
idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
t0 = time.time()
X = res.reset().run(u, readout_idx=idx)
np.savez_compressed(r"D:\sphertt-0.1.0\prototype\009\X_d15_guided.npz",
                    X=X, u=u, idx=idx)
evaluate(X, "tt-state-guided", n_dims=15, N=res.N, beta=0.99, chi_x=48,
         ms_per_step=(time.time() - t0) / T * 1e3,
         delta_bar=float(res.deltas_[300:].mean()))
print("done.")
