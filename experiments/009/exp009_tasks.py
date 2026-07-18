"""Experiment 009 — real tasks (run AFTER the exp008 seed batch).

009a (Paper I): dense-state SphereTTReservoir, N = 1024 / 16384 / 262144.
    Long-memory tasks: delay-k reproduction (k = 50/100/200) + NARMA-20/30.
    One state trajectory per N, many readouts.
009b (Paper II): TT-state at n_dims=15 (N > 1e9): theory-guided
    (beta=0.99, chi_x=48) vs naive (beta=0.8, chi_x=16).
Results -> results009.jsonl
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import RidgeReadout, SphereTTReservoir, TTStateReservoir  # noqa: E402
from sphertt.model import _sample_idx                                   # noqa: E402
from sphertt.tasks import narma, nrmse                                  # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\009\results009.jsonl"
T = 4000
WASHOUT = 300
rng_u = np.random.default_rng(7)
u = rng_u.uniform(0, 0.5, T)


def narma_target(u, order):
    """NARMA-y driven by our fixed input u (no redraw; clip guard)."""
    y = np.zeros(len(u))
    for t in range(order - 1, len(u) - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - order + 1:t + 1])
                    + 1.5 * u[t - order + 1] * u[t] + 0.1)
        if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 100:
            return None
    return y


def targets():
    out = {}
    for k in (50, 100, 200):
        y = np.zeros(T)
        y[k:] = u[:-k]
        out[f"delay{k}"] = (y, max(WASHOUT, k))
    for order in (20, 30):
        y = narma_target(u, order)
        if y is not None:
            out[f"narma{order}"] = (y, max(WASHOUT, order))
    return out


TARGETS = targets()


def evaluate(X, tag, **meta):
    n_tr = int(T * 0.6)
    for name, (y, start) in TARGETS.items():
        ro = RidgeReadout()
        ro.fit(X[start:n_tr], y[start:n_tr])
        pred = ro.predict(X[n_tr:])
        e = nrmse(y[n_tr:], pred)
        c = np.corrcoef(pred, y[n_tr:])[0, 1]
        rec = {"tag": tag, "task": name, "nrmse": float(e),
               "r2": float(max(c, 0) ** 2 if np.isfinite(c) else 0.0),
               **meta}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


print("=== 009a: dense-state, N sweep ===")
for n_dims in (5, 7, 9):
    res = SphereTTReservoir(n_dims=n_dims, seed=0)
    idx = (_sample_idx(np.random.default_rng(0), res.N, 2048)
           if res.N > 2048 else None)
    t0 = time.time()
    X = res.reset().run(u, readout_idx=idx)
    ms = (time.time() - t0) / T * 1e3
    evaluate(X, "dense-state", n_dims=n_dims, N=res.N, beta=0.8,
             ms_per_step=ms)

print("=== 009b: TT-state at n_dims=15, guided vs naive ===")
for beta, chi_x, label in [(0.99, 48, "guided"), (0.8, 16, "naive")]:
    res = TTStateReservoir(n_dims=15, beta=beta, chi_x=chi_x, chi_in=4,
                           seed=0)
    idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u, readout_idx=idx)
    ms = (time.time() - t0) / T * 1e3
    evaluate(X, f"tt-state-{label}", n_dims=15, N=res.N, beta=beta,
             chi_x=chi_x, ms_per_step=ms,
             delta_bar=float(res.deltas_[300:].mean()))
print("done.")
