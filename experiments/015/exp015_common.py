"""Shared: definitive-protocol MC (validated alpha, leak-free 60/20/20)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\015\results015.jsonl"
T, WASHOUT, DELAYS = 16000, 300, 300
U = np.random.default_rng(7).uniform(0, 0.5, T)


def mc_validated(X, u, alphas=(1e-6, 1e-4, 1e-2)):
    n_tr, n_va = int(T * 0.6), int(T * 0.8)
    start = max(WASHOUT, DELAYS)
    A_tr = np.hstack([X[start:n_tr], np.ones((n_tr - start, 1))])
    A_va = np.hstack([X[n_tr:n_va], np.ones((n_va - n_tr, 1))])
    A_te = np.hstack([X[n_va:], np.ones((T - n_va, 1))])
    G = A_tr.T @ A_tr
    ATy = {k: A_tr.T @ u[start - k:n_tr - k] for k in range(1, DELAYS + 1)}
    best = None
    for a in alphas:
        P = np.linalg.inv(G + a * np.eye(G.shape[0]))
        val = 0.0
        for k in range(1, DELAYS + 1):
            w = P @ ATy[k]
            c = np.corrcoef(A_va @ w, u[n_tr - k:n_va - k])[0, 1]
            val += max(c, 0.0) ** 2 if np.isfinite(c) else 0.0
        if best is None or val > best[0]:
            best = (val, a, P)
    _, a_star, P = best
    mc = 0.0
    for k in range(1, DELAYS + 1):
        w = P @ ATy[k]
        c = np.corrcoef(A_te @ w, u[n_va - k:T - k])[0, 1]
        mc += max(c, 0.0) ** 2 if np.isfinite(c) else 0.0
    return float(mc), a_star


def run_config(n_dims, w_kind, beta, K, backend, dtype, runner):
    res = TTStateReservoir(n_dims=n_dims, beta=beta, chi_w=8, chi_x=48,
                           chi_in=4, w_kind=w_kind, seed=0)
    if backend != "numpy":
        res.to(backend, dtype)
    idx = _sample_idx(np.random.default_rng(0), res.N, K)
    t0 = time.time()
    X = res.reset().run(U, readout_idx=idx).astype(np.float64)
    ms = (time.time() - t0) / T * 1e3
    mc, a_star = mc_validated(X, U)
    rec = {"part": "curve", "n_dims": n_dims, "w_kind": w_kind,
           "beta": beta, "chi_x": 48, "K_readout": K, "T": T,
           "runner": runner, "mc300_test": round(mc, 2),
           "alpha_star": a_star,
           "delta_bar": round(float(res.deltas_[300:].mean()), 5),
           "ms_per_step": round(ms, 1)}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
