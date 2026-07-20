"""Experiment 017 — shared protocol.

Same leak-free MC protocol as 014/015 (alpha on a validation slice,
MC on the held-out test slice, 60/20/20), generalized:

* T, WASHOUT, DELAYS are parameters (017 extends the window to 1200 —
  the 300-delay window censored every d8 measurement in 013-016).
* The alpha grid gains both endpoints (1e-8 ... 1.0) because 015's d15
  optimum sat on the old grid boundary (1e-6).
* The per-delay ridge solves are vectorized (DELAYS=1200 x K=8192 is
  too slow as a python loop).

The vectorized implementation is verified against the 014 loop
implementation by `test_mc_equivalence()` (run on import in exp017a).
"""
import json
import time

import numpy as np

ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)


def _col_corr(P, Y):
    """Column-wise Pearson r between P[:, j] and Y[:, j]."""
    Pc = P - P.mean(0)
    Yc = Y - Y.mean(0)
    num = (Pc * Yc).sum(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / den
    return np.where(np.isfinite(c), c, 0.0)


def _targets(u, lo, hi, delays):
    """Matrix of delayed inputs: column k-1 is u[lo-k:hi-k]."""
    return np.stack([u[lo - k:hi - k] for k in range(1, delays + 1)], axis=1)


def mc_validated(X, u, T, washout=300, delays=300, alphas=ALPHAS):
    """Leak-free linear MC (validated alpha), vectorized.

    Returns (mc, alpha_star, profile) — profile is the per-delay r^2 on
    the test slice."""
    n_tr, n_va = int(T * 0.6), int(T * 0.8)
    start = max(int(washout), int(delays))
    A_tr = np.hstack([X[start:n_tr], np.ones((n_tr - start, 1))])
    A_va = np.hstack([X[n_tr:n_va], np.ones((n_va - n_tr, 1))])
    A_te = np.hstack([X[n_va:T], np.ones((T - n_va, 1))])
    G = A_tr.T @ A_tr
    Y_tr = _targets(u, start, n_tr, delays)
    Y_va = _targets(u, n_tr, n_va, delays)
    Y_te = _targets(u, n_va, T, delays)
    ATY = A_tr.T @ Y_tr                      # (features+1, delays)
    best = None
    for a in alphas:
        P = np.linalg.inv(G + a * np.eye(G.shape[0]))
        W = P @ ATY
        c = _col_corr(A_va @ W, Y_va)
        val = float(np.sum(np.maximum(c, 0.0) ** 2))
        if best is None or val > best[0]:
            best = (val, a, W)
    _, a_star, W = best
    c = _col_corr(A_te @ W, Y_te)
    prof = np.maximum(c, 0.0) ** 2
    return float(np.sum(prof)), a_star, prof


def mc_validated_loop_014(X, u, T, washout=300, delays=300,
                          alphas=(1e-6, 1e-4, 1e-2)):
    """Verbatim port of exp014_definitive.mc_validated (loop version),
    kept only to verify the vectorized implementation."""
    n_tr, n_va = int(T * 0.6), int(T * 0.8)
    start = max(washout, delays)
    A_tr = np.hstack([X[start:n_tr], np.ones((n_tr - start, 1))])
    A_va = np.hstack([X[n_tr:n_va], np.ones((n_va - n_tr, 1))])
    A_te = np.hstack([X[n_va:T], np.ones((T - n_va, 1))])
    G = A_tr.T @ A_tr
    ATy = {k: A_tr.T @ u[start - k:n_tr - k] for k in range(1, delays + 1)}
    best = None
    for a in alphas:
        P = np.linalg.inv(G + a * np.eye(G.shape[0]))
        r2s = []
        for k in range(1, delays + 1):
            w = P @ ATy[k]
            c = np.corrcoef(A_va @ w, u[n_tr - k:n_va - k])[0, 1]
            r2s.append(max(c, 0.0) ** 2 if np.isfinite(c) else 0.0)
        val = float(np.sum(r2s))
        if best is None or val > best[0]:
            best = (val, a, P)
    _, a_star, P = best
    prof = []
    for k in range(1, delays + 1):
        w = P @ ATy[k]
        c = np.corrcoef(A_te @ w, u[n_va - k:T - k])[0, 1]
        prof.append(max(c, 0.0) ** 2 if np.isfinite(c) else 0.0)
    return float(np.sum(prof)), a_star, prof


def test_mc_equivalence():
    rng = np.random.default_rng(42)
    T = 900
    u = rng.uniform(0, 0.5, T)
    X = rng.standard_normal((T, 40))
    for k in range(1, 30):
        X[k:, k] += 3.0 * u[:T - k]          # planted memory
    a = mc_validated(X, u, T, washout=60, delays=60, alphas=(1e-6, 1e-4, 1e-2))
    b = mc_validated_loop_014(X, u, T, washout=60, delays=60)
    assert abs(a[0] - b[0]) < 1e-8, (a[0], b[0])
    assert a[1] == b[1]
    return a[0]


def log(path, rec):
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in rec.items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k != "r2_profile"}), flush=True)


def timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0
