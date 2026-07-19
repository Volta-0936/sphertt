"""The definitive beyond-the-wall MC measurement.

d15, kron-sum, beta=0.99, chi_x=48, T=16000, K_readout=8192, and a
leak-free protocol: alpha chosen on a validation slice (60/20/20 split),
MC reported on the held-out test slice.  Nested readout subsets give the
MC(K) curve with adequate rows everywhere.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\014\results014.jsonl"
T, WASHOUT, DELAYS = 16000, 300, 300
u = np.random.default_rng(7).uniform(0, 0.5, T)


def mc_validated(X, u, alphas=(1e-6, 1e-4, 1e-2)):
    """Linear MC with alpha picked on a validation slice (no test leak)."""
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
        r2s = []
        for k in range(1, DELAYS + 1):
            w = P @ ATy[k]
            c = np.corrcoef(A_va @ w, u[n_tr - k:n_va - k])[0, 1]
            r2s.append(max(c, 0.0) ** 2 if np.isfinite(c) else 0.0)
        val = float(np.sum(r2s))
        if best is None or val > best[0]:
            best = (val, a, P)
    _, a_star, P = best
    prof = []
    for k in range(1, DELAYS + 1):
        w = P @ ATy[k]
        c = np.corrcoef(A_te @ w, u[n_va - k:T - k])[0, 1]
        prof.append(max(c, 0.0) ** 2 if np.isfinite(c) else 0.0)
    return float(np.sum(prof)), a_star, prof


res = TTStateReservoir(n_dims=15, beta=0.99, chi_w=8, chi_x=48, chi_in=4,
                       w_kind="kron-sum", seed=0)
res.to("cupy", "float32")
idx = _sample_idx(np.random.default_rng(0), res.N, 8192)
t0 = time.time()
X = res.reset().run(u, readout_idx=idx).astype(np.float64)
ms = (time.time() - t0) / T * 1e3
np.savez_compressed(
    r"D:\sphertt-0.1.0\prototype\014\X_d15_kron_T16k_K8192.npz",
    X=X.astype(np.float32), u=u, idx=idx)

for K in (2048, 4096, 8192):
    cols = np.random.default_rng(0).choice(8192, K, replace=False)
    mc, a_star, prof = mc_validated(X[:, cols], u)
    rec = {"part": "definitive", "w_kind": "kron-sum", "n_dims": 15,
           "N": res.N, "beta": 0.99, "chi_x": 48, "T": T, "K_readout": K,
           "alpha_star": a_star, "mc300_test": round(mc, 2),
           "ms_per_step": round(ms, 1),
           "delta_bar": round(float(res.deltas_[300:].mean()), 5),
           "r2_profile_head": [round(p, 3) for p in prof[:10]]}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("done.")
