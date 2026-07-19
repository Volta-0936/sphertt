"""CPU lane: A) stock term vs depth (beta=1, fast lane), C) bond spectra
of steady states (kron vs random, d8 vs d15)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.tt import get_array_module                   # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\013\results013.jsonl"
u = np.random.default_rng(7).uniform(0, 0.5, 300)


def log(rec):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: (v if not isinstance(v, list) else f"[{len(v)}]")
                      for k, v in rec.items()}), flush=True)


def bond_spectra(cores):
    """Schmidt spectrum at every bond of a TT vector (no truncation),
    via RL orthogonalization + LR SVD sweep.  Never densifies."""
    xp = get_array_module(cores[0])
    cs = [c.copy() for c in cores]
    d = len(cs)
    for k in range(d - 1, 0, -1):
        r0, n, r1 = cs[k].shape
        Q, R = xp.linalg.qr(cs[k].reshape(r0, n * r1).T)
        cs[k] = Q.T.reshape(-1, n, r1)
        cs[k - 1] = xp.tensordot(cs[k - 1], R.T, axes=([2], [0]))
    spectra = []
    for k in range(d - 1):
        r0, n, r1 = cs[k].shape
        U, S, Vt = xp.linalg.svd(cs[k].reshape(r0 * n, r1),
                                 full_matrices=False)
        Sn = S / xp.linalg.norm(S)
        spectra.append([float(x) for x in Sn])
        r = S.shape[0]
        cs[k] = U[:, :r].reshape(r0, n, r)
        cs[k + 1] = xp.tensordot(S[:, None] * Vt, cs[k + 1],
                                 axes=([1], [0]))
    return spectra


print("=== A: stock term vs depth (beta=1) ===")
for chi_x in (16, 48):
    for n_dims in (8, 10, 12, 15):
        res = TTStateReservoir(n_dims=n_dims, beta=1.0, chi_x=chi_x,
                               chi_in=4, seed=0)
        t0 = time.time()
        res.run(u, readout_idx=np.arange(8))
        log({"part": "A", "n_dims": n_dims, "chi_x": chi_x, "chi_in": 4,
             "beta": 1.0, "delta_stock": round(
                 float(res.deltas_[100:].mean()), 5),
             "ms_per_step": round((time.time() - t0) / 300 * 1e3, 1)})
print("=== A2: chi_in dependence at d15 ===")
for chi_in in (1, 4, 16):
    res = TTStateReservoir(n_dims=15, beta=1.0, chi_x=48, chi_in=chi_in,
                           seed=0)
    res.run(u, readout_idx=np.arange(8))
    log({"part": "A2", "n_dims": 15, "chi_x": 48, "chi_in": chi_in,
         "beta": 1.0,
         "delta_stock": round(float(res.deltas_[100:].mean()), 5)})

print("=== C: steady-state bond spectra (beta=0.99, chi_x=48) ===")
for n_dims in (8, 15):
    for w_kind in ("random-tt", "kron-sum"):
        res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_w=8,
                               chi_x=48, chi_in=4, w_kind=w_kind, seed=0)
        res.run(u, readout_idx=np.arange(8))
        spec = bond_spectra(res.x)
        mid = spec[len(spec) // 2]
        log({"part": "C", "n_dims": n_dims, "w_kind": w_kind,
             "beta": 0.99, "chi_x": 48,
             "delta_bar": round(float(res.deltas_[100:].mean()), 5),
             "mid_bond_spectrum": mid,
             "tail_mass_mid": round(float(np.sqrt(
                 np.sum(np.asarray(mid[24:]) ** 2))), 5)})
print("cpu lane done.")
