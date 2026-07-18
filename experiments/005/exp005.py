"""Experiment 005: native TT-state evolution (sphertt v0.3.0, TTStateReservoir).

H2: NARMA10 vs chi_x at N=1024 (native TT-state vs dense-state twin).
H3: scaling n_dims 5..15 (N up to 4^15 > 10^9, dense state 8.6 GB —
    infeasible): timing, memory ledger, NARMA10, MC(window 300), deltas.

Results -> results005.jsonl (one JSON object per line).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import ESN, SphereTTReservoir, TTStateReservoir  # noqa: E402
from sphertt.model import _sample_idx                          # noqa: E402
from sphertt.tasks import memory_capacity, narma               # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\005\results005.jsonl"


def log(rec):
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in rec.items()}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    brief = {k: v for k, v in rec.items() if k != "mc_profile"}
    print(json.dumps(brief), flush=True)


# ---------------------------------------------------------------- H2
print("=== H2: NARMA10 vs chi_x at N=1024 (n_dims=5, [4]^5, max rank 16) ===")
u, y = narma(2500, order=10, rng=np.random.default_rng(0))
for seed in (0, 1):
    esn = ESN(reservoir=SphereTTReservoir(n_dims=5, seed=seed), washout=200)
    esn.fit(u[:1500], y[:1500])
    log({"exp": "H2", "engine": "dense-state", "n_dims": 5, "seed": seed,
         "narma10": esn.score(u[1500:], y[1500:])})
for chi_x in (4, 8, 12, 16):
    for seed in (0, 1):
        res = TTStateReservoir(n_dims=5, chi_x=chi_x, chi_in=4, seed=seed)
        esn = ESN(reservoir=res, washout=200)
        t0 = time.time()
        esn.fit(u[:1500], y[:1500])
        sc = esn.score(u[1500:], y[1500:])
        log({"exp": "H2", "engine": "tt-state", "n_dims": 5, "chi_x": chi_x,
             "seed": seed, "narma10": sc,
             "ms_per_step": (time.time() - t0) / 2500 * 1e3,
             "delta_mean": float(res.deltas_[200:].mean()),
             "delta_max": float(res.deltas_.max())})

# ---------------------------------------------------------------- H3
print("=== H3: scaling in n_dims at chi_x=16 ===")
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)
for n_dims in (5, 8, 10, 12, 15):
    res = TTStateReservoir(n_dims=n_dims, chi_x=16, chi_in=4, seed=0)
    rng = np.random.default_rng(0)
    k = min(res.N, 2048)
    idx = _sample_idx(rng, res.N, k) if res.N > k else np.arange(res.N)

    # timing
    t0 = time.time()
    res.reset().run(u_mc[:100], readout_idx=idx[:8])
    ms = (time.time() - t0) / 100 * 1e3

    # NARMA10
    esn = ESN(reservoir=TTStateReservoir(n_dims=n_dims, chi_x=16, chi_in=4,
                                         seed=0), washout=200)
    esn.fit(u[:1500], y[:1500])
    sc = esn.score(u[1500:], y[1500:])

    # memory capacity, 300-delay window
    X = res.reset().run(u_mc, readout_idx=idx)
    mc, prof = memory_capacity(X, u_mc, washout=300, delays=300,
                               return_profile=True)
    log({"exp": "H3", "engine": "tt-state", "n_dims": n_dims, "N": res.N,
         "chi_x": 16, "seed": 0, "ms_per_step": ms, "narma10": sc,
         "mc300": mc, "mc_profile": np.asarray(prof),
         "delta_mean": float(res.deltas_[200:].mean()),
         "delta_max": float(res.deltas_.max()),
         **res.memory_ledger()})

print("done.")
