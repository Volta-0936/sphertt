"""Experiment 007 — the decisive test of the two-resource theory.

A: lift the noise limit at N=65536, chi_x=48: beta 0.98 / 0.99 (MC + delta)
B: delta_bar vs n_dims at the winning beta (flow accumulation across bonds)
C: headline — MC at n_dims 12 and 15 (N = 16.7M / 1.07e9; dense state
   134 MB / 8.6 GB) with the winning config.
Predictions were logged in REPORT.md BEFORE this run.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\007\results007.jsonl"
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)


def log(rec):
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in rec.items()}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "mc_profile"}),
          flush=True)


def run_mc(n_dims, beta, chi_x, part):
    res = TTStateReservoir(n_dims=n_dims, beta=beta, chi_x=chi_x, chi_in=4,
                           seed=0)
    idx = _sample_idx(np.random.default_rng(0), res.N, min(res.N, 2048))
    t0 = time.time()
    X = res.reset().run(u_mc, readout_idx=idx)
    mc, prof = memory_capacity(X, u_mc, washout=300, delays=300,
                               return_profile=True)
    log({"part": part, "n_dims": n_dims, "N": res.N, "beta": beta,
         "chi_x": chi_x, "mc300": mc,
         "delta_bar": float(res.deltas_[300:].mean()),
         "ms_per_step": (time.time() - t0) / 4000 * 1e3,
         "mc_profile": np.asarray(prof), **res.memory_ledger()})
    return mc


# ---------------------------------------------------------------- A
print("=== A: lifting the noise limit (N=65536, chi_x=48) ===")
mcs = {}
for beta in (0.98, 0.99):
    mcs[beta] = run_mc(8, beta, 48, "A")
beta_win = max(mcs, key=mcs.get)
print(f"winning beta: {beta_win}")

# ---------------------------------------------------------------- B
print("=== B: delta_bar vs n_dims at the winning config ===")
for n_dims in (10, 12, 15):
    res = TTStateReservoir(n_dims=n_dims, beta=beta_win, chi_x=48, chi_in=4,
                           seed=0)
    t0 = time.time()
    res.run(u_mc[:300], readout_idx=np.arange(8))
    log({"part": "B", "n_dims": n_dims, "N": res.N, "beta": beta_win,
         "chi_x": 48, "delta_bar": float(res.deltas_[100:].mean()),
         "ms_per_step": (time.time() - t0) / 300 * 1e3})

# ---------------------------------------------------------------- C
print("=== C: headline — memory beyond the state wall ===")
for n_dims in (12, 15):
    run_mc(n_dims, beta_win, 48, "C")

print("done.")
