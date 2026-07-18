"""Experiment 008 — seed robustness for the exp 005-007 results.

Everything so far was seed 0.  Re-measure with seeds 1, 2:
  A: decomposition grid  (beta x chi_x, delta_bar)      [cheap, first]
  B: two-resource crossover (beta=0.95, chi sweep, MC)  [medium]
  C: headline (beta=0.99, chi_x=48: d=8 and d=15, MC)   [expensive, last]
Results -> results008.jsonl
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\008\results008.jsonl"
u_short = np.random.default_rng(7).uniform(0, 0.5, 300)
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)
SEEDS = (1, 2)


def log(rec):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


print("=== A: decomposition grid, seeds 1-2 ===")
for seed in SEEDS:
    for chi_x in (8, 16, 32):
        for beta in (0.8, 0.9, 0.95, 0.98, 1.0):
            res = TTStateReservoir(n_dims=8, beta=beta, chi_x=chi_x,
                                   chi_in=4, seed=seed)
            res.run(u_short, readout_idx=np.arange(8))
            log({"part": "A", "seed": seed, "beta": beta, "chi_x": chi_x,
                 "delta_bar": float(res.deltas_[100:].mean())})

print("=== B: crossover MC, beta=0.95, seeds 1-2 ===")
for seed in SEEDS:
    for chi_x in (8, 16, 24, 32, 48):
        res = TTStateReservoir(n_dims=8, beta=0.95, chi_x=chi_x, chi_in=4,
                               seed=seed)
        idx = _sample_idx(np.random.default_rng(seed), res.N, 2048)
        t0 = time.time()
        X = res.reset().run(u_mc, readout_idx=idx)
        mc = memory_capacity(X, u_mc, washout=300, delays=300)
        log({"part": "B", "seed": seed, "beta": 0.95, "chi_x": chi_x,
             "mc300": mc, "delta_bar": float(res.deltas_[300:].mean()),
             "ms_per_step": (time.time() - t0) / 4000 * 1e3})

print("=== C: headline, seeds 1-2 ===")
for seed in SEEDS:
    for n_dims in (8, 15):
        res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_x=48,
                               chi_in=4, seed=seed)
        idx = _sample_idx(np.random.default_rng(seed), res.N, 2048)
        t0 = time.time()
        X = res.reset().run(u_mc, readout_idx=idx)
        mc = memory_capacity(X, u_mc, washout=300, delays=300)
        log({"part": "C", "seed": seed, "n_dims": n_dims, "N": res.N,
             "beta": 0.99, "chi_x": 48, "mc300": mc,
             "delta_bar": float(res.deltas_[300:].mean()),
             "ms_per_step": (time.time() - t0) / 4000 * 1e3})
print("done.")
