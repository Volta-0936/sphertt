"""Experiment 011 - GPU/precision benchmark and fp32 physics validation.

A: ms/step for the heavy configs across (backend, dtype).
B: physics check - MC at the reference config (N=65536, beta=0.95,
   chi_x=32) on gpu-fp32 vs the known cpu-fp64 value 64.8 (seed 0).
Results -> results011.jsonl
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                      # noqa: E402
from sphertt.model import _sample_idx                     # noqa: E402
from sphertt.tasks import memory_capacity                 # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\011\results011.jsonl"


def log(rec):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


u = np.random.default_rng(7).uniform(0, 0.5, 4000)

print("=== A: step-time benchmark ===")
CONFIGS = [(8, 48), (15, 48), (15, 16)]
BACKENDS = [("numpy", "float64"), ("cupy", "float64"), ("cupy", "float32")]
for n_dims, chi_x in CONFIGS:
    for backend, dtype in BACKENDS:
        res = TTStateReservoir(n_dims=n_dims, beta=0.99, chi_x=chi_x,
                               chi_in=4, seed=0).to(backend, dtype)
        res.run(u[:5], readout_idx=np.arange(8))          # warmup
        T = 60 if backend == "numpy" else 200
        t0 = time.time()
        res.reset().run(u[:T], readout_idx=np.arange(8))
        ms = (time.time() - t0) / T * 1e3
        log({"part": "A", "n_dims": n_dims, "chi_x": chi_x,
             "backend": backend, "dtype": dtype,
             "ms_per_step": round(ms, 2),
             "delta_bar": float(res.deltas_[30:].mean())})

print("=== B: fp32 physics validation (MC at reference config) ===")
for backend, dtype in [("cupy", "float32"), ("cupy", "float64")]:
    res = TTStateReservoir(n_dims=8, beta=0.95, chi_x=32, chi_in=4,
                           seed=0).to(backend, dtype)
    idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
    t0 = time.time()
    X = res.reset().run(u, readout_idx=idx)
    mc = memory_capacity(X.astype(np.float64), u, washout=300, delays=300)
    log({"part": "B", "backend": backend, "dtype": dtype,
         "mc300": mc, "delta_bar": float(res.deltas_[300:].mean()),
         "ms_per_step": round((time.time() - t0) / 4000 * 1e3, 2),
         "reference_cpu_fp64_mc": 64.8})
print("done.")
