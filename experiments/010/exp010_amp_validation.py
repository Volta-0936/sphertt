"""Experiment 010 — self-contained re-validation of the amplification law
    A ~= min{(1 - g_bar^2)^(-1/2), sqrt(2)/delta_bar}
using sphertt's own diagnostics (the original 52-point validation lives in
the hsmps-rc research repo; this reproduces it inside this repo so Paper I
is self-contained).

Grid: beta x chi_x x seeds at N=1024, state tensorization [2]^10
(max rank 32), T=1500 NARMA10 input, as in experiments B/C/D.
Results -> results010.jsonl, figure -> fig5_amp_validation.png
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import SphereTTReservoir, amplification_report  # noqa: E402
from sphertt.tasks import narma                               # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\010\results010.jsonl"
u, _ = narma(1500, order=10, rng=np.random.default_rng(0))

BETAS = (0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 1.0)
CHIS = (8, 12, 16, 24)
SEEDS = (0, 1)

t0 = time.time()
for seed in SEEDS:
    for beta in BETAS:
        res = SphereTTReservoir(n_dims=10, mode_size=2, beta=beta, chi_w=8,
                                seed=seed)
        for chi_x in CHIS:
            rep = amplification_report(res, u, chi_x=chi_x)
            rec = {"beta": beta, "chi_x": chi_x, "seed": seed,
                   **{k: (None if isinstance(v, float) and not np.isfinite(v)
                          else v) for k, v in rep.items()}}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
print(f"total {time.time()-t0:.0f}s")
