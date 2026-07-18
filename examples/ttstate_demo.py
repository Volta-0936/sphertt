"""A reservoir with N = 4**15 > 10^9 units on a laptop.

The dense state alone would be 8.6 GB and the dense W ~ 9 exabytes; in TT
format the whole model (W + state + input coupling) is under 1 MB.
"""
import time

import numpy as np

from sphertt import TTStateReservoir
from sphertt.model import _sample_idx

res = TTStateReservoir(n_dims=15, chi_x=16, seed=0)   # N = 1,073,741,824
print(res)
print(res.memory_ledger())

idx = _sample_idx(np.random.default_rng(0), res.N, 2048)
u = np.random.default_rng(0).uniform(0, 0.5, 200)
t0 = time.time()
X = res.run(u, readout_idx=idx)
print(f"{(time.time() - t0) / len(u) * 1e3:.1f} ms/step, "
      f"state ranks {res.state_ranks}, "
      f"mean rounding error {res.deltas_[50:].mean():.4f}")
