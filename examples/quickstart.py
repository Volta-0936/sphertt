"""Minimal end-to-end example: NARMA10 at N = 65,536 (dense W would be 32 GB)."""
import numpy as np

from sphertt import ESN
from sphertt.tasks import narma

u, y = narma(4000, order=10, rng=np.random.default_rng(0))

esn = ESN(n_dims=8, seed=0)               # N = 4**8 = 65,536
esn.fit(u[:2400], y[:2400])
print("test NRMSE :", round(esn.score(u[2400:], y[2400:]), 4))
print("ledger     :", esn.reservoir.memory_ledger())
