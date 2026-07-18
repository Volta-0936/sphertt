"""sphertt — hyperspherical reservoir computing with tensor-train connectivity.

Core idea: the sphere normalization x -> z/||z|| is the (only practical)
nonlinearity that never increases tensor-train rank, and a quasi-orthogonal
TT connectivity  W = beta*Q_kron + (1-beta)*W_tt  (rank <= 1 + chi_w)
scales reservoir computing to state dimensions where a dense W cannot even
be allocated (512 GB -> 73 KB at N = 4**9), with memory capacity that
deepens with N.

Quickstart::

    import numpy as np
    from sphertt import ESN
    from sphertt.tasks import narma, nrmse

    u, y = narma(4000, order=10, rng=np.random.default_rng(0))
    esn = ESN(n_dims=8, seed=0)          # N = 4**8 = 65,536
    esn.fit(u[:2400], y[:2400])
    print("NRMSE:", esn.score(u[2400:], y[2400:]))
"""
from .diagnostics import (amplification_report, error_growth_rate,
                          predict_amplification)
from .model import ESN
from .readout import RidgeReadout
from .reservoir import SphereTTReservoir
from .tasks import mackey_glass, memory_capacity, narma, nrmse
from .ttstate import TTStateReservoir

__version__ = "0.4.1"

__all__ = [
    "ESN", "SphereTTReservoir", "TTStateReservoir", "RidgeReadout",
    "narma", "mackey_glass", "memory_capacity", "nrmse",
    "error_growth_rate", "predict_amplification", "amplification_report",
    "__version__",
]
