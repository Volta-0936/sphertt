"""H5: can a higher orthogonal weight beta rescue the truncated regime?

H4 found delta_bar ~ (1-beta): the Kronecker-orthogonal backbone preserves
the state's TT structure exactly; only the random part injects
incompressible perturbation.  Here we measure what that buys in TASK terms
at N=65536, chi_x=16 (deep truncation: max rank 256), against dense-state
references at the same beta (= the ceiling the truncation must reach).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import ESN, SphereTTReservoir, TTStateReservoir  # noqa: E402
from sphertt.model import _sample_idx                          # noqa: E402
from sphertt.tasks import memory_capacity, narma               # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\005\results005c.jsonl"
u, y = narma(2500, order=10, rng=np.random.default_rng(0))
u_mc = np.random.default_rng(7).uniform(0, 0.5, 4000)
idx = _sample_idx(np.random.default_rng(0), 65536, 2048)


def log(rec):
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in rec.items()}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "mc_profile"}),
          flush=True)


def evaluate(res, tag, **meta):
    esn = ESN(reservoir=res, washout=200)
    t0 = time.time()
    esn.fit(u[:1500], y[:1500])
    sc = esn.score(u[1500:], y[1500:])
    ms = (time.time() - t0) / 2500 * 1e3
    X = res.reset().run(u_mc, readout_idx=idx)
    mc, prof = memory_capacity(X, u_mc, washout=300, delays=300,
                               return_profile=True)
    rec = {"exp": "H5", "engine": tag, "narma10": sc, "mc300": mc,
           "ms_per_step": ms, "mc_profile": np.asarray(prof), **meta}
    if hasattr(res, "deltas_"):
        rec["delta_mean"] = float(res.deltas_[200:].mean())
    log(rec)


# dense-state ceilings at each beta (what truncation must reach)
for beta in (0.8, 0.9, 0.95, 0.98):
    res = SphereTTReservoir(n_dims=8, beta=beta, seed=0)
    evaluate(res, "dense-state", beta=beta, chi_w=8)

# truncated TT-state candidates
for beta, chi_w in [(0.9, 8), (0.95, 8), (0.9, 2), (0.95, 2), (0.98, 2)]:
    res = TTStateReservoir(n_dims=8, beta=beta, chi_w=chi_w, chi_x=16,
                           chi_in=4, seed=0)
    evaluate(res, "tt-state", beta=beta, chi_w=chi_w, chi_x=16)

print("done.")
