"""Part D — Schmidt spectra of the TRUNCATED system's own attractor.

Part A showed the exact attractor is near-maximally entangled (flat middle
Schmidt spectrum, one-shot 0.93 at chi=16).  Yet the TT run only loses
delta_bar ~ 0.26/step: the truncated dynamics lives on its own low-rank
attractor.  Here we measure, at steady state:
  * spectrum of the truncated state x (rank chi_x),
  * spectrum of the pre-rounding z = W x + u w_in (rank (1+chi_w)chi_x+chi_in),
  * one-shot error of z at chi_x  (consistency: should equal delta_bar).
The tail of z's spectrum beyond chi_x IS the floor mechanism.
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import TTStateReservoir                     # noqa: E402
from sphertt.tt import tt_add, tt_to_dense, ttm_ttv      # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\006\results006b.jsonl"
u = np.random.default_rng(7).uniform(0, 0.5, 400)

for beta, chi_x in [(0.8, 16), (0.95, 16), (1.0, 16), (0.8, 32), (0.95, 32)]:
    res = TTStateReservoir(n_dims=8, beta=beta, chi_x=chi_x, chi_in=4,
                           seed=0)
    spec_x, spec_z, oneshot_z, deltas = [], [], [], []
    res.reset()
    for t, ut in enumerate(u):
        if t >= 300 and t % 10 == 0:
            # pre-rounding z from the current state
            z_tt = tt_add(ttm_ttv(res.W, res.x), res.w_in_tt[0], 1.0,
                          float(ut))
            z = tt_to_dense(z_tt)
            s = np.linalg.svd(z.reshape(256, 256), compute_uv=False)
            spec_z.append(s / np.linalg.norm(s))
            oneshot_z.append(np.sqrt((s[chi_x:] ** 2).sum() / (s ** 2).sum()))
            x = tt_to_dense(res.x)
            sx = np.linalg.svd(x.reshape(256, 256), compute_uv=False)
            spec_x.append(sx / np.linalg.norm(sx))
        res.step(ut)
        if t >= 300:
            deltas.append(res.last_delta)
    rec = {"part": "D", "beta": beta, "chi_x": chi_x,
           "delta_bar_run": float(np.mean(deltas)),
           "oneshot_z_mid": float(np.mean(oneshot_z)),
           "spec_x": np.mean(spec_x, axis=0).tolist(),
           "spec_z": np.mean(spec_z, axis=0).tolist()}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: (v if not isinstance(v, list) else f"[{len(v)}]")
                      for k, v in rec.items()}), flush=True)
print("done.")
