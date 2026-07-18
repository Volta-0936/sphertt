"""Experiment 006 — decomposing the rounding error of TT-state dynamics.

Hypothesis (from exp 005):  delta_bar(beta, chi_x)^2 ~= delta_flow^2 + delta_stock^2
  * flow  : incompressible perturbation injected per step by (1-beta)*W_tt
            -> proportional to (1-beta), vanishes at beta=1
  * stock : one-shot incompressibility of the steady state itself
            (accumulated input history) -> measurable at beta=1, and
            predictable from the Schmidt spectrum of the exact attractor.

Parts:
  A : exact (dense-state) attractor at N=65536 — one-shot truncation error
      delta_attr(beta, chi) + middle-bond Schmidt spectrum
  B : TT-state runs — delta_bar on a (beta, chi_x) grid + beta=1 stock
      isolation + chi_in effect
  B2: dense-twin error dynamics for the chi_x=16 row: g_bar, steady error
Results -> results006.jsonl
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\src")
from sphertt import SphereTTReservoir, TTStateReservoir  # noqa: E402
from sphertt.tt import tt_svd, tt_to_dense, ttm_matvec   # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\006\results006.jsonl"
DIMS = [4] * 8
N = 65536
BETAS = (0.8, 0.9, 0.95, 0.98, 1.0)
CHI_GRID = (4, 8, 16, 24, 32, 48, 64, 96, 128)

rng_u = np.random.default_rng(7)
u = rng_u.uniform(0, 0.5, 1000)


def log(rec):
    rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in rec.items()}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    brief = {k: (v if not isinstance(v, list) else f"[{len(v)}]")
             for k, v in rec.items()}
    print(json.dumps(brief), flush=True)


def oneshot(x, chi):
    """Relative one-shot TT truncation error of dense vector x."""
    xt = tt_to_dense(tt_svd(x, DIMS, chi))
    return float(np.linalg.norm(xt - x) / np.linalg.norm(x))


# ---------------------------------------------------------------- Part A
print("=== A: exact attractor compressibility ===")
for beta in BETAS:
    res = SphereTTReservoir(n_dims=8, beta=beta, seed=0)
    t0 = time.time()
    states = []
    res.reset()
    for t, ut in enumerate(u):
        x = res.step(ut)
        if t >= 200 and (t - 200) % 20 == 0:
            states.append(x.copy())
    errs = {chi: [oneshot(x, chi) for x in states] for chi in CHI_GRID}
    spec = np.mean([np.linalg.svd(x.reshape(256, 256), compute_uv=False)
                    for x in states], axis=0)
    log({"part": "A", "beta": beta, "n_states": len(states),
         "delta_attr_mean": {str(c): float(np.mean(e))
                             for c, e in errs.items()},
         "delta_attr_max": {str(c): float(np.max(e))
                            for c, e in errs.items()},
         "schmidt_mid": spec / np.linalg.norm(spec),
         "sec": time.time() - t0})

# ---------------------------------------------------------------- Part B
print("=== B: TT-state delta_bar grid ===")
for chi_x in (8, 16, 32):
    for beta in BETAS:
        res = TTStateReservoir(n_dims=8, beta=beta, chi_x=chi_x, chi_in=4,
                               seed=0)
        t0 = time.time()
        res.run(u[:300], readout_idx=np.arange(8))
        log({"part": "B", "beta": beta, "chi_x": chi_x, "chi_in": 4,
             "delta_bar": float(res.deltas_[100:].mean()),
             "ms_per_step": (time.time() - t0) / 300 * 1e3})
for chi_in in (1, 16):
    res = TTStateReservoir(n_dims=8, beta=1.0, chi_x=16, chi_in=chi_in,
                           seed=0)
    res.run(u[:300], readout_idx=np.arange(8))
    log({"part": "B", "beta": 1.0, "chi_x": 16, "chi_in": chi_in,
         "delta_bar": float(res.deltas_[100:].mean())})

# ---------------------------------------------------------------- Part B2
print("=== B2: twin-trajectory error dynamics (chi_x=16 row) ===")
for beta in BETAS:
    res = TTStateReservoir(n_dims=8, beta=beta, chi_x=16, chi_in=4, seed=0)
    W, wdims = res.W, res.dims
    w_in = tt_to_dense(res.w_in_tt[0])
    xe = np.zeros(N)
    errs, logs_g = [], []
    eps_prev = None
    res.reset()
    for t, ut in enumerate(u[:600]):
        ze = ttm_matvec(W, xe, wdims) + w_in * ut
        ne = np.linalg.norm(ze)
        xe = ze / ne if ne > 0 else ze
        res.step(ut)
        xt = tt_to_dense(res.x)
        err = np.linalg.norm(xt - xe)
        if eps_prev is not None and 1e-12 < eps_prev < 0.8 and t >= 20:
            logs_g.append(np.log(err / eps_prev) if err > 0 else 0.0)
        eps_prev = err
        if t >= 200:
            errs.append(err)
    log({"part": "B2", "beta": beta, "chi_x": 16,
         "g_bar": float(np.exp(np.mean(logs_g))) if logs_g else None,
         "err_steady": float(np.mean(errs[-100:]))})

print("done.")
