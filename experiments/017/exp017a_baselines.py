"""017a — budget-matched classical baselines (CPU, minutes).

The review of 2026-07-20 found that Papers I/II compare the TT design
only against (i) an impossible dense-W strawman and (ii) a naive
configuration of the same architecture.  This experiment adds the
baselines a referee will demand: what does the same byte budget buy in
a *boring* design, under the identical MC protocol?

Baselines (all linear-memory-optimal families, T=24000, DELAYS=1200):

  dense-qo N=328  : dense quasi-orthogonal hypersphere reservoir,
                    W = beta*Q + (1-beta)*P, sphere-normalized.
                    N=328 -> dense W = 861 KB = the d15 TT model total.
  dense-qo N=104  : same at 86 KB (the "MC=104 equivalence budget").
  ring-in1 N=2048 : cyclic shift * r, input into unit 0 only — a decaying
                    delay line (Rodan & Tino SCR topology, linear).
                    Model = 1 scalar + input scalar: ~33 KB with state.
  ring-full N=2048: same ring, dense random-sign input vector.

Seeds 0/1/2 each.  MC window 1200 (uncensored; the old 300 window is
also reported for comparability).
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\sphertt-0.1.0\sphertt\experiments\017")
from exp017_common import log, mc_validated, test_mc_equivalence  # noqa: E402

OUT = r"D:\sphertt-0.1.0\prototype\017\results017a.jsonl"
T, WASHOUT, DELAYS = 24000, 1200, 1200
U = np.random.default_rng(7).uniform(0, 0.5, T)

print("mc_equivalence check:", test_mc_equivalence())


def run_dense_qo(N, beta, seed):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((N, N)))
    Q = Q * np.sign(np.diag(R))
    P = rng.standard_normal((N, N))
    P /= np.max(np.abs(np.linalg.eigvals(P)))
    W = beta * Q + (1.0 - beta) * P
    w_in = 0.02 * np.sqrt(1024.0 / N) * rng.uniform(-1, 1, N)
    x = np.zeros(N)
    X = np.empty((T, N))
    for t in range(T):
        z = W @ x + w_in * U[t]
        n = np.linalg.norm(z)
        x = z / n if n > 0 else z
        X[t] = x
    mc, a_star, prof = mc_validated(X, U, T, WASHOUT, DELAYS)
    log(OUT, {"part": "dense-qo", "N": N, "beta": beta, "seed": seed,
              "model_bytes": N * N * 8 + N * 8 + N * 8,
              "mc1200_test": round(mc, 2),
              "mc300_test": round(float(np.sum(prof[:300])), 2),
              "alpha_star": a_star,
              "r2_profile": np.round(prof, 4)})


def run_ring(N, r, seed, input_mode):
    rng = np.random.default_rng(seed)
    if input_mode == "in1":
        w_in = np.zeros(N)
        w_in[0] = 1.0
        model_bytes = 2 * 8 + N * 8          # r + one input weight + state
    else:
        w_in = rng.choice([-1.0, 1.0], N)
        model_bytes = 8 + N * 8 + N * 8      # r + input vector + state
    x = np.zeros(N)
    X = np.empty((T, N))
    for t in range(T):
        x = r * np.roll(x, 1) + w_in * U[t]
        X[t] = x
    mc, a_star, prof = mc_validated(X, U, T, WASHOUT, DELAYS)
    log(OUT, {"part": f"ring-{input_mode}", "N": N, "r": r, "seed": seed,
              "model_bytes": model_bytes,
              "mc1200_test": round(mc, 2),
              "mc300_test": round(float(np.sum(prof[:300])), 2),
              "alpha_star": a_star,
              "r2_profile": np.round(prof, 4)})


for seed in (0, 1, 2):
    for N, beta in ((328, 0.99), (328, 0.8), (104, 0.99)):
        run_dense_qo(N, beta, seed)
    run_ring(2048, 0.99, seed, "in1")
    run_ring(2048, 0.99, seed, "full")

print("017a done.")
