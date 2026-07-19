"""Stability diagnostics for truncated norm-preserving dynamics.

This module implements the quantitative theory behind sphertt:

* In normalized (norm-preserving) reservoirs the largest tangent Lyapunov
  exponent is ~= 1 for ANY parameters — it carries no information.  What
  decides the fate of truncation errors is the mean one-step growth rate
  along the ACTUAL error direction, ``g_bar``.

* Amplification law (validated: 94% of measurements within a factor of 2):

      A  ~=  min( (1 - g_bar**2)**-0.5,  sqrt(2) / delta_bar )

  where ``delta_bar`` is the mean one-shot truncation error.  The first
  branch is incoherent error accumulation (bounded regime, g_bar < 1); the
  second is the saturation ceiling imposed by the sphere geometry.

Use :func:`error_growth_rate` to measure ``g_bar`` and ``delta_bar`` for a
reservoir + state-truncation level, then :func:`predict_amplification`.
"""
from __future__ import annotations

import numpy as np

from .tt import tt_svd, tt_to_dense, ttm_matvec

__all__ = ["error_growth_rate", "predict_amplification",
           "amplification_report"]


def _truncate_state(x, dims, chi):
    return tt_to_dense(tt_svd(x, dims, chi))


def error_growth_rate(reservoir, u, chi_x, state_dims=None, washout=200):
    """Twin-trajectory measurement of the error-direction growth rate.

    Runs the reservoir exactly and with the state TT-truncated to ``chi_x``
    at every step, and measures:

    * ``g_bar``      : geometric-mean one-step growth of the actual error
    * ``delta_bar``  : mean one-shot truncation error (injected per step)
    * ``err_final``  : mean relative state error over the last 50 steps
    * ``amp_measured`` : err_final / delta_bar

    Parameters
    ----------
    reservoir : SphereTTReservoir
    u : (T,) input sequence, or (T, n_in) for multi-channel reservoirs
    chi_x : state bond dimension for the truncated twin
    state_dims : mode sizes for the state tensorization
        (default: reservoir.dims — note the max exact rank this implies).
    """
    dims = state_dims or reservoir.dims
    from .tt import to_numpy
    W = [to_numpy(c) for c in reservoir.W]    # twin runs on CPU float64
    w_in = to_numpy(reservoir.w_in).astype(np.float64)
    wdims = reservoir.dims
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    xe = np.zeros(reservoir.N)
    xt = np.zeros(reservoir.N)
    logs_g, deltas, errs = [], [], []
    eps_prev = None
    for t, ut in enumerate(u):
        inj = w_in @ ut
        ze = ttm_matvec(W, xe, wdims) + inj
        ne = np.linalg.norm(ze)
        xe_new = ze / ne if ne > 0 else ze
        zt = ttm_matvec(W, xt, wdims) + inj
        nt = np.linalg.norm(zt)
        phi_xt = zt / nt if nt > 0 else zt
        xt_new = _truncate_state(phi_xt, dims, chi_x)
        if eps_prev is not None:
            pe = np.linalg.norm(eps_prev)
            # growth samples: from early transient on, while unsaturated
            # (in the runaway regime the error saturates quickly, so waiting
            #  for the washout would leave no valid samples)
            if t >= min(washout, 20) and 1e-12 < pe < 0.8:
                logs_g.append(np.log(np.linalg.norm(phi_xt - xe_new) / pe))
            if t >= washout:
                deltas.append(np.linalg.norm(xt_new - phi_xt))
                errs.append(np.linalg.norm(xt_new - xe_new))
        eps_prev = xt_new - xe_new
        xe, xt = xe_new, xt_new
    g_bar = float(np.exp(np.mean(logs_g))) if logs_g else float("nan")
    delta_bar = float(np.mean(deltas)) if deltas else float("nan")
    err_final = float(np.mean(errs[-50:])) if errs else float("nan")
    return {
        "g_bar": g_bar,
        "delta_bar": delta_bar,
        "err_final": err_final,
        "amp_measured": err_final / delta_bar if delta_bar > 0 else float("inf"),
        "n_growth_samples": len(logs_g),
    }


def predict_amplification(g_bar, delta_bar, eps_sat=np.sqrt(2.0)):
    """Amplification law:  A ~= min((1-g^2)^-1/2, eps_sat/delta)."""
    if not np.isfinite(g_bar):
        return float("nan")
    acc = (1.0 / np.sqrt(1.0 - g_bar * g_bar)
           if g_bar < 1.0 else float("inf"))
    sat = eps_sat / delta_bar if delta_bar > 0 else float("inf")
    return float(min(acc, sat))


def amplification_report(reservoir, u, chi_x, **kwargs):
    """Convenience: measure and compare against the law in one call."""
    m = error_growth_rate(reservoir, u, chi_x, **kwargs)
    m["amp_predicted"] = predict_amplification(m["g_bar"], m["delta_bar"])
    m["runaway"] = bool(m["g_bar"] >= 1.0)
    return m
