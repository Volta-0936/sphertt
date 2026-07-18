"""Benchmark tasks and metrics (NARMA, Mackey-Glass, memory capacity)."""
from __future__ import annotations

import numpy as np

__all__ = ["narma", "mackey_glass", "nrmse", "memory_capacity"]


def nrmse(y_true, y_pred):
    """Normalized root-mean-square error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / np.var(y_true)))


def narma(T, order=10, rng=None, retries=10):
    """NARMA-``order`` benchmark. Returns (u, y); redraws u if it diverges."""
    rng = rng or np.random.default_rng(0)
    for _ in range(retries):
        u = rng.uniform(0.0, 0.5, size=T)
        y = np.zeros(T)
        ok = True
        for t in range(order - 1, T - 1):
            y[t + 1] = (0.3 * y[t]
                        + 0.05 * y[t] * np.sum(y[t - order + 1:t + 1])
                        + 1.5 * u[t - order + 1] * u[t] + 0.1)
            if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 10:
                ok = False
                break
        if ok:
            return u, y
    raise RuntimeError(f"NARMA-{order} diverged repeatedly")


def mackey_glass(T, tau=17, dt=1.0, x0=1.2, warmup=500, rng=None):
    """Mackey-Glass series via RK4 with interpolated delay (beta=0.2,
    gamma=0.1, n=10). Returns a length-``T`` array."""
    beta, gamma, n = 0.2, 0.1, 10
    h = dt / 10.0
    hist = int(np.ceil(tau / h)) + 1
    total = int((T + warmup) * dt / h)
    buf = np.empty(total + hist)
    buf[:hist] = x0
    if rng is not None:
        buf[:hist] += 0.01 * rng.standard_normal(hist)
    delay = tau / h

    def f(x, x_tau):
        return beta * x_tau / (1.0 + x_tau ** n) - gamma * x

    for i in range(hist, hist + total):
        def x_tau(offset):
            idx = i - 1 + offset - delay
            i0 = int(np.floor(idx))
            frac = idx - i0
            return buf[i0] * (1 - frac) + buf[min(i0 + 1, i - 1)] * frac
        x = buf[i - 1]
        k1 = f(x, x_tau(0.0))
        k2 = f(x + 0.5 * h * k1, x_tau(0.5))
        k3 = f(x + 0.5 * h * k2, x_tau(0.5))
        k4 = f(x + h * k3, x_tau(1.0))
        buf[i] = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    series = buf[hist::int(dt / h)][warmup:][:T]
    if len(series) != T:
        raise RuntimeError("mackey_glass length mismatch")
    return series


def memory_capacity(X, u, washout=200, train_frac=0.6, delays=100,
                    alpha=1e-6, return_profile=False):
    """Linear memory capacity: sum over k of r^2 between u(t-k) and a ridge
    readout of the states X. Standard linear-features definition.

    Fitting starts at ``max(washout, delays)`` so that every target u(t-k)
    exists for every delay (no wrap-around); with the default
    ``delays <= washout`` this matches the plain washout start."""
    X = np.asarray(X)
    u = np.asarray(u, dtype=float)
    n_tr = int(len(u) * train_frac)
    start = max(int(washout), int(delays))
    if n_tr <= start + 1 or len(u) - n_tr < 2:
        raise ValueError("sequence too short for the requested "
                         "washout/delays/train_frac")
    A_tr = np.hstack([X[start:n_tr], np.ones((n_tr - start, 1))])
    A_te = np.hstack([X[n_tr:], np.ones((len(u) - n_tr, 1))])
    P = np.linalg.inv(A_tr.T @ A_tr + alpha * np.eye(A_tr.shape[1])) @ A_tr.T
    prof = []
    for k in range(1, delays + 1):
        w = P @ u[start - k:n_tr - k]
        c = np.corrcoef(A_te @ w, u[n_tr - k:len(u) - k])[0, 1]
        prof.append(max(c, 0.0) ** 2 if np.isfinite(c) else 0.0)
    mc = float(np.sum(prof))
    return (mc, prof) if return_profile else mc
