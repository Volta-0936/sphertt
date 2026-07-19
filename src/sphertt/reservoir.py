"""The quasi-orthogonal hyperspherical tensor-train reservoir.

State update (norm-preserving, the projection itself is the nonlinearity):

    z_t = W x_{t-1} + w_in * u_t
    x_t = z_t / ||z_t||

with  W(beta) = beta * Q_kron + (1 - beta) * W_tt,  stored ONLY as a TT
matrix of rank <= 1 + chi_w.  Q_kron is exactly orthogonal (TT rank 1),
W_tt is a random TT matrix rescaled to unit spectral radius.

beta in [0.8, 0.85] is the empirically validated stability band: memory
capacity near its maximum while TT-truncation-error amplification stays
bounded (see the paper / diagnostics module).
"""
from __future__ import annotations

import numpy as np

from .tt import (power_iteration_norm, power_iteration_norm_tt, to_numpy,
                 ttm_add, ttm_kron_orthogonal, ttm_kron_sum, ttm_matvec,
                 ttm_params, ttm_random)

__all__ = ["SphereTTReservoir", "build_w"]


def build_w(dims, beta, chi_w, rng, kind="random-tt"):
    """Build the quasi-orthogonal TT connectivity W(beta) = beta*Q_kron +
    (1-beta)*W_tt (total TT rank <= 1 + chi_w).  Shared by the dense-state
    and TT-state reservoirs; identical rng state gives identical W.

    ``kind`` selects the perturbation family W_tt:
      "random-tt"     : dense-rank random Gaussian TT matrix (default)
      "kron-sum"      : sum of chi_w random Kronecker rank-1 terms
      "kron-orth-sum" : sum of chi_w orthogonal Kronecker terms
    The kron families are the *structured perturbations*: rank-chi states
    stay closer to low rank under them, shrinking both rounding cost and
    incompressible-error injection.
    """
    if kind == "random-tt":
        W_tt = ttm_random(dims, dims, chi_w, rng)
    elif kind == "kron-sum":
        W_tt = ttm_kron_sum(dims, chi_w, rng, orthogonal=False)
    elif kind == "kron-orth-sum":
        W_tt = ttm_kron_sum(dims, chi_w, rng, orthogonal=True)
    else:
        raise ValueError(f"unknown connectivity kind {kind!r}")
    N = int(np.prod(dims))
    # dense-vector power iteration up to N = 4^9 (keeps v0.1/v0.2
    # reproducibility there); TT-native beyond, where a dense iterate
    # would cost minutes and gigabytes
    sr = (power_iteration_norm(W_tt, dims, rng=rng) if N <= 2 ** 18
          else power_iteration_norm_tt(W_tt, dims, rng=rng))
    if sr > 0:
        W_tt[0] = W_tt[0] / sr
    Q = ttm_kron_orthogonal(dims, rng)
    if beta == 1.0:
        return Q
    if beta == 0.0:
        return W_tt
    return ttm_add(Q, W_tt, beta, 1.0 - beta)


class SphereTTReservoir:
    """Hyperspherical reservoir with tensor-train connectivity.

    Parameters
    ----------
    n_dims : int
        Number of TT modes ``d``.  State dimension is ``N = mode_size**d``.
    mode_size : int, default 4
        Size of each TT mode.
    beta : float, default 0.8
        Convex-combination weight of the orthogonal (isometric) part.
        1.0 = pure Kronecker orthogonal (degenerate mixing — not recommended),
        0.0 = pure random TT (seed-dependent truncation-error runaway).
    chi_w : int, default 8
        TT rank of the random part.  Total W rank <= 1 + chi_w.
    in_scale : float or "auto", default "auto"
        Input scaling.  "auto" uses 0.02 * sqrt(1024 / N) / sqrt(n_in),
        which keeps the input-to-state norm ratio invariant across N and
        across the number of input channels (validated heuristic).
    n_in : int, default 1
        Number of input channels.  ``step`` then takes an (n_in,) vector
        and ``run`` a (T, n_in) array; with n_in=1, plain scalars / (T,)
        arrays keep working and reproduce v0.1 trajectories exactly.
    seed : int, default 0
    """

    def __init__(self, n_dims, mode_size=4, beta=0.8, chi_w=8,
                 in_scale="auto", n_in=1, w_kind="random-tt", seed=0):
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if int(n_in) < 1:
            raise ValueError("n_in must be >= 1")
        self.dims = [int(mode_size)] * int(n_dims)
        self.N = int(mode_size) ** int(n_dims)
        self.beta = float(beta)
        self.chi_w = int(chi_w)
        self.n_in = int(n_in)
        self.w_kind = str(w_kind)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        self.W = build_w(self.dims, self.beta, self.chi_w, rng, self.w_kind)

        if in_scale == "auto":
            in_scale = 0.02 * np.sqrt(1024.0 / self.N) / np.sqrt(self.n_in)
        self.in_scale = float(in_scale)
        self.w_in = self.in_scale * rng.uniform(-1.0, 1.0, (self.N, self.n_in))
        self._xp = np
        self._dtype = np.float64
        self.reset()

    def to(self, backend="numpy", dtype=None):
        """Move the reservoir to a compute backend / precision, in place.
        See :meth:`TTStateReservoir.to` for semantics."""
        if backend == "cupy":
            import cupy as xp
        elif backend == "numpy":
            xp = np
        else:
            raise ValueError(f"unknown backend {backend!r}")
        dt = self._dtype if dtype is None else np.dtype(dtype)
        conv = lambda a: xp.asarray(to_numpy(a), dtype=dt)  # noqa: E731
        self.W = [conv(c) for c in self.W]
        self.w_in = conv(self.w_in)
        self.x = conv(self.x)
        self._xp = xp
        self._dtype = dt
        return self

    # ------------------------------------------------------------------ api

    def reset(self):
        """Reset the reservoir state to the origin."""
        self.x = self._xp.zeros(self.N, dtype=self._dtype)
        return self

    def step(self, u):
        """Advance one time step; ``u`` is a scalar (n_in=1) or an (n_in,)
        vector.  Returns the state."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if u.shape != (self.n_in,):
            raise ValueError(f"expected input of shape ({self.n_in},), "
                             f"got {u.shape}")
        u = self._xp.asarray(u, dtype=self._dtype)
        z = ttm_matvec(self.W, self.x, self.dims) + self.w_in @ u
        n = float(self._xp.linalg.norm(z))
        self.x = z / n if n > 0 else z
        return self.x

    def run(self, u, readout_idx=None):
        """Run over an input sequence.

        Parameters
        ----------
        u : (T,) array of scalar inputs (n_in=1), or (T, n_in) array
        readout_idx : optional (K,) int array — collect only these state
            components (memory-friendly for very large N).

        Returns
        -------
        (T, K or N) array of states (the reservoir state is NOT reset first;
        call :meth:`reset` for a fresh run).
        """
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            if self.n_in != 1:
                raise ValueError(f"reservoir has n_in={self.n_in}; "
                                 "pass input of shape (T, n_in)")
            u = u[:, None]
        elif u.ndim != 2 or u.shape[1] != self.n_in:
            raise ValueError(f"expected input of shape (T, {self.n_in}), "
                             f"got {u.shape}")
        k = self.N if readout_idx is None else len(readout_idx)
        idx = (None if readout_idx is None
               else self._xp.asarray(np.asarray(readout_idx, dtype=np.intp)))
        X = self._xp.empty((len(u), k), dtype=self._dtype)
        for t in range(len(u)):
            x = self.step(u[t])
            X[t] = x if idx is None else x[idx]
        return to_numpy(X)

    # ----------------------------------------------------------------- info

    @property
    def n_params_w(self):
        """Number of parameters actually stored for W."""
        return ttm_params(self.W)

    @property
    def dense_w_bytes(self):
        """Bytes a dense W would need (for the memory-wall ledger)."""
        return self.N * self.N * 8

    def memory_ledger(self):
        return {
            "N": self.N,
            "tt_w_bytes": self.n_params_w * 8,
            "dense_w_bytes": self.dense_w_bytes,
            "compression": self.dense_w_bytes / (self.n_params_w * 8),
        }

    def __repr__(self):
        return (f"SphereTTReservoir(N={self.N}, dims={self.dims}, "
                f"beta={self.beta}, chi_w={self.chi_w}, n_in={self.n_in}, "
                f"in_scale={self.in_scale:.2e}, seed={self.seed})")
