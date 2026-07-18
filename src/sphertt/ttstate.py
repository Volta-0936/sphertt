"""TT-state sphere reservoir — the state itself never exists densely.

Dynamics (identical in spirit to :class:`SphereTTReservoir`, but every
object stays in tensor-train format):

    z_t = W x_{t-1} + sum_j u_t[j] * w_in[j]     (TT matvec + TT adds)
    z_t -> tt_round(z_t, chi_x)                  (recompression, error delta_t)
    x_t = z_t / ||z_t||                          (scalar division: rank-free)

Structural facts this engine exploits:

* The orthogonal backbone Q_kron has TT rank 1, so the isometric part of W
  does NOT inflate the state rank at all; only the random part (rank chi_w)
  and the input (rank chi_in) do.
* The sphere normalization is a scalar division of one core — the *only*
  practical nonlinearity that is exactly rank-neutral.
* A dense i.i.d. input vector is TT-incompressible (research finding 1), so
  the input coupling is born structured: a sum of ``chi_in`` random
  Kronecker rank-1 vectors, norm-matched to the dense-reservoir heuristic
  ``||w_in|| = in_scale * sqrt(N/3)``.

The per-step rounding error ``delta_t`` is recorded for free (discarded
singular-value mass), so the amplification diagnostics of the accompanying
theory come built in: after :meth:`run`, see ``deltas_``.
"""
from __future__ import annotations

import numpy as np

from .reservoir import build_w
from .tt import (tt_add, tt_entries, tt_norm, tt_params, tt_ranks,
                 tt_rank1_kron_sum, tt_round, tt_to_dense, tt_zeros, ttm_ttv,
                 ttm_params)

__all__ = ["TTStateReservoir"]


class TTStateReservoir:
    """Sphere reservoir with TT connectivity AND a TT-compressed state.

    Parameters
    ----------
    n_dims : int
        Number of TT modes ``d``.  State dimension is ``N = mode_size**d``
        — with the state in TT format, ``N`` beyond dense-state reach
        (e.g. 4**15 > 10**9) is fine.
    mode_size : int, default 4
    beta : float, default 0.8
        Orthogonal-mix weight; [0.8, 0.85] is the validated stability band.
    chi_w : int, default 8
        TT rank of the random part of W.
    chi_x : int, default 16
        State bond dimension: the state is recompressed to this rank every
        step.  Larger = more faithful, slower.  If chi_x reaches the maximal
        rank of the tensorization the evolution is EXACT (equals the
        dense-state reservoir with the same W and w_in to machine
        precision).
    chi_in : int, default 4
        TT rank of each input-coupling vector (sum of chi_in random
        Kronecker rank-1 vectors).
    in_scale : float or "auto", default "auto"
        "auto" uses the validated dense-reservoir heuristic
        0.02*sqrt(1024/N)/sqrt(n_in); the TT w_in is norm-matched to it.
    n_in : int, default 1
    seed : int, default 0
        Same seed => same W as ``SphereTTReservoir`` (w_in differs by
        construction).
    """

    def __init__(self, n_dims, mode_size=4, beta=0.8, chi_w=8, chi_x=16,
                 chi_in=4, in_scale="auto", n_in=1, seed=0):
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if int(n_in) < 1 or int(chi_in) < 1 or int(chi_x) < 1:
            raise ValueError("n_in, chi_in and chi_x must be >= 1")
        self.dims = [int(mode_size)] * int(n_dims)
        self.N = int(mode_size) ** int(n_dims)
        self.beta = float(beta)
        self.chi_w = int(chi_w)
        self.chi_x = int(chi_x)
        self.chi_in = int(chi_in)
        self.n_in = int(n_in)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        self.W = build_w(self.dims, self.beta, self.chi_w, rng)

        if in_scale == "auto":
            in_scale = 0.02 * np.sqrt(1024.0 / self.N) / np.sqrt(self.n_in)
        self.in_scale = float(in_scale)
        target = self.in_scale * np.sqrt(self.N / 3.0)   # dense-w_in norm
        self.w_in_tt = []
        for _ in range(self.n_in):
            v = tt_rank1_kron_sum(self.dims, self.chi_in, rng)
            nrm = tt_norm(v)
            if nrm > 0:
                v[0] = v[0] * (target / nrm)
            self.w_in_tt.append(v)
        self.reset()

    # ------------------------------------------------------------------ api

    def reset(self):
        """Reset the reservoir state to the (rank-1) zero TT vector."""
        self.x = tt_zeros(self.dims)
        self.last_delta = 0.0
        return self

    def step(self, u):
        """Advance one time step; ``u`` is a scalar (n_in=1) or an (n_in,)
        vector.  Returns self (the state stays in TT format: ``self.x``)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if u.shape != (self.n_in,):
            raise ValueError(f"expected input of shape ({self.n_in},), "
                             f"got {u.shape}")
        z = ttm_ttv(self.W, self.x)
        for j in range(self.n_in):
            if u[j] != 0.0:
                z = tt_add(z, self.w_in_tt[j], 1.0, float(u[j]))
        z, err = tt_round(z, self.chi_x)
        nrm = float(np.linalg.norm(z[-1]))   # valid: cores left-orthogonal
        self.last_delta = err / nrm if nrm > 0 else 0.0
        if nrm > 0:
            z[-1] = z[-1] / nrm
        self.x = z
        return self

    def run(self, u, readout_idx=None):
        """Run over an input sequence, collecting state entries.

        Parameters
        ----------
        u : (T,) array of scalar inputs (n_in=1), or (T, n_in) array
        readout_idx : (K,) int array of state components to collect.
            Defaults to all N components — only allowed for N <= 65536;
            beyond that you must subsample (that is the point).

        Returns
        -------
        (T, K) array.  Per-step relative rounding errors are stored in
        ``self.deltas_`` (the delta_bar of the amplification law).
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
        if readout_idx is None:
            if self.N > 65536:
                raise ValueError("N > 65536: pass readout_idx (the dense "
                                 "state is exactly what we avoid building)")
            readout_idx = np.arange(self.N)
        readout_idx = np.asarray(readout_idx, dtype=np.intp)
        X = np.empty((len(u), len(readout_idx)))
        deltas = np.empty(len(u))
        for t in range(len(u)):
            self.step(u[t])
            X[t] = tt_entries(self.x, readout_idx, self.dims)
            deltas[t] = self.last_delta
        self.deltas_ = deltas
        return X

    def to_dense_state(self):
        """Materialize the current state as a dense vector (small N only)."""
        if self.N * 8 > 2 ** 30:
            raise MemoryError("state too large to densify; use tt_entries")
        return tt_to_dense(self.x)

    # ----------------------------------------------------------------- info

    @property
    def state_ranks(self):
        return tt_ranks(self.x)

    @property
    def n_params_w(self):
        return ttm_params(self.W)

    @property
    def n_params_state(self):
        return tt_params(self.x)

    def memory_ledger(self):
        win = sum(tt_params(v) for v in self.w_in_tt)
        return {
            "N": self.N,
            "tt_w_bytes": self.n_params_w * 8,
            "dense_w_bytes": self.N * self.N * 8,
            "tt_state_bytes": self.n_params_state * 8,
            "dense_state_bytes": self.N * 8,
            "tt_w_in_bytes": win * 8,
            "model_bytes_total": (self.n_params_w + self.n_params_state
                                  + win) * 8,
        }

    def __repr__(self):
        return (f"TTStateReservoir(N={self.N}, dims={self.dims}, "
                f"beta={self.beta}, chi_w={self.chi_w}, chi_x={self.chi_x}, "
                f"chi_in={self.chi_in}, n_in={self.n_in}, "
                f"in_scale={self.in_scale:.2e}, seed={self.seed})")
