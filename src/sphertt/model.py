"""High-level estimator combining reservoir + readout (sklearn-flavored)."""
from __future__ import annotations

import numpy as np

from .readout import RidgeReadout
from .reservoir import SphereTTReservoir
from .ttstate import TTStateReservoir

__all__ = ["ESN"]


def _sample_idx(rng, N, k):
    """k distinct indices in [0, N) without materializing arange(N)."""
    if N <= (1 << 24):
        return rng.choice(N, k, replace=False)
    idx = np.unique(rng.integers(0, N, int(k * 1.05) + 8))
    while len(idx) < k:
        idx = np.unique(np.concatenate([idx, rng.integers(0, N, k)]))
    return idx[:k]


class ESN:
    """Echo-state-style model: SphereTTReservoir + RidgeReadout.

    Parameters
    ----------
    reservoir : SphereTTReservoir or None
        If None, one is built from ``reservoir_kwargs``.
    n_readout : int, default 2048
        Number of randomly selected state components fed to the readout
        (all components if the state is smaller).
    washout : int, default 200
        Initial steps discarded before fitting.
    readout : RidgeReadout or None

    Examples
    --------
    >>> from sphertt import ESN
    >>> import numpy as np
    >>> esn = ESN(n_dims=5, seed=0)              # N = 4**5 = 1024
    >>> u = np.random.default_rng(0).uniform(0, 0.5, 1000)
    >>> y = np.roll(u, 3)                        # toy 3-step-delay target
    >>> esn.fit(u[:800], y[:800])                # doctest: +ELLIPSIS
    ESN(...)
    >>> pred = esn.predict(u[800:])
    """

    def __init__(self, reservoir=None, n_readout=2048, washout=200,
                 readout=None, seed=0, **reservoir_kwargs):
        if reservoir is None:
            reservoir_kwargs.setdefault("n_dims", 5)
            reservoir = SphereTTReservoir(seed=seed, **reservoir_kwargs)
        self.reservoir = reservoir
        self.readout = readout or RidgeReadout()
        self.washout = int(washout)
        rng = np.random.default_rng(seed)
        N = reservoir.N
        self.readout_idx = (_sample_idx(rng, N, n_readout)
                            if N > n_readout else None)

    def fit(self, u, y):
        """Reset the reservoir, run over ``u``, fit readout on ``y``.

        ``u`` is (T,) for scalar input or (T, n_in) for multi-channel
        reservoirs (build with ``ESN(n_in=..., ...)``).
        """
        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)
        if u.shape[0] != y.shape[0]:
            raise ValueError("u and y must have the same length")
        if u.shape[0] <= self.washout:
            raise ValueError("sequence shorter than washout")
        self.reservoir.reset()
        X = self.reservoir.run(u, self.readout_idx)
        self.readout.fit(X[self.washout:], y[self.washout:])
        return self

    def predict(self, u):
        """Continue from the current reservoir state and predict targets."""
        X = self.reservoir.run(np.asarray(u, dtype=float), self.readout_idx)
        return self.readout.predict(X)

    def score(self, u, y):
        """NRMSE (lower is better) of predictions on (u, y)."""
        from .tasks import nrmse
        return nrmse(np.asarray(y, dtype=float), self.predict(u))

    # -------------------------------------------------------------- save/load

    def save(self, path):
        """Serialize the fitted model (reservoir + readout) to ``.npz``.

        Works for both reservoir kinds; everything lives in TT format, so
        the file stays small (under 1 MB even at N > 10^9 for the TT-state
        reservoir)."""
        if self.readout.w is None:
            raise RuntimeError("cannot save an unfitted model")
        res = self.reservoir
        payload = {
            "dims": np.asarray(res.dims, dtype=np.int64),
            "beta": np.float64(res.beta),
            "chi_w": np.int64(res.chi_w),
            "n_in": np.int64(res.n_in),
            "in_scale": np.float64(res.in_scale),
            "seed": np.int64(res.seed),
            "washout": np.int64(self.washout),
            "has_idx": np.bool_(self.readout_idx is not None),
            "readout_idx": (self.readout_idx if self.readout_idx is not None
                            else np.zeros(0, dtype=np.int64)),
            "readout_w": self.readout.w,
            "readout_alpha": np.float64(self.readout.alpha_),
            "readout_quadratic": np.bool_(self.readout.quadratic),
            "n_cores": np.int64(len(res.W)),
        }
        from .tt import to_numpy
        for i, c in enumerate(res.W):
            payload[f"w_core_{i}"] = to_numpy(c)
        if isinstance(res, SphereTTReservoir):
            payload["format"] = np.int64(1)
            payload["w_in"] = to_numpy(res.w_in)
            payload["state"] = to_numpy(res.x)
        elif isinstance(res, TTStateReservoir):
            payload["format"] = np.int64(2)
            payload["chi_x"] = np.int64(res.chi_x)
            payload["chi_in"] = np.int64(res.chi_in)
            for k in range(len(res.dims)):
                payload[f"x_core_{k}"] = to_numpy(res.x[k])
            for j, v in enumerate(res.w_in_tt):
                for k, c in enumerate(v):
                    payload[f"win_{j}_core_{k}"] = to_numpy(c)
        else:
            raise NotImplementedError(
                f"save not supported for {type(res).__name__}")
        np.savez_compressed(path, **payload)
        return self

    @classmethod
    def load(cls, path):
        """Load a model saved with :meth:`save`, including the reservoir
        state at save time (predictions continue from where it left off)."""
        d = np.load(path)
        fmt = int(d["format"])
        dims = [int(v) for v in d["dims"]]
        if fmt == 1:
            res = SphereTTReservoir.__new__(SphereTTReservoir)
        elif fmt == 2:
            res = TTStateReservoir.__new__(TTStateReservoir)
            res.chi_x = int(d["chi_x"])
            res.chi_in = int(d["chi_in"])
        else:
            raise ValueError(f"unknown save format {fmt}")
        res.dims = dims
        res.N = int(np.prod(dims))
        res._xp = np
        res._dtype = np.float64
        res.beta = float(d["beta"])
        res.chi_w = int(d["chi_w"])
        res.n_in = int(d["n_in"])
        res.in_scale = float(d["in_scale"])
        res.seed = int(d["seed"])
        res.W = [d[f"w_core_{i}"] for i in range(int(d["n_cores"]))]
        if fmt == 1:
            res.w_in = d["w_in"]
            res.x = d["state"]
        else:
            res.x = [d[f"x_core_{k}"] for k in range(len(dims))]
            res.w_in_tt = [[d[f"win_{j}_core_{k}"] for k in range(len(dims))]
                           for j in range(res.n_in)]
            res.last_delta = 0.0
        readout = RidgeReadout(alpha=float(d["readout_alpha"]),
                               quadratic=bool(d["readout_quadratic"]))
        readout.alpha_ = float(d["readout_alpha"])
        readout.w = d["readout_w"]
        model = cls.__new__(cls)
        model.reservoir = res
        model.readout = readout
        model.washout = int(d["washout"])
        model.readout_idx = d["readout_idx"] if bool(d["has_idx"]) else None
        return model

    def __repr__(self):
        return (f"ESN(reservoir={self.reservoir!r}, washout={self.washout}, "
                f"n_readout={'all' if self.readout_idx is None else len(self.readout_idx)})")
