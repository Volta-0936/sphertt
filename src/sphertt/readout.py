"""Ridge readout with optional quadratic features and validated alpha."""
from __future__ import annotations

import numpy as np

__all__ = ["RidgeReadout"]


class RidgeReadout:
    """Linear readout trained by ridge regression.

    Parameters
    ----------
    alpha : float or "auto", default "auto"
        Ridge coefficient.  "auto" selects from ``alphas`` on the last
        ``val_frac`` slice of the training window.
    quadratic : bool, default True
        Append element-wise squares [x, x**2] (recommended for the sphere
        reservoir, whose only dynamical nonlinearity is the normalization).
    """

    def __init__(self, alpha="auto", quadratic=True,
                 alphas=(1e-8, 1e-6, 1e-4, 1e-2), val_frac=0.2):
        self.alpha = alpha
        self.quadratic = bool(quadratic)
        self.alphas = tuple(alphas)
        self.val_frac = float(val_frac)
        self.w = None
        self.alpha_ = None

    def _features(self, X):
        X = np.asarray(X)
        if self.quadratic:
            X = np.hstack([X, X ** 2])
        return np.hstack([X, np.ones((len(X), 1))])

    @staticmethod
    def _solve(A, y, alpha):
        G = A.T @ A + alpha * np.eye(A.shape[1])
        return np.linalg.solve(G, A.T @ y)

    def fit(self, X, y):
        A = self._features(X)
        y = np.asarray(y, dtype=float)
        if self.alpha == "auto":
            n_val = max(1, int(self.val_frac * len(A)))
            A_tr, y_tr = A[:-n_val], y[:-n_val]
            A_va, y_va = A[-n_val:], y[-n_val:]
            best = None
            for a in self.alphas:
                w = self._solve(A_tr, y_tr, a)
                err = float(np.mean((A_va @ w - y_va) ** 2))
                if best is None or err < best[0]:
                    best = (err, a)
            self.alpha_ = best[1]
        else:
            self.alpha_ = float(self.alpha)
        self.w = self._solve(A, y, self.alpha_)
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("readout is not fitted")
        return self._features(X) @ self.w
