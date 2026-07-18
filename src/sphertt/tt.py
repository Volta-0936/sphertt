"""Tensor-train (TT/MPS) primitives.

TT vector : cores[k] of shape (r_{k-1}, n_k, r_k), with r_0 = r_d = 1.
TT matrix : cores[k] of shape (r_{k-1}, m_k, n_k, r_k)  (m = out, n = in).

All functions are dtype-agnostic (float64 / complex128).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "tt_svd", "tt_to_dense", "tt_ranks", "tt_params",
    "tt_zeros", "tt_add", "tt_norm", "tt_round", "tt_entries",
    "tt_rank1_kron_sum",
    "ttm_svd", "ttm_random", "ttm_kron_orthogonal", "ttm_add",
    "ttm_to_dense", "ttm_params", "ttm_matvec", "ttm_ttv",
    "power_iteration_norm",
]


# ---------------------------------------------------------------- TT vector

def tt_svd(x, dims, chi):
    """Compress a dense vector into TT cores via sequential SVD.

    Parameters
    ----------
    x : (N,) array with N == prod(dims)
    dims : list of mode sizes
    chi : maximum bond dimension (rank)
    """
    d = len(dims)
    cores = []
    t = np.asarray(x).reshape(dims)
    r_prev = 1
    for k in range(d - 1):
        m = t.reshape(r_prev * dims[k], -1)
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        tol_rank = int(np.sum(S > 1e-14 * S[0])) if S.size and S[0] > 0 else 1
        r = max(1, min(int(chi), tol_rank, len(S)))
        cores.append(U[:, :r].reshape(r_prev, dims[k], r))
        t = S[:r, None] * Vt[:r]
        r_prev = r
    cores.append(t.reshape(r_prev, dims[-1], 1))
    return cores


def tt_to_dense(cores):
    t = cores[0]
    for c in cores[1:]:
        t = np.tensordot(t, c, axes=([-1], [0]))
    return t.reshape(-1)


def tt_ranks(cores):
    return [c.shape[0] for c in cores] + [cores[-1].shape[-1]]


def tt_params(cores):
    return int(sum(c.size for c in cores))


def tt_zeros(dims):
    """Rank-1 TT representation of the zero vector."""
    return [np.zeros((1, n, 1)) for n in dims]


def tt_add(A, B, wa=1.0, wb=1.0):
    """Weighted sum ``wa*A + wb*B`` of two TT vectors (ranks add)."""
    d = len(A)
    out = []
    for k in range(d):
        a, b = A[k], B[k]
        ra0, n, ra1 = a.shape
        rb0, _, rb1 = b.shape
        if d == 1:
            out.append(wa * a + wb * b)
        elif k == 0:
            out.append(np.concatenate([wa * a, wb * b], axis=2))
        elif k == d - 1:
            out.append(np.concatenate([a, b], axis=0))
        else:
            c = np.zeros((ra0 + rb0, n, ra1 + rb1),
                         dtype=np.result_type(a, b))
            c[:ra0, :, :ra1] = a
            c[ra0:, :, ra1:] = b
            out.append(c)
    return out


def tt_norm(cores):
    """Euclidean norm of a TT vector via a right-to-left Gram sweep."""
    G = np.ones((1, 1))
    for c in reversed(cores):
        t = np.tensordot(c, G, axes=([2], [0]))
        G = np.tensordot(t, c.conj(), axes=([1, 2], [1, 2]))
    return float(np.sqrt(abs(G[0, 0])))


def tt_round(cores, chi):
    """Round (recompress) a TT vector to maximum bond dimension ``chi``.

    Standard TT rounding: right-to-left QR orthogonalization followed by a
    left-to-right truncated-SVD sweep.

    Returns
    -------
    (cores, trunc_err) : the rounded cores and the absolute truncation
        error  sqrt(sum of discarded singular values squared).
        The returned cores are left-orthogonal except the last one, so the
        vector norm equals ``np.linalg.norm(cores[-1])``.
    """
    d = len(cores)
    cores = list(cores)
    for k in range(d - 1, 0, -1):
        r0, n, r1 = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(r0, n * r1).T)
        cores[k] = Q.T.reshape(-1, n, r1)
        cores[k - 1] = np.tensordot(cores[k - 1], R.T, axes=([2], [0]))
    err2 = 0.0
    for k in range(d - 1):
        r0, n, r1 = cores[k].shape
        U, S, Vt = np.linalg.svd(cores[k].reshape(r0 * n, r1),
                                 full_matrices=False)
        tol_rank = int(np.sum(S > 1e-14 * S[0])) if S.size and S[0] > 0 else 1
        r = max(1, min(int(chi), tol_rank))
        err2 += float(np.sum(S[r:] ** 2))
        cores[k] = U[:, :r].reshape(r0, n, r)
        cores[k + 1] = np.tensordot(S[:r, None] * Vt[:r], cores[k + 1],
                                    axes=([1], [0]))
    return cores, float(np.sqrt(err2))


def tt_entries(cores, indices, dims):
    """Extract selected entries of a TT vector without densifying.

    Cost O(K * d * chi^2) for K indices — this is the readout path for
    reservoirs whose state never exists as a dense vector.
    """
    idx = np.unravel_index(np.asarray(indices, dtype=np.intp), dims)
    K = idx[0].size
    v = np.ones((K, 1))
    for k, c in enumerate(cores):
        sel = c[:, idx[k], :]                       # (r0, K, r1)
        v = np.einsum('kr,rks->ks', v, sel)
    return v[:, 0]


def tt_rank1_kron_sum(dims, R, rng):
    """Sum of ``R`` random Kronecker rank-1 vectors as a TT vector (rank R).

    Used as the input-coupling vector w_in for TT-state reservoirs: a dense
    i.i.d. w_in is TT-incompressible, so the input coupling must be born
    structured.
    """
    d = len(dims)
    vecs = [[rng.standard_normal(n) for n in dims] for _ in range(R)]
    cores = []
    for k, n in enumerate(dims):
        r0 = 1 if k == 0 else R
        r1 = 1 if k == d - 1 else R
        c = np.zeros((r0, n, r1))
        for r in range(R):
            c[0 if k == 0 else r, :, 0 if k == d - 1 else r] += vecs[r][k]
        cores.append(c)
    return cores


# ---------------------------------------------------------------- TT matrix

def ttm_svd(W, row_dims, col_dims, chi):
    """Compress a dense matrix into TT-matrix cores (TT-SVD on paired modes)."""
    d = len(row_dims)
    T = np.asarray(W).reshape(list(row_dims) + list(col_dims))
    perm = []
    for k in range(d):
        perm += [k, d + k]
    T = T.transpose(perm)
    cores = []
    r_prev = 1
    t = T
    for k in range(d - 1):
        m = t.reshape(r_prev * row_dims[k] * col_dims[k], -1)
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        tol_rank = int(np.sum(S > 1e-14 * S[0])) if S.size and S[0] > 0 else 1
        r = max(1, min(int(chi), tol_rank, len(S)))
        cores.append(U[:, :r].reshape(r_prev, row_dims[k], col_dims[k], r))
        t = S[:r, None] * Vt[:r]
        r_prev = r
    cores.append(t.reshape(r_prev, row_dims[-1], col_dims[-1], 1))
    return cores


def ttm_random(row_dims, col_dims, chi, rng):
    """Random Gaussian TT matrix with all internal ranks equal to ``chi``."""
    d = len(row_dims)
    ranks = [1] + [int(chi)] * (d - 1) + [1]
    return [rng.standard_normal((ranks[k], row_dims[k], col_dims[k],
                                 ranks[k + 1])) for k in range(d)]


def ttm_kron_orthogonal(dims, rng):
    """Kronecker product of per-mode random orthogonal matrices.

    Exactly orthogonal as an (N x N) operator, yet TT rank 1 — this is the
    isometric backbone of the quasi-orthogonal reservoir.
    """
    cores = []
    for n in dims:
        q, r = np.linalg.qr(rng.standard_normal((n, n)))
        q = q * np.sign(np.diag(r))
        cores.append(q.reshape(1, n, n, 1))
    return cores


def ttm_add(A, B, wa=1.0, wb=1.0):
    """Weighted sum ``wa*A + wb*B`` of two TT matrices (ranks add)."""
    d = len(A)
    out = []
    for k in range(d):
        a, b = A[k], B[k]
        ra0, m, n, ra1 = a.shape
        rb0, _, _, rb1 = b.shape
        if d == 1:
            out.append(wa * a + wb * b)
        elif k == 0:
            out.append(np.concatenate([wa * a, wb * b], axis=3))
        elif k == d - 1:
            out.append(np.concatenate([a, b], axis=0))
        else:
            c = np.zeros((ra0 + rb0, m, n, ra1 + rb1),
                         dtype=np.result_type(a, b))
            c[:ra0, :, :, :ra1] = a
            c[ra0:, :, :, ra1:] = b
            out.append(c)
    return out


def ttm_to_dense(cores):
    """Materialize a TT matrix as dense (guard: refuses > ~1 GB)."""
    d = len(cores)
    M = int(np.prod([c.shape[1] for c in cores]))
    N = int(np.prod([c.shape[2] for c in cores]))
    if M * N * 8 > 2 ** 30:
        raise MemoryError(
            f"dense materialization would need {M * N * 8 / 2**30:.1f} GB; "
            "use ttm_matvec instead")
    t = cores[0]
    for c in cores[1:]:
        t = np.tensordot(t, c, axes=([-1], [0]))
    t = t.reshape(t.shape[1:-1])
    perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    return t.transpose(perm).reshape(M, N)


def ttm_params(cores):
    return int(sum(c.size for c in cores))


def ttm_matvec(cores, x, col_dims):
    """y = W @ x for W in TT-matrix format and dense x. Never densifies W."""
    d = len(cores)
    t = np.asarray(x).reshape(col_dims)[None, ...]
    out_dims = []
    for k in range(d):
        c = cores[k]
        n_axis = 1 + len(out_dims)
        t = np.tensordot(c, t, axes=([0, 2], [0, n_axis]))
        t = np.moveaxis(t, 0, 1 + len(out_dims))
        out_dims.append(c.shape[1])
    return t.reshape(-1)


def ttm_ttv(W, x):
    """y = W @ x with BOTH operands in TT format; output ranks multiply.

    W cores (a0, m, n, a1), x cores (b0, n, b1) -> y cores (a0*b0, m, a1*b1).
    Follow with :func:`tt_round` to recompress.
    """
    out = []
    for wc, xc in zip(W, x):
        t = np.einsum('amnb,cnd->acmbd', wc, xc)
        a, c_, m, b, d_ = t.shape
        out.append(t.reshape(a * c_, m, b * d_))
    return out


def power_iteration_norm_tt(cores, dims, chi=16, iters=30, rng=None):
    """Spectral-radius estimate with the iterate kept in TT format.

    Same contract as :func:`power_iteration_norm` but never allocates a
    dense length-N vector — required beyond N ~ 10^6.  The per-iteration
    rounding adds a small bias; accuracy is comparable to the dense
    estimate (~10%), which is sufficient for connectivity rescaling.
    """
    rng = rng or np.random.default_rng(0)
    v = tt_rank1_kron_sum(dims, min(int(chi), 4), rng)
    nrm = tt_norm(v)
    if nrm > 0:
        v[0] = v[0] / nrm
    lam = 1.0
    for _ in range(iters):
        w = ttm_ttv(cores, v)
        w, _ = tt_round(w, chi)
        lam = float(np.linalg.norm(w[-1]))
        if lam < 1e-300:
            return 0.0
        w[-1] = w[-1] / lam
        v = w
    return lam


def power_iteration_norm(cores, dims, iters=60, rng=None):
    """Estimate the spectral radius of a TT matrix by power iteration.

    Note: for non-normal matrices this is an estimate (typically within
    ~10%), which is sufficient for rescaling reservoir connectivity; the
    sphere normalization removes any residual global-scale sensitivity.
    """
    N = int(np.prod(dims))
    v = (rng or np.random.default_rng(0)).standard_normal(N)
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(iters):
        w = ttm_matvec(cores, v, dims)
        lam = float(np.linalg.norm(w))
        if lam < 1e-300:
            return 0.0
        v = w / lam
    return lam
