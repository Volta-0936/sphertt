import numpy as np
import pytest

from sphertt.tt import (power_iteration_norm, tt_params, tt_svd, tt_to_dense,
                        ttm_add, ttm_kron_orthogonal, ttm_matvec, ttm_random,
                        ttm_svd, ttm_to_dense)

RNG = np.random.default_rng(0)
DIMS = [4] * 5
N = 4 ** 5


def test_tt_svd_exact_reconstruction():
    x = RNG.standard_normal(N)
    cores = tt_svd(x, DIMS, chi=10 ** 9)
    assert np.linalg.norm(tt_to_dense(cores) - x) < 1e-10 * np.linalg.norm(x)


def test_tt_truncation_monotone():
    x = RNG.standard_normal(N)
    prev = np.inf
    for chi in [1, 2, 4, 8, 16]:
        e = np.linalg.norm(tt_to_dense(tt_svd(x, DIMS, chi)) - x)
        assert e <= prev + 1e-12
        prev = e


def test_ttm_svd_roundtrip():
    W = RNG.standard_normal((N, N)) / np.sqrt(N)
    cores = ttm_svd(W, DIMS, DIMS, chi=10 ** 9)
    assert np.linalg.norm(ttm_to_dense(cores) - W) < 1e-9 * np.linalg.norm(W)


def test_ttm_matvec_matches_dense():
    cores = ttm_random(DIMS, DIMS, 5, RNG)
    x = RNG.standard_normal(N)
    y1 = ttm_matvec(cores, x, DIMS)
    y2 = ttm_to_dense(cores) @ x
    assert np.linalg.norm(y1 - y2) < 1e-10 * np.linalg.norm(y2)


def test_kron_orthogonal_is_orthogonal_and_rank1():
    cores = ttm_kron_orthogonal([4] * 4, RNG)
    assert all(c.shape[0] == c.shape[3] == 1 for c in cores)
    Q = ttm_to_dense(cores)
    assert np.linalg.norm(Q @ Q.T - np.eye(4 ** 4)) < 1e-10


def test_ttm_add_weighted_sum():
    A = ttm_kron_orthogonal([4] * 4, RNG)
    B = ttm_random([4] * 4, [4] * 4, 3, RNG)
    S = ttm_add(A, B, 0.8, 0.2)
    ref = 0.8 * ttm_to_dense(A) + 0.2 * ttm_to_dense(B)
    assert np.linalg.norm(ttm_to_dense(S) - ref) < 1e-10 * np.linalg.norm(ref)


def test_power_iteration_close_to_exact():
    # Power iteration on non-normal matrices is an estimate: 15% tolerance
    # (sufficient for spectral-radius rescaling of W_tt).
    cores = ttm_random([4] * 4, [4] * 4, 4, RNG)
    est = power_iteration_norm(cores, [4] * 4, iters=300, rng=RNG)
    exact = float(np.max(np.abs(np.linalg.eigvals(ttm_to_dense(cores)))))
    assert abs(est - exact) < 0.15 * exact


def test_ttm_to_dense_memory_guard():
    cores = ttm_random([4] * 10, [4] * 10, 2, RNG)
    with pytest.raises(MemoryError):
        ttm_to_dense(cores)


def test_tt_params_counts():
    cores = tt_svd(RNG.standard_normal(N), DIMS, 4)
    assert tt_params(cores) == sum(c.size for c in cores)
