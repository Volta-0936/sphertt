import numpy as np

from sphertt import SphereTTReservoir
from sphertt.tt import ttm_matvec, ttm_to_dense


def test_state_on_unit_sphere():
    res = SphereTTReservoir(n_dims=5, seed=0)
    u = np.random.default_rng(0).uniform(0, 0.5, 300)
    X = res.run(u)
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms[5:], 1.0, atol=1e-12)


def test_deterministic_given_seed():
    u = np.random.default_rng(1).uniform(0, 0.5, 100)
    X1 = SphereTTReservoir(n_dims=5, seed=42).run(u)
    X2 = SphereTTReservoir(n_dims=5, seed=42).run(u)
    assert np.array_equal(X1, X2)


def test_w_is_quasi_orthogonal_mix():
    res = SphereTTReservoir(n_dims=4, beta=0.8, chi_w=4, seed=0)
    W = ttm_to_dense(res.W)
    # spectral radius of the mix should be within ~[0.6, 1.05]
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    assert 0.5 < sr < 1.1
    # rank bound: 1 + chi_w
    assert max(c.shape[0] for c in res.W) <= 1 + 4


def test_matvec_never_densifies_at_large_n():
    res = SphereTTReservoir(n_dims=9, seed=0)   # N = 262144, dense W = 512 GB
    assert res.dense_w_bytes > 5e11
    assert res.n_params_w * 8 < 200 * 1024      # under 200 KB
    x = res.step(0.3)                           # one step must just work
    assert np.isfinite(x).all()


def test_readout_idx_subsampling():
    res = SphereTTReservoir(n_dims=5, seed=0)
    idx = np.arange(16)
    X = res.reset().run(np.zeros(10) + 0.2, readout_idx=idx)
    assert X.shape == (10, 16)


def test_multi_input_shapes():
    res = SphereTTReservoir(n_dims=5, n_in=3, seed=0)
    u = np.random.default_rng(0).uniform(0, 0.5, (50, 3))
    X = res.run(u)
    assert X.shape == (50, res.N)
    assert np.allclose(np.linalg.norm(X[5:], axis=1), 1.0, atol=1e-12)
    import pytest
    with pytest.raises(ValueError):
        res.run(u[:, 0])                    # 1-D input to a 3-channel reservoir
    with pytest.raises(ValueError):
        res.step(np.zeros(2))               # wrong channel count


def test_single_channel_accepts_1d_and_2d():
    u = np.random.default_rng(1).uniform(0, 0.5, 100)
    X1 = SphereTTReservoir(n_dims=5, seed=0).run(u)
    X2 = SphereTTReservoir(n_dims=5, seed=0).run(u[:, None])
    assert np.array_equal(X1, X2)


def test_kron_sum_kinds_build_and_run():
    from sphertt.tt import ttm_to_dense
    for kind in ("kron-sum", "kron-orth-sum"):
        res = SphereTTReservoir(n_dims=4, chi_w=3, w_kind=kind, seed=0)
        assert max(c.shape[0] for c in res.W) <= 1 + 3      # rank <= 1+K
        X = res.run(np.random.default_rng(0).uniform(0, 0.5, 50))
        assert np.isfinite(X).all()
        assert np.allclose(np.linalg.norm(X[5:], axis=1), 1.0, atol=1e-12)
    # orthogonal variant: each Kronecker term is an exact isometry
    from sphertt.tt import ttm_kron_sum
    cores = ttm_kron_sum([4] * 3, 1, np.random.default_rng(0),
                         orthogonal=True)
    Q = ttm_to_dense(cores)
    assert np.linalg.norm(Q @ Q.T - np.eye(64)) < 1e-10


def test_auto_in_scale_scaling():
    r5 = SphereTTReservoir(n_dims=5, seed=0)
    r7 = SphereTTReservoir(n_dims=7, seed=0)
    assert np.isclose(r5.in_scale / r7.in_scale,
                      np.sqrt(r7.N / r5.N), rtol=1e-12)
