import numpy as np
import pytest

from sphertt import ESN, TTStateReservoir
from sphertt.tt import tt_norm, tt_to_dense, ttm_to_dense


def test_exact_when_chi_x_at_max_rank():
    # [4]^5 needs bond dims (4, 16, 16, 4) for exactness -> chi_x=16 is
    # lossless and the TT-state evolution must equal the dense simulation.
    res = TTStateReservoir(n_dims=5, chi_x=16, chi_in=4, seed=0)
    W = ttm_to_dense(res.W)
    w_in = tt_to_dense(res.w_in_tt[0])
    u = np.random.default_rng(0).uniform(0, 0.5, 100)

    X_tt = res.run(u)

    x = np.zeros(res.N)
    X_dense = np.empty((len(u), res.N))
    for t, ut in enumerate(u):
        z = W @ x + w_in * ut
        n = np.linalg.norm(z)
        x = z / n if n > 0 else z
        X_dense[t] = x

    assert np.allclose(X_tt, X_dense, atol=1e-9)
    assert res.deltas_.max() < 1e-12          # lossless -> no rounding error


def test_state_stays_normalized_and_rank_bounded():
    res = TTStateReservoir(n_dims=6, chi_x=8, seed=1)
    u = np.random.default_rng(1).uniform(0, 0.5, 60)
    X = res.run(u, readout_idx=np.arange(32))
    assert np.isfinite(X).all()
    assert abs(tt_norm(res.x) - 1.0) < 1e-10
    assert max(res.state_ranks) <= 8
    assert res.deltas_.shape == (60,)
    assert (res.deltas_[5:] > 0).any()        # truncation actually happens


def test_esn_integration_learns_delay_line():
    # chi_x=16 is lossless for [4]^4; chi_x=8 is a deep truncation (middle
    # bond 16 -> 8, per-step delta ~ 0.27) that rightly destroys memory.
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, 800)
    y = np.roll(u, 3)
    esn = ESN(reservoir=TTStateReservoir(n_dims=4, chi_x=16, seed=0),
              washout=100)
    esn.fit(u[:500], y[:500])
    assert esn.score(u[500:], y[500:]) < 0.2


def test_large_n_never_densifies():
    res = TTStateReservoir(n_dims=12, chi_x=4, chi_in=2, seed=0)  # N = 16.7M
    assert res.memory_ledger()["dense_w_bytes"] > 1e15
    X = res.run(np.full(3, 0.3), readout_idx=np.arange(8))
    assert X.shape == (3, 8)
    assert np.isfinite(X).all()
    assert res.memory_ledger()["model_bytes_total"] < 200 * 1024


def test_run_requires_readout_idx_at_large_n():
    res = TTStateReservoir(n_dims=9, chi_x=4, seed=0)             # N = 262144
    with pytest.raises(ValueError):
        res.run(np.zeros(3) + 0.2)


def test_ttstate_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, 700)
    y = np.roll(u, 3)
    esn = ESN(reservoir=TTStateReservoir(n_dims=4, chi_x=16, seed=0),
              washout=100)
    esn.fit(u[:500], y[:500])
    path = tmp_path / "ttmodel.npz"
    esn.save(path)
    loaded = ESN.load(path)
    assert isinstance(loaded.reservoir, TTStateReservoir)
    p1 = esn.predict(u[500:])
    p2 = loaded.predict(u[500:])
    assert np.allclose(p1, p2, atol=1e-12)
    assert loaded.reservoir.chi_x == 16


def test_multi_input_shapes():
    res = TTStateReservoir(n_dims=4, n_in=2, chi_x=8, seed=0)
    u = np.random.default_rng(0).uniform(0, 0.5, (20, 2))
    X = res.run(u)
    assert X.shape == (20, res.N)
    with pytest.raises(ValueError):
        res.step(0.3)                          # scalar to 2-channel reservoir
