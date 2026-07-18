import numpy as np
import pytest

from sphertt import ESN, SphereTTReservoir
from sphertt.tasks import memory_capacity, narma, nrmse


def test_esn_learns_narma10():
    u, y = narma(2500, order=10, rng=np.random.default_rng(0))
    esn = ESN(n_dims=5, seed=0, washout=200)
    esn.fit(u[:1500], y[:1500])
    score = esn.score(u[1500:], y[1500:])
    assert score < 0.6          # dense-equivalent quality band


def test_esn_learns_delay_line():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, 2000)
    y = np.roll(u, 5)
    esn = ESN(n_dims=5, seed=0, washout=200)
    esn.fit(u[:1200], y[:1200])
    assert esn.score(u[1200:], y[1200:]) < 0.1


def test_memory_capacity_reasonable():
    res = SphereTTReservoir(n_dims=5, seed=0)
    u = np.random.default_rng(0).uniform(0, 0.5, 3000)
    X = res.run(u)
    mc = memory_capacity(X, u, delays=50)
    assert mc > 40              # near-perfect memory over 50 delays


def test_fit_rejects_bad_lengths():
    esn = ESN(n_dims=5, seed=0)
    with pytest.raises(ValueError):
        esn.fit(np.zeros(100), np.zeros(99))
    with pytest.raises(ValueError):
        esn.fit(np.zeros(100), np.zeros(100))   # shorter than washout


def test_memory_capacity_no_wraparound():
    # X is an exact 30-step delay embedding of u: r^2 must be ~1 for k <= 30
    # and ~0 beyond, even when delays > washout (the wrap-around regime).
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, 3000)
    m = 30
    X = np.column_stack([np.concatenate([np.zeros(k), u[:-k]])
                         for k in range(1, m + 1)])
    mc, prof = memory_capacity(X, u, washout=10, delays=45,
                               return_profile=True)
    assert all(p > 0.99 for p in prof[:m])
    assert all(p < 0.1 for p in prof[m:])
    assert m - 0.5 < mc < m + 1.5


def test_esn_multi_input_cross_channel_delay():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, (2000, 2))
    y = np.concatenate([[0.0] * 3, u[:-3, 1]])   # y_t = u2(t-3)
    esn = ESN(n_dims=5, n_in=2, seed=0, washout=200)
    esn.fit(u[:1200], y[:1200])
    assert esn.score(u[1200:], y[1200:]) < 0.1


def test_save_load_roundtrip(tmp_path):
    u, y = narma(1200, order=10, rng=np.random.default_rng(0))
    esn = ESN(n_dims=5, seed=0, washout=200)
    esn.fit(u[:900], y[:900])
    path = tmp_path / "model.npz"
    esn.save(path)
    loaded = ESN.load(path)
    # both continue from the same saved reservoir state
    p1 = esn.predict(u[900:])
    p2 = loaded.predict(u[900:])
    assert np.allclose(p1, p2, atol=1e-12)
    assert loaded.reservoir.memory_ledger() == esn.reservoir.memory_ledger()


def test_save_rejects_unfitted(tmp_path):
    esn = ESN(n_dims=5, seed=0)
    with pytest.raises(RuntimeError):
        esn.save(tmp_path / "nope.npz")


def test_nrmse_zero_for_perfect_prediction():
    y = np.random.default_rng(0).standard_normal(100)
    assert nrmse(y, y) == 0.0
