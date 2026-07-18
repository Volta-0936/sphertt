import numpy as np

from sphertt import (SphereTTReservoir, amplification_report,
                     error_growth_rate, predict_amplification)


def test_predict_amplification_branches():
    # bounded regime: incoherent accumulation
    assert np.isclose(predict_amplification(0.968, 1e-3),
                      1 / np.sqrt(1 - 0.968 ** 2))
    # runaway regime: saturation ceiling
    assert np.isclose(predict_amplification(1.01, 0.05), np.sqrt(2) / 0.05)
    # tiny delta never beats saturation in bounded regime
    assert predict_amplification(0.5, 1.0) < 2.0


def test_growth_rate_bounded_for_quasi_orthogonal():
    res = SphereTTReservoir(n_dims=5, mode_size=2, beta=0.8, chi_w=8, seed=0)
    # mode_size=2, n_dims=5 -> N=32 tiny; use bigger: dims [2]*10 via mode_size=2,n_dims=10
    res = SphereTTReservoir(n_dims=10, mode_size=2, beta=0.8, chi_w=8, seed=0)
    u = np.random.default_rng(0).uniform(0, 0.5, 600)
    rep = amplification_report(res, u, chi_x=16, washout=200)
    assert rep["n_growth_samples"] > 100
    assert rep["g_bar"] < 1.0          # stability band: no runaway
    assert rep["amp_predicted"] < 10
    # measured amplification within a factor ~3 of the law (loose CI check)
    assert rep["amp_measured"] < 3 * rep["amp_predicted"] + 1


def test_exact_rank_gives_zero_error():
    res = SphereTTReservoir(n_dims=10, mode_size=2, beta=0.8, chi_w=8, seed=0)
    u = np.random.default_rng(0).uniform(0, 0.5, 300)
    m = error_growth_rate(res, u, chi_x=32, washout=100)  # 32 = max exact rank
    assert m["err_final"] < 1e-10
