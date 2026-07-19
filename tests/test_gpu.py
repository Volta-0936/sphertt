"""Backend/precision tests. GPU tests are skipped when CuPy is absent
(e.g. on CI runners without a GPU)."""
import numpy as np
import pytest

from sphertt import SphereTTReservoir, TTStateReservoir

cupy = pytest.importorskip("cupy")
try:
    cupy.cuda.runtime.getDeviceCount()
except Exception:                                     # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

U = np.random.default_rng(0).uniform(0, 0.5, 120)


def test_ttstate_gpu_matches_cpu_fp64():
    r_cpu = TTStateReservoir(n_dims=5, chi_x=12, seed=0)
    r_gpu = TTStateReservoir(n_dims=5, chi_x=12, seed=0).to("cupy",
                                                            "float64")
    X_cpu = r_cpu.run(U, readout_idx=np.arange(64))
    X_gpu = r_gpu.run(U, readout_idx=np.arange(64))
    assert isinstance(X_gpu, np.ndarray)
    # SVD/QR sign conventions may differ between LAPACK and cuSOLVER, but
    # the represented state (and hence entries) must agree closely.
    assert np.allclose(X_cpu, X_gpu, atol=1e-8)
    assert np.allclose(r_cpu.deltas_, r_gpu.deltas_, atol=1e-8)


def test_ttstate_gpu_fp32_close_to_fp64():
    r64 = TTStateReservoir(n_dims=5, chi_x=12, seed=0)
    r32 = TTStateReservoir(n_dims=5, chi_x=12, seed=0).to("cupy",
                                                          "float32")
    r64.run(U, readout_idx=np.arange(8))
    r32.run(U, readout_idx=np.arange(8))
    d64 = r64.deltas_[20:].mean()
    d32 = r32.deltas_[20:].mean()
    # fp32 roundoff (~1e-7) is far below the truncation noise delta_bar
    assert abs(d32 - d64) < 0.1 * max(d64, 1e-3)


def test_dense_reservoir_gpu_matches_cpu():
    r_cpu = SphereTTReservoir(n_dims=5, seed=0)
    r_gpu = SphereTTReservoir(n_dims=5, seed=0).to("cupy", "float64")
    X_cpu = r_cpu.run(U)
    X_gpu = r_gpu.run(U)
    assert isinstance(X_gpu, np.ndarray)
    assert np.allclose(X_cpu, X_gpu, atol=1e-10)


def test_roundtrip_back_to_numpy():
    r = TTStateReservoir(n_dims=4, chi_x=8, seed=0).to("cupy", "float32")
    r.run(U[:20], readout_idx=np.arange(8))
    r.to("numpy", "float64")
    X = r.run(U[:20], readout_idx=np.arange(8))
    assert isinstance(X, np.ndarray) and np.isfinite(X).all()
