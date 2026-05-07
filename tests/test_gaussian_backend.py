import numpy as np

from qgen import GaussianBackend, Model


def test_steady_state_variances_match_analytic():
    backend = GaussianBackend(Model())
    _, ys = backend.variance_solver(n_periods=80, dt=0.01)
    last = ys[:, -1]
    expected = np.array(backend.steady_state_variances())
    rel = np.abs(last - expected) / np.abs(expected)
    idx = int(np.argmax(rel))
    assert rel.max() < 1e-3, (
        f"ERROR steady-state variance mismatch idx={idx} expected={expected[idx]:.6e} "
        f"actual={last[idx]:.6e} rel={rel[idx]:.3e}"
    )


def test_variance_solver_positivity():
    backend = GaussianBackend(Model())
    _, ys = backend.variance_solver(n_periods=20, dt=0.01)
    n_total = ys.shape[1] * 2  # Vx and Vp entries
    n_pos = int(np.sum(ys[0] > 0) + np.sum(ys[1] > 0))
    assert n_pos == n_total, f"ERROR variance positivity pass-rate {n_pos}/{n_total}"


def test_seed_reproducibility():
    backend = GaussianBackend(Model())
    r1 = backend.simulate(n_periods=2, dt=0.01, seed=0)
    r2 = backend.simulate(n_periods=2, dt=0.01, seed=0)
    assert np.array_equal(r1.photocurrent, r2.photocurrent), "ERROR seed reproducibility failed"
    assert np.array_equal(r1.xc, r2.xc) and np.array_equal(r1.pc, r2.pc)
