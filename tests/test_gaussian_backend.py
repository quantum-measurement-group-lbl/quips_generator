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


def test_kalman_self_consistency():
    backend = GaussianBackend(Model())
    init = (0.3, -0.2)
    res = backend.simulate(n_periods=5, dt=0.01, seed=0, initial_means=init)
    filt = backend.kalman_filter(res, initial_means=init)
    err_x = float(np.max(np.abs(filt.xc_kalman - res.xc)))
    err_p = float(np.max(np.abs(filt.pc_kalman - res.pc)))
    assert err_x < 1e-12 and err_p < 1e-12, (
        f"ERROR kalman self-consistency max|dx|={err_x:.3e} max|dp|={err_p:.3e}"
    )


def test_kalman_converges_from_wrong_init():
    backend = GaussianBackend(Model())
    res = backend.simulate(n_periods=20, dt=0.01, seed=1, initial_means=(0.0, 0.0))
    filt = backend.kalman_filter(res, initial_means=(2.0, 2.0))
    n = len(res.xc)
    early = float(np.max(np.abs(filt.xc_kalman[: n // 20] - res.xc[: n // 20])))
    late = float(np.max(np.abs(filt.xc_kalman[-n // 4 :] - res.xc[-n // 4 :])))
    assert late < 0.1 * (early + 1e-12), (
        f"ERROR kalman convergence early={early:.3e} late={late:.3e}"
    )


def test_kalman_innovation_matches_simulator_dW():
    # Recompute innovations from the filter run; they must equal the rng dW the
    # simulator drew (since algorithms are identical with matched initial means).
    backend = GaussianBackend(Model())
    m = backend.model
    init = (0.0, 0.0)
    res = backend.simulate(n_periods=3, dt=0.01, seed=2, initial_means=init)
    filt = backend.kalman_filter(res, initial_means=init)

    sqrt_2eta_gm = 2.0 * np.sqrt(m.eta * m.gamma_meas_dim)
    dt = float(res.meta["dt"])
    # Reproduce the simulator's dW sequence
    rng = np.random.default_rng(2)
    dW_sim = np.sqrt(dt) * rng.standard_normal(len(res.xc))

    n = len(res.xc)
    dW_recon = np.zeros(n)
    for i in range(1, n):
        dY_i = res.photocurrent[i] * dt
        dW_recon[i - 1] = sqrt_2eta_gm * (dY_i - filt.xc_kalman[i - 1] * dt)

    err = float(np.max(np.abs(dW_recon[: n - 1] - dW_sim[: n - 1])))
    assert err < 1e-12, f"ERROR kalman innovation mismatch max|ddW|={err:.3e}"
