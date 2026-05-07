import numpy as np

from qgen import GaussianBackend, GaussianState, Model, auto_grid, wigner, wigner_auto


def _trapz2d(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(W, p, axis=1), x))


def test_wigner_vacuum_normalization():
    state = GaussianState(mean=np.zeros(2), cov=np.eye(2))
    x = np.linspace(-8, 8, 401)
    p = np.linspace(-8, 8, 401)
    integral = _trapz2d(wigner(state, x, p), x, p)
    assert abs(integral - 1.0) < 1e-4, f"ERROR vacuum integral {integral:.6e} != 1"


def test_wigner_vacuum_peak():
    state = GaussianState(mean=np.zeros(2), cov=np.eye(2))
    W = wigner(state, np.array([0.0]), np.array([0.0]))
    expected = 1.0 / np.pi
    assert abs(W[0, 0] - expected) < 1e-12, f"ERROR vacuum peak {W[0, 0]:.6e} != {expected:.6e}"


def test_wigner_displaced_squeezed():
    theta = 0.6
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov = R @ np.diag([0.5, 2.0]) @ R.T
    mean = np.array([1.5, -0.7])
    state = GaussianState(mean=mean, cov=cov)

    x, p, W = wigner_auto(state, n_sigma=8.0, n_points=401)
    integral = _trapz2d(W, x, p)
    assert abs(integral - 1.0) < 1e-4, f"ERROR squeezed integral {integral:.6e} != 1"

    ix, ip = np.unravel_index(int(np.argmax(W)), W.shape)
    peak_err = max(abs(x[ix] - mean[0]), abs(p[ip] - mean[1]))
    grid_step = max(x[1] - x[0], p[1] - p[0])
    assert peak_err <= grid_step, f"ERROR peak location off by {peak_err:.3e} (grid step {grid_step:.3e})"


def test_auto_grid_extent():
    cov = np.diag([0.25, 9.0])
    state = GaussianState(mean=np.array([2.0, -1.0]), cov=cov)
    x, p = auto_grid(state, n_sigma=3.0, n_points=11)
    assert abs(x[0] - (2.0 - 3.0 * 0.5)) < 1e-12
    assert abs(p[-1] - (-1.0 + 3.0 * 3.0)) < 1e-12


def test_result_gaussian_state_extraction():
    backend = GaussianBackend(Model())
    r = backend.simulate(n_periods=1, dt=0.05, seed=0)
    s = r.gaussian_state(3)
    assert s.mean[0] == r.xc[3] and s.mean[1] == r.pc[3]
    assert s.cov[0, 0] == r.Vxx[3] and s.cov[1, 1] == r.Vpp[3]
    assert s.cov[0, 1] == r.Cxp[3] and s.cov[1, 0] == r.Cxp[3]
