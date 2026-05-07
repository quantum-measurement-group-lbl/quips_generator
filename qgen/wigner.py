import numpy as np
from numpy.typing import NDArray

from qgen.state import GaussianState


def wigner(
    state: GaussianState,
    x_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the Wigner function of a Gaussian state on a (x, p) grid.

    Uses the qgen convention (vacuum covariance = I), in which

        W(x, p) = (1 / (pi * sqrt(det V))) * exp(-(r - r_bar)^T V^-1 (r - r_bar))

    Returns an array of shape `(len(x_grid), len(p_grid))`.
    """
    cov_inv = np.linalg.inv(state.cov)
    det_cov = np.linalg.det(state.cov)
    norm = 1.0 / (np.pi * np.sqrt(det_cov))

    dx = x_grid - state.mean[0]
    dp = p_grid - state.mean[1]
    DX, DP = np.meshgrid(dx, dp, indexing="ij")

    quad = (
        cov_inv[0, 0] * DX * DX
        + 2.0 * cov_inv[0, 1] * DX * DP
        + cov_inv[1, 1] * DP * DP
    )
    return norm * np.exp(-quad)


def auto_grid(
    state: GaussianState,
    n_sigma: float = 4.0,
    n_points: int = 201,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a +-n_sigma grid centered on the state's mean."""
    sx = np.sqrt(state.cov[0, 0])
    sp = np.sqrt(state.cov[1, 1])
    x_grid = np.linspace(state.mean[0] - n_sigma * sx, state.mean[0] + n_sigma * sx, n_points)
    p_grid = np.linspace(state.mean[1] - n_sigma * sp, state.mean[1] + n_sigma * sp, n_points)
    return x_grid, p_grid


def wigner_auto(
    state: GaussianState,
    n_sigma: float = 4.0,
    n_points: int = 201,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convenience wrapper: build an auto grid and evaluate the Wigner on it."""
    x_grid, p_grid = auto_grid(state, n_sigma=n_sigma, n_points=n_points)
    W = wigner(state, x_grid, p_grid)
    return x_grid, p_grid, W
