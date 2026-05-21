from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from qgen.backends.base import Backend
from qgen.backends.collision import apply_kick, average_collision
from qgen.model import Model
from qgen.result import Result


def dvariance(model: Model, covariances: NDArray[np.float64]) -> NDArray[np.float64]:
    Vx, Vp, Cxp = covariances
    omega = model.omega_dim
    g = model.gamma_meas_dim
    eta = model.eta
    dVx = 2 * omega * Cxp - 4 * eta * g * Vx**2
    dVp = -2 * omega * Cxp + 4 * g - 4 * eta * g * Cxp**2
    dCxp = omega * (Vp - Vx) - 4 * eta * g * Vx * Cxp
    return np.array([dVx, dVp, dCxp])


def variance_solver(
    model: Model, n_periods: float, dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    init = np.array([1.0 + model.n_thermal, 1.0 + model.n_thermal, 0.0])
    n_times = int(2 * np.pi * n_periods / dt)
    y = np.zeros((3, n_times))
    times = np.zeros(n_times)
    y[:, 0] = init
    for i in range(1, n_times):
        t = (i - 1) * dt
        cur = y[:, i - 1]
        k1 = dvariance(model, cur)
        k2 = dvariance(model, cur + dt / 2 * k1)
        k3 = dvariance(model, cur + dt / 2 * k2)
        k4 = dvariance(model, cur + dt * k3)
        y[:, i] = cur + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        times[i] = t + dt
    return times, y


def steady_state_variances(model: Model) -> tuple[float, float, float]:
    eta = model.eta
    # NOTE: factor 16 (not 4 as in the reference gaussian_oscillator.py) — derived
    # from solving the variance EOMs at steady state; matches the simulator fixed
    # point. The 4 in the reference was inconsistent with its own EOMs.
    xi = np.sqrt(1.0 + 16.0 * eta * model.gamma_meas_dim**2 / model.omega_dim**2)
    Vx = 2.0 / (np.sqrt(2.0 * eta) * np.sqrt(xi + 1.0))
    Vp = 2.0 * xi / (np.sqrt(2.0 * eta) * np.sqrt(xi + 1.0))
    Cxp = np.sqrt(xi - 1.0) / (np.sqrt(eta) * np.sqrt(xi + 1.0))
    return float(Vx), float(Vp), float(Cxp)


def kalman_filter(
    model: Model,
    dt: float,
    photocurrent: NDArray[np.float64],
    var_x: NDArray[np.float64],
    cov_xp: NDArray[np.float64],
    initial_means: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_times = photocurrent.shape[0]
    x_est = np.zeros(n_times)
    p_est = np.zeros(n_times)
    x_est[0], p_est[0] = initial_means

    sqrt_2eta_gm = 2.0 * np.sqrt(model.eta * model.gamma_meas_dim)
    omega = model.omega_dim

    for i in range(1, n_times):
        x = x_est[i - 1]
        p = p_est[i - 1]
        Vx = var_x[i - 1]
        Cxp = cov_xp[i - 1]
        # Reconstruct innovation from photocurrent: dY = x dt + dW / sqrt_2eta_gm
        # photocurrent[i] was stored as drecord/dt where drecord used x at step i-1
        dY_i = photocurrent[i] * dt
        dW_i = sqrt_2eta_gm * (dY_i - x * dt)

        # Same semi-implicit Euler step as simulate(): momentum first
        p_new = p - omega * x * dt + sqrt_2eta_gm * Cxp * dW_i
        x_new = x + omega * p_new * dt + sqrt_2eta_gm * Vx * dW_i

        x_est[i] = x_new
        p_est[i] = p_new

    return x_est, p_est


class GaussianBackend(Backend):
    def __init__(self, model: Model) -> None:
        super().__init__(model)

    def dvariance(self, covariances: NDArray[np.float64]) -> NDArray[np.float64]:
        return dvariance(self.model, covariances)

    def variance_solver(
        self, n_periods: float, dt: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return variance_solver(self.model, n_periods, dt)

    def steady_state_variances(self) -> tuple[float, float, float]:
        return steady_state_variances(self.model)

    def kalman_filter(
        self,
        result: Result,
        initial_means: tuple[float, float] = (0.0, 0.0),
    ) -> Result:
        dt = float(result.meta["dt"])
        x_est, p_est = kalman_filter(
            self.model, dt, result.photocurrent, result.Vxx, result.Cxp, initial_means
        )
        return replace(result, xc_kalman=x_est, pc_kalman=p_est)

    def simulate(
        self,
        n_periods: float,
        dt: float,
        seed: int | None = None,
        initial_means: tuple[float, float] = (0.0, 0.0),
        use_steady_state_covs: bool = False,
    ) -> Result:
        m = self.model
        n_times = int(2 * np.pi * n_periods / dt)

        # Inserting gas collisions throughout the simulation
        # For now, only 1 kick halfway through the simulation
        kick_step = n_times // 2
        # Possible Parameters: Kr, Xe, SF6, N2, H2 -> refer to GAS_SPECIES in collisions.py
        # Calculated in SI units so converted to dimensionless units
        kick_delta_p = average_collision(gas="SF6") / m.p_zpf

        if use_steady_state_covs:
            Vx_ss, Vp_ss, Cxp_ss = steady_state_variances(m)
            var_x = np.full(n_times, Vx_ss)
            var_p = np.full(n_times, Vp_ss)
            cov_xp = np.full(n_times, Cxp_ss)
            times_var = np.arange(n_times) * dt
        else:
            times_var, ys = variance_solver(m, n_periods, dt)
            var_x, var_p, cov_xp = ys[0], ys[1], ys[2]

        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal(n_times)

        xc = np.zeros(n_times)
        pc = np.zeros(n_times)
        photocurrent = np.zeros(n_times)
        xc[0], pc[0] = initial_means

        sqrt_2eta_gm = 2.0 * np.sqrt(m.eta * m.gamma_meas_dim)

        for i in range(1, n_times):
            x = xc[i - 1]
            p = pc[i - 1]

            # Applying collision at the kick_step
            if i == kick_step:
                p = apply_kick(p, kick_delta_p)

            Vx = var_x[i - 1]
            Cxp = cov_xp[i - 1]
            dW_i = dW[i - 1]

            # Semi-implicit Euler: momentum first, then position uses p_new
            p_new = p - m.omega_dim * x * dt + sqrt_2eta_gm * Cxp * dW_i
            x_new = x + m.omega_dim * p_new * dt + sqrt_2eta_gm * Vx * dW_i

            drecord = x * dt + dW_i / sqrt_2eta_gm

            xc[i] = x_new
            pc[i] = p_new
            photocurrent[i] = drecord / dt

        meta = {
            "model": m,
            "n_periods": n_periods,
            "dt": dt,
            "seed": seed,
            "use_steady_state_covs": use_steady_state_covs,
        }

        return Result(
            times=times_var,
            xc=xc,
            pc=pc,
            photocurrent=photocurrent,
            Vxx=var_x,
            Vpp=var_p,
            Cxp=cov_xp,
            meta=meta,
        )
