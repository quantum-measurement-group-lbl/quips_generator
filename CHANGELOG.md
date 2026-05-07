# q-gen Development Progress

## Status: Initial framework spedcified

**Project scope has been defined, work is ongoing to build a first pass**

### May 5, 2026: Project begins

### May 7, 2026: Minimal Gaussian backend skeleton
- Created `qgen/` package with `Model`, `Result`, `Backend` (ABC), `GaussianBackend`.
- Ported variance and conditional-mean EOMs from `gaussian_oscillator.py` using
  the same constants (`omega`, `gamma_meas`, `eta`, `n_thermal`). Variance ODE
  uses RK4; conditional-mean SDE uses semi-implicit Euler–Maruyama (momentum
  step first, then position uses `p_new`).
- Photocurrent: `dY = x dt + dW/(2·sqrt(eta·gamma_meas))`.
- Added `tests/test_gaussian_backend.py` (steady-state match, positivity,
  seed reproducibility) and `examples/single_mode_gaussian.py`.
- pytest configured via `[tool.pytest.ini_options]` in `pyproject.toml`
  (`pythonpath = ["."]`); pytest added to dev dependency group.
- **Correction vs reference**: the analytic steady-state formula in
  `gaussian_oscillator.py` had `xi = sqrt(1 + 4·eta·gamma_meas²/omega²)`,
  but solving the EOMs at fixed point gives `xi = sqrt(1 + 16·eta·gamma_meas²/omega²)`
  (factor of 4 different inside the radical). The simulator agrees with the
  corrected formula. Used the corrected version in `qgen.backends.gaussian`.
### May 7, 2026: SI units backend
- Refactored `Model` to hold SI fields with experimental defaults: `mass=5fg`,
  `omega=2*pi*50 kHz`, `gamma_ba=2*pi*1 kHz`, `eta=1`. Dimensionless quantities
  are exposed via properties (`omega_dim=1`, `gamma_meas_dim=gamma_ba/omega`,
  `x_zpf`, `p_zpf`). Backend math unchanged; just consumes the `*_dim` views.
- New `qgen/units.py` with `HBAR`, `DimensionalResult` dataclass, and
  `to_si(result, model)` for re-dimensionalising trajectories.
  Conversion: `t/=omega`, `x*=x_zpf`, `p*=p_zpf`, `V` accordingly,
  `photocurrent *= x_zpf*omega` (units m/s; signal part is `x(t)*omega`).
- Added `tests/test_units.py` (zpf scales, dimensionless rate at defaults,
  round-trip identities). All 12 tests pass.
- Example `examples/single_mode_gaussian.py` now plots in SI (pm, ms, m/s).

### May 7, 2026: Kalman filter from photocurrent
- Added low-level `kalman_filter(model, dt, photocurrent, var_x, cov_xp, initial_means)`
  in `qgen/backends/gaussian.py` and `GaussianBackend.kalman_filter(result, initial_means)`
  wrapper. Filter uses the SAME semi-implicit Euler step as `simulate()` driven by the
  innovation `dW = 2*sqrt(eta*g)*(dY - x_est dt)` reconstructed from the recorded
  photocurrent — so feeding back the simulator's own photocurrent (with matching
  initial means) reproduces `xc`/`pc` to ~1e-16.
- Extended `Result` and `DimensionalResult` with optional `xc_kalman`, `pc_kalman`
  fields; `to_si` re-dimensionalises them when present.
- Tests in `tests/test_gaussian_backend.py`: self-consistency (filter == sim mean to
  numerical precision), convergence from a wrong initial mean, and innovation
  reconstruction matching the simulator's RNG-drawn `dW` sequence. All 15 tests pass.

### May 7, 2026: Minimal Gaussian backend skeleton (continued)
- **Deferred** (out of scope for this slice): `H_ext(t)` callable,
  feedback / parametric drive, mechanical damping + thermal-bath terms in
  variance EOMs, multi-trajectory batching, separate SDE integrator module,
  plotting helpers, PSD-of-photocurrent test.
