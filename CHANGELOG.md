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
- **Deferred** (out of scope for this slice): units layer, `H_ext(t)` callable,
  feedback / parametric drive, mechanical damping + thermal-bath terms in
  variance EOMs, multi-trajectory batching, separate SDE integrator module,
  plotting helpers, PSD-of-photocurrent test.
