# q-gen

## Table of Contents

1. [Goals](#1-goals)
2. [Physics & Numerics](#2-physics--numerics)
3. [Architecture Overview](#3-architecture-overview)
4. [Repository Layout](#4-repository-layout)
5. [Implementation Roadmap](#5-implementation-roadmap)

## 1. Goals

Build an event generator for a continuously-measured levitated nanoparticle.
Given a physical model and measurement channel, q-gen simulates the
conditional evolution of the particle's density matrix and emits a
measurement record as a function of time.

The system should:

- Represent the particle's state through its density matrix, with the option to
  evolve only its first two moments when the state and channel are Gaussian
- Support a Gaussian backend (fast, exact for linear systems with Gaussian
  noise) and a full stochastic master equation (SME) backend
- Begin with a single mechanical mode in 1D, with an architecture that
  generalises to 3D and multi-mode systems
- Allow extra Hamiltonian terms (possibly time-dependent) to be added on
  top of the base optomechanical model — e.g. effect of gas collisions, frequency drift
- Make measurement-conditioned feedback straightforward to add as a later
  extension
- Output dimensional measurement records (and other observables) while
  computing internally in dimensionless units
- Eventually be exposed through a browser-based interface (Python backend +
  JS frontend), with a clean programmatic API in the meantime

## 2. Physics & Numerics

### 2.1 The system

A single mechanical mode of a levitated nanoparticle, weakly and
continuously measured (homodyne/heterodyne detection of light scattered
or transmitted by the particle). The conditional state ρ_c(t) obeys a
stochastic master equation of the form

```
dρ_c = -i[H, ρ_c] dt
       + Σ_k D[c_k] ρ_c dt              (unconditional dissipators)
       + H[c_m] ρ_c dW_t                 (measurement backaction)
```

where `D[c]ρ = cρc† - ½{c†c, ρ}` is the Lindblad dissipator,
`H[c]ρ = cρ + ρc† - Tr[(c+c†)ρ] ρ` is the measurement superoperator, and
`dW_t` is a Wiener increment. The measurement record is

```
dY_t = Tr[(c_m + c_m†) ρ_c] dt + dW_t
```

scaled appropriately to give a photocurrent in physical units.

`H` decomposes as `H = H_0 + H_ext(t)`, where `H_0` is the bare mechanical
Hamiltonian and `H_ext(t)` collects user-supplied extra terms (drives,
feedback, etc.).

### 2.2 The Gaussian regime

When `H` is at most quadratic in `(x, p)` and all `c_k`, `c_m` are linear
in `(x, p)`, the conditional state stays Gaussian and is fully described
by its mean vector `μ = (⟨x⟩, ⟨p⟩)` and covariance matrix
`V = ⟨{Δr, Δr^T}⟩/2`. The SME reduces to a closed pair of equations:

```
dμ = A μ dt + B u(t) dt + (V C^T + Γ^T) dW_t
dV/dt = A V + V A^T + D - (V C^T + Γ^T)(C V + Γ)
```

(a Kalman–Bucy filter). The covariance equation is deterministic; only
the mean is stochastic. This is the fast path.

### 2.3 The full SME

When the model is non-Gaussian (e.g. a quartic potential, or jump-like
collision events), the full ρ must be propagated. We can figure out 
different ways to solve this problem.

### 2.4 Units

Internally, time is in units of `1/ω_m`, position in units of the
zero-point motion `x_zpf = √(ħ/2mω_m)`, and momentum in units of
`p_zpf = √(ħmω_m/2)`. Model parameters are supplied with units (via 
a small in-house unit handler), converted to dimensionless form
at model-build time, and outputs are re-dimensionalised before being
returned to the user.


### 2.5 Backends and the model abstraction

A `Model` object holds:

- The base Hamiltonian `H_0` (specified symbolically or as operator data)
- A list of dissipators `c_k`
- A measurement operator `c_m` with a detection efficiency `η`
- An optional `H_ext(t)` callable
- Physical parameters with units

A `Backend` consumes a `Model` and produces trajectories. Two backends
are planned:

- `GaussianBackend` — checks the model is quadratic + linear; integrates
  the moment equations.
- `SMEBackend` — represents ρ and integrates the full SME via various methods.

The same `Model` can be run under either backend (when both apply),
which is also a useful correctness check.

## 3. Architecture Overview

### 3.1 Layers

```
       ┌────────────────────────────────────────────┐
       │  Frontend (deferred — JS + Python server)  │
       └────────────────────────────────────────────┘
                          │
       ┌────────────────────────────────────────────┐
       │  Programmatic API   (qgen.simulate(...))   │
       └────────────────────────────────────────────┘
                          │
       ┌──────────────┬───────────────┬─────────────┐
       │   Model      │   Backends    │   Output    │
       │  (physics)   │  (numerics)   │ (records,   │
       │              │               │  rescaling) │
       └──────────────┴───────────────┴─────────────┘
                          │
       ┌────────────────────────────────────────────┐
       │  NumPy / SciPy   (later: QuTiP optional)   │
       └────────────────────────────────────────────┘
```

### 3.2 Programmatic API (sketch)

```python
import qgen
from qgen import Model, GaussianBackend
import qgen.units as u

model = Model(
    omega_m = 2*np.pi * 100e3 * u.Hz,
    mass    = 1e-18 * u.kg,
    gamma   = 2*np.pi * 10 * u.Hz,        # measurement rate
    eta     = 0.5,                         # detection efficiency
    T_bath  = 300 * u.K,
)

backend = GaussianBackend(model)

result = backend.simulate(
    t_span = (0, 1e-3) * u.s,
    dt     = 1e-7 * u.s,
    n_traj = 4,
    seed   = 0,
)

result.t            # times, with units
result.photocurrent # shape (n_traj, n_steps), with units
result.mean         # shape (n_traj, n_steps, 2), with units
result.cov          # shape (n_steps, 2, 2), with units
```

A `Model` can be extended with extra Hamiltonian terms:

```python
model.add_hamiltonian(lambda t, x, p: F_drive(t) * x)
```

### 3.3 Trajectory object

A `Trajectory` (or `Result`) bundles:

- `t` — time array (with units)
- `photocurrent` — measurement record `dY_t / dt`, scaled to physical units
- `mean`, `cov` — conditional moments (Gaussian backend) or expectation
  values + variances reconstructed from ρ (SME backend)
- `meta` — model parameters, backend, seed, integrator settings

Persistence (HDF5/Zarr) is deferred; results live in memory for now.

## 4. Repository Layout

```
q-gen/
├── DESIGN.md                  # This document
├── README.md
├── pyproject.toml
├── qgen/
│   ├── __init__.py            # Public API re-exports
│   ├── units.py               # Unit registry and rescaling helpers
│   ├── model.py               # Model class: H_0, dissipators, c_m, H_ext
│   ├── operators.py           # x, p, a, a† builders in Fock basis
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py            # Backend abstract base class
│   │   └── gaussian.py        # Kalman–Bucy moment-equation backend
│   ├── integrators/
│   │   ├── __init__.py
│   │   └── sde.py             # Euler–Maruyama, Milstein
│   ├── result.py              # Trajectory / Result container
│   └── plotting.py            # Convenience plots (matplotlib)
├── examples/
│   └── single_mode_gaussian.py
└── tests/
    ├── test_units.py
    ├── test_gaussian_backend.py
    └── test_steady_state.py
```

## 5. Implementation Roadmap

### Phase 1 — Foundation (Gaussian, single mode)

| Task | Files | Deliverable |
|------|-------|-------------|
| Package setup | `pyproject.toml`, `qgen/__init__.py` | Installable package |
| Units layer | `qgen/units.py` | SI-in / dimensionless-internal / SI-out conversions |
| Model class | `qgen/model.py` | Single-mode optomechanical model with optional `H_ext(t)` |
| SDE integrator | `qgen/integrators/sde.py` | Euler–Maruyama with seed control |
| Gaussian backend | `qgen/backends/gaussian.py` | Conditional moment evolution + photocurrent |
| Result container | `qgen/result.py` | Trajectory bundle with units |
| Example + plot | `examples/single_mode_gaussian.py`, `qgen/plotting.py` | End-to-end demo: parameters → photocurrent plot |
| Validation | `tests/` | Steady-state variance matches analytic Kalman result; PSD of photocurrent matches expected Lorentzian + shot noise |

### Phase 2 — Processes & feedback

- A `Process` interface for adding contributions to the SME (extra
  dissipators, jump operators for collisions, time-dependent `H_ext`)
- Gas-collision model (initially: Brownian-motion limit as an extra
  thermal dissipator; later: Poisson jump events)
- Measurement-conditioned feedback: `H_ext` may depend on a causal
  functional of the photocurrent (e.g. filtered estimate of `μ`)
- 3D / multi-mode generalisation

### Phase 3— Web interface

- FastAPI (or similar) server wrapping the programmatic API
- Streaming endpoints (WebSocket) for trajectory chunks
- JS frontend with parameter controls and live plotting
- Persistence layer (HDF5 or Zarr) for saved runs

### Phase 4— Full SME backend
- TBD
