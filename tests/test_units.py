from math import pi, sqrt

import numpy as np

from qgen import GaussianBackend, Model, to_si
from qgen.model import HBAR


def test_default_zpf_scales():
    m = Model()
    expected_x_zpf = sqrt(HBAR / (2.0 * m.mass * m.omega))
    expected_p_zpf = sqrt(HBAR * m.mass * m.omega / 2.0)
    assert abs(m.x_zpf - expected_x_zpf) / expected_x_zpf < 1e-12
    assert abs(m.p_zpf - expected_p_zpf) / expected_p_zpf < 1e-12
    # sanity: 5fg @ 50kHz gives x_zpf in the ~1e-12 m range
    assert 1e-13 < m.x_zpf < 1e-11, f"ERROR x_zpf out of expected range: {m.x_zpf:.3e}"


def test_default_dimensionless_rates():
    m = Model()
    # gamma_meas_dim = Gamma_BA / omega = (2*pi*1e3) / (2*pi*50e3) = 0.02
    assert abs(m.gamma_meas_dim - 0.02) < 1e-12
    assert m.omega_dim == 1.0


def test_custom_si_inputs():
    m = Model(mass=1e-18, omega=2 * pi * 100e3, gamma_ba=2 * pi * 10.0, eta=0.5)
    assert abs(m.gamma_meas_dim - 1e-4) < 1e-15
    assert m.eta == 0.5


def test_to_si_round_trip():
    m = Model()
    backend = GaussianBackend(m)
    res = backend.simulate(n_periods=1, dt=0.05, seed=0)
    si = to_si(res, m)
    rel = np.max(np.abs(si.xc / m.x_zpf - res.xc) / (np.abs(res.xc) + 1e-300))
    assert rel < 1e-12, f"ERROR xc round-trip rel error {rel:.3e}"
    rel_t = np.max(np.abs(si.times * m.omega - res.times) / (np.abs(res.times) + 1e-300))
    assert rel_t < 1e-12, f"ERROR times round-trip rel error {rel_t:.3e}"
    rel_v = np.max(np.abs(si.Vxx / m.x_zpf**2 - res.Vxx) / (np.abs(res.Vxx) + 1e-300))
    assert rel_v < 1e-12, f"ERROR Vxx round-trip rel error {rel_v:.3e}"
