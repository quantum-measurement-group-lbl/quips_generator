from dataclasses import dataclass
from math import pi, sqrt

HBAR = 1.054_571_817e-34  # J*s


@dataclass(frozen=True)
class Model:
    """Single-mode optomechanical model in SI units.

    Internal calculations use dimensionless quantities exposed through the
    `*_dim` properties (time in 1/omega, position in x_zpf, momentum in p_zpf).
    """

    mass: float = 5e-18                 # kg   (5 fg)
    omega: float = 2 * pi * 50e3        # rad/s
    gamma_ba: float = 2 * pi * 1e3      # rad/s, backaction rate Gamma_BA
    eta: float = 1.0                    # detection efficiency
    n_thermal: float = 100.0            # initial thermal occupation
    quality_factor: float = 1e4         # for future damping term

    @property
    def omega_dim(self) -> float:
        return 1.0

    @property
    def gamma_meas_dim(self) -> float:
        return self.gamma_ba / self.omega

    @property
    def x_zpf(self) -> float:
        return sqrt(HBAR / (2.0 * self.mass * self.omega))

    @property
    def p_zpf(self) -> float:
        return sqrt(HBAR * self.mass * self.omega / 2.0)

    @property
    def gamma(self) -> float:
        return self.omega / self.quality_factor

    @property
    def k_jacobs(self) -> float:
        return self.gamma_meas_dim * self.omega_dim / 2.0
