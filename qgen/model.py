from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    omega: float = 1.0
    quality_factor: float = 1e4
    gamma_meas: float = 5e-2
    eta: float = 1.0
    n_thermal: float = 100.0

    @property
    def gamma(self) -> float:
        return self.omega / self.quality_factor

    @property
    def k_jacobs(self) -> float:
        return self.gamma_meas * self.omega / 2
