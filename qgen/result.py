from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qgen.state import GaussianState


@dataclass(frozen=True)
class Result:
    times: NDArray[np.float64]
    xc: NDArray[np.float64]
    pc: NDArray[np.float64]
    photocurrent: NDArray[np.float64]
    Vxx: NDArray[np.float64]
    Vpp: NDArray[np.float64]
    Cxp: NDArray[np.float64]
    meta: dict = field(default_factory=dict)

    def gaussian_state(self, t_idx: int) -> GaussianState:
        mean = np.array([self.xc[t_idx], self.pc[t_idx]])
        cov = np.array(
            [
                [self.Vxx[t_idx], self.Cxp[t_idx]],
                [self.Cxp[t_idx], self.Vpp[t_idx]],
            ]
        )
        return GaussianState(mean=mean, cov=cov)
