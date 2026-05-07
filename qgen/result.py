from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


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
