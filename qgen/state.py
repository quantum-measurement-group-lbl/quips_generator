from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianState:
    """Single-mode Gaussian state at one instant.

    `mean` is `[x, p]`. `cov` is the 2x2 symmetric covariance in the qgen
    convention (vacuum has `cov = I`, i.e. 2x the standard hbar=1 covariance).
    """

    mean: NDArray[np.float64]
    cov: NDArray[np.float64]
