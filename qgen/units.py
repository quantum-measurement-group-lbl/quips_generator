from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qgen.model import HBAR, Model
from qgen.result import Result

__all__ = ["HBAR", "DimensionalResult", "to_si"]


@dataclass(frozen=True)
class DimensionalResult:
    """SI-units mirror of `Result`.

    Units:
      times [s], xc [m], pc [kg*m/s],
      Vxx [m^2], Vpp [(kg*m/s)^2], Cxp [m * kg*m/s],
      photocurrent [m/s] (homodyne current; signal part is x(t)*omega).
    """

    times: NDArray[np.float64]
    xc: NDArray[np.float64]
    pc: NDArray[np.float64]
    photocurrent: NDArray[np.float64]
    Vxx: NDArray[np.float64]
    Vpp: NDArray[np.float64]
    Cxp: NDArray[np.float64]
    xc_kalman: NDArray[np.float64] | None = None
    pc_kalman: NDArray[np.float64] | None = None
    meta: dict = field(default_factory=dict)


def to_si(result: Result, model: Model) -> DimensionalResult:
    """Re-dimensionalise a `Result` (computed in dimensionless internal units)
    using a `Model` for the physical scales.
    """
    x_zpf = model.x_zpf
    p_zpf = model.p_zpf
    omega = model.omega

    xc_kalman = result.xc_kalman * x_zpf if result.xc_kalman is not None else None
    pc_kalman = result.pc_kalman * p_zpf if result.pc_kalman is not None else None

    return DimensionalResult(
        times=result.times / omega,
        xc=result.xc * x_zpf,
        pc=result.pc * p_zpf,
        photocurrent=result.photocurrent * x_zpf * omega,
        Vxx=result.Vxx * x_zpf**2,
        Vpp=result.Vpp * p_zpf**2,
        Cxp=result.Cxp * x_zpf * p_zpf,
        xc_kalman=xc_kalman,
        pc_kalman=pc_kalman,
        meta={**result.meta, "units": "SI"},
    )
