"""FastAPI app: serves the qgen browser visualizer.

Run with:  uv run uvicorn viz.server:app --reload --port 8000
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from qgen import GaussianBackend, Model, to_si, wigner

N_PERIODS = 30
DT = 0.01
SEED = 0
N_GRID = 121
N_FRAMES = 200
N_SIGMA = 4.0


class SimulateRequest(BaseModel):
    omega_hz: float = Field(gt=0)
    mass_kg: float = Field(gt=0)
    gamma_ba_hz: float = Field(gt=0)
    eta: float = Field(gt=0, le=1.0)


@dataclass
class RunCache:
    run_id: str
    params: dict
    times_ms: list[float]
    xc_pm: list[float]
    pc: list[float]
    xc_kalman_pm: list[float]
    pc_kalman: list[float]
    frame_indices: NDArray[np.int64]
    frame_time_ms: list[float]
    x_grid_pm: list[float]
    p_grid: list[float]
    wigner_min: float
    wigner_max: float
    frames: NDArray[np.float64]  # (n_frames, N_GRID, N_GRID)


def run_simulation(omega_hz: float, mass_kg: float, gamma_ba_hz: float, eta: float) -> RunCache:
    model = Model(
        mass=mass_kg,
        omega=2 * pi * omega_hz,
        gamma_ba=2 * pi * gamma_ba_hz,
        eta=eta,
    )
    backend = GaussianBackend(model)
    result = backend.simulate(n_periods=N_PERIODS, dt=DT, seed=SEED)
    result = backend.kalman_filter(result)
    si = to_si(result, model)

    sigma_x = float(np.max(np.sqrt(result.Vxx)))
    sigma_p = float(np.max(np.sqrt(result.Vpp)))
    x_lo = float(np.min(result.xc)) - N_SIGMA * sigma_x
    x_hi = float(np.max(result.xc)) + N_SIGMA * sigma_x
    p_lo = float(np.min(result.pc)) - N_SIGMA * sigma_p
    p_hi = float(np.max(result.pc)) + N_SIGMA * sigma_p
    x_grid = np.linspace(x_lo, x_hi, N_GRID)
    p_grid = np.linspace(p_lo, p_hi, N_GRID)

    n = len(result.times)
    frame_indices = np.linspace(0, n - 1, N_FRAMES, dtype=np.int64)
    frames = np.empty((N_FRAMES, N_GRID, N_GRID), dtype=np.float64)
    for k, idx in enumerate(frame_indices):
        state = result.gaussian_state(int(idx))
        # wigner returns shape (len(x), len(p)); transpose so axis 0 = p (heatmap rows)
        frames[k] = wigner(state, x_grid, p_grid).T

    x_grid_pm = (x_grid * model.x_zpf * 1e12).tolist()
    p_grid_si = (p_grid * model.p_zpf).tolist()

    times_ms = (si.times * 1e3).tolist()
    return RunCache(
        run_id=str(uuid.uuid4()),
        params={
            "omega_hz": omega_hz,
            "mass_kg": mass_kg,
            "gamma_ba_hz": gamma_ba_hz,
            "eta": eta,
        },
        times_ms=times_ms,
        xc_pm=(si.xc * 1e12).tolist(),
        pc=si.pc.tolist(),
        xc_kalman_pm=(si.xc_kalman * 1e12).tolist(),
        pc_kalman=si.pc_kalman.tolist(),
        frame_indices=frame_indices,
        frame_time_ms=[times_ms[int(i)] for i in frame_indices],
        x_grid_pm=x_grid_pm,
        p_grid=p_grid_si,
        wigner_min=float(frames.min()),
        wigner_max=float(frames.max()),
        frames=frames,
    )


def _payload(run: RunCache) -> dict:
    return {
        "run_id": run.run_id,
        "params": run.params,
        "times_ms": run.times_ms,
        "xc_pm": run.xc_pm,
        "pc": run.pc,
        "xc_kalman_pm": run.xc_kalman_pm,
        "pc_kalman": run.pc_kalman,
        "frame_time_ms": run.frame_time_ms,
        "x_grid_pm": run.x_grid_pm,
        "p_grid": run.p_grid,
        "wigner_min": run.wigner_min,
        "wigner_max": run.wigner_max,
        "n_frames": len(run.frame_indices),
    }


_DEFAULT_MODEL = Model()
current_run: RunCache = run_simulation(
    omega_hz=_DEFAULT_MODEL.omega / (2 * pi),
    mass_kg=_DEFAULT_MODEL.mass,
    gamma_ba_hz=_DEFAULT_MODEL.gamma_ba / (2 * pi),
    eta=_DEFAULT_MODEL.eta,
)


app = FastAPI()


@app.get("/api/data")
def get_data() -> dict:
    return _payload(current_run)


@app.get("/api/wigner/{frame_idx}")
def get_wigner(frame_idx: int, run_id: str) -> dict:
    if run_id != current_run.run_id:
        raise HTTPException(status_code=409, detail="stale run_id")
    if frame_idx < 0 or frame_idx >= len(current_run.frame_indices):
        raise HTTPException(status_code=404, detail="frame out of range")
    return {"W": current_run.frames[frame_idx].tolist()}


@app.post("/api/simulate")
def post_simulate(req: SimulateRequest) -> dict:
    global current_run
    current_run = run_simulation(
        omega_hz=req.omega_hz,
        mass_kg=req.mass_kg,
        gamma_ba_hz=req.gamma_ba_hz,
        eta=req.eta,
    )
    return _payload(current_run)


_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
