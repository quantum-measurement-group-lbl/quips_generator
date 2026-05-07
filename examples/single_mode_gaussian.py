"""End-to-end demo: SI parameters -> Gaussian-backend trajectory -> SI plot.

Run with:  uv run python examples/single_mode_gaussian.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from qgen import GaussianBackend, Model, to_si


def n_bar(xc, pc, Vx, Vp):
    return (xc**2 + Vx + pc**2 + Vp) / 4.0 - 0.5


def main():
    model = Model()  # SI defaults: m=5fg, omega/2pi=50kHz, Gamma_BA/2pi=1kHz, eta=1
    backend = GaussianBackend(model)
    res = backend.simulate(n_periods=30, dt=0.01, seed=0)
    si = to_si(res, model)

    n = len(res.times)
    half = n // 2
    n_bar_steady = float(
        np.mean(n_bar(res.xc[half:], res.pc[half:], res.Vxx[half:], res.Vpp[half:]))
    )
    print(f"x_zpf = {model.x_zpf:.3e} m   p_zpf = {model.p_zpf:.3e} kg*m/s")
    print(f"gamma_meas_dim = Gamma_BA/omega = {model.gamma_meas_dim:.4f}")
    print(f"n_bar (second half mean): {n_bar_steady:.4f}")
    print(
        f"Vxx[-1]={res.Vxx[-1]:.4f}  Vpp[-1]={res.Vpp[-1]:.4f}  Cxp[-1]={res.Cxp[-1]:.4f}"
    )

    out_dir = Path(__file__).parent / "_out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "single_mode.png"

    times_ms = si.times * 1e3
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times_ms, si.xc * 1e12, lw=0.6)
    axes[0].set_ylabel(r"$\langle x \rangle$ [pm]")
    axes[1].plot(times_ms, si.pc, lw=0.6)
    axes[1].set_ylabel(r"$\langle p \rangle$ [kg$\cdot$m/s]")
    axes[2].plot(times_ms, si.photocurrent, lw=0.3)
    axes[2].set_ylabel("photocurrent [m/s]")
    axes[2].set_xlabel("time [ms]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
