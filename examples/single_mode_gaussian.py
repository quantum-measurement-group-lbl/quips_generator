"""End-to-end demo: parameters -> Gaussian-backend trajectory -> plot.

Run with:  uv run python examples/single_mode_gaussian.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from qgen import GaussianBackend, Model


def n_bar(xc, pc, Vx, Vp):
    return (xc**2 + Vx + pc**2 + Vp) / 4.0 - 0.5


def main():
    model = Model(omega=1.0, gamma_meas=5e-2, eta=1.0, n_thermal=100.0)
    backend = GaussianBackend(model)
    res = backend.simulate(n_periods=30, dt=0.01, seed=0)

    n = len(res.times)
    half = n // 2
    n_bar_steady = float(
        np.mean(n_bar(res.xc[half:], res.pc[half:], res.Vxx[half:], res.Vpp[half:]))
    )
    print(f"n_bar (second half mean): {n_bar_steady:.4f}")
    print(f"Vxx[-1]={res.Vxx[-1]:.4f}  Vpp[-1]={res.Vpp[-1]:.4f}  Cxp[-1]={res.Cxp[-1]:.4f}")

    out_dir = Path(__file__).parent / "_out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "single_mode.png"

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(res.times, res.xc, lw=0.6)
    axes[0].set_ylabel(r"$\langle x \rangle$")
    axes[1].plot(res.times, res.pc, lw=0.6)
    axes[1].set_ylabel(r"$\langle p \rangle$")
    axes[2].plot(res.times, res.photocurrent, lw=0.3)
    axes[2].set_ylabel("photocurrent")
    axes[2].set_xlabel(r"time $[\,1/\omega\,]$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
