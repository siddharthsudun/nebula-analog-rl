"""Render the eye diagram: channel-only (closed) vs channel + CTLE + 1-tap DFE (open).

The single most legible proof that the equalizer works. Saves results/eye.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eqrl.circuits.ctle import DesignVars
from eqrl.sim.eye import EyeResult, compute_eye
from eqrl.sim.server import get_server


def _centered(res: EyeResult) -> np.ndarray:
    """Roll each UI trace so the optimal sampling instant sits at the centre."""
    M = res.samples_per_ui
    shift = M // 2 - res.sample_phase
    return np.roll(res.eye_matrix, shift, axis=1)


def render(design_path: str, out: str = "results/eye.png", loss_db: float = 12.0,
           n_traces: int = 500) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dv = DesignVars(**json.load(open(design_path)))
    srv = get_server("tt")
    H = srv.ac_complex(dv)
    flat = np.ones_like(H["H"])

    eq = compute_eye(H["freq"], H["H"], channel_loss_db=loss_db)
    raw = compute_eye(H["freq"], flat, channel_loss_db=loss_db, dfe_taps=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    M = eq.samples_per_ui
    x = (np.arange(M) - M / 2) / M    # UI, centred
    for ax, res, title, col in [
        (axes[0], raw, f"Channel only  ·  {loss_db:.4g} dB loss", "#c0603a"),
        (axes[1], eq, "Channel + CTLE + 1-tap DFE (SILQ)", "#17b7a8"),
    ]:
        E = _centered(res)
        for row in E[:n_traces]:
            ax.plot(x, row * 1e3, color=col, alpha=0.05, linewidth=0.8)
        ax.axhline(0, color="#888", lw=0.6, ls=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("time (UI)")
        ax.text(0.02, 0.04, f"H = {res.height_v*1e3:.0f} mV\nW = {res.width_ui:.2f} UI",
                transform=ax.transAxes, fontsize=10, family="monospace",
                va="bottom", color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))
    axes[0].set_ylabel("differential (mV)")
    fig.suptitle("PCIe Gen2 receiver eye — 5 Gb/s NRZ, SKY130", fontsize=12, y=1.02)
    fig.tight_layout()
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"eye -> {out}   (no-EQ {raw.height_v*1e3:.0f}mV/{raw.width_ui:.2f}UI  →  "
          f"SILQ {eq.height_v*1e3:.0f}mV/{eq.width_ui:.2f}UI)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--design", default="results/rl_design_pvt.json")
    p.add_argument("--out", default="results/eye.png")
    p.add_argument("--loss", type=float, default=12.0)
    args = p.parse_args()
    render(args.design, args.out, args.loss)


if __name__ == "__main__":
    main()
