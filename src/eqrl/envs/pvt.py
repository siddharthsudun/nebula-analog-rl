"""PVT corner evaluation — the robustness differentiator.

A design that meets spec at TT often breaks at SS/FF or temperature/voltage extremes.
`worst_corner` evaluates a design across the corner set and returns the WORST-performing
corner's measurements, so the RL reward optimizes worst-case robustness, not nominal.

Corner reload cost: each process corner is a separate resident server (models differ),
so the first visit to a corner pays the ~15 s model-load once, then caches.
"""
from __future__ import annotations

from eqrl.circuits.ctle import DesignVars
from eqrl.sim.measures import Measures, measure_all
from eqrl.sim.server import get_server
from eqrl.specs import Spec


def corner_grid(spec: Spec, mode: str = "reduced") -> list[tuple[str, float, float]]:
    """(process, vdd, temp_c) tuples to evaluate.

    mode="reduced": the stress corners that dominate failure (fast, for training).
    mode="full": the complete PVT cross-product (for final sign-off).
    """
    vlo, vnom, vhi = spec.vdd_corners()
    if mode == "reduced":
        return [
            ("tt", vnom, 27.0),
            ("ss", vlo, 125.0),   # slow/low-V/hot: worst speed & headroom
            ("ff", vhi, 0.0),     # fast/high-V/cold: worst linearity & power
        ]
    grid = []
    for proc in spec.process_corners:
        for v in spec.vdd_corners():
            for t in spec.temps_c:
                grid.append((proc, v, t))
    return grid


def evaluate_corners(dv: DesignVars, spec: Spec, mode: str = "reduced",
                     fast: bool = True) -> dict[tuple, Measures]:
    """Measure a design at every corner in the grid."""
    results = {}
    for proc, vdd, temp in corner_grid(spec, mode):
        srv = get_server(proc)
        results[(proc, vdd, temp)] = measure_all(
            dv, vdd=vdd, temp_c=temp, corner=proc, fast=fast, srv=srv)
    return results


def worst_corner(dv: DesignVars, spec: Spec, mode: str = "reduced",
                 fast: bool = True) -> Measures:
    """Return the measurements of the worst corner, by reward.

    Worst = the corner whose reward is lowest (imported lazily to avoid a cycle).
    Any non-convergent corner immediately dominates as the worst (ok=False).
    """
    from eqrl.envs.equalizer_env import compute_reward

    results = evaluate_corners(dv, spec, mode, fast)
    worst_m, worst_r = None, 1e18
    for m in results.values():
        if not m.ok:
            return m
        r, _, _ = compute_reward(m, spec)
        if r < worst_r:
            worst_r, worst_m = r, m
    return worst_m if worst_m is not None else Measures(ok=False)
