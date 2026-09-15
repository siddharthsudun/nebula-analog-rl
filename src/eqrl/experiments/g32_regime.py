"""Regime-aware peak-axis calibration for a *new* G3.2 comparison arm.

The frozen G3.2 controller remains in :mod:`final_comparison` and is intentionally not
edited here.  Its peak coordinate (``l_in``) was measured on designs above the allowed
peak band.  The low-peak probe later found that, below the band, ``r_load`` moves the
peak reliably while ``l_in`` is commonly below measurement resolution.

This module turns that observation into a bounded, auditable experiment:

* retain ``rs`` and every frozen G3.2 decision unchanged;
* when the selected handoff peak is outside the band, first probe the coordinate measured
  for that regime (``r_load`` below, ``l_in`` above);
* use its locally measured slope only when it clears the existing information floor;
* otherwise try the other coordinate, then fall back to the frozen plane.

The probe cost comes out of the same stage-two budget.  This is deliberately a new arm,
not a modification of the frozen path or a claim about its historical result.
"""
from __future__ import annotations

import math

import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE
from eqrl.experiments.g32_selfcal import (MIN_ABS_D_LN_PEAK, PROBE_H, _probe_axis)

DIMS = list(ACTION_SPACE.keys())


def probe_start(xs, s1trace, target: float) -> tuple[np.ndarray, dict | None, str]:
    """Choose the same starting point the frozen solver would choose before repair.

    The calibration must be taken at the design the comparison arm will actually repair,
    not automatically at the final PPO handoff.  Guard-invalid handoffs have no usable
    AC measurement, so they retain the frozen plane and use the frozen rescue branch.
    """
    feasible = [(abs(e["boost_db"] - target), j) for j, e in enumerate(s1trace)
                if e and e["loose_pass"]]
    if feasible:
        j = min(feasible)[1]
        return np.asarray(xs[j], dtype=np.float64), s1trace[j], "best stage-1 feasible"
    valid = [(len(e["failing"]), abs(e["boost_db"] - target), j)
             for j, e in enumerate(s1trace) if e]
    if valid:
        j = min(valid)[2]
        return np.asarray(xs[j], dtype=np.float64), s1trace[j], "fewest failing checks"
    return np.asarray(xs[-1], dtype=np.float64), None, "guard-invalid handoff"


def regime_plane(evaluate, x0, target: float, frozen: dict, *, base_rec: dict | None,
                 h: float = PROBE_H, budget: int | None = None) -> tuple[dict, int, dict]:
    """Return a locally measured peak-coordinate plane, or the frozen plane.

    ``evaluate`` has the same contract as ``final_comparison.g32_solve``.  The returned
    plane is a drop-in for that solver; only peak-axis metadata and peak-related slopes
    can change.  One-sided finite differences reuse the stage-one measurement and normally
    cost one evaluation.  A rejected or uninformative preferred axis can consume a second
    side and then try the alternate axis, never exceeding ``budget``.
    """
    plane = dict(frozen)
    spent = 0
    info: dict = {"evaluations": 0, "regime": None, "candidates": [],
                  "selected_axis": frozen["peak_axis"], "selected": False,
                  "reason": None, "notes": {}}

    if base_rec is None or not base_rec.get("peak_freq_ghz"):
        info["reason"] = "no guard-valid handoff to probe; retained frozen plane"
        return plane, spent, info

    lo, hi = frozen["band_ghz"]
    peak = float(base_rec["peak_freq_ghz"])
    if lo <= peak <= hi:
        info.update(regime="in_band", reason="peak already in band; retained frozen plane")
        return plane, spent, info

    # The order is measured, not tuned: g32a_lowpeak_report established r_load below the
    # band, while g32_peak_probe established l_in above it.  The alternate is attempted
    # only if the preferred coordinate carries no usable local peak information.
    if peak < lo:
        regime, candidates = "below_band", ("r_load", "l_in")
    else:
        regime, candidates = "above_band", ("l_in", "r_load")
    info["regime"] = regime

    def take(x):
        nonlocal spent
        if budget is not None and spent >= budget:
            return None
        spent += 1
        rec, _score, _guard = evaluate(x, target)
        return rec

    chosen = None
    for axis in candidates:
        if budget is not None and spent >= budget:
            info["candidates"].append(axis + ": no budget")
            continue
        slope, notes = _probe_axis(take, np.asarray(x0, dtype=np.float64),
                                   DIMS.index(axis), h, base_rec, two_sided=False)
        info["notes"][axis] = notes
        gain = None if slope is None else slope.get("d_ln_peak")
        if gain is None or not math.isfinite(gain) or abs(gain) < MIN_ABS_D_LN_PEAK:
            info["candidates"].append(axis + ": below information floor")
            continue
        chosen = (axis, slope)
        info["candidates"].append(axis + ": selected")
        break

    info["evaluations"] = spent
    if chosen is None:
        info["reason"] = "no locally measurable peak axis; retained frozen plane"
        return plane, spent, info

    axis, slope = chosen
    plane.update({
        "peak_axis": axis,
        "d_peak_ghz_per_unit": float(slope["d_peak"]),
        "d_ln_peak_per_unit": float(slope["d_ln_peak"]),
        "peak_axis_d_boost_db_per_unit": float(slope["d_boost"]),
        "source": "runtime regime-aware peak-axis calibration (fallback: %s)"
                  % frozen.get("source", "frozen plane"),
        "n_specs_measured": 1,
        "spec_seed": None,
    })
    plane["regime_axis"] = {"regime": regime, "axis": axis, "h": h,
                            "evaluations": spent}
    info.update(selected_axis=axis, selected=True,
                reason="local %s slope cleared information floor" % axis)
    return plane, spent, info
