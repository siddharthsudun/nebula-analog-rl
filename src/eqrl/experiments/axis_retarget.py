"""Second-axis retargeting: what to do when G3.2's boost axis is pinned at a bound.

THE MEASURED PROBLEM (docs/DESIGN_MODES_V2.md section 6). `g32_solve`'s stage B is a
ONE-DIMENSIONAL line search along `plane["boost_axis"]`, which is `rs`. When the handoff
design already sits at `rs = 1.0`, the very first trial step clips to nothing, the loop
exits immediately with `reason = "no admissible step remains"`, and the solver returns
with its ENTIRE budget unspent. On spec 11.0/12.5 that is 1.936 dB of error for zero
evaluations out of ten.

That is not a tuning failure and no mode preset touches it: budget, stop tolerance,
restart count and guard relaxation were all measured against that spec and all left it at
exactly 1.936 dB. Only a different axis can move it. `scratchpad/axis_probe.py` stepped
every axis from that stuck design:

    cs     +0.08   boost 10.230   err 1.936 -> 0.770   feasible   <-- one evaluation
    w_in   +0.20   boost  9.784   err        -> 1.216   feasible
    i_tail -0.08   boost  9.546   err        -> 1.454   feasible
    rs     -0.08   boost  8.593   err        -> 2.407   feasible   (the only axis tried today)

`cs` cuts the error by 60% in a single evaluation and the solver never tries it. That is
the right physics: `rs * cs` sets the CTLE zero, so with `rs` railed, `cs` is the
remaining degeneration control.

WHY THIS IS A NEW FILE AND NOT A CHANGE TO `g32_solve`. `final_comparison.g32_solve` is a
transcribed, frozen controller -- see that module's docstring. It is called here TWICE,
completely unmodified, and no branch, threshold or step inside it is touched. The only new
thing is what happens BETWEEN the two calls, which lives here, in its own clearly-labelled
file, exactly as `fastest_hedge` does for Fastest mode.

HOW THE SECOND CALL SEARCHES A DIFFERENT AXIS WITHOUT EDITING THE SOLVER. `g32_solve`
reads its boost coordinate and that coordinate's slope out of the `plane` dict it is
handed. So the retarget builds a DERIVED plane whose `boost_axis` is the newly chosen axis
and whose `d_boost_db_per_unit` is the slope MEASURED on the probe step that chose it --
which is precisely what that field means. The peak coordinate is left alone. Passing a
different plane is using the solver's own interface, not reaching inside it.

    THE SLOPE IS MEASURED, NOT ASSUMED. `plane_from_probe`'s slopes are medians over a
    committed probe artifact; this one is a local finite difference taken on THIS design,
    one evaluation ago. It is used only to pick the search DIRECTION and the first step
    size -- `g32_solve` re-measures at every step and brackets on real numbers, so a wrong
    slope costs step efficiency, never correctness.

THE SURROGATE RANKS, THE EVALUATOR JUDGES. Candidate (axis, step) pairs are ordered by the
kNN surrogate at ~7 microseconds each, then the top few are spent on REAL guarded
evaluations. This matters because `eqrl.surrogate` explicitly does NOT predict guard
validity -- and guard validity, not boost, is the binding constraint here: of the twenty
probe steps measured above, nine were rejected outright by the guard. So the surrogate may
only propose an ORDER; every candidate that survives has been through the same guarded
`fast=False` evaluator every other arm uses. If the surrogate is unavailable the ranking
falls back to a fixed axis order and the mechanism is otherwise identical.

WHAT THIS IS NOT. It is not a new optimizer, it does not touch the reward, the policy, the
observation space or any guard, and it is not part of `default`. It ships as its own arm
under its own name and must be reported on a fresh, nonzero `--spec-seed`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from eqrl.experiments import final_comparison as fc
from eqrl.experiments.final_comparison import DIMS

#: Step magnitudes offered to the ranker, in normalized action units. Both signs of each
#: are tried. 0.04 is a little above `PREREG["min_t"]`-scale noise; 0.24 is below
#: `PREREG["step_cap"]` (0.35) so a probe never proposes a jump larger than the frozen
#: repair step is allowed to take.
PROBE_STEPS: tuple[float, ...] = (0.04, 0.08, 0.16, 0.24)

#: How many REAL guarded evaluations the probe may spend before handing back to the
#: solver. Three, because the case this fires on has its whole budget unspent, and because
#: at most one candidate per axis is tried (see `_rank`) -- so three buys three distinct
#: axes rather than three magnitudes of the same one.
PROBE_EVALS = 3

#: Only a candidate that beats the stuck design by more than this is accepted. Guards
#: against spending the remaining budget re-searching from a point that is no better.
MIN_IMPROVEMENT_DB = 0.05


def RETARGET_SURROGATE_OPTIONAL():
    """The ranker, or None if it cannot be loaded.

    Returning None is a supported state, not a failure: `_rank` falls back to a fixed axis
    order and every candidate is still judged by the real guarded evaluator. So a machine
    without `results/surrogate_corpus.npz` gets a worse ORDER, never a wrong answer -- and
    the mode is not silently disabled, which is what an exception here would do.
    """
    try:
        from eqrl.experiments.fastest_hedge import load_fastest_assets

        surrogate, _corpus_X, _radius = load_fastest_assets()
        return surrogate
    except Exception:
        return None


def candidate_axes(plane: dict) -> list[str]:
    """Every axis the retarget may search, in the fixed fallback order.

    The current boost axis is excluded because it is pinned -- that is what triggered this
    -- and the PEAK axis is excluded because `g32_solve` uses the boost and peak
    coordinates as two separate dimensions in its 2-D repair; making them the same axis
    would make that step degenerate.
    """
    return [d for d in DIMS if d not in (plane["boost_axis"], plane["peak_axis"])]


def _rank(x_f: np.ndarray, target: float, axes: list[str], surrogate) -> list[tuple[str, float]]:
    """(axis, step) candidates, best first, at most one per axis.

    Ranked by |predicted boost - target| when a surrogate is available, else left in the
    fixed `axes` order at the smallest useful step. One per axis is deliberate: the point
    of spending a probe evaluation is to learn which AXIS moves this design, and three
    magnitudes of one axis answers that question three times over.
    """
    cands: list[tuple[str, float, float]] = []
    for rank, name in enumerate(axes):
        j = DIMS.index(name)
        steps = [(name, s * sign) for s in PROBE_STEPS for sign in (1.0, -1.0)
                 if 0.0 <= x_f[j] + s * sign <= 1.0]
        if not steps:
            continue                      # this axis is pinned too; nothing to try
        if surrogate is None:
            # No ranking available: smallest step, axes in their fixed order. `rank`
            # keeps that order stable through the sort below.
            cands.append((steps[0][0], steps[0][1], float(rank)))
            continue
        X = np.repeat(np.asarray(x_f, dtype=np.float64)[None, :], len(steps), axis=0)
        for row, (_n, st) in enumerate(steps):
            X[row, j] += st
        pred, _dist = surrogate.predict_boost(X)
        best = int(np.argmin(np.abs(np.asarray(pred, dtype=np.float64) - target)))
        cands.append((steps[best][0], steps[best][1], float(abs(pred[best] - target))))
    cands.sort(key=lambda c: c[2])
    return [(name, step) for name, step, _key in cands]


def _derived_plane(plane: dict, axis: str, slope_db_per_unit: float) -> dict:
    """`plane` with its boost coordinate swapped for `axis` and that axis's measured slope.

    Every other field -- the peak coordinate, its slopes, the band and the aim -- is
    carried through untouched, because only the boost search direction is being changed.
    """
    out = dict(plane)
    out["boost_axis"] = axis
    out["d_boost_db_per_unit"] = float(slope_db_per_unit)
    out["derived_from"] = {"original_boost_axis": plane["boost_axis"],
                           "slope_source": "local finite difference measured on this design"}
    return out


def axis_is_pinned(x_f, plane: dict, atol: float = 1e-9) -> bool:
    """Is the boost coordinate sitting on a bound of the normalized action space?"""
    if x_f is None:
        return False
    v = float(np.asarray(x_f, dtype=np.float64)[DIMS.index(plane["boost_axis"])])
    return v <= atol or v >= 1.0 - atol


def retarget_stage2(evaluate, xs, s1trace, target: float, plane: dict, ladder, budget: int,
                    *, surrogate=None, probe_evals: int = PROBE_EVALS,
                    stop_abs_err_db: float | None = None):
    """Frozen G3.2, and -- only if its boost axis was pinned -- a second pass on a new axis.

    Returns `(trace, info, x_f, rec_f, budget_left)`, the same five values `g32_solve`
    returns, so this is a drop-in for it. `info["retarget"]` records what happened; it is
    absent when the retarget did not fire.
    """
    trace, info, x_f, rec_f, left = fc.g32_solve(
        evaluate, xs, s1trace, target, plane, ladder, budget,
        stop_abs_err_db=stop_abs_err_db)

    fired = (not info["reached_target"] and left > 0 and x_f is not None
             and rec_f is not None and rec_f.get("loose_pass")
             and axis_is_pinned(x_f, plane))
    if not fired:
        info["retarget"] = {"fired": False, "why_not": _why_not(info, left, x_f, rec_f, plane)}
        return trace, info, x_f, rec_f, left

    base_err = abs(rec_f["boost_db"] - target)
    probes: list[dict[str, Any]] = []
    # (err, axis, slope, x_probe, rec_probe, usable_as_start)
    best: tuple[float, str, float, np.ndarray, dict, bool] | None = None

    for name, step in _rank(np.asarray(x_f, dtype=np.float64), target,
                            candidate_axes(plane), surrogate)[:probe_evals]:
        if left <= 0:
            break
        j = DIMS.index(name)
        x_try = np.array(x_f, dtype=np.float64)
        x_try[j] = float(np.clip(x_try[j] + step, 0.0, 1.0))
        if np.allclose(x_try, x_f):
            continue
        left -= 1
        rec, _score, gcheck = evaluate(x_try, target)
        moved = float(x_try[j] - x_f[j])
        # "phase" is not decoration: `pipeline.design` renders every entry of
        # `info["steps"]` through `s["phase"]`, so a probe row without it raises there.
        row = {"phase": "retarget_probe:%s" % name,
               "axis": name, "step": moved, "guard_check": gcheck,
               "valid": rec is not None,
               "feasible": bool(rec and rec["loose_pass"]),
               "boost_db": None if rec is None else rec["boost_db"],
               "abs_err": None if rec is None else abs(rec["boost_db"] - target)}
        if rec is not None:
            trace.append(rec)
            # The slope that will steer the second solve: measured, one evaluation old.
            row["slope_db_per_unit"] = (rec["boost_db"] - rec_f["boost_db"]) / moved
            # AN INFEASIBLE PROBE STILL IDENTIFIES THE RIGHT AXIS. Requiring the probe
            # step itself to be feasible was the first version of this rule and it was
            # wrong: on spec 11.0/12.5 the ranker's top pick `cs +0.16` reaches 11.226 dB
            # (err 0.226) but overshoots out of the feasible set, so the axis that
            # actually moves this design was discarded in favour of `w_in`, which moves it
            # a third as hard. Bracketing between a feasible point and a wall is precisely
            # what `g32_solve`'s stage B already does -- its `wall_t` branch exists for
            # this -- so the axis is accepted on the strength of the MEASUREMENT and the
            # step is left to the solver. What is still required is guard validity: a
            # guard-invalid probe measured nothing, so it says nothing about its axis.
            usable_start = bool(rec["loose_pass"])
            row["usable_as_start"] = usable_start
            if row["abs_err"] < base_err - MIN_IMPROVEMENT_DB:
                if best is None or row["abs_err"] < best[0]:
                    best = (row["abs_err"], name, row["slope_db_per_unit"], x_try, rec,
                            usable_start)
        probes.append(row)

    detail: dict[str, Any] = {"fired": True, "base_abs_err_db": base_err,
                              "probes": probes, "probe_evals_spent": len(probes),
                              "ranked_by": "surrogate" if surrogate is not None else "fixed order"}

    if best is None:
        detail.update(accepted_axis=None,
                      reason="no probed axis was guard-valid and closer to target than "
                             "the stuck design by > %.2f dB" % MIN_IMPROVEMENT_DB)
        info["retarget"] = detail
        return trace, info, x_f, rec_f, left

    _err, axis, slope, x_new, rec_new, usable_start = best
    detail.update(accepted_axis=axis, accepted_step_slope_db_per_unit=slope,
                  abs_err_after_probe_db=_err, probe_usable_as_start=usable_start)

    # Hand the frozen solver a plane whose boost coordinate is the accepted axis, and --
    # only when the probe point is itself feasible -- that point folded into the handoff
    # trace, where `g32_solve`'s own "already feasible, closest to target" scan will pick
    # it up as the start. An INFEASIBLE probe is deliberately NOT appended: the solver
    # would ignore it anyway (its scan filters on `loose_pass`), and the search should
    # start from the feasible design the first pass was holding and bisect toward the
    # wall the probe just located. No branch inside the solver is aware anything changed.
    xs2 = list(xs) + ([x_new] if usable_start else [])
    s1_2 = list(s1trace) + ([rec_new] if usable_start else [])
    trace2, info2, x_f2, rec_f2, left2 = fc.g32_solve(
        evaluate, xs2, s1_2, target, _derived_plane(plane, axis, slope), ladder, left,
        stop_abs_err_db=stop_abs_err_db)

    detail["second_pass"] = {"reason": info2["reason"],
                             "reached_target": bool(info2["reached_target"]),
                             "evals": len(info2["steps"])}
    # The first pass's steps are real simulator work and must stay in the record.
    info2["steps"] = info["steps"] + probes + info2["steps"]
    info2["retarget"] = detail
    info2["first_pass_reason"] = info["reason"]
    if info2["start_source"]:
        info2["start_source"] += " -> retargeted onto %s" % axis

    # Never hand back something worse than what the first pass already held.
    if rec_f2 is None or (rec_f is not None
                          and abs(rec_f["boost_db"] - target) < abs(rec_f2["boost_db"] - target)
                          and not info2["reached_target"]):
        detail["kept_first_pass_design"] = True
        return trace + trace2, info2, x_f, rec_f, left2
    return trace + trace2, info2, x_f2, rec_f2, left2


def _why_not(info: dict, left: int, x_f, rec_f, plane: dict) -> str:
    """Why the retarget stayed out of the way. Recorded so a null result is legible."""
    if info["reached_target"]:
        return "target already reached"
    if x_f is None or rec_f is None:
        return "first pass produced no design to retarget from"
    if not rec_f.get("loose_pass"):
        return "first pass never reached the feasible set; that is the repair stage's job"
    if left <= 0:
        return "no budget left after the first pass"
    if not axis_is_pinned(x_f, plane):
        return ("boost axis %s is not at a bound -- this exit is %r, which a different "
                "mechanism owns" % (plane["boost_axis"], info["reason"]))
    return "unknown"
