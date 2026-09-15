"""Measure G3.2's repair plane at the design under test, instead of reading frozen slopes.

WHAT THIS ANSWERS. G3.2 steers with a plane whose slopes were fitted once, offline, on
FOUR specs of spec-seed 2 -- `plane_from_probe("results/g32_peak_probe.json")`, giving
d_boost_db_per_unit = 12.45 dB/unit on `rs` and d_ln_peak_per_unit = -1.15/unit on `l_in`.
(Not `results/g32b_plane.json`, despite the name: that is an earlier G3.2b run log whose
peak axis is `r_load`.) Those numbers are medians over a development slice, hard-coded
into every subsequent run. The fair criticism is not that
they are wrong -- the probe that produced them is in the repo -- but that a reader cannot
tell a measured controller from a hand-fitted one, and four specs is a thin basis for a
constant applied everywhere. This module removes the constant: it spends 2-3 evaluations
at the handoff point to measure the local slopes on the design actually in hand, and
returns a plane with the SAME schema so it drops straight into `g32_solve(..., plane, ...)`.

THIS IS NOT A NEW MECHANISM, IT IS AN EXISTING ONE MOVED EARLIER. The frozen solver already
re-measures the peak slope at runtime -- after its first repair step it overwrites `gain`
with `log(rec.peak / pk) / moved` and steers on that for every step after. So G3.2 already
trusts a locally measured slope more than the table; it just cannot do so until it has
spent a step blind. This calibrates the same quantity (and the boost slope, which the
frozen code never re-measures) BEFORE the first step rather than after it.

WHAT IT DELIBERATELY DOES NOT DO. It does not re-choose the axes. Picking `rs` for boost
and `l_in` for peak came from a full six-dimensional probe, and re-deriving that per design
would cost far more than the whole repair budget. `boost_axis`, `peak_axis`, `band_ghz` and
`peak_aim_ghz` are carried over from the frozen plane unchanged; only the four slopes are
measured. Any claim from this module is therefore about the SLOPES being measured, never
about the coordinate choice, which remains a seed-2 development artifact.

EVERY MEASUREMENT CAN FAIL, AND FAILURE MEANS FALL BACK. A perturbation can be rejected by
the guard, or land with the AC peak on the sweep edge where the peak (and hence the boost)
is not a real measurement. A slope can also come back so small that it carries no
information, and the solver divides by both `d_boost_db_per_unit` and `d_ln_peak_per_unit`.
In each of those cases this module keeps the FROZEN value for that entry and records which
entries it fell back on. With every measurement failing it returns the frozen plane exactly,
which is what makes it safe to switch on: the worst case is the current behaviour plus the
evaluations it spent finding out.

Changes no reward, PPO hyperparameter, design bound, guard, hard_pass, controller constant,
frozen model or benchmark criterion. Imports nothing from `g32_repair.py` and edits nothing
in `final_comparison.py`; the solver is untouched and its equivalence gate still covers it.

    from silq.experiments.g32_selfcal import calibrate_plane
    plane, spent, info = calibrate_plane(evaluate, x_f, target, frozen_plane)
    trace, sinfo, x, rec, left = g32_solve(evaluate, xs, s1, target, plane, ladder,
                                           budget - spent)
"""
from __future__ import annotations

import math

import numpy as np

from silq.circuits.ctle import ACTION_SPACE
from silq.experiments.g32_peak_report import at_sweep_edge

DIMS = list(ACTION_SPACE.keys())

#: Perturbation size in normalized design units. Identical to `g32_peak_probe.PROBE_H`,
#: the step the frozen plane itself was measured with, so a calibrated slope and the
#: constant it replaces are estimates of the same finite difference rather than of two
#: different ones. `tests/test_g32_selfcal.py` fails if the two drift apart.
PROBE_H = 0.05

#: A slope below this magnitude is treated as "not measured" rather than as "measured to be
#: small". Both are divided by inside the solver, so a near-zero estimate turns the first
#: step into a saturated guess wearing the costume of a measurement. The floors sit roughly
#: an order of magnitude under the frozen values (12.45 dB/unit and 1.15/unit); a design
#: whose real sensitivity is genuinely that flat is one where the frozen constant is no
#: better, so falling back loses nothing.
MIN_ABS_D_BOOST = 1.0        # dB per unit
MIN_ABS_D_LN_PEAK = 0.05     # ln(GHz) per unit


def _usable(rec: dict | None) -> bool:
    """A point is a measurement only if the guard kept it AND the peak is interior.

    `at_sweep_edge` is the frozen probe's own filter, imported rather than re-implemented.
    When argmax lands on the first or last sweep point there is no peak, so neither the
    peak nor the boost derived from it means anything -- the probe drops such points for
    both quantities and so does this.
    """
    return rec is not None and not at_sweep_edge(rec.get("peak_freq_ghz"))


def _finite_difference(base: dict, plus: dict | None, minus: dict | None) -> dict | None:
    """Central difference where both ends survived, one-sided against base otherwise.

    Arithmetic and span convention are `g32_peak_report.slopes`: `span` is the distance the
    coordinate actually moved, which is not 2h when a perturbation clipped at a bound.
    """
    if plus is not None and minus is not None:
        span = plus["moved"] + minus["moved"]
        if span <= 0:
            return None
        a, b = plus["rec"], minus["rec"]
    elif plus is not None or minus is not None:
        q = plus if plus is not None else minus
        span = q["moved"]
        if span <= 0 or not _usable(base):
            return None
        # One-sided against the base point. Negating for a minus-probe keeps the sign
        # convention identical to the central case: slope per +1 unit of the coordinate.
        a, b = (q["rec"], base) if plus is not None else (base, q["rec"])
    else:
        return None
    if a["peak_freq_ghz"] <= 0 or b["peak_freq_ghz"] <= 0:
        return None
    return {
        "d_boost": (a["boost_db"] - b["boost_db"]) / span,
        "d_peak": (a["peak_freq_ghz"] - b["peak_freq_ghz"]) / span,
        "d_ln_peak": math.log(a["peak_freq_ghz"] / b["peak_freq_ghz"]) / span,
        "span": span,
        "sided": "central" if (plus is not None and minus is not None) else "one",
    }


def _probe_axis(take, x0: np.ndarray, j: int, h: float, base: dict | None,
                two_sided: bool) -> tuple[dict | None, list[str]]:
    """Perturb one coordinate and return its measured slopes, or None if unmeasurable.

    Order matters for cost. The +h probe is spent first; -h is spent only when +h was
    unusable (or when `two_sided`), so the common case costs one evaluation per axis and
    the guard-blocked case costs two. That is the "2-3 perturbations" budget.
    """
    notes: list[str] = []
    ends: dict[str, dict | None] = {"plus": None, "minus": None}
    have_base = base is not None and _usable(base)

    def spend(tag: str) -> None:
        s = h if tag == "plus" else -h
        e = np.zeros(len(DIMS))
        e[j] = 1.0
        xt = np.clip(np.asarray(x0, dtype=np.float64) + s * e, 0.0, 1.0)
        moved = float(abs(xt[j] - float(x0[j])))
        if moved <= 0.0:
            notes.append("%s: coordinate is already at its bound" % tag)
            return
        rec = take(xt)
        if not _usable(rec):
            notes.append("%s: %s" % (tag, "guard rejected" if rec is None
                                     else "peak on the sweep edge"))
            return
        ends[tag] = {"rec": rec, "moved": moved}

    def formable() -> bool:
        """Enough points on hand for a difference: two ends, or one end and a base."""
        if ends["plus"] is not None and ends["minus"] is not None:
            return True
        return have_base and (ends["plus"] is not None or ends["minus"] is not None)

    spend("plus")
    # Spend -h when it would buy something: because two_sided was asked for, or because
    # what we have so far cannot form a difference. The second case covers BOTH failures --
    # a rejected +h probe, and a usable +h probe with no usable base to difference it
    # against. Skipping the retry in that second case would throw away a good measurement
    # for want of one more evaluation.
    if two_sided or not formable():
        spend("minus")
    if not formable():
        if not have_base:
            notes.append("no usable base point, so a one-sided difference is unavailable")
        return None, notes
    return _finite_difference(base or {}, ends["plus"], ends["minus"]), notes


def calibrate_plane(evaluate, x0, target: float, frozen: dict, *,
                    base_rec: dict | None = None, h: float = PROBE_H,
                    two_sided: bool = False, budget: int | None = None
                    ) -> tuple[dict, int, dict]:
    """Measure the repair plane at `x0`. Returns (plane, evaluations_spent, info).

    `evaluate(x, target) -> (rec, score, guard_check)` is the same callable `g32_solve`
    takes. `frozen` is the plane to fall back to, entry by entry -- normally the one
    `g32_peak_report.plane_from_probe` builds. `base_rec` is the handoff measurement if the
    caller already has it, which it normally does; passing it makes the one-sided
    difference free rather than costing an extra evaluation.

    The returned plane always carries every key `frozen` has, so it is a drop-in. Which
    entries were measured and which fell back is in `info["measured"]` / `info["fellback"]`
    -- read them before quoting any of the slopes as measured.
    """
    spent = 0

    def take(x):
        nonlocal spent
        if budget is not None and spent >= budget:
            return None
        spent += 1
        rec, _score, _gcheck = evaluate(x, target)
        return rec

    jb, jp = DIMS.index(frozen["boost_axis"]), DIMS.index(frozen["peak_axis"])
    boost, bnotes = _probe_axis(take, x0, jb, h, base_rec, two_sided)
    peak, pnotes = _probe_axis(take, x0, jp, h, base_rec, two_sided)

    plane = dict(frozen)
    measured: list[str] = []
    fellback: list[str] = []

    def put(key: str, value: float | None, floor: float | None = None) -> None:
        """Take the measurement only if it is one. Otherwise keep the frozen entry."""
        if value is None or not math.isfinite(value):
            fellback.append(key)
            return
        if floor is not None and abs(value) < floor:
            fellback.append(key + " (below the information floor)")
            return
        plane[key] = float(value)
        measured.append(key)

    put("d_boost_db_per_unit", boost and boost["d_boost"], MIN_ABS_D_BOOST)
    put("boost_axis_d_peak_ghz_per_unit", boost and boost["d_peak"])
    put("d_peak_ghz_per_unit", peak and peak["d_peak"])
    put("d_ln_peak_per_unit", peak and peak["d_ln_peak"], MIN_ABS_D_LN_PEAK)
    put("peak_axis_d_boost_db_per_unit", peak and peak["d_boost"])

    #: Provenance, not decoration. `source` is what the report prints, and a plane that is
    #: partly measured and partly frozen must not be describable as either one.
    #:
    #: The three fields below describe HOW THE SLOPES IN THIS DICT WERE OBTAINED, and the
    #: frozen plane's answers ("spec_seed 2", "4 specs", "peak under resolution on 3 of 4
    #: specs") stop being true the moment a slope is measured here. The solver reads none of
    #: them, but a report would, and a calibrated plane that still claims seed 2 is exactly
    #: the kind of stale provenance that turns into a false sentence in a paper. They are
    #: rewritten to describe this measurement, or left frozen when nothing was measured.
    if measured:
        plane["source"] = "runtime self-calibration at the handoff (fallback: %s)" \
            % frozen.get("source", "frozen plane")
        plane["n_specs_measured"] = 1          # this design, not a development slice
        plane["spec_seed"] = None              # no offline spec draw stands behind these
        if "boost_axis_d_peak_ghz_per_unit" in measured:
            plane["boost_axis_peak_under_resolution_on"] = None
    plane["selfcal"] = {
        "h": h, "two_sided": bool(two_sided), "evaluations": spent,
        "measured": measured, "fellback": fellback,
        "boost_axis_sided": boost and boost["sided"],
        "peak_axis_sided": peak and peak["sided"],
    }

    info = {
        "evaluations": spent, "measured": measured, "fellback": fellback,
        "notes": {"boost_axis": bnotes, "peak_axis": pnotes},
        "fully_frozen": not measured,
        "deltas": {k: (plane[k] - frozen[k]) for k in measured if k in frozen},
        "ratios": {k: (plane[k] / frozen[k]) for k in measured
                   if k in frozen and frozen[k] not in (0, 0.0)},
    }
    return plane, spent, info
