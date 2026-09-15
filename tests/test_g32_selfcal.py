"""Unit tests for runtime plane calibration. No SPICE, no model, no network.

Every test drives `calibrate_plane` with an analytic stand-in for `evaluate`, so each one
pins a property of the calibration logic rather than a property of the circuit. The two
that matter most are the fallback identity (a calibration where nothing measurable happens
returns the frozen plane byte for byte) and the cross-check against
`g32_peak_report.slopes`, which is the function the frozen plane itself was built with.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from silq.circuits.ctle import ACTION_SPACE
from silq.experiments.g32_peak_report import AC_FSTART_GHZ, slopes
from silq.experiments.g32_selfcal import (
    MIN_ABS_D_BOOST, MIN_ABS_D_LN_PEAK, PROBE_H, calibrate_plane,
)

DIMS = list(ACTION_SPACE.keys())
JB, JP = DIMS.index("rs"), DIMS.index("l_in")

# A frozen plane with the real schema and the real values, so a fallback is a fallback to
# the numbers actually in use. These are what plane_from_probe("results/g32_peak_probe.json")
# returns -- the plane final_comparison builds at runtime. Transcribed rather than loaded,
# because a test that reads the artifact it is validating cannot detect that artifact
# changing underneath it; test_frozen_constants_still_match_the_live_plane closes the loop.
FROZEN = {
    "boost_axis": "rs", "peak_axis": "l_in",
    "d_boost_db_per_unit": 12.454385847686456,
    "boost_axis_d_peak_ghz_per_unit": 0.0,
    "boost_axis_peak_under_resolution_on": "3 of 4 specs",
    "d_peak_ghz_per_unit": -3.143996950000001,
    "d_ln_peak_per_unit": -1.1526639022851284,
    "peak_axis_d_boost_db_per_unit": -5.287655146126827,
    "band_ghz": [1.25, 2.5], "peak_aim_ghz": 1.7677669529663689,
    "grid_step_pct": 5.9253725177288885, "source": "results/g32_peak_probe.json",
    "n_specs_measured": 4, "spec_seed": 2,
}

# The synthetic circuit. Boost is linear in both axes; the peak is log-linear in the peak
# axis, which is the form d_ln_peak is defined to recover exactly. Values are the same
# order of magnitude as the frozen plane so the information floors are not accidentally
# in play.
B0, SB, SPB = 6.0, 9.0, -4.0        # dB, dB/unit on rs, dB/unit on l_in
P0, GLN = 1.8, -0.9                 # GHz at x=0, ln(GHz) per unit on l_in
X0 = np.full(len(DIMS), 0.5)


def truth(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "boost_db": B0 + SB * x[JB] + SPB * x[JP],
        "peak_freq_ghz": P0 * math.exp(GLN * x[JP]),
        "dc_gain_db": 10.0,
        "loose_pass": False, "failing": [], "design": {},
    }


def make_eval(reject=(), edge=(), rec_fn=truth):
    """`evaluate(x, target)` with scripted failures.

    `reject` and `edge` are coordinate indices whose perturbations come back guard-rejected
    or with the peak pinned to the sweep edge -- the two ways a real probe point dies.
    """
    calls = []

    def evaluate(x, target):
        x = np.asarray(x, dtype=np.float64)
        calls.append(x.copy())
        moved = [j for j in range(len(DIMS)) if abs(x[j] - X0[j]) > 1e-12]
        if any(j in reject for j in moved):
            return None, -1e9, "guard"
        rec = dict(rec_fn(x))
        if any(j in edge for j in moved):
            rec["peak_freq_ghz"] = AC_FSTART_GHZ
        return rec, 0.0, "ok"

    evaluate.calls = calls
    return evaluate


def frozen_part(plane: dict) -> dict:
    """The plane minus the bookkeeping key calibration always adds."""
    return {k: v for k, v in plane.items() if k != "selfcal"}


def plane_fellback(info: dict, key: str) -> bool:
    """`fellback` annotates the floor case, so match on the key rather than equality."""
    return any(f == key or f.startswith(key + " ") for f in info["fellback"])


# --- the fallback identity ---------------------------------------------------------

def test_total_measurement_failure_returns_the_frozen_plane_exactly():
    """The property that makes this safe to switch on: worst case is current behaviour."""
    ev = make_eval(reject=(JB, JP))
    plane, spent, info = calibrate_plane(ev, X0, 12.0, FROZEN, base_rec=truth(X0))
    assert frozen_part(plane) == FROZEN
    assert info["fully_frozen"] is True
    assert info["measured"] == []
    assert spent > 0, "it should have paid to find out, not skipped the attempt"


def test_sweep_edge_points_are_not_treated_as_measurements():
    """A peak pinned to the sweep edge is a degenerate response, not a slope."""
    ev = make_eval(edge=(JB, JP))
    plane, _spent, info = calibrate_plane(ev, X0, 12.0, FROZEN, base_rec=truth(X0))
    assert frozen_part(plane) == FROZEN
    assert info["fully_frozen"] is True


def test_an_unusable_base_alone_does_not_prevent_calibration():
    """With no usable base the two ends still form a central difference."""
    ev = make_eval()
    plane, _spent, info = calibrate_plane(ev, X0, 12.0, FROZEN, base_rec=None)
    assert not info["fully_frozen"]
    assert plane["d_boost_db_per_unit"] == pytest.approx(SB)
    assert plane["selfcal"]["boost_axis_sided"] == "central"


def test_frozen_plane_is_not_mutated():
    before = dict(FROZEN)
    calibrate_plane(make_eval(), X0, 12.0, FROZEN, base_rec=truth(X0))
    assert FROZEN == before


# --- the arithmetic ----------------------------------------------------------------

def test_recovers_slopes_it_was_given():
    ev = make_eval()
    plane, _spent, _info = calibrate_plane(ev, X0, 12.0, FROZEN,
                                           base_rec=truth(X0), two_sided=True)
    assert plane["d_boost_db_per_unit"] == pytest.approx(SB)
    assert plane["peak_axis_d_boost_db_per_unit"] == pytest.approx(SPB)
    assert plane["d_ln_peak_per_unit"] == pytest.approx(GLN)
    # rs does not move the peak in the stand-in, and the calibration should say so rather
    # than manufacturing a small number.
    assert plane["boost_axis_d_peak_ghz_per_unit"] == pytest.approx(0.0, abs=1e-12)


def test_matches_g32_peak_report_slopes_on_the_same_points():
    """The frozen plane's own arithmetic, fed the same measurements, must agree.

    This is the test that makes the comparison meaningful: if calibration used a different
    finite difference than the one that produced the constants, a frozen-vs-calibrated
    result would be measuring the two formulas against each other, not the two planes.
    """
    ev = make_eval()
    plane, _spent, _info = calibrate_plane(ev, X0, 12.0, FROZEN,
                                           base_rec=truth(X0), two_sided=True)

    def point(j, s):
        xt = np.clip(X0 + s * PROBE_H * np.eye(len(DIMS))[j], 0.0, 1.0)
        r = dict(truth(xt))
        r.update(valid=True, moved=float(abs(xt[j] - X0[j])))
        return r

    artifact = {"rows": [{"spec": 0, "base": dict(truth(X0), valid=True), "dims": {
        name: {"points": {"plus": point(j, +1.0), "minus": point(j, -1.0)}}
        for name, j in (("rs", JB), ("l_in", JP))
    }}]}
    per_dim, _seen, _dropped = slopes(artifact)

    assert plane["d_boost_db_per_unit"] == pytest.approx(per_dim["rs"][0]["d_boost"])
    assert plane["boost_axis_d_peak_ghz_per_unit"] == pytest.approx(
        per_dim["rs"][0]["d_peak"])
    assert plane["peak_axis_d_boost_db_per_unit"] == pytest.approx(
        per_dim["l_in"][0]["d_boost"])
    assert plane["d_peak_ghz_per_unit"] == pytest.approx(per_dim["l_in"][0]["d_peak"])
    assert plane["d_ln_peak_per_unit"] == pytest.approx(per_dim["l_in"][0]["d_ln_peak"])


def test_one_sided_keeps_the_sign_convention_of_the_central_difference():
    """A minus-only probe must not report a slope with the wrong sign.

    Sign is not cosmetic here: `g32_solve`'s Stage B reads
    `way = 1 if (target - b_f) * S_BOOST > 0 else -1`, so a flipped S_BOOST sends the
    search the wrong way down the axis.
    """
    two = calibrate_plane(make_eval(), X0, 12.0, FROZEN,
                          base_rec=truth(X0), two_sided=True)[0]
    # Block the +h end on both axes, forcing the -h retry and a one-sided difference.
    x_hi = X0.copy()

    def ev_minus_only(x, target):
        x = np.asarray(x, dtype=np.float64)
        moved_up = any(x[j] > X0[j] + 1e-12 for j in (JB, JP))
        return (None, -1e9, "guard") if moved_up else (truth(x), 0.0, "ok")

    one, _spent, info = calibrate_plane(ev_minus_only, x_hi, 12.0, FROZEN,
                                        base_rec=truth(X0))
    assert info["fully_frozen"] is False
    assert one["selfcal"]["boost_axis_sided"] == "one"
    for key in ("d_boost_db_per_unit", "peak_axis_d_boost_db_per_unit",
                "d_ln_peak_per_unit"):
        assert math.copysign(1.0, one[key]) == math.copysign(1.0, two[key]), key
    assert one["d_boost_db_per_unit"] == pytest.approx(SB)
    assert one["d_ln_peak_per_unit"] == pytest.approx(GLN)


def test_a_coordinate_at_its_bound_falls_back_to_the_other_side():
    """x0 = 1.0 on the boost axis: +h cannot move, so -h must carry the measurement."""
    x0 = X0.copy()
    x0[JB] = 1.0
    plane, _spent, info = calibrate_plane(make_eval(), x0, 12.0, FROZEN,
                                          base_rec=truth(x0))
    assert plane["d_boost_db_per_unit"] == pytest.approx(SB)
    assert any("already at its bound" in n for n in info["notes"]["boost_axis"])


# --- the floors --------------------------------------------------------------------

def test_an_uninformative_boost_slope_falls_back_rather_than_being_believed():
    """A near-flat measured slope is divided by; the frozen constant is the safer read."""
    def flat(x):
        r = dict(truth(x))
        r["boost_db"] = B0 + 0.5 * MIN_ABS_D_BOOST * np.asarray(x)[JB]
        return r

    plane, _spent, info = calibrate_plane(make_eval(rec_fn=flat), X0, 12.0, FROZEN,
                                          base_rec=flat(X0))
    assert plane["d_boost_db_per_unit"] == FROZEN["d_boost_db_per_unit"]
    assert any("information floor" in f for f in info["fellback"])
    # The floor is per entry: the peak axis was still measured.
    assert "d_ln_peak_per_unit" in info["measured"]


def test_an_uninformative_peak_slope_falls_back():
    def flat_peak(x):
        r = dict(truth(x))
        r["peak_freq_ghz"] = P0 * math.exp(0.5 * MIN_ABS_D_LN_PEAK * np.asarray(x)[JP])
        return r

    plane, _spent, info = calibrate_plane(make_eval(rec_fn=flat_peak), X0, 12.0, FROZEN,
                                          base_rec=flat_peak(X0))
    assert plane["d_ln_peak_per_unit"] == FROZEN["d_ln_peak_per_unit"]
    assert "d_boost_db_per_unit" in info["measured"]


def test_failure_on_one_axis_does_not_cost_the_other():
    plane, _spent, info = calibrate_plane(make_eval(reject=(JP,)), X0, 12.0, FROZEN,
                                          base_rec=truth(X0))
    assert plane["d_boost_db_per_unit"] == pytest.approx(SB)
    assert plane["d_ln_peak_per_unit"] == FROZEN["d_ln_peak_per_unit"]
    assert plane["peak_axis_d_boost_db_per_unit"] == FROZEN["peak_axis_d_boost_db_per_unit"]


# --- cost and schema ---------------------------------------------------------------

def test_cost_is_two_evaluations_in_the_clean_case():
    ev = make_eval()
    _plane, spent, _info = calibrate_plane(ev, X0, 12.0, FROZEN, base_rec=truth(X0))
    assert spent == 2 == len(ev.calls)


def test_cost_is_four_when_both_sides_are_asked_for():
    ev = make_eval()
    _plane, spent, _info = calibrate_plane(ev, X0, 12.0, FROZEN,
                                           base_rec=truth(X0), two_sided=True)
    assert spent == 4 == len(ev.calls)


def test_budget_is_respected():
    ev = make_eval()
    _plane, spent, info = calibrate_plane(ev, X0, 12.0, FROZEN, base_rec=truth(X0),
                                          two_sided=True, budget=1)
    assert spent == 1 and len(ev.calls) == 1
    # The one evaluation bought the boost axis one-sided; the peak axis got nothing and
    # fell back, which is the graceful degradation the budget is there to produce.
    assert "d_boost_db_per_unit" in info["measured"]
    assert plane_fellback(info, "d_ln_peak_per_unit")


def test_the_returned_plane_is_a_drop_in_for_the_frozen_one():
    """Every key g32_solve or a report might read must still be there."""
    plane, _spent, _info = calibrate_plane(make_eval(), X0, 12.0, FROZEN,
                                           base_rec=truth(X0))
    assert set(FROZEN).issubset(plane)
    for key in ("boost_axis", "peak_axis", "band_ghz", "peak_aim_ghz"):
        assert plane[key] == FROZEN[key], "%s is not calibrated and must not move" % key


def test_calibrated_plane_does_not_inherit_the_frozen_probes_provenance():
    """A measured plane claiming spec_seed 2 would put a false sentence in a report."""
    plane, _spent, info = calibrate_plane(make_eval(), X0, 12.0, FROZEN,
                                          base_rec=truth(X0))
    assert info["measured"]
    assert plane["spec_seed"] is None
    assert plane["n_specs_measured"] == 1
    assert plane["source"] != FROZEN["source"]
    assert plane["selfcal"]["evaluations"] == 2


def test_probe_step_matches_the_one_the_frozen_plane_was_measured_with():
    """Read as source text: importing g32_peak_probe would bootstrap the simulator."""
    src = (Path(__file__).resolve().parents[1]
           / "src/silq/experiments/g32_peak_probe.py").read_text(encoding="utf-8")
    m = re.search(r"^PROBE_H\s*=\s*([0-9.eE+-]+)", src, re.M)
    assert m, "PROBE_H not found in g32_peak_probe.py"
    assert float(m.group(1)) == PROBE_H


def test_frozen_constants_still_match_the_live_plane():
    """Close the loop on the transcription above.

    FROZEN is typed out rather than loaded so the other tests cannot be silently
    invalidated by the artifact changing. That trick only works if something notices when
    the artifact DOES change, which is this test. Skipped where the probe is unavailable,
    since it reads a result file rather than testing logic.
    """
    probe = Path(__file__).resolve().parents[1] / "results/g32_peak_probe.json"
    if not probe.exists():
        pytest.skip("frozen probe artifact not present")
    from silq.experiments.g32_peak_report import plane_from_probe
    live = plane_from_probe(str(probe))
    # `source` echoes the path the caller passed, so it differs between an absolute path
    # here and the relative one final_comparison uses. Every entry that carries meaning is
    # compared.
    assert {k: v for k, v in live.items() if k != "source"}         == {k: v for k, v in FROZEN.items() if k != "source"}
    assert Path(live["source"]).name == Path(FROZEN["source"]).name
