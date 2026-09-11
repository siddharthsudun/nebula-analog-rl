"""PVT search must preserve acceptance semantics and expose real corner settings."""
from copy import deepcopy
from dataclasses import replace

import pytest

from eqrl.circuits.ctle import DesignVars
from eqrl.envs.pvt import corner_grid
from eqrl.pvt_refinement import (
    EvaluationBudgetExceeded, PVTEvaluator, full_grid_pass, rank_key,
    recorded_netlist, signed_slacks, stress_corners, summarize,
)
from eqrl.sim.measures import Measures
from eqrl.specs import DEFAULT_SPEC, hard_pass


SPEC = replace(DEFAULT_SPEC, boost_target_tol_db=1.5)


def good_row(corner=("tt", 1.8, 27)):
    m = Measures(ok=True, boost_db=9, peak_freq_ghz=1.8, dc_gain_db=2,
                 hd3_db=-45, noise_vrms=0.0005, power_w=0.003,
                 area_mm2=0.002, eye_h_ui=0.6, eye_v_mv=200)
    health = {"worst_headroom_v": 0.15, "tail_delivery_ratio": 0.98}
    return {"corner": list(corner), "guard_valid": True, "passed": True,
            "target_error_db": 0, "health": health, "slacks": signed_slacks(m, SPEC, health)}


def test_partial_or_duplicate_grid_can_never_be_accepted():
    grid = corner_grid(SPEC, "full")
    rows = [good_row(c) for c in grid]
    assert len(grid) == 45
    assert full_grid_pass(rows, grid)
    assert not full_grid_pass(rows[:8], grid)
    assert not full_grid_pass(rows[:-1] + [rows[0]], grid)
    rows[-1]["passed"] = False
    assert not full_grid_pass(rows, grid)


def test_missing_guard_health_does_not_count_as_a_pass():
    grid = corner_grid(SPEC, "full")
    rows = [good_row(c) for c in grid]
    rows[-1]["guard_valid"] = False
    assert not full_grid_pass(rows, grid)
    assert summarize(rows)["guard_valid"] == 44


def test_stress_selection_covers_every_process_and_includes_failing_corner():
    grid = corner_grid(SPEC, "full")
    rows = [good_row(c) for c in grid]
    rows[-1].update(guard_valid=False, passed=False)
    selected = stress_corners(rows, grid)
    assert len(selected) == 8
    assert {c[0] for c in selected} == set(SPEC.process_corners)
    assert tuple(rows[-1]["corner"]) in selected
    assert ("tt", 1.8, 27) in selected


def test_target_failure_cannot_be_hidden_by_good_other_metrics():
    good = good_row()
    failed = deepcopy(good)
    failed.update(passed=False, target_error_db=1.6)
    failed["slacks"]["boost_target"] = -0.1
    assert rank_key([good]) < rank_key([failed])
    invalid = deepcopy(good)
    invalid["guard_valid"] = False
    assert rank_key([failed]) < rank_key([invalid])


def test_strict_noise_limit_is_decided_by_hard_pass_not_zero_slack():
    m = Measures(ok=True, boost_db=9, peak_freq_ghz=1.8, dc_gain_db=2,
                 hd3_db=-45, noise_vrms=SPEC.noise_vrms_max, power_w=.003,
                 area_mm2=.002, eye_h_ui=.6, eye_v_mv=200)
    slacks = signed_slacks(m, SPEC, good_row()["health"])
    assert slacks["noise"] == 0
    assert not hard_pass(m, SPEC)[0]


def test_target_tolerance_is_required():
    with pytest.raises(ValueError, match="explicit positive target tolerance"):
        signed_slacks(Measures(), DEFAULT_SPEC, good_row()["health"])


def test_batch_over_budget_is_rejected_before_simulation(tmp_path):
    ev = PVTEvaluator(SPEC, tmp_path, limit=1)
    with pytest.raises(EvaluationBudgetExceeded):
        ev.evaluate_batch([{"id": "a"}, {"id": "b"}], [("tt", 1.8, 27)])
    assert ev.evaluations == 0


def test_recorded_netlist_matches_resident_vdd_scaling(monkeypatch):
    from eqrl.circuits import pdk
    monkeypatch.setattr(pdk, "lib_include", lambda c: f"* process {c}")
    deck = recorded_netlist(DesignVars(), vdd=1.71, temp_c=125, corner="ss")
    assert "Vcm cm 0 'vddp*0.72'" in deck
    assert ".param vddp=1.71\n" in deck
    assert ".param tempc=125\n" in deck
    assert "* process ss" in deck
