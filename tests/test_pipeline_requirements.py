"""User acceptance constraints and Auto mode: the labelling contract, PDK-free.

Same stub as tests/test_solve_pipeline.py. What is pinned here:

  * a run with no requirements is stamped as scored against the competition spec;
  * a run WITH requirements is stamped as user-modified, lists every override with its
    direction, and `verify` reports the competition verdict alongside the user one;
  * an unknown requirement is an error, never a silent drop;
  * Auto escalates exactly when the first attempt was not SOLVED, and keeps the first
    attempt's cost and verdict in the result rather than discarding them.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eqrl import pipeline
from eqrl.specs import DEFAULT_SPEC

from tests.test_solve_pipeline import AI_DESIGN, stub  # noqa: F401  (fixture re-use)


class TestSpecFor:
    def test_no_requirements_is_the_competition_spec(self):
        s = pipeline.spec_for(9.0, 14.0, 1.5)
        assert s.power_w_max == DEFAULT_SPEC.power_w_max
        assert pipeline.requirements_diff(s) == []

    def test_requirement_is_applied_and_labelled(self):
        s = pipeline.spec_for(9.0, 14.0, 1.5, {"power_w_max": 0.010, "hd3_db_max": -25.0})
        d = {q["field"]: q for q in pipeline.requirements_diff(s)}
        assert d["power_w_max"]["direction"] == "tighter"
        assert d["hd3_db_max"]["direction"] == "looser"
        assert s.power_w_max == 0.010

    def test_unknown_field_is_an_error(self):
        with pytest.raises(ValueError, match="unknown requirement"):
            pipeline.spec_for(9.0, 14.0, 1.5, {"vdd_nominal": 1.2})

    def test_dc_gain_floor_cannot_go_below_the_guard(self):
        s = pipeline.spec_for(9.0, 14.0, 1.5, {"dc_gain_db_min": -3.0})
        assert s.dc_gain_db_min == DEFAULT_SPEC.dc_gain_db_min

    def test_none_means_default(self):
        s = pipeline.spec_for(9.0, 14.0, 1.5, {"power_w_max": None})
        assert pipeline.requirements_diff(s) == []


class TestStamping:
    def test_plain_run_is_scored_against_the_competition(self, stub):
        stub(best_design=AI_DESIGN)
        r = pipeline.design(8.92, 14.83)
        assert r["spec"]["requirements"] == []
        assert r["spec"]["scored_against"] == "the competition specification"

    def test_requirements_run_is_stamped_user_modified(self, stub):
        stub(best_design=AI_DESIGN)
        r = pipeline.design(8.92, 14.83, requirements={"power_w_max": 0.010})
        assert r["spec"]["scored_against"].startswith("user-modified")
        assert r["spec"]["requirements"][0]["field"] == "power_w_max"
        assert "USER-MODIFIED" in r["spec"]["requirement_set"]
        assert "USER-MODIFIED" in pipeline.describe(r)


class TestVerifyReportsBothBars:
    def test_competition_verdict_alongside_user_verdict(self, monkeypatch):
        """`verify` on a relaxed spec must also say whether the unmodified spec passes."""
        from eqrl.sim.measures import Measures

        class _Verdict:
            is_valid = True
            check = None

            def unwrap(self):
                return Measures(dc_gain_db=1.0, peak_gain_db=10.0, boost_db=9.0,
                                peak_freq_ghz=1.6, hd3_db=-28.0, noise_vrms=6e-4,
                                power_w=2e-3, area_mm2=1e-3, eye_h_ui=0.7, eye_v_mv=600.0)

        class _Ev:
            def evaluate(self, dv, vdd):
                return _Verdict()

        import eqrl.evaluator as evmod
        monkeypatch.setattr(evmod, "build_evaluator", lambda *a, **k: _Ev())
        spec = pipeline.spec_for(9.0, 14.0, 1.5, {"hd3_db_max": -25.0})
        v = pipeline.verify(pipeline.DesignVars(**AI_DESIGN), spec)
        assert v["passed"] is True                   # -28 dB clears the relaxed -25
        assert v["competition_passed"] is False      # but not the competition's -30
        assert v["competition_failing"] == ["hd3"]


class TestAuto:
    def _fake(self, monkeypatch, statuses):
        calls = []

        def fake_design(t, c, *, mode, **kw):
            calls.append(mode)
            st = statuses[mode]
            return {"status": st, "mode": mode, "guidance": {"headline": "h"},
                    "solver": {"best_boost_db": 8.0},
                    "cost": {"optimizer_evals": 4, "measure_all_total": 5,
                             "spice_analyses_total": 30}}
        monkeypatch.setattr(pipeline, "design", fake_design)
        return calls

    def test_no_escalation_when_first_attempt_verifies(self, monkeypatch):
        calls = self._fake(monkeypatch, {"fastest": pipeline.SOLVED})
        r = pipeline._auto(9.0, 14.0, {})
        assert calls == ["fastest"]
        assert r["auto"] == {"escalated": False, "first_mode": "fastest",
                             "first_status": pipeline.SOLVED}
        assert r["mode"] == "auto"

    def test_escalates_and_keeps_the_first_attempt(self, monkeypatch):
        calls = self._fake(monkeypatch, {"fastest": pipeline.UNSOLVED,
                                         "thinking": pipeline.SOLVED})
        r = pipeline._auto(9.0, 14.0, {})
        assert calls == ["fastest", "thinking"]
        assert r["auto"]["escalated"] is True
        assert r["auto"]["first_status"] == pipeline.UNSOLVED
        assert r["cost"]["measure_all_prior_attempts"] == 5
        assert r["cost"]["measure_all_total"] == 10
        assert r["status"] == pipeline.SOLVED

    def test_auto_is_a_mode(self):
        assert "auto" in pipeline.MODES


class TestReselection:
    def test_next_candidate_is_taken_when_only_a_user_check_fails(self, monkeypatch):
        """The most accurate design fails the user's power cap; the next one passes."""
        spec = pipeline.spec_for(9.0, 14.0, 1.5, {"power_w_max": 0.010})
        alt = dict(AI_DESIGN, rs=3000.0)
        result = {
            "design": dict(AI_DESIGN),
            "verification": {"passed": False, "guard_valid": True, "failing": ["power"]},
            "solver": {"best_design": dict(AI_DESIGN), "best_boost_db": 9.0,
                       "best_abs_err": 0.0},
            "provenance": {},
            "cost": {"measure_all_verification": 1, "measure_all_total": 6,
                     "spice_analyses_verification": 5, "spice_analyses_total": 30},
        }
        trace = [{"boost_db": 9.0, "loose_pass": True, "design": dict(AI_DESIGN)},
                 {"boost_db": 9.3, "loose_pass": True, "design": alt},
                 None]
        seen = []

        def fake_verify(dv, s):
            seen.append(dataclasses.asdict(dv)["rs"])
            return {"passed": True, "guard_valid": True, "failing": [],
                    "measures": {}, "checks": {}}
        monkeypatch.setattr(pipeline, "verify", fake_verify)

        class _Counting:
            def __enter__(self):
                return {"measure_all": 1, "analysis": 5}

            def __exit__(self, *a):
                return False
        pipeline._reselect_for_requirements(result, trace, spec, 1.5, _Counting)
        assert seen == [3000.0]
        assert result["design"] == alt
        assert result["verification"]["passed"] is True
        assert result["cost"]["measure_all_total"] == 7
        assert result["provenance"]["requirement_reselection"]["applied"] is True

    def test_no_reselection_when_the_target_check_fails(self, monkeypatch):
        spec = pipeline.spec_for(9.0, 14.0, 1.5, {"power_w_max": 0.010})
        result = {"design": dict(AI_DESIGN),
                  "verification": {"passed": False, "guard_valid": True,
                                   "failing": ["boost_target", "power"]},
                  "solver": {}, "provenance": {}, "cost": {}}
        called = []
        monkeypatch.setattr(pipeline, "verify", lambda *a: called.append(1))
        pipeline._reselect_for_requirements(result, [], spec, 1.5, None)
        assert called == []
        assert "requirement_reselection" not in result["provenance"]
