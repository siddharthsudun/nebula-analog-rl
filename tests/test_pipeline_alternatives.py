"""result["alternatives"]: bounded, independently-verified tradeoffs from the same trace.

PDK-free throughout -- `verify` is monkeypatched, no simulator involved. What is pinned:

  * only a SOLVED result ever gets an alternatives block;
  * the pool filter mirrors _reselect_for_requirements exactly (loose_pass, within tol,
    distinct from the chosen design) and excludes the POST-reselection design when both
    fire;
  * ALTERNATIVES_CAP bounds real verify() calls even when more candidates qualify;
  * only independently-passing candidates land in "items", each carrying the full
    verify() output (so noise/power/area/HD3/eye are present, not just boost_db);
  * cost is folded additively into new alternatives_* buckets, never double-counted;
  * design()'s PVT stage still runs exactly once per request, never once per alternative;
  * _auto() passes the key through untouched.
"""
from __future__ import annotations

import dataclasses

from silq import pipeline

from tests.test_solve_pipeline import AI_DESIGN, stub  # noqa: F401  (fixture re-use)


class _Counting:
    def __enter__(self):
        return {"measure_all": 1, "analysis": 5}

    def __exit__(self, *a):
        return False


def _base_result(status=pipeline.SOLVED):
    return {
        "design": dict(AI_DESIGN),
        "status": status,
        "cost": {"measure_all_total": 6, "spice_analyses_total": 30},
    }


class TestAttachAlternatives:
    def test_no_op_when_not_solved(self, monkeypatch):
        result = _base_result(status=pipeline.CLOSED_NOT_VERIFIED)
        called = []
        monkeypatch.setattr(pipeline, "verify", lambda *a: called.append(1))
        pipeline._attach_alternatives(result, [], pipeline.spec_for(9.0, 14.0, 1.5), 1.5,
                                      _Counting)
        assert called == []
        assert "alternatives" not in result

    def test_pool_filter_matches_reselection(self, monkeypatch):
        spec = pipeline.spec_for(9.0, 14.0, 1.5)
        result = _base_result()
        chosen = dict(AI_DESIGN)
        in_range = dict(AI_DESIGN, rs=3000.0)
        out_of_tol = dict(AI_DESIGN, rs=2500.0)
        trace = [
            {"boost_db": 9.0, "loose_pass": True, "design": chosen},          # == chosen
            {"boost_db": 9.2, "loose_pass": False, "design": dict(AI_DESIGN, rs=3100.0)},
            {"boost_db": 20.0, "loose_pass": True, "design": out_of_tol},     # outside tol
            {"boost_db": 9.3, "loose_pass": True, "design": in_range},        # qualifies
            None,
        ]
        monkeypatch.setattr(pipeline, "verify", lambda dv, s: {"passed": True, "failing": [],
                                                                "measures": {}})
        pipeline._attach_alternatives(result, trace, spec, 1.5, _Counting)
        assert result["alternatives"]["pool_size"] == 1
        assert result["alternatives"]["items"][0]["design"] == in_range

    def test_cap_bounds_real_verify_calls(self, monkeypatch):
        spec = pipeline.spec_for(9.0, 14.0, 1.5)
        result = _base_result()
        trace = [{"boost_db": 9.0 + i * 0.01, "loose_pass": True,
                  "design": dict(AI_DESIGN, rs=3000.0 + i)}
                 for i in range(1, pipeline.ALTERNATIVES_CAP + 3)]
        seen = []

        def fake_verify(dv, s):
            seen.append(dataclasses.asdict(dv)["rs"])
            return {"passed": True, "failing": [], "measures": {}}
        monkeypatch.setattr(pipeline, "verify", fake_verify)
        pipeline._attach_alternatives(result, trace, spec, 1.5, _Counting)
        assert len(seen) == pipeline.ALTERNATIVES_CAP
        assert result["alternatives"]["pool_size"] == len(trace)
        assert len(result["alternatives"]["tried"]) == pipeline.ALTERNATIVES_CAP

    def test_items_only_independently_passing_with_measures(self, monkeypatch):
        spec = pipeline.spec_for(9.0, 14.0, 1.5)
        result = _base_result()
        fails = dict(AI_DESIGN, rs=3000.0)
        passes = dict(AI_DESIGN, rs=3100.0)
        trace = [{"boost_db": 9.1, "loose_pass": True, "design": fails},
                 {"boost_db": 9.2, "loose_pass": True, "design": passes}]

        def fake_verify(dv, s):
            rs = dataclasses.asdict(dv)["rs"]
            ok = rs != 3000.0
            return {"passed": ok, "failing": [] if ok else ["hd3"],
                    "measures": {"noise_vrms": 6e-4, "power_w": 2e-3}}
        monkeypatch.setattr(pipeline, "verify", fake_verify)
        pipeline._attach_alternatives(result, trace, spec, 1.5, _Counting)
        alt = result["alternatives"]
        assert len(alt["tried"]) == 2
        assert len(alt["items"]) == 1
        assert alt["items"][0]["design"] == passes
        assert alt["items"][0]["verification"]["measures"]["noise_vrms"] == 6e-4

    def test_cost_folds_additively(self, monkeypatch):
        spec = pipeline.spec_for(9.0, 14.0, 1.5)
        result = _base_result()
        result["cost"] = {"measure_all_total": 6, "spice_analyses_total": 30}
        trace = [{"boost_db": 9.0 + i * 0.01, "loose_pass": True,
                  "design": dict(AI_DESIGN, rs=3000.0 + i)}
                 for i in range(1, pipeline.ALTERNATIVES_CAP + 1)]
        monkeypatch.setattr(pipeline, "verify", lambda dv, s: {"passed": True, "failing": [],
                                                                "measures": {}})
        pipeline._attach_alternatives(result, trace, spec, 1.5, _Counting)
        cost = result["cost"]
        assert cost["measure_all_alternatives"] == pipeline.ALTERNATIVES_CAP
        assert cost["measure_all_total"] == 6 + pipeline.ALTERNATIVES_CAP
        assert cost["spice_analyses_alternatives"] == pipeline.ALTERNATIVES_CAP * 5
        assert cost["spice_analyses_total"] == 30 + pipeline.ALTERNATIVES_CAP * 5

    def test_excludes_post_reselection_design_not_original(self, monkeypatch):
        """If reselection already swapped result["design"], alternatives must read the
        NEW chosen design, not the original arm_b winner -- otherwise the reselected
        design could reappear as its own "alternative"."""
        spec = pipeline.spec_for(9.0, 14.0, 1.5)
        original = dict(AI_DESIGN)
        reselected = dict(AI_DESIGN, rs=3050.0)
        result = _base_result()
        result["design"] = reselected  # simulates _reselect_for_requirements having run
        trace = [{"boost_db": 9.0, "loose_pass": True, "design": original},
                 {"boost_db": 9.1, "loose_pass": True, "design": reselected}]
        monkeypatch.setattr(pipeline, "verify", lambda dv, s: {"passed": True, "failing": [],
                                                                "measures": {}})
        pipeline._attach_alternatives(result, trace, spec, 1.5, _Counting)
        designs = [item["design"] for item in result["alternatives"]["items"]]
        assert reselected not in designs
        assert original in designs


class TestDesignWiring:
    def test_alternatives_populated_and_pvt_runs_once(self, monkeypatch, stub):
        """End-to-end through design(): multiple distinct guard-valid candidates in the
        trace produce alternatives, and apply_pvt_stage -- though it accepts the request
        at the design() level -- is never invoked per-alternative, only once overall."""
        fake = stub(best_design=AI_DESIGN)
        alt1 = dict(AI_DESIGN, rs=3100.0)
        alt2 = dict(AI_DESIGN, rs=3200.0)
        monkeypatch.setattr(type(fake), "stage1_rollout", lambda self, *a, **k: (
            [], [
                {"boost_db": 8.92, "loose_pass": True, "design": dict(AI_DESIGN),
                 "peak_freq_ghz": 1.6, "dc_gain_db": 1.2, "failing": []},
                {"boost_db": 8.95, "loose_pass": True, "design": alt1,
                 "peak_freq_ghz": 1.6, "dc_gain_db": 1.2, "failing": []},
                {"boost_db": 9.0, "loose_pass": True, "design": alt2,
                 "peak_freq_ghz": 1.6, "dc_gain_db": 1.2, "failing": []},
            ], None))
        monkeypatch.setattr(type(fake), "g32_solve", lambda self, *a, **k: (
            [], {"steps": [], "start_source": "stage1 (free)", "reached_target": True,
                 "reason": "stub", "rescue_used": 0, "repair_used": 0},
            None, None, 0))

        pvt_calls = []
        monkeypatch.setattr("silq.pvt_repair.apply_pvt_stage",
                            lambda result, *a, **k: pvt_calls.append(1))

        r = pipeline.design(8.92, 14.83)
        assert r["status"] == pipeline.SOLVED
        assert 0 < len(r["alternatives"]["items"]) <= pipeline.ALTERNATIVES_CAP
        assert r["alternatives"]["scope"].startswith("nominal")
        assert len(pvt_calls) == 1

    def test_auto_passes_alternatives_through_untouched(self, monkeypatch):
        sentinel = {"cap": 3, "pool_size": 0, "tried": [], "items": [], "scope": "nominal"}

        def fake_design(t, c, *, mode, **kw):
            return {"status": pipeline.SOLVED, "mode": mode, "guidance": {"headline": "h"},
                    "solver": {"best_boost_db": 8.0},
                    "cost": {"optimizer_evals": 4, "measure_all_total": 5,
                             "spice_analyses_total": 30},
                    "alternatives": sentinel}
        monkeypatch.setattr(pipeline, "design", fake_design)
        r = pipeline._auto(9.0, 14.0, {})
        assert r["alternatives"] is sentinel
