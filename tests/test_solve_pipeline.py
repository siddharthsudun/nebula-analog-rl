"""The public entry point must run the delivered architecture, and must never present the
fixed fallback design as something the AI produced.

Two layers, deliberately:

  PDK-FREE   the labelling contract. Every branch of `pipeline.design` is exercised with
             the architecture stubbed out, so CI enforces -- on every push, forever --
             that a fixed design can never come back wearing `is_ai_generated: True`.
             This is the test that matters for honesty, and it must not be skippable.

  PDK-GATED  the architecture itself. Runs PPO -> G3.2 -> verification for real and
             checks that the entry point reproduces `results/delivered_circuit.json`,
             the circuit the 45-corner sign-off was performed on.

Nothing here changes a reward, bound, guard, controller constant or benchmark criterion.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest

from eqrl import pipeline
from eqrl.baselines.robust import robust_design
from eqrl.circuits import pdk

ROOT = Path(__file__).resolve().parent.parent
DELIVERED = ROOT / "results" / "delivered_circuit.json"


# ======================================================================================
# PDK-free: the labelling contract
# ======================================================================================
class _FakeSolver:
    """A stand-in for experiments.final_comparison with scriptable outcomes.

    It reproduces only the surface `pipeline.design` touches. The point is to drive every
    branch -- solved, unsolved, fallback -- without a simulator, so the labelling rules
    are checked by CI rather than by a machine that happens to have SKY130 installed.
    """

    PREREG = {"k": 5, "r": 10, "budget_measure_all": 20,
              "ppo_measure_all_per_eval": 2.0, "search_measure_all_per_eval": 1.0,
              "tol": 1.5}

    def __init__(self, *, best_design, reached=True, n_steps=3):
        self._best = best_design
        self._reached = reached
        self._n_steps = n_steps

    # -- the bits pipeline.design calls ------------------------------------------------
    def _check_constants(self):
        return None

    def plane_from_probe(self, path):
        return {"boost_axis": "rs", "peak_axis": "l_in"}

    def rescue_order_from_probe(self, path):
        return []

    class Evaluation:
        def __init__(self):
            self.n_sim = 7

        def make_eval(self, channel):
            def evaluate(x, target):
                return None, -10.0, "stub"
            return evaluate

    def load_policy(self, model, *, seed=123):
        return object(), object()

    def stage1_rollout(self, evaluate, policy, env, i, target, channel, k):
        xs = [np.zeros(6)]
        trace = [None]
        return xs, trace, None

    def g32_solve(self, evaluate, xs, s1, target, plane, ladder, budget):
        info = {"steps": [{"phase": "advance", "boost_db": 7.0, "abs_err": 1.9}]
                         * self._n_steps,
                "start_source": "stage1 (free)", "reached_target": self._reached,
                "reason": "stub", "rescue_used": 0, "repair_used": 0}
        return [], info, None, None, budget - self._n_steps

    def summarize(self, trace, target, tol):
        return {"best_design": self._best, "best_boost_db": None, "best_abs_err": None,
                "n_evals": 5, "n_valid": 0, "strict_solved_at": None,
                "loose_solved_at": None}


@pytest.fixture
def stub(monkeypatch):
    """Install a fake solver + a fake verifier + a fake netlist writer."""
    def install(*, best_design, passed=True, reached=True):
        fake = _FakeSolver(best_design=best_design, reached=reached)
        monkeypatch.setattr(pipeline, "_fc", lambda: fake)
        monkeypatch.setattr(pipeline, "netlist", lambda *a, **k: "* stub netlist\n")
        monkeypatch.setattr(pipeline, "verify", lambda dv, spec: {
            "guard_valid": True, "guard_check": None, "passed": passed,
            "checks": {}, "failing": [] if passed else ["boost_target"],
            "measures": {"boost_db": 9.0, "peak_freq_ghz": 1.6, "dc_gain_db": 1.2,
                         "hd3_db": -55.0, "noise_vrms": 6e-4, "power_w": 2e-3,
                         "area_mm2": 1e-3, "eye_h_ui": 0.75, "eye_v_mv": 650.0},
            "abs_err_db": 0.08})
        return fake
    return install


AI_DESIGN = {"w_in": 55.784e-6, "l_in": 0.3921e-6, "i_tail": 712.27e-6,
             "rs": 3287.45, "cs": 181.62e-15, "r_load": 2450.30, "w_dfe": 0.0}


class TestTheArchitectureIsTheNormalPath:
    def test_a_solved_run_is_labelled_ai_generated_and_used_g32(self, stub):
        stub(best_design=AI_DESIGN)
        r = pipeline.design(8.92, 14.83)
        assert r["status"] == pipeline.SOLVED
        assert r["provenance"]["is_ai_generated"] is True
        assert r["provenance"]["g32_used"] is True
        assert r["provenance"]["fallback_invoked"] is False
        assert r["design"] == AI_DESIGN

    def test_the_returned_circuit_is_the_solvers_circuit(self, stub):
        """The bug this whole module exists to prevent: shipping a different circuit."""
        stub(best_design=AI_DESIGN)
        r = pipeline.design(8.92, 14.83)
        assert r["design"] != dataclasses.asdict(robust_design())
        assert r["design"] is not None and r["design"]["rs"] == AI_DESIGN["rs"]

    def test_provenance_records_every_stage(self, stub):
        stub(best_design=AI_DESIGN)
        p = pipeline.design(8.92, 14.83)["provenance"]
        for field in ("policy", "ppo_produced_handoff", "g32_used", "g32_start_source",
                      "g32_steps", "g32_reached_target", "fallback_invoked",
                      "is_ai_generated", "hand_tuning"):
            assert field in p, field

    def test_cost_is_reported_in_both_units(self, stub):
        stub(best_design=AI_DESIGN)
        c = pipeline.design(8.92, 14.83)["cost"]
        assert c["stage1_evals"] == 5 and c["stage2_evals"] == 3
        assert c["measure_all_spent"] == 5 * 2.0 + 3 * 1.0
        assert c["measure_all_budget"] == 20

    def test_closing_but_failing_verification_is_its_own_status(self, stub):
        """A design G3.2 was happy with but the independent re-measure rejects."""
        stub(best_design=AI_DESIGN, passed=False)
        r = pipeline.design(8.92, 14.83)
        assert r["status"] == pipeline.CLOSED_NOT_VERIFIED
        assert r["verification"]["passed"] is False
        assert r["provenance"]["is_ai_generated"] is True


class TestG32CannotCompleteAndFallbackIsExplicit:
    def test_no_guard_valid_design_and_no_fallback_returns_nothing(self, stub):
        stub(best_design=None, reached=False)
        r = pipeline.design(8.92, 14.83)
        assert r["status"] == pipeline.UNSOLVED
        assert r["design"] is None
        assert r["verification"] is None
        assert r["provenance"]["fallback_invoked"] is False
        # An unsolved AI run is still an AI run. It reports failure, not a substitute.
        assert r["provenance"]["is_ai_generated"] is True

    def test_fallback_is_off_unless_asked_for(self, stub):
        stub(best_design=None, reached=False)
        assert pipeline.design(8.92, 14.83)["design"] is None

    def test_fallback_is_never_labelled_ai_generated(self, stub):
        stub(best_design=None, reached=False)
        r = pipeline.design(8.92, 14.83, allow_fallback=True)
        assert r["status"] == pipeline.FALLBACK
        assert r["provenance"]["is_ai_generated"] is False
        assert r["provenance"]["fallback_invoked"] is True
        assert "not an output of the SILQ architecture" in \
            r["provenance"]["fallback_note"]

    def test_the_fallback_design_is_the_fixed_baseline_not_a_search_result(self, stub):
        stub(best_design=None, reached=False)
        r = pipeline.design(8.92, 14.83, allow_fallback=True)
        assert r["design"] == dataclasses.asdict(robust_design())

    def test_fallback_does_not_fire_when_the_architecture_produced_a_design(self, stub):
        """Even with the flag on: the escape hatch is for nothing-came-back, not for
        'the target was missed'. A missed target is a result, and it is reported."""
        stub(best_design=AI_DESIGN, reached=False, passed=False)
        r = pipeline.design(8.92, 14.83, allow_fallback=True)
        assert r["status"] == pipeline.CLOSED_NOT_VERIFIED
        assert r["provenance"]["fallback_invoked"] is False
        assert r["provenance"]["is_ai_generated"] is True

    def test_describe_shouts_about_the_fallback(self, stub):
        stub(best_design=None, reached=False)
        text = pipeline.describe(pipeline.design(8.92, 14.83, allow_fallback=True))
        assert "NOT generated by PPO or G3.2" in text

    def test_describe_does_not_shout_about_a_real_result(self, stub):
        stub(best_design=AI_DESIGN)
        text = pipeline.describe(pipeline.design(8.92, 14.83))
        assert "FIXED FALLBACK" not in text
        assert "PPO (5 evals) -> G3.2 (3 evals)" in text


class TestTheResultSurvivesBeingWrittenDown:
    """`eqrl.solve` dumps the result to JSON. A run that solves the spec and then dies in
    the encoder has not delivered anything, so the coercion is tested, not assumed."""

    def test_numpy_scalars_are_coerced(self):
        got = pipeline._plain({"ok": np.bool_(True), "n": np.int64(3),
                               "x": np.float64(1.5), "steps": [np.bool_(False)]})
        assert got == {"ok": True, "n": 3, "x": 1.5, "steps": [False]}
        assert type(got["ok"]) is bool and type(got["n"]) is int
        json.dumps(got)

    def test_plain_values_pass_through_untouched(self):
        d = {"a": 1.0, "b": None, "c": "s", "d": {"e": [1, 2]}}
        assert pipeline._plain(d) == d

    def test_a_stubbed_result_is_json_serialisable(self, stub):
        stub(best_design=AI_DESIGN)
        json.dumps(pipeline.design(8.92, 14.83))


class TestTheEntryPointUsesThePipeline:
    def test_solve_imports_the_architecture_and_not_a_private_copy(self):
        """`eqrl.solve` must be a front end, not a second implementation."""
        import inspect

        from eqrl import solve
        src = inspect.getsource(solve)
        assert "from eqrl.pipeline import" in src
        assert "robust_design" not in src, \
            "solve.py must not reach for the fixed design itself; the labelled fallback " \
            "lives in pipeline.design(allow_fallback=True)"

    def test_solve_does_not_overwrite_the_historical_artifact(self):
        """results/solved_design.json is cited by README/RESULTS/HANDOVER as the design
        that fails 10 of 45 corners. A demo run must not quietly rewrite it."""
        import inspect

        from eqrl import solve
        m = re.search(r'"--out",\s*default="([^"]+)"', inspect.getsource(solve))
        assert m, "could not find the --out default in solve.py"
        assert m.group(1) != "results/solved_design.json"


# ======================================================================================
# PDK-gated: the architecture itself
# ======================================================================================
def _sim_available() -> bool:
    if not pdk.available():
        return False
    try:
        from eqrl.sim.server import get_server
        get_server("tt")
        return True
    except Exception:
        return False


needs_sim = pytest.mark.skipif(not _sim_available(),
                               reason="needs the SKY130 PDK and a usable libngspice")


# Module-scoped so the whole architecture runs once, not once per assertion: loading the
# policy and driving ngspice costs ~18 s a go.
@pytest.fixture(scope="module")
def manifest():
    if not DELIVERED.exists():
        pytest.skip("results/delivered_circuit.json not present")
    return json.loads(DELIVERED.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run(manifest):
    """The entry point, driven from the delivered circuit's own specification."""
    s = manifest["spec"]
    return pipeline.design(s["target_boost_db"], s["channel_loss_db"],
                           spec_index=s["spec_index"])


@pytest.fixture(scope="module")
def unreachable():
    # 20 dB is far outside the brief's 3-12 dB range and outside anything the action
    # space reaches, so the solver must fail to close it.
    return pipeline.design(20.0, 14.83, spec_index=2)


@needs_sim
class TestEndToEndReproducesTheDeliveredCircuit:
    """The whole point of the rewiring: the demo and the delivered artifact agree."""

    def test_the_architecture_solved_it(self, run):
        assert run["status"] == pipeline.SOLVED
        assert run["provenance"]["g32_reached_target"] is True
        assert run["provenance"]["fallback_invoked"] is False

    def test_the_six_device_values_match_the_frozen_manifest(self, run, manifest):
        got, want = run["design"], manifest["design"]
        for key in want:
            assert got[key] == pytest.approx(want[key], rel=1e-9), key

    def test_the_independent_verification_reproduces_the_frozen_tt_boost(self, run,
                                                                        manifest):
        want = manifest["pvt"]["tt_nominal"]["boost_db"]
        assert run["verification"]["measures"]["boost_db"] == pytest.approx(want,
                                                                            abs=1e-6)

    def test_it_passes_all_ten_checks_on_re_measurement(self, run):
        assert run["verification"]["guard_valid"] is True
        assert run["verification"]["passed"] is True
        assert run["verification"]["failing"] == []

    def test_it_emits_a_schematic_that_names_real_sky130_devices(self, run):
        assert "sky130_fd_pr__nfet_01v8" in run["netlist"]

    def test_the_budget_was_not_exceeded(self, run):
        assert run["cost"]["measure_all_spent"] <= run["cost"]["measure_all_budget"]

    def test_a_real_result_is_json_serialisable(self, run):
        """The stubbed version of this cannot fail; only a real `hard_pass` returns the
        numpy booleans that broke the first end-to-end `eqrl.solve` run."""
        json.dumps(run)


@needs_sim
class TestAnUnreachableTargetIsReportedNotSubstituted:
    """G3.2 cannot close every specification. When it cannot, the run says so."""

    def test_the_target_was_not_reached(self, unreachable):
        assert unreachable["provenance"]["g32_reached_target"] is False
        assert unreachable["status"] != pipeline.SOLVED

    def test_no_fixed_design_was_substituted(self, unreachable):
        if unreachable["design"] is not None:
            assert unreachable["design"] != dataclasses.asdict(robust_design())
        assert unreachable["provenance"]["fallback_invoked"] is False
        assert unreachable["provenance"]["is_ai_generated"] is True
