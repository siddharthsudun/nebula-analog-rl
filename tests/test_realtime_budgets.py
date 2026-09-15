"""Budget arithmetic and startup pre-warming for the live (wall-clock bounded) path.

Two defects motivated this file, both of which returned NO design rather than a slow one,
and neither of which any existing test could see:

1. The surrogate corpus (374,588 records, ~1.7 s to build a kNN tree over) was not
   pre-warmed on the in-process path, and `pipeline.design` builds it inside the request
   before the first evaluation. That is longer than Fastest's entire search window, so a
   cold server failed its first Fastest request on evaluation zero.
2. Fastest reserved 72% of its 5 s budget for a PVT certification stage it never ran,
   leaving 1.4 s for a search that needs roughly four guarded SPICE evaluations.

Fastest now DOES certify, against the three stress corners rather than all 45 (see
`realtime.GRID_MODE`), so defect 2 is live again in a new form: the reserve must cover a
real sweep without falling back below the 4 s the search needs. Both ends are asserted
below, and the 3-corner sweep was measured at 0.50 s warm against 3.12 s for 45.

Both are static/arithmetic properties, so these tests need no simulator.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silq import realtime                                       # noqa: E402


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() is missing from server.py")


class TestStartupPrewarm:
    """Anything costing more than a request's budget must be paid for before the request.

    The user's constraint was explicit -- "initialise those workers pre-simulation if
    reqd" -- and it applies to the surrogate exactly as it does to the PVT pool.
    """

    def test_warm_up_builds_the_surrogate_corpus(self):
        warm = _function((ROOT / "server.py").read_text(encoding="utf-8"), "_warm_up")
        called = {n.func.id for n in ast.walk(warm)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "load_fastest_assets" in called, (
            "the surrogate corpus costs more to build than Fastest's whole search "
            "window; if it is not warmed at startup the first Fastest request returns "
            "no design at all")

    def test_warm_up_still_builds_the_pool_evaluator_and_policy(self):
        """The new pre-warm must be additive -- it cannot displace the existing ones."""
        warm = _function((ROOT / "server.py").read_text(encoding="utf-8"), "_warm_up")
        called = {n.func.id for n in ast.walk(warm)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for required in ("prepare", "get_evaluator", "load_policy"):
            assert required in called, f"{required} was dropped from warm-up"
        worker = _function((ROOT / "src/silq/runtime.py").read_text(encoding="utf-8"), "worker_main")
        worker_calls = {n.func.id for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert {"get_pool","load_policy","load_fastest_assets"} <= worker_calls


class TestSearchReserve:
    def test_fastest_reserve_covers_a_three_corner_sweep_without_starving_the_search(self):
        """Both ends matter, and they pull against each other inside one 5 s budget."""
        budget = realtime.LIMITS["fastest"]
        reserve = realtime.search_reserve("fastest", budget)
        assert reserve >= 1.0, (
            "Fastest certifies tt/ss/ff now; a sweep measured at 0.50 s warm plus the one "
            "in-flight evaluation the search may overrun by does not fit in less")
        assert budget - reserve >= 4.0, (
            "Fastest needs room for one surrogate seed plus FASTEST_BUDGET refinements")
        assert reserve < realtime.FULL_RESERVE, (
            "3 corners cost a fraction of 45; reserving the certifying modes' share would "
            "spend Fastest's budget on time it cannot use")

    def test_fastest_certifies_the_reduced_grid_and_nothing_else_does(self):
        """The mode->grid map is the single place this decision is made."""
        assert realtime.GRID_MODE.get("fastest") == "reduced"
        for mode in ("auto", "thinking", "default", "retarget", "g32_acceptance"):
            assert realtime.GRID_MODE.get(mode, "full") == "full", (
                f"{mode} must still certify all 45 corners")

    def test_the_reduced_grid_is_exactly_tt_ss_ff(self):
        """What "reduced" means is defined in envs.pvt, not restated here."""
        from silq.envs.pvt import corner_grid
        from silq import pipeline as pl
        spec = pl.spec_for(9.0, 12.0, 1.5, None)
        assert [p for p, _, _ in corner_grid(spec, "reduced")] == ["tt", "ss", "ff"]
        assert len(corner_grid(spec, "full")) == 45

    @pytest.mark.parametrize("mode", ["auto", "thinking", "default", "retarget"])
    def test_certifying_modes_keep_a_real_reserve(self, mode):
        budget = realtime.LIMITS[mode]
        reserve = realtime.search_reserve(mode, budget)
        assert reserve >= 1.0, (
            f"{mode} certifies, so it must hold time back for the corner sweep")

    @pytest.mark.parametrize("mode", ["fastest", "auto", "thinking"])
    def test_reserve_never_consumes_the_whole_budget(self, mode):
        budget = realtime.LIMITS[mode]
        assert 0 < realtime.search_reserve(mode, budget) < budget

    def test_a_tiny_budget_still_leaves_some_search_time(self):
        """The proportional term protects modes whose budget is smaller than the reserve."""
        assert realtime.search_reserve("auto", 2.0) < 2.0


class TestBudgetOverrunStaysHonest:
    """Running out of time must degrade to an honest partial, never to a 500.

    `pvt_workers.remaining()` RAISES queue.Empty once the deadline passes rather than
    returning, and the certify call sits outside design_realtime's try/finally. Before this
    was guarded, an overrun escaped to the API as a bare 500 with an empty message and
    discarded a circuit the engineer already had on screen -- the opposite of the agreed
    contract, which is to return best effort and say what is unverified.
    """

    def test_certify_call_is_guarded(self):
        source = (ROOT / "src" / "silq" / "realtime.py").read_text(encoding="utf-8")
        design_realtime = None
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == "design_realtime":
                design_realtime = node
        assert design_realtime is not None

        guarded = set()
        for node in ast.walk(design_realtime):
            if isinstance(node, ast.Try):
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
                        guarded.add(inner.func.id)
        assert "certify" in guarded, (
            "certify() must run inside a try block; an overrun raises queue.Empty and "
            "would otherwise reach the client as an unexplained 500")

    def test_overrun_handler_does_not_swallow_interrupts(self):
        """Exception, not BaseException -- Ctrl-C and SystemExit must still propagate."""
        source = (ROOT / "src" / "silq" / "realtime.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Try):
                continue
            calls = {i.func.id for i in ast.walk(node)
                     if isinstance(i, ast.Call) and isinstance(i.func, ast.Name)}
            if "certify" not in calls:
                continue
            for handler in node.handlers:
                assert handler.type is not None, "a bare except would swallow interrupts"
                name = getattr(handler.type, "id", None)
                assert name != "BaseException", (
                    "catching BaseException would swallow KeyboardInterrupt/SystemExit")

    def test_partial_result_is_marked_unverified_and_keeps_the_circuit(self):
        """checkpoint_result is what the overrun branch returns: it must not claim a pass."""
        result = {
            "design": {"w_in": 1.0}, "netlist": "* circuit",
            "status": "solved", "overall_passed": True,
            "pvt": {"accepted": False, "status": "pending"},
        }
        partial = realtime.checkpoint_result(result)
        assert partial["design"] == {"w_in": 1.0}, "the measured circuit must survive"
        assert partial["overall_passed"] is False
        assert partial["status"] == "pvt_not_verified"
        assert partial["pvt"]["accepted"] is False
        assert partial["pvt"]["cost_complete"] is False
        assert "pending" in partial["pvt"]["scope"].lower()

    def test_partial_result_does_not_leak_internal_trace(self):
        partial = realtime.checkpoint_result({
            "design": {"w_in": 1.0}, "_pareto_trace": [{"secret": 1}],
            "pvt": {"accepted": False, "status": "pending"}})
        assert "_pareto_trace" not in partial

    def test_an_accepted_pvt_result_is_left_alone(self):
        """Only unverified results get rewritten -- a real pass must not be downgraded."""
        partial = realtime.checkpoint_result({
            "design": {"w_in": 1.0}, "status": "solved", "overall_passed": True,
            "pvt": {"accepted": True, "status": "verified"}})
        assert partial["pvt"]["accepted"] is True
        assert partial["status"] == "solved"


class TestModeBudgets:
    """The ceilings the user set: Fastest 5 s, Auto 15 s, Thinking 30-40 s."""

    @pytest.mark.parametrize("mode,ceiling", [("fastest", 5.0), ("auto", 15.0), ("thinking", 40.0)])
    def test_declared_budget_matches_the_requested_ceiling(self, mode, ceiling):
        assert realtime.LIMITS[mode] == ceiling

    def test_every_ui_mode_has_a_budget(self):
        for mode in ("fastest", "auto", "thinking"):
            assert mode in realtime.LIMITS
