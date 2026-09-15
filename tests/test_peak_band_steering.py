"""The requested peak-frequency band must STEER the search, not just judge it afterwards.

THE DEFECT THIS FILE PINS DOWN. `peak_freq_lo_ghz`/`peak_freq_hi_ghz` are settable
requirements and appear in `hard_pass`, so a run that missed the band was correctly
reported as failed. But nothing in the SEARCH could see them: `Evaluation.make_eval` built
its acceptance spec from DEFAULT_SPEC, so `loose_pass` -- which is what `g32_solve` repairs
against and what `summarize` selects the returned design from -- meant "inside the
COMPETITION band" no matter what the user asked for; and `g32_solve`'s 2-D peak repair
aimed at the probe's band rather than the requested one. The band was a filter bolted onto
the end of a search that had never heard of it.

MEASURED BEFORE THE FIX (fastest mode, target 8 dB, channel 12 dB): asking for the
default band, for 2.00-2.50 GHz, and for 1.25-1.60 GHz all returned the SAME circuit --
7.99946757129063 dB at 1.5093219 GHz, `g32_repair_used` 0 in every case. Only the verdict
differed: the 2.00-2.50 request was failed on `peak_in_band`. AFTER: the same request
returns a different circuit peaking at 2.0134 GHz, on target, passing.

The sharpest tests here are the ones in TestNothingMovesAtTheCompetitionDefaults: with no
requirement set every steering input is None and the frozen path is reconstructed exactly.
That is what keeps this repo's published numbers reproducible while the feature exists.
"""
from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eqrl import pipeline as pl                                    # noqa: E402
from eqrl.experiments import final_comparison as fc                # noqa: E402
from eqrl.sim.measures import Measures                             # noqa: E402
from eqrl.specs import DEFAULT_SPEC                                # noqa: E402

#: A design that passes every check EXCEPT, for some bands, where its peak sits. Every
#: other value is comfortably inside the competition limits so only the band is in play.
IN_DEFAULT_BAND = Measures(dc_gain_db=2.0, peak_gain_db=10.0, boost_db=8.0,
                           peak_freq_ghz=1.50, hd3_db=-40.0, noise_vrms=0.5e-3,
                           power_w=8e-3, area_mm2=0.02, eye_h_ui=0.6, eye_v_mv=180.0)


class _Verdict:
    is_valid = True
    check = None

    def __init__(self, m):
        self._m = m

    def unwrap(self):
        return self._m


class _Guard:
    """Stands in for the guarded evaluator so these tests need no simulator.

    It returns the SAME measurement for every design, which is the point: any difference
    in `loose_pass` below therefore comes from the acceptance spec and nothing else.
    """

    def __init__(self, m):
        self.m = m

    def evaluate(self, dv, vdd):
        return _Verdict(self.m)


def _eval_with(accept, m=IN_DEFAULT_BAND, channel=12.0):
    ev = fc.Evaluation()
    ev.guards[channel] = _Guard(m)
    return ev.make_eval(channel, accept=accept)


def _spec(**over):
    return dataclasses.replace(DEFAULT_SPEC, target_boost_db=8.0, channel_loss_db=12.0,
                               **over)


def _source(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


class TestFeasibilityFollowsTheUsersBand:
    """`loose_pass` is the search's steering signal, so it must mean the user's band."""

    def test_a_peak_outside_the_requested_band_is_not_feasible(self):
        evaluate = _eval_with(_spec(peak_freq_lo_ghz=2.0))
        rec, _score, _g = evaluate(np.full(6, 0.5), 8.0)
        assert rec["loose_pass"] is False, (
            "a 1.50 GHz peak cannot be 'feasible' for a run that asked for 2.00-2.50 GHz; "
            "if it is, g32_solve has nothing to repair and summarize returns it anyway")
        assert "peak_in_band" in rec["failing"]

    def test_the_same_peak_is_feasible_when_the_band_allows_it(self):
        evaluate = _eval_with(_spec(peak_freq_lo_ghz=1.25, peak_freq_hi_ghz=1.60))
        rec, _score, _g = evaluate(np.full(6, 0.5), 8.0)
        assert rec["loose_pass"] is True
        assert rec["failing"] == []

    def test_the_competition_verdict_is_recorded_alongside_the_users(self):
        """So a search that steered on a tighter bar can still report what it found."""
        evaluate = _eval_with(_spec(peak_freq_lo_ghz=2.0))
        rec, _score, _g = evaluate(np.full(6, 0.5), 8.0)
        assert rec["competition_pass"] is True, (
            "the design meets the competition band; only the user's band rejects it")
        assert rec["competition_failing"] == []

    def test_the_guard_is_never_relaxed_by_a_user_requirement(self):
        """Acceptance is the user's to set. Device physics is not."""
        src = _source("src/eqrl/experiments/final_comparison.py")
        block = src[src.index("def guard_for("):src.index("def make_eval(")]
        assert "DEFAULT_SPEC" in block, (
            "guard_for must keep building on DEFAULT_SPEC; a user-relaxed guard would let "
            "a design that is not a real circuit through")

    def test_the_boost_target_is_not_folded_into_feasibility(self):
        """`hard_pass` would add a `boost_target` check; Stage B needs it absent.

        g32_solve walks boost toward the target THROUGH the feasible set. If being on
        target were a condition of being feasible there would be nothing left to walk.
        """
        evaluate = _eval_with(_spec(boost_target_tol_db=0.01))
        rec, _score, _g = evaluate(np.full(6, 0.5), 11.0)   # 3 dB off target
        assert "boost_target" not in rec["failing"]
        assert rec["loose_pass"] is True


class TestNothingMovesAtTheCompetitionDefaults:
    """The frozen benchmark must be reconstructed exactly when no requirement is set."""

    def test_no_requirement_means_no_search_spec(self):
        user = pl.spec_for(8.0, 12.0, 1.5)
        assert pl._search_spec(user, pl.requirements_diff(user)) is None

    def test_no_requirement_means_no_band_override(self):
        user = pl.spec_for(8.0, 12.0, 1.5)
        assert pl._search_band(user, pl.requirements_diff(user)) is None

    def test_a_non_band_requirement_does_not_retarget_the_repair(self):
        """Tightening power is a gate, not a reason to aim the peak somewhere new."""
        user = pl.spec_for(8.0, 12.0, 1.5, {"power_w_max": 10e-3})
        diff = pl.requirements_diff(user)
        assert diff, "fixture must actually set a requirement"
        assert pl._search_band(user, diff) is None
        assert pl._search_spec(user, diff) is not None, (
            "a tightened power ceiling should still steer feasibility")

    def test_an_unsteered_record_keeps_the_frozen_shape(self):
        """`accept=None` must not add the two keys the steered path records."""
        evaluate = _eval_with(None)
        rec, _score, _g = evaluate(np.full(6, 0.5), 8.0)
        assert set(rec) == {"boost_db", "peak_freq_ghz", "dc_gain_db", "loose_pass",
                            "failing", "design"}

    def test_the_probes_band_is_the_competition_band(self):
        """Which is why band=None and passing the default band are the same search."""
        plane = fc.plane_from_probe("results/g32_peak_probe.json")
        assert plane["band_ghz"] == [DEFAULT_SPEC.peak_freq_lo_ghz,
                                     DEFAULT_SPEC.peak_freq_hi_ghz]

    def test_only_a_band_ask_may_reorder_the_starts(self):
        """PPO first, corpus as rescue, is the FROZEN order.

        Every published Thinking number was produced by it. Band steering reverses that
        order -- the corpus is the only generator that can see a requested band -- but it
        must do so only when a band was actually asked for. So what this pins is the
        GUARD, not the reordering: widen it to `req_diff` (a tightened power ceiling, say)
        and the frozen order silently changes on runs that are reproducing the benchmark,
        and nothing else in this repo would notice.
        """
        src = _source("src/eqrl/pipeline.py")
        i = src.index("band_first = search_band is not None")
        block = src[i:i + 1400]
        assert "reached = _run_seeds(_corpus_seeds(None)) if band_first else False" in block, (
            "the band-first draw must be conditional on band_first alone")
        assert "if (not band_first and candidates" in block, (
            "the post-PPO rescue draw must not fire when the same corpus pool was already "
            "drawn before PPO -- that would spend the budget on it twice")

    def test_the_ppo_loop_runs_unless_the_band_first_draw_already_finished(self):
        """AST, not a string match: the rollout loop must sit inside `if not reached:`.

        A reordering that left the PPO loop unguarded would still pass the source test
        above and would still measure faster on a band ask, while quietly paying for five
        blind rollouts nobody needed.
        """
        tree = ast.parse(_source("src/eqrl/pipeline.py"))

        def _rollout_loops(node):
            return [n for n in ast.walk(node) if isinstance(n, ast.For)
                    and ast.unparse(n.iter).startswith("THINKING_SEED_OFFSETS")]

        assert len(_rollout_loops(tree)) == 1, (
            "expected exactly one PPO rollout loop in pipeline.py")
        guards = [n for n in ast.walk(tree)
                  if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
                  and isinstance(n.test.op, ast.Not)
                  and isinstance(n.test.operand, ast.Name)
                  and n.test.operand.id == "reached"]
        assert len(guards) == 1, "expected exactly one `if not reached:` guard"
        assert _rollout_loops(guards[0]), (
            "the PPO rollout loop must sit inside `if not reached:`, or a corpus start "
            "that already reached target still pays for every blind rollout")

    def test_the_seed_runner_cannot_stop_before_it_has_spent_anything(self):
        """`candidates and` on the governor guard is load-bearing on the band-first path.

        Without it the pre-PPO draw is measured against a budget nothing has spent yet;
        the check happens to pass today because `_spent()` starts at 0, but the guard is
        what makes that true by construction rather than by arithmetic accident.
        """
        src = _source("src/eqrl/pipeline.py")
        i = src.index("def _run_seeds(seeds)")
        block = src[i:i + 900]
        assert "if candidates and _spent() >= THINKING_MEASURE_ALL_BUDGET:" in block


class TestTheBandReachesTheRepairLoop:
    def test_search_band_is_returned_when_an_edge_moves(self):
        user = pl.spec_for(8.0, 12.0, 1.5, {"peak_freq_lo_ghz": 2.0})
        assert pl._search_band(user, pl.requirements_diff(user)) == (2.0, 2.5)

    def test_search_spec_drops_the_boost_target_tolerance(self):
        user = pl.spec_for(8.0, 12.0, 1.5, {"peak_freq_lo_ghz": 2.0})
        assert user.boost_target_tol_db == 1.5
        assert pl._search_spec(user,
                               pl.requirements_diff(user)).boost_target_tol_db is None

    def test_g32_solve_aims_at_the_requested_bands_geometric_mean(self):
        """The aim point is derived the same way `plane_from_probe` derives its own."""
        src = _source("src/eqrl/experiments/final_comparison.py")
        i = src.index("def g32_solve(")
        block = src[i:i + 3000]
        assert "band is None" in block, "g32_solve must honour a caller-supplied band"
        assert "(LO * HI) ** 0.5" in block, (
            "AIM must be the requested band's geometric mean, matching plane_from_probe")

    def test_every_g32_solve_call_site_passes_the_band(self):
        """A call site that forgets `band=` silently reverts to the old behaviour."""
        tree = ast.parse(_source("src/eqrl/pipeline.py"))
        sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "g32_solve"]
        assert sites, "no g32_solve call sites found -- has the search been restructured?"
        for site in sites:
            assert any(kw.arg == "band" for kw in site.keywords), (
                f"the g32_solve call at line {site.lineno} does not pass band=; that arm "
                f"would keep aiming at the probe's band whatever the user asked for")

    def test_the_search_evaluator_is_built_with_the_acceptance_spec(self):
        tree = ast.parse(_source("src/eqrl/pipeline.py"))
        sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "make_eval"]
        assert sites and all(any(kw.arg == "accept" for kw in s.keywords) for s in sites)

    def test_fastest_seeds_the_corpus_lookup_from_the_users_band(self):
        """The corpus records each design's own peak; the lookup must be given the band."""
        src = _source("src/eqrl/pipeline.py")
        i = src.index('elif mode == "fastest":')
        block = src[i:i + 1500]
        assert "seed_spec = user_spec if req_diff" in block, (
            "Fastest's corpus seed must narrow to the requested band; otherwise its "
            "3-evaluation stage-2 budget is spent walking there from an arbitrary start")


class TestAnUnreachableBandStillReturnsACircuit:
    """Steering toward a bar nothing clears must not turn a near-miss into nothing."""

    def _trace(self):
        return [
            {"boost_db": 8.0, "peak_freq_ghz": 1.50, "dc_gain_db": 2.0,
             "loose_pass": False, "failing": ["peak_in_band"],
             "competition_pass": True, "competition_failing": [],
             "design": {"w_in": 10.0}},
            {"boost_db": 6.0, "peak_freq_ghz": 1.40, "dc_gain_db": 2.0,
             "loose_pass": False, "failing": ["peak_in_band"],
             "competition_pass": True, "competition_failing": [],
             "design": {"w_in": 20.0}},
        ]

    def test_the_relaxed_summary_recovers_the_best_competition_design(self):
        out = pl._summarize_relaxed(fc, self._trace(), 8.0, 1.5)
        assert out["best_design"] == {"w_in": 10.0}, (
            "the 8.0 dB design is nearest the 8.0 dB target under the competition bar")

    def test_the_strict_summary_finds_nothing(self):
        """Which is exactly why the fallback exists."""
        assert fc.summarize(self._trace(), 8.0, 1.5)["best_design"] is None

    def test_the_relaxed_summary_does_not_mutate_the_trace(self):
        trace = self._trace()
        pl._summarize_relaxed(fc, trace, 8.0, 1.5)
        assert all(e["loose_pass"] is False for e in trace), (
            "the fallback must re-label a COPY; mutating the trace would hand the "
            "verifier and the Pareto stage designs marked as passing a bar they failed")

    def test_a_record_with_no_competition_verdict_falls_back_to_its_own(self):
        """Unsteered records carry no `competition_pass`; they must still summarize."""
        trace = [{"boost_db": 8.0, "peak_freq_ghz": 1.5, "dc_gain_db": 2.0,
                  "loose_pass": True, "failing": [], "design": {"w_in": 10.0}}]
        assert pl._summarize_relaxed(fc, trace, 8.0, 1.5)["best_design"] == {"w_in": 10.0}

    def test_the_result_says_when_it_fell_back(self):
        """A relaxed answer that did not announce itself would be the worst outcome."""
        src = _source("src/eqrl/pipeline.py")
        assert "fell_back_to_competition_bar" in src
        i = src.index("steering_relaxed = False")
        block = src[i:i + 1800]
        assert "requirement_steering" in block
        assert "peak_band_ghz" in block, (
            "the result must record which band the search actually aimed at")


class TestTheLiveNarrationWrapperDoesNotDropArguments:
    """`server._narrating` monkeypatches the frozen solver; it must stay transparent.

    This class exists because of a bug the rest of this file could not see. Every test
    above calls `fc.Evaluation.make_eval` directly, but a real request runs inside
    `server._narrating()`, which REPLACES `make_eval` with a wrapper that emits a progress
    event per candidate. That wrapper spelled its parameters out as `(self, channel)`, so
    the moment `make_eval` grew `accept=` every live run died with

        TypeError: _narrating.<locals>.make_eval() got an unexpected keyword argument 'accept'

    -- a 500 on the dashboard for all three modes, default band included, with the whole
    unit suite green. Found by running the production `runtime.Runtime.run` path, not by a
    test. These tests make the wrapper's transparency itself the thing under test.
    """

    def test_the_wrapper_forwards_every_keyword_the_real_one_takes(self):
        """Bind the frozen signature's kwargs against the wrapper's, inside narration."""
        import inspect
        server = pytest.importorskip("server")
        real = inspect.signature(fc.Evaluation.make_eval)
        with server._narrating():
            wrapper = fc.Evaluation.make_eval
            assert wrapper is not real, "narration did not install its wrapper"
            sig = inspect.signature(wrapper)
            for name, param in real.parameters.items():
                if param.kind is param.POSITIONAL_OR_KEYWORD and name not in ("self",):
                    # Either named outright, or swallowed by *args/**kwargs.
                    ok = name in sig.parameters or any(
                        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                        for p in sig.parameters.values())
                    assert ok, f"narration wrapper drops `{name}` from make_eval"

    def test_the_wrapper_restores_the_frozen_function(self):
        """A leaked wrapper narrates later runs through a dead closure."""
        server = pytest.importorskip("server")
        before = fc.Evaluation.make_eval
        with server._narrating():
            pass
        assert fc.Evaluation.make_eval is before

    def test_g32_solve_narration_is_already_transparent(self):
        """The sibling wrapper takes `*a, **kw`; `band=` reaches the solver through it."""
        import inspect
        server = pytest.importorskip("server")
        with server._narrating():
            kinds = {p.kind for p in
                     inspect.signature(fc.g32_solve).parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD in kinds, (
            "g32_solve's narration wrapper must keep **kw or `band=` cannot reach the "
            "repair loop on a live request")
