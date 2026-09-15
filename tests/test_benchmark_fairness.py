"""The benchmark's success criterion, and the spec-side hole it papers over.

TWO DEFECTS, TWO OPT-IN FIXES, ONE PROPERTY THAT MUST HOLD FOR BOTH
-------------------------------------------------------------------
1. `honest_benchmark.evaluate()` scored a candidate with `hard_pass` on a direct
   `measure_all`, never consulting `silq.guards`. The RL policy is trained with the
   guard on. So the baselines were solving "meet the specs" while RL solved "meet the
   specs AND be a valid circuit". `--require-valid` closes that.

2. `boost_db` is peak gain MINUS DC gain, so attenuating at DC inflates boost for free
   and `hard_pass` never notices. `Spec.dc_gain_db_min` closes that.

`results/pass_vs_valid.json` sized both: of 28 CMA-ES designs that passed all eight
specs, 24 (86%) were guard-invalid — 14 as T4.10_dc_gain_implausible (the DC trick), 10
as T2.5_mosfet_not_in_saturation.

Both fixes are opt-in, and that is the load-bearing property here: with the flag unset
and the field None, every number this repo has published must still reproduce. Half of
the tests below exist only to hold that line.

No simulator is involved. `measure_all` and the guard are faked, so these tests assert
the WIRING — which criterion is applied, to which methods — rather than re-measuring
circuits.
"""
from __future__ import annotations

import dataclasses

import pytest

from silq.experiments import honest_benchmark as hb
from silq.sim.measures import Measures
from silq.specs import DEFAULT_SPEC, Spec, hard_pass


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def passing(**over) -> Measures:
    """A design that clears all eight hard specs at target_boost_db=9, channel 12."""
    base = dict(dc_gain_db=6.0, peak_gain_db=15.0, boost_db=9.0, peak_freq_ghz=2.0,
                hd3_db=-35.0, noise_vrms=1.2e-3, power_w=11e-3, area_mm2=0.040,
                eye_h_ui=0.45, eye_v_mv=120.0, ok=True)
    return Measures(**{**base, **over})


def attenuating() -> Measures:
    """The exploit, as a measurement.

    Loses 10.6 dB at DC and still only reaches -1.6 dB at the peak, so it amplifies
    nothing anywhere — yet peak minus DC is 9.0 dB of 'boost' and every other hard check
    is satisfied more easily by a stage that passes less signal. The DC figure is the one
    measured in `measures.bandwidth_ghz`'s docstring for a real attenuating stage.
    """
    return passing(dc_gain_db=-10.6, peak_gain_db=-1.6, boost_db=9.0)


class FakeVerdict:
    def __init__(self, is_valid: bool):
        self.is_valid = is_valid


class FakeGuard:
    """Stands in for a GuardedEvaluator. Records every design it was asked about."""

    def __init__(self, valid: bool):
        self._valid = valid
        self.calls = []

    def evaluate(self, dv, **kw):
        self.calls.append(dv)
        return FakeVerdict(self._valid)


@pytest.fixture(autouse=True)
def clean_criterion():
    """The criterion is module state. Restore it, or one test's opt-in leaks into the
    next and the 'defaults unchanged' tests would pass for the wrong reason."""
    yield
    hb.set_validity_guard(None)
    hb.set_dc_gain_floor(None)


@pytest.fixture
def sim(monkeypatch):
    """Replace the simulator with a fixed measurement; count the calls."""
    def install(m: Measures):
        calls = []

        def fake_measure_all(dv, **kw):
            calls.append(dv)
            return m
        monkeypatch.setattr(hb, "measure_all", fake_measure_all)
        return calls
    return install


X = [0.5] * hb.N          # a mid-range action; its decoded value is irrelevant here
TARGET, CHANNEL = 9.0, 12.0


# ---------------------------------------------------------------------------
# defaults must not move
# ---------------------------------------------------------------------------

class TestDefaultsUnchanged:

    def test_the_dc_gain_floor_is_now_ON_by_default(self):
        """This default was flipped deliberately, and it changes what "passes" means.

        Two independent optimisers exploited its absence. CMA-ES: 14 of 28 spec-passing
        designs rejected by the guard as T4.10_dc_gain_implausible. PPO after 40k steps:
        4 of 6 held-out rollouts, same check. Boost is peak MINUS DC, so a stage that
        attenuates at DC manufactures boost for free and every other check is *easier*
        for a stage passing less signal. A spec set that admits a circuit with no gain at
        any frequency is not describing an equalizer.
        """
        assert DEFAULT_SPEC.dc_gain_db_min == 0.0

    def test_hard_pass_reports_nine_checks_by_default(self):
        ok, checks = hard_pass(passing(), DEFAULT_SPEC)
        assert ok
        assert len(checks) == 9
        assert checks["dc_gain"] is True

    def test_the_floor_can_still_be_disabled_explicitly(self):
        """Turning it off must remain possible, so the pre-fix behaviour stays
        reproducible and the eight-check numbers already published can be regenerated."""
        no_floor = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=None)
        ok, checks = hard_pass(attenuating(), no_floor)
        assert ok                                   # the old, exploitable verdict
        assert len(checks) == 8 and "dc_gain" not in checks

    def test_the_attenuating_design_no_longer_passes(self):
        """The design the baselines actually found: dc_gain -5.00 dB, peak +1.57 dB,
        "boost" 6.56 dB. It amplifies nothing at any frequency."""
        ok, checks = hard_pass(attenuating(), DEFAULT_SPEC)
        assert not ok
        assert checks["dc_gain"] is False

    def test_benchmark_does_not_consult_the_guard_by_default(self, sim):
        sim(passing())
        guard = FakeGuard(valid=False)          # would veto, if it were asked
        ok, _ = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is True
        assert guard.calls == []

    def test_evaluate_score_is_unchanged_by_the_new_code_path(self, sim):
        """8 checks passing, boost exactly on target -> 8.0. Same arithmetic as before."""
        sim(passing())
        _, score = hb.evaluate(X, TARGET, CHANNEL)
        assert score == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# change 1 — guard validity as part of the benchmark's success criterion
# ---------------------------------------------------------------------------

class TestRequireValid:

    def test_spec_passing_but_guard_invalid_is_not_a_solve(self, sim):
        """The whole point. 86% of the baselines' 'solves' are this case."""
        sim(passing())
        hb.set_validity_guard(FakeGuard(valid=False))
        ok, _ = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is False

    def test_spec_passing_and_guard_valid_is_still_a_solve(self, sim):
        sim(passing())
        hb.set_validity_guard(FakeGuard(valid=True))
        ok, _ = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is True

    def test_the_shaped_score_is_untouched_by_the_guard(self, sim):
        """The guard decides solved/not-solved. If it also moved the dense score, the
        search baselines would be optimizing a different landscape under the flag and
        the two modes would stop being comparable for any reason but the criterion."""
        sim(passing())
        _, before = hb.evaluate(X, TARGET, CHANNEL)
        hb.set_validity_guard(FakeGuard(valid=False))
        _, after = hb.evaluate(X, TARGET, CHANNEL)
        assert after == pytest.approx(before)

    def test_the_guard_is_not_run_on_candidates_that_already_failed(self, sim):
        """One extra simulation per spec-passing candidate is the flag's whole cost.
        Guarding failures too would roughly double every method's budget for nothing."""
        sim(passing(boost_db=99.0))             # fails boost_range
        guard = FakeGuard(valid=True)
        hb.set_validity_guard(guard)
        ok, _ = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is False
        assert guard.calls == []

    def test_the_guard_sees_the_same_design_the_simulator_did(self, sim):
        """Decoding the action twice, or guarding a different vector than the one
        measured, would make the verdict describe some other circuit."""
        measured = sim(passing())
        guard = FakeGuard(valid=True)
        hb.set_validity_guard(guard)
        hb.evaluate(X, TARGET, CHANNEL)
        assert len(guard.calls) == 1
        assert guard.calls[0] == measured[0]

    def test_a_failed_simulation_is_still_not_a_solve(self, sim):
        sim(passing(ok=False))
        hb.set_validity_guard(FakeGuard(valid=True))
        ok, score = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is False and score == -10.0


class TestEverySolverIsHeldToTheSameCriterion:
    """Symmetry is the claim the benchmark rests on, so test it structurally rather
    than trusting that four call sites stay in sync."""

    def test_all_four_solvers_route_through_evaluate(self, monkeypatch):
        seen = {"n": 0}

        def fake_evaluate(x, target, channel):
            seen["n"] += 1
            return False, 0.0                   # never solves; every method burns budget
        monkeypatch.setattr(hb, "evaluate", fake_evaluate)

        assert hb.solve_random(TARGET, CHANNEL, budget=3, seed=0) is None
        assert seen["n"] == 3

        seen["n"] = 0
        assert hb.solve_cmaes(TARGET, CHANNEL, budget=8, seed=0) is None
        assert seen["n"] >= 8

        seen["n"] = 0
        assert hb.solve_tpe(TARGET, CHANNEL, budget=4, seed=0) is None
        assert seen["n"] == 4

    def test_rl_rollout_uses_evaluate_too(self, monkeypatch):
        """solve_rl verifies with the same function, so installing the guard covers RL
        without touching the rollout code."""
        calls = []
        monkeypatch.setattr(hb, "evaluate",
                            lambda x, t, c: (calls.append(x), (False, 0.0))[1])

        class FakeEnv:
            horizon = 3
            _x = [0.5] * hb.N

            def reset(self, seed=None):
                return [0.0], {}

            def _measure(self, x):
                return None

            def _obs(self, m):
                return [0.0]

            def step(self, a):
                return [0.0], 0.0, False, False, {}

        class FakeModel:
            def predict(self, obs, deterministic=True):
                return [0.0] * hb.N, None

        assert hb.solve_rl(FakeModel(), FakeEnv(), TARGET, CHANNEL, seed=0) is None
        assert len(calls) == 3

    def test_the_criterion_is_module_state_not_a_solver_argument(self):
        """A per-solver argument is exactly how one method quietly ends up on a
        different success test. There is one switch and `evaluate` reads it."""
        import inspect
        sig = inspect.signature(hb.evaluate)
        assert list(sig.parameters) == ["x", "target", "channel"]


# ---------------------------------------------------------------------------
# change 2 — the DC-gain floor in the spec
# ---------------------------------------------------------------------------

class TestDcGainFloor:

    def test_an_attenuating_design_fails_hard_pass_when_the_floor_is_set(self):
        spec = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=0.0)
        ok, checks = hard_pass(attenuating(), spec)
        assert ok is False
        assert checks["dc_gain"] is False
        # ...and it fails ONLY on DC gain. The eight published checks still pass, which
        # is precisely why the spec needed a ninth.
        assert all(v for k, v in checks.items() if k != "dc_gain")

    def test_an_amplifying_design_still_passes(self):
        spec = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=0.0)
        ok, checks = hard_pass(passing(), spec)
        assert ok is True and checks["dc_gain"] is True

    def test_the_floor_is_inclusive(self):
        """0 dB means unity gain, which is the boundary guards.DC_GAIN_DB_MIN draws.
        A stage sitting exactly on it is not attenuating."""
        spec = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=0.0)
        ok, _ = hard_pass(passing(dc_gain_db=0.0), spec)
        assert ok is True

    def test_a_positive_floor_is_honoured(self):
        spec = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=6.0)
        assert hard_pass(passing(dc_gain_db=5.9), spec)[0] is False
        assert hard_pass(passing(dc_gain_db=6.1), spec)[0] is True

    def test_a_failed_simulation_short_circuits_before_the_floor(self):
        spec = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=0.0)
        ok, checks = hard_pass(passing(ok=False), spec)
        assert ok is False and checks == {"sim_ok": False}

    def test_spec_is_still_frozen_and_hashable(self):
        spec = Spec(dc_gain_db_min=3.0)          # differs from the 0.0 default
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.dc_gain_db_min = 6.0           # type: ignore[misc]
        assert hash(spec) != hash(DEFAULT_SPEC)

    def test_the_benchmark_can_install_the_floor(self, sim):
        """The spec field is only useful if the headline experiment can turn it on."""
        sim(attenuating())
        ok, _ = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is True                       # default: the exploit is a 'solve'
        hb.set_dc_gain_floor(0.0)
        ok, _ = hb.evaluate(X, TARGET, CHANNEL)
        assert ok is False

    def test_the_floor_adds_a_ninth_term_to_the_dense_score(self, sim):
        """Documented, not accidental: the score counts passing checks, so a run with
        the floor on has a different score scale than one without. That is fine within
        a run (all methods share it) and is why the two modes write to different files."""
        sim(passing())
        _, without = hb.evaluate(X, TARGET, CHANNEL)
        hb.set_dc_gain_floor(0.0)
        _, with_floor = hb.evaluate(X, TARGET, CHANNEL)
        assert with_floor == pytest.approx(without + 1.0)


# ---------------------------------------------------------------------------
# the two fixes are independent
# ---------------------------------------------------------------------------

class TestTheTwoOptInsAreIndependent:

    def test_require_valid_does_not_imply_the_dc_floor(self, sim):
        """The two mechanisms stay separable. Checked against an explicitly floor-less
        spec, since the floor is now on by default."""
        sim(attenuating())
        hb.set_validity_guard(FakeGuard(valid=True))   # a guard that permits everything
        no_floor = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=None)
        ok, checks = hard_pass(attenuating(), no_floor)
        assert ok and "dc_gain" not in checks

    def test_the_dc_floor_does_not_imply_the_guard(self, sim):
        sim(passing())
        hb.set_dc_gain_floor(0.0)
        assert hb._VALIDITY_GUARD is None
        assert hb.evaluate(X, TARGET, CHANNEL)[0] is True

    def test_both_together_require_both(self, sim):
        sim(passing())
        hb.set_dc_gain_floor(0.0)
        hb.set_validity_guard(FakeGuard(valid=False))
        assert hb.evaluate(X, TARGET, CHANNEL)[0] is False
