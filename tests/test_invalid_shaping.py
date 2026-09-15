"""The opt-in shaped INVALID penalty: a near-miss must not score like an impossibility.

WHY THIS EXISTS
---------------
`INVALID_REWARD` is a flat constant, so a device missing saturation by 10 mV scores
exactly what a design demanding 50 V across a 1.8 V supply scores. Over a 40,000-step
PPO run that rejected 99.90% of 43,009 candidates, the reward was therefore very nearly
constant everywhere the policy looked; the value function flattened and the entropy bonus
walked the Gaussian mean out to the edges of the action box (43.5% of visited coordinates
within 2% of a range edge, results/policy_diagnosis.json). The invalid rate ended worse
than uniform random sampling of the same space (99.90% vs 99.6%).

WHAT THESE TESTS PIN DOWN
-------------------------
1. With shaping OFF — the default — the reward is exactly the old constant, and nothing
   about the info dict changes. This is the property that makes the option safe to add.
2. A nearly-valid design scores strictly better than a grossly-invalid one.
3. No rejection can outscore a measured design, because the shaped band is bounded ABOVE
   by the flat penalty it replaces. Shaping only ever moves a rejection downward.
4. A Tier 1 failure — the solver never returned, so no distance to any bound exists —
   gets the floor, not a guess.

The guard-side half of the contract is tested too: `Invalid.violation` has to carry the
right magnitude, or the env is shaping on noise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from silq.circuits.ctle import ACTION_SPACE, DesignVars
from silq.envs.sequential_env import (
    INVALID_REWARD, INVALID_SHAPING_SCALE, INVALID_SHAPING_SPAN, SequentialEqualizerEnv,
    shaped_invalid_reward,
)
from silq.guards import (
    ArtifactStore, Check, DeviceOP, Invalid, OperatingPoint, RunArtifacts, Valid,
    check_circuit_sanity, check_run_integrity,
)
from silq.sim.measures import Measures

N = len(ACTION_SPACE)
FLOOR = INVALID_REWARD - INVALID_SHAPING_SPAN          # -10.0 at the defaults


# ---------------------------------------------------------------------------
# fixtures — the guarded env, with the simulator replaced by a scripted verdict
# ---------------------------------------------------------------------------

def marginal() -> Measures:
    """Passes Tier 4 and stays clear of Tier-5 check 20. Same design as
    tests/test_env_guarding.marginal, so the two files agree on what 'valid' means."""
    return Measures(dc_gain_db=5.0, peak_gain_db=14.0, boost_db=9.0, peak_freq_ghz=2.0,
                    hd3_db=-35.0, noise_vrms=1.2e-3, power_w=11e-3, area_mm2=0.040,
                    eye_h_ui=0.45, eye_v_mv=120.0, ok=True)


class FakeGuard:
    """Stands in for GuardedEvaluator: returns whatever verdict the test wants."""

    def __init__(self, verdicts):
        self._v = list(verdicts)
        self.calls = 0

    def evaluate(self, dv, **kw):
        self.calls += 1
        return self._v[min(self.calls - 1, len(self._v) - 1)]


def invalid_verdict(tmp_path, violation, check=Check.T2_NOT_SATURATED):
    return Invalid(check=check, reason="XM2 in triode", run_id="r2",
                   artifact_dir=tmp_path, violation=violation)


def valid_verdict(tmp_path):
    return Valid(metrics=marginal(), run_id="r1", artifact_dir=tmp_path)


@pytest.fixture
def env_factory(tmp_path):
    def make(verdicts, **kw):
        env = SequentialEqualizerEnv(seed=0, **kw)
        env._guard = FakeGuard(verdicts)
        return env
    return make


def step_reward(env) -> float:
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.zeros(N, dtype=np.float32))
    return reward, info


@pytest.fixture
def art(tmp_path) -> RunArtifacts:
    a = ArtifactStore(tmp_path / "raw").new_run()
    a.exit_code = 0
    return a


def sane_dv(**over) -> DesignVars:
    """Legal on every axis check 8 looks at (default l_in sits exactly on the SKY130
    minimum, which check 8 correctly rejects)."""
    base = dict(l_in=0.30e-6)
    base.update(over)
    return DesignVars(**base)


def good_op(**over) -> OperatingPoint:
    devices = over.pop("devices", (
        DeviceOP(name="XM1", vds=0.60, vdsat=0.20),
        DeviceOP(name="XM2", vds=0.60, vdsat=0.20),
    ))
    return OperatingPoint(
        devices=devices,
        node_voltages=over.pop("node_voltages", {"outp": 1.2, "outn": 1.2, "sp": 0.45}),
        tail_currents=over.pop("tail_currents", {"Itp": 1e-3, "Itn": 1e-3}),
    )


# ===========================================================================
# (a) OFF BY DEFAULT, AND OFF MEANS UNCHANGED
# ===========================================================================

class TestShapingIsOptIn:

    def test_off_by_default(self):
        env = SequentialEqualizerEnv(seed=0)
        assert env.invalid_shaping is False, (
            "shaping must default OFF; every published number was produced with the flat "
            "penalty and turning this on silently would change what they measure"
        )

    @pytest.mark.parametrize("violation", [None, 0.0, 0.5, 1.2, 9.0, 1e6])
    def test_reward_is_exactly_the_old_constant_when_off(self, env_factory, tmp_path,
                                                         violation):
        env = env_factory([invalid_verdict(tmp_path, violation)])
        reward, info = step_reward(env)
        assert reward == INVALID_REWARD, (
            f"shaping off must give the flat constant regardless of violation "
            f"({violation!r}); got {reward}"
        )
        assert "invalid_violation" not in info, (
            "shaping off must not add keys to info either — 'unchanged' includes the "
            "info dict a callback might be counting"
        )

    def test_a_custom_flat_penalty_is_still_honoured_when_off(self, env_factory, tmp_path):
        env = env_factory([invalid_verdict(tmp_path, 9.0)], invalid_reward=-7.5)
        reward, _ = step_reward(env)
        assert reward == pytest.approx(-7.5)

    def test_on_reports_the_violation_it_shaped_on(self, env_factory, tmp_path):
        env = env_factory([invalid_verdict(tmp_path, 1.2)], invalid_shaping=True)
        reward, info = step_reward(env)
        assert info["invalid_violation"] == pytest.approx(1.2)
        assert info["invalid_check"] == Check.T2_NOT_SATURATED.value
        assert reward != INVALID_REWARD


# ===========================================================================
# (b) NEARLY VALID BEATS GROSSLY INVALID
# ===========================================================================

class TestOrdering:

    def test_near_miss_scores_strictly_better_than_an_impossibility(self):
        """10 mV the wrong side of the 50 mV headroom bound (violation 1.2) against a
        node 27 rail-widths outside the supply — the 20 mA x 5 kohm corner of the box."""
        near = shaped_invalid_reward(invalid_stub(1.2))
        gross = shaped_invalid_reward(invalid_stub(26.8))
        assert near > gross, f"near-miss {near} must beat impossibility {gross}"
        assert near == pytest.approx(-5.85, abs=0.02)
        assert gross == pytest.approx(-8.60, abs=0.02)

    def test_the_penalty_is_monotone_in_the_violation(self):
        vs = [0.0, 0.01, 0.1, 1.0, 1.2, 9.0, 26.8, 99.0, 100.0]
        rewards = [shaped_invalid_reward(invalid_stub(v)) for v in vs]
        assert all(a > b for a, b in zip(rewards, rewards[1:])), (
            f"a larger violation must always score worse: {list(zip(vs, rewards))}"
        )

    def test_ordering_survives_the_full_env_path(self, env_factory, tmp_path):
        near = env_factory([invalid_verdict(tmp_path, 1.2)], invalid_shaping=True)
        gross = env_factory([invalid_verdict(tmp_path, 26.8,
                                             check=Check.T2_NODE_OUT_OF_RAILS)],
                            invalid_shaping=True)
        assert step_reward(near)[0] > step_reward(gross)[0]

    def test_sitting_exactly_on_a_bound_scores_the_old_constant(self):
        """Violation 0 is the closest an invalid design can be to buildable, so it gets
        the ceiling — which is exactly the penalty it would have received unshaped."""
        assert shaped_invalid_reward(invalid_stub(0.0)) == pytest.approx(INVALID_REWARD)


# ===========================================================================
# (c) NO REJECTION EVER OUTSCORES A MEASUREMENT
# ===========================================================================

class TestBoundedness:

    @pytest.mark.parametrize("violation", [
        None, 0.0, -0.0, 1e-12, 0.5, 1.2, 9.0, 26.8, 100.0, 1e3, 1e12,
        float("inf"), float("nan"), -3.0,
    ])
    def test_shaped_penalty_stays_inside_the_declared_band(self, violation):
        r = shaped_invalid_reward(invalid_stub(violation))
        assert FLOOR <= r <= INVALID_REWARD, (
            f"violation {violation!r} produced {r}, outside [{FLOOR}, {INVALID_REWARD}]"
        )

    def test_the_ceiling_is_the_flat_penalty_it_replaces(self):
        """The load-bearing safety property. Because the shaped band hangs DOWNWARD from
        the flat constant, every ordering between a valid step and an invalid one that
        held under the flat penalty still holds — shaping can only make a rejection look
        worse, never better, so it cannot invent a preference for being invalid."""
        vs = [None, 0.0, 1e-9, 1.0, 50.0, 1e9, float("nan")]
        assert all(shaped_invalid_reward(invalid_stub(v)) <= INVALID_REWARD for v in vs)

    def test_a_custom_flat_penalty_moves_the_whole_band(self):
        for base in (-1.0, -5.0, -20.0):
            rs = [shaped_invalid_reward(invalid_stub(v), base) for v in (0.0, 1.0, None)]
            assert max(rs) == pytest.approx(base)
            assert min(rs) == pytest.approx(base - INVALID_SHAPING_SPAN)

    def test_a_valid_step_outscores_every_possible_rejection(self, env_factory, tmp_path):
        """A valid step that merely holds its score already earns -0.05 (the per-step
        time cost); one that passes earns +10. Both are far above the shaping ceiling."""
        env = env_factory([valid_verdict(tmp_path)], invalid_shaping=True)
        good, info = step_reward(env)
        assert "invalid_check" not in info
        assert good > INVALID_REWARD >= max(
            shaped_invalid_reward(invalid_stub(v))
            for v in (None, 0.0, 1.2, 26.8, 1e9)
        )


# ===========================================================================
# (d) TIER 1 — NO DISTANCE EXISTS, SO IT GETS THE FLOOR
# ===========================================================================

class TestTier1GetsTheWorstPenalty:

    def test_a_missing_violation_scores_the_floor(self):
        assert shaped_invalid_reward(invalid_stub(None)) == pytest.approx(FLOOR)

    def test_nothing_scores_below_a_tier_1_failure(self):
        worst = shaped_invalid_reward(invalid_stub(None))
        assert all(shaped_invalid_reward(invalid_stub(v)) >= worst
                   for v in (0.0, 1.0, 26.8, 1e9, float("inf")))

    @pytest.mark.parametrize("break_it, expected", [
        (lambda a: setattr(a, "exit_code", 1), Check.T1_EXIT_CODE),
        (lambda a: setattr(a, "stderr", "doAnalyses: TRAN:  Timestep too small"),
         Check.T1_STDERR_FAILURE),
        (lambda a: setattr(a, "exit_code", None), Check.T1_EXIT_CODE),
    ])
    def test_the_guard_reports_no_distance_for_a_tier_1_failure(self, art, break_it,
                                                                expected):
        """A solver that never returned says nothing about how close the design was.
        Reporting 0.0 here would pay the agent for crashing the simulator."""
        break_it(art)
        result = check_run_integrity(art, expect_output=False)
        assert result is not None and result.check is expected
        assert result.violation is None
        assert shaped_invalid_reward(result) == pytest.approx(FLOOR)

    def test_a_tier_1_failure_beats_nothing_in_the_env(self, env_factory, tmp_path):
        t1 = env_factory([invalid_verdict(tmp_path, None, check=Check.T1_EXIT_CODE)],
                         invalid_shaping=True)
        t2 = env_factory([invalid_verdict(tmp_path, 9.0)], invalid_shaping=True)
        assert step_reward(t1)[0] == pytest.approx(FLOOR)
        assert step_reward(t2)[0] > step_reward(t1)[0]


# ===========================================================================
# THE GUARD SIDE — the magnitudes the env shapes on have to be the right ones
# ===========================================================================

class TestGuardReportsTheMagnitude:

    def test_saturation_distance_is_measured_from_the_headroom_bound(self, art):
        """Bound is 50 mV of headroom. A device AT -10 mV is 60 mV short of it, i.e.
        1.2 bound-widths outside."""
        op = good_op(devices=(DeviceOP(name="XM1", vds=0.60, vdsat=0.20),
                              DeviceOP(name="XM2", vds=0.19, vdsat=0.20)))
        result = check_circuit_sanity(op, sane_dv(), art, vdd=1.8)
        assert result.check is Check.T2_NOT_SATURATED
        assert result.violation == pytest.approx(1.2)

    def test_the_worst_device_sets_the_distance(self, art):
        """Fixing the near-miss still leaves the design invalid, so the binding device
        is the one furthest out."""
        op = good_op(devices=(DeviceOP(name="XM1", vds=0.19, vdsat=0.20),   # -10 mV
                              DeviceOP(name="XM2", vds=-0.20, vdsat=0.20)))  # -400 mV
        result = check_circuit_sanity(op, sane_dv(), art, vdd=1.8)
        assert result.violation == pytest.approx((0.050 + 0.400) / 0.050)

    def test_a_worse_device_reports_a_larger_distance(self, art):
        def v(vds):
            op = good_op(devices=(DeviceOP(name="XM1", vds=vds, vdsat=0.20),))
            return check_circuit_sanity(op, sane_dv(), art, vdd=1.8).violation
        assert v(0.24) < v(0.19) < v(0.0) < v(-0.4)

    def test_a_non_finite_operating_point_reports_no_distance(self, art):
        """NaN Vds fails `saturated` (nan >= 0.05 is False) and lands here. Its distance
        is unknown, not small — it must not be shaped as a near-miss."""
        op = good_op(devices=(DeviceOP(name="XM1", vds=float("nan"), vdsat=0.20),))
        result = check_circuit_sanity(op, sane_dv(), art, vdd=1.8)
        assert result.check is Check.T2_NOT_SATURATED
        assert result.violation is None

    def test_tail_current_distance_is_measured_from_the_tolerance(self, art):
        """Bound is 10%. Delivering 1.5 mA against a 1 mA request is 50% off, i.e.
        (0.50 - 0.10) / 0.10 = 4 tolerance-widths outside."""
        op = good_op(tail_currents={"Itp": 0.75e-3, "Itn": 0.75e-3})
        result = check_circuit_sanity(op, sane_dv(i_tail=1e-3), art, vdd=1.8)
        assert result.check is Check.T2_TAIL_CURRENT
        assert result.violation == pytest.approx(4.0)

    def test_a_dead_mirror_is_a_full_hundred_percent_deviation(self, art):
        op = good_op(tail_currents={"Itp": 0.0, "Itn": 0.0})
        result = check_circuit_sanity(op, sane_dv(i_tail=1e-3), art, vdd=1.8)
        assert result.check is Check.T2_TAIL_CURRENT
        assert result.violation == pytest.approx(9.0)      # (1.0 - 0.1) / 0.1

    def test_rail_distance_is_measured_in_rail_widths(self, art):
        """The 20 mA x 5 kohm corner of the action space asks for 100 V across the load.
        A node at -50 V on a 1.8 V supply is 27.8 supplies below ground."""
        op = good_op(node_voltages={"outp": 1.2, "outn": -50.0})
        result = check_circuit_sanity(op, sane_dv(), art, vdd=1.8)
        assert result.check is Check.T2_NODE_OUT_OF_RAILS
        assert result.violation == pytest.approx(50.0 / 1.8)

    def test_a_node_just_outside_the_rail_is_a_small_distance(self, art):
        op = good_op(node_voltages={"outp": 1.85, "outn": 1.2})
        result = check_circuit_sanity(op, sane_dv(), art, vdd=1.8)
        assert result.violation == pytest.approx(0.05 / 1.8)

    def test_a_missing_probe_reports_no_distance_rather_than_zero(self, art):
        """No devices probed, no branches probed, no nodes probed: in each case the
        guard has no measurement to take a distance from."""
        for op in (good_op(devices=()), good_op(tail_currents={}),
                   good_op(node_voltages={})):
            result = check_circuit_sanity(op, sane_dv(), art, vdd=1.8)
            assert result is not None and result.violation is None

    def test_sitting_exactly_on_a_pdk_bound_is_distance_zero(self, art):
        """On the bound, not past it — the smallest violation there is."""
        result = check_circuit_sanity(good_op(), DesignVars(l_in=0.15e-6), art, vdd=1.8)
        assert result.check is Check.T2_AT_PDK_BOUND
        assert result.violation == pytest.approx(0.0)

    def test_outside_a_pdk_bound_is_a_relative_distance(self, art):
        result = check_circuit_sanity(good_op(), sane_dv(w_in=200e-6), art, vdd=1.8)
        assert result.check is Check.T2_AT_PDK_BOUND
        assert result.violation == pytest.approx(1.0)      # 200 um past a 100 um max

    def test_violation_defaults_to_none_so_nothing_has_to_supply_it(self):
        """`Invalid` is constructed in several places; none of them are required to know
        about shaping, and the safe default for 'no information' is the worst penalty."""
        stub = Invalid(check=Check.T3_CORNER_IDENTICAL, reason="x", run_id="r",
                       artifact_dir="/tmp")
        assert stub.violation is None
        assert shaped_invalid_reward(stub) == pytest.approx(FLOOR)


# ===========================================================================
# the squash itself
# ===========================================================================

class TestSquash:

    def test_saturates_at_the_declared_scale(self):
        assert shaped_invalid_reward(
            invalid_stub(INVALID_SHAPING_SCALE)) == pytest.approx(FLOOR)

    def test_is_logarithmic_so_the_decades_share_the_resolution(self):
        """A linear squash would spend all of its range above violation 10 and none of
        it near the bound: violation 0.1 and violation 1.2 would be indistinguishable at
        the resolution PPO can see. Each decade must get a real slice of the span."""
        def sev(v):
            return (INVALID_REWARD - shaped_invalid_reward(invalid_stub(v))) \
                / INVALID_SHAPING_SPAN
        decades = [sev(10 ** k) - sev(10 ** (k - 1)) for k in (0, 1, 2)]
        assert all(d > 0.10 for d in decades), decades
        assert sev(1.0) == pytest.approx(math.log(2) / math.log1p(100), abs=1e-9)


def invalid_stub(violation) -> Invalid:
    return Invalid(check=Check.T2_NOT_SATURATED, reason="stub", run_id="r",
                   artifact_dir="/tmp", violation=violation)
