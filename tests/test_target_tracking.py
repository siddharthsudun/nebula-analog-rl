"""The target must be SCORED and PRICED, not merely observed.

The submission claims retargeting: give the system a spec, get a design meeting THAT spec.
Two defects prevented that from ever being true, and both are closed by
`Spec.boost_target_tol_db`:

1. `hard_pass` never referenced `target_boost_db`. Boost only had to land somewhere in the
   3-12 dB range, so ONE fixed design passed all 32 held-out specs and random search solved
   30/30. The benchmark was not measuring the claim.

2. The training reward mentioned the target only through a soft term worth at most +0.5
   against a margin sum spanning ~27 points -- 1.8% of the available signal. Re-simulating
   the 26 designs the 22 Aug policy solved gave a correlation between requested and
   achieved boost of 0.114, with the policy scoring the same 4/32 within +/-0.5 dB as a
   fixed design that ignores the spec entirely (results/target_tracking_clean40k.json).

These tests pin both halves, and pin that the feature stays OPT-IN -- turning it on
silently would make every artifact predating it incomparable.
"""
import dataclasses

import pytest

from eqrl.envs.equalizer_env import _margins
from eqrl.envs.sequential_env import SequentialEqualizerEnv, _shaped
from eqrl.sim.measures import Measures
from eqrl.specs import DEFAULT_SPEC, hard_pass


def passing(**over) -> Measures:
    """A design clearing every hard spec at target_boost_db=9, channel 12. Defined here
    rather than imported from test_benchmark_fairness so the two files stay independent."""
    base = dict(dc_gain_db=6.0, peak_gain_db=15.0, boost_db=9.0, peak_freq_ghz=2.0,
                hd3_db=-35.0, noise_vrms=1.2e-3, power_w=11e-3, area_mm2=0.040,
                eye_h_ui=0.45, eye_v_mv=120.0, ok=True)
    return Measures(**{**base, **over})


TOL = 1.5


def spec_at(target: float, tol: float | None = TOL):
    return dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                               boost_target_tol_db=tol)


class TestHardPassScoresTheTarget:
    def test_off_by_default_so_older_artifacts_stay_comparable(self):
        assert DEFAULT_SPEC.boost_target_tol_db is None
        _, checks = hard_pass(passing(), DEFAULT_SPEC)
        # absent, not False -- honest_benchmark's dense score counts this dict
        assert "boost_target" not in checks

    def test_a_design_that_misses_the_target_now_fails(self):
        # 9.0 dB delivered against a 5.0 dB request: passes the 3-12 range, misses by 4
        m = passing(boost_db=9.0)
        assert hard_pass(m, dataclasses.replace(DEFAULT_SPEC, target_boost_db=5.0))[0]
        ok, checks = hard_pass(m, spec_at(5.0))
        assert ok is False
        assert checks["boost_target"] is False

    def test_a_design_that_hits_the_target_still_passes(self):
        ok, checks = hard_pass(passing(boost_db=9.0), spec_at(9.0))
        assert ok is True
        assert checks["boost_target"] is True

    @pytest.mark.parametrize("err,expected", [(0.0, True), (1.4, True), (1.5, True),
                                              (1.6, False), (4.0, False)])
    def test_the_boundary_is_inclusive_and_symmetric(self, err, expected):
        for signed in (err, -err):
            m = passing(boost_db=9.0 + signed)
            assert hard_pass(m, spec_at(9.0))[1]["boost_target"] is expected


class TestTheRewardPricesTheTarget:
    def test_margin_absent_unless_the_spec_asks(self):
        assert "boost_target" not in _margins(passing(), DEFAULT_SPEC)
        assert "boost_target" in _margins(passing(), spec_at(9.0))

    def test_margin_is_positive_inside_tolerance_and_negative_outside(self):
        assert _margins(passing(boost_db=9.0), spec_at(9.0))["boost_target"] == 1.0
        assert _margins(passing(boost_db=10.5), spec_at(9.0))["boost_target"] == 0.0
        assert _margins(passing(boost_db=12.0), spec_at(9.0))["boost_target"] < 0.0

    def test_hitting_the_target_is_worth_as_much_as_any_other_spec(self):
        """The whole fix. Before, the target was worth at most +0.5 against ~27 points of
        margin. It must now be worth the same 3-point swing as every other check, which is
        what the shared [-2, +1] clip in _shaped gives it."""
        on_target = _shaped(passing(boost_db=9.0), spec_at(9.0))[0]
        far_off = _shaped(passing(boost_db=13.5), spec_at(9.0))[0]   # margin clipped at -2
        # 1.0 -> -2.0 on the boost_target margin alone
        assert on_target - far_off >= 3.0 - 1e-9

    def test_the_target_is_priced_once_not_twice(self):
        """With the margin active the old soft term must be dropped, not added to it."""
        m = passing(boost_db=9.0)
        with_margin = _shaped(m, spec_at(9.0))[0]
        margin_sum = sum(min(max(v, -2.0), 1.0)
                         for v in _margins(m, spec_at(9.0)).values())
        assert with_margin == pytest.approx(margin_sum)

    def test_the_old_soft_term_survives_when_the_feature_is_off(self):
        """Unchanged behaviour for every config that predates this."""
        m = passing(boost_db=9.0)
        off = dataclasses.replace(DEFAULT_SPEC, target_boost_db=9.0)
        margin_sum = sum(min(max(v, -2.0), 1.0) for v in _margins(m, off).values())
        assert _shaped(m, off)[0] == pytest.approx(margin_sum + 0.5)

    def test_passed_gates_on_the_target(self):
        assert _shaped(passing(boost_db=9.0), spec_at(9.0))[1] is True
        assert _shaped(passing(boost_db=9.0), spec_at(5.0))[1] is False


class TestEnvPlumbing:
    def test_off_by_default(self):
        env = SequentialEqualizerEnv(fast=True)
        assert env.boost_tol is None
        env._target, env._channel = 7.0, 12.0
        assert env._spec().boost_target_tol_db is None

    def test_reaches_the_per_episode_spec_when_set(self):
        env = SequentialEqualizerEnv(fast=True, boost_tol=TOL)
        env._target, env._channel = 7.0, 12.0
        spec = env._spec()
        assert spec.boost_target_tol_db == TOL
        assert spec.target_boost_db == 7.0        # still randomized per episode

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_a_tolerance_that_cannot_be_met(self, bad):
        with pytest.raises(ValueError):
            SequentialEqualizerEnv(fast=True, boost_tol=bad)
