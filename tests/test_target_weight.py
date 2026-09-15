"""Pricing the target must be separable from gating termination on it.

`Spec.boost_target_tol_db` (tested in test_target_tracking.py) does both at once: it adds a
`boost_target` margin, and because `passed = all(v >= 0)` over the margins while
`terminated = passed`, the episode now also ends only when the target is hit. The run built
on it came back negative, and that result is uninterpretable -- it cannot separate "paying
for the target does not teach retargeting" from "we sparsified the terminal reward".

`target_weight` is the same price with none of the gating. These tests pin the three things
that make it a clean instrument:

1. OFF BY DEFAULT and bit-identical when off, so every artifact that predates it stands.
2. It never touches `passed`, at any weight, for any error.
3. It is dense -- it carries gradient across the whole 5-11 dB target range, unlike the
   +0.5 soft term it replaces, which saturates at 3 dB and is flat beyond.

None of these needs a simulator; they are properties of the scoring function.
"""
import dataclasses

import numpy as np
import pytest

from eqrl.envs.sequential_env import (TARGET_PRICE_REF_DB, SequentialEqualizerEnv,
                                      _shaped)
from eqrl.sim.measures import Measures
from eqrl.specs import DEFAULT_SPEC


def passing(**over) -> Measures:
    """A design clearing every hard spec at target_boost_db=9, channel 12."""
    base = dict(dc_gain_db=6.0, peak_gain_db=15.0, boost_db=9.0, peak_freq_ghz=2.0,
                hd3_db=-35.0, noise_vrms=1.2e-3, power_w=11e-3, area_mm2=0.040,
                eye_h_ui=0.45, eye_v_mv=120.0, ok=True)
    return Measures(**{**base, **over})


def spec_at(target: float):
    """A spec asking for `target`, with the confounded tolerance path left OFF."""
    return dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                               boost_target_tol_db=None)


#: Changing `boost_db` to vary the target error ALSO moves the boost_lo/boost_hi
#: feasibility margins, so a raw score difference between two boosts measures both at once.
#: These isolate the target's own contribution: an infinitesimal weight contributes nothing
#: while still taking the priced branch, leaving the margin sum alone.
def _margins_only(m: Measures, spec) -> float:
    return _shaped(m, spec, 1e-12)[0]


def priced_term(m: Measures, spec, w: float) -> float:
    return _shaped(m, spec, w)[0] - _margins_only(m, spec)


def soft_term(m: Measures, spec) -> float:
    return _shaped(m, spec)[0] - _margins_only(m, spec)


class TestOffByDefaultAndUnchangedWhenOff:
    def test_the_default_call_is_the_historical_formula(self):
        """No third argument must mean exactly what it meant before the flag existed."""
        m, spec = passing(boost_db=7.0), spec_at(9.0)
        got, passed = _shaped(m, spec)
        soft = 0.5 * (1.0 - min(abs(7.0 - 9.0) / 3.0, 1.0))
        from eqrl.envs.equalizer_env import _margins
        base = float(sum(np.clip(v, -2.0, 1.0) for v in _margins(m, spec).values()))
        assert got == pytest.approx(base + soft)
        assert passed is True

    def test_passing_none_is_the_same_as_passing_nothing(self):
        m, spec = passing(boost_db=6.2), spec_at(10.0)
        assert _shaped(m, spec) == _shaped(m, spec, None)

    def test_the_env_defaults_to_off(self):
        env = SequentialEqualizerEnv.__new__(SequentialEqualizerEnv)
        import inspect
        sig = inspect.signature(SequentialEqualizerEnv.__init__)
        assert sig.parameters["target_weight"].default is None
        del env


class TestItNeverTouchesPassed:
    @pytest.mark.parametrize("err", [0.0, 1.5, 3.0, 6.0])
    @pytest.mark.parametrize("w", [1.0, 3.0])
    def test_a_feasible_design_stays_passed_however_far_off_target(self, err, w):
        """This is the whole point of the flag. `passed` is feasibility, nothing else."""
        spec = spec_at(9.0)
        m = passing(boost_db=9.0 - err)
        _, passed_off = _shaped(m, spec)
        _, passed_on = _shaped(m, spec, w)
        assert passed_off is True
        assert passed_on is passed_off

    def test_an_infeasible_design_stays_failed_even_at_a_perfect_hit(self):
        """A perfect target hit must not buy its way past a hard check."""
        spec = spec_at(9.0)
        m = passing(boost_db=9.0, power_w=1.0)          # 1 W against a 15 mW ceiling
        _, passed_off = _shaped(m, spec)
        _, passed_on = _shaped(m, spec, 3.0)
        assert passed_off is False
        assert passed_on is False

    def test_the_weight_shifts_only_the_score(self):
        spec = spec_at(9.0)
        m = passing(boost_db=9.0)
        off, _ = _shaped(m, spec)
        on, _ = _shaped(m, spec, 1.0)
        soft = 0.5                                       # perfect hit -> the old term maxes
        assert on - (off - soft) == pytest.approx(1.0)   # replaced, not stacked


class TestItIsDense:
    def test_a_perfect_hit_is_worth_exactly_the_weight(self):
        spec = spec_at(8.0)
        for w in (1.0, 2.5):
            assert priced_term(passing(boost_db=8.0), spec, w) == pytest.approx(w, abs=1e-6)

    def test_the_term_falls_monotonically_as_the_error_grows(self):
        spec = spec_at(8.0)
        terms = [priced_term(passing(boost_db=8.0 - e), spec, 1.0)
                 for e in (0.0, 0.5, 1.0, 2.0, 3.0)]
        assert all(a > b for a, b in zip(terms, terms[1:]))

    def test_it_still_has_gradient_where_the_old_soft_term_went_flat(self):
        """The soft term is min(|err|/3, 1)-clipped: beyond 3 dB it is identically 0 and
        the agent is told nothing. Targets span 5-11 dB, so that is half the range."""
        spec = spec_at(9.0)
        old = [soft_term(passing(boost_db=9.0 - e), spec) for e in (3.5, 5.0)]
        assert old == pytest.approx([0.0, 0.0], abs=1e-9)         # flat, the defect

        new = [priced_term(passing(boost_db=9.0 - e), spec, 1.0) for e in (3.5, 5.0)]
        assert new[0] > new[1]                                    # still descending

    def test_the_term_spans_one_margin_check(self):
        """Same [-2, +1] clip every other spec gets, scaled by the weight -- so weight 1.0
        prices the target at exactly one check's worth, which is what the repo argued for
        and what the confounded run could not test cleanly."""
        spec, ref = spec_at(9.0), TARGET_PRICE_REF_DB
        best = priced_term(passing(boost_db=9.0), spec, 1.0)
        floor = priced_term(passing(boost_db=9.0 - 3.0 * ref), spec, 1.0)   # t = -2
        deeper = priced_term(passing(boost_db=9.0 - 9.0 * ref), spec, 1.0)
        assert best == pytest.approx(1.0, abs=1e-6)
        assert floor == pytest.approx(-2.0, abs=1e-6)
        assert best - floor == pytest.approx(3.0, abs=1e-6)
        assert floor == pytest.approx(deeper)                               # clip holds


class TestItRefusesTheConfound:
    def test_setting_both_flags_is_an_error(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SequentialEqualizerEnv(boost_tol=1.5, target_weight=1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_meaningless_weight_is_an_error(self, bad):
        with pytest.raises(ValueError, match="target_weight"):
            SequentialEqualizerEnv(target_weight=bad)
