"""The improvement baseline must never be set from an untrustworthy measurement.

Regression test for the reward loop that made extended training counterproductive: a
rejected candidate scored -5.0 and was allowed to become the baseline, so the next valid
step collected the whole gap as improvement and the excursion could be repeated forever.
"""
import numpy as np
import pytest

from eqrl.sim.measures import Measures

gym = pytest.importorskip("gymnasium")
from eqrl.envs.sequential_env import SequentialEqualizerEnv, _shaped


def _env(monkeypatch, sequence):
    """Env whose measurements are scripted, so no simulator is involved."""
    env = SequentialEqualizerEnv(fast=True, seed=0)
    it = iter(sequence)
    monkeypatch.setattr(env, "_measure", lambda x: next(it))
    return env


def test_invalid_measurement_does_not_move_the_baseline(monkeypatch):
    good = Measures(ok=True, boost_db=8.0, peak_freq_ghz=1.8, hd3_db=-42.0,
                    noise_vrms=1e-4, power_w=1e-4, area_mm2=1e-4,
                    eye_h_ui=0.5, eye_v_mv=250.0, dc_gain_db=4.0)
    env = _env(monkeypatch, [good, Measures(ok=False), good])
    env.reset(seed=0)
    baseline = env._score
    env.step(np.zeros(env.action_space.shape, dtype=np.float32))    # the rejection
    assert env._score == baseline, "a failed measurement moved the improvement baseline"


def test_returning_from_invalid_pays_no_more_than_staying_valid(monkeypatch):
    """The excursion must not be worth more than never leaving."""
    good = Measures(ok=True, boost_db=8.0, peak_freq_ghz=1.8, hd3_db=-42.0,
                    noise_vrms=1e-4, power_w=1e-4, area_mm2=1e-4,
                    eye_h_ui=0.5, eye_v_mv=250.0, dc_gain_db=4.0)
    a = np.zeros(6, dtype=np.float32)

    straight = _env(monkeypatch, [good, good])
    straight.reset(seed=0)
    _, r_valid, *_ = straight.step(a)

    excursion = _env(monkeypatch, [good, Measures(ok=False), good])
    excursion.reset(seed=0)
    excursion.step(a)                                   # go invalid
    _, r_return, *_ = excursion.step(a)                 # come back

    assert r_return <= r_valid + 1e-6, (
        f"returning from an invalid state paid {r_return:+.3f} versus {r_valid:+.3f} for "
        f"an ordinary valid step; the baseline is still being poisoned")


def test_invalid_reset_does_not_give_first_valid_step_a_recovery_bonus(monkeypatch):
    """An episode that starts invalid must not manufacture improvement on recovery."""
    good = Measures(ok=True, boost_db=8.0, peak_freq_ghz=1.8, hd3_db=-42.0,
                    noise_vrms=1e-4, power_w=1e-4, area_mm2=1e-4,
                    eye_h_ui=0.5, eye_v_mv=250.0, dc_gain_db=4.0)

    invalid_first = _env(monkeypatch, [Measures(ok=False), good])
    invalid_first.reset(seed=0)
    _, recovered, *_ = invalid_first.step(np.zeros(invalid_first.action_space.shape,
                                                    dtype=np.float32))

    normal = _env(monkeypatch, [good, good])
    normal.reset(seed=0)
    _, ordinary, *_ = normal.step(np.zeros(normal.action_space.shape, dtype=np.float32))

    assert recovered <= ordinary + 1e-6, (
        f"first valid step after invalid reset paid {recovered:+.3f} versus "
        f"{ordinary:+.3f} for an ordinary step")
