"""The measured 32/32 baseline must remain a stable anchor for the residual policy."""
import numpy as np
import pytest

from eqrl.baselines.robust import robust_design, robust_unit_action
from eqrl.circuits.ctle import decode_action
from eqrl.sim.measures import Measures
from eqrl.envs.sequential_env import SequentialEqualizerEnv


def _good():
    return Measures(ok=True, boost_db=8.0, peak_freq_ghz=1.8, hd3_db=-42.0,
                    noise_vrms=1e-4, power_w=1e-4, area_mm2=1e-4,
                    eye_h_ui=0.5, eye_v_mv=250.0, dc_gain_db=4.0)


def test_robust_baseline_roundtrips_through_action_space():
    dv = decode_action(robust_unit_action())
    expected = robust_design()
    for name in ("w_in", "l_in", "i_tail", "rs", "cs", "r_load"):
        assert getattr(dv, name) == pytest.approx(getattr(expected, name), rel=1e-6)


def test_zero_residual_keeps_anchor(monkeypatch):
    env = SequentialEqualizerEnv(fast=True, seed=0, anchor_design=robust_design())
    monkeypatch.setattr(env, "_measure", lambda x: _good())
    env.reset(seed=0)
    expected = robust_unit_action()
    np.testing.assert_allclose(env._x, expected, rtol=0, atol=1e-7)
    env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    np.testing.assert_allclose(env._x, expected, rtol=0, atol=1e-7)


def test_anchor_noise_must_be_non_negative():
    with pytest.raises(ValueError, match="anchor_noise"):
        SequentialEqualizerEnv(anchor_design=robust_design(), anchor_noise=-1.0)
