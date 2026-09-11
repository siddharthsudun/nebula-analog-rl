import numpy as np
import pytest

from eqrl.envs.noise_env import NoiseEqualizerEnv
from eqrl.sim.measures import Measures


@pytest.fixture
def env(monkeypatch):
    # The unit tests never initialize a simulator. Integration is separate.
    monkeypatch.setattr('eqrl.evaluator.build_evaluator', lambda *a, **kw: object())
    obj = NoiseEqualizerEnv(seed=31)
    def measurement(x):
        obj._noise_score = 1.0
        obj._noise_passed = False
        return Measures(ok=True, eye_v_mv=120, eye_h_ui=0.5)
    monkeypatch.setattr(obj, '_measure', measurement)
    return obj


def test_modes_change_observation_and_preserve_supplied_range(env):
    specific, _ = env.reset(seed=123, options={'noise_request': {'mode': 'specific', 'value_vrms': .005}})
    ranged, info = env.reset(seed=123, options={'noise_request': {'mode': 'range', 'low_vrms': .005, 'high_vrms': .02}})
    assert specific.shape == env.observation_space.shape
    assert not np.array_equal(specific, ranged)
    assert env.noise_request.endpoints() == (.005, .02)
    assert info['noise']['mode'] == 'estimated'
    assert len(env.noise_request.points()) == 5
    assert np.isfinite(ranged).all()


def test_unknown_retains_assumed_nonzero_interval(env):
    env.reset(seed=123, options={'noise_request': {'mode': 'unknown'}})
    assert env.noise_request.endpoints()[0] > 0
    assert env.noise_request.endpoints()[1] > env.noise_request.endpoints()[0]


def test_failure_does_not_pay_a_recovery_bonus(env, monkeypatch):
    env.reset(seed=123)
    env._score = 3.0
    def failure(x):
        env._noise_score = None
        env._noise_passed = False
        return Measures(ok=False)
    monkeypatch.setattr(env, '_measure', failure)
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    assert reward == env.invalid_reward
    assert not terminated and not info['sim_ok']
    assert env._score == 3.0
    def recovery(x):
        env._noise_score = 3.0
        env._noise_passed = False
        return Measures(ok=True)
    monkeypatch.setattr(env, '_measure', recovery)
    assert env.step(np.zeros(env.action_space.shape))[1] == pytest.approx(-.05)


def test_endpoint_failure_prevents_success_bonus(env, monkeypatch):
    env.reset(seed=123)
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    assert not terminated and not info['passed']
    assert reward == pytest.approx(-.05)


def test_invalid_action_is_rejected_before_measurement(env):
    env.reset(seed=123)
    with pytest.raises(ValueError):
        env.step(np.full(env.action_space.shape, np.nan))
