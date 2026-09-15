"""The guarded env path: a rejected candidate must never reach the reward as a metric.

`guarded=True` is OFF by default and these tests document why: with the current
fast-mode stubs, Tier-5 check 20 halts on the first evaluation. Turning it on is a
one-line change once the stubbed metrics are resolved.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from silq.circuits.ctle import ACTION_SPACE, DesignVars
from silq.guards import Check, DeviceOP, Invalid, OperatingPoint, Valid
from silq.sim.measures import Measures
from silq.specs import DEFAULT_SPEC

N = len(ACTION_SPACE)


def marginal() -> Measures:
    """Passes Tier 4 and stays clear of check 20."""
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


def valid_verdict(tmp_path, m=None):
    return Valid(metrics=m or marginal(), run_id="r1", artifact_dir=tmp_path)


def invalid_verdict(tmp_path, check=Check.T2_NOT_SATURATED):
    return Invalid(check=check, reason="XM2 in triode", run_id="r2",
                   artifact_dir=tmp_path)


@pytest.fixture
def env_factory(monkeypatch, tmp_path):
    from silq.envs import sequential_env as se

    def make(verdicts, **kw):
        env = se.SequentialEqualizerEnv(seed=0, **kw)
        env._guard = FakeGuard(verdicts)
        return env
    return make


class TestDefaultIsUnguarded:

    def test_guarding_is_off_by_default(self):
        from silq.envs.sequential_env import SequentialEqualizerEnv
        env = SequentialEqualizerEnv(seed=0)
        assert env.guarded is False and env._guard is None, (
            "guarded must default OFF: with the current fast-mode stubs, check 20 "
            "halts on the first evaluation"
        )

    def test_invalid_reward_matches_the_non_convergent_penalty(self):
        """A rejected design and a design that would not simulate are the same thing:
        no trustworthy measurement. They must not be scored differently."""
        from silq.envs.sequential_env import INVALID_REWARD, _shaped
        bad, _ = _shaped(Measures(ok=False), DEFAULT_SPEC)
        assert INVALID_REWARD == bad


class TestGuardedPath:

    def test_valid_verdict_supplies_the_metrics(self, env_factory, tmp_path):
        env = env_factory([valid_verdict(tmp_path)])
        obs, _ = env.reset(seed=1)
        assert env._guard.calls == 1
        assert env.n_invalid == 0

    def test_invalid_verdict_never_yields_a_metric(self, env_factory, tmp_path):
        env = env_factory([invalid_verdict(tmp_path)])
        env.reset(seed=1)
        obs, reward, term, trunc, info = env.step(np.zeros(N, dtype=np.float32))
        assert info["sim_ok"] is False
        assert info["measures"]["boost_db"] == 0.0, (
            "an INVALID must not leak a measured value into the observation"
        )
        assert info["invalid_check"] == Check.T2_NOT_SATURATED.value
        assert "XM2" in info["invalid_reason"]
        assert info["artifact_dir"], "an INVALID must point at its raw output"

    def test_invalid_reward_is_absolute_not_a_delta(self, env_factory, tmp_path):
        env = env_factory([invalid_verdict(tmp_path)], invalid_reward=-7.5)
        env.reset(seed=1)
        _, reward, _, _, _ = env.step(np.zeros(N, dtype=np.float32))
        assert reward == pytest.approx(-7.5), (
            "an INVALID carries no measurement, so 'improvement since last score' is "
            "not meaningful — the penalty must be absolute"
        )

    def test_invalid_candidates_are_counted(self, env_factory, tmp_path):
        env = env_factory([invalid_verdict(tmp_path)])
        env.reset(seed=1)
        for _ in range(3):
            env.step(np.zeros(N, dtype=np.float32))
        assert env.n_invalid == 4          # 1 from reset + 3 steps

    def test_recovery_after_an_invalid(self, env_factory, tmp_path):
        """A later valid candidate must be scored normally, not poisoned by the
        previous rejection."""
        env = env_factory([valid_verdict(tmp_path), invalid_verdict(tmp_path),
                           valid_verdict(tmp_path)])
        env.reset(seed=1)
        _, bad, _, _, info_bad = env.step(np.zeros(N, dtype=np.float32))
        _, good, _, _, info_good = env.step(np.zeros(N, dtype=np.float32))
        assert "invalid_check" in info_bad
        assert "invalid_check" not in info_good
        assert good != bad

    def test_every_evaluation_goes_through_the_guard(self, env_factory, tmp_path):
        env = env_factory([valid_verdict(tmp_path)])
        env.reset(seed=1)
        for _ in range(5):
            env.step(np.zeros(N, dtype=np.float32))
        assert env._guard.calls == 6, "reset + 5 steps must all be validated"
        assert env.n_sims == 6
