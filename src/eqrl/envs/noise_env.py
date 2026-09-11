"""Experimental noise-conditioned PPO environment; frozen environment untouched.

All nominal measurements pass the existing guards with fast=False. The new
task scores noisy v2 eyes at each supplied noise sample and gates success on all
sampled points. This is a finite-pattern stress test, not a guarantee over
the continuous interval or a BER certification. Unknown means an explicit
assumed interval, never zero noise. TT only for this bounded pilot.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from gymnasium import spaces

from eqrl.circuits.ctle import decode_action
from eqrl.envs.sequential_env import SequentialEqualizerEnv, _shaped
from eqrl.evaluator import build_evaluator
from eqrl.guards import ArtifactStore
from eqrl.snr_spec import SNRRequest, experimental_request
from eqrl.sim.snr_evaluation import evaluate_snr, signal_calibration
from eqrl.sim.measures import Measures
from eqrl.sim.ngspice_runner import NgspiceError
from eqrl.sim.server import get_server


class NoiseEqualizerEnv(SequentialEqualizerEnv):
    """Append noise bounds, signal swing and provenance to the observation."""

    def __init__(self, *, noise_request=None, eye_bits=512, artifact_root=None, **kwargs):
        if kwargs.get("fast", False) or not kwargs.get("guarded", True):
            raise ValueError("noise pilot requires guarded=True and fast=False")
        if kwargs.get("pvt", False) or kwargs.get("feasible_decode", False):
            raise ValueError("pilot supports nominal TT and the original decoder only")
        if kwargs.get("corner", "tt") != "tt":
            raise ValueError("pilot is nominal TT only")
        kwargs.update(fast=False, guarded=True)
        kwargs.setdefault("boost_tol", 1.5)
        super().__init__(**kwargs)
        if not isinstance(eye_bits, int) or not 128 <= eye_bits <= 2048:
            raise ValueError("eye_bits must be an integer in [128, 2048]")
        self.eye_bits = eye_bits
        self._pilot_store = ArtifactStore(artifact_root) if artifact_root is not None else None
        if isinstance(noise_request, dict):
            noise_request = experimental_request(noise_request)
        self.fixed_noise_request = noise_request
        self.noise_request = experimental_request(noise_request) if noise_request is not None else experimental_request({"mode": "unknown"})
        self._equivalent_tx_vpp = 1.0
        self.heartbeat_path = None
        extra = len(self.noise_request.observation())
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(self.observation_space.shape[0] + extra,), dtype=np.float32)
        self.n_noise_failures = 0
        self.n_noise_evaluations = 0
        self.noise_details = {}
        self._noise_seed = 0
        self._noise_score = None
        self._noise_passed = False

    def _sample_noise(self, rng):
        mode = str(rng.choice(["measured", "estimated", "unknown"]))
        band = ((1e7, 5e9), (1e7, 2.5e9), (1e8, 5e9), (1e8, 2.5e9))[int(rng.integers(4))]
        swing = float(rng.uniform(.5, 1.))
        reference = str(rng.choice(["tx_vpp", "ctle_input_vrms"]))
        # The channel is set by the base reset; normalize the alternate reference there.
        self._sampled_tx_swing = swing
        data = {"mode": mode, "signal_reference": reference, "signal_value_v": swing,
                "bandwidth_hz": band}
        if mode == "measured":
            data['value_vrms'] = float(10**rng.uniform(-3, np.log10(.05)))
        elif mode == "estimated":
            lo, hi = sorted(10**rng.uniform(-3, np.log10(.05), 2))
            if rng.random() < .5:
                data['budget_vrms'] = float(hi)
            else:
                data.update(low_vrms=float(lo), high_vrms=float(hi))
        return SNRRequest.from_dict(data)

    def reset(self, *, seed=None, options=None):
        # A separate episode stream avoids exposing a sampled hidden noise value.
        if seed is not None or not hasattr(self, "_noise_rng"):
            self._noise_rng = np.random.default_rng(seed if seed is not None else
                int(self._rng.integers(0, 2**31)))
        self._sampled_tx_swing = None
        request = (options or {}).get("noise_request", self.fixed_noise_request)
        if isinstance(request, dict):
            request = experimental_request(request)
        self.noise_request = experimental_request(request) if request is not None else self._sample_noise(self._noise_rng)
        self._noise_seed = int(self._noise_rng.integers(0, 2**31))
        observation, info = super().reset(seed=seed, options=options)
        self._score = self._noise_score
        info["noise"] = self.noise_request.to_dict()
        info["noise_evaluation"] = self.noise_details
        return observation, info

    def _measure(self, x):
        self._heartbeat("measuring")
        if self._sampled_tx_swing is not None:
            from eqrl.sim.snr_evaluation import channel_unit_rms
            if self.noise_request.signal_reference == "ctle_input_vrms":
                self.noise_request = replace(self.noise_request, signal_value_v=
                    self._sampled_tx_swing * channel_unit_rms(float(self._channel), self.noise_request.bandwidth_hz))
            self._sampled_tx_swing = None
        self._equivalent_tx_vpp, _ = signal_calibration(self.noise_request, self._channel)
        # The base env builds its guard before target/channel randomization.
        # Build this experiment's guard against the actual episode specification.
        self._guard = build_evaluator(self._spec(), corner=self.corner, fast=False,
                                     store=self._pilot_store)
        self._noise_score, self._noise_passed = None, False
        self.noise_details = {"status": "nominal_invalid"}
        nominal = super()._measure(x)
        if not nominal.ok:
            return nominal
        dv = decode_action(x)
        srv = get_server(self.corner)
        try:
            measured, score, passed, details = evaluate_snr(srv, dv, nominal, self._spec(),
                self.noise_request, seed=self._noise_seed, eye_bits=self.eye_bits, target_weight=self.target_weight)
            self.n_noise_evaluations += len(details['points'])
            self._noise_score, self._noise_passed, self.noise_details = score, passed, details
            self._heartbeat("scored")
            return measured
        except (NgspiceError, ValueError) as exc:
            self.n_noise_failures += 1
            self.n_invalid += 1
            self.noise_details = {"status": "noise_measurement_failed", "reason": str(exc)}
            return Measures(ok=False)

    def _heartbeat(self, stage):
        if self.heartbeat_path is not None:
            from eqrl.agents.train_noise_pilot import _write_json
            import os, time
            _write_json(self.heartbeat_path, {"pid": os.getpid(), "updated_at_unix": time.time(),
                "stage": stage, "n_sims": self.n_sims, "n_noise_evaluations": self.n_noise_evaluations})

    def _obs(self, measures):
        return np.concatenate((super()._obs(measures), self.noise_request.observation(self._equivalent_tx_vpp))).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("action must have the policy's finite action shape")
        delta = self.step_size * np.clip(action, -1, 1)
        self._x = np.clip((self._x if self._anchor_x is None else self._anchor_x) + delta, 0, 1)
        measured = self._measure(self._x)
        passed = bool(measured.ok and self._noise_passed)
        if measured.ok:
            score = self._noise_score
            reward = score - self._score - 0.05 if self._score is not None else -0.05
            self._score = score
            if passed:
                reward += 10.0
        else:
            # Rejections never reset the last trustworthy improvement baseline.
            reward = self.invalid_reward
        self._heartbeat("step_complete")
        self._t += 1
        info = {"passed": passed, "sim_ok": measured.ok, "target_boost": self._target,
            "design": decode_action(self._x).__dict__, "measures": measured.as_dict(),
            "noise": self.noise_request.to_dict(), "noise_evaluation": self.noise_details}
        if self._last_invalid is not None:
            info["invalid_check"] = self._last_invalid.check.value
        elif not measured.ok:
            info["invalid_check"] = "pilot_noise_measurement_failed"
        return self._obs(measured), float(reward), passed, self._t >= self.horizon, info
