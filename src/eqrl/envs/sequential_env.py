"""Sequential, target-randomized equalizer env — where RL actually wins.

Unlike the 1-step bandit (equalizer_env.py), here an episode is a *design trajectory*:
the agent starts from a random sizing, OBSERVES the current performance-vs-target gap,
and nudges the device sizes over several steps to close it — exactly how a human analog
designer iterates. Crucially the **target boost is randomized every episode**, so the
policy learns to hit *any* spec, not one. That is the win condition the poster asks for:
"given circuit specifications ... reach near-optimal ... with zero human intervention."

After training, the agent retargets to a *new* spec in a handful of sims, while
random/Bayesian search must restart a full search for every new spec.
"""
from __future__ import annotations

import dataclasses

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = object  # type: ignore
    spaces = None  # type: ignore

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.envs.equalizer_env import _margins, compute_reward
from eqrl.sim.measures import Measures, measure_all
from eqrl.specs import DEFAULT_SPEC, Spec

N_PARAM = len(ACTION_SPACE)          # 7 design variables (normalized in [0,1])
_MEAS_KEYS = ["boost_db", "peak_freq_ghz", "power_w", "area_mm2"]

#: Reward for a candidate the guard layer rejects (`guarded=True`).
#:
#: Chosen to match the env's existing penalty for a non-convergent simulation (-5.0 in
#: `_shaped`), so a design that fails validation is treated exactly like a design that
#: failed to simulate — both are "we have no trustworthy measurement here", and neither
#: should look more attractive than a real but poor design.
#:
#: This is a MODELLING CHOICE, not a fact. A larger magnitude teaches stronger avoidance
#: of the infeasible region; a smaller one lets the agent ignore it. Override with
#: `SequentialEqualizerEnv(invalid_reward=...)` once you have evidence either way.
INVALID_REWARD = -5.0


def _shaped(m: Measures, spec: Spec) -> tuple[float, bool]:
    """Dense score (sum of clipped margins) + all-pass flag, without the terminal bonus."""
    if not m.ok:
        return -5.0, False
    mg = _margins(m, spec)
    passed = all(v >= 0 for v in mg.values())
    return float(sum(np.clip(v, -2.0, 1.0) for v in mg.values())), passed


class SequentialEqualizerEnv(gym.Env):  # type: ignore[misc]
    """Design-trajectory env with a randomized target boost each episode."""

    metadata = {"render_modes": []}

    def __init__(self, spec: Spec = DEFAULT_SPEC, horizon: int = 20,
                 corner: str = "tt", fast: bool = True, step_size: float = 0.18,
                 target_range: tuple[float, float] = (4.0, 11.0), seed: int | None = None,
                 pvt: bool = False, guarded: bool = False,
                 invalid_reward: float = INVALID_REWARD):
        super().__init__()
        self.guarded = guarded
        self.invalid_reward = invalid_reward
        self._guard = None
        self._last_invalid = None
        self.n_invalid = 0
        if guarded:
            from eqrl.evaluator import build_evaluator
            self._guard = build_evaluator(spec, corner=corner, fast=fast)
        self.base_spec = spec
        self.horizon = horizon
        self.corner = corner
        self.fast = fast
        self.step_size = step_size
        self.target_range = target_range
        self.pvt = pvt
        # voltage x temperature stress points on the current process corner (no reload;
        # V and T are alterparam'd). Nominal + hot/low-V is the binding pair for this
        # topology (boost & peak-freq drift down hot); cold/high-V is verified in final
        # characterization. Two corners keeps per-step cost ~2x instead of 3x.
        vlo, vnom, vhi = spec.vdd_corners()
        self._vt = [(vnom, 27.0), (vlo, 125.0)]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_PARAM,), dtype=np.float32)
        # obs = params(7) + norm measures(4) + target(1) + boost_gap(1) + fpk_gap(1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_PARAM + 7,),
                                            dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.n_sims = 0

    # -- helpers -----------------------------------------------------------
    def _spec(self) -> Spec:
        return dataclasses.replace(self.base_spec, target_boost_db=self._target)

    def _measure(self, x: np.ndarray) -> Measures:
        dv = decode_action(x)                       # x in [0,1]; decode accepts it
        if self._guard is not None:
            # Guarded path: nothing reaches the reward without passing Tiers 1-4.
            # A rejected candidate is returned as Measures(ok=False), which is exactly
            # how a non-convergent simulation is already represented — both mean "no
            # trustworthy measurement here", and the reward path already handles it.
            self.n_sims += 1
            verdict = self._guard.evaluate(dv, vdd=self.base_spec.vdd_nominal)
            if verdict.is_valid:
                self._last_invalid = None
                return verdict.unwrap()
            self._last_invalid = verdict
            self.n_invalid += 1
            return Measures(ok=False)
        if not self.pvt:
            self.n_sims += 1
            return measure_all(dv, corner=self.corner, vdd=self.base_spec.vdd_nominal,
                               fast=self.fast)
        # PVT-aware: return the worst (lowest-reward) V x T corner on this process corner
        worst_m, worst_r = None, 1e18
        for vdd, temp in self._vt:
            self.n_sims += 1
            m = measure_all(dv, corner=self.corner, vdd=vdd, temp_c=temp, fast=self.fast)
            if not m.ok:
                return m
            r, _, _ = compute_reward(m, self._spec())
            if r < worst_r:
                worst_r, worst_m = r, m
        return worst_m

    def _obs(self, m: Measures) -> np.ndarray:
        spec = self._spec()
        if m.ok:
            norm = [m.boost_db / 12.0, m.peak_freq_ghz / 5.0,
                    m.power_w / spec.power_w_max, m.area_mm2 / spec.area_mm2_max]
            boost_gap = (m.boost_db - self._target) / spec.boost_tol_db
            fc = m.peak_freq_ghz
            if spec.peak_freq_lo_ghz <= fc <= spec.peak_freq_hi_ghz:
                fpk_gap = 0.0
            else:
                edge = spec.peak_freq_lo_ghz if fc < spec.peak_freq_lo_ghz else spec.peak_freq_hi_ghz
                fpk_gap = (fc - edge) / edge
        else:
            norm, boost_gap, fpk_gap = [0, 0, 0, 0], 0.0, 0.0
        return np.array([*self._x, *norm, self._target / 12.0, boost_gap, fpk_gap],
                        dtype=np.float32)

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._target = float(self._rng.uniform(*self.target_range))
        self._x = self._rng.uniform(0.0, 1.0, size=N_PARAM).astype(np.float32)
        self._t = 0
        m = self._measure(self._x)
        self._score, _ = _shaped(m, self._spec())
        return self._obs(m), {"target_boost": self._target}

    def step(self, action):
        self._x = np.clip(self._x + self.step_size * np.asarray(action, dtype=np.float32),
                          0.0, 1.0)
        m = self._measure(self._x)
        spec = self._spec()
        score, passed = _shaped(m, spec)
        reward = (score - self._score) - 0.05           # improvement, small time cost
        self._score = score
        self._t += 1
        if passed:
            reward += 10.0
        terminated = passed
        truncated = self._t >= self.horizon
        info = {"passed": passed, "sim_ok": m.ok, "target_boost": self._target,
                "design": decode_action(self._x).__dict__, "measures": m.as_dict()}
        if self._guard is not None and self._last_invalid is not None:
            # Absolute penalty, not a delta: an INVALID carries no measurement, so
            # "improvement since the last score" is not a meaningful quantity here.
            reward = self.invalid_reward
            info["invalid_check"] = self._last_invalid.check.value
            info["invalid_reason"] = self._last_invalid.reason
            info["artifact_dir"] = str(self._last_invalid.artifact_dir)
        return self._obs(m), reward, terminated, truncated, info
