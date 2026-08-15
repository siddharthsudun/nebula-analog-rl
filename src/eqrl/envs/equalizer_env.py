"""Gymnasium environment: the SPICE testbench as an RL environment.

Single-step ("bandit"/contextual) formulation by default — each episode is one design
attempt: the agent emits a full size vector, we simulate, and reward = how close to spec.
This matches the analog-sizing literature (AutoCkt) and is the most sample-efficient
framing. Set `horizon > 1` to allow iterative refinement instead.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # keep import-safe before deps are installed
    gym = object  # type: ignore
    spaces = None  # type: ignore

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.sim.measures import Measures, measure_all
from eqrl.specs import DEFAULT_SPEC, Spec


def _margins(m: Measures, spec: Spec) -> dict[str, float]:
    """Normalized signed margins per spec. Positive = meets, magnitude ~ how comfortably.

    Each is scaled so ~1.0 means 'comfortably met' and negative means 'violated'.
    """
    boost_err = (spec.boost_tol_db - abs(m.boost_db - spec.target_boost_db)) / spec.boost_tol_db  # >=0 within tol
    lo, hi, f = spec.peak_freq_lo_ghz, spec.peak_freq_hi_ghz, m.peak_freq_ghz
    if lo <= f <= hi:
        fpk = 1.0
    else:                                                                # smooth gradient toward band
        edge = lo if f < lo else hi
        fpk = -min(abs(f - edge) / edge, 2.0)
    return {
        "boost":  boost_err,
        "fpeak":  fpk,
        "hd3":    (spec.hd3_db_max - m.hd3_db) / 10.0,                   # lower is better
        "noise":  (spec.noise_vrms_max - m.noise_vrms) / spec.noise_vrms_max,
        "power":  (spec.power_w_max - m.power_w) / spec.power_w_max,
        "area":   (spec.area_mm2_max - m.area_mm2) / spec.area_mm2_max,
        "eye_h":  (m.eye_h_ui - spec.eye_h_ui_min) / spec.eye_h_ui_min,
        "eye_v":  (m.eye_v_mv - spec.eye_v_mv_min) / spec.eye_v_mv_min,
    }


def compute_reward(m: Measures, spec: Spec) -> tuple[float, bool, dict]:
    """Reward = clipped sum of margins + big bonus when every spec passes."""
    if not m.ok:
        return -10.0, False, {"sim_ok": False}
    mg = _margins(m, spec)
    passed = all(v >= 0 for v in mg.values())
    shaped = sum(np.clip(v, -2.0, 1.0) for v in mg.values())
    reward = shaped + (10.0 if passed else 0.0)
    return float(reward), passed, {"sim_ok": True, "margins": mg, "passed": passed}


class EqualizerEnv(gym.Env):  # type: ignore[misc]
    """RL environment wrapping the CTLE+DFE SPICE testbench."""

    metadata = {"render_modes": []}

    def __init__(self, spec: Spec = DEFAULT_SPEC, horizon: int = 1,
                 corner: str = "tt", pvt: bool = False, fast: bool = True):
        super().__init__()
        self.target = spec
        self.horizon = horizon
        self.corner = corner
        self.pvt = pvt
        self.fast = fast   # fast measurement (AC+power+area) for training speed
        n = len(ACTION_SPACE)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        # obs = normalized measurement vector (fixed 10 dims, see Measures)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self._t = 0

    def _obs(self, m: Measures) -> np.ndarray:
        d = m.as_dict()
        keys = ["dc_gain_db", "peak_gain_db", "boost_db", "peak_freq_ghz", "hd3_db",
                "noise_vrms", "power_w", "area_mm2", "eye_h_ui", "eye_v_mv"]
        return np.array([d[k] for k in keys], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(10, dtype=np.float32), {}

    def _evaluate(self, dv):
        if not self.pvt:
            return measure_all(dv, corner=self.corner, vdd=self.target.vdd_nominal,
                               fast=self.fast)
        # PVT: return the WORST corner (Phase 3). Simplified worst-by-reward selection.
        from eqrl.envs.pvt import worst_corner  # lazy import; added in Phase 3
        return worst_corner(dv, self.target)

    def step(self, action):
        # action_space is Box(-1, 1) and the action is an ABSOLUTE sizing, so the
        # decoder must be told the domain. (SequentialEqualizerEnv is different: its
        # Box(-1,1) action is a *delta* applied to an internal [0,1] state, so it
        # decodes with the default "unit" domain.)
        dv = decode_action(action, domain="pm1")
        m = self._evaluate(dv)
        reward, passed, info = compute_reward(m, self.target)
        info["design"] = dv.__dict__
        self._t += 1
        terminated = passed or self._t >= self.horizon
        return self._obs(m), reward, terminated, False, info
