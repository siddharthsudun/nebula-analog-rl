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
    """Normalized signed margins for the 8 HARD specs — one per pass/fail check in
    specs.hard_pass, so the training reward optimizes exactly what is scored. Each is >=0
    iff that spec is met; magnitude ~ how comfortably. Boost is the tunable 3-12 dB range
    (the agent picks the peaking the channel needs), not a per-target tolerance.
    """
    lo, hi, f = spec.peak_freq_lo_ghz, spec.peak_freq_hi_ghz, m.peak_freq_ghz
    if lo <= f <= hi:
        fpk = 1.0
    else:                                                                # smooth gradient toward band
        edge = lo if f < lo else hi
        fpk = -min(abs(f - edge) / edge, 2.0)
    out = {
        "boost_lo": (m.boost_db - spec.boost_db_min) / 3.0,             # >= 3 dB peaking
        "boost_hi": (spec.boost_db_max - m.boost_db) / 3.0,             # <= 12 dB
        "fpeak":  fpk,
        "hd3":    (spec.hd3_db_max - m.hd3_db) / 10.0,                   # lower is better
        "noise":  (spec.noise_vrms_max - m.noise_vrms) / spec.noise_vrms_max,
        "power":  (spec.power_w_max - m.power_w) / spec.power_w_max,
        "area":   (spec.area_mm2_max - m.area_mm2) / spec.area_mm2_max,
        "eye_h":  (m.eye_h_ui - spec.eye_h_ui_min) / spec.eye_h_ui_min,
        "eye_v":  (m.eye_v_mv - spec.eye_v_mv_min) / spec.eye_v_mv_min,
    }
    # DC gain, when the spec asks for it. Its ABSENCE was the defect that stalled this
    # project: boost is peak MINUS DC, so a stage that attenuates at DC manufactures boost
    # for free, and nothing in the other nine margins notices -- HD3, noise, power, area
    # and both eye terms are all EASIER for a stage that passes less signal.
    #
    # Two independent optimisers found this unprompted. CMA-ES: 14 of 28 spec-passing
    # designs were rejected by the guard as T4.10_dc_gain_implausible
    # (results/pass_vs_valid.json). PPO after 40k steps: 4 of 6 held-out rollouts, same
    # check (results/policy_rollout.json). One verified example measured
    # dc_gain -5.00 dB, peak +1.57 dB, "boost" 6.56 dB -- a circuit with no gain at any
    # frequency, passing every published spec.
    #
    # The agent was optimising exactly what it was scored on. The score was wrong.
    if spec.dc_gain_db_min is not None:
        out["dc_gain"] = (m.dc_gain_db - spec.dc_gain_db_min) / 3.0
    return out


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
