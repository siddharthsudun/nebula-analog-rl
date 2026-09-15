"""Tracking wrapper: log every SPICE evaluation so we can plot sample efficiency.

Wraps the env and records, per step: cumulative sim count, reward, whether the design
passed, and the design + measurements. This is what produces the headline curve
(best-reward / spec-met vs #SPICE-evals, RL vs baselines).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import gymnasium as gym


@dataclass
class History:
    n_sims: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    best_reward: list[float] = field(default_factory=list)
    passed: list[bool] = field(default_factory=list)
    first_pass_sim: int | None = None
    best_design: dict | None = None
    best_measures: dict | None = None

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_sims", "reward", "best_reward", "passed"])
            for row in zip(self.n_sims, self.reward, self.best_reward, self.passed):
                w.writerow(row)


class TrackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.h = History()
        self._n = 0
        self._best = -1e18

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._n += 1
        if reward > self._best:
            self._best = reward
            self.h.best_design = info.get("design")
            self.h.best_measures = info.get("margins")
        self.h.n_sims.append(self._n)
        self.h.reward.append(float(reward))
        self.h.best_reward.append(float(self._best))
        passed = bool(info.get("passed"))
        self.h.passed.append(passed)
        if passed and self.h.first_pass_sim is None:
            self.h.first_pass_sim = self._n
        return obs, reward, term, trunc, info
