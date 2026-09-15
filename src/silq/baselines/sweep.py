"""Baseline searches — the thing RL must beat.

Random search and grid search over the action space, plus an optional Bayesian
optimizer (Optuna). We record how many SPICE evaluations each needs to first meet spec;
the RL agent's curve should dominate these. This comparison IS the submission's headline.
"""
from __future__ import annotations

import argparse

import numpy as np

from silq.circuits.ctle import ACTION_SPACE
from silq.envs.equalizer_env import EqualizerEnv


def random_search(env: EqualizerEnv, budget: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(ACTION_SPACE)
    best = -1e9
    first_pass = None
    for i in range(budget):
        a = rng.uniform(-1, 1, size=n)
        _, r, _, _, info = env.step(a)
        if r > best:
            best = r
        if first_pass is None and info.get("passed"):
            first_pass = i + 1
    return {"method": "random", "best_reward": best, "sims_to_spec": first_pass,
            "budget": budget}


def bayesian_search(env: EqualizerEnv, budget: int, seed: int = 0):
    try:
        import optuna
    except ImportError:
        return {"method": "bayesian", "error": "pip install optuna"}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(ACTION_SPACE)
    state = {"first_pass": None, "i": 0}

    def objective(trial):
        a = [trial.suggest_float(f"a{j}", -1, 1) for j in range(n)]
        _, r, _, _, info = env.step(np.array(a))
        state["i"] += 1
        if state["first_pass"] is None and info.get("passed"):
            state["first_pass"] = state["i"]
        return r

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget)
    return {"method": "bayesian", "best_reward": study.best_value,
            "sims_to_spec": state["first_pass"], "budget": budget}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--method", default="all", choices=["random", "bayesian", "all"])
    args = p.parse_args()

    env = EqualizerEnv()
    if args.method in ("random", "all"):
        print(random_search(env, args.budget))
    if args.method in ("bayesian", "all"):
        print(bayesian_search(env, args.budget))


if __name__ == "__main__":
    main()
