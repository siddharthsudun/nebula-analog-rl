"""Headline experiment: RL vs sweep sample efficiency.

Runs random search, (optional) Bayesian, and a PPO agent on the SAME env with the SAME
per-eval tracking, then writes histories + a plot of best-reward-so-far vs #SPICE-evals.
This is the graph that proves the poster's own ask: reach spec in fewer simulations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eqrl.agents.tracking import TrackWrapper
from eqrl.envs.equalizer_env import EqualizerEnv
from eqrl.specs import DEFAULT_SPEC


def run_random(budget: int, seed: int = 0) -> TrackWrapper:
    env = TrackWrapper(EqualizerEnv(fast=True))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        env.step(rng.uniform(-1, 1, env.action_space.shape[0]))
    return env


def run_bayes(budget: int, seed: int = 0) -> TrackWrapper | None:
    try:
        import optuna
    except ImportError:
        return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    env = TrackWrapper(EqualizerEnv(fast=True))
    env.reset(seed=seed)
    n = env.action_space.shape[0]

    def obj(trial):
        a = np.array([trial.suggest_float(f"a{j}", -1, 1) for j in range(n)])
        _, r, _, _, _ = env.step(a)
        return r

    optuna.create_study(direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=seed)
                        ).optimize(obj, n_trials=budget)
    return env


def run_rl(budget: int, seed: int = 0) -> TrackWrapper:
    from stable_baselines3 import PPO

    env = TrackWrapper(EqualizerEnv(fast=True, horizon=1))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                n_steps=256, batch_size=64, gamma=0.0, ent_coef=0.01)
    model.learn(total_timesteps=budget)
    return env


def plot(histories: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in histories.items():
        ax.plot(h.n_sims, h.best_reward, label=name, linewidth=2)
    ax.set_xlabel("# SPICE evaluations")
    ax.set_ylabel("best reward so far")
    ax.set_title("Sample efficiency: RL vs sweep (SKY130 CTLE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)

    runs = {}
    print("== random =="); runs["random"] = run_random(args.budget, args.seed)
    b = run_bayes(args.budget, args.seed)
    if b:
        print("== bayesian =="); runs["bayesian"] = b
    print("== PPO =="); runs["PPO (RL)"] = run_rl(args.budget, args.seed)

    summary = {}
    for name, env in runs.items():
        h = env.h
        h.to_csv(f"{args.outdir}/history_{name.split()[0].lower()}.csv")
        summary[name] = {
            "first_pass_sim": h.first_pass_sim,
            "best_reward": max(h.best_reward),
            "best_design": h.best_design,
        }
        print(f"{name:14s} first_pass={h.first_pass_sim} best_reward={max(h.best_reward):.2f}")

    Path(f"{args.outdir}/summary.json").write_text(json.dumps(summary, indent=2))
    plot({n: e.h for n, e in runs.items()}, f"{args.outdir}/sample_efficiency.png")


if __name__ == "__main__":
    main()
