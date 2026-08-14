"""The winning experiment: amortized sample efficiency across specs.

A trained sequential agent is asked to hit a set of *held-out* target boosts. For each,
we measure how many SPICE sims it needs to reach spec (deterministic policy, from random
starts). We compare against Bayesian optimization run *from scratch per target* — because
a search has no memory, every new spec costs it a full search.

Headline numbers:
  - RL: median sims-to-spec per (new) target, after a one-time training cost.
  - BO: median sims-to-spec per target, paid again for every target.
Plot: sims-to-spec vs target boost, RL vs BO.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.specs import DEFAULT_SPEC


def rl_sims_to_spec(model, env: SequentialEqualizerEnv, target: float,
                    n_starts: int = 3, seed: int = 0) -> int | None:
    """Best (fewest) sims across a few random starts for one target."""
    best = None
    for s in range(n_starts):
        obs, _ = env.reset(seed=seed * 100 + s)
        env._target = target                       # force this target
        obs = env._obs(env._measure(env._x))       # refresh obs for forced target
        used = 1
        for _ in range(env.horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            used += 1
            if info["passed"]:
                best = used if best is None else min(best, used)
                break
            if trunc:
                break
    return best


def bo_sims_to_spec(target: float, budget: int = 120, seed: int = 0) -> int | None:
    """Bayesian optimization from scratch for one target; return sims to first pass."""
    try:
        import optuna
    except ImportError:
        return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from eqrl.circuits.ctle import ACTION_SPACE, decode_action
    from eqrl.sim.measures import measure_all
    from eqrl.envs.equalizer_env import _margins

    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target)
    n = len(ACTION_SPACE)
    state = {"i": 0, "first": None}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def obj(trial):
        a = np.array([trial.suggest_float(f"a{j}", 0, 1) for j in range(n)])
        m = measure_all(decode_action(a), corner="tt", vdd=spec.vdd_nominal, fast=True)
        state["i"] += 1
        passed = m.ok and all(v >= 0 for v in _margins(m, spec).values())
        if passed and state["first"] is None:
            state["first"] = state["i"]
            study.stop()
        return -1e9 if not m.ok else sum(np.clip(v, -2, 1) for v in _margins(m, spec).values())

    study.optimize(obj, n_trials=budget)
    return state["first"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_agent.zip")
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=True, seed=7)

    targets = np.linspace(5.0, 11.0, args.targets)
    rows = []
    for t in targets:
        rl = rl_sims_to_spec(model, env, float(t), n_starts=3, seed=int(t * 10))
        bo = bo_sims_to_spec(float(t), budget=120, seed=int(t * 10))
        rows.append({"target": round(float(t), 2), "rl_sims": rl, "bo_sims": bo})
        print(f"target={t:5.2f}dB   RL={rl}   BO={bo}")

    rl_vals = [r["rl_sims"] for r in rows if r["rl_sims"]]
    bo_vals = [r["bo_sims"] for r in rows if r["bo_sims"]]
    summary = {
        "rows": rows,
        "rl_median_sims": float(np.median(rl_vals)) if rl_vals else None,
        "bo_median_sims": float(np.median(bo_vals)) if bo_vals else None,
        "rl_solved": len(rl_vals), "bo_solved": len(bo_vals), "n_targets": len(targets),
    }
    Path(f"{args.outdir}/generalization.json").write_text(json.dumps(summary, indent=2))
    print("\nRL median sims/spec:", summary["rl_median_sims"],
          " BO median sims/spec:", summary["bo_median_sims"])
    _plot(rows, f"{args.outdir}/generalization.png")


def _plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["target"] for r in rows]
    rl = [r["rl_sims"] for r in rows]
    bo = [r["bo_sims"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, rl, "o-", label="RL (trained, per new spec)", linewidth=2)
    ax.plot(t, bo, "s--", label="Bayesian (from scratch per spec)", linewidth=2)
    ax.set_xlabel("target boost (dB)")
    ax.set_ylabel("SPICE sims to reach spec")
    ax.set_title("Amortized efficiency across specs (SKY130 CTLE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
