"""Sample efficiency at ONE fixed target: RL vs random vs Bayesian.

Companion to `generalization.py`, which sweeps across targets. Same estimator in both:
**total SPICE evaluations until the first design that passes**, counted identically for
every arm.

WHAT CHANGED AND WHY
--------------------
The previous version ran PPO against `EqualizerEnv(horizon=1)` — the one-step bandit
whose `reset()` returns a vector of zeros. With a constant observation and gamma=0 a
policy cannot condition on anything; it can only learn a state-independent action
distribution, which is random search with extra machinery. That is why the committed
`results/summary.json` showed PPO at 659 sims against Bayesian's 57: the number was
measuring the wrong environment, not the algorithm.

RL now runs on `SequentialEqualizerEnv`, the target-randomized MDP the policy is
actually trained on, and every arm counts sims the same way. Both the random and
Bayesian arms sample the design space directly, so no arm gets a free reset.

Counting note: `SequentialEqualizerEnv.n_sims` increments on reset AND on each step, so
reading it captures the resets that a per-step counter would miss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.envs.equalizer_env import _margins
from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.experiments.generalization import live_constraints
from eqrl.specs import DEFAULT_SPEC
from eqrl.sim.measures import measure_all

N_PARAM = len(ACTION_SPACE)


def _passes(m, spec) -> bool:
    return bool(m.ok) and all(v >= 0 for v in _margins(m, spec).values())


def _score(m, spec) -> float:
    if not m.ok:
        return -1e9
    return float(sum(np.clip(v, -2, 1) for v in _margins(m, spec).values()))


def run_random(spec, budget: int, seed: int = 0) -> dict:
    """Uniform sampling of the design box. Domain is [0,1] — the decoder's own units."""
    rng = np.random.default_rng(seed)
    best, best_design, first = -1e18, None, None
    for i in range(budget):
        a = rng.uniform(0.0, 1.0, size=N_PARAM)
        dv = decode_action(a, domain="unit")
        m = measure_all(dv, corner="tt", vdd=spec.vdd_nominal, fast=True)
        s = _score(m, spec)
        if s > best:
            best, best_design = s, dv.__dict__.copy()
        if first is None and _passes(m, spec):
            first = i + 1
            break
    return {"method": "random", "first_pass_sim": first, "best_score": best,
            "best_design": best_design, "budget": budget}


def run_bayes(spec, budget: int, seed: int = 0) -> dict | None:
    try:
        import optuna
    except ImportError:
        return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    state = {"i": 0, "first": None, "best": -1e18, "design": None}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def obj(trial):
        a = np.array([trial.suggest_float(f"a{j}", 0, 1) for j in range(N_PARAM)])
        dv = decode_action(a, domain="unit")
        m = measure_all(dv, corner="tt", vdd=spec.vdd_nominal, fast=True)
        state["i"] += 1
        s = _score(m, spec)
        if s > state["best"]:
            state["best"], state["design"] = s, dv.__dict__.copy()
        if state["first"] is None and _passes(m, spec):
            state["first"] = state["i"]
            study.stop()
        return s

    study.optimize(obj, n_trials=budget)
    return {"method": "bayesian", "first_pass_sim": state["first"],
            "best_score": state["best"], "best_design": state["design"],
            "budget": budget}


def run_rl(model_path: str, spec, budget: int, starts: int = 3, seed: int = 0) -> dict:
    """Trained policy on the sequential env, counting every restart."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = SequentialEqualizerEnv(fast=True, seed=seed)
    n0, first, best, design = env.n_sims, None, -1e18, None
    for s in range(starts):
        obs, _ = env.reset(seed=seed * 100 + s)
        env._target = spec.target_boost_db
        obs = env._obs(env._measure(env._x))
        for _ in range(env.horizon):
            if env.n_sims - n0 >= budget:
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if r > best:
                best, design = float(r), info.get("design")
            if info.get("passed"):
                first = env.n_sims - n0
                break
            if trunc:
                break
        if first is not None:
            break
    return {"method": "PPO (RL)", "first_pass_sim": first, "best_score": best,
            "best_design": design, "budget": budget,
            "train_env_steps": int(getattr(model, "num_timesteps", 0)) or None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="results/seq_agent.zip",
                   help="trained sequential policy; RL arm is skipped if absent")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)
    spec = DEFAULT_SPEC

    runs = {}
    print("== random ==")
    runs["random"] = run_random(spec, args.budget, args.seed)
    b = run_bayes(spec, args.budget, args.seed)
    if b:
        print("== bayesian ==")
        runs["bayesian"] = b
    if Path(args.model).exists():
        print("== PPO (RL) ==")
        runs["PPO (RL)"] = run_rl(args.model, spec, args.budget, seed=args.seed)
    else:
        print(f"[skip] RL arm: no trained policy at {args.model}. Train one with "
              f"`python -m eqrl.agents.train_sequential` first — an untrained or "
              f"missing policy must not be reported as an RL result.")

    for name, r in runs.items():
        print(f"{name:14s} first_pass={r['first_pass_sim']} best={r['best_score']:.2f}")

    summary = {
        "accounting": "total SPICE evaluations to first pass; identical for every arm",
        "target_boost_db": spec.target_boost_db,
        "live_constraints": live_constraints(spec, fast=True),
        **runs,
    }
    Path(f"{args.outdir}/summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {args.outdir}/summary.json")


if __name__ == "__main__":
    main()
