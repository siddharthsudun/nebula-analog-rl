"""The decisive experiment (audit Part 18), done honestly.

Compares Random, Bayesian (Optuna TPE), CMA-ES, and the trained RL policy on the SAME
task: reach a design that passes ALL EIGHT hard specs (verified at fast=False), for a set
of (target boost, channel loss) specifications the RL policy never trained on.

Fairness rules:
  - success = all 8 specs pass, measured full (fast=False), identical for every method
  - RL gets a single deterministic rollout (no best-of-k)
  - every method uses the identical simulator, ranges, and success test
  - RL additionally carries its measured training cost as an upfront debt
  - 1 "simulation" = 1 candidate evaluation (one measure_all)

Outputs: results/benchmark.json and two figures — per-spec cost and the cumulative-cost
amortization curve (where RL overtakes search).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.sim.measures import measure_all
from eqrl.specs import DEFAULT_SPEC, hard_pass

N = len(ACTION_SPACE)


def evaluate(x, target, channel):
    """One candidate -> (passed_all_8, shaped_score). Full spec (fast=False)."""
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target, channel_loss_db=channel)
    m = measure_all(decode_action(np.asarray(x)), corner="tt", vdd=spec.vdd_nominal,
                    fast=False, channel_loss_db=channel)
    ok, checks = hard_pass(m, spec)
    if not m.ok:
        return False, -10.0
    # dense score: how many checks pass + soft boost closeness
    score = sum(1.0 for v in checks.values() if v) - abs(m.boost_db - target) / 3.0
    return ok, score


# ---- search baselines: return sims-to-first-full-pass (or None) ----
def solve_random(target, channel, budget, seed):
    rng = np.random.default_rng(seed)
    for i in range(budget):
        ok, _ = evaluate(rng.uniform(0, 1, N), target, channel)
        if ok:
            return i + 1
    return None


def solve_cmaes(target, channel, budget, seed):
    import cma
    es = cma.CMAEvolutionStrategy(N * [0.5], 0.25,
                                  {"bounds": [0, 1], "maxfevals": budget, "seed": seed,
                                   "verbose": -9})
    used = 0
    while not es.stop() and used < budget:
        xs = es.ask()
        costs = []
        for x in xs:
            ok, score = evaluate(x, target, channel)
            used += 1
            if ok:
                return used
            costs.append(-score)
        es.tell(xs, costs)
    return None


def solve_tpe(target, channel, budget, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    state = {"i": 0, "first": None}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def obj(trial):
        x = [trial.suggest_float(f"a{j}", 0, 1) for j in range(N)]
        ok, score = evaluate(x, target, channel)
        state["i"] += 1
        if ok and state["first"] is None:
            state["first"] = state["i"]
            study.stop()
        return score

    study.optimize(obj, n_trials=budget)
    return state["first"]


def solve_rl(model, env, target, channel, seed):
    """Single deterministic rollout; sims-to-full-pass (verified fast=False)."""
    obs, _ = env.reset(seed=seed)
    env._target, env._channel = target, channel
    obs = env._obs(env._measure(env._x))
    used = 1
    for _ in range(env.horizon):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        used += 1
        ok, _ = evaluate(env._x, target, channel)     # full-spec verification
        if ok:
            return used
        if trunc:
            break
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_agent.zip",
                   help="trained policy; must be a path train_sequential actually wrote")
    p.add_argument("--train-cost", type=int, default=None,
                   help="RL training sims (debt). Read from the model's _train.json "
                        "sidecar when omitted — this number sets the break-even point, "
                        "so it has to come from the run that produced the model.")
    p.add_argument("--rl-specs", type=int, default=24)
    p.add_argument("--search-specs", type=int, default=6)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=150)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"no model at {model_path}. Train one first:\n"
            f"  python -m eqrl.agents.train_sequential --guarded --no-fast "
            f"--out {model_path}")
    train_cost = args.train_cost
    if train_cost is None:
        sidecar = model_path.with_name(model_path.stem + "_train.json")
        if not sidecar.exists():
            raise SystemExit(
                f"no training record at {sidecar}, so the RL training debt is unknown. "
                "That number decides where RL breaks even against the search baselines, "
                "and guessing it would make the amortization plot fiction. Retrain (the "
                "sidecar is written automatically) or pass --train-cost explicitly.")
        rec = json.loads(sidecar.read_text())
        train_cost = int(rec["n_sims"])
        print(f"training debt: {train_cost} sims (from {sidecar.name}, "
              f"{rec.get('timesteps')} timesteps, guarded={rec.get('guarded')}, "
              f"fast={rec.get('fast')})")

    rng = np.random.default_rng(0)
    specs = [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16)))
             for _ in range(args.rl_specs)]

    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(str(model_path))
    env = SequentialEqualizerEnv(fast=False, seed=123)

    # RL on all specs (cheap)
    rl = [solve_rl(model, env, t, c, seed=i) for i, (t, c) in enumerate(specs)]
    rl_solved = [v for v in rl if v]
    print(f"RL: solved {len(rl_solved)}/{len(specs)}  median {np.median(rl_solved):.1f} sims/spec")

    # search methods on a subset, multiple seeds
    def med(fn):
        vals = []
        for (t, c) in specs[:args.search_specs]:
            for s in range(args.seeds):
                v = fn(t, c, args.budget, s)
                if v:
                    vals.append(v)
        return vals
    print("running random ..."); rand = med(solve_random)
    print("running CMA-ES ..."); cmaes = med(solve_cmaes)
    print("running TPE ...");    tpe = med(solve_tpe)

    def summ(v, n):
        return {"solved": len(v), "attempts": n, "median": float(np.median(v)) if v else None,
                "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v else None}

    ns = args.search_specs * args.seeds
    out = {
        "rl": {"solved": len(rl_solved), "specs": len(specs),
               "median": float(np.median(rl_solved)) if rl_solved else None,
               "train_cost": train_cost},
        "random": summ(rand, ns), "cmaes": summ(cmaes, ns), "tpe": summ(tpe, ns),
    }
    Path(f"{args.outdir}/benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    _plot(out, rl_solved, f"{args.outdir}/amortization.png")


def _plot(out, rl_sims, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n = max(len(rl_sims), 20)
    xs = np.arange(1, n + 1)
    # RL: upfront training debt, then measured per-spec cost
    rl_med = np.median(rl_sims) if rl_sims else 15
    ax.plot(xs, out["rl"]["train_cost"] + xs * rl_med, label="RL (trained)", lw=2.4,
            color="#17b7a8")
    for key, col, lab in [("cmaes", "#c0603a", "CMA-ES"),
                          ("tpe", "#d9a441", "Bayesian (TPE)"),
                          ("random", "#8a8f98", "Random")]:
        mv = out[key]["median"]
        if mv:
            ax.plot(xs, xs * mv, label=f"{lab} (from scratch)", lw=2, ls="--", color=col)
    # break-even annotation vs CMA-ES
    if out["cmaes"]["median"]:
        be = out["rl"]["train_cost"] / max(out["cmaes"]["median"] - rl_med, 1e-6)
        if 0 < be < n:
            ax.axvline(be, color="#17b7a8", ls=":", alpha=0.6)
            ax.text(be, ax.get_ylim()[1] * 0.9, f"  break-even ≈ {be:.0f} specs",
                    color="#17b7a8", fontsize=9)
    ax.set_xlabel("number of specifications solved")
    ax.set_ylabel("cumulative SPICE simulations")
    ax.set_title("Amortized cost: RL vs search-from-scratch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"amortization curve -> {path}")


if __name__ == "__main__":
    main()
