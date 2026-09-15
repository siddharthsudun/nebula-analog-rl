"""Amortized sample efficiency across specs — measured symmetrically.

A trained sequential agent is asked to hit held-out target boosts. For each we count
the SPICE simulations it needs, and compare against Bayesian optimization run from
scratch per target.

ACCOUNTING (this is the part that has to be right)
--------------------------------------------------
Both arms report the SAME estimator: **total SPICE evaluations until the first design
that passes**, under one procedure, counting everything spent along the way.

The previous version did not. It ran the RL policy from `n_starts=3` random starts and
reported `min()` of the three, discarding the simulations spent by the two losing
starts, while the BO arm reported a first-hitting time under a single run. A best-of-3
order statistic is not comparable to a first-hitting time, and the gap between them is
not a property of the algorithm. It also initialised its counter to 1 while `reset()`
had already run one simulation and the observation refresh ran a second.

`rl_sims_to_spec` now reads `env.n_sims` — the counter the environment already
maintains — so restarts, resets and observation refreshes are all included. The
best-of-3 figure is still reported alongside, labelled, so the two can be compared.

TRAINING COST
-------------
The RL number is per *new* spec and excludes the one-time training run. That training
cost is real and is now reported in the output (`train_env_steps`), because "7x fewer
simulations" is only meaningful next to the number of simulations training consumed and
the break-even spec count.

WHAT "PASSES" MEANS
-------------------
Both arms evaluate with fast=True. Under that setting `measures.measure_all` returns
hardcoded values for HD3, noise and both eye metrics, and power/area cannot be violated
anywhere inside ACTION_SPACE. `live_constraints()` computes which margins can actually
go negative and records them in the output, so the artifact states what was really
being solved rather than implying the full spec was.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from silq.envs.sequential_env import SequentialEqualizerEnv
from silq.specs import DEFAULT_SPEC


def live_constraints(spec=DEFAULT_SPEC, *, fast: bool = True) -> dict:
    """Which spec margins can actually be violated inside the search box?

    A constraint that no reachable design can break is not constraining the search,
    and a benchmark that counts it as 'solved' overstates the difficulty.
    """
    from silq.circuits.ctle import ACTION_SPACE

    frozen = {}
    if fast:
        # measures.measure_all(fast=True) substitutes these constants.
        frozen.update(hd3_db=-40.0, noise_vrms=1.0e-3, eye_h_ui=0.5, eye_v_mv=120.0)

    max_power = 1.8 * ACTION_SPACE["i_tail"][1]
    w_hi, l_hi = ACTION_SPACE["w_in"][1], ACTION_SPACE["l_in"][1]
    max_area = (2 * w_hi * l_hi + 2e-9) * 1e6

    verdict = {}
    verdict["boost"] = "LIVE (varies with the target)"
    verdict["peak_freq"] = f"LIVE (band {spec.peak_freq_lo_ghz}-{spec.peak_freq_hi_ghz} GHz)"
    verdict["power"] = (
        f"DEAD: max reachable {max_power * 1e3:.1f} mW < limit {spec.power_w_max * 1e3:.0f} mW"
        if max_power < spec.power_w_max else "LIVE")
    verdict["area"] = (
        f"DEAD: max reachable {max_area:.4f} mm2 < limit {spec.area_mm2_max} mm2"
        if max_area < spec.area_mm2_max else "LIVE")
    for k, v in frozen.items():
        verdict[k] = f"DEAD: hardcoded to {v!r} under fast=True"
    return verdict


def rl_sims_to_spec(model, env: SequentialEqualizerEnv, target: float,
                    n_starts: int = 3, seed: int = 0) -> dict:
    """Total SPICE sims until the first pass, counting every restart.

    Returns {"total": int|None, "best_start": int|None, "starts_used": int}.
    `total` is the headline, directly comparable to the BO first-hitting time.
    `best_start` is the old best-of-3 statistic, kept for comparison only.
    """
    n0 = env.n_sims
    per_start, total = [], None
    for s in range(n_starts):
        s0 = env.n_sims
        obs, _ = env.reset(seed=seed * 100 + s)
        env._target = target                       # force this target
        obs = env._obs(env._measure(env._x))       # refresh obs for the forced target
        hit = None
        for _ in range(env.horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if info["passed"]:
                hit = env.n_sims - s0
                break
            if trunc:
                break
        per_start.append(hit)
        if hit is not None and total is None:
            total = env.n_sims - n0                # cumulative, incl. failed restarts
            break                                  # first-hitting time: stop here
    hits = [h for h in per_start if h]
    return {"total": total,
            "best_start": min(hits) if hits else None,
            "starts_used": len(per_start)}


def bo_sims_to_spec(target: float, budget: int = 120, seed: int = 0) -> int | None:
    """Bayesian optimization from scratch for one target; sims to the first pass."""
    try:
        import optuna
    except ImportError:
        return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from silq.circuits.ctle import ACTION_SPACE, decode_action
    from silq.sim.measures import measure_all
    from silq.envs.equalizer_env import _margins

    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target)
    n = len(ACTION_SPACE)
    state = {"i": 0, "first": None}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def obj(trial):
        a = np.array([trial.suggest_float(f"a{j}", 0, 1) for j in range(n)])
        m = measure_all(decode_action(a, domain="unit"), corner="tt",
                        vdd=spec.vdd_nominal, fast=True)
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
    p.add_argument("--starts", type=int, default=3)
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=True, seed=7)

    # SB3 records how many env steps the policy was trained for. Each step is one SPICE
    # evaluation; per-episode resets add more, so this is a LOWER BOUND on training cost.
    train_steps = int(getattr(model, "num_timesteps", 0)) or None

    targets = np.linspace(5.0, 11.0, args.targets)
    rows = []
    for t in targets:
        rl = rl_sims_to_spec(model, env, float(t), n_starts=args.starts, seed=int(t * 10))
        bo = bo_sims_to_spec(float(t), budget=args.budget, seed=int(t * 10))
        rows.append({"target": round(float(t), 2),
                     "rl_sims": rl["total"], "rl_best_start": rl["best_start"],
                     "rl_starts_used": rl["starts_used"], "bo_sims": bo})
        print(f"target={t:5.2f}dB   RL(total)={rl['total']}   "
              f"RL(best start)={rl['best_start']}   BO={bo}")

    rl_vals = [r["rl_sims"] for r in rows if r["rl_sims"]]
    bo_vals = [r["bo_sims"] for r in rows if r["bo_sims"]]
    rl_med = float(np.median(rl_vals)) if rl_vals else None
    bo_med = float(np.median(bo_vals)) if bo_vals else None

    breakeven = None
    if train_steps and rl_med and bo_med and bo_med > rl_med:
        breakeven = int(np.ceil(train_steps / (bo_med - rl_med)))

    summary = {
        "accounting": "total SPICE evaluations to first pass, both arms, all restarts counted",
        "rows": rows,
        "rl_median_sims": rl_med,
        "bo_median_sims": bo_med,
        "rl_solved": len(rl_vals), "bo_solved": len(bo_vals), "n_targets": len(targets),
        "train_env_steps": train_steps,
        "train_cost_note": ("lower bound: SB3 num_timesteps; per-episode resets add "
                            "one SPICE eval each and are not included"),
        "breakeven_specs": breakeven,
        "breakeven_note": ("number of new specs before the amortized RL cost undercuts "
                           "running Bayesian from scratch each time"),
        "live_constraints": live_constraints(DEFAULT_SPEC, fast=True),
        "median_excludes_failures": True,
    }
    Path(f"{args.outdir}/generalization.json").write_text(json.dumps(summary, indent=2))

    print("\nRL median sims/spec:", rl_med, " BO median sims/spec:", bo_med)
    if train_steps:
        print(f"training cost: >= {train_steps} SPICE evals"
              + (f"; break-even after ~{breakeven} new specs" if breakeven else ""))
    live = [k for k, v in summary["live_constraints"].items() if v.startswith("LIVE")]
    print(f"constraints actually live during this benchmark: {live}")
    _plot(rows, f"{args.outdir}/generalization.png")


def _plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["target"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [r["rl_sims"] for r in rows], "o-", linewidth=2,
            label="RL (trained; total sims incl. restarts)")
    ax.plot(t, [r["rl_best_start"] for r in rows], "^:", linewidth=1, alpha=0.6,
            label="RL (best single start — not comparable)")
    ax.plot(t, [r["bo_sims"] for r in rows], "s--", linewidth=2,
            label="Bayesian (from scratch per spec)")
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
