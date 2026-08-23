"""Unbiased target-tracking audit: PPO vs random search vs chance, on ONE protocol.

WHY THIS EXISTS, AND WHY IT IS NOT policy_rollout.

policy_rollout answers "how many simulations to first success". It records the design at
the moment it succeeds and `None` otherwise, which is the right artifact for that question
and the wrong one for this one. Correlating requested against achieved boost over its rows
conditions on success; under `--boost-tol` success MEANS |achieved - requested| <= tol, so
the correlation gets computed on a population truncated to exactly the quantity it claims
to measure. That produced a reported correlation of +0.962 for a policy whose
unconditioned correlation is +0.114 (results/target_tracking_compare.json against
results/target_tracking_clean40k.json). Selection on the dependent variable.

This module records EVERY spec's trajectory whether or not it ever succeeds, so the
population is unfiltered by construction and the correlation is honest.

It also puts the methods on ONE protocol, which the existing benchmark does not:
honest_benchmark gives PPO 32 specs at horizon 20 and gives the search baselines 6 specs
x 5 seeds at budget 150, so "PPO 26/32 versus random 30/30" compares different specs at
different budgets. Here every method sees the SAME 32 specs, the SAME evaluation budget,
and the SAME validity test.

Reads only. Changes no threshold, no bound, no reward, and no benchmark criterion.
"""
import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{P/'shim'};{P/'Library'/'bin'};{os.environ['PATH']}"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.specs import DEFAULT_SPEC, hard_pass


def make_specs(n: int, seed: int = 0) -> list[tuple[float, float]]:
    """IDENTICAL to policy_rollout's spec generation -- same seed, same bounds, same
    order -- so spec i here is spec i there and the numbers are directly comparable.

    `seed` defaults to 0, which IS the historical set. Any other value draws from the
    same distribution in the same order and has never been seen by a model-selection
    decision, which is what makes it a clean held-out set (see REPRODUCE.md section 15).
    The distribution itself is never changed -- only which draw you take."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16))) for _ in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--budget", type=int, default=20,
                   help="evaluations per spec per method; 20 = policy_rollout's horizon")
    p.add_argument("--tol", type=float, default=1.5, help="strict target tolerance, dB")
    p.add_argument("--fresh-restarts", action="store_true",
                   help="After an episode terminates, restart from a NEW random sizing "
                        "instead of the same one. WHY THIS MATTERS: the env terminates on "
                        "the first loose pass, and a deterministic policy re-run from the "
                        "same seed replays the identical trajectory -- so without this "
                        "flag PPO spends its budget on ~7 distinct designs while random "
                        "search gets 20. Off by default so the original run reproduces.")
    p.add_argument("--spec-seed", type=int, default=0,
                   help="RNG seed for the spec set. 0 is the historical 32 specs; any "
                        "other value is a clean held-out set at the same distribution")
    p.add_argument("--match-train-env", action="store_true",
                   help="Roll the policy out in the environment it was TRAINED in "
                        "(guarded=True, fast=True) instead of the historical rollout env "
                        "(guarded=False, fast=False). The verification evaluator is "
                        "unaffected -- it stays guarded and fast=False -- so success "
                        "means the same thing. Off by default; on, the trajectory really "
                        "differs and the result is NOT comparable to historical numbers.")
    p.add_argument("--out", default="results/target_audit.json")
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    specs = make_specs(args.specs, args.spec_seed)
    env = SequentialEqualizerEnv(fast=args.match_train_env, seed=123,
                                 guarded=args.match_train_env)

    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        """One evaluation under the SAME validity test both arms use. None = rejected."""
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception:
            return None
        if not v.is_valid:
            return None
        m = v.unwrap()
        ok, _ = hard_pass(m, spec)          # the LOOSE test, unchanged
        return {"boost_db": float(m.boost_db), "dc_gain_db": float(m.dc_gain_db),
                "loose_pass": bool(ok), "design": dataclasses.asdict(dv)}

    def summarize(trace, target):
        """Per-spec outcome. `best` is the closest-to-target design among the LOOSE-passing
        guard-valid ones: the fairest single representative of what the method achieved,
        recorded whether or not it ever met the strict tolerance."""
        ok = [e for e in trace if e and e["loose_pass"]]
        loose_at = next((i + 1 for i, e in enumerate(trace) if e and e["loose_pass"]), None)
        strict_at = next((i + 1 for i, e in enumerate(trace)
                          if e and e["loose_pass"]
                          and abs(e["boost_db"] - target) <= args.tol), None)
        best = min(ok, key=lambda e: abs(e["boost_db"] - target)) if ok else None
        return {"loose_solved_at": loose_at, "strict_solved_at": strict_at,
                "n_valid": sum(1 for e in trace if e), "n_loose_pass": len(ok),
                "best_boost_db": None if best is None else best["boost_db"],
                "best_abs_err": None if best is None else abs(best["boost_db"] - target),
                "best_design": None if best is None else best["design"],
                "all_valid_boosts": [e["boost_db"] for e in trace if e]}

    rng = np.random.default_rng(20260823)   # fixed: the random arm must be reproducible
    n_dim = len(ACTION_SPACE)
    rows = []
    print("%d specs, budget %d evals per method, strict tol +/-%.1f dB\n"
          % (len(specs), args.budget, args.tol))
    for i, (target, channel) in enumerate(specs):
        # --- PPO arm: sequential rollout, identical seeding to policy_rollout ---
        restarts = 0
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        ppo_trace = [evaluate(env._x, target, channel)]
        while len(ppo_trace) < args.budget:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            ppo_trace.append(evaluate(env._x, target, channel))
            if term or trunc:
                restarts += 1
                env.reset(seed=1000 + i
                          + (100000 * restarts if args.fresh_restarts else 0))
                env._target, env._channel = target, channel
                obs = env._obs(env._measure(env._x))
        # --- Random-search arm: same budget, same validity test, independent draws ---
        rand_trace = [evaluate(rng.uniform(0.0, 1.0, size=n_dim), target, channel)
                      for _ in range(args.budget)]

        pp = summarize(ppo_trace, target)
        rr = summarize(rand_trace, target)
        rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                     "ppo": pp, "random": rr})
        print("  spec %2d  target %5.2f  chan %5.2f | PPO loose %-4s strict %-4s err %-6s"
              " | RAND loose %-4s strict %-4s err %s"
              % (i, target, channel,
                 pp["loose_solved_at"], pp["strict_solved_at"],
                 "-" if pp["best_abs_err"] is None else "%.2f" % pp["best_abs_err"],
                 rr["loose_solved_at"], rr["strict_solved_at"],
                 "-" if rr["best_abs_err"] is None else "%.2f" % rr["best_abs_err"]),
              flush=True)
        # Write after EVERY spec. A 32-spec run is ~5 hours of simulation and a killed
        # process used to lose all of it; the file is small and rewriting it is free.
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "specs": args.specs, "budget": args.budget,
             "tol": args.tol, "fresh_restarts": bool(args.fresh_restarts),
             "spec_seed": args.spec_seed,
             "match_train_env": bool(args.match_train_env),
             "complete": len(rows) == len(specs), "rows": rows}, indent=1))

    Path(args.out).write_text(json.dumps(
        {"model": args.model, "specs": args.specs, "budget": args.budget,
         "tol": args.tol, "fresh_restarts": bool(args.fresh_restarts),
         "spec_seed": args.spec_seed,
         "match_train_env": bool(args.match_train_env),
         "complete": True, "rows": rows}, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
