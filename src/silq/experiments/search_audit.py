"""CMA-ES and TPE under the SAME matched protocol as `target_audit`.

WHY THIS EXISTS. `honest_benchmark` runs the search baselines on a different protocol
from the policy: budget 150, on the first 6 specs x 5 seeds, against an unguarded
success test. The policy runs 32 specs at horizon 20 against a guarded one. "RL 26/32
versus CMA-ES 30/30" therefore compares different methods on different specs at
different budgets under different criteria, and means nothing.

This module puts CMA-ES and TPE on the protocol `target_audit` froze:

    same 32 specs        -- imported from target_audit.make_specs, not re-derived
    same budget          -- 20 evaluations per spec, default
    same evaluator       -- silq.evaluator.build_evaluator, corner tt, fast=False,
                            one guard per channel loss
    same validity test   -- guard-valid (Tiers 1-4) AND specs.hard_pass
    same loose criterion -- hard_pass with the target UNSCORED (3-12 dB range)
    same strict criterion-- |achieved - requested| <= tol, tol = 1.5 dB
    same reporting       -- the same solved_at / best / correlation logic as
                            target_audit.summarize, spelled out identically
    full trajectory      -- every evaluation recorded, success or not, so correlation
                            is computed on the unfiltered population

THE OPTIMIZERS ARE NOT TUNED. Both configurations are the ones already in
`honest_benchmark`: CMA-ES from x0 = 0.5 with sigma0 = 0.25, bounds [0, 1], library
default population size; TPE with optuna's default sampler. Only the EVALUATION WRAPPER
was adapted -- the objective the optimizer minimizes is `honest_benchmark`'s existing
dense score, unchanged.

ONE DELIBERATE ASYMMETRY, STATED. `honest_benchmark` stops a solver the moment it
succeeds. Here neither optimizer stops early: both spend the full budget, because the
protocol records the whole trajectory and because stopping at the LOOSE success would
throw away exactly the later evaluations that could have hit the STRICT target. Both
`loose_solved_at` and `strict_solved_at` are still the index of the FIRST evaluation
that qualified, so the cost numbers stay comparable to a stopping run.

Reads only. Changes no threshold, bound, reward, or benchmark criterion.
"""
import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "silq-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from silq.circuits.ctle import ACTION_SPACE, decode_action
from silq.evaluator import build_evaluator
from silq.experiments.target_audit import make_specs
from silq.specs import DEFAULT_SPEC, hard_pass

N = len(ACTION_SPACE)

#: Score a guard-rejected or non-simulating candidate. Matches the constant
#: `honest_benchmark.evaluate` already returns when `m.ok` is False, so the optimizers
#: see the same shape of landscape they see there.
REJECT_SCORE = -10.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["cmaes", "tpe"])
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--budget", type=int, default=20,
                   help="evaluations per spec; 20 = the matched protocol's budget")
    p.add_argument("--tol", type=float, default=1.5, help="strict target tolerance, dB")
    p.add_argument("--seed", type=int, default=20260823,
                   help="base seed; spec i runs at seed + i. Fixed and reported, not "
                        "searched over -- picking a seed by result would be cherry-picking")
    p.add_argument("--sigma0", type=float, default=0.25,
                   help="CMA-ES initial step. 0.25 is honest_benchmark's existing value")
    p.add_argument("--spec-seed", type=int, default=0,
                   help="RNG seed for the spec set, passed straight to "
                        "target_audit.make_specs. 0 is the historical 32 specs; any "
                        "other value is a clean held-out set at the same distribution. "
                        "The OPTIMIZER seeds are unchanged by this, so a seed-1 run "
                        "differs from a seed-0 run in the specs and nothing else.")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or (f"results/search_audit_{args.method}"
                       + (f"_seed{args.spec_seed}" if args.spec_seed else "")
                       + ".json")

    specs = make_specs(args.specs, args.spec_seed)
    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        """One evaluation. Returns (record_or_None, objective_score).

        The record is exactly what `target_audit.evaluate` returns, so `summarize` reads
        it unchanged. The score is `honest_benchmark`'s dense objective: how many hard
        checks pass, minus the target miss in units of 3 dB.
        """
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception:
            return None, REJECT_SCORE
        if not v.is_valid:
            return None, REJECT_SCORE
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        score = (sum(1.0 for c in checks.values() if c)
                 - abs(m.boost_db - target) / 3.0)
        rec = {"boost_db": float(m.boost_db), "dc_gain_db": float(m.dc_gain_db),
               "loose_pass": bool(ok), "design": dataclasses.asdict(dv)}
        return rec, float(score)

    def run_cmaes(target, channel, seed):
        """Existing honest_benchmark configuration, run to the full budget."""
        import cma
        es = cma.CMAEvolutionStrategy(N * [0.5], args.sigma0,
                                      {"bounds": [0, 1], "maxfevals": args.budget,
                                       "seed": seed, "verbose": -9})
        trace = []
        while len(trace) < args.budget:
            xs = es.ask()
            costs = []
            for x in xs:
                if len(trace) < args.budget:
                    rec, score = evaluate(x, target, channel)
                    trace.append(rec)
                else:
                    # The population outran the budget. Do not simulate: the candidate
                    # would cost a real evaluation this method is not allowed to spend.
                    score = REJECT_SCORE
                costs.append(-score)
            es.tell(xs, costs)
        return trace, {"popsize": int(es.popsize), "sigma0": args.sigma0,
                       "x0": 0.5, "bounds": [0, 1], "seed": seed,
                       "restarts": 0, "generations": -(-args.budget // int(es.popsize))}

    def run_tpe(target, channel, seed):
        """Optuna's default TPE sampler, run to the full budget."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        trace = []
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def obj(trial):
            x = [trial.suggest_float(f"a{j}", 0, 1) for j in range(N)]
            rec, score = evaluate(x, target, channel)
            trace.append(rec)
            return score

        study.optimize(obj, n_trials=args.budget)
        return trace, {"sampler": "TPESampler", "n_startup_trials":
                       int(sampler._n_startup_trials), "seed": seed,
                       "n_trials": args.budget, "restarts": 0}

    runner = run_cmaes if args.method == "cmaes" else run_tpe
    rows = []
    print("%s | %d specs, budget %d evals, strict tol +/-%.1f dB, base seed %d\n"
          % (args.method.upper(), len(specs), args.budget, args.tol, args.seed),
          flush=True)
    for i, (target, channel) in enumerate(specs):
        trace, cfg = runner(target, channel, args.seed + i)
        # target_audit.summarize closes over its own CLI's tol, so the identical logic is
        # spelled out here rather than imported with a different tolerance baked in.
        ok = [e for e in trace if e and e["loose_pass"]]
        loose_at = next((k + 1 for k, e in enumerate(trace) if e and e["loose_pass"]), None)
        strict_at = next((k + 1 for k, e in enumerate(trace)
                          if e and e["loose_pass"]
                          and abs(e["boost_db"] - target) <= args.tol), None)
        best = min(ok, key=lambda e: abs(e["boost_db"] - target)) if ok else None
        res = {"loose_solved_at": loose_at, "strict_solved_at": strict_at,
               "n_valid": sum(1 for e in trace if e), "n_loose_pass": len(ok),
               "n_evals": len(trace),
               "n_distinct_boosts": len({round(e["boost_db"], 6)
                                         for e in trace if e}),
               "best_boost_db": None if best is None else best["boost_db"],
               "best_abs_err": None if best is None else abs(best["boost_db"] - target),
               "best_design": None if best is None else best["design"],
               "all_valid_boosts": [e["boost_db"] for e in trace if e]}
        rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                     "config": cfg, args.method: res})
        print("  spec %2d  target %5.2f  chan %5.2f | loose %-4s strict %-4s "
              "valid %2d/%d  err %s"
              % (i, target, channel, res["loose_solved_at"], res["strict_solved_at"],
                 res["n_valid"], res["n_evals"],
                 "-" if res["best_abs_err"] is None else "%.2f" % res["best_abs_err"]),
              flush=True)
        Path(out).write_text(json.dumps(
            {"method": args.method, "specs": args.specs, "budget": args.budget,
             "tol": args.tol, "base_seed": args.seed, "spec_seed": args.spec_seed,
             "complete": len(rows) == len(specs), "rows": rows}, indent=1))

    print("\nwrote", out)


if __name__ == "__main__":
    main()
