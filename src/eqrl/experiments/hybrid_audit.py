"""The two-stage designer: PPO supplies feasibility, CMA-ES supplies precision.

PRE-REGISTERED. docs/REPRODUCE.md sections 17 and 18, both committed before this file
existed and before any spec-seed-2 or spec-seed-3 simulation ran. Every constant below is
read from that protocol, not chosen here.

WHY. `seq_clean40k` is measured-good at feasibility -- 26/32 loose, median 4 evaluations --
and measured-not-good at hitting a requested boost: it sits ON its matched chance line on
both spec sets tested. Rather than ask one policy to do both jobs, this hands the design
off once the policy has produced something feasible, and lets a local optimizer aim it.

    spec (target, channel)
      -> STAGE 1  frozen seq_clean40k, k = 5 evaluations     [2.00 measure_all each]
      -> the design the policy currently holds
      -> STAGE 2  CMA-ES from that design, r = 10 evaluations [1.00 measure_all each]
      -> verification: guarded, fast=False, OUTSIDE the loop, identical to every other arm

    2k + r = 20 measure_all, exactly what every baseline arm spends (section 13 measured
    the 2.00 / 1.00 costs; matching SIMULATION rather than evaluation count is what makes
    the comparison fair).

TWO ARMS, DIFFERING IN EXACTLY ONE THING (amendment 1, section 18.2):

    H1  each generation: ask CMA-ES for popsize candidates
                         SPICE all of them
    H2  each generation: ask CMA-ES for 20 x popsize candidates
                         rank by SURROGATE-PREDICTED abs(boost - target)
                         SPICE only the popsize best

Same optimizer, same seed, same sigma0, same handoff, same budget, same objective. The
only difference is WHICH candidates get simulated, so `H2 - H1` is the value of surrogate
pre-screening and nothing else.

WHAT THIS CANNOT CLAIM. If it works, the honest statement is that a two-stage designer
hits requested specs above matched chance -- with the RL policy supplying feasibility and
the refiner supplying precision. It is NOT evidence that the policy learned to retarget.
That distinction is the point of the experiment and has to survive into the writeup.

THE SUCCESS CRITERION IS NOT THE STRICT SOLVE COUNT. Section 8 already measured a
6/32 -> 16/32 and 8/32 -> 20/32 strict gain that was pure coverage: more feasible designs
produce more accidental target hits. This arm must clear its OWN per-method matched chance
line, with k counted as DISTINCT designs, on spec seed 2, and replicate on seed 3.

Changes no reward, PPO hyperparameter, design bound, benchmark criterion, or frozen
checkpoint. Adds an arm; modifies nothing.
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
from eqrl.experiments.target_audit import make_specs
from eqrl.specs import DEFAULT_SPEC, hard_pass

N = len(ACTION_SPACE)

#: Same constant `search_audit` and `honest_benchmark` give a rejected candidate, so the
#: optimizer sees the same shape of landscape it sees in the baseline arms.
REJECT_SCORE = -10.0

#: PRE-REGISTERED (section 17). Duplicated here so a drift from the committed protocol is
#: a loud failure rather than a silent one; `main` asserts the CLI has not moved them.
PREREG = {"k": 5, "r": 10, "sigma0": 0.05, "budget_measure_all": 20,
          "ppo_measure_all_per_eval": 2.0, "search_measure_all_per_eval": 1.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["h1", "h2"])
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--k", type=int, default=PREREG["k"],
                   help="PPO evaluations in stage 1. PRE-REGISTERED at 5.")
    p.add_argument("--r", type=int, default=PREREG["r"],
                   help="refinement evaluations in stage 2. PRE-REGISTERED at 10.")
    p.add_argument("--sigma0", type=float, default=PREREG["sigma0"],
                   help="CMA-ES initial step. PRE-REGISTERED at 0.05.")
    p.add_argument("--oversample", type=int, default=20,
                   help="H2 only: candidates proposed per generation, as a multiple of "
                        "popsize. PRE-REGISTERED at 20.")
    p.add_argument("--tol", type=float, default=1.5, help="strict target tolerance, dB")
    p.add_argument("--spec-seed", type=int, default=2,
                   help="0 and 1 are the historical sets and have both been used in "
                        "model-selection-adjacent decisions. This arm is pre-registered "
                        "on 2 and 3, neither of which has been seen by any decision.")
    p.add_argument("--seed", type=int, default=20260823,
                   help="base optimizer seed; spec i runs at seed + i. Same base the "
                        "search baselines use, so H1/H2/CMA-ES share it.")
    p.add_argument("--corpus", default="results/surrogate_corpus.npz")
    p.add_argument("--allow-protocol-deviation", action="store_true",
                   help="required to run with any pre-registered constant changed. Any "
                        "such run is EXPLORATORY and may not be reported as the result.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    deviations = {n: (getattr(args, n), PREREG[n]) for n in ("k", "r", "sigma0")
                  if getattr(args, n) != PREREG[n]}
    if 2 * args.k + args.r != PREREG["budget_measure_all"]:
        deviations["budget"] = (2 * args.k + args.r, PREREG["budget_measure_all"])
    if deviations and not args.allow_protocol_deviation:
        raise SystemExit(
            "REFUSING TO RUN: this deviates from the pre-registered protocol in "
            + ", ".join(f"{k} = {got} (registered {want})"
                        for k, (got, want) in deviations.items())
            + ".\ndocs/REPRODUCE.md sections 17-18 fix these before any compute ran. If "
              "the deviation is deliberate, pass --allow-protocol-deviation; the result "
              "is then EXPLORATORY and may not be reported as the headline.")

    out = args.out or (f"results/hybrid_audit_{args.arm}"
                       + (f"_seed{args.spec_seed}" if args.spec_seed else "") + ".json")

    specs = make_specs(args.specs, args.spec_seed)
    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        """One verified evaluation. Returns (record_or_None, objective_score).

        Byte-identical in behaviour to `search_audit.evaluate` -- same guard, same
        `hard_pass`, same dense objective -- so the records `summarize` reads below are
        the same records every other arm produced.
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

    # -- the surrogate, for H2 only ------------------------------------------------
    sur = None
    if args.arm == "h2":
        from eqrl.surrogate import Surrogate, load_corpus
        c = load_corpus(args.corpus)
        sur = Surrogate(c["X"], c["Y"], k=5)
        print(f"surrogate: {len(c['X'])} recorded designs from {args.corpus}\n"
              f"  it RANKS candidates and never decides a pass. It is STATIC -- the "
              f"evaluations this run makes are not fed back into it, because online "
              f"updating is a second mechanism and was not pre-registered.", flush=True)

    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    # Identical construction to target_audit's PPO arm, so stage 1 IS that arm's opening.
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        """Exactly k evaluations, NO reset on `terminated` (section 18.4).

        Resetting would re-roll a good design away -- the replay artifact section 8
        measured -- and stopping early on termination would hand the easiest specs the
        most refinement budget. Fixed k keeps 2k + r at exactly 20 on every spec.
        """
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        trace = [evaluate(env._x, target, channel)[0]]
        terminated_at = None
        while len(trace) < args.k:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, _trunc, _ = env.step(a)
            trace.append(evaluate(env._x, target, channel)[0])
            if term and terminated_at is None:
                terminated_at = len(trace)
        return np.array(env._x, dtype=np.float64), trace, terminated_at

    def stage2(x0, target, channel, seed):
        """CMA-ES from the handoff design. H2 differs ONLY in which candidates are SPICEd."""
        import cma
        es = cma.CMAEvolutionStrategy(list(np.clip(x0, 0.0, 1.0)), args.sigma0,
                                      {"bounds": [0, 1], "maxfevals": args.r,
                                       "seed": seed, "verbose": -9})
        trace: list[dict | None] = []
        screened = 0
        generations = 0
        while len(trace) < args.r:
            afford = min(int(es.popsize), args.r - len(trace))
            preds = None
            if args.arm == "h2":
                cand = es.ask(number=int(es.popsize) * args.oversample)
                pred, dist = sur.predict_boost(np.clip(np.array(cand), 0.0, 1.0))
                pick = np.argsort(np.abs(pred - target))[:afford]
                screened += len(cand)
                chosen = [cand[j] for j in pick]
                # Recorded so "did the surrogate aim, and was it right?" is answerable
                # from the artifact instead of by re-running. A screen that picks
                # candidates it predicts are on target, which then measure far off, is a
                # DIFFERENT failure from one where nothing on target was available --
                # and the two need opposite responses.
                preds = [(float(pred[j]), float(dist[j])) for j in pick]
                pool_best = float(np.abs(pred - target).min())
            else:
                cand = es.ask()
                chosen = list(cand[:afford])
                pool_best = None
            costs = []
            for n_i, x in enumerate(chosen):
                rec, score = evaluate(x, target, channel)
                if rec is not None:
                    rec["gen"] = generations
                    if preds is not None:
                        rec["pred_boost_db"] = preds[n_i][0]
                        rec["pred_nn_dist"] = preds[n_i][1]
                        rec["pred_err_db"] = rec["boost_db"] - preds[n_i][0]
                        rec["pool_best_pred_err_db"] = pool_best
                trace.append(rec)
                costs.append(-score)
            generations += 1
            if afford < int(es.popsize):
                break          # budget exhausted mid-generation; nothing left to adapt for
            es.tell(chosen, costs)
        return trace, {"popsize": int(es.popsize), "sigma0": args.sigma0,
                       "generations": generations, "seed": seed,
                       "candidates_screened": screened,
                       "arm": args.arm, "oversample": args.oversample
                       if args.arm == "h2" else None}

    def summarize(trace, target):
        """The same solved_at / best / correlation logic `target_audit.summarize` and
        `search_audit` use, spelled out identically so the arms stay comparable."""
        ok = [e for e in trace if e and e["loose_pass"]]
        loose_at = next((i + 1 for i, e in enumerate(trace) if e and e["loose_pass"]), None)
        strict_at = next((i + 1 for i, e in enumerate(trace)
                          if e and e["loose_pass"]
                          and abs(e["boost_db"] - target) <= args.tol), None)
        best = min(ok, key=lambda e: abs(e["boost_db"] - target)) if ok else None
        return {"loose_solved_at": loose_at, "strict_solved_at": strict_at,
                "n_valid": sum(1 for e in trace if e), "n_loose_pass": len(ok),
                "n_evals": len(trace),
                "n_distinct_boosts": len({round(e["boost_db"], 6) for e in trace if e}),
                "best_boost_db": None if best is None else best["boost_db"],
                "best_abs_err": None if best is None else abs(best["boost_db"] - target),
                "best_design": None if best is None else best["design"],
                "all_valid_boosts": [e["boost_db"] for e in trace if e]}

    rows = []
    print("HYBRID %s | %d specs, spec-seed %d | stage 1 = %d PPO evals, "
          "stage 2 = %d CMA-ES evals, sigma0 %.3f | %d measure_all total, "
          "strict tol +/-%.1f dB\n"
          % (args.arm.upper(), len(specs), args.spec_seed, args.k, args.r, args.sigma0,
             2 * args.k + args.r, args.tol), flush=True)
    for i, (target, channel) in enumerate(specs):
        x0, t1, term_at = stage1(i, target, channel)
        t2, cfg = stage2(x0, target, channel, args.seed + i)
        # The reported trajectory is the WHOLE arm -- both stages -- because both spent
        # the shared budget. Reporting stage 2 alone would hide the 10 measure_all stage 1
        # cost and flatter the arm.
        full = t1 + t2
        res = summarize(full, target)
        res["stage1"] = summarize(t1, target)
        res["stage2"] = summarize(t2, target)
        res["handoff"] = {
            "loose_pass": bool(t1[-1]["loose_pass"]) if t1[-1] else False,
            "boost_db": t1[-1]["boost_db"] if t1[-1] else None,
            "guard_valid": t1[-1] is not None,
            "ppo_terminated_at_eval": term_at,
        }
        res["measure_all_spent"] = (args.k * PREREG["ppo_measure_all_per_eval"]
                                    + len(t2) * PREREG["search_measure_all_per_eval"])
        rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                     "config": cfg, args.arm: res})
        print("  spec %2d  target %5.2f  chan %5.2f | handoff %-5s %6s | "
              "loose %-4s strict %-4s valid %2d/%d  err %s"
              % (i, target, channel,
                 "PASS" if res["handoff"]["loose_pass"] else "-",
                 "%.2f" % res["handoff"]["boost_db"] if res["handoff"]["boost_db"]
                 is not None else "-",
                 res["loose_solved_at"], res["strict_solved_at"],
                 res["n_valid"], res["n_evals"],
                 "-" if res["best_abs_err"] is None else "%.2f" % res["best_abs_err"]),
              flush=True)
        Path(out).write_text(json.dumps(
            {"method": f"hybrid_{args.arm}", "specs": args.specs, "budget": args.k + args.r,
             "measure_all_budget": 2 * args.k + args.r, "k": args.k, "r": args.r,
             "sigma0": args.sigma0, "tol": args.tol, "base_seed": args.seed,
             "spec_seed": args.spec_seed, "model": args.model,
             "protocol_deviation": bool(deviations),
             "complete": len(rows) == len(specs), "rows": rows}, indent=1))

    print("\nwrote", out)


if __name__ == "__main__":
    main()
