"""G3.1: advance toward the target along the direction, and stop at the feasibility wall.

WHAT G3 ESTABLISHED. The 1-D root find solves b(x0 + t*d) = target to a median 0.08 dB in
two to four SPICE evaluations. It also returns zero usable circuits, because every on-target
point it finds fails one check -- peak_in_band, 4 of 4 -- and 10 of its 11 guard-valid
designs fail the hard spec. The direction that raises boost is the same direction that moves
the peak out of band. The solver went exactly where it was sent; it was sent outside the
feasible set.

THE FIX IS NOT TO PUT peak_in_band INTO THE EQUATION. Folding constraints into a scalar
f(x) = |boost - target| + w * penalty is how this project got a reward whose retargeting
correlation was 0.114, and re-deriving one here under a new name would earn the same result.
Feasibility is a GATE, not a term:

    eligible   guard-valid AND hard_pass          (the same loose test every arm uses)
    ranked     among eligible only, min |boost - target|
    rejected   anything not eligible, whatever its boost

That ordering is lexicographic and has no weights to tune. An infeasible point can never
outrank a feasible one no matter how exactly it hits the target.

THE SEARCH THIS IMPLIES. If t = 0 is feasible and the on-target t is not, then somewhere
between them is a feasibility wall, and the best design available along d is the one just
inside it. So instead of solving for the target, G3.1 solves for the WALL: advance while
feasible, bisect when a step lands outside, and keep the best feasible point found. The
target still terminates the search when reached -- it just no longer overrides the gate.
This reuses G3's bracketing machinery on a different quantity, which is why it costs the
same two-to-four evaluations rather than needing a new mechanism.

REPAIR, for the handoffs the solver could not previously start from. 4 of 8 PPO handoffs
were guard-invalid, and a root find has no baseline to bracket from. Two fallbacks, in
order, both free of any new objective:
  1. stage 1 already evaluated k designs. If any is feasible, start from the best of them.
     This costs nothing -- those simulations are already paid for.
  2. otherwise step AGAINST the boost direction. The diagnosis says over-boosting is what
     breaks peak_in_band, so backing boost off is the principled way back into the feasible
     set, not a guess.

Budget parity is exact and unchanged: k = 5 PPO evaluations at 2.00 measure_all, leaving 10
at 1.00 for repair and refinement COMBINED. Repair is not free extra budget.

Every rejected point records WHICH checks it failed, so "is peak_in_band still dominant"
is answered by the artifact rather than by a follow-up run.

Changes no threshold, bound, reward, model, or benchmark criterion.
"""
from __future__ import annotations

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
from silq.specs import DEFAULT_SPEC, hard_pass
from silq.experiments.target_audit import make_specs
from silq.experiments.g3_rootfind import direction_from_probe

DIMS = list(ACTION_SPACE.keys())

PREREG = {
    "k": 5,                    # PPO evals, 2.00 measure_all each -- H1's stage 1, unchanged
    "r": 10,                   # repair + refinement COMBINED, 1.00 each
    "budget_measure_all": 20,
    "t0": 0.05,                # opening step, CMA-ES's sigma0
    "expand": 2.0,
    "t_max": 1.0,
    "min_t": 1e-3,             # below this a step is no longer a move
    "stop_abs_err_db": 0.25,   # inside the 1.5 dB strict criterion
    "repair_max": 3,           # evaluations repair may consume out of r
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--specs", type=int, default=8)
    p.add_argument("--spec-seed", type=int, default=3)
    p.add_argument("--probe", default="results/g3_probe_check.json")
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--out", default="results/g3_constrained_smoke.json")
    args = p.parse_args()

    k, r = PREREG["k"], PREREG["r"]
    d_unit, sign_table = direction_from_probe(args.probe)
    specs = make_specs(32, args.spec_seed)[:args.specs]
    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        """As hybrid_audit.evaluate, plus the per-check dict so refusals are explainable."""
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
        ok, checks = hard_pass(m, spec)
        return {"boost_db": float(m.boost_db), "dc_gain_db": float(m.dc_gain_db),
                "loose_pass": bool(ok),
                "failing": [c for c, good in checks.items() if not good],
                "design": dataclasses.asdict(dv)}

    from stable_baselines3 import PPO
    from silq.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        """Byte-identical to hybrid_audit.stage1 -- k evals, NO reset on terminated."""
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        xs = [np.array(env._x, dtype=np.float64)]
        trace = [evaluate(env._x, target, channel)]
        while len(trace) < k:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, _t, _tr, _ = env.step(a)
            xs.append(np.array(env._x, dtype=np.float64))
            trace.append(evaluate(env._x, target, channel))
        return xs, trace

    def run(xs, t1, target, channel):
        """Repair to a feasible point, then advance to the feasibility wall."""
        trace, steps = [], []
        budget = r
        info = {"steps": steps, "repair_used": 0, "repair_source": None,
                "started_feasible": False, "reached_target": False,
                "wall_hit": False, "reason": None}

        def take(x, phase):
            """One SPICE evaluation. Records feasibility and the checks that refused it."""
            nonlocal budget
            if budget <= 0:
                return None
            budget -= 1
            rec = evaluate(x, target, channel)
            feas = bool(rec and rec["loose_pass"])
            steps.append({"phase": phase, "valid": rec is not None, "feasible": feas,
                          "boost_db": None if rec is None else rec["boost_db"],
                          "abs_err": None if rec is None else abs(rec["boost_db"] - target),
                          "failing": [] if rec is None else rec["failing"]})
            if rec is not None:
                trace.append(rec)
            return rec

        # ---- 1. a feasible starting point -------------------------------------------
        # Free first: stage 1 already paid for k evaluations. The best FEASIBLE one is a
        # legitimate start and costs nothing, which is why this precedes any repair step.
        cand = [(abs(e["boost_db"] - target), j) for j, e in enumerate(t1)
                if e and e["loose_pass"]]
        if cand:
            j = min(cand)[1]
            x_f, b_f = xs[j], t1[j]["boost_db"]
            info["started_feasible"] = True
            info["repair_source"] = "stage1 (free)"
        else:
            # Step AGAINST the boost direction: over-boosting is what the G3 diagnosis
            # showed breaks peak_in_band, so reducing boost is the principled way back in.
            x_f = b_f = None
            base = xs[-1]
            t = PREREG["t0"]
            while info["repair_used"] < PREREG["repair_max"] and budget > 0:
                x_try = np.clip(base - t * d_unit, 0.0, 1.0)
                rec = take(x_try, "repair")
                info["repair_used"] += 1
                if rec is not None and rec["loose_pass"]:
                    x_f, b_f = x_try, rec["boost_db"]
                    info["repair_source"] = "repair step against d (t=%.3f)" % t
                    break
                t *= PREREG["expand"]
            if x_f is None:
                info["reason"] = "no feasible starting point after repair"
                return trace, info

        best_err = abs(b_f - target)
        if best_err <= PREREG["stop_abs_err_db"]:
            info["reached_target"] = True
            info["reason"] = "handoff already on target"
            return trace, info

        # ---- 2. advance, tracking TWO brackets ---------------------------------------
        # There are two different things that can stop the advance and they need separate
        # brackets. A first version tracked only the feasibility wall, so once a feasible
        # step overshot the target the loop read "still feasible, push further" and walked
        # away from the answer -- spec 2 ran 5.02 -> 6.16 -> 8.26 -> 10.10 dB against a
        # 5.56 dB target, every point feasible and every point worse. The target crossing
        # is a bracket in its own right, and when both exist the target bracket binds,
        # because both of its ends are already feasible.
        way = 1.0 if target > b_f else -1.0
        side = np.sign(b_f - target)   # which side of the target we start on
        lo_t, lo_b = 0.0, b_f          # feasible, still on the starting side
        over_t = over_b = None         # feasible, overshot -> target bracket
        wall_t = None                  # smallest t known infeasible
        t = PREREG["t0"]
        while budget > 0:
            x_try = np.clip(x_f + t * way * d_unit, 0.0, 1.0)
            if np.allclose(x_try, x_f) or t < PREREG["min_t"]:
                info["reason"] = "no admissible step remains"
                break
            rec = take(x_try, "advance" if (over_t is None and wall_t is None) else "refine")
            if rec is not None and rec["loose_pass"]:
                b = rec["boost_db"]
                err = abs(b - target)
                best_err = min(best_err, err)
                if err <= PREREG["stop_abs_err_db"]:
                    info["reached_target"] = True
                    info["reason"] = "target reached inside the feasible set"
                    break
                if np.sign(b - target) != side:
                    over_t, over_b = t, b
                else:
                    lo_t, lo_b = t, b
            else:
                info["wall_hit"] = True
                wall_t = t if wall_t is None else min(wall_t, t)

            if over_t is not None:
                # Target bracketed by two FEASIBLE points: secant, bisection on escape.
                if abs(over_b - lo_b) > 1e-9:
                    t = lo_t + (target - lo_b) * (over_t - lo_t) / (over_b - lo_b)
                else:
                    t = 0.5 * (lo_t + over_t)
                if not (min(lo_t, over_t) < t < max(lo_t, over_t)):
                    t = 0.5 * (lo_t + over_t)
                if abs(over_t - lo_t) < PREREG["min_t"]:
                    info["reason"] = "converged inside the feasible set"
                    break
            elif wall_t is not None:
                # No overshoot yet and the wall is in the way: the best design available
                # along d is the one just inside it.
                t = 0.5 * (lo_t + wall_t)
                if abs(wall_t - lo_t) < PREREG["min_t"]:
                    info["reason"] = "converged onto the feasibility wall"
                    break
            else:
                t = min(t * PREREG["expand"], PREREG["t_max"])
        info["reason"] = info["reason"] or "budget exhausted"
        return trace, info

    def summarize(trace, target):
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

    print("G3.1 CONSTRAINED | %d specs, spec-seed %d | stage 1 = %d PPO evals, "
          "stage 2 = %d evals (repair + advance)" % (len(specs), args.spec_seed, k, r),
          flush=True)
    print("  feasibility is a GATE (guard-valid AND hard_pass), never a weighted term.",
          flush=True)
    print("  direction: %s\n" % {n: ("+" if sign_table[n] > 0 else "-") for n in DIMS},
          flush=True)

    rows, fails = [], {}
    for i, (target, channel) in enumerate(specs):
        xs, t1 = stage1(i, target, channel)
        t2, info = run(xs, t1, target, channel)
        full = [e for e in t1 if e] + t2
        s_all = summarize(full, target)
        s_1 = summarize([e for e in t1 if e], target)
        n2 = len(info["steps"])
        for s in info["steps"]:
            for c in s["failing"]:
                fails[c] = fails.get(c, 0) + 1
        h = t1[-1]
        row = {"spec": i, "target_boost_db": target, "channel_loss_db": channel,
               "g31": {**s_all,
                       "n_distinct_boosts": len({round(b, 4)
                                                 for b in s_all["all_valid_boosts"]}),
                       "stage1": s_1, "stage2": summarize(t2, target),
                       "handoff": {"boost_db": None if h is None else h["boost_db"],
                                   "guard_valid": h is not None,
                                   "loose_pass": bool(h and h["loose_pass"])},
                       "solver": info, "n_solver_evals": n2,
                       "measure_all_spent": k * 2.0 + n2}}
        rows.append(row)
        print("  spec %2d  target %5.2f | start %-22s | best %-6s err %-5s %-6s | "
              "%d evals%s"
              % (i, target, info["repair_source"] or "NONE",
                 "-" if s_all["best_boost_db"] is None else "%.2f" % s_all["best_boost_db"],
                 "-" if s_all["best_abs_err"] is None else "%.2f" % s_all["best_abs_err"],
                 "strict" if s_all["strict_solved_at"] else
                 ("loose" if s_all["loose_solved_at"] else "-"), n2,
                 "  WALL" if info["wall_hit"] else ""), flush=True)
        Path(args.out).write_text(json.dumps(
            {"prereg": PREREG, "spec_seed": args.spec_seed, "model": args.model,
             "direction": sign_table, "probe_source": args.probe, "tol": args.tol,
             "failing_checks_seen": fails, "complete": False, "rows": rows}, indent=1))

    Path(args.out).write_text(json.dumps(
        {"prereg": PREREG, "spec_seed": args.spec_seed, "model": args.model,
         "direction": sign_table, "probe_source": args.probe, "tol": args.tol,
         "failing_checks_seen": fails, "complete": True, "rows": rows}, indent=1))
    print("\nfailing checks across all stage-2 points: %s" % fails, flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
