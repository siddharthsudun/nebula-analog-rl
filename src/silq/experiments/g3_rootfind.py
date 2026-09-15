"""G3: solve b(x0 + t*d) = target along a known direction, instead of searching around x0.

THE ARGUMENT. H1 gave CMA-ES ten evaluations to wander the neighbourhood of a PPO handoff
and it landed on its matched chance line: more distinct feasible designs, more accidental
target hits, no aim. The G3 probe experiment then showed that the thing we assumed was
missing -- knowledge of which way boost moves -- is not missing at all. Five of six action
coordinates have a FIXED slope sign across every design probed, and a constant sign table
predicts the direction of a real SPICE step as well as a 31-simulation-per-spec finite
difference does (95.7% against 97.8%, p = 0.232). So direction is free.

What nothing in the pipeline does is solve for the target VALUE. That is a
one-dimensional root find:

    find t such that  b(x0 + t*d) = target

and every evaluation it spends has a specific purpose, which is what separates it from
CMA-ES sampling. If the Strategy A story is right, this should show up as the signature the
matched chance line was built to detect: the SAME or MORE strict solves from FEWER distinct
designs. Coverage turning into aim.

MONOTONICITY IS NOT ASSUMED, and this is the part that has to be defensive. The probe
established local, single-axis behaviour on 11 designs. It says nothing about whether
b(x0 + t*d) stays monotone over a long step, nothing about cross terms between coordinates
moved together, and nothing about designs whose handoff differs from those 11. So the
direction is an initial HYPOTHESIS that every spec re-tests:

    * the first step must move boost TOWARD the target, or the hypothesis is rejected for
      this design and the controller stops rather than pushing on
    * a bracket must actually be observed -- two real SPICE points straddling the target --
      before any interpolation is allowed
    * if no bracket appears within budget, the best VALID point measured is returned and
      the spec is recorded as unbracketed. It is not forced.

BUDGET PARITY IS EXACT. Stage 1 is H1's stage 1, unchanged: k = 5 PPO evaluations at 2.00
measure_all each. That leaves 10 measure_all for the solver at 1.00 each -- precisely the r
= 10 CMA-ES got. Same handoff, same budget, different second stage, so the comparison
isolates the mechanism.

Changes no threshold, bound, reward, model, or benchmark criterion. H1 and the surrogate
are untouched; this is a third arm beside them, not a replacement for either.
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

DIMS = list(ACTION_SPACE.keys())

# Declared before the run and not tuned afterwards.
PREREG = {
    "k": 5,                    # PPO evaluations, 2.00 measure_all each -- H1's stage 1
    "r": 10,                   # solver evaluations, 1.00 each -- H1's stage 2 budget
    "budget_measure_all": 20,
    "t0": 0.05,                # first step, the sigma0 CMA-ES used, so the opening move
                               # is the same size H1's was
    "expand": 2.0,             # geometric bracket expansion
    "t_max": 1.0,              # never step further than the unit box can carry
    "stop_abs_err_db": 0.25,   # stop early well inside the 1.5 dB strict criterion;
                               # spending the remaining budget could only add designs
    "invalid_backoff": 0.5,    # halve the step when a point is guard-rejected
}


def direction_from_probe(path: str) -> tuple[np.ndarray, dict]:
    """The boost-INCREASING unit direction, taken from the committed probe artifact.

    Majority sign of d(boost)/dx_i over the influential observations. Using the artifact
    rather than a literal in the source keeps the provenance of this vector explicit: it
    is a measurement, and if the measurement is ever redone the direction moves with it.
    The magnitudes are deliberately NOT used. Weighting coordinates by their measured
    slope would be a second mechanism, tuned on the same 11 designs, and the point of this
    experiment is to test root finding, not a hand-fitted gradient.
    """
    d = json.loads(Path(path).read_text())
    thr = d["influential_db"]
    h = str(d["probe_h"][-1])
    sign = {}
    for name in DIMS:
        s = []
        for r in d["rows"]:
            if not r["handoff_valid"]:
                continue
            pr = r["dims"][name]["probes"][h]
            if pr["slope_db_per_unit"] is None or abs(pr["pred_delta_db"]) < thr:
                continue
            s.append(np.sign(pr["slope_db_per_unit"]))
        sign[name] = 1.0 if (s and sum(1 for v in s if v > 0) >= len(s) / 2) else -1.0
    v = np.array([sign[n] for n in DIMS], dtype=np.float64)
    return v / np.linalg.norm(v), sign


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--specs", type=int, default=8)
    p.add_argument("--spec-seed", type=int, default=3)
    p.add_argument("--probe", default="results/g3_probe_check.json")
    p.add_argument("--tol", type=float, default=1.5, help="strict tolerance, dB")
    p.add_argument("--out", default="results/g3_rootfind_smoke.json")
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
        """Identical to hybrid_audit.evaluate, so these records are the same records."""
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
        ok, _ = hard_pass(m, spec)
        return {"boost_db": float(m.boost_db), "dc_gain_db": float(m.dc_gain_db),
                "loose_pass": bool(ok), "design": dataclasses.asdict(dv)}

    from stable_baselines3 import PPO
    from silq.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        """Byte-identical to hybrid_audit.stage1 -- k evals, NO reset on terminated."""
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        trace = [evaluate(env._x, target, channel)]
        while len(trace) < k:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, _t, _tr, _ = env.step(a)
            trace.append(evaluate(env._x, target, channel))
        return np.array(env._x, dtype=np.float64), trace

    def solve(x0, b0, target, channel):
        """Bracket, then secant-with-bisection-fallback. Returns (trace, info).

        Every accepted point is a real SPICE evaluation. `t` is a displacement along the
        unit direction, clipped into the box; `eff` records the displacement that actually
        survived clipping, which is what the interpolation must use when a coordinate
        saturates.
        """
        trace, steps = [], []
        budget = r
        info = {"direction_rejected": False, "bracketed": False,
                "reason": None, "n_invalid": 0}
        # Bound to the SAME list object `probe` appends to, and bound here rather than at
        # the end, so that every early return still reports the budget it actually spent.
        # Setting it only on the success path would have made a rejected direction look
        # free when it costs a real simulation.
        info["steps"] = steps
        if b0 is None:
            info["reason"] = "handoff invalid: no baseline boost to solve from"
            return trace, info

        # Which way along d moves boost toward the target.
        way = 1.0 if target > b0 else -1.0
        pts = [(0.0, b0)]                       # (t, boost) -- t = 0 is the handoff, free

        def probe(t):
            nonlocal budget
            if budget <= 0:
                return None
            x = np.clip(x0 + t * way * d_unit, 0.0, 1.0)
            if np.allclose(x, x0):
                return None                     # fully clipped: no move left in this way
            budget -= 1
            rec = evaluate(x, target, channel)
            steps.append({"t": float(t), "valid": rec is not None,
                          "boost_db": None if rec is None else rec["boost_db"],
                          "abs_err": None if rec is None else abs(rec["boost_db"] - target),
                          "phase": phase})
            if rec is None:
                info["n_invalid"] += 1
                return None
            trace.append(rec)
            pts.append((float(t), rec["boost_db"]))
            return rec["boost_db"]

        # --- 1. first step: does the direction hypothesis hold for THIS design? -------
        phase = "hypothesis"
        t = PREREG["t0"]
        b = None
        while budget > 0 and b is None and t <= PREREG["t_max"]:
            b = probe(t)
            if b is None:
                t *= PREREG["invalid_backoff"]      # guard-rejected: try a smaller move
                if t < 1e-3:
                    break
        if b is None:
            info["reason"] = "no valid point along the direction"
            return trace, info
        if abs(b - target) >= abs(b0 - target):
            # Moved away from the target, or not toward it. The probe's direction was a
            # hypothesis and this design rejects it; pushing on would be assuming the
            # monotonicity we explicitly refused to assume.
            info["direction_rejected"] = True
            info["reason"] = "first step did not reduce target error"
            return trace, info

        # --- 2. expand until the target is bracketed ---------------------------------
        phase = "expand"
        lo_t, lo_b = 0.0, b0
        hi_t, hi_b = t, b
        while budget > 0 and (hi_b - target) * (b0 - target) > 0 and hi_t < PREREG["t_max"]:
            lo_t, lo_b = hi_t, hi_b
            hi_t = min(hi_t * PREREG["expand"], PREREG["t_max"])
            nb = probe(hi_t)
            if nb is None:
                hi_t = lo_t                     # invalid: cannot expand further
                break
            hi_b = nb
            if abs(hi_b - target) <= PREREG["stop_abs_err_db"]:
                break
        if (hi_b - target) * (b0 - target) > 0:
            info["reason"] = ("target not bracketed within budget/box; best valid point "
                              "returned")
            return trace, info
        info["bracketed"] = True

        # --- 3. secant, with bisection whenever secant leaves the bracket -------------
        phase = "refine"
        while budget > 0 and abs(hi_b - target) > PREREG["stop_abs_err_db"] \
                and abs(lo_b - target) > PREREG["stop_abs_err_db"]:
            if abs(hi_b - lo_b) > 1e-9:
                t_new = lo_t + (target - lo_b) * (hi_t - lo_t) / (hi_b - lo_b)
            else:
                t_new = 0.5 * (lo_t + hi_t)
            if not (min(lo_t, hi_t) < t_new < max(lo_t, hi_t)):
                t_new = 0.5 * (lo_t + hi_t)     # secant escaped: fall back to bisection
            nb = probe(t_new)
            if nb is None:
                t_new = 0.5 * (lo_t + hi_t)
                nb = probe(t_new)
                if nb is None:
                    info["reason"] = "refinement hit guard-invalid points"
                    break
            if (nb - target) * (lo_b - target) > 0:
                lo_t, lo_b = t_new, nb
            else:
                hi_t, hi_b = t_new, nb
        info["reason"] = info["reason"] or "converged or budget exhausted"
        info["steps"] = steps
        return trace, info

    def summarize(trace, target):
        ok = [e for e in trace if e and e["loose_pass"]]
        loose_at = next((i + 1 for i, e in enumerate(trace) if e and e["loose_pass"]), None)
        strict_at = next((i + 1 for i, e in enumerate(trace)
                          if e and e["loose_pass"] and abs(e["boost_db"] - target) <= args.tol),
                         None)
        best = min(ok, key=lambda e: abs(e["boost_db"] - target)) if ok else None
        return {"loose_solved_at": loose_at, "strict_solved_at": strict_at,
                "n_valid": sum(1 for e in trace if e), "n_loose_pass": len(ok),
                "best_boost_db": None if best is None else best["boost_db"],
                "best_abs_err": None if best is None else abs(best["boost_db"] - target),
                "best_design": None if best is None else best["design"],
                "all_valid_boosts": [e["boost_db"] for e in trace if e]}

    print("G3 ROOT-FIND | %d specs, spec-seed %d | stage 1 = %d PPO evals, "
          "stage 2 = up to %d solver evals | %d measure_all total"
          % (len(specs), args.spec_seed, k, r, PREREG["budget_measure_all"]), flush=True)
    print("  direction (boost-increasing, from %s): %s"
          % (args.probe, {n: ("+" if sign_table[n] > 0 else "-") for n in DIMS}), flush=True)
    print("  monotonicity is NOT assumed: the first step must reduce target error and a "
          "bracket must be observed.\n", flush=True)

    rows = []
    for i, (target, channel) in enumerate(specs):
        x0, t1 = stage1(i, target, channel)
        h = t1[-1]
        b0 = None if h is None else h["boost_db"]
        t2, info = solve(x0, b0, target, channel)

        full = [e for e in t1 if e] + t2
        s_all, s_1, s_2 = (summarize(full, target), summarize([e for e in t1 if e], target),
                           summarize(t2, target))
        n_solver = len(info.get("steps", []))
        e1 = s_1["best_abs_err"]
        ef = s_all["best_abs_err"]
        row = {"spec": i, "target_boost_db": target, "channel_loss_db": channel,
               "g3": {**s_all,
                      "n_distinct_boosts": len({round(b, 4) for b in s_all["all_valid_boosts"]}),
                      "stage1": s_1, "stage2": s_2,
                      "handoff": {"boost_db": b0, "guard_valid": h is not None,
                                  "loose_pass": bool(h and h["loose_pass"])},
                      "solver": info,
                      "n_solver_evals": n_solver,
                      "efficiency_db_per_eval": (
                          None if (e1 is None or ef is None or not n_solver)
                          else (e1 - ef) / n_solver),
                      "measure_all_spent": k * 2.0 + n_solver}}
        rows.append(row)
        print("  spec %2d  target %5.2f | handoff %-6s | %-22s | best %-6s err %-5s %-6s "
              "| %d solver evals"
              % (i, target, "-" if b0 is None else "%.2f" % b0,
                 ("REJECTED " + ("" if not info["direction_rejected"] else "dir"))
                 if not info["bracketed"] else "bracketed",
                 "-" if s_all["best_boost_db"] is None else "%.2f" % s_all["best_boost_db"],
                 "-" if ef is None else "%.2f" % ef,
                 "strict" if s_all["strict_solved_at"] else
                 ("loose" if s_all["loose_solved_at"] else "-"), n_solver), flush=True)
        Path(args.out).write_text(json.dumps(
            {"prereg": PREREG, "spec_seed": args.spec_seed, "model": args.model,
             "direction": {n: sign_table[n] for n in DIMS}, "probe_source": args.probe,
             "tol": args.tol, "complete": False, "rows": rows}, indent=1))

    Path(args.out).write_text(json.dumps(
        {"prereg": PREREG, "spec_seed": args.spec_seed, "model": args.model,
         "direction": {n: sign_table[n] for n in DIMS}, "probe_source": args.probe,
         "tol": args.tol, "complete": True, "rows": rows}, indent=1))
    print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
