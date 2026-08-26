"""G3.2: rescue the handoff into the feasible set, then solve the target inside it.

WHAT G3.1 LEFT. Where it could start, it was exact -- 0.01 and 0.07 dB, on two specs H1
missed strictly, in 3 and 4 evaluations against H1's 10. It could not start on 5 of 8. That
is a STARTING-POINT problem, not a search problem, and it has two distinct causes which
this file keeps distinct rather than pouring into one objective:

  Case 1  the handoff is guard-INVALID (3 of 8 seed-3 specs had 0 of 5 valid stage-1
          designs). No boost, no peak, nothing to difference or bracket.
  Case 2  the handoff is guard-valid but fails a hard check, and the single direction G3.1
          owned moved boost and peak together, so repairing one broke the other. 0 of 5.

BOTH BRANCHES ARE MEASURED BEFORE THEY ARE BUILT, on spec-seed 2, disjoint from the seed-3
specs tested here.

  results/g32_peak_probe.json  ->  the plane. rs moves boost at 12.45 dB/unit with peak
      movement under the 5.9% measurement resolution on 3 of 4 specs; l_in moves the peak
      at -3.14 GHz/unit for -5.29 dB/unit of boost. Those are the two coordinates, and they
      are nearly decoupled in exactly the way G3.1 needed and did not have.
  results/g32_rescue_probe.json  ->  the Case-1 ladder. 4 of 4 guard-invalid handoffs were
      returned to guard-valid by a SINGLE-axis move, 2 of them straight to hard_pass.

THE ONE MECHANISM CHANGE, stated plainly. G3 and G3.1 used a six-dimensional composite
d_boost. G3.2 uses rs alone as the boost coordinate. The composite is precisely what walked
the peak out of band on 4 of 4 bracketed specs in G3; the probe says rs moves boost harder
than the composite while leaving the peak inside the measurement floor. This is the change
the measurement asked for, and it is the only one.

NO SCALAR REWARD, ANYWHERE. Each coordinate answers to one quantity:

    t2 (l_in)   set ONLY by the peak constraint -- aim at the geometric centre of the band
    t1 (rs)     set ONLY by boost -- aim at the requested target

There is nothing to weight against anything, because the two are never compared. The
target is a legitimate aim point for t1 during rescue rather than a competing objective:
boost_range is 3-12 dB and targets are drawn from 5-11 dB, so the requested target is
ALWAYS strictly inside the boost constraint. Satisfying the constraint and hitting the
target are the same instruction, and no trade-off exists to be tuned.

The peak is aimed at the band CENTRE (geometric, 1.77 GHz, since the grid is logarithmic)
and never at an edge. With a 5.9% quantum and a band 12 steps wide, aiming at an edge is a
coin flip; aiming at the centre leaves six steps of margin on both sides.

Feasibility stays a lexicographic GATE, as in G3.1: eligible = guard-valid AND hard_pass;
ranked among eligible only by |boost - target|; an infeasible point never outranks a
feasible one however exactly it hits the target.

BUDGET PARITY IS UNCHANGED. k = 5 PPO evaluations at 2.00 measure_all, leaving 10 at 1.00
for rescue, repair and refinement COMBINED. Rescue is not free extra budget.

G3.1's advance loop is reimplemented here rather than imported, so that the committed
results/g3_constrained_smoke.json stays reproducible from unmodified code.

Changes no threshold, bound, reward, model, PPO configuration, or benchmark criterion.
"""
from __future__ import annotations

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
from eqrl.specs import DEFAULT_SPEC, hard_pass
from eqrl.experiments.target_audit import make_specs
from eqrl.experiments.g32_peak_report import plane_from_probe
from eqrl.experiments.g32_rescue_probe import rescue_order_from_probe

DIMS = list(ACTION_SPACE.keys())

PREREG = {
    "k": 5,                    # PPO evals, 2.00 measure_all each -- H1's stage 1, unchanged
    "r": 10,                   # rescue + repair + refine COMBINED, 1.00 each
    "budget_measure_all": 20,
    "rescue_max": 3,           # Case-1 ladder ceiling, out of r
    "repair_max": 3,           # Case-2 2-D steps ceiling, out of r
    "step_cap": 0.35,          # no single computed step may exceed this in either coord
    "t0": 0.05, "expand": 2.0, "t_max": 1.0, "min_t": 1e-3,
    "stop_abs_err_db": 0.25,   # well inside the 1.5 dB strict criterion
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--spec-seed", type=int, default=3)
    p.add_argument("--first", type=int, default=8, help="first spec index (8 = fresh)")
    p.add_argument("--specs", type=int, default=10)
    p.add_argument("--peak-probe", default="results/g32_peak_probe.json")
    p.add_argument("--rescue-probe", default="results/g32_rescue_probe.json")
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--out", default="results/g32_repair_smoke.json")
    args = p.parse_args()

    k, r = PREREG["k"], PREREG["r"]
    plane = plane_from_probe(args.peak_probe)
    ladder = rescue_order_from_probe(args.rescue_probe)
    jb = DIMS.index(plane["boost_axis"])
    jp = DIMS.index(plane["peak_axis"])
    S_BOOST = plane["d_boost_db_per_unit"]              # dB per unit of t1
    S_PEAK = plane["d_peak_ghz_per_unit"]               # GHz per unit of t2
    S_PEAK_BOOST = plane["peak_axis_d_boost_db_per_unit"]   # t2's boost side-effect
    LO, HI = plane["band_ghz"]
    AIM = plane["peak_aim_ghz"]
    B_LO, B_HI = DEFAULT_SPEC.boost_db_min, DEFAULT_SPEC.boost_db_max

    all_specs = make_specs(32, args.spec_seed)
    specs = list(enumerate(all_specs))[args.first:args.first + args.specs]
    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        """SPICE -> guard -> peak -> boost -> hard constraints, in that order.

        Returns None ONLY on guard rejection, matching every other arm's convention, but
        the caller needs the guard's Check to steer Case 1, so that is returned separately.
        """
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception as e:
            return None, "exception:%s" % repr(e)[:80]
        if not v.is_valid:
            return None, str(getattr(v.check, "value", v.check))
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        return {"boost_db": float(m.boost_db), "peak_freq_ghz": float(m.peak_freq_ghz),
                "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                "failing": [c for c, good in checks.items() if not good],
                "design": dataclasses.asdict(dv)}, None

    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        """Byte-identical to hybrid_audit.stage1 -- k evals, NO reset on terminated."""
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        xs = [np.array(env._x, dtype=np.float64)]
        trace = [evaluate(env._x, target, channel)[0]]
        while len(trace) < k:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, _t, _tr, _ = env.step(a)
            xs.append(np.array(env._x, dtype=np.float64))
            trace.append(evaluate(env._x, target, channel)[0])
        return xs, trace

    def run(xs, s1trace, target, channel):
        trace, steps = [], []
        budget = r
        info = {"steps": steps, "case": None, "rescue_used": 0, "repair_used": 0,
                "rescue_entry": None, "started_feasible": False, "start_source": None,
                "reached_target": False, "wall_hit": False, "reason": None,
                "blocked_by": None, "measured_gain": None}

        def take(x, phase):
            """One SPICE evaluation, recorded with the constraint verdict that produced it."""
            nonlocal budget
            if budget <= 0:
                return None
            budget -= 1
            rec, gcheck = evaluate(x, target, channel)
            steps.append({"phase": phase, "valid": rec is not None,
                          "feasible": bool(rec and rec["loose_pass"]),
                          "guard_check": gcheck,
                          "boost_db": None if rec is None else rec["boost_db"],
                          "peak_freq_ghz": None if rec is None else rec["peak_freq_ghz"],
                          "abs_err": None if rec is None else abs(rec["boost_db"] - target),
                          "failing": [] if rec is None else rec["failing"]})
            if rec is not None:
                trace.append(rec)
            return rec

        # ---- 0. is there already a feasible point? -----------------------------------
        # Free first: stage 1 already paid for k evaluations.
        cand = [(abs(e["boost_db"] - target), j) for j, e in enumerate(s1trace)
                if e and e["loose_pass"]]
        if cand:
            j = min(cand)[1]
            x_f, rec_f = xs[j], s1trace[j]
            info.update(case="already feasible", started_feasible=True,
                        start_source="stage1 (free)")
        else:
            # ---- STAGE A -------------------------------------------------------------
            valid = [(len(e["failing"]), abs(e["boost_db"] - target), j)
                     for j, e in enumerate(s1trace) if e]
            x_f = rec_f = None

            if not valid:
                # CASE 1: nothing guard-valid anywhere in stage 1. Walk the measured
                # rescue ladder from the handoff. Each rung is one axis, one sign, one
                # magnitude, ordered by how many seed-2 handoffs it rescued.
                info["case"] = "case 1: guard-invalid handoff"
                base = xs[-1]
                for entry in ladder:
                    if info["rescue_used"] >= PREREG["rescue_max"] or budget <= 0:
                        break
                    e = np.zeros(len(DIMS)); e[DIMS.index(entry["axis"])] = 1.0
                    x_try = np.clip(base + entry["sign"] * entry["mag"] * e, 0.0, 1.0)
                    if np.allclose(x_try, base):
                        continue        # clipped at a bound: not a move, not an attempt
                    rec = take(x_try, "rescue")
                    info["rescue_used"] += 1
                    if rec is not None:
                        x_f, rec_f = x_try, rec
                        info["rescue_entry"] = "%s %+g x %.2f" % (
                            entry["axis"], entry["sign"], entry["mag"])
                        info["start_source"] = "rescue ladder (%s)" % info["rescue_entry"]
                        break
                if rec_f is None:
                    info["reason"] = "rescue ladder did not restore guard validity"
                    info["blocked_by"] = "guard: " + ",".join(
                        sorted({s["guard_check"] for s in steps if s["guard_check"]}))
                    return trace, info
            else:
                # CASE 2: guard-valid but failing a hard check. Fewest failing checks
                # first, then closest in boost -- no weighting between the two, the check
                # count is a strict outer key.
                j = min(valid)[2]
                x_f, rec_f = xs[j], s1trace[j]
                info["case"] = "case 2: valid handoff, failing " + ",".join(rec_f["failing"])
                info["start_source"] = "stage1 (free)"

            # ---- 2-D constrained repair, if the start is still not feasible -----------
            # Each coordinate serves ONE quantity: t2 the peak constraint, aimed at the
            # band centre; t1 boost, aimed at the requested target, which is always inside
            # boost_range so there is nothing to weigh against anything.
            #
            # THE PEAK STEP IS SIZED BY MEASUREMENT, NOT BY THE TABLE. The calibrated
            # slope is good enough to pick the coordinate and its sign and is NOT good
            # enough to size a step: on spec 16 a step the table said would land near
            # 1.77 GHz landed at 0.714, an implied local gain of -4.88 against the table's
            # -1.15. So the table is used only for the FIRST step, and from the second
            # onward the gain is the secant through the two points actually measured.
            # This is the bracketed search the design called for; the first version was a
            # Newton step with a constant Jacobian, which cannot converge when the
            # Jacobian is four times wrong.
            #
            # The peak is stepped in LOG space. The AC sweep is `dec 40`, the band is a
            # ratio, and peak_freq_ghz only ever takes values on that logarithmic grid, so
            # ln(peak) is the coordinate the quantity actually lives in. Across the
            # calibration specs the log slope is -1.153 on 3 of 4; the linear slope ranges
            # over -2.19 to -4.03 for the same designs.
            cap = PREREG["step_cap"]
            gain = plane["d_ln_peak_per_unit"]   # table, first step only
            back = 1.0                           # backoff after leaving the guard-valid set
            while rec_f is not None and not rec_f["loose_pass"]:
                if info["repair_used"] >= PREREG["repair_max"] or budget <= 0:
                    break
                pk, bo = rec_f["peak_freq_ghz"], rec_f["boost_db"]
                t2 = 0.0 if LO <= pk <= HI else float(np.log(AIM / pk) / gain)
                t2 = float(np.clip(t2, -cap, cap)) * back
                bo_after = bo + S_PEAK_BOOST * t2
                t1 = float(np.clip((target - bo_after) / S_BOOST, -cap, cap))
                x_try = np.array(x_f, dtype=np.float64)
                x_try[jp] = np.clip(x_try[jp] + t2, 0.0, 1.0)
                x_try[jb] = np.clip(x_try[jb] + t1, 0.0, 1.0)
                if np.allclose(x_try, x_f):
                    info["reason"] = "2-D repair step clipped to nothing at a bound"
                    break
                rec = take(x_try, "repair2d")
                info["repair_used"] += 1
                if rec is None:
                    # The step left the guard-valid set. That is a wall, not a dead end --
                    # halve and try again from the SAME point, exactly as the advance loop
                    # bisects onto a wall. The first version gave up here and returned with
                    # 8 of its 10 evaluations unspent on specs 9 and 16.
                    back *= 0.5
                    info["blocked_by"] = "guard: " + (steps[-1]["guard_check"] or "?")
                    if back < 0.125:
                        info["reason"] = "2-D repair walled in by the guard"
                        break
                    continue
                back = 1.0
                moved = x_try[jp] - x_f[jp]
                if abs(moved) > PREREG["min_t"] and rec["peak_freq_ghz"] > 0 and pk > 0:
                    # The gain actually observed on this design, replacing the table.
                    gain = float(np.log(rec["peak_freq_ghz"] / pk) / moved)
                    info["measured_gain"] = gain
                x_f, rec_f = x_try, rec

            if rec_f is None or not rec_f["loose_pass"]:
                info["reason"] = info["reason"] or "2-D repair did not reach feasibility"
                if rec_f is not None:
                    info["blocked_by"] = "hard_pass: " + ",".join(rec_f["failing"])
                return trace, info
            info["started_feasible"] = True

        # ---- STAGE B: precision targeting inside the feasible set --------------------
        # G3.1's two-bracket advance, along rs alone. The target bracket binds when both
        # exist, because both of its ends are already feasible.
        b_f = rec_f["boost_db"]
        best_err = abs(b_f - target)
        if best_err <= PREREG["stop_abs_err_db"]:
            info["reached_target"] = True
            info["reason"] = "start already on target"
            return trace, info

        way = 1.0 if (target - b_f) * S_BOOST > 0 else -1.0
        side = np.sign(b_f - target)
        lo_t, lo_b = 0.0, b_f
        over_t = over_b = None
        wall_t = None
        t = PREREG["t0"]
        while budget > 0:
            x_try = np.array(x_f, dtype=np.float64)
            x_try[jb] = np.clip(x_try[jb] + t * way, 0.0, 1.0)
            if np.allclose(x_try, x_f) or t < PREREG["min_t"]:
                info["reason"] = "no admissible step remains"
                break
            rec = take(x_try, "advance" if (over_t is None and wall_t is None) else "refine")
            if rec is not None and rec["loose_pass"]:
                b = rec["boost_db"]
                best_err = min(best_err, abs(b - target))
                if abs(b - target) <= PREREG["stop_abs_err_db"]:
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
                if rec is not None:
                    info["blocked_by"] = "hard_pass: " + ",".join(rec["failing"])
                else:
                    info["blocked_by"] = "guard: " + (steps[-1]["guard_check"] or "?")

            if over_t is not None:
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

    print("G3.2 REPAIR | specs %d-%d of spec-seed %d (fresh) | stage 1 = %d PPO evals, "
          "stage 2 = %d" % (args.first, args.first + len(specs) - 1, args.spec_seed, k, r),
          flush=True)
    print("  plane from %s: boost=%s %.2f dB/unit (peak under resolution on %s), "
          "peak=%s %+.2f GHz/unit"
          % (plane["source"], plane["boost_axis"], S_BOOST,
             plane["boost_axis_peak_under_resolution_on"], plane["peak_axis"], S_PEAK), flush=True)
    print("  peak aimed at the band CENTRE %.3f GHz (quantum %.1f%%), never an edge"
          % (AIM, plane["grid_step_pct"]), flush=True)
    print("  rescue ladder from %s: %s" % (
        args.rescue_probe,
        " > ".join("%s%+g@%.2f" % (e["axis"], e["sign"], e["mag"]) for e in ladder)),
        flush=True)
    print("  boost_range %.1f-%.1f dB contains every target, so t1 has no trade-off to "
          "weigh\n" % (B_LO, B_HI), flush=True)

    rows, fails, blocked = [], {}, {}
    for i, (target, channel) in specs:
        xs, s1trace = stage1(i, target, channel)
        t2r, info = run(xs, s1trace, target, channel)
        full = [e for e in s1trace if e] + t2r
        s_all = summarize(full, target)
        n2 = len(info["steps"])
        for s in info["steps"]:
            for c in s["failing"]:
                fails[c] = fails.get(c, 0) + 1
        if info["blocked_by"]:
            blocked[info["blocked_by"]] = blocked.get(info["blocked_by"], 0) + 1
        h = s1trace[-1]
        rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                     "g32": {**s_all,
                             "n_distinct_boosts": len({round(b, 4)
                                                       for b in s_all["all_valid_boosts"]}),
                             "stage1": summarize([e for e in s1trace if e], target),
                             "stage2": summarize(t2r, target),
                             "handoff": {"boost_db": None if h is None else h["boost_db"],
                                         "guard_valid": h is not None,
                                         "loose_pass": bool(h and h["loose_pass"])},
                             "solver": info, "n_solver_evals": n2,
                             "measure_all_spent": k * 2.0 + n2}})
        print("  spec %2d  target %5.2f | %-34s | best %-6s err %-5s %-6s | %d evals%s"
              % (i, target, (info["case"] or "?")[:34],
                 "-" if s_all["best_boost_db"] is None else "%.2f" % s_all["best_boost_db"],
                 "-" if s_all["best_abs_err"] is None else "%.2f" % s_all["best_abs_err"],
                 "strict" if s_all["strict_solved_at"] else
                 ("loose" if s_all["loose_solved_at"] else "-"), n2,
                 "  blocked: %s" % info["blocked_by"] if info["blocked_by"] else ""),
              flush=True)
        Path(args.out).write_text(json.dumps(
            {"prereg": PREREG, "plane": plane, "ladder": ladder,
             "spec_seed": args.spec_seed, "first": args.first, "model": args.model,
             "tol": args.tol, "failing_checks_seen": fails, "blocked_by_seen": blocked,
             "complete": False, "rows": rows}, indent=1))

    Path(args.out).write_text(json.dumps(
        {"prereg": PREREG, "plane": plane, "ladder": ladder, "spec_seed": args.spec_seed,
         "first": args.first, "model": args.model, "tol": args.tol,
         "failing_checks_seen": fails, "blocked_by_seen": blocked, "complete": True,
         "rows": rows}, indent=1))
    print("\nfailing checks across all stage-2 points: %s" % fails, flush=True)
    print("what blocked the specs that did not finish: %s" % blocked, flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
