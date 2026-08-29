"""The final three-arm comparison, under one fixed 20-measure_all budget per spec.

    A   PPO -> H1 CMA-ES                        the standalone precision baseline
    B   PPO -> G3.2 constrained repair          the standalone constrained controller
    C   PPO -> G3.2 -> H1 on the REMAINDER      the system under test

Pre-registered in docs/PREREG_FINAL_COMPARISON.md, with amendment 1 fixing the reporting
additions, arm C's start rule, and the gate below -- all before any spec of seed 23 ran.

WHY ONE PROCESS AND NOT THREE SCRIPT INVOCATIONS. Arm C spends from the SAME pool arm B
does: if G3.2 stops after 4 evaluations, the fallback gets 6 and not 10. That is a within-
spec handoff of an unspent budget, so it cannot be assembled by running two scripts back to
back. Both solvers live inside `main()` in their own files and cannot be imported, so they
are TRANSCRIBED here -- for exactly the reason g32_repair.py gives for re-implementing
G3.1's advance loop rather than importing it: results/g32_repair_smoke.json and
results/hybrid_audit_h1_seed3.json must stay reproducible from UNMODIFIED committed code.
No frozen file is edited by this experiment.

TRANSCRIPTION IS TESTED, NOT ASSERTED. `--gate` re-runs arms A and B on spec-seed 3 specs
8-17 -- the burned development slice, never seed 23 -- and diffs every outcome field against
the two frozen artifacts. Both controllers are deterministic given the spec index, so a
faithful transcription reproduces them exactly and a drifted one cannot. The gate must pass
before the held-out set is touched.

THREE THINGS ARE COMPUTED ONCE AND SHARED, because sharing them is an identity and not an
approximation:

  stage 1   deterministic given (spec index, target, channel) -- one PPO rollout serves all
            three arms. Each arm is still CHARGED the full 10.00 measure_all it would have
            paid alone; only the duplicate simulation is skipped, never the cost.
  G3.2      deterministic, no RNG anywhere in it -- so arm C's first leg IS arm B's run,
            bit for bit. Reusing it is what makes "C minus B = the fallback" true by
            construction rather than by assumption.
  guards    one evaluator per channel loss, as every other arm builds them.

Changes no reward, PPO hyperparameter, design bound, guard, hard_pass, controller constant,
frozen model or benchmark criterion. Adds an arm; modifies nothing.
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
from eqrl.experiments import g32_repair as _g32
from eqrl.experiments import hybrid_audit as _h1

DIMS = list(ACTION_SPACE.keys())

#: hybrid_audit.REJECT_SCORE, the constant search_audit and honest_benchmark also give a
#: rejected candidate, so CMA-ES sees the landscape shape every other search arm saw.
REJECT_SCORE = -10.0

#: Every constant this file uses, named here so that a drift from either frozen controller
#: is a loud import-time failure rather than a silent difference in the benchmark. Nothing
#: here is a new choice: each value is checked against the module it came from below.
PREREG = {
    "k": 5, "r": 10, "budget_measure_all": 20,
    "ppo_measure_all_per_eval": 2.0, "search_measure_all_per_eval": 1.0,
    "sigma0": 0.05, "base_seed": 20260823,        # H1's, so the fallback is H1's optimizer
    "rescue_max": 3, "repair_max": 3, "step_cap": 0.35,
    "t0": 0.05, "expand": 2.0, "t_max": 1.0, "min_t": 1e-3,
    "stop_abs_err_db": 0.25,
    "tol": 1.5,                                    # the project's strict criterion
}


def _check_constants() -> None:
    """Refuse to run if any transcribed constant has moved in the file it came from."""
    bad = []
    for name in ("k", "r", "budget_measure_all", "rescue_max", "repair_max", "step_cap",
                 "t0", "expand", "t_max", "min_t", "stop_abs_err_db"):
        if _g32.PREREG.get(name) != PREREG[name]:
            bad.append("g32_repair.PREREG[%r] = %r, transcribed as %r"
                       % (name, _g32.PREREG.get(name), PREREG[name]))
    for name in ("k", "r", "sigma0", "budget_measure_all", "ppo_measure_all_per_eval",
                 "search_measure_all_per_eval"):
        if _h1.PREREG.get(name) != PREREG[name]:
            bad.append("hybrid_audit.PREREG[%r] = %r, transcribed as %r"
                       % (name, _h1.PREREG.get(name), PREREG[name]))
    if _h1.REJECT_SCORE != REJECT_SCORE:
        bad.append("hybrid_audit.REJECT_SCORE = %r, transcribed as %r"
                   % (_h1.REJECT_SCORE, REJECT_SCORE))
    if PREREG["ppo_measure_all_per_eval"] * PREREG["k"] + PREREG["r"] \
            != PREREG["budget_measure_all"]:
        bad.append("2k + r != budget_measure_all")
    if bad:
        raise SystemExit("REFUSING TO RUN: transcribed constants no longer match their "
                         "source:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------------------
# The G3.2 constrained solver, transcribed from g32_repair.main().run
# --------------------------------------------------------------------------------------
def g32_solve(evaluate, xs, s1trace, target, plane, ladder, budget):
    """Returns (trace, info, x_f, rec_f, budget_left).

    The three extra return values are what arm C needs to continue: the design the solver
    was last holding, its measurement, and the unspent budget. They are OUTPUTS ONLY --
    no branch, threshold or step in the solver reads them, so the control flow below is
    the frozen one line for line.
    """
    jb = DIMS.index(plane["boost_axis"])
    jp = DIMS.index(plane["peak_axis"])
    S_BOOST = plane["d_boost_db_per_unit"]
    S_PEAK_BOOST = plane["peak_axis_d_boost_db_per_unit"]
    LO, HI = plane["band_ghz"]
    AIM = plane["peak_aim_ghz"]

    trace, steps = [], []
    left = [budget]
    info = {"steps": steps, "case": None, "rescue_used": 0, "repair_used": 0,
            "rescue_entry": None, "started_feasible": False, "start_source": None,
            "reached_target": False, "wall_hit": False, "reason": None,
            "blocked_by": None, "measured_gain": None}

    def take(x, phase):
        if left[0] <= 0:
            return None
        left[0] -= 1
        rec, _score, gcheck = evaluate(x, target)
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

    # ---- 0. is there already a feasible point? ---------------------------------------
    cand = [(abs(e["boost_db"] - target), j) for j, e in enumerate(s1trace)
            if e and e["loose_pass"]]
    if cand:
        j = min(cand)[1]
        x_f, rec_f = xs[j], s1trace[j]
        info.update(case="already feasible", started_feasible=True,
                    start_source="stage1 (free)")
    else:
        valid = [(len(e["failing"]), abs(e["boost_db"] - target), j)
                 for j, e in enumerate(s1trace) if e]
        x_f = rec_f = None

        if not valid:
            info["case"] = "case 1: guard-invalid handoff"
            base = xs[-1]
            for entry in ladder:
                if info["rescue_used"] >= PREREG["rescue_max"] or left[0] <= 0:
                    break
                e = np.zeros(len(DIMS))
                e[DIMS.index(entry["axis"])] = 1.0
                x_try = np.clip(base + entry["sign"] * entry["mag"] * e, 0.0, 1.0)
                if np.allclose(x_try, base):
                    continue
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
                return trace, info, None, None, left[0]
        else:
            j = min(valid)[2]
            x_f, rec_f = xs[j], s1trace[j]
            info["case"] = "case 2: valid handoff, failing " + ",".join(rec_f["failing"])
            info["start_source"] = "stage1 (free)"

        cap = PREREG["step_cap"]
        gain = plane["d_ln_peak_per_unit"]
        back = 1.0
        while rec_f is not None and not rec_f["loose_pass"]:
            if info["repair_used"] >= PREREG["repair_max"] or left[0] <= 0:
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
                back *= 0.5
                info["blocked_by"] = "guard: " + (steps[-1]["guard_check"] or "?")
                if back < 0.125:
                    info["reason"] = "2-D repair walled in by the guard"
                    break
                continue
            back = 1.0
            moved = x_try[jp] - x_f[jp]
            if abs(moved) > PREREG["min_t"] and rec["peak_freq_ghz"] > 0 and pk > 0:
                gain = float(np.log(rec["peak_freq_ghz"] / pk) / moved)
                info["measured_gain"] = gain
            x_f, rec_f = x_try, rec

        if rec_f is None or not rec_f["loose_pass"]:
            info["reason"] = info["reason"] or "2-D repair did not reach feasibility"
            if rec_f is not None:
                info["blocked_by"] = "hard_pass: " + ",".join(rec_f["failing"])
            return trace, info, x_f, rec_f, left[0]
        info["started_feasible"] = True

    # ---- STAGE B: precision targeting inside the feasible set -------------------------
    b_f = rec_f["boost_db"]
    best_err = abs(b_f - target)
    if best_err <= PREREG["stop_abs_err_db"]:
        info["reached_target"] = True
        info["reason"] = "start already on target"
        return trace, info, x_f, rec_f, left[0]

    way = 1.0 if (target - b_f) * S_BOOST > 0 else -1.0
    side = np.sign(b_f - target)
    lo_t, lo_b = 0.0, b_f
    over_t = over_b = None
    wall_t = None
    t = PREREG["t0"]
    while left[0] > 0:
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
    return trace, info, x_f, rec_f, left[0]


# --------------------------------------------------------------------------------------
# H1's CMA-ES refinement, transcribed from hybrid_audit.main().stage2, h1 branch only
# --------------------------------------------------------------------------------------
def cma_refine(evaluate, x0, target, seed, r, sigma0):
    if r <= 0:
        return [], {"popsize": None, "sigma0": sigma0, "generations": 0, "seed": seed,
                    "arm": "h1", "evals_granted": 0}
    import cma
    es = cma.CMAEvolutionStrategy(list(np.clip(np.asarray(x0, dtype=np.float64), 0.0, 1.0)),
                                  sigma0, {"bounds": [0, 1], "maxfevals": r,
                                           "seed": seed, "verbose": -9})
    trace: list[dict | None] = []
    generations = 0
    while len(trace) < r:
        afford = min(int(es.popsize), r - len(trace))
        cand = es.ask()
        chosen = list(cand[:afford])
        costs = []
        for x in chosen:
            rec, score, _g = evaluate(x, target)
            if rec is not None:
                rec["gen"] = generations
            trace.append(rec)
            costs.append(-score)
        generations += 1
        if afford < int(es.popsize):
            break
        es.tell(chosen, costs)
    return trace, {"popsize": int(es.popsize), "sigma0": sigma0,
                   "generations": generations, "seed": seed, "arm": "h1",
                   "evals_granted": r}


# --------------------------------------------------------------------------------------
def summarize(trace, target, tol):
    """The union of g32_repair.summarize and hybrid_audit.summarize.

    Every field the gate compares -- loose_solved_at, strict_solved_at, n_valid,
    best_boost_db, best_abs_err -- is defined identically in both frozen files, so one
    function can serve both arms without either becoming a different measurement.
    """
    ok = [e for e in trace if e and e["loose_pass"]]
    loose_at = next((i + 1 for i, e in enumerate(trace) if e and e["loose_pass"]), None)
    strict_at = next((i + 1 for i, e in enumerate(trace)
                      if e and e["loose_pass"] and abs(e["boost_db"] - target) <= tol), None)
    best = min(ok, key=lambda e: abs(e["boost_db"] - target)) if ok else None
    return {"loose_solved_at": loose_at, "strict_solved_at": strict_at,
            "n_valid": sum(1 for e in trace if e), "n_loose_pass": len(ok),
            "n_evals": len(trace),
            "n_distinct_boosts": len({round(e["boost_db"], 4) for e in trace if e}),
            "best_boost_db": None if best is None else best["boost_db"],
            "best_abs_err": None if best is None else abs(best["boost_db"] - target),
            "best_design": None if best is None else best["design"],
            "all_valid_boosts": [e["boost_db"] for e in trace if e]}


def pick_fallback_x0(xs, s1trace, x_f, rec_f, target):
    """Amendment 1 §4, in the order it is written there.

    Candidates are the stage-1 designs (each with its x) and the single design G3.2 was
    last holding (x_f, rec_f). Those are exactly the guard-valid designs whose COORDINATES
    the arm still has: G3.2's intermediate points are recorded as measurements, not as
    positions, and inventing an inverse of decode_action to recover them would be a new
    mechanism rather than a frozen one.
    """
    pool = [(x, e) for x, e in zip(xs, s1trace) if e is not None]
    if rec_f is not None and x_f is not None:
        pool.append((np.asarray(x_f, dtype=np.float64), rec_f))
    feas = [(abs(e["boost_db"] - target), n) for n, (_x, e) in enumerate(pool)
            if e["loose_pass"]]
    if feas:
        n = min(feas)[1]
        return np.asarray(pool[n][0], dtype=np.float64), "best hard_pass design"
    if pool:
        n = min((len(e["failing"]), abs(e["boost_db"] - target), n)
                for n, (_x, e) in enumerate(pool))[2]
        return np.asarray(pool[n][0], dtype=np.float64), "fewest failing checks"
    return np.asarray(xs[-1], dtype=np.float64), "PPO handoff (nothing guard-valid)"


# --------------------------------------------------------------------------------------
# Stage 1 and the evaluation both arms share.
#
# These were closures inside main() until the public entry point (eqrl.pipeline) needed
# them. They are lifted here VERBATIM -- same bodies, same constants, same order -- so the
# benchmark and the shipped demo run one implementation rather than two transcriptions of
# it. `main()` binds them below exactly where the closures used to sit, and
# `--gate` re-proves the whole path against the frozen artifacts.
# --------------------------------------------------------------------------------------
class Evaluation:
    """The guarded evaluation both frozen arms record, plus its simulation counter.

    One instance per run. `guards` is memoised per channel loss because building an
    evaluator is the expensive part, exactly as the closure did.
    """

    def __init__(self) -> None:
        self.guards: dict[float, object] = {}
        self.n_sim = 0

    def guard_for(self, channel: float):
        if channel not in self.guards:
            self.guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                                   channel_loss_db=channel)
        return self.guards[channel]

    def make_eval(self, channel: float):
        """One evaluation, recording the UNION of what both frozen arms record.

        The guard, hard_pass and the CMA-ES objective are byte-identical to
        hybrid_audit.evaluate; peak_freq_ghz and `failing` are what g32_repair.evaluate
        additionally needs. Recording a field neither solver reads changes no decision.
        """
        def evaluate(x, target):
            self.n_sim += 1
            dv = decode_action(np.asarray(x))
            spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                       channel_loss_db=channel)
            try:
                v = self.guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
            except Exception as e:
                return None, REJECT_SCORE, "exception:%s" % repr(e)[:80]
            if not v.is_valid:
                return None, REJECT_SCORE, str(getattr(v.check, "value", v.check))
            m = v.unwrap()
            ok, checks = hard_pass(m, spec)
            score = sum(1.0 for c in checks.values() if c) - abs(m.boost_db - target) / 3.0
            rec = {"boost_db": float(m.boost_db), "peak_freq_ghz": float(m.peak_freq_ghz),
                   "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                   "failing": [c for c, good in checks.items() if not good],
                   "design": dataclasses.asdict(dv)}
            return rec, float(score), None
        return evaluate


def stage1_rollout(evaluate, model, env, i, target, channel, k):
    """g32_repair.stage1 and hybrid_audit.stage1 are the same rollout; this is it.

    k evaluations, NO reset on `terminated`, env seeded at 1000 + spec index. Returns
    the designs as well as the measurements, because G3.2 needs the coordinates.
    """
    env.reset(seed=1000 + i)
    env._target, env._channel = target, channel
    obs = env._obs(env._measure(env._x))
    xs = [np.array(env._x, dtype=np.float64)]
    trace = [evaluate(env._x, target)[0]]
    term_at = None
    while len(trace) < k:
        a, _ = model.predict(obs, deterministic=True)
        obs, _rw, term, _tr, _inf = env.step(a)
        xs.append(np.array(env._x, dtype=np.float64))
        trace.append(evaluate(env._x, target)[0])
        if term and term_at is None:
            term_at = len(trace)
    return xs, trace, term_at


def load_policy(model_path: str, *, seed: int = 123):
    """The (model, env) pair stage 1 rolls out in. Seed 123 is the frozen benchmark's."""
    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    return PPO.load(model_path), SequentialEqualizerEnv(fast=False, seed=seed,
                                                        guarded=False)


def main() -> None:
    _check_constants()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--spec-seed", type=int, default=23)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--specs", type=int, default=40)
    p.add_argument("--arms", default="abc")
    p.add_argument("--peak-probe", default="results/g32_peak_probe.json")
    p.add_argument("--rescue-probe", default="results/g32_rescue_probe.json")
    p.add_argument("--tol", type=float, default=PREREG["tol"])
    p.add_argument("--seed", type=int, default=PREREG["base_seed"])
    p.add_argument("--gate", action="store_true",
                   help="equivalence mode: diff arms A and B against the frozen artifacts")
    p.add_argument("--gate-a", default="results/hybrid_audit_h1_seed3.json")
    p.add_argument("--gate-b", default="results/g32_repair_smoke.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or ("results/final_comparison_gate.json" if args.gate
                       else "results/final_comparison_seed%d.json" % args.spec_seed)
    k, r = PREREG["k"], PREREG["r"]
    arms = [a for a in "abc" if a in args.arms.lower()]
    tol = args.tol

    plane = plane_from_probe(args.peak_probe)
    ladder = rescue_order_from_probe(args.rescue_probe)

    specs = list(enumerate(make_specs(args.first + args.specs, args.spec_seed)))
    specs = specs[args.first:args.first + args.specs]

    ev = Evaluation()
    make_eval = ev.make_eval
    model, env = load_policy(args.model)

    def stage1(evaluate, i, target, channel):
        return stage1_rollout(evaluate, model, env, i, target, channel, k)

    print("FINAL COMPARISON | arms %s | spec-seed %d, specs %d-%d | %d measure_all ceiling"
          % ("+".join(a.upper() for a in arms), args.spec_seed, args.first,
             args.first + len(specs) - 1, PREREG["budget_measure_all"]), flush=True)
    print("  stage 1  PPO k=%d at %.2f measure_all each = %.2f, charged to every arm"
          % (k, PREREG["ppo_measure_all_per_eval"],
             k * PREREG["ppo_measure_all_per_eval"]), flush=True)
    print("  stage 2  up to %d evaluations at %.2f each; arm C splits these ONE pool"
          % (r, PREREG["search_measure_all_per_eval"]), flush=True)
    print("  routing  G3.2 reached_target ? stop : hand the REMAINDER to CMA-ES", flush=True)
    if args.gate:
        print("\n  *** EQUIVALENCE GATE: seed 3 specs 8-17, diffed against the frozen "
              "artifacts.\n      Seed 23 is not touched by this mode. ***", flush=True)
    print(flush=True)

    rows = []
    for i, (target, channel) in specs:
        evaluate = make_eval(channel)
        xs, s1, term_at = stage1(evaluate, i, target, channel)
        s1_only = [e for e in s1 if e]
        row = {"spec": i, "target_boost_db": target, "channel_loss_db": channel,
               "handoff": {"boost_db": None if s1[-1] is None else s1[-1]["boost_db"],
                           "guard_valid": s1[-1] is not None,
                           "loose_pass": bool(s1[-1] and s1[-1]["loose_pass"]),
                           "ppo_terminated_at_eval": term_at},
               "stage1": summarize(s1_only, target, tol)}

        # ---- arm A ----------------------------------------------------------------
        if "a" in arms:
            t2, cfg = cma_refine(evaluate, xs[-1], target, args.seed + i, r,
                                 PREREG["sigma0"])
            res = summarize(s1 + t2, target, tol)
            res["stage2"] = summarize(t2, target, tol)
            res["config"] = cfg
            res["stage2_evals"] = len(t2)
            res["measure_all_spent"] = (k * PREREG["ppo_measure_all_per_eval"]
                                        + len(t2) * PREREG["search_measure_all_per_eval"])
            row["a"] = res

        # ---- arms B and C share one G3.2 run --------------------------------------
        if "b" in arms or "c" in arms:
            g2, info, x_f, rec_f, left = g32_solve(evaluate, xs, s1, target, plane,
                                                   ladder, r)
            b_res = summarize(s1_only + g2, target, tol)
            b_res["stage2"] = summarize(g2, target, tol)
            b_res["solver"] = info
            b_res["stage2_evals"] = len(info["steps"])
            b_res["measure_all_spent"] = (
                k * PREREG["ppo_measure_all_per_eval"]
                + len(info["steps"]) * PREREG["search_measure_all_per_eval"])
            if "b" in arms:
                row["b"] = b_res

            if "c" in arms:
                fired = (not info["reached_target"]) and left > 0
                fb, fcfg, x0src = [], None, None
                if fired:
                    x0, x0src = pick_fallback_x0(xs, s1, x_f, rec_f, target)
                    fb, fcfg = cma_refine(evaluate, x0, target, args.seed + i, left,
                                          PREREG["sigma0"])
                c_res = summarize(s1_only + g2 + fb, target, tol)
                c_res["stage2"] = summarize(g2 + fb, target, tol)
                c_res["solver"] = info
                c_res["fallback"] = {
                    "fired": bool(fired),
                    "reason_not_fired": (None if fired else
                                         ("g32 reached target" if info["reached_target"]
                                          else "no budget remained")),
                    "budget_granted": int(left) if fired else 0,
                    "evals_used": len(fb), "x0_source": x0src, "config": fcfg,
                    "g32_best_abs_err": b_res["best_abs_err"],
                    "g32_strict": b_res["strict_solved_at"] is not None,
                }
                c_res["stage2_evals"] = len(info["steps"]) + len(fb)
                c_res["measure_all_spent"] = (
                    k * PREREG["ppo_measure_all_per_eval"]
                    + (len(info["steps"]) + len(fb))
                    * PREREG["search_measure_all_per_eval"])
                row["c"] = c_res

        rows.append(row)
        bits = []
        for a in arms:
            e = row[a]
            bits.append("%s %-6s %-5s %-6s" % (
                a.upper(),
                "-" if e["best_boost_db"] is None else "%.2f" % e["best_boost_db"],
                "-" if e["best_abs_err"] is None else "%.2f" % e["best_abs_err"],
                "strict" if e["strict_solved_at"] else
                ("loose" if e["loose_solved_at"] else "-")))
        tail = ""
        if "c" in arms:
            f = row["c"]["fallback"]
            tail = ("  fb %d evals from %s" % (f["evals_used"], f["x0_source"])
                    if f["fired"] else "  fb no (%s)" % f["reason_not_fired"])
        print("  spec %2d  tgt %5.2f | %s |%s" % (i, target, " | ".join(bits), tail),
              flush=True)

        Path(out).write_text(json.dumps(
            {"prereg": PREREG, "arms": arms, "spec_seed": args.spec_seed,
             "first": args.first, "specs": args.specs, "tol": tol,
             "base_seed": args.seed, "model": args.model, "plane": plane,
             "ladder": ladder, "gate": bool(args.gate),
             "n_simulations_run": ev.n_sim, "complete": len(rows) == len(specs),
             "rows": rows}, indent=1))

    print("\n%d simulations actually run (stage 1 is shared across arms; each arm is "
          "still\ncharged the full %.2f measure_all it would have paid alone)"
          % (ev.n_sim, k * PREREG["ppo_measure_all_per_eval"]), flush=True)
    print("wrote", out, flush=True)

    if args.gate:
        ok = gate(rows, args.gate_a, args.gate_b, arms)
        raise SystemExit(0 if ok else 1)


def gate(rows, path_a, path_b, arms) -> bool:
    """Diff every outcome field against the frozen artifacts. Any mismatch fails."""
    FIELDS = ["best_boost_db", "best_abs_err", "loose_solved_at", "strict_solved_at",
              "n_valid"]
    SOLVER = ["reason", "case", "reached_target", "wall_hit"]
    bad = 0
    print("\n" + "=" * 86)
    print("EQUIVALENCE GATE -- transcribed solvers vs the frozen committed artifacts")
    print("=" * 86)
    for tag, path, key in (("A / H1", path_a, "h1"), ("B / G3.2", path_b, "g32")):
        if tag[0].lower() not in arms:
            continue
        if not Path(path).exists():
            print("  %-9s MISSING artifact %s -- cannot verify" % (tag, path))
            bad += 1
            continue
        froz = {r["spec"]: r[key] for r in json.loads(Path(path).read_text())["rows"]}
        a = tag[0].lower()
        n_ok = 0
        for row in rows:
            i = row["spec"]
            if i not in froz:
                print("  %-9s spec %d absent from %s" % (tag, i, path))
                bad += 1
                continue
            f, g = froz[i], row[a]
            diffs = []
            for fld in FIELDS:
                u, v = f.get(fld), g.get(fld)
                same = (u is None and v is None) or (
                    u is not None and v is not None
                    and (abs(u - v) < 1e-9 if isinstance(u, (int, float))
                         and not isinstance(u, bool) else u == v))
                if not same:
                    diffs.append("%s %r != %r" % (fld, u, v))
            if a == "b":
                if f.get("n_solver_evals") != g.get("stage2_evals"):
                    diffs.append("n_solver_evals %r != %r"
                                 % (f.get("n_solver_evals"), g.get("stage2_evals")))
                for fld in SOLVER:
                    if f["solver"].get(fld) != g["solver"].get(fld):
                        diffs.append("solver.%s %r != %r"
                                     % (fld, f["solver"].get(fld), g["solver"].get(fld)))
            if diffs:
                bad += 1
                print("  %-9s spec %2d  MISMATCH: %s" % (tag, i, "; ".join(diffs)))
            else:
                n_ok += 1
        print("  %-9s %d/%d specs reproduce the frozen artifact exactly  (%s)"
              % (tag, n_ok, len(rows), path))
    print("=" * 86)
    print("GATE %s" % ("PASSED -- the transcription is the frozen controller" if not bad
                       else "FAILED -- %d mismatch(es). The held-out set is NOT to be run "
                            "until this is explained." % bad))
    return bad == 0


if __name__ == "__main__":
    main()
