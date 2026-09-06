"""SILQ's delivered architecture as one callable: a target specification in, a verified
circuit out.

    target specification
        |
    frozen seq_clean40k PPO          global feasibility search        (k evaluations)
        |
    G3.2 constrained refinement      specification closure            (up to r evaluations)
        |
    independent guarded SPICE verification
        |
    final circuit + resulting specs

WHY THIS MODULE EXISTS. `results/delivered_circuit.json` was produced by arm B of
`experiments/final_comparison.py`, but the public entry point `eqrl.solve` ran PPO alone
and then fell back to a fixed hand-verified design. The polished system and the executable
demo therefore described two different architectures, and the demo could not reproduce the
delivered circuit. This module is the one place that architecture lives, and `eqrl.solve`
is now a thin front end over it.

NOTHING HERE IS A NEW CONTROLLER. Stage 1, the guarded evaluation, and G3.2 are IMPORTED
from `experiments.final_comparison` -- the same functions arm B runs, under the same
constants, checked by the same `_check_constants()` guard, and re-proved against the frozen
artifacts by that module's `--gate`. A second transcription of G3.2 would be a second
controller with the same name, which is exactly what the frozen record must not acquire.
Changes no reward, PPO hyperparameter, design bound, guard, hard_pass, controller constant,
frozen model or benchmark criterion.

THE VERIFICATION STEP IS NOT THE SEARCH'S OWN OPINION. G3.2 stops on the nine-check
`loose_pass` plus a target-error test; this module then re-measures the design it returned
through a FRESH evaluator on all ten checks -- the nine plus `boost_target` -- which is the
criterion `results/delivered_circuit.json` was held to. It is a second, independent
measurement, so `verification.boost_db` reproducing `solver.best_boost_db` is evidence the
result is real and not a cached number.

    from eqrl.pipeline import design
    r = design(target_boost_db=8.92, channel_loss_db=14.83)
    r["status"], r["design"], r["verification"]["measures"]["boost_db"]

Importing this module pulls in `experiments.final_comparison`, which performs the ngspice /
PDK environment bootstrap every experiment module performs at import. That import is
deliberately deferred to call time so that `import eqrl.pipeline` stays cheap and works on a
machine with no PDK -- CI imports this file and never calls it.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.specs import DEFAULT_SPEC, Spec, hard_pass

#: The policy the delivered circuit was produced by. Frozen; see RESULTS.md section 20.
POLICY = "results/seq_clean40k.zip"

#: The two probe artifacts G3.2 reads its move plane and rescue order from. Both are
#: committed results, not tunables -- G3.2 is not G3.2 without them.
PEAK_PROBE = "results/g32_peak_probe.json"
RESCUE_PROBE = "results/g32_rescue_probe.json"

#: Labels for `status`, so a caller never has to parse prose to find out what produced the
#: circuit it is holding.
SOLVED = "solved"                 #: PPO -> G3.2 closed the target and verification agrees
CLOSED_NOT_VERIFIED = "closed_but_failed_verification"
UNSOLVED = "unsolved"             #: the architecture ran and did not reach the target
FALLBACK = "fallback_fixed_design_not_ai"   #: see `design(..., allow_fallback=True)`

#: Every mode runs the SAME frozen `g32_solve`. Stage 1 is also the same frozen PPO
#: rollout for "default", "thinking" and "retarget" -- but NOT for "fastest",
#: which replaces stage 1 with a corpus lookup; see `eqrl.experiments.fastest_hedge`.
#: "default" is byte-identical to this module's pre-mode behaviour: r = fc.PREREG["r"],
#: stop_abs_err_db left at None so g32_solve reads fc.PREREG itself.
MODES = ("default", "fastest", "thinking", "retarget")

#: "accurate" WAS a mode here -- same feasibility gate, r = 30, stop = 0.05 dB. It was
#: removed 06 Sep 2026 after `scratchpad/mode_bench32.py` measured it against `default` on
#: spec-seed 137 and found it produced an IDENTICAL result, to four decimal places, on
#: every spec: its larger budget and tighter tolerance changed nothing because neither
#: budget nor tolerance was the binding constraint. A mode that never differs from another
#: mode is a label, not a mode. What section 4 of docs/DESIGN_MODES_V2.md actually proposed
#: for it -- relaxing the DC-gain floor so the search may cross the feasibility wall -- was
#: never implemented and is the only thing that would have made it distinct. That remains
#: unbuilt; do not resurrect the name without it.

#: Fastest: stage 1 is a corpus lookup (eqrl.experiments.fastest_hedge.surrogate_stage1) --
#: no PPO rollout, one real evaluation instead of PREREG["k"]=5 -- then the unmodified
#: `g32_solve` with this fixed budget. No floor-budget escalation: that safety net (used
#: while stage 1 was still the full PPO rollout) traded speed for accuracy-parity with
#: default, which is the opposite of what this mode is now for. A rejected/weak seed costs
#: at most this many evaluations, same as any other spec -- it does not fall back to
#: `default`'s r=10.
FASTEST_BUDGET = 3

#: Thinking: up to THINKING_ROLLOUTS independent PPO stage-1 rollouts, each closed by the
#: frozen G3.2, best of N kept; then, only if none of them reached target, up to
#: THINKING_SURROGATE_STARTS corpus-proposed starts (`eqrl.experiments.thinking_starts`).
#: Diversity in the PPO phase comes from calling the frozen `stage1_rollout` with N
#: different spec indices -- it seeds its env at `1000 + i`, so a different `i` is a
#: different rollout without touching its seeding rule. The offsets are fixed and
#: arbitrary, chosen only to be distinct and reproducible.
#:
#: The three numbers below are docs/DESIGN_MODES_V2.md section 3 item 1 and item 3:
#: 3 -> 8 rollouts, r 15 -> 25, stop 0.05 -> 0.01 dB. The stop tolerance is deliberately
#: TIGHTER than the mode's own accuracy goal (+-0.1 dB): the mode should stop because the
#: bisection converged, not because it hit its own tolerance and gave up early.
THINKING_ROLLOUTS = 8
THINKING_R = 25
THINKING_STOP_ABS_ERR_DB = 0.01
THINKING_SEED_OFFSETS = (0, 4001, 9007, 15013, 23021, 31033, 42043, 55049)

#: Corpus-proposed restarts, tried ONLY after every PPO rollout has failed to reach target.
#: Motivated by the n=32 measurement in DESIGN_MODES_V2 section 6: 8 of 32 specs failed
#: because no start ever reached the feasible set, which is a start-quality problem that
#: more budget from the same start cannot fix. Each costs ONE real evaluation instead of
#: PPO's five.
THINKING_SURROGATE_STARTS = 4

#: Wall-clock governor, in `measure_all` units. The requested envelope is ~60 s and a
#: `measure_all` costs ~0.55 s (DESIGN_MODES_V2 section 0), so ~100 is the budget. Without
#: this, 8 rollouts x (5 + 25) is 240 measure_all -- over two minutes -- in the worst case,
#: which is the case where nothing is working and the user is waiting longest. No restart
#: is STARTED once the spend reaches this; a restart already running is always finished, so
#: this trims the search rather than truncating a solve into a misleading partial answer.
THINKING_MEASURE_ALL_BUDGET = 100

#: Retarget: DEFAULT, plus one thing -- when G3.2's boost axis is pinned at a bound, probe
#: the other axes and resume the line search along the best one
#: (`eqrl.experiments.axis_retarget`). Budget and stopping tolerance are deliberately left
#: at Default's own `fc.PREREG` values so that ANY difference between this mode and
#: "default" is attributable to the second axis and to nothing else. That is what makes
#: this arm measurable; do not "improve" it by also raising r.
RETARGET_PROBE_EVALS = 3


def _plain(o: Any) -> Any:
    """Coerce numpy scalars to built-ins so a result survives `json.dumps`.

    `hard_pass` compares numpy floats and so returns numpy booleans, and `json` refuses
    those: `np.float64` subclasses `float` and serialises fine, `np.bool_` does not
    subclass `bool` and raises. This changes no value, only its Python type, and it runs
    once on the finished result rather than being sprinkled through the code that
    produces it.
    """
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _fc():
    """Import the benchmark module (and its env bootstrap) at call time, not import time."""
    from eqrl.experiments import final_comparison as fc
    return fc


def spec_for(target_boost_db: float, channel_loss_db: float, tol: float) -> Spec:
    """The requirement set the result is verified against.

    DEFAULT_SPEC plus this run's target and channel, with `boost_target_tol_db` ON. That
    tenth check is what `results/delivered_circuit.json` was scored on, and what makes
    "meets the specification" mean the REQUESTED boost rather than anywhere in 3-12 dB.
    `dc_gain_db_min` is already 0.0 in DEFAULT_SPEC and is left alone.
    """
    return dataclasses.replace(DEFAULT_SPEC, target_boost_db=target_boost_db,
                               channel_loss_db=channel_loss_db,
                               boost_target_tol_db=tol)


def verify(dv: DesignVars, spec: Spec) -> dict[str, Any]:
    """Re-measure one design through a fresh guarded evaluator on all ten checks.

    Independent of the search: a new evaluator, a new measurement, no cached record. The
    guard runs first, so a design that is not a real circuit fails here even if every
    number it reported looked good.
    """
    from eqrl.evaluator import build_evaluator

    ev = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                         channel_loss_db=spec.channel_loss_db)
    verdict = ev.evaluate(dv, vdd=spec.vdd_nominal)
    if not verdict.is_valid:
        return {"guard_valid": False,
                "guard_check": str(getattr(verdict.check, "value", verdict.check)),
                "passed": False, "checks": None, "measures": None}
    m = verdict.unwrap()
    ok, checks = hard_pass(m, spec)
    return {"guard_valid": True, "guard_check": None, "passed": bool(ok),
            "checks": checks, "failing": [c for c, good in checks.items() if not good],
            "measures": m.as_dict(),
            "abs_err_db": abs(m.boost_db - spec.target_boost_db)}


def _cost(fc, k: int, info: dict, left: int, c1: dict, c2: dict) -> dict[str, Any]:
    """The run's simulator cost in all three units REPRODUCE.md section 13 distinguishes.

    `c1` and `c2` are counted, not derived. Verification is added later by `design`,
    because it is a genuinely separate measurement and merging it into the search cost
    would overstate what the search spent.

    WHY THE CHARGED AND MEASURED NUMBERS DIFFER, which is a finding rather than a bug in
    this function. The benchmark charges stage 1 at `k * ppo_measure_all_per_eval`
    (5 x 2.00 = 10), the factor section 13 measured for `env.step` plus guarded verify.
    The rollout actually spends 11: `stage1_rollout` calls `env.reset(seed=...)`, which
    measures, and then re-measures the SAME unchanged design through `env._measure(env._x)`
    to build the first observation. That extra measurement is real simulator work the
    charge does not count, so it is reported here rather than absorbed. Changing it would
    mean editing the frozen `stage1_rollout` and re-running the benchmark record, which is
    not this task's call to make.
    """
    steps = len(info["steps"])
    charged = (k * fc.PREREG["ppo_measure_all_per_eval"]
               + steps * fc.PREREG["search_measure_all_per_eval"])
    search_m = c1["measure_all"] + c2["measure_all"]
    return {
        # Unit 1 -- optimizer evaluations. What `--budget` counts, and what "median 4
        # simulations to first solve" has always meant.
        "optimizer_evals": k + steps,
        "stage1_evals": k,
        "stage2_evals": steps,
        "budget_evals_unspent": int(left),

        # Unit 2 -- measure_all calls, counted at `measures.measure_all`.
        "measure_all_search": search_m,
        "measure_all_stage1": c1["measure_all"],
        "measure_all_stage2": c2["measure_all"],
        "measure_all_verification": 0,          # filled in once verification has run
        "measure_all_total": search_m,
        "measure_all_budget": fc.PREREG["budget_measure_all"],

        # Unit 3 -- SPICE analyses inside libngspice, counted at `NgspiceServer._analysis`.
        "spice_analyses_search": c1["analysis"] + c2["analysis"],
        "spice_analyses_verification": 0,
        "spice_analyses_total": c1["analysis"] + c2["analysis"],

        # The benchmark's accounting, kept so this run stays comparable to the frozen
        # record, with the measured shortfall stated instead of hidden.
        "measure_all_charged_by_prereg": charged,
        "measure_all_uncharged_by_prereg": search_m - charged,
        "counting_method":
            "counted at measures.measure_all and NgspiceServer._analysis "
            "(REPRODUCE.md section 13), not derived from a per-evaluation factor",
    }


def _add_verification_cost(cost: dict[str, Any], c3: dict) -> None:
    """Fold the independent verification's measured cost into the run's totals.

    Kept separate from the search figures on purpose: verification is the second, fresh
    measurement that makes `verification.boost_db` evidence rather than a cached number,
    and charging it to the search budget would misstate what the search spent.
    """
    cost["measure_all_verification"] = c3["measure_all"]
    cost["measure_all_total"] = cost["measure_all_search"] + c3["measure_all"]
    cost["spice_analyses_verification"] = c3["analysis"]
    cost["spice_analyses_total"] = cost["spice_analyses_search"] + c3["analysis"]


def _guidance(mode: str, info: dict, arm_b: dict, target_boost_db: float) -> dict:
    """What to tell the user about WHY this run stopped where it did.

    docs/DESIGN_MODES_V2.md section 3. The user asked Thinking to say "default already
    found the best circuit" when nothing better is available. **That claim cannot be made
    honestly** -- none of these solvers proves global optimality, and asserting it would be
    the single easiest way for this project to be caught overclaiming. What CAN be said is
    the mechanism, which `g32_solve` already records in `info["reason"]` at no cost, and
    each reason licenses a different, true statement.

    The `wall` case is the one that matters most and is the one a naive implementation gets
    backwards: converging onto the feasibility wall does NOT mean nothing better exists. It
    means something better exists and costs DC gain below the guard's floor. Telling the
    user "this is the best possible" there would be false. It routes to Accurate instead.

    `actionable` is the flag a UI should use to decide whether to offer another mode; it is
    False exactly when no other mode in this codebase is known to help.
    """
    reason = info.get("reason") or ""
    err = arm_b.get("best_abs_err")
    got = arm_b.get("best_boost_db")
    near = "" if got is None else " Closest achievable found: %.2f dB." % got

    if info.get("reached_target"):
        return {"headline": "Target reached inside the feasible set.",
                "detail": "The design meets the boost target and passes the feasibility "
                          "guard. Nothing further is required.",
                "suggest_mode": None, "actionable": False, "reason": reason}
    if "feasibility wall" in reason:
        # Measured, not assumed: on spec-seed 137 Thinking cleared two of default's three
        # wall cases (3.92 -> 0.0007, 2.28 -> 0.0007) WITHOUT relaxing the guard, because a
        # different restart lands in a basin where the wall is not binding. So the honest
        # advice here is a different start, not a weaker guard.
        return {"headline": "Stopped at the feasibility wall, not at the best circuit.",
                "detail": "Getting closer to %.2f dB from THIS starting point requires DC "
                          "gain below the guard's 0 dB floor, which is refused.%s That is "
                          "a statement about this basin, not about the circuit family -- "
                          "Thinking mode restarts from other starting points and clears "
                          "most walls without weakening the guard."
                          % (target_boost_db, near),
                "suggest_mode": None if mode == "thinking" else "thinking",
                "actionable": mode != "thinking", "reason": reason}
    if "no admissible step" in reason:
        return {"headline": "The boost control is at its limit for this channel.",
                "detail": "The primary boost axis has reached a bound, so the line search "
                          "has no admissible step left.%s This target may not be reachable "
                          "with this topology. Retarget mode probes the other axes, though "
                          "it is measured to fire rarely." % near,
                "suggest_mode": "retarget", "actionable": True, "reason": reason}
    if "budget exhausted" in reason:
        return {"headline": "Ran out of evaluation budget before converging.",
                "detail": "The search was still improving when its budget ran out.%s "
                          "Thinking mode spends a larger budget on the same search."
                          % near,
                "suggest_mode": None if mode == "thinking" else "thinking",
                "actionable": mode != "thinking", "reason": reason}
    if arm_b.get("best_design") is None:
        return {"headline": "No design passed the feasibility guard.",
                "detail": "Neither the policy nor the repair ladder produced a "
                          "guard-valid circuit for this request, so there is nothing to "
                          "report rather than a circuit that does not hold up. Thinking "
                          "mode tries additional and corpus-proposed starting points.",
                "suggest_mode": None if mode == "thinking" else "thinking",
                "actionable": mode != "thinking", "reason": reason}
    return {"headline": "Stopped short of the target.",
            "detail": "Solver reason: %s.%s" % (reason or "unrecorded", near),
            "suggest_mode": None, "actionable": False, "reason": reason}


def design(target_boost_db: float, channel_loss_db: float = DEFAULT_SPEC.channel_loss_db,
           *, model: str = POLICY, spec_index: int = 0, tol: float | None = None,
           peak_probe: str = PEAK_PROBE, rescue_probe: str = RESCUE_PROBE,
           allow_fallback: bool = False, mode: str = "default") -> dict[str, Any]:
    """Run the delivered architecture for one specification.

    `spec_index` seeds stage 1's environment (`1000 + spec_index`), exactly as the
    benchmark does. It is the only knob that changes which circuit comes out for a given
    target, and it is recorded in the result so any run is reproducible.

    `mode` selects one of `MODES`. For "default", "thinking" and "retarget"
    this changes only the stage-2 budget/tolerance (and, for "thinking", how many
    independent stage-1 rollouts are tried) -- stage 1 is always the same frozen PPO
    rollout. "fastest" is the one exception: it replaces stage 1 itself with a corpus
    lookup plus a single real evaluation (`eqrl.experiments.fastest_hedge.surrogate_stage1`)
    instead of PPO's `k` rollout evaluations, because that rollout, not stage 2, was
    measured to be the majority of the wall-clock cost a "fast" mode is supposed to cut.
    No mode ever changes the PPO policy itself, the guard, `hard_pass`, or `g32_solve`'s
    control flow -- see the module-level `MODES`/`FASTEST_*`/`THINKING_*`
    constants for exactly what each preset is. "default" reproduces this function's
    pre-mode behaviour bit for bit.

    `allow_fallback` is OFF by default and is not part of the architecture. When on, and
    only when PPO -> G3.2 produced nothing guard-valid at all, the fixed
    `baselines.robust.robust_design()` is returned with `status` set to
    `fallback_fixed_design_not_ai` and `provenance.is_ai_generated` False. That design
    ignores the requested target -- it is a product-robustness escape hatch, never an
    answer the framework searched for, and the labelling exists so it can never be
    mistaken for one.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    fc = _fc()
    fc._check_constants()
    k = fc.PREREG["k"]
    tol = fc.PREREG["tol"] if tol is None else tol

    plane = fc.plane_from_probe(peak_probe)
    ladder = fc.rescue_order_from_probe(rescue_probe)

    ev = fc.Evaluation()
    evaluate = ev.make_eval(channel_loss_db)
    # "fastest" never rolls out PPO -- see the mode dispatch below -- so it has no use for
    # the policy/env pair, and loading them would be pure overhead this mode exists to cut.
    policy, env = (None, None) if mode == "fastest" else fc.load_policy(model)

    # Cost is COUNTED at the two chokepoints REPRODUCE.md section 13 defines, never
    # derived from a per-evaluation factor. The factor is what went wrong before: this
    # module reported `ev.n_sim`, which counts `evaluate()` calls and is neither of
    # section 13's units, and charged stage 1 at k x 2.00 measure_all, which the
    # measurement below shows is one short of what the rollout actually spends.
    from eqrl.simcount import counting

    mode_detail: dict[str, Any] = {"mode": mode}

    if mode == "thinking":
        # ---- N independent PPO rollouts, each closed by the frozen G3.2 -------------
        # Adaptive (docs/RESULTS_INFERENCE_MODES.md section 6.2): offsets are spent in
        # order and stop the moment one reaches target. Restart diversity is worth its
        # cost only when the rollout so far did NOT land on target -- paying for more
        # independent rollouts after the first already solved it recovers nothing (the
        # winner is already picked by _rank below) and only adds SPICE cost.
        #
        # Two things bound the worst case, which is the case where nothing works and the
        # user is waiting longest: THINKING_ROLLOUTS caps how many restarts exist, and
        # THINKING_MEASURE_ALL_BUDGET stops NEW ones being started once the run has spent
        # its wall-clock envelope. A restart already under way is always finished -- a
        # truncated solve would report a partial search as if it were a converged one.
        from eqrl.experiments.thinking_starts import diverse_seeds

        candidates = []
        c1 = {"measure_all": 0, "analysis": 0}
        c2 = {"measure_all": 0, "analysis": 0}
        governor_stopped = False

        def _spent() -> int:
            return c1["measure_all"] + c2["measure_all"]

        def _close(xs_i, s1_i, term_i, label):
            """Run the frozen g32_solve from one start and record the candidate."""
            s1_only_i = [e for e in s1_i if e]
            with counting() as c2_i:
                g2_i, info_i, _xf, _rf, left_i = fc.g32_solve(
                    evaluate, xs_i, s1_i, target_boost_db, plane, ladder, THINKING_R,
                    stop_abs_err_db=THINKING_STOP_ABS_ERR_DB)
            for key in c2:
                c2[key] += c2_i[key]
            candidates.append({
                "start": label, "xs": xs_i, "s1": s1_i, "s1_only": s1_only_i,
                "term_at": term_i, "g2": g2_i, "info": info_i, "left": left_i,
                "arm": fc.summarize(s1_only_i + g2_i, target_boost_db, tol)})
            return info_i["reached_target"]

        for offset in THINKING_SEED_OFFSETS[:THINKING_ROLLOUTS]:
            if candidates and _spent() >= THINKING_MEASURE_ALL_BUDGET:
                governor_stopped = True
                break
            with counting() as c1_i:
                xs_i, s1_i, term_i = fc.stage1_rollout(
                    evaluate, policy, env, spec_index + offset, target_boost_db,
                    channel_loss_db, k)
            for key in c1:
                c1[key] += c1_i[key]
            if _close(xs_i, s1_i, term_i, "ppo:%d" % offset):
                break

        # ---- corpus-proposed starts, ONLY if no PPO rollout reached target -----------
        # See eqrl.experiments.thinking_starts: the failures this addresses are ones where
        # no start ever reached the feasible set, which more budget from the same start
        # cannot fix. Each start costs ONE real guarded evaluation, not PPO's five, and the
        # corpus never judges -- g32_solve and the verifier still decide.
        n_surrogate = 0
        if candidates and not any(c["info"]["reached_target"] for c in candidates):
            try:
                from eqrl.experiments.fastest_hedge import load_fastest_assets

                surrogate, _cx, _rad = load_fastest_assets()
                tried = np.array([x for c in candidates for x in c["xs"]],
                                 dtype=np.float64)
                seeds = diverse_seeds(target_boost_db, surrogate,
                                      spec_for(target_boost_db, channel_loss_db, tol),
                                      THINKING_SURROGATE_STARTS, exclude=tried)
            except Exception:
                seeds = []          # no corpus on disk -> Thinking is just best-of-N PPO
            for x0 in seeds:
                if _spent() >= THINKING_MEASURE_ALL_BUDGET:
                    governor_stopped = True
                    break
                with counting() as c1_i:
                    rec0, _s, _g = evaluate(x0, target_boost_db)
                for key in c1:
                    c1[key] += c1_i[key]
                n_surrogate += 1
                if _close([x0], [rec0], None, "surrogate:%d" % n_surrogate):
                    break

        def _rank(c):
            has_design = c["arm"]["best_design"] is not None
            err = c["arm"]["best_abs_err"]
            return (0 if has_design else 1, err if err is not None else float("inf"))

        winner = min(candidates, key=_rank)
        s1, s1_only, term_at = winner["s1"], winner["s1_only"], winner["term_at"]
        g2, info, left = winner["g2"], winner["info"], winner["left"]
        arm_b = winner["arm"]
        # Stage 1 cost is charged per PPO rollout at k, plus one per corpus start. Counting
        # every candidate at k would overstate what the corpus starts actually spent.
        n_ppo = len(candidates) - n_surrogate
        k_eff = k * n_ppo + n_surrogate
        mode_detail["rollouts"] = [
            {"start": c["start"], "best_abs_err_db": c["arm"]["best_abs_err"],
             "n_loose_pass": c["arm"]["n_loose_pass"],
             "reached_target": bool(c["info"]["reached_target"])}
            for c in candidates]
        mode_detail["winner_start"] = winner["start"]
        mode_detail["n_ppo_rollouts"] = n_ppo
        mode_detail["n_surrogate_starts"] = n_surrogate
        mode_detail["measure_all_spent"] = _spent()
        mode_detail["governor_stopped"] = governor_stopped
        mode_detail["adaptive_stopped_early"] = (
            n_ppo < min(THINKING_ROLLOUTS, len(THINKING_SEED_OFFSETS))
            and not governor_stopped)

    elif mode == "fastest":
        # ---- stage 1 REPLACEMENT: corpus lookup, no PPO, one real evaluation ---------
        # See eqrl.experiments.fastest_hedge.surrogate_stage1's docstring for why this is
        # sound with no channel awareness in the corpus, and why exactly one real
        # evaluation (not zero) is spent before g32_solve ever sees the candidate.
        from eqrl.experiments.fastest_hedge import load_fastest_assets, surrogate_stage1

        surrogate, _corpus_X, _radius = load_fastest_assets()
        seed_spec = spec_for(target_boost_db, channel_loss_db, tol)
        with counting() as c1:
            xs, s1, term_at = surrogate_stage1(evaluate, target_boost_db, surrogate,
                                               seed_spec)
        s1_only = [e for e in s1 if e]
        k_eff = 1  # one real evaluation was spent (confirming the corpus candidate), not
                   # PREREG["k"]=5 -- `optimizer_evals`/`stage1_evals` must say so honestly

        # ---- stage 2: the SAME frozen g32_solve, fixed small budget, no floor --------
        with counting() as c2:
            g2, info, _x_f, _rec_f, left = fc.g32_solve(
                evaluate, xs, s1, target_boost_db, plane, ladder, FASTEST_BUDGET)
        seed_rec = s1[0]
        mode_detail["surrogate_seed"] = {
            "candidate_boost_db": None if seed_rec is None else seed_rec["boost_db"],
            "target_boost_db": target_boost_db,
            "candidate_guard_valid": seed_rec is not None,
            "candidate_loose_pass": bool(seed_rec and seed_rec["loose_pass"]),
        }
        arm_b = fc.summarize(s1_only + g2, target_boost_db, tol)

    else:
        # ---- stage 1: the frozen PPO policy, k evaluations ---------------------------
        with counting() as c1:
            xs, s1, term_at = fc.stage1_rollout(evaluate, policy, env, spec_index,
                                                target_boost_db, channel_loss_db, k)
        s1_only = [e for e in s1 if e]
        k_eff = k

        # ---- stage 2: G3.2 constrained refinement, mode-dependent budget -------------
        if mode == "retarget":
            from eqrl.experiments.axis_retarget import (RETARGET_SURROGATE_OPTIONAL,
                                                        retarget_stage2)

            surrogate = RETARGET_SURROGATE_OPTIONAL()
            with counting() as c2:
                g2, info, _x_f, _rec_f, left = retarget_stage2(
                    evaluate, xs, s1, target_boost_db, plane, ladder, fc.PREREG["r"],
                    surrogate=surrogate, probe_evals=RETARGET_PROBE_EVALS,
                    stop_abs_err_db=None)
            mode_detail["retarget"] = info.get("retarget")
        else:
            r, stop = fc.PREREG["r"], None
            with counting() as c2:
                g2, info, _x_f, _rec_f, left = fc.g32_solve(
                    evaluate, xs, s1, target_boost_db, plane, ladder, r,
                    stop_abs_err_db=stop)

        arm_b = fc.summarize(s1_only + g2, target_boost_db, tol)

    stage1 = fc.summarize(s1_only, target_boost_db, tol)

    result: dict[str, Any] = {
        "architecture": "PPO (global feasibility) -> G3.2 constrained refinement",
        "mode": mode,
        "spec": {"target_boost_db": target_boost_db, "channel_loss_db": channel_loss_db,
                 "boost_tol_db": tol, "spec_index": spec_index,
                 "requirement_set": "eqrl.specs.Spec defaults + this target/channel; "
                                    "dc_gain_db_min=0.0 and boost_target_tol_db both ON"},
        "provenance": {
            "is_ai_generated": True,
            "policy": model,
            "mode": mode,
            "mode_detail": mode_detail,
            "ppo_produced_handoff": s1[-1] is not None,
            "ppo_terminated_at_eval": term_at,
            "stage1": {"n_evals": stage1["n_evals"], "n_valid": stage1["n_valid"],
                       "best_boost_db": stage1["best_boost_db"],
                       "best_abs_err_db": stage1["best_abs_err"]},
            "g32_used": True,
            "g32_start_source": info["start_source"],
            "g32_steps": [{"phase": s["phase"], "boost_db": s["boost_db"],
                           "abs_err_db": s["abs_err"]} for s in info["steps"]],
            "g32_reached_target": bool(info["reached_target"]),
            "g32_reason": info["reason"],
            "g32_rescue_used": info["rescue_used"],
            "g32_repair_used": info["repair_used"],
            "fallback_invoked": False,
            "hand_tuning": "none",
        },
        "guidance": _guidance(mode, info, arm_b, target_boost_db),
        "cost": _cost(fc, k_eff, info, left, c1, c2),
        "solver": arm_b,
        "design": arm_b["best_design"],
        "netlist": None,
        "verification": None,
        "status": UNSOLVED,
    }

    # ---- independent verification -----------------------------------------------------
    if arm_b["best_design"] is not None:
        dv = DesignVars(**arm_b["best_design"])
        spec = spec_for(target_boost_db, channel_loss_db, tol)
        with counting() as c3:
            v = verify(dv, spec)
        result["verification"] = v
        _add_verification_cost(result["cost"], c3)
        result["netlist"] = netlist(dv, vdd=spec.vdd_nominal, temp_c=27.0, corner="tt",
                                    analysis="ac", models="sky130")
        result["status"] = SOLVED if v["passed"] else CLOSED_NOT_VERIFIED
        return _plain(result)

    # ---- nothing guard-valid came out of the architecture -----------------------------
    if allow_fallback:
        from eqrl.baselines.robust import robust_design

        dv = robust_design()
        spec = spec_for(target_boost_db, channel_loss_db, tol)
        with counting() as c3:
            v = verify(dv, spec)
        _add_verification_cost(result["cost"], c3)
        result["design"] = dataclasses.asdict(dv)
        result["verification"] = v
        result["netlist"] = netlist(dv, vdd=spec.vdd_nominal, temp_c=27.0, corner="tt",
                                    analysis="ac", models="sky130")
        result["status"] = FALLBACK
        result["provenance"].update({
            "is_ai_generated": False,
            "fallback_invoked": True,
            "fallback_source": "eqrl.baselines.robust.robust_design",
            "fallback_note": "A FIXED, HAND-VERIFIED design. It does not read the "
                             "requested target and was not searched for by PPO or G3.2. "
                             "It is not an output of the SILQ architecture and must "
                             "never be reported as one.",
        })
    return _plain(result)


def describe(r: dict[str, Any]) -> str:
    """A human-readable rendering of one result, for the CLI and for reports."""
    d = r["design"]
    p, c = r["provenance"], r["cost"]
    out = []
    if r["status"] == FALLBACK:
        out.append("  !! FIXED FALLBACK DESIGN -- NOT generated by PPO or G3.2 !!")
        out.append("     The architecture returned nothing guard-valid for this spec.")
    else:
        out.append("  architecture  PPO (%d evals) -> G3.2 (%d evals)%s"
                   % (c["stage1_evals"], c["stage2_evals"],
                      "" if p["g32_reached_target"] else "  [target NOT reached]"))
        out.append("  provenance    policy %s, started from %s, %s"
                   % (p["policy"], p["g32_start_source"], p["g32_reason"]))
        # Counted, not derived, and in all three units -- printing one number called
        # "simulations" is what made this line misleading before.
        out.append("  cost          %d optimizer evals = %d measure_all "
                   "(%d search + %d verification) = %d SPICE analyses"
                   % (c["optimizer_evals"], c["measure_all_total"],
                      c["measure_all_search"], c["measure_all_verification"],
                      c["spice_analyses_total"]))
    if d is None:
        out.append("\n  no design: the architecture produced nothing guard-valid.")
        return "\n".join(out)
    out.append("")
    out.append("    W/L        %.2f / %.4f um" % (d["w_in"] * 1e6, d["l_in"] * 1e6))
    out.append("    I_tail     %.1f uA" % (d["i_tail"] * 1e6))
    out.append("    Rs / Cs    %.2f kOhm / %.1f fF" % (d["rs"] / 1e3, d["cs"] * 1e15))
    out.append("    R_load     %.1f Ohm" % d["r_load"])
    v = r["verification"]
    if v and v["measures"]:
        m = v["measures"]
        out.append("")
        out.append("    boost      %.3f dB @ %.3f GHz   (requested %.3f +/- %.2f)"
                   % (m["boost_db"], m["peak_freq_ghz"], r["spec"]["target_boost_db"],
                      r["spec"]["boost_tol_db"]))
        out.append("    dc gain    %.3f dB" % m["dc_gain_db"])
        out.append("    HD3        %.1f dB      noise %.0f uVrms"
                   % (m["hd3_db"], m["noise_vrms"] * 1e6))
        out.append("    power      %.3f mW     area  %.6f mm^2"
                   % (m["power_w"] * 1e3, m["area_mm2"]))
        out.append("    eye        %.3f UI / %.0f mV" % (m["eye_h_ui"], m["eye_v_mv"]))
    out.append("")
    if v is None:
        out.append("  verification  not run")
    elif not v["guard_valid"]:
        out.append("  verification  GUARD REJECTED: %s" % v["guard_check"])
    elif v["passed"]:
        out.append("  verification  ALL TEN CHECKS PASS (independent re-measurement)")
    else:
        out.append("  verification  FAILS: %s" % ", ".join(v["failing"]))
    out.append("  status        %s" % r["status"])
    return "\n".join(out)
