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

#: Every mode runs the SAME frozen PPO stage 1 and the SAME frozen `g32_solve` -- see that
#: function's docstring and `eqrl.experiments.fastest_hedge` for what actually differs.
#: "default" is byte-identical to this module's pre-mode behaviour: r = fc.PREREG["r"],
#: stop_abs_err_db left at None so g32_solve reads fc.PREREG itself.
MODES = ("default", "accurate", "fastest", "thinking")

#: Accurate: same feasibility gate, a bigger stage-2 budget and a tighter stopping test on
#: the SAME bisection loop. Nothing here is a new search -- just more of the existing one,
#: asked to stop closer to the target than PREREG's own 0.25 dB.
ACCURATE_R = 30
ACCURATE_STOP_ABS_ERR_DB = 0.05

#: Fastest: one surrogate-guided real evaluation (eqrl.experiments.fastest_hedge), then the
#: unmodified `g32_solve` with whatever budget is left. 4 is 1 hedge + 3 for G3.2 -- roughly
#: a third of default's r=10, which is where the time saving comes from.
FASTEST_BUDGET = 4

#: Thinking: THINKING_ROLLOUTS independent PPO stage-1 rollouts, each closed by G3.2 at
#: Accurate's tolerance, the best of the N kept. Diversity comes from calling the frozen
#: `stage1_rollout` with N different spec indices -- it seeds its env at `1000 + i`, so a
#: different `i` is a different rollout without touching its seeding rule. The offsets are
#: fixed and arbitrary, chosen only to be distinct and reproducible.
THINKING_ROLLOUTS = 3
THINKING_R = 15
THINKING_STOP_ABS_ERR_DB = 0.05
THINKING_SEED_OFFSETS = (0, 4001, 9007)


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


def design(target_boost_db: float, channel_loss_db: float = DEFAULT_SPEC.channel_loss_db,
           *, model: str = POLICY, spec_index: int = 0, tol: float | None = None,
           peak_probe: str = PEAK_PROBE, rescue_probe: str = RESCUE_PROBE,
           allow_fallback: bool = False, mode: str = "default") -> dict[str, Any]:
    """Run the delivered architecture for one specification.

    `spec_index` seeds stage 1's environment (`1000 + spec_index`), exactly as the
    benchmark does. It is the only knob that changes which circuit comes out for a given
    target, and it is recorded in the result so any run is reproducible.

    `mode` selects one of `MODES` and changes ONLY the stage-2 budget/tolerance (and, for
    "thinking", how many independent stage-1 rollouts are tried). It never changes the PPO
    policy, the guard, `hard_pass`, or `g32_solve`'s control flow -- see the module-level
    `MODES`/`ACCURATE_*`/`FASTEST_*`/`THINKING_*` constants for exactly what each preset is,
    and `eqrl.experiments.fastest_hedge` for Fastest's one new mechanism. "default"
    reproduces this function's pre-mode behaviour bit for bit.

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
    policy, env = fc.load_policy(model)

    # Cost is COUNTED at the two chokepoints REPRODUCE.md section 13 defines, never
    # derived from a per-evaluation factor. The factor is what went wrong before: this
    # module reported `ev.n_sim`, which counts `evaluate()` calls and is neither of
    # section 13's units, and charged stage 1 at k x 2.00 measure_all, which the
    # measurement below shows is one short of what the rollout actually spends.
    from eqrl.simcount import counting

    mode_detail: dict[str, Any] = {"mode": mode}

    if mode == "thinking":
        # ---- N independent PPO rollouts, each closed by the frozen G3.2 -------------
        candidates = []
        c1 = {"measure_all": 0, "analysis": 0}
        c2 = {"measure_all": 0, "analysis": 0}
        for offset in THINKING_SEED_OFFSETS:
            with counting() as c1_i:
                xs_i, s1_i, term_i = fc.stage1_rollout(
                    evaluate, policy, env, spec_index + offset, target_boost_db,
                    channel_loss_db, k)
            for key in c1:
                c1[key] += c1_i[key]
            s1_only_i = [e for e in s1_i if e]
            with counting() as c2_i:
                g2_i, info_i, _xf, _rf, left_i = fc.g32_solve(
                    evaluate, xs_i, s1_i, target_boost_db, plane, ladder, THINKING_R,
                    stop_abs_err_db=THINKING_STOP_ABS_ERR_DB)
            for key in c2:
                c2[key] += c2_i[key]
            candidates.append({
                "seed_offset": offset, "xs": xs_i, "s1": s1_i, "s1_only": s1_only_i,
                "term_at": term_i, "g2": g2_i, "info": info_i, "left": left_i,
                "arm": fc.summarize(s1_only_i + g2_i, target_boost_db, tol)})

        def _rank(c):
            has_design = c["arm"]["best_design"] is not None
            err = c["arm"]["best_abs_err"]
            return (0 if has_design else 1, err if err is not None else float("inf"))

        winner = min(candidates, key=_rank)
        s1, s1_only, term_at = winner["s1"], winner["s1_only"], winner["term_at"]
        g2, info, left = winner["g2"], winner["info"], winner["left"]
        arm_b = winner["arm"]
        k_eff = k * len(THINKING_SEED_OFFSETS)
        mode_detail["rollouts"] = [
            {"seed_offset": c["seed_offset"], "best_abs_err_db": c["arm"]["best_abs_err"],
             "n_loose_pass": c["arm"]["n_loose_pass"],
             "reached_target": bool(c["info"]["reached_target"])}
            for c in candidates]
        mode_detail["winner_seed_offset"] = winner["seed_offset"]

    else:
        # ---- stage 1: the frozen PPO policy, k evaluations ---------------------------
        with counting() as c1:
            xs, s1, term_at = fc.stage1_rollout(evaluate, policy, env, spec_index,
                                                target_boost_db, channel_loss_db, k)
        s1_only = [e for e in s1 if e]
        k_eff = k

        # ---- stage 2: G3.2 constrained refinement, mode-dependent budget -------------
        if mode == "fastest":
            from eqrl.experiments.fastest_hedge import fastest_stage2, load_fastest_assets

            surrogate, corpus_X, radius = load_fastest_assets()
            with counting() as c2:
                g2, info, _x_f, _rec_f, left = fastest_stage2(
                    evaluate, xs, s1, target_boost_db, plane, ladder, FASTEST_BUDGET,
                    surrogate=surrogate, corpus_X=corpus_X, safety_radius=radius)
            mode_detail["hedge"] = info.get("hedge")
        else:
            r = ACCURATE_R if mode == "accurate" else fc.PREREG["r"]
            stop = ACCURATE_STOP_ABS_ERR_DB if mode == "accurate" else None
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
