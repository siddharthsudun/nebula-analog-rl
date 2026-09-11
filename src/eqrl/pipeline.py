"""SILQ's delivered architecture as one callable: a target specification in, a verified
circuit out.

    target specification
        |
    frozen seq_clean40k PPO          global feasibility search        (k evaluations)
        |
    G3.2 constrained refinement      specification closure            (up to r evaluations)
        |
    guarded nominal verification -> PVT sizing repair -> independent full-grid verification
        |
    final circuit + resulting specs

WHY THIS MODULE EXISTS. `results/delivered_circuit.json` was produced by arm B of
`experiments/final_comparison.py`, but the public entry point `eqrl.solve` ran PPO alone
and then fell back to a fixed hand-verified design. The polished system and the executable
demo therefore described two different architectures, and the demo could not reproduce the
delivered circuit. This module is the one place that architecture lives, and `eqrl.solve`
is now a thin front end over it.

NOMINAL STAGES REUSE EXISTING CONTROLLERS. Stage 1, the guarded evaluation, and G3.2 are IMPORTED
from `experiments.final_comparison` -- the same functions arm B runs, under the same
constants, checked by the same `_check_constants()` guard, and re-proved against the frozen
artifacts by that module's `--gate`. A second transcription of G3.2 would be a second
controller with the same name, which is exactly what the frozen record must not acquire.
Changes no reward, PPO hyperparameter, design bound, guard, hard_pass, controller constant,
frozen model or benchmark criterion. The additional PVT sizing stage lives in
`pvt_repair`; production defaults to this stage after nominal verification.
Use pvt=False only to explicitly reproduce the historical nominal benchmark.

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
MODES = ("default", "fastest", "thinking", "retarget", "auto", "g32_acceptance")

#: g32_acceptance: "thinking", with the user's ACCEPTANCE CONSTRAINTS applied to the corpus
#: pool that proposes restart designs, instead of only to the verdict at the end. It is a
#: separate mode and not a change to "thinking", so every published thinking number stands.
#:
#: IT IS INERT ON THE BENCHMARK, AND THE UI MUST SAY SO -- but "inert on the benchmark" is
#: not "inert at the defaults", and an earlier version of this comment conflated the two.
#: Measured on the 32 spec draws (`scratchpad/g32_acceptance_viability.py`): at the default
#: spec it removes ZERO rows from the 512-row near-target pool and changes ZERO of 32
#: selected seed sets, bit-identically. `area` cannot bite at the default ceiling at all --
#: the analytic supremum over the whole action space is 0.002227 mm^2 against 0.05 mm^2.
#:
#: `boost_range` HOWEVER CAN BITE AT THE DEFAULTS NEAR THE EDGES, because the pool is the
#: rows nearest the target IN BOOST and a target near a range boundary pulls in rows from
#: the far side of it. Scanned, not bisected -- a first pass bisected for a threshold and
#: reported [3.0664, 11.2125], and there is no threshold there to find. At 0.01 dB:
#:
#:      3.00-3.04 dB    seeds differ
#:      3.05-11.13 dB   seeds IDENTICAL, no exceptions across the whole interval
#:      11.14-12.00 dB  differ at 76 of 121 sampled targets, RAGGED -- 11.15-11.18, 11.20,
#:                      11.21, 11.27, 11.28, 11.30, 11.32 and 11.87 are still identical
#:
#: Removing pool rows does not have to move the selection: farthest-point can pick the same
#: designs from the smaller pool. At 3.05 dB the mask drops 76 rows and the seeds do not
#: change at all. So the quiet interval is a measured fact and the noisy region is genuinely
#: patchy; quoting a two-sided band would assert a monotonicity that was measured to fail.
#:
#: The 32 draws span 5.01-10.99 dB, wholly inside the quiet interval, which is why the sweep
#: could not see any of this. Where it does bite, the mode is a small correctness
#: improvement rather than nothing: it stops restarts being proposed from designs whose
#: recorded boost is outside the range the run will accept.
#:
#: So the UI copy must say it cannot change the result FOR A TARGET IN 3.05-11.13 dB AT THE
#: DEFAULT SPEC -- not that it is inert unconditionally, which would be false at the edges,
#: and not that it always helps outside, which would be false at 11.87.
#:
#: So it exists for the case the competition does not score: a user who tightens the boost
#: range or the area ceiling and wants the SEARCH to respect that, rather than discovering
#: at verification time that every restart was proposed from a region their constraint
#: excludes. Its comparison is against `_reselect_for_requirements` at
#: REQUIREMENT_RESELECT_CAP, under tightened requirements -- not against chance, and not at
#: the defaults, where both are provably the same number.
#:
#: NOT AN ESCALATION TARGET. `AUTO_ESCALATE` stays "thinking" deliberately: "auto" is the
#: mode for someone who has not read this file, and silently routing them into an arm whose
#: numbers are not the preregistered ones is exactly the substitution this project keeps
#: refusing to make elsewhere.
G32_ACCEPTANCE_BASE = "thinking"

#: Auto: run AUTO_FIRST, and only if that attempt did not come back SOLVED, run
#: AUTO_ESCALATE. Nothing new is searched -- it is the two frozen modes in sequence. What
#: follows is read off `_auto` below, which is the whole mechanism:
#:
#:   * "fastest" runs first, with this call's model / spec_index / tol / probe paths and
#:     the caller's `requirements` passed through unchanged.
#:   * The escalation test is exactly `first["status"] == SOLVED`, so UNSOLVED,
#:     CLOSED_NOT_VERIFIED and FALLBACK all escalate. SOLVED already means the independent
#:     ten-check verification passed, so a verified answer is never re-searched.
#:   * The escalation is a FULL, independent "thinking" run. Nothing from the first attempt
#:     warms it up -- no start, no design, no trace is carried over. Only the first
#:     attempt's status, guidance, best boost and cost are kept, under `result["auto"]`.
#:   * On escalation the request pays for both runs. `measure_all_total` and
#:     `spice_analyses_total` fold the first attempt in; `optimizer_evals` deliberately
#:     does not, and reports the escalated run alone with the first attempt's figure beside
#:     it as `optimizer_evals_prior_attempts`.
#:   * Only the TOP-LEVEL `mode` is relabelled "auto". `provenance.mode` and
#:     `provenance.mode_detail` still name the mode that actually searched.
#:
#: THE ROUTING RULE RESTS ON NO COMMITTED MEASUREMENT, and this is worse than the numbers
#: merely being stale. An earlier version of this comment justified the ordering with an
#: n=32 sweep -- "fastest 22/32, thinking 30/32" -- read out of
#: `results/mode_sweep_seed99.json`. THAT FILE WAS NEVER COMMITTED TO THIS REPOSITORY. It
#: appears in no commit on any branch; it exists only as an untracked local file, so no
#: reader of this tree can check the claim it carried. The numbers are therefore withdrawn
#: rather than restated. Both arms of that comparison were superseded anyway: `30a90901f`
#: replaced "fastest"'s stage 1 outright (PPO rollout -> corpus lookup plus one real
#: evaluation), and `4b9863d05` gave "thinking" 8 rollouts, r=25, a 0.01 dB stop and
#: corpus-proposed restarts in place of 3 rollouts at r=15. Section 4.3 of
#: `docs/RESULTS_INFERENCE_MODES_V2.md` carries the warning banner for that sweep, and its
#: own item 2 separately retracts the comparison those numbers were used to license.
#:
#: What survives is a structural argument, not evidence. By construction auto verifies on a
#: SUPERSET of the specs "fastest" verifies on, because it escalates on exactly the ones
#: "fastest" did not verify. The cost trade is genuinely two-sided: auto is cheaper than
#: "thinking" on every spec "fastest" verifies, and strictly MORE expensive than "thinking"
#: on every spec it escalates on, since the first attempt is paid for and then discarded.
#: Whether that trade is favourable on average depends on how often "fastest" verifies --
#: which is exactly the quantity the withdrawn numbers claimed to supply and which nothing
#: in this tree currently measures. A committed 32-spec sweep of the shipped modes is owed;
#: until one lands, quote no solve rate and no cost saving for this mode.
AUTO_FIRST, AUTO_ESCALATE = "fastest", "thinking"

#: Acceptance constraints a caller may set alongside the target and channel. These are
#: exactly the fields `hard_pass` reads, so setting one changes what "verified" means for
#: this run and NOTHING else: the guard evaluator still runs on DEFAULT_SPEC (device
#: physics is not the user's to relax), and the frozen search still steers on boost.
#: Every result records the full diff against the competition specification, and when a
#: requirement was relaxed the competition verdict is reported alongside the user one so
#: a loosened bar can never be mistaken for the benchmark's.
REQUIREMENT_FIELDS = ("power_w_max", "noise_vrms_max", "hd3_db_max", "area_mm2_max",
                      "eye_h_ui_min", "eye_v_mv_min", "boost_db_min", "boost_db_max",
                      "peak_freq_lo_ghz", "peak_freq_hi_ghz", "dc_gain_db_min")

#: For each requirement, the direction that makes it STRICTER than the default: a lower
#: ceiling or a higher floor. Used to label a user's value "tighter" or "looser".
_TIGHTER_IS_LOWER = frozenset({"power_w_max", "noise_vrms_max", "hd3_db_max",
                               "area_mm2_max", "boost_db_max", "peak_freq_hi_ghz"})

#: How many alternative designs from the search trace the requirement-aware selection
#: may re-verify. Each costs one measure_all. Bounded so a user constraint the search did
#: not steer on cannot turn a five-evaluation run into a fifty-evaluation one.
REQUIREMENT_RESELECT_CAP = 4

#: How many alternative designs from the same search trace get a full, independent
#: verify() call so result["alternatives"] can offer real tradeoffs, not estimates.
#: Unlike REQUIREMENT_RESELECT_CAP this runs on EVERY solved request, not only on a
#: failure, so it is deliberately smaller.
ALTERNATIVES_CAP = 3

#: Optional progress hook: `notify(stage, text)`. The dashboard installs one so events
#: that happen INSIDE design() (auto-mode escalation, requirement re-selection) reach
#: the live trace. None means silent. Never affects what is computed.
notify = None


def _note(stage: str, text: str) -> None:
    if notify is not None:
        notify(stage, text)

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

#: Search-depth governor, in `measure_all` units. NOT a wall-clock guarantee, and an
#: earlier version of this comment claimed one -- corrected 06 Sep 2026. The ~0.55 s per
#: `measure_all` in DESIGN_MODES_V2 section 0 is an IDLE machine; measured under the
#: contention this repo actually creates (a bench and the dashboard both driving ngspice)
#: it is ~2.6 s, so 100 measure_all was ~268 s on spec 10, not the ~55 s first claimed.
#: The cap is kept in measure_all rather than seconds on purpose: a deterministic budget
#: makes the search reproducible, where a wall-clock stop would make its depth depend on
#: the machine. Without
#: this, 8 rollouts x (5 + 25) is 240 measure_all -- over two minutes -- in the worst case,
#: which is the case where nothing is working and the user is waiting longest. No restart
#: is STARTED once the spend reaches this; a restart already running is always finished, so
#: this trims the search rather than truncating a solve into a misleading partial answer.
THINKING_MEASURE_ALL_BUDGET = 100

#: Corner-robustness tiebreak. Thinking finishes up to 8 candidates and reports one; until
#: now it picked purely on TT accuracy. Measured 06 Sep 2026 over the 84 designs already in
#: results/pvt_*.json (`scratchpad/dcgain_proxy.py`, no new simulation), against the number
#: of corners whose HARD DEVICE GUARDS hold -- 82% of the frozen pool's failing corners:
#:
#:     predictor                 all corners passed      guard-valid corners
#:     TT abs_err                 -0.65 / -0.54           -0.21 / -0.12
#:     TT DC-gain headroom        +0.27 / +0.24           +0.60 / +0.59   (pearson/spearman)
#:
#: So error ranks spec drift and headroom ranks device robustness, and the old key saw only
#: the first. The first column runs the other way because `boost_target` is itself one of
#: the ten corner checks, which makes a high-error design fail corners almost by
#: construction -- so trading real accuracy for headroom would lose more than it wins.
#:
#: Hence a BAND, not a replacement: only candidates within this many dB of the most
#: accurate one are treated as interchangeable, and headroom decides among those. The value
#: is not fitted -- it is `final_comparison.PREREG["stop_abs_err_db"]`, the distance below
#: which the frozen solver itself stops distinguishing outcomes. No headroom threshold is
#: applied: the quartile data suggests a cliff near 0.9 dB, but fitting a cutoff on the same
#: 84 points that motivated the rule is exactly the trap `g32_headroom.py` warns about.
THINKING_TIEBREAK_BAND_DB = 0.25

#: Retarget: DEFAULT, plus one thing -- when G3.2's boost axis is pinned at a bound, probe
#: the other axes and resume the line search along the best one
#: (`eqrl.experiments.axis_retarget`). Budget and stopping tolerance are deliberately left
#: at Default's own `fc.PREREG` values so that ANY difference between this mode and
#: "default" is attributable to the second axis and to nothing else. That is what makes
#: this arm measurable; do not "improve" it by also raising r.
#:
#: MEASURED, INERT, AND KEPT ANYWAY -- recorded here so the fact does not live only in a
#: commit message. Measured in `437412e96` at n=32 on spec-seed 137, a FRESH draw: the n=8
#: result that motivated this arm came from a set that contained the known pinned case and
#: so was selection-biased. On that draw the arm FIRED ON 0 OF 32 SPECS. Re-derived from
#: the committed rows in `scratchpad/retarget_ab32.json`: 0/32 fired, 32/32 identical
#: status and abs_err against "default", and a measure_all delta of exactly 0 on every one
#: of the 32. "no admissible step remains" -- the only exit this arm acts on -- does not
#: occur once in the file. (The commit message's bucket table quotes 6 budget-exhausted
#: and 3 wall specs; the committed rows say 4 and 5. Trust the rows.)
#:
#: It stays because it is free -- 32/32 identical outputs at +0 measure_all cannot regress
#: anything -- and because it is budget-starved rather than merely useless: specs that exit
#: "budget exhausted" never get far enough for a second axis to be tried at all. Whether
#: raising r would make the pinned exit appear is UNTESTED and must not be assumed.
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


def clean_requirements(requirements: dict | None) -> dict[str, float]:
    """Validate a caller's acceptance constraints. Unknown fields are an error, not a
    silent drop: a constraint the caller believes is in force but is not would be the
    quietest possible way to mislead them."""
    if not requirements:
        return {}
    bad = sorted(set(requirements) - set(REQUIREMENT_FIELDS))
    if bad:
        raise ValueError(f"unknown requirement field(s) {bad}; "
                         f"settable fields are {REQUIREMENT_FIELDS}")
    out = {k: float(v) for k, v in requirements.items() if v is not None}
    if "dc_gain_db_min" in out and out["dc_gain_db_min"] < DEFAULT_SPEC.dc_gain_db_min:
        # The guard's own floor (guards.DC_GAIN_DB_MIN) rejects any design below 0 dB
        # before hard_pass ever sees it, so a lower floor here would be a number with no
        # effect. It is clamped rather than refused so the request still runs.
        out["dc_gain_db_min"] = DEFAULT_SPEC.dc_gain_db_min
    return out


def requirements_diff(spec: Spec) -> list[dict[str, Any]]:
    """Every acceptance constraint on `spec` that differs from the competition default,
    labelled by whether the user made it tighter or looser."""
    out = []
    for k in REQUIREMENT_FIELDS:
        v, d = getattr(spec, k), getattr(DEFAULT_SPEC, k)
        if v is None or d is None or abs(float(v) - float(d)) <= 1e-12:
            continue
        lower = float(v) < float(d)
        out.append({"field": k, "value": float(v), "default": float(d),
                    "direction": ("tighter" if lower == (k in _TIGHTER_IS_LOWER)
                                  else "looser")})
    return out


def spec_for(target_boost_db: float, channel_loss_db: float, tol: float,
             requirements: dict | None = None) -> Spec:
    """The requirement set the result is verified against.

    DEFAULT_SPEC plus this run's target and channel, with `boost_target_tol_db` ON. That
    tenth check is what `results/delivered_circuit.json` was scored on, and what makes
    "meets the specification" mean the REQUESTED boost rather than anywhere in 3-12 dB.
    `dc_gain_db_min` is already 0.0 in DEFAULT_SPEC and is left alone.

    `requirements` (see REQUIREMENT_FIELDS) overrides individual acceptance constraints.
    Without it the result is the competition specification exactly.
    """
    return dataclasses.replace(DEFAULT_SPEC, target_boost_db=target_boost_db,
                               channel_loss_db=channel_loss_db,
                               boost_target_tol_db=tol,
                               **clean_requirements(requirements))


def competition_spec(spec: Spec) -> Spec:
    """The same target, channel and tolerance with every acceptance constraint at the
    competition default. What `spec` would have been with no user requirements."""
    return dataclasses.replace(DEFAULT_SPEC, target_boost_db=spec.target_boost_db,
                               channel_loss_db=spec.channel_loss_db,
                               boost_target_tol_db=spec.boost_target_tol_db)


def verify(dv: DesignVars, spec: Spec) -> dict[str, Any]:
    """Re-measure one design through a fresh guarded evaluator on all ten checks.

    Independent of the search: a new evaluator, a new measurement, no cached record. The
    guard runs first, so a design that is not a real circuit fails here even if every
    number it reported looked good.

    When `spec` carries user requirements that differ from the competition defaults, the
    same measurement is ALSO scored against the competition specification and reported
    under `competition_*`, so a relaxed bar and the benchmark's bar are never conflated.
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
    out = {"guard_valid": True, "guard_check": None, "passed": bool(ok),
           "checks": checks, "failing": [c for c, good in checks.items() if not good],
           "measures": m.as_dict(),
           "abs_err_db": abs(m.boost_db - spec.target_boost_db)}
    if requirements_diff(spec):
        cok, cchecks = hard_pass(m, competition_spec(spec))
        out["competition_passed"] = bool(cok)
        out["competition_checks"] = cchecks
        out["competition_failing"] = [c for c, good in cchecks.items() if not good]
    return out


def _cost(fc, k: int, info: dict, left: int, c1: dict, c2: dict, *, corpus_seed=False) -> dict[str, Any]:
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
    charged = (k * (1 if corpus_seed else fc.PREREG["ppo_measure_all_per_eval"])
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


def _auto(target_boost_db, channel_loss_db, kw) -> dict[str, Any]:
    """AUTO_FIRST, then AUTO_ESCALATE only if the first attempt was not SOLVED."""
    _note("search", "Auto mode: trying the corpus-seeded fast search first.")
    first = design(target_boost_db, channel_loss_db, mode=AUTO_FIRST, pvt=False, **kw)
    if first["status"] == SOLVED:
        first["auto"] = {"escalated": False, "first_mode": AUTO_FIRST,
                         "first_status": SOLVED}
        first["mode"] = "auto"
        return first
    why = first["guidance"]["headline"]
    _note("search", f"Fast search did not verify ({why}). Escalating to Thinking: "
                    f"multiple independent restarts with a larger budget.")
    final = design(target_boost_db, channel_loss_db, mode=AUTO_ESCALATE, pvt=False, **kw)
    fc1 = first["cost"]
    final["auto"] = {
        "escalated": True, "first_mode": AUTO_FIRST, "first_status": first["status"],
        "first_guidance": first["guidance"],
        "first_best_boost_db": first["solver"].get("best_boost_db"),
        "first_cost": {"optimizer_evals": fc1["optimizer_evals"],
                       "measure_all_total": fc1["measure_all_total"],
                       "spice_analyses_total": fc1["spice_analyses_total"]},
        "escalated_mode": AUTO_ESCALATE,
    }
    # The first attempt's simulator work is real work this request paid for. It is added
    # to the totals under its own key so the escalated run's own figures stay comparable
    # to a plain Thinking run.
    c = final["cost"]
    c["measure_all_prior_attempts"] = fc1["measure_all_total"]
    c["spice_analyses_prior_attempts"] = fc1["spice_analyses_total"]
    c["optimizer_evals_prior_attempts"] = fc1["optimizer_evals"]
    c["measure_all_total"] += fc1["measure_all_total"]
    c["spice_analyses_total"] += fc1["spice_analyses_total"]
    final["mode"] = "auto"
    return final


def _reselect_for_requirements(result: dict, trace: list, spec: Spec, tol: float,
                               counting) -> None:
    """If the most accurate design fails a USER requirement the search never steered on,
    re-verify the next most accurate guard-valid designs from the same trace and take the
    first that passes. Bounded by REQUIREMENT_RESELECT_CAP; every extra measurement is
    counted and reported. Mutates `result` in place."""
    v = result["verification"]
    if v["passed"] or not v.get("guard_valid"):
        return
    # Only a failure on a check the user tightened is a reason to look again. A failure
    # on the boost target or a competition default is the search's verdict, not a
    # selection problem.
    diff = {d["field"] for d in requirements_diff(spec)}
    check_of = {"power_w_max": "power", "noise_vrms_max": "noise", "hd3_db_max": "hd3",
                "area_mm2_max": "area", "eye_h_ui_min": "eye_h", "eye_v_mv_min": "eye_v",
                "boost_db_min": "boost_range", "boost_db_max": "boost_range",
                "peak_freq_lo_ghz": "peak_in_band", "peak_freq_hi_ghz": "peak_in_band",
                "dc_gain_db_min": "dc_gain"}
    user_checks = {check_of[f] for f in diff}
    if not set(v.get("failing") or []) <= user_checks:
        return
    from eqrl.request_budget import active
    if active.get() is not None:
        result["_pareto_trace"] = trace
        return
    chosen = result["design"]
    pool = [e for e in trace if e and e["loose_pass"] and e.get("design")
            and e["design"] != chosen
            and abs(e["boost_db"] - spec.target_boost_db) <= tol]
    pool.sort(key=lambda e: abs(e["boost_db"] - spec.target_boost_db))
    tried = []
    for e in pool[:REQUIREMENT_RESELECT_CAP]:
        dv = DesignVars(**e["design"])
        _note("verify", f"Best design fails your {', '.join(sorted(user_checks))} "
                        f"requirement; re-verifying the next candidate "
                        f"({e['boost_db']:.2f} dB).")
        with counting() as c:
            v2 = verify(dv, spec)
        _add_verification_cost_extra(result["cost"], c)
        tried.append({"boost_db": e["boost_db"], "passed": bool(v2["passed"]),
                      "failing": v2.get("failing")})
        if v2["passed"]:
            result["design"] = dict(e["design"])
            result["verification"] = v2
            result["solver"] = dict(result["solver"],
                                    best_design=dict(e["design"]),
                                    best_boost_db=e["boost_db"],
                                    best_abs_err=abs(e["boost_db"] - spec.target_boost_db))
            break
    result["provenance"]["requirement_reselection"] = {
        "reason": sorted(v.get("failing") or []), "tried": tried,
        "applied": bool(tried) and tried[-1]["passed"]}


def _add_verification_cost_extra(cost: dict[str, Any], c: dict) -> None:
    cost["measure_all_verification"] += c["measure_all"]
    cost["measure_all_total"] += c["measure_all"]
    cost["spice_analyses_verification"] += c["analysis"]
    cost["spice_analyses_total"] += c["analysis"]


def _add_alternatives_cost(cost: dict[str, Any], c: dict) -> None:
    cost["measure_all_alternatives"] = cost.get("measure_all_alternatives", 0) + c["measure_all"]
    cost["measure_all_total"] += c["measure_all"]
    cost["spice_analyses_alternatives"] = cost.get("spice_analyses_alternatives", 0) + c["analysis"]
    cost["spice_analyses_total"] += c["analysis"]


def _attach_alternatives(result: dict, trace: list, spec: Spec, tol: float, counting) -> None:
    """After a SOLVED result, independently verify up to ALTERNATIVES_CAP other
    guard-valid designs from the same search trace, so the result can offer real
    tradeoffs instead of exactly one circuit. Nominal (pre-PVT) only -- PVT repair runs
    once, in design(), after this returns, and is never re-run per alternative. A no-op
    unless result["status"] == SOLVED. Mutates `result` in place."""
    if result["status"] != SOLVED:
        return
    chosen = result["design"]
    pool = [e for e in trace if e and e["loose_pass"] and e.get("design")
            and e["design"] != chosen
            and abs(e["boost_db"] - spec.target_boost_db) <= tol]
    pool.sort(key=lambda e: abs(e["boost_db"] - spec.target_boost_db))

    cost = result["cost"]
    cost.setdefault("measure_all_alternatives", 0)
    cost.setdefault("spice_analyses_alternatives", 0)

    tried, items = [], []
    for e in pool[:ALTERNATIVES_CAP]:
        dv = DesignVars(**e["design"])
        with counting() as c:
            v2 = verify(dv, spec)
        _add_alternatives_cost(cost, c)
        tried.append({"boost_db": e["boost_db"], "passed": bool(v2["passed"]),
                      "failing": v2.get("failing")})
        if v2["passed"]:
            items.append({"design": dict(e["design"]), "boost_db": e["boost_db"],
                          "abs_err_db": abs(e["boost_db"] - spec.target_boost_db),
                          "verification": v2})

    result["alternatives"] = {
        "cap": ALTERNATIVES_CAP,
        "pool_size": len(pool),
        "tried": tried,
        "items": items,
        "scope": "nominal (pre-PVT) designs only; not run through PVT repair",
    }


def _design_nominal(target_boost_db: float, channel_loss_db: float = DEFAULT_SPEC.channel_loss_db,
           *, model: str = POLICY, spec_index: int = 0, tol: float | None = None,
           peak_probe: str = PEAK_PROBE, rescue_probe: str = RESCUE_PROBE,
           allow_fallback: bool = False, mode: str = "default",
           requirements: dict | None = None) -> dict[str, Any]:
    """Run the delivered architecture for one specification.

    `requirements` optionally overrides acceptance constraints (REQUIREMENT_FIELDS). The
    result's `spec.requirements` lists every override with its direction, and
    `spec.scored_against` says in words whether the verdict is the competition
    specification's or a user-modified one.

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
    requirements = clean_requirements(requirements)
    if mode == "auto":
        return _auto(target_boost_db, channel_loss_db, dict(
            model=model, spec_index=spec_index, tol=tol, peak_probe=peak_probe,
            rescue_probe=rescue_probe, allow_fallback=allow_fallback,
            requirements=requirements))
    fc = _fc()
    fc._check_constants()
    k = fc.PREREG["k"]
    tol = fc.PREREG["tol"] if tol is None else tol
    user_spec = spec_for(target_boost_db, channel_loss_db, tol, requirements)
    req_diff = requirements_diff(user_spec)

    plane = fc.plane_from_probe(peak_probe)
    ladder = fc.rescue_order_from_probe(rescue_probe)

    ev = fc.Evaluation()
    evaluate = ev.make_eval(channel_loss_db)
    from eqrl.request_budget import active
    if active.get() is not None:
        evaluate = active.get().bind(evaluate)
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

    # g32_acceptance shares this branch entirely -- see G32_ACCEPTANCE_BASE. The two differ
    # at exactly one line, the diverse_seeds call below, and nowhere else.
    if mode in ("thinking", "g32_acceptance"):
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
                # THE ONE LINE THAT DIFFERS BETWEEN "thinking" AND "g32_acceptance".
                # "thinking" passes the COMPETITION spec here, deliberately: its published
                # numbers must not move when a user sets a requirement, so its restarts are
                # proposed from the same pool on every run. "g32_acceptance" passes
                # `user_spec` and turns the filter on, so a tightened boost range or area
                # ceiling narrows the pool the restarts are drawn from instead of only
                # rejecting them at verification. Both still hand every proposal to the
                # same guarded evaluator -- the corpus decides where to look, never what
                # passes.
                accept = mode == "g32_acceptance"
                seeds = diverse_seeds(
                    target_boost_db, surrogate,
                    user_spec if accept else spec_for(target_boost_db, channel_loss_db, tol),
                    THINKING_SURROGATE_STARTS, exclude=tried, acceptance=accept)
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

        # ---- winner selection: accuracy first, then corner robustness ---------------
        # See THINKING_TIEBREAK_BAND_DB for the measurement this rests on. Accuracy is
        # still the primary key; headroom only decides between candidates the frozen
        # solver's own stopping tolerance would not distinguish.
        from eqrl.guards import DC_GAIN_DB_MIN

        def _headroom(c):
            """DC-gain headroom, in dB, of the design this candidate would report.

            Picks the same record `fc.summarize` does -- min |boost - target| over the
            loose passes of the same trace -- restated here only because summarize is
            frozen and carries no dc_gain_db. Headroom is the guard's own bound and not a
            fitted quantity (`eqrl.experiments.g32_headroom`): DC_GAIN_DB_MIN is 0.0 dB.
            """
            ok = [e for e in c["s1_only"] + c["g2"] if e and e["loose_pass"]]
            if not ok:
                return None
            best = min(ok, key=lambda e: abs(e["boost_db"] - target_boost_db))
            return float(best["dc_gain_db"]) - DC_GAIN_DB_MIN

        def _rank(c):
            has_design = c["arm"]["best_design"] is not None
            err = c["arm"]["best_abs_err"]
            return (0 if has_design else 1, err if err is not None else float("inf"))

        for c in candidates:
            c["headroom_db"] = _headroom(c)

        most_accurate = min(candidates, key=_rank)
        best_err = most_accurate["arm"]["best_abs_err"]
        band = [] if best_err is None else [
            c for c in candidates
            if c["arm"]["best_design"] is not None
            and c["arm"]["best_abs_err"] <= best_err + THINKING_TIEBREAK_BAND_DB]
        # CAN THIS BRANCH EVER CHANGE AN OUTCOME? YES -- established by reading the code,
        # not assumed, and stated here because "this arm is inert" is the kind of claim
        # this file keeps having to retract. `most_accurate` is always a member of `band`
        # whenever it has a design (its own error trivially satisfies the bound), so the
        # branch fires exactly when a SECOND candidate carries a design within
        # THINKING_TIEBREAK_BAND_DB and beats it on headroom. That state is reachable:
        # `candidates` only grows past one when no earlier restart reached target, and
        # independent restarts land on different designs with different `dc_gain_db`. The
        # winner then supplies `design`, `netlist`, `info` and `guidance`, so a swap is
        # fully observable in the result.
        #
        # Two bounds on a swap, both read off the code rather than measured:
        #   * Every band member has a real headroom. `fc.summarize` returns `best_design`
        #     None exactly when the trace holds no `loose_pass`, and `_headroom` scans the
        #     same records, so band membership implies `headroom_db` is not None. The
        #     float("-inf") fallback below is unreachable for a band member and is there
        #     only to keep the sort key total.
        #   * A swap cannot by itself flip the `boost_target` check: the band is 0.25 dB
        #     and `hard_pass` scores that check at `fc.PREREG["tol"]`, which is 1.5 dB. It
        #     CAN change every other reported measure, the netlist, and
        #     `g32_reached_target` -- band membership does not test `reached_target`, so a
        #     candidate that stopped short may displace one that reached target.
        #
        # HOW OFTEN IT ACTUALLY FIRES IS NOT MEASURED. No committed artifact in this tree
        # records a firing rate for this rule; `mode_detail["tiebreak"]` below exists to
        # collect one. Do not describe this branch as a no-op -- the code does not say so.
        if len(band) > 1:
            # Highest headroom wins; a candidate with no headroom to report cannot win the
            # tiebreak, and remaining ties fall back to the more accurate one.
            winner = max(band, key=lambda c: (
                c["headroom_db"] if c["headroom_db"] is not None else float("-inf"),
                -c["arm"]["best_abs_err"]))
        else:
            winner = most_accurate
        tiebreak_applied = winner is not most_accurate
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
             "headroom_db": c["headroom_db"],
             "reached_target": bool(c["info"]["reached_target"])}
            for c in candidates]
        mode_detail["winner_start"] = winner["start"]
        # Recorded so the tiebreak can be audited rather than assumed: how often it fires
        # at all, and what accuracy it gave up when it did. The `retarget` arm fired 0/32
        # on a fresh seed; nothing here should be trusted until this says otherwise.
        mode_detail["tiebreak"] = {
            "band_db": THINKING_TIEBREAK_BAND_DB,
            "n_in_band": len(band),
            "applied": tiebreak_applied,
            "winner_headroom_db": winner["headroom_db"],
            "most_accurate_start": most_accurate["start"],
            "most_accurate_headroom_db": most_accurate["headroom_db"],
            # The design the tiebreak DISPLACED, kept only when it actually displaced one.
            # Without it the firing cannot be audited after the fact -- the corner sweep
            # that decides whether this rule helps needs both sides of the swap, and
            # re-deriving the loser would mean re-running the whole search.
            "most_accurate_design": (most_accurate["arm"]["best_design"]
                                     if tiebreak_applied else None),
            "abs_err_given_up_db": (
                None if not tiebreak_applied or best_err is None
                else winner["arm"]["best_abs_err"] - best_err)}
        mode_detail["n_ppo_rollouts"] = n_ppo
        mode_detail["n_surrogate_starts"] = n_surrogate
        mode_detail["measure_all_spent"] = _spent()
        mode_detail["governor_stopped"] = governor_stopped
        mode_detail["adaptive_stopped_early"] = (
            n_ppo < min(THINKING_ROLLOUTS, len(THINKING_SEED_OFFSETS))
            and not governor_stopped)
        if mode == "g32_acceptance":
            # SELF-REPORTING, because this mode's honest answer is usually "I did nothing".
            # `pool_narrowed` is False whenever the acceptance filter admitted every row
            # the unfiltered pool would have, which at the competition defaults is always.
            # Recording it means a result can state its own inertness instead of leaving a
            # reader to infer that a mode ran and mattered because it was selected.
            mode_detail["acceptance"] = {
                "applied_to": "restart seed pool (thinking_starts.diverse_seeds)",
                "verification_path": "UNCHANGED -- same guarded evaluator and hard_pass",
                "requirements_set": sorted(req_diff),
                "inert_at_defaults": not req_diff,
                "note": ("no requirement was tightened, so this run is bit-identical to "
                         "thinking" if not req_diff else
                         "requirements tightened; restart pool was filtered before "
                         "farthest-point selection"),
            }

    elif mode == "fastest":
        # ---- stage 1: corpus lookup, bounded retries on guard-invalid seeds ---------
        # See eqrl.experiments.fastest_hedge.surrogate_stage1's docstring for why this is
        # sound with no channel awareness in the corpus. Every attempted seed is
        # actually measured; a guard-valid first seed ends this stage immediately.
        from eqrl.experiments.fastest_hedge import load_fastest_assets, surrogate_stage1

        surrogate, _corpus_X, _radius = load_fastest_assets()
        seed_spec = spec_for(target_boost_db, channel_loss_db, tol)
        with counting() as c1:
            xs, s1, term_at = surrogate_stage1(evaluate, target_boost_db, surrogate,
                                               seed_spec)
        s1_only = [e for e in s1 if e]
        # 1 real evaluation ordinarily, up to 4 when rejected seeds require another
        # distinct corpus
        # seed -- either way this is len(xs), not PREREG["k"]=5.
        # `optimizer_evals`/`stage1_evals` must say so honestly.
        k_eff = len(xs)

        # ---- stage 2: the SAME frozen g32_solve, fixed small budget, no floor --------
        with counting() as c2:
            g2, info, _x_f, _rec_f, left = fc.g32_solve(
                evaluate, xs, s1, target_boost_db, plane, ladder, FASTEST_BUDGET)
        seed_rec = s1[0]
        mode_detail["surrogate_seed"] = {
            "attempts": len(xs),
            "guard_valid_attempts": sum(e is not None for e in s1),
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
            # Called exactly as the benchmark calls it: no stop kwarg, so g32_solve reads
            # fc.PREREG itself and "default" stays byte-identical to the frozen record.
            with counting() as c2:
                g2, info, _x_f, _rec_f, left = fc.g32_solve(
                    evaluate, xs, s1, target_boost_db, plane, ladder, fc.PREREG["r"])

        arm_b = fc.summarize(s1_only + g2, target_boost_db, tol)

    stage1 = fc.summarize(s1_only, target_boost_db, tol)

    result: dict[str, Any] = {
        "architecture": "PPO (global feasibility) -> G3.2 constrained refinement",
        "mode": mode,
        "spec": {"target_boost_db": target_boost_db, "channel_loss_db": channel_loss_db,
                 "boost_tol_db": tol, "spec_index": spec_index,
                 "requirement_set": ("eqrl.specs.Spec defaults + this target/channel; "
                                     "dc_gain_db_min=0.0 and boost_target_tol_db both ON"
                                     if not req_diff else
                                     "USER-MODIFIED: this target/channel plus %d "
                                     "acceptance constraint(s) changed from the "
                                     "competition defaults; see `requirements`"
                                     % len(req_diff)),
                 "requirements": req_diff,
                 "scored_against": ("the competition specification" if not req_diff
                                    else "user-modified requirements, not the "
                                         "competition specification")},
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
        "cost": _cost(fc, k_eff, info, left, c1, c2, corpus_seed=mode == "fastest"),
        "solver": arm_b,
        "design": arm_b["best_design"],
        "netlist": None,
        "verification": None,
        "status": UNSOLVED,
    }

    # ---- independent verification -----------------------------------------------------
    if arm_b["best_design"] is not None:
        dv = DesignVars(**arm_b["best_design"])
        spec = user_spec
        with counting() as c3:
            v = verify(dv, spec)
        result["verification"] = v
        _add_verification_cost(result["cost"], c3)
        if req_diff:
            _reselect_for_requirements(result, s1_only + g2, spec, tol, counting)
            v = result["verification"]
            dv = DesignVars(**result["design"])
        result["netlist"] = netlist(dv, vdd=spec.vdd_nominal, temp_c=27.0, corner="tt",
                                    analysis="ac", models="sky130")
        result["status"] = SOLVED if v["passed"] else CLOSED_NOT_VERIFIED
        _attach_alternatives(result, s1_only + g2, spec, tol, counting)
        return _plain(result)

    # ---- nothing guard-valid came out of the architecture -----------------------------
    if allow_fallback:
        from eqrl.baselines.robust import robust_design

        dv = robust_design()
        spec = user_spec
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


def design(target_boost_db: float, channel_loss_db: float = DEFAULT_SPEC.channel_loss_db,
           *, model: str = POLICY, spec_index: int = 0, tol: float | None = None,
           peak_probe: str = PEAK_PROBE, rescue_probe: str = RESCUE_PROBE,
           allow_fallback: bool = False, mode: str = "default", requirements: dict | None = None,
           pvt: bool = True, pvt_output=None, pvt_wall_seconds: float = 900,
           wall_seconds: float | None = None, checkpoint=None) -> dict[str, Any]:
    """PPO/G3.2 followed by automatic bounded PVT repair and independent acceptance.

    pvt=False retains the historical nominal benchmark path explicitly. Production
    requests use PVT by default; failures cannot report a solved circuit.
    """
    if wall_seconds is not None:
        from eqrl.realtime import design_realtime
        return design_realtime(target_boost_db, channel_loss_db, mode=mode, requirements=requirements,
            model=model, spec_index=spec_index, tol=tol, peak_probe=peak_probe, rescue_probe=rescue_probe,
            allow_fallback=allow_fallback, wall_seconds=wall_seconds, checkpoint=checkpoint)
    result = _design_nominal(target_boost_db, channel_loss_db, model=model, spec_index=spec_index,
        tol=tol, peak_probe=peak_probe, rescue_probe=rescue_probe, allow_fallback=allow_fallback,
        mode=mode, requirements=requirements)
    if pvt:
        from eqrl.pvt_repair import apply_pvt_stage
        spec = spec_for(target_boost_db, channel_loss_db, result['spec']['boost_tol_db'], requirements)
        apply_pvt_stage(result, spec, output=pvt_output, seed=20260910+spec_index,
                        wall_seconds=pvt_wall_seconds, notify=_note)
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
        out.append("  cost          %d nominal optimizer evals; %d measure_all "
                   "(%d nominal search + %d nominal verification + %d PVT) = %d SPICE analyses"
                   % (c["optimizer_evals"], c["measure_all_total"],
                      c["measure_all_search"], c["measure_all_verification"], c.get("measure_all_pvt", 0),
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
    reqs = r.get("spec", {}).get("requirements") or []
    if reqs:
        out.append("  requirements  USER-MODIFIED (%d): %s"
                   % (len(reqs), ", ".join("%s=%g (%s)" % (q["field"], q["value"],
                                                          q["direction"]) for q in reqs)))
        if v and "competition_passed" in v:
            out.append("  competition   %s" % ("PASSES the unmodified specification too"
                                               if v["competition_passed"] else
                                               "FAILS the unmodified specification: "
                                               + ", ".join(v["competition_failing"])))
    auto = r.get("auto")
    if auto:
        out.append("  auto          %s" % (
            "fastest verified on the first attempt" if not auto["escalated"] else
            "fastest returned %s; escalated to thinking" % auto["first_status"]))
    if r.get("pvt"):
        pvt = r["pvt"]
        out.append("  PVT           %s; %d corner evaluations; independent 45-corner acceptance: %s" %
                   (pvt["status"], pvt["evaluations"], pvt["accepted"]))
        if r["provenance"].get("fixed_anchor_reused"):
            out.append("  final source  fixed delivered sizing, reverified for this specification")
    out.append("  status        %s" % r["status"])
    return "\n".join(out)
