"""Qualify the delivered SILQ circuit: margins, binding constraint, failure radius.

This module MEASURES the already-frozen delivered circuit and OPTIMIZES NOTHING. It does
not train, does not touch a reward, a design bound, or a guard threshold, and it never
changes results/delivered_circuit.json. It reads that circuit, runs it and small
perturbations of it back through the SAME guard the rest of the project uses
(eqrl.evaluator.build_evaluator), and writes results/qualification_delivered.json -- an
engineering "passport" for the delivered design.

It answers three questions the optimizer itself never exposes:

    1. MARGINS        For each of the 8 hard specs, how far is the delivered circuit from
                      failing that spec? Which one is binding (smallest margin)? This is a
                      presentation of the frozen numbers against the transcribed limits --
                      it should corroborate what pvt_signoff already found (DC gain).

    2. FAILURE RADIUS Part A. One parameter at a time, sweep the delivered value from -20%
                      to +20% and record the first % step at which the guard rejects the
                      circuit or a spec crosses its limit, and why. The smallest such radius
                      over all parameters is the "empirical local failure radius". This is
                      a LOCAL, one-parameter-at-a-time probe -- not a yield estimate.

    3. TOLERANCE      Part B. A Monte Carlo over simultaneous component perturbations:
                      each parameter drawn independent Gaussian, 3 sigma = +/-10%, truncated
                      at 3 sigma. Reports the empirical pass fraction with a Clopper-Pearson
                      95% interval and which tier dominates the failures.

HONESTY, stated where the numbers are produced so a reader cannot miss it:

  * Part B is a COMPONENT-TOLERANCE STRESS TEST under an ASSUMED independent-Gaussian model
    (3 sigma = +/-10%). It is NOT a foundry mismatch / Pcell model and NOT a yield
    guarantee. The +/-10% is a stated stress level, chosen before the run, not fitted to it.
  * This is DISTINCT from results/delivered_circuit.json's 45-corner PVT sweep, which uses
    real SKY130 process-corner models. Do not conflate or add the two robustness numbers.
  * Part A is local and single-parameter; it cannot see interactions between parameters.
  * Part B here uses the fast (AC-path) evaluator for speed, so HD3 and input-referred
    noise are NOT re-simulated per sample -- failures on those tiers cannot appear in the
    MC histogram. Part A uses the full evaluator, so its margins DO include HD3 and noise.

  The perturbation constants below are the preregistration. Report whatever comes out of
  them; do not shrink the stress after seeing the pass fraction.

    PYTHONPATH=src python -m eqrl.experiments.qualify_delivered            # 2000-sample MC
    PYTHONPATH=src python -m eqrl.experiments.qualify_delivered --mc-n 10000
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from eqrl.circuits.ctle import DesignVars
from eqrl.evaluator import build_evaluator
from eqrl.experiments.freeze_delivered import LIMITS
from eqrl.specs import DEFAULT_SPEC

DELIVERED = "results/delivered_circuit.json"
OUT = "results/qualification_delivered.json"

# -- Preregistration: fix these before running, report whatever they yield. ----------------
PARAMS = ["w_in", "l_in", "i_tail", "rs", "cs", "r_load"]  # w_dfe is fixed at 0, excluded
SWEEP_LO, SWEEP_HI, SWEEP_STEP = -0.20, 0.20, 0.01          # L1: -20%..+20% by 1%
MC_SIGMA = 0.10 / 3.0                                       # MC: 3 sigma = +/-10%
MC_TRUNC = 3.0                                              # truncate draws at +/- this many sigma
MC_SEED = 20260903
PAIR_GRID = [-0.10, -0.05, 0.0, 0.05, 0.10]                # L2: pairwise 5x5 grid, per parameter
ADV_BUDGET = 0.10                                           # L3: +/-10% per-parameter search box
ADV_SEED = 20260903                                         # L3: differential-evolution seed


def _margins(metrics: dict, target_boost_db: float, boost_tol_db: float) -> dict:
    """Fractional margin to failure for each hard spec, from the metrics dict.

    A positive margin means the spec is met with that fraction of slack; a negative margin
    means the spec is violated. The binding spec is the one with the smallest margin. The
    normalization is stated per metric so the number is auditable, not magic.
    """
    out = {}
    for name, lo, hi in LIMITS:
        if name not in metrics:
            continue
        v = metrics[name]
        if name == "boost_db":
            # bounded as |boost - target| <= tol
            margin = (boost_tol_db - abs(v - target_boost_db)) / boost_tol_db
            out[name] = {"value": v, "kind": "target", "target": target_boost_db,
                         "tol": boost_tol_db, "margin": margin}
        elif lo is not None and hi is None:
            denom = abs(lo) if lo != 0 else 1.0
            out[name] = {"value": v, "kind": "min", "limit": lo,
                         "margin": (v - lo) / denom}
        elif hi is not None and lo is None:
            denom = abs(hi) if hi != 0 else 1.0
            out[name] = {"value": v, "kind": "max", "limit": hi,
                         "margin": (hi - v) / denom}
        elif lo is not None and hi is not None:
            span = (hi - lo) or 1.0
            out[name] = {"value": v, "kind": "band", "limit_lo": lo, "limit_hi": hi,
                         "margin": min(v - lo, hi - v) / span}
    return out


def _evaluate(design_si: dict, evaluator, vdd: float):
    """Run one design through the guard. Returns (valid, reason, metrics_or_None)."""
    dv = DesignVars(**design_si)
    verdict = evaluator.evaluate(dv, vdd=vdd)
    if verdict.is_valid:
        return True, None, verdict.unwrap().as_dict()
    return False, verdict.check.value, None


def _first_failure(metrics: dict, target: float, tol: float) -> str | None:
    """Name the first hard spec a valid-guard design still violates, or None if it meets all."""
    for name, info in _margins(metrics, target, tol).items():
        if info["margin"] < 0:
            return name
    return None


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


# -- Level 2: pairwise interaction ---------------------------------------------------------
def _pairwise(design: dict, evaluator, pair: tuple[str, str],
              vdd: float, target: float, tol: float) -> dict:
    """Sweep two parameters together on a PAIR_GRID x PAIR_GRID grid, others nominal.

    A cell that a single-parameter sweep would call safe on both axes can still fail here --
    that is the whole point: it exposes interaction the 1-D sweep cannot see.
    """
    p1, p2 = pair
    cells = []
    for f1 in PAIR_GRID:
        row = []
        for f2 in PAIR_GRID:
            trial = dict(design)
            trial[p1] = design[p1] * (1.0 + f1)
            trial[p2] = design[p2] * (1.0 + f2)
            ok, reason, m = _evaluate(trial, evaluator, vdd)
            why = reason if not ok else _first_failure(m, target, tol)
            row.append({"pass": why is None, "reason": why})
        cells.append(row)
    return {"param1": p1, "param2": p2,
            "grid_pct": [round(f * 100, 1) for f in PAIR_GRID], "cells": cells}


# -- Level 3: bounded adversarial stress ---------------------------------------------------
def _worst_margin(delta, design: dict, evaluator, vdd: float,
                  target: float, tol: float) -> float:
    """Scalar objective: the worst (smallest) spec margin at design*(1+delta).

    Guard-invalid designs return a sentinel below any real margin, so the search treats a
    physical-validity failure as the deepest failure -- which it is.
    """
    trial = {p: design[p] * (1.0 + float(delta[i])) for i, p in enumerate(PARAMS)}
    ok, _reason, m = _evaluate(trial, evaluator, vdd)
    if not ok:
        return -1.0
    return min(v["margin"] for v in _margins(m, target, tol).values())


def _describe_point(delta, design: dict, evaluator, vdd: float,
                    target: float, tol: float) -> dict:
    trial = {p: design[p] * (1.0 + float(delta[i])) for i, p in enumerate(PARAMS)}
    ok, reason, m = _evaluate(trial, evaluator, vdd)
    out = {"delta_pct": {p: round(float(delta[i]) * 100, 2) for i, p in enumerate(PARAMS)},
           "max_component_pct": round(float(np.max(np.abs(delta))) * 100, 2),
           "guard_valid": ok}
    if not ok:
        out["first_failing_spec"] = reason
        out["min_margin"] = None
    else:
        mg = _margins(m, target, tol)
        closest = min(mg, key=lambda k: mg[k]["margin"])
        out["first_failing_spec"] = closest if mg[closest]["margin"] < 0 else None
        out["closest_spec"] = closest
        out["min_margin"] = mg[closest]["margin"]
    return out


def _adversarial(design: dict, fast_eval, full_eval, vdd: float,
                 target: float, tol: float, de_maxiter: int, de_popsize: int) -> dict:
    """Search the +/-ADV_BUDGET box for the perturbation closest to failure, then find the
    nearest failure along that discovered direction. LOCAL heuristic -- reports the worst
    case it FOUND, never claims a global optimum or the globally nearest failure.
    """
    from scipy.optimize import differential_evolution

    bounds = [(-ADV_BUDGET, ADV_BUDGET)] * len(PARAMS)
    res = differential_evolution(
        _worst_margin, bounds, args=(design, fast_eval, vdd, target, tol),
        seed=ADV_SEED, maxiter=de_maxiter, popsize=de_popsize, tol=1e-4,
        polish=False, updating="immediate")
    worst = np.asarray(res.x)

    # Re-check the worst point found with the FULL guard so the reported margin is accurate.
    worst_full = _describe_point(worst, design, full_eval, vdd, target, tol)

    # Nearest failure along the discovered direction: smallest scale t in (0,1] whose
    # fast-path worst-margin is <= 0. If the full budget in this direction still passes, say so.
    nearest: dict
    if _worst_margin(worst, design, fast_eval, vdd, target, tol) > 0:
        nearest = {"note": "the circuit survives the full +/-%.0f%% box along the worst "
                           "direction found; nearest failure is beyond the budget."
                           % (ADV_BUDGET * 100)}
    else:
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if _worst_margin(mid * worst, design, fast_eval, vdd, target, tol) <= 0:
                hi = mid
            else:
                lo = mid
        nearest = _describe_point(hi * worst, design, full_eval, vdd, target, tol)

    return {
        "protocol": "bounded local search (differential evolution, fast/AC path) for the "
                    "+/-%.0f%%-per-parameter perturbation that minimizes the worst spec "
                    "margin; the worst point and the nearest-failure point are re-checked "
                    "with the full guard." % (ADV_BUDGET * 100),
        "scope": "LOCAL heuristic. Reports the worst case FOUND within the budget, not a "
                 "proof of the global worst case or the globally nearest failure. Search "
                 "objective uses the fast (AC) path, so HD3/noise do not steer it.",
        "budget_pct": ADV_BUDGET * 100,
        "de": {"maxiter": de_maxiter, "popsize": de_popsize, "seed": ADV_SEED,
               "func_evals": int(res.nfev)},
        "worst_case_within_budget": worst_full,
        "nearest_failure": nearest,
    }


def build(mc_n: int, do_pairwise: bool, do_adversarial: bool,
          de_maxiter: int, de_popsize: int) -> dict:
    delivered = json.loads(Path(DELIVERED).read_text())
    design = dict(delivered["design"])
    channel = delivered["spec"]["channel_loss_db"]
    target = delivered["spec"]["target_boost_db"]
    tol = delivered["spec"]["boost_tol_db"]
    vdd = DEFAULT_SPEC.vdd_nominal

    full = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False, channel_loss_db=channel)
    fast = build_evaluator(DEFAULT_SPEC, corner="tt", fast=True, channel_loss_db=channel)

    # -- Nominal margins -------------------------------------------------------------------
    # Reported per-metric as a fingerprint. NOT ranked across metrics to crown a "binding"
    # spec: the limits are normalized incommensurably (dc_gain_db's floor is 0 dB, so its
    # margin is absolute dB, while peak_freq_ghz's is a fraction of a 1.25 GHz band), and a
    # TT-nominal ranking answers a different question than the 45-corner sign-off anyway.
    # The authoritative binding constraint is quoted from the frozen PVT sweep below; the
    # unit-comparable "what breaks first" is Part A.
    ok, reason, metrics = _evaluate(design, full, vdd)
    if not ok:
        raise SystemExit("REFUSING: the delivered circuit does not pass its own guard "
                         "(%s). Something is wrong upstream, not here." % reason)
    margins = _margins(metrics, target, tol)

    # Binding constraint, quoted (not re-derived) from the frozen 45-corner PVT worst case.
    wc = delivered.get("pvt", {}).get("worst_case_by_metric", {})
    pvt_binding = None
    if "dc_gain_db" in wc:
        floor = wc["dc_gain_db"].get("limit_lo") or 0.0
        pvt_binding = {
            "spec": "dc_gain_db",
            "source": "%s -> pvt.worst_case_by_metric (frozen; not re-simulated here)"
                      % DELIVERED,
            "worst_corner_value_db": wc["dc_gain_db"]["min"],
            "worst_corner_at": wc["dc_gain_db"]["min_at"],
            "floor_db": floor,
            "worst_corner_slack_db": wc["dc_gain_db"]["min"] - floor,
            "note": "DC gain is the constraint the PVT sign-off found binding. It has ample "
                    "slack at TT nominal; it only binds at the worst process/VDD/temp corner.",
        }

    # -- Part A: single-parameter failure radius -------------------------------------------
    steps = np.round(np.arange(SWEEP_LO, SWEEP_HI + 1e-9, SWEEP_STEP), 4)
    single = {}
    for p in PARAMS:
        pos_fail = pos_reason = neg_fail = neg_reason = None
        for frac in steps:
            if frac == 0.0:
                continue
            trial = dict(design)
            trial[p] = design[p] * (1.0 + float(frac))
            v_ok, v_reason, v_metrics = _evaluate(trial, full, vdd)
            failed = (not v_ok) or (_first_failure(v_metrics, target, tol) is not None)
            why = v_reason if not v_ok else _first_failure(v_metrics, target, tol)
            if frac > 0 and failed and pos_fail is None:
                pos_fail, pos_reason = float(frac), why
            if frac < 0 and failed and neg_fail is None:
                neg_fail, neg_reason = float(frac), why
        radii = [abs(x) for x in (pos_fail, neg_fail) if x is not None]
        single[p] = {
            "first_fail_pos_pct": None if pos_fail is None else round(pos_fail * 100, 1),
            "reason_pos": pos_reason,
            "first_fail_neg_pct": None if neg_fail is None else round(neg_fail * 100, 1),
            "reason_neg": neg_reason,
            "radius_pct": round(min(radii) * 100, 1) if radii else None,
        }
    with_radius = {p: s for p, s in single.items() if s["radius_pct"] is not None}
    binding_param = (min(with_radius, key=lambda k: with_radius[k]["radius_pct"])
                     if with_radius else None)
    # Two most-sensitive parameters (smallest single-param radius) drive the L2 grid.
    ranked = sorted(with_radius, key=lambda k: with_radius[k]["radius_pct"])

    # -- Level 2: pairwise interaction -----------------------------------------------------
    pairwise = None
    if do_pairwise and len(ranked) >= 2:
        pair = (ranked[0], ranked[1])
        pairwise = _pairwise(design, full, pair, vdd, target, tol)
        pairwise["scope"] = ("The two most single-parameter-sensitive params (%s, %s); "
                             "others held nominal. Full guard. Exposes interaction the "
                             "1-D sweep in Part A cannot see." % pair)

    # -- Level 3: bounded adversarial stress -----------------------------------------------
    adversarial = None
    if do_adversarial:
        adversarial = _adversarial(design, fast, full, vdd, target, tol,
                                   de_maxiter, de_popsize)

    # -- Part B: Monte Carlo component-tolerance stress ------------------------------------
    rng = np.random.default_rng(MC_SEED)
    n_pass = 0
    fail_by_tier: dict[str, int] = {}
    base = np.array([design[p] for p in PARAMS])
    for _ in range(mc_n):
        draw = rng.normal(0.0, MC_SIGMA, size=len(PARAMS))
        draw = np.clip(draw, -MC_TRUNC * MC_SIGMA, MC_TRUNC * MC_SIGMA)
        trial = {p: float(base[i] * (1.0 + draw[i])) for i, p in enumerate(PARAMS)}
        v_ok, v_reason, v_metrics = _evaluate(trial, fast, vdd)
        why = v_reason if not v_ok else _first_failure(v_metrics, target, tol)
        if why is None:
            n_pass += 1
        else:
            fail_by_tier[why] = fail_by_tier.get(why, 0) + 1
    frac = n_pass / mc_n
    ci_lo, ci_hi = _clopper_pearson(n_pass, mc_n)

    return {
        "what": "post-design qualification of the delivered SILQ circuit -- measures the "
                "frozen circuit, optimizes nothing, changes no recorded number",
        "generated_by": "eqrl.experiments.qualify_delivered",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "circuit_source": DELIVERED,
        "circuit_commit": delivered.get("commit"),
        "guard": "eqrl.evaluator.build_evaluator -- the same guard the CLI and dashboard use",

        "nominal": {
            "evaluator": "fast=False (full guard: real HD3 and input-referred noise)",
            "target_boost_db": target, "boost_tol_db": tol, "channel_loss_db": channel,
            "margins": margins,
            "margins_note": "Per-metric fingerprint only. NOT ranked across metrics to crown "
                            "a binding spec -- the normalizations are incommensurable "
                            "(dc_gain_db floor is 0 dB -> absolute dB; peak_freq_ghz -> "
                            "fraction of the band). The authoritative binding constraint is "
                            "pvt_binding; the unit-comparable 'what breaks first' is Part A.",
            "pvt_binding": pvt_binding,
        },

        "failure_radius": {
            "protocol": "Part A: one parameter at a time, %d%%..%d%% by %d%%, others nominal, "
                        "full guard." % (SWEEP_LO * 100, SWEEP_HI * 100, SWEEP_STEP * 100),
            "scope": "LOCAL and single-parameter -- cannot see parameter interactions.",
            "per_param": single,
            "binding_param": binding_param,
            "empirical_local_failure_radius_pct": (
                with_radius[binding_param]["radius_pct"] if binding_param else None),
        },

        "pairwise": pairwise,
        "adversarial": adversarial,

        "monte_carlo": {
            "label": "COMPONENT-TOLERANCE STRESS TEST -- assumed independent Gaussian, "
                     "3 sigma = +/-10%%. NOT a foundry mismatch/yield model. Distinct from "
                     "the 45-corner PVT result, which uses real SKY130 corner models.",
            "evaluator": "fast=True (AC path): HD3 and noise NOT re-simulated per sample.",
            "n": mc_n, "sigma_pct": round(MC_SIGMA * 100, 3), "trunc_sigma": MC_TRUNC,
            "rng_seed": MC_SEED, "params": PARAMS,
            "pass_fraction": frac,
            "ci95_low": ci_lo, "ci95_high": ci_hi,
            "fail_by_tier": dict(sorted(fail_by_tier.items(), key=lambda kv: -kv[1])),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mc-n", type=int, default=2000,
                   help="Monte Carlo sample count (default 2000; use 10000 for the final).")
    p.add_argument("--no-pairwise", action="store_true", help="skip Level 2 (pairwise grid).")
    p.add_argument("--no-adversarial", action="store_true",
                   help="skip Level 3 (bounded adversarial search).")
    p.add_argument("--de-maxiter", type=int, default=40,
                   help="Level 3 differential-evolution max iterations (default 40).")
    p.add_argument("--de-popsize", type=int, default=12,
                   help="Level 3 differential-evolution population size (default 12).")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    m = build(args.mc_n, not args.no_pairwise, not args.no_adversarial,
              args.de_maxiter, args.de_popsize)
    Path(args.out).write_text(json.dumps(m, indent=1))
    nom, fr, mc = m["nominal"], m["failure_radius"], m["monte_carlo"]
    print("qualification -> %s" % args.out)
    if nom["pvt_binding"]:
        pb = nom["pvt_binding"]
        print("  binding (PVT)        %s: worst corner %.2f dB, floor %.1f dB (slack %.2f dB)"
              % (pb["spec"], pb["worst_corner_value_db"], pb["floor_db"],
                 pb["worst_corner_slack_db"]))
    if fr["binding_param"]:
        print("  local failure radius %.1f%% (first: %s, breaks %s)"
              % (fr["empirical_local_failure_radius_pct"], fr["binding_param"],
                 fr["per_param"][fr["binding_param"]]["reason_pos"]
                 or fr["per_param"][fr["binding_param"]]["reason_neg"]))
    if m["pairwise"]:
        pw = m["pairwise"]
        n_fail = sum(1 for row in pw["cells"] for c in row if not c["pass"])
        print("  L2 pairwise (%s x %s)  %d of %d grid cells fail"
              % (pw["param1"], pw["param2"], n_fail, len(pw["cells"]) ** 2))
    if m["adversarial"]:
        wc = m["adversarial"]["worst_case_within_budget"]
        nf = m["adversarial"]["nearest_failure"]
        print("  L3 worst-in-box      valid=%s, min_margin=%s, max component %.1f%%"
              % (wc["guard_valid"], wc["min_margin"], wc["max_component_pct"]))
        if "max_component_pct" in nf:
            print("  L3 nearest failure   %s at max component %.1f%%"
                  % (nf.get("first_failing_spec"), nf["max_component_pct"]))
        else:
            print("  L3 nearest failure   %s" % nf.get("note"))
    print("  MC pass fraction     %.3f  (95%% CI %.3f - %.3f, n=%d, +/-10%% 3-sigma stress)"
          % (mc["pass_fraction"], mc["ci95_low"], mc["ci95_high"], mc["n"]))
    if mc["fail_by_tier"]:
        top = next(iter(mc["fail_by_tier"]))
        print("  MC dominant failure  %s (%d of %d)"
              % (top, mc["fail_by_tier"][top], mc["n"] - int(mc["pass_fraction"] * mc["n"])))


if __name__ == "__main__":
    main()
