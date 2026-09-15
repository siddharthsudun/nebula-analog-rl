"""Does G3.2 still work when its slope table is MEASURED instead of read from the table?

WHAT THIS IS FOR. The fair criticism of G3.2 is that a reader cannot tell its five slope
constants from hand-set numbers, because the controller ships with them baked in. This runs
the SAME solver twice per spec from the SAME stage-1 handoff -- once with the frozen
seed-2 plane, once with a plane measured at runtime by `g32_selfcal.calibrate_plane` -- and
diffs the outcomes. If the measured plane does as well, the constants are demonstrably
derivable from the design in hand and were frozen for reproducibility rather than fitted
for performance. Criterion, slice and verdict rule are in `docs/PREREG_G32_SELFCAL.md`,
committed before this file ever ran.

THIS DEMONSTRATES, IT DOES NOT REPLACE. The delivered circuit stays the frozen-slope one.
Nothing here is re-frozen, `g32_repair.py` is untouched, and `final_comparison.py` is
imported, never edited -- the transcribed `g32_solve` this calls is the same function its
own equivalence gate covers, which is what makes the frozen arm here checkable against a
committed artifact rather than merely plausible.

THE COMPARISON IS PAIRED AND THE CALIBRATION PAYS ITS OWN WAY. Both arms get the same
`evaluate`, the same rescue ladder, the same stage-1 rollout and the same stage-2 pool of
r = 10 evaluations; the perturbations the calibrated arm spends come OUT of that pool, so
it runs on `r - spent`. The only other difference between the arms is the plane dict. Exact
byte-match is not expected: different slopes put the secant on a different path.

    PYTHONPATH=src python -m eqrl.experiments.g32_selfcal_bench

Exit status is 1 if the internal validity gate fails (the frozen arm here must reproduce
`results/g32_repair_smoke.json` exactly), NOT if the verdict is WORSE. A worse verdict is
a result and is reported as one.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from eqrl.experiments.g32_peak_report import plane_from_probe
from eqrl.experiments.g32_selfcal import PROBE_H, calibrate_plane

# `final_comparison`, `target_audit` and `g32_rescue_probe` each perform the Windows
# ngspice/PDK bootstrap at import time. That bootstrap used to read
# `os.environ["USERPROFILE"]` outright and so raised `KeyError` on Linux; it now falls back
# to `Path.home()`, which is why CI can collect the five test files that used to error at
# import. The deferred imports below stay anyway: the bootstrap being survivable is not the
# same as the PDK being present, and `verdict`, `validity_gate` and the preregistered
# constants must stay importable with no PDK and no simulator at all. That is not a
# tidiness point: those are exactly the checks CI has to be able to run, since they are
# what stop the criterion being edited after the fact.

#: The five fields `final_comparison.gate` diffs. The frozen arm in THIS harness has to
#: reproduce the committed artifact on all of them, or the harness -- not the hypothesis --
#: is what the run measured.
GATE_FIELDS = ["best_boost_db", "best_abs_err", "loose_solved_at", "strict_solved_at",
               "n_valid"]

#: Preregistered in docs/PREREG_G32_SELFCAL.md sections 2 and 4. Named constants here so
#: the verdict is computed from that document rather than from whatever looks good after.
CAL_BUDGET = 4           # hard cap on calibration evaluations: 2 axes x at most 2 probes
LOSS_ALLOWANCE = 1       # specs out of 10 that may be lost and still count as equivalent
ERR_ALLOWANCE_DB = 0.15  # 10% of the +/-1.5 dB criterion tolerance


def run_arm(evaluate, xs, s1, target, plane, ladder, budget, tol, k) -> dict:
    """One G3.2 refinement, summarized exactly the way `final_comparison` arm B is.

    `xs` and `s1` are copied because two arms share them. `g32_solve` only reads them, but
    a comparison whose two halves alias each other is not one worth defending.
    """
    from eqrl.experiments.final_comparison import PREREG, g32_solve, summarize

    g2, info, _x_f, _rec_f, _left = g32_solve(
        evaluate, [np.array(x, dtype=np.float64) for x in xs], list(s1), target, plane,
        ladder, budget)
    s1_only = [e for e in s1 if e]
    res = summarize(s1_only + g2, target, tol)
    res["stage2"] = summarize(g2, target, tol)
    res["solver"] = info
    res["stage2_evals"] = len(info["steps"])
    res["measure_all_spent"] = (
        k * PREREG["ppo_measure_all_per_eval"]
        + len(info["steps"]) * PREREG["search_measure_all_per_eval"])
    return res


def validity_gate(rows, path: str) -> tuple[bool, list[str]]:
    """The frozen arm here must BE the committed frozen arm. Checked before any verdict."""
    froz = {r["spec"]: r["g32"] for r in json.loads(Path(path).read_text())["rows"]}
    bad = []
    for row in rows:
        i = row["spec"]
        if i not in froz:
            bad.append("spec %d absent from %s" % (i, path))
            continue
        f, g = froz[i], row["frozen"]
        for fld in GATE_FIELDS:
            u, v = f.get(fld), g.get(fld)
            same = (u is None and v is None) or (
                u is not None and v is not None
                and (abs(u - v) < 1e-9 if isinstance(u, (int, float))
                     and not isinstance(u, bool) else u == v))
            if not same:
                bad.append("spec %d %s %r != %r" % (i, fld, u, v))
    return not bad, bad


def verdict(rows) -> dict:
    """Apply the preregistered table. No metric is invented here that is not in it."""
    def count(arm, field):
        return sum(bool(r[arm][field]) for r in rows)

    def n_valid(arm):
        return sum(r[arm]["best_abs_err"] is not None for r in rows)

    paired = [(r["spec"], r["cal"]["best_abs_err"] - r["frozen"]["best_abs_err"])
              for r in rows
              if r["cal"]["best_abs_err"] is not None
              and r["frozen"]["best_abs_err"] is not None]
    med = statistics.median([d for _i, d in paired]) if paired else None

    m = {
        "strict_frozen": count("frozen", "strict_solved_at"),
        "strict_cal": count("cal", "strict_solved_at"),
        "loose_frozen": count("frozen", "loose_solved_at"),
        "loose_cal": count("cal", "loose_solved_at"),
        "valid_frozen": n_valid("frozen"),
        "valid_cal": n_valid("cal"),
        "paired_n": len(paired),
        "paired_deltas": {str(i): d for i, d in paired},
        "median_delta_abs_err_db": med,
    }
    m["P1_strict"] = m["strict_cal"] >= m["strict_frozen"] - LOSS_ALLOWANCE
    m["P2_loose"] = m["loose_cal"] >= m["loose_frozen"] - LOSS_ALLOWANCE
    m["S1_valid"] = m["valid_cal"] >= m["valid_frozen"] - LOSS_ALLOWANCE
    m["S2_err"] = med is not None and med <= ERR_ALLOWANCE_DB
    equivalent = all(m[key] for key in ("P1_strict", "P2_loose", "S1_valid", "S2_err"))
    better = equivalent and (m["strict_cal"] > m["strict_frozen"]
                             or (med is not None and med < -ERR_ALLOWANCE_DB))
    m["verdict"] = "BETTER" if better else ("EQUIVALENT" if equivalent else "WORSE")
    return m


def main() -> None:
    from eqrl.experiments.final_comparison import (PREREG, Evaluation, _check_constants,
                                                   load_policy, stage1_rollout)
    from eqrl.experiments.g32_rescue_probe import rescue_order_from_probe
    from eqrl.experiments.target_audit import make_specs

    _check_constants()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--spec-seed", type=int, default=3)
    p.add_argument("--first", type=int, default=8)
    p.add_argument("--specs", type=int, default=10)
    p.add_argument("--peak-probe", default="results/g32_peak_probe.json")
    p.add_argument("--rescue-probe", default="results/g32_rescue_probe.json")
    p.add_argument("--frozen-artifact", default="results/g32_repair_smoke.json")
    p.add_argument("--tol", type=float, default=PREREG["tol"])
    p.add_argument("--out", default="results/g32_selfcal_bench.json")
    args = p.parse_args()

    k, r, tol = PREREG["k"], PREREG["r"], args.tol
    frozen_plane = plane_from_probe(args.peak_probe)
    ladder = rescue_order_from_probe(args.rescue_probe)

    specs = list(enumerate(make_specs(args.first + args.specs, args.spec_seed)))
    specs = specs[args.first:args.first + args.specs]

    ev = Evaluation()
    model, env = load_policy(args.model)

    print("H-SC | G3.2 frozen slopes vs slopes measured at the handoff", flush=True)
    print("  slice     spec-seed %d, specs %d-%d (burned development slice)"
          % (args.spec_seed, args.first, args.first + len(specs) - 1), flush=True)
    print("  paired    same stage-1 rollout, same ladder, same r=%d pool; only the plane "
          "differs" % r, flush=True)
    print("  cost      calibration spends out of that pool: the calibrated arm runs on "
          "r - spent", flush=True)
    print("  criterion docs/PREREG_G32_SELFCAL.md section 4, committed before this run",
          flush=True)
    print(flush=True)

    def show(res):
        return "%-6s %-5s %-6s" % (
            "-" if res["best_boost_db"] is None else "%.2f" % res["best_boost_db"],
            "-" if res["best_abs_err"] is None else "%.2f" % res["best_abs_err"],
            "strict" if res["strict_solved_at"] else
            ("loose" if res["loose_solved_at"] else "-"))

    rows = []
    for i, (target, channel) in specs:
        evaluate = ev.make_eval(channel)
        xs, s1, term_at = stage1_rollout(evaluate, model, env, i, target, channel, k)

        frozen_res = run_arm(evaluate, xs, s1, target, frozen_plane, ladder, r, tol, k)

        cal_plane, spent, cal_info = calibrate_plane(
            evaluate, xs[-1], target, frozen_plane, base_rec=s1[-1], h=PROBE_H,
            two_sided=False, budget=CAL_BUDGET)
        cal_res = run_arm(evaluate, xs, s1, target, cal_plane, ladder, r - spent, tol, k)
        cal_res["measure_all_spent"] += spent * PREREG["search_measure_all_per_eval"]

        rows.append({
            "spec": i, "target_boost_db": target, "channel_loss_db": channel,
            "handoff": {"boost_db": None if s1[-1] is None else s1[-1]["boost_db"],
                        "guard_valid": s1[-1] is not None,
                        "ppo_terminated_at_eval": term_at},
            "frozen": frozen_res, "cal": cal_res,
            "calibration": {
                "evaluations": spent, "measured": cal_info["measured"],
                "fellback": cal_info["fellback"], "notes": cal_info["notes"],
                "fully_frozen": cal_info["fully_frozen"], "ratios": cal_info["ratios"],
                "slopes": {key: cal_plane[key] for key in frozen_plane
                           if isinstance(frozen_plane[key], (int, float))},
                # The solver may start from an earlier stage-1 state than the handoff the
                # slopes were measured at. Section 5 of the prereg; recorded per spec so
                # that limitation is auditable rather than merely asserted.
                "solver_start_source_frozen": frozen_res["solver"]["start_source"],
                "solver_start_source_cal": cal_res["solver"]["start_source"]}})

        print("  spec %2d  tgt %5.2f | FROZEN %s | CAL %s | cal %d ev, measured %d/5"
              % (i, target, show(frozen_res), show(cal_res), spent,
                 len(cal_info["measured"])), flush=True)

    ok, bad = validity_gate(rows, args.frozen_artifact)
    m = verdict(rows)
    Path(args.out).write_text(json.dumps(
        {"prereg": "docs/PREREG_G32_SELFCAL.md",
         "constants": {"cal_budget": CAL_BUDGET, "loss_allowance": LOSS_ALLOWANCE,
                       "err_allowance_db": ERR_ALLOWANCE_DB, "probe_h": PROBE_H},
         "spec_seed": args.spec_seed, "first": args.first, "specs": args.specs,
         "tol": tol, "model": args.model, "r": r, "k": k,
         "frozen_plane": frozen_plane, "frozen_artifact": args.frozen_artifact,
         "validity_gate_passed": ok, "validity_gate_failures": bad,
         "metrics": m, "n_simulations_run": ev.n_sim,
         "complete": len(rows) == len(specs), "rows": rows}, indent=1))
    print("\nwrote", args.out, flush=True)

    print("\n" + "=" * 86, flush=True)
    print("INTERNAL VALIDITY GATE -- the frozen arm here vs %s" % args.frozen_artifact,
          flush=True)
    if not ok:
        for line in bad:
            print("  MISMATCH", line, flush=True)
        print("  FAILED -- the harness is wrong. No verdict is reported.", flush=True)
        raise SystemExit(1)
    print("  PASSED: all %d specs reproduce the committed frozen arm on %s"
          % (len(rows), ", ".join(GATE_FIELDS)), flush=True)

    print("=" * 86, flush=True)
    print("H-SC RESULT (descriptive; burned development slice, n=10, no inferential test)",
          flush=True)
    print("  P1 strict solves   frozen %d/10  calibrated %d/10   %s"
          % (m["strict_frozen"], m["strict_cal"], "ok" if m["P1_strict"] else "FAILS"),
          flush=True)
    print("  P2 loose solves    frozen %d/10  calibrated %d/10   %s"
          % (m["loose_frozen"], m["loose_cal"], "ok" if m["P2_loose"] else "FAILS"),
          flush=True)
    print("  S1 valid designs   frozen %d/10  calibrated %d/10   %s"
          % (m["valid_frozen"], m["valid_cal"], "ok" if m["S1_valid"] else "FAILS"),
          flush=True)
    print("  S2 paired median delta abs_err  %s dB over %d specs   %s"
          % ("n/a" if m["median_delta_abs_err_db"] is None
             else "%+.4f" % m["median_delta_abs_err_db"], m["paired_n"],
             "ok" if m["S2_err"] else "FAILS"), flush=True)
    print("=" * 86, flush=True)
    print("VERDICT: %s" % m["verdict"], flush=True)
    print("The delivered circuit is unchanged either way: it is the frozen-slope one.",
          flush=True)


if __name__ == "__main__":
    main()
