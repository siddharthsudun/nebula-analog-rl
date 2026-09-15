"""Compare frozen G3.2 against a regime-aware peak-axis arm on a fresh spec set.

This does not alter the frozen controller, its artifacts, or the shipped pipeline.  The
new arm spends up to four of its ten stage-two evaluations identifying a usable local peak
axis, then invokes ``final_comparison.g32_solve`` unchanged with the remaining budget.
Run only after choosing a fresh non-zero ``--spec-seed`` and recording it as the evaluation
set; this script is intentionally separate from all historical benchmark artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eqrl.experiments.g32_regime import probe_start, regime_plane
from eqrl.experiments.g32_selfcal_bench import GATE_FIELDS, run_arm, validity_gate


def main() -> None:
    # Imports with the Windows simulator bootstrap stay inside main so pure helpers and
    # their unit tests remain importable without a PDK.
    from eqrl.experiments.final_comparison import (PREREG, Evaluation, _check_constants,
                                                   load_policy, stage1_rollout)
    from eqrl.experiments.g32_peak_report import plane_from_probe
    from eqrl.experiments.g32_rescue_probe import rescue_order_from_probe
    from eqrl.experiments.target_audit import make_specs

    _check_constants()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--spec-seed", type=int, required=True,
                   help="fresh non-zero held-out specification seed")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--peak-probe", default="results/g32_peak_probe.json")
    p.add_argument("--rescue-probe", default="results/g32_rescue_probe.json")
    p.add_argument("--frozen-artifact", default=None,
                   help="optional matching frozen run; enables the equivalence gate")
    p.add_argument("--tol", type=float, default=PREREG["tol"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.spec_seed == 0:
        p.error("--spec-seed must be non-zero for this held-out comparison")

    out = args.out or "results/g32_regime_seed%d.json" % args.spec_seed
    k, r = PREREG["k"], PREREG["r"]
    frozen_plane = plane_from_probe(args.peak_probe)
    ladder = rescue_order_from_probe(args.rescue_probe)
    all_specs = list(enumerate(make_specs(args.first + args.specs, args.spec_seed)))
    specs = all_specs[args.first:args.first + args.specs]
    ev = Evaluation()
    model, env = load_policy(args.model)

    print("G3.2 REGIME AXIS | fresh spec-seed %d, specs %d-%d" %
          (args.spec_seed, args.first, args.first + len(specs) - 1), flush=True)
    print("  frozen arm and new arm share PPO rollout, guard, ladder, and r=%d." % r,
          flush=True)
    print("  new arm probes r_load below band / l_in above band; probe cost is deducted.",
          flush=True)

    rows = []
    for i, (target, channel) in specs:
        evaluate = ev.make_eval(channel)
        xs, s1, term_at = stage1_rollout(evaluate, model, env, i, target, channel, k)
        frozen = run_arm(evaluate, xs, s1, target, frozen_plane, ladder, r, args.tol, k)
        x0, base, start_source = probe_start(xs, s1, target)
        plane, spent, selection = regime_plane(evaluate, x0, target, frozen_plane,
                                                base_rec=base, budget=r)
        regime = run_arm(evaluate, xs, s1, target, plane, ladder, r - spent, args.tol, k)
        regime["measure_all_spent"] += spent * PREREG["search_measure_all_per_eval"]
        rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                     "handoff": {"boost_db": None if s1[-1] is None else s1[-1]["boost_db"],
                                 "guard_valid": s1[-1] is not None,
                                 "ppo_terminated_at_eval": term_at},
                     "frozen": frozen, "regime": regime,
                     "axis_selection": {"start_source": start_source, **selection}})
        print("  spec %2d  %s -> %-6s (%d probe evals) | frozen err %-5s  regime err %-5s" %
              (i, selection["regime"], selection["selected_axis"], spent,
               "-" if frozen["best_abs_err"] is None else "%.2f" % frozen["best_abs_err"],
               "-" if regime["best_abs_err"] is None else "%.2f" % regime["best_abs_err"]),
              flush=True)
        Path(out).write_text(json.dumps({"spec_seed": args.spec_seed, "first": args.first,
            "specs": args.specs, "tol": args.tol, "model": args.model,
            "frozen_plane": frozen_plane, "rows": rows, "complete": False}, indent=1))

    gate_ok, gate_bad = (None, [])
    if args.frozen_artifact:
        gate_rows = [{"spec": row["spec"], "frozen": row["frozen"]} for row in rows]
        gate_ok, gate_bad = validity_gate(gate_rows, args.frozen_artifact)
    Path(out).write_text(json.dumps({"spec_seed": args.spec_seed, "first": args.first,
        "specs": args.specs, "tol": args.tol, "model": args.model,
        "r": r, "k": k, "frozen_plane": frozen_plane, "gate_fields": GATE_FIELDS,
        "frozen_artifact": args.frozen_artifact, "frozen_gate_passed": gate_ok,
        "frozen_gate_failures": gate_bad, "n_simulations_run": ev.n_sim,
        "rows": rows, "complete": True}, indent=1))
    print("wrote", out, flush=True)
    if gate_ok is False:
        raise SystemExit("frozen equivalence gate failed: " + "; ".join(gate_bad))


if __name__ == "__main__":
    main()
