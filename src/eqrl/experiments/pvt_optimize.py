"""Bounded PVT sizing refinement and independent full-grid acceptance.

Default repair case: first historically failing candidate by spec index, chosen
before new measurements. Existing delivered design is an explicitly named anchor.
No production artifact is replaced. Output directories must be new.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from eqrl.circuits.ctle import DesignVars, decode_action, encode_action
from eqrl.envs.pvt import corner_grid
from eqrl.experiments.tail_mirror_pilot import ROOT, fingerprints, write_json
from eqrl.pvt_refinement import PVTEvaluator, full_grid_pass, rank_key, recorded_netlist, stress_corners, summarize
from eqrl.specs import DEFAULT_SPEC, Spec


def initial_case():
    history = json.loads((ROOT / "results/pvt_signoff_seed23.json").read_text())
    failures = [c for c in history["candidates"]
                if not all(r.get("guard_valid") and r.get("pass10") for r in c["corners"].values())]
    first = min(failures, key=lambda c: c["spec_index"])
    delivered = json.loads((ROOT / "results/delivered_circuit.json").read_text())
    return {
        "selection": "first historically non-clean candidate by spec_index; no fresh-result selection",
        "spec_index": first["spec_index"],
        "spec": asdict(replace(DEFAULT_SPEC, target_boost_db=first["target_boost_db"],
                               channel_loss_db=first["channel_loss_db"], boost_target_tol_db=1.5)),
        "initial": [
            {"id": "input", "origin": "historical PPO + G3.2 candidate", "design": first["design"]},
            {"id": "anchor", "origin": "frozen delivered sizing, rescored at this case's target/channel",
             "design": delivered["design"]},
        ],
    }


def optimize(output):
    import cma
    config = json.loads((output / "config.json").read_text())
    spec = Spec(**config["spec"])
    grid = corner_grid(spec, "full")
    ev = PVTEvaluator(spec, output / "raw", limit=500)
    initial = config["initial"]
    resume = Path(config["resume_from"]) if config.get("resume_from") else None
    if resume:
        cached = json.loads((resume / "baselines.json").read_text())
        baseline = {k: v["rows"] for k, v in cached.items()}
    else:
        baseline = ev.evaluate_batch(initial, grid)
    write_json(output / "baselines.json", {c["id"]: dict(candidate=c, rows=baseline[c["id"]],
               summary=summarize(baseline[c["id"]])) for c in initial})
    stresses = stress_corners(baseline["input"] + baseline["anchor"], grid, limit=8)
    write_json(output / "search_corners.json", stresses)
    selected = set(stresses)
    search_rows = {c["id"]: [r for r in baseline[c["id"]] if tuple(r["corner"]) in selected]
                   for c in initial}
    candidates = {c["id"]: c for c in initial}
    best_initial = min(initial, key=lambda c: rank_key(baseline[c["id"]]))
    es = cma.CMAEvolutionStrategy(
        encode_action(DesignVars(**best_initial["design"])).astype(float), 0.035,
        {"bounds": [0, 1], "popsize": 8, "seed": 20260910, "verbose": -9, "maxiter": 5},
    )
    for generation in range(5):
        xs = es.ask()
        batch = [{"id": f"g{generation}_c{i}", "origin": "PVT CMA-ES refinement",
                  "design": asdict(decode_action(x))} for i, x in enumerate(xs)]
        saved = resume / f"generation_{generation}.json" if resume else None
        if saved and saved.exists():
            previous = json.loads(saved.read_text())
            if [r["candidate"] for r in previous] != batch:
                raise RuntimeError("CMA replay disagrees with saved candidates; cannot resume")
            results = {r["candidate"]["id"]: r["rows"] for r in previous}
            print(f"replayed generation {generation} without simulations", flush=True)
        else:
            results = ev.evaluate_batch(batch, stresses)
        candidates.update({c["id"]: c for c in batch})
        search_rows.update(results)
        order = sorted(range(len(batch)), key=lambda i: rank_key(results[batch[i]["id"]]))
        costs = np.empty(len(batch))
        for rank, i in enumerate(order):
            costs[i] = rank
        es.tell(xs, costs.tolist())
        write_json(output / f"generation_{generation}.json", [
            dict(candidate=c, rows=results[c["id"]], summary=summarize(results[c["id"]])) for c in batch])
        winner = min(candidates, key=lambda k: rank_key(search_rows[k]))
        print(f"generation {generation}: leader {winner} {summarize(search_rows[winner])}", flush=True)
    ranked = sorted(candidates, key=lambda k: rank_key(search_rows[k]))
    # Baseline full-grid measurements are already available. Validate two NEW
    # candidates selected solely by search scores; never label eight-corner success
    # as PVT acceptance. A separate process repeats the chosen winner afterward.
    finalists = [candidates[k] for k in ranked if k not in baseline][:2]
    full = ev.evaluate_batch(finalists, grid)
    full.update(baseline)
    considered = initial + finalists
    clean = [c for c in considered if full_grid_pass(full[c["id"]], grid)]
    pool = clean or considered
    winner = min(pool, key=lambda c: rank_key(full[c["id"]]))
    write_json(output / "selection.json", {
        "winner": winner, "provisionally_full_grid_passed": bool(clean),
        "summary": summarize(full[winner["id"]]),
        "full_candidates": {c["id"]: dict(candidate=c, rows=full[c["id"]],
                              summary=summarize(full[c["id"]])) for c in considered},
        "evaluations": ev.evaluations,
        "corner_integrity_tt_ss_passed": ev.corner_integrity_passed,
        "search_rule": "min guard failures, then failed corners, guard violation, maximin normalized slack, target error",
    })


def verify(output, evaluator=None):
    config = json.loads((output / "config.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    spec, winner = Spec(**config["spec"]), selection["winner"]
    grid = corner_grid(spec, "full")
    ev = evaluator or PVTEvaluator(spec, output / "verification_raw", limit=45)
    rows = ev.evaluate_batch_parallel([winner], grid)[winner["id"]]
    accepted = full_grid_pass(rows, grid)
    result = {"candidate": winner, "spec": config["spec"], "rows": rows,
              "summary": summarize(rows), "accepted": accepted,
              "evaluations": ev.evaluations, "independent_process": True,
              "corner_integrity_tt_ss_passed": ev.corner_integrity_passed,
              "meaning": "full guarded 45-corner acceptance at this one target/channel; not general policy robustness"}
    write_json(output / "verification.json", result)
    # Every export is explicit about acceptance; a failed design is never exported
    # under a verified filename or substituted into the shipped result.
    if accepted:
        write_json(output / "verified_candidate.json", result)
        (output / "verified_candidate.cir").write_text(recorded_netlist(
            DesignVars(**winner["design"]), vdd=1.8, temp_c=27, corner="tt"), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=("optimize", "verify"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.worker:
        (optimize if args.worker == "optimize" else verify)(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    config = initial_case()
    if args.resume_from:
        prior = json.loads((args.resume_from / "config.json").read_text())
        if (json.loads(json.dumps(config["spec"])) != prior["spec"] or
                config["initial"] != prior["initial"]):
            raise RuntimeError("resume inputs differ from original case")
        config["resume_from"] = str(args.resume_from.resolve())
    config.update(seed=20260910, evaluation_limit=550, wall_limit_seconds=900,
                  cma_population=8, cma_generations=5, cma_sigma=0.035,
                  tolerance_db=1.5, vcm_vdd_ratio=0.72)
    if args.resume_from:
        config.update(evaluation_limit=330, wall_limit_seconds=600)
    write_json(args.output / "config.json", config)
    before = fingerprints()
    manifest = {"frozen_before": before, "workers": [], "complete": False,
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in ("src/eqrl/pvt_refinement.py", "src/eqrl/experiments/pvt_optimize.py",
                                            "results/pvt_signoff_seed23.json", "results/delivered_circuit.json")}}
    started = time.monotonic()
    write_json(args.output / "manifest.json", manifest)
    try:
        for phase in ("optimize", "verify"):
            remaining = config["wall_limit_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                break
            with (args.output / f"{phase}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, "-m", "eqrl.experiments.pvt_optimize",
                                         "--worker", phase, "--output", str(args.output.resolve())],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    code = "wall_limit"
            manifest["workers"].append({"phase": phase, "returncode": code})
            write_json(args.output / "manifest.json", manifest)
            print(f"{phase}: {code}", flush=True)
            if code != 0:
                break
        manifest["complete"] = len(manifest["workers"]) == 2 and all(w["returncode"] == 0 for w in manifest["workers"])
    finally:
        manifest.update(seconds=time.monotonic() - started, frozen_after=fingerprints(),
                        frozen_unchanged=before == fingerprints())
        write_json(args.output / "manifest.json", manifest)
    if not manifest["complete"] or not manifest["frozen_unchanged"]:
        raise SystemExit("incomplete or changed protected inputs; review manifest")


if __name__ == "__main__":
    main()
