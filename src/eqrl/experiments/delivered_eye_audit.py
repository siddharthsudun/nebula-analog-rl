"""Task 6: prospective 45-corner comparison, NEVER an optimizer.

Default invocation prints a plan only. --execute runs one serial child process per
process-corner group, with a 300-second deadline each: at most 25 minutes of child
time (not an expected runtime), plus launch/file overhead. Each child permits 18
full measure_all calls and 300 SPICE analyses. Actual counts/timing are recorded.

Two freshly guarded paths per corner: frozen WORKING-TREE compute_eye (not old
HEAD) and compute_eye_v2, each with real HD3/noise. Archived legacy values are a
third, clearly labelled column. V2 enters the existing guard before any metric can
be exposed as valid. Existing artifacts/rewards/thresholds are not changed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import time

from eqrl.experiments.audit_support import (AuditBudgetExceeded, bounded_calls,
                                           fingerprint, write_new)

PROCESSES = ("tt", "ss", "ff", "sf", "fs")
SOURCE = "results/delivered_circuit.json"
OUT = "results/delivered_eye_audit_v2_20260909"


def audit_one(dv, spec, proc, vdd, temp, version, store):
    """Fresh guard. Invalid deliberately has no metrics; never unwrap it."""
    import numpy as np
    from eqrl.evaluator import build_evaluator
    from eqrl.guards import ArtifactStore, SearchHalted
    from eqrl.sim.eye import compute_eye_v2
    from eqrl.sim.server import get_server
    from eqrl.specs import hard_pass
    ev = build_evaluator(spec, corner=proc, temp_c=temp, fast=False,
                         channel_loss_db=spec.channel_loss_db,
                         store=ArtifactStore(store))
    eye_details = {}
    if version == "audited_v2":
        original = ev.raw_eval

        def raw_v2(candidate, *, artifacts, vdd, **kwargs):
            m, art, op = original(candidate, artifacts=artifacts, vdd=vdd, **kwargs)
            # A separate AC acquisition is charged and retained. V2 replaces only
            # eye measurements BEFORE GuardedEvaluator applies its frozen tiers.
            ac = get_server(proc).ac_complex(candidate, vdd=vdd, temp_c=temp)
            e = compute_eye_v2(ac["freq"], ac["H"],
                               channel_loss_db=spec.channel_loss_db)
            eye_details.update(signed_opening_v=e.signed_opening_v, errors=e.errors,
                               count=e.count, ber=e.ber, sample_phase=e.sample_phase)
            art.meta["eye_audit_version"] = "audited_v2"
            art.meta["eye_audit"] = eye_details.copy()
            # Preserve measured complex response, including any run later rejected.
            dest = art.directory / "audited_eye_ac.data"
            np.savetxt(dest, np.column_stack((ac["freq"], ac["H"].real, ac["H"].imag)))
            art.data_files["audited_eye_ac.data"] = dest
            return dataclasses.replace(m, eye_v_mv=1000 * e.height_v,
                                       eye_h_ui=e.width_ui), art, op

        ev.raw_eval = raw_v2
    try:
        verdict = ev.evaluate(dv, vdd=vdd)
    except SearchHalted as exc:
        return {"version": version, "guard_valid": False, "pass10": False,
                "halted": str(exc)}
    if not verdict.is_valid:
        return {"version": version, "guard_valid": False, "pass10": False,
                "guard_check": verdict.check.value, "reason": verdict.reason,
                "artifact_dir": str(verdict.artifact_dir)}
    m = verdict.unwrap()
    passed, checks = hard_pass(m, spec)
    return {"version": version, "guard_valid": True, "pass10": bool(passed),
            "checks": {k: bool(v) for k, v in checks.items()}, "metrics": m.as_dict(),
            "eye_audit": eye_details or None, "artifact_dir": str(verdict.artifact_dir)}


def worker(proc, directory):
    from eqrl.circuits.ctle import DesignVars
    from eqrl.envs.pvt import corner_grid
    from eqrl.specs import DEFAULT_SPEC
    # Use the documented bootstrap without importing/running a benchmark main().
    from eqrl.experiments import final_comparison  # noqa: F401
    d = json.loads(Path(SOURCE).read_text(encoding="utf-8"))
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=d["spec"]["target_boost_db"],
                               channel_loss_db=d["spec"]["channel_loss_db"],
                               boost_target_tol_db=1.5)
    grid = corner_grid(spec, mode="full")
    if len(grid) != 45 or set(p for p, _, _ in grid) != set(PROCESSES):
        raise ValueError("grid drift: this audit requires the delivered 45 corners")
    dv = DesignVars(**d["design"])
    rows, started, problem = [], time.monotonic(), None
    try:
        with bounded_calls(18, 300) as cost:
            for p, v, t in grid:
                if p != proc:
                    continue
                key = f"{p}|{v:.3f}|{t:g}"
                row = {"corner": key, "archived_legacy": d["pvt"]["corners"][key]}
                for version in ("frozen_working_tree", "audited_v2"):
                    row[version] = audit_one(dv, spec, p, v, t, version,
                                              Path(directory) / "raw" / key.replace("|", "_") / version)
                rows.append(row)
                write_new(Path(directory) / f"{proc}_{len(rows):02d}.json", row)
    except (AuditBudgetExceeded, Exception) as exc:
        problem = f"{type(exc).__name__}: {exc}"
    write_new(Path(directory) / f"{proc}_summary.json",
              {"complete": len(rows) == 9 and problem is None, "rows": rows,
               "cost": cost, "wall_seconds_observed": time.monotonic() - started,
               "error": problem})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--out-dir", default=OUT)
    p.add_argument("--worker", choices=PROCESSES, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if not a.execute:
            p.error("worker requires --execute")
        worker(a.worker, a.out_dir)
        return
    plan = {"generator": "eqrl.experiments.delivered_eye_audit", "corners": 45,
            "versions": ["frozen_working_tree", "audited_v2"],
            "max_measure_all": 90, "max_analyses": 1500,
            "child_timeout_seconds": 300, "max_child_seconds": 1500,
            "expected_wall_seconds": None,
            "expected_cost_formula": "5 model loads + 90 full measurements + 45 AC/v2 computations",
            "scope": "ideal linear behavioural eye; no silicon or low-BER certification"}
    print(json.dumps(plan, indent=2))
    if not a.execute:
        return
    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    plan["inputs"] = [fingerprint(n) for n in
                      (SOURCE, "src/eqrl/sim/eye.py", "src/eqrl/sim/measures.py",
                       "src/eqrl/guards.py", "src/eqrl/specs.py", __file__)]
    write_new(root / "plan.json", plan)
    outcomes = []
    for proc in PROCESSES:
        with (root / f"{proc}.log").open("x", encoding="utf-8") as log:
            try:
                r = subprocess.run([sys.executable, "-m", __spec__.name, "--execute",
                                    "--worker", proc, "--out-dir", str(root)],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=300)
                outcomes.append({"process": proc, "returncode": r.returncode})
            except subprocess.TimeoutExpired:
                outcomes.append({"process": proc, "timeout": True})
    parts = []
    for proc in PROCESSES:
        f = root / f"{proc}_summary.json"
        if f.exists():
            parts.append(json.loads(f.read_text(encoding="utf-8")))
    rows = [r for part in parts for r in part["rows"]]
    complete = len(rows) == 45 and all(part["complete"] for part in parts)
    counts = {ver: sum(r[ver]["guard_valid"] and r[ver]["pass10"] for r in rows)
              for ver in plan["versions"]}
    write_new(root / "comparison.json", {"plan": plan, "complete": complete,
              "rows": rows, "corners_recorded": len(rows), "corners_required": 45,
              "corners_pass10": counts, "processes": outcomes,
              "pvt_clean": {ver: complete and n == 45 for ver, n in counts.items()}})


if __name__ == "__main__":
    main()
