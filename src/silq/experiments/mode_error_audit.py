"""Prospective shipped-mode audit. Print protocol by default; --execute is explicit.

Dedicated serial subprocesses prevent simulator/cache leakage between search and
independent verification. No training. No frozen controller/statistic changes.
See docs/PROTOCOL_MODE_ERROR_AUDIT_20260909.md before executing this experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from silq.experiments.audit_support import (AuditBudgetExceeded, bounded_calls,
                                           fingerprint, write_new)

MODELS = ("results/seq_clean40k.zip", "results/seq_seed1_40k.zip",
          "results/seq_seed2_40k.zip")
MODES = ("auto", "fastest", "thinking")
SPEC_SEED = 202609091
N_SPECS = 32
MAX_CALLS = 192
TIMEOUT = 300
TOL = 1.5


def protocol():
    return {"schema": "mode_error_audit_v1", "spec_seed": SPEC_SEED, "n_specs": N_SPECS,
            "models": list(MODELS), "modes": list(MODES), "tol_db": TOL,
            "conditions": ["informed", "target_blind_8db", "target_shuffled"],
            "max_calls_per_search": MAX_CALLS, "max_analyses_per_search": 2000,
            "max_calls_per_external_verification": 1,
            "max_analyses_per_external_verification": 40,
            "child_timeout_seconds": TIMEOUT,
            "max_search_jobs": N_SPECS * len(MODELS) * len(MODES) * 3,
            "max_verify_jobs": N_SPECS * len(MODELS) * len(MODES) * 3,
            "status": "prospective; no observations; historical tolerance unchanged",
            "acceptance": "all 10 hard checks and guarded independent TT verification",
            "hypothesis": "paired advantage vs each target-withheld control; see protocol"}


def child(job, destination):
    """Run only in its own process; no process-global changes escape this job."""
    from silq.experiments import final_comparison  # toolchain setup, no main()
    from silq import pipeline
    from silq.guards import SearchHalted
    from silq.circuits.ctle import DesignVars
    started = time.monotonic()
    data = {"job": job, "complete": False, "result": None, "error": None}
    cost = {"measure_all": 0, "analysis": 0}
    try:
        with bounded_calls(job["budget"], 40 if job["kind"] == "verify" else 2000) as cost:
            if job["kind"] == "verify":
                spec = pipeline.spec_for(job["target"], job["channel"], TOL)
                # Fresh process and fresh evaluator; this is deliberately in addition
                # to pipeline.design's existing independent-in-process verifier.
                data["result"] = pipeline.verify(DesignVars(**job["design"]), spec)
            else:
                # pvt=False: the protocol (docs/PROTOCOL_MODE_ERROR_AUDIT_20260909.md)
                # audits search-stage error rates across auto/fastest/thinking under
                # the fixed `bounded_calls` budget above. PVT repair (pipeline.design's
                # default now) spends its own uncounted-by-this-audit corner
                # evaluations and would both blow the analysis budget unpredictably and
                # confound the per-mode comparison with a stage none of the modes
                # control.
                data["result"] = pipeline.design(
                    job["search_target"], job["channel"], model=job["model"],
                    mode=job["mode"], spec_index=job["spec_index"], tol=TOL,
                    allow_fallback=False, pvt=False)
            data["complete"] = True
    except (SearchHalted, AuditBudgetExceeded) as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
    data["actual_cost"] = cost
    data["wall_seconds_observed"] = time.monotonic() - started
    write_new(destination, data)


def dispatch(job, root, name):
    input_path, output = root / f"{name}_job.json", root / f"{name}.json"
    write_new(input_path, job)
    with (root / f"{name}.log").open("x", encoding="utf-8") as log:
        try:
            r = subprocess.run([sys.executable, "-m", "silq.experiments.mode_error_audit",
                                "--execute", "--worker-job", str(input_path),
                                "--worker-out", str(output)], stdout=log,
                               stderr=subprocess.STDOUT, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"job": job, "complete": False, "result": None,
                    "error": "subprocess timeout; partial cost unknown"}
    if r.returncode != 0 or not output.exists():
        return {"job": job, "complete": False, "result": None,
                "error": f"worker exit {r.returncode}; see log"}
    return json.loads(output.read_text(encoding="utf-8"))


def run(root):
    import numpy as np
    # Identical draw distribution/order to target_audit.make_specs; no PDK import
    # required in the parent. This set is first generated at authorized execution.
    rng = np.random.default_rng(SPEC_SEED)
    specs = [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16))) for _ in range(N_SPECS)]
    # Random cycle: a permutation with no fixed points, retaining the target
    # multiset. Channels and spec_index remain attached to the original row.
    order = np.random.default_rng(SPEC_SEED + 1).permutation(N_SPECS)
    shuffled = np.empty(N_SPECS, dtype=int)
    shuffled[order] = np.roll(order, 1)
    manifest = protocol()
    inputs = [*MODELS, *(str(Path(m).with_name(Path(m).stem + "_train.json")) for m in MODELS),
              "results/surrogate_corpus.npz", "results/target_tracking_clean40k.json",
              "results/g32_peak_probe.json", "results/g32_rescue_probe.json",
              "src/silq/pipeline.py", "src/silq/experiments/fastest_hedge.py",
              "src/silq/experiments/final_comparison.py", "src/silq/sim/eye.py",
              "src/silq/guards.py", "src/silq/specs.py", __file__]
    # All prerequisites checked before the first simulator worker.
    manifest["inputs"] = [fingerprint(p) for p in inputs]
    manifest["specs"] = specs
    manifest["shuffled_target_indices"] = shuffled.tolist()
    manifest["training"] = [json.loads(Path(m).with_name(Path(m).stem + "_train.json")
                                         .read_text(encoding="utf-8")) for m in MODELS]
    if [t["seed"] for t in manifest["training"]] != [0, 1, 2]:
        raise ValueError("training seed manifest mismatch")
    comparable = ("timesteps", "fast", "guarded", "shaped_invalid", "pvt", "horizon",
                  "n_envs", "step_size", "target_range", "channel_range",
                  "feasible_decode", "anchor_baseline", "anchor_noise")
    if any(t[k] != manifest["training"][0][k]
           for t in manifest["training"] for k in comparable):
        raise ValueError("policy training configs differ; prospectively amend protocol")
    write_new(root / "manifest.json", manifest)
    rows = []
    for i, (target, channel) in enumerate(specs):
        for model in MODELS:
            for mode in MODES:
                cap = MAX_CALLS
                for condition in manifest["conditions"]:
                    search_target = (target if condition == "informed" else
                                     8.0 if condition == "target_blind_8db" else
                                     specs[int(shuffled[i])][0])
                    job = {"kind": "search", "spec_index": i, "target": target,
                           "channel": channel, "model": model, "mode": mode,
                           "condition": condition, "search_target": search_target,
                           "budget": cap}
                    name = f"s{i:02d}_{Path(model).stem}_{mode}_{condition}"
                    search = dispatch(job, root, name)
                    result = search.get("result") or {}
                    verification = None
                    if result.get("design") is not None:
                        verify_job = {**job, "kind": "verify", "budget": 1,
                                      "design": result["design"]}
                        verification = dispatch(verify_job, root, name + "_verify")
                    rows.append({"search": search, "independent_verification": verification})
                    if condition == "informed":
                        # Controls use this mode/spec/model's observed total deployment
                        # measure_all budget, including its internal verification.
                        # Unknown timeout cost stays explicitly unmatched, not imputed.
                        actual = search.get("actual_cost")
                        if actual is None:
                            rows[-1]["controls_omitted"] = "informed cost unknown"
                            break
                        cap = actual["measure_all"]
    drift = [r["path"] for r in manifest["inputs"]
             if fingerprint(r["path"])["sha256"] != r["sha256"]]
    write_new(root / "observations.json", {"manifest": manifest, "rows": rows,
              "input_drift": drift, "complete_schedule": len(rows) == manifest["max_search_jobs"],
              "eligible_for_inference": not drift and all(r["search"]["complete"] for r in rows)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--out-dir", default="results/mode_error_audit_20260909")
    p.add_argument("--worker-job", help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker_job:
        if not a.execute or not a.worker_out:
            p.error("worker requires --execute and --worker-out")
        child(json.loads(Path(a.worker_job).read_text(encoding="utf-8")), a.worker_out)
        return
    print(json.dumps(protocol(), indent=2))
    if a.execute:
        root = Path(a.out_dir)
        root.mkdir(parents=True, exist_ok=False)
        run(root)


if __name__ == "__main__":
    main()
