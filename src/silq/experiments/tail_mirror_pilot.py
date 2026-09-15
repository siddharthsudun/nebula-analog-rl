"""Bounded matched-design tail comparison, isolated from the shipped singleton.

Run: python -m silq.experiments.tail_mirror_pilot --output results/<new-directory>
Screen: at most 36 full evaluations, 600 seconds for workers combined.
Refine: at most 24 full evaluations, 300 seconds. Confirm: six, 120 seconds.
No optimizer, training, checkpoint promotion, or claims of PVT sign-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

from silq.circuits import ctle
from silq.circuits.tail_variants import (
    TailConfig, TailDesign, instances, nodes, param_deck, parameters, snapshot, tail_devices,
)

ROOT = Path(__file__).resolve().parents[3]
FROZEN = (
    "results/seq_clean40k.zip", "results/delivered_circuit.json",
    "src/silq/guards.py", "src/silq/sim/eye.py",
    "src/silq/experiments/final_report.py", "src/silq/specs.py",
    "src/silq/envs/equalizer_env.py", "src/silq/circuits/ctle.py",
)
CONFIGS = {
    "simple": TailConfig(),
    "wide_swing_1x": TailConfig("wide_swing", 1),
    "wide_swing_4x": TailConfig("wide_swing", 4),
    "wide_swing_4x_bias16": TailConfig("wide_swing", 4, 0.16),
    "wide_swing_4x_bias14": TailConfig("wide_swing", 4, 0.14),
    "wide_swing_8x_bias16": TailConfig("wide_swing", 8, 0.16),
    "wide_swing_8x_bias14": TailConfig("wide_swing", 8, 0.14),
    "wide_swing_8x_bias12": TailConfig("wide_swing", 8, 0.12),
}
RATIOS = (0.60, 0.64, 0.68, 0.72)
PHASES = {
    "screen": (tuple(CONFIGS)[:3], RATIOS, 600),
    "refine": (tuple(CONFIGS)[3:7], (0.68, 0.72), 300),
    "confirm": (("wide_swing_8x_bias12",), (0.68, 0.72), 120),
}


def fingerprints():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FROZEN}


def samples():
    delivered = json.loads((ROOT / "results/delivered_circuit.json").read_text())
    sweep = json.loads((ROOT / "results/vcm_headroom_sweep.json").read_text())
    result = [("delivered", delivered["design"])]
    for ratio in (0.68, 0.72):
        row = next(r for r in sweep["rows"] if r["vcm_ratio"] == ratio)
        result.append((f"historical_label_{ratio}", row["best"]["design"]))
    return result


def write_json(path, data):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported result type: {type(value).__name__}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=scalar), encoding="utf-8")
    tmp.replace(path)


def worker(label: str, output: Path, ratios=RATIOS):
    # Import only inside a worker: libngspice is one singleton per process.
    from silq.sim.server import NgspiceServer
    from silq.sim.probe import ProbeError, probe_operating_point
    from silq.evaluator import build_evaluator
    from silq.guards import ArtifactStore
    from silq.specs import DEFAULT_SPEC, hard_pass

    config = CONFIGS[label]

    class TailServer(NgspiceServer):
        vcm_ratio = 0.72
        last_op = None

        def _load(self, corner):
            deck = self._dir / f"tail_{corner}.cir"
            deck.write_text(param_deck(corner, config), encoding="utf-8")
            self._ng.exec_command(f"source {deck}")
            self._ng.exec_command("reset")
            self._ng.exec_command("op")

        def _prime(self, dv, vdd, temp_c):
            if not isinstance(dv, TailDesign) or dv.tail != config:
                raise ValueError("candidate tail does not match loaded deck")
            self.expected_cm = vdd * self.vcm_ratio
            self.last_params = parameters(dv, vdd=vdd, temp_c=temp_c, vcm_ratio=self.vcm_ratio)
            for k, value in self.last_params.items():
                self._ng.exec_command(f"alterparam {k}={value:.6g}")
            self._ng.exec_command("reset")

    def read_op(srv):
        op = probe_operating_point(srv, instances=instances(config), nodes=nodes(config),
                                   tail_devices=tail_devices(config))
        srv.last_op = op
        actual_cm = op.node_voltages["cm"]
        if abs(actual_cm - srv.expected_cm) > 1e-5:
            raise ProbeError(f"VCM mismatch: requested {srv.expected_cm}, measured {actual_cm}")
        return op

    started = time.monotonic()
    report = {"label": label, "config": asdict(config), "corner": "tt",
              "vdd": 1.8, "temp_c": 27, "fast": False, "rows": []}
    path = output / f"{label}.json"
    with TailServer("tt") as srv:
        ev = build_evaluator(
            DEFAULT_SPEC, corner="tt", fast=False,
            server_factory=lambda corner: srv,
            store=ArtifactStore(output / "raw" / label),
            operating_point_probe=read_op,
            netlist_builder=lambda dv, **kw: snapshot(dv, vcm_ratio=srv.vcm_ratio, **kw),
        )
        for sample, design in samples():
            dv = TailDesign(**design, tail=config)
            for ratio in ratios:
                srv.vcm_ratio, srv.last_op = ratio, None
                verdict = ev.evaluate(dv, vdd=1.8)
                row = {"sample": sample, "design": asdict(dv), "vcm_ratio": ratio,
                       "requested_cm_v": ratio * 1.8, "guard_valid": verdict.is_valid,
                       "artifact_dir": str(verdict.artifact_dir),
                       "simulator_parameters": srv.last_params}
                if srv.last_op is not None:
                    op = srv.last_op
                    row["health"] = {
                        "measured_cm_v": op.node_voltages["cm"],
                        "nodes_v": dict(op.node_voltages),
                        "all_saturated": all(d.saturated for d in op.devices),
                        "worst_headroom_mv": min(d.headroom for d in op.devices) * 1e3,
                        "tail_delivered_pct": 100 * sum(abs(v) for v in op.tail_currents.values()) / dv.i_tail,
                        "devices": [dict(asdict(d), headroom_mv=d.headroom * 1e3,
                                         saturated=d.saturated) for d in op.devices],
                    }
                if verdict.is_valid:
                    m = verdict.unwrap()
                    passed, checks = hard_pass(m, DEFAULT_SPEC)
                    row.update(measures=asdict(m), spec_passed=passed, checks=checks,
                               in_band=1.25 <= m.peak_freq_ghz <= 2.5)
                else:
                    row.update(rejection=verdict.check.value, reason=verdict.reason)
                report["rows"].append(row)
                report["seconds"] = time.monotonic() - started
                write_json(path, report)
                print(json.dumps({"variant": label, "sample": sample, "vcm": ratio,
                                  "valid": verdict.is_valid,
                                  "headroom_mv": row.get("health", {}).get("worst_headroom_mv"),
                                  "boost_db": row.get("measures", {}).get("boost_db"),
                                  "rejection": row.get("rejection")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=CONFIGS)
    parser.add_argument("--phase", choices=PHASES, default="screen")
    args = parser.parse_args()
    labels, ratios, wall_limit = PHASES[args.phase]
    if args.worker:
        worker(args.worker, args.output, ratios)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    before = fingerprints()
    manifest = {"eval_limit": len(labels) * len(ratios) * 3, "wall_limit_seconds": wall_limit,
                "phase": args.phase, "variants": labels, "vcm_ratios": ratios,
                "method": "matched preselected designs, measured VCM, explicit tail sizing",
                "scope": "TT only; no optimization; historical labels are not verified historical VCM values",
                "source_artifact_sha256": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in ("results/delivered_circuit.json", "results/vcm_headroom_sweep.json")},
                "code_sha256": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in ("src/silq/circuits/tail_variants.py", "src/silq/evaluator.py",
                              "src/silq/experiments/tail_mirror_pilot.py")},
                "frozen_before": before, "workers": []}
    write_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    deadline = started + wall_limit
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        for label in labels:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            command = [sys.executable, "-m", "silq.experiments.tail_mirror_pilot",
                       "--output", str(args.output.resolve()), "--worker", label,
                       "--phase", args.phase]
            with (args.output / f"{label}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, cwd=ROOT)
                try:
                    code = proc.wait(timeout=remaining)
                    status = "complete" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    code, status = proc.returncode, "wall_limit"
            manifest["workers"].append({"label": label, "status": status, "returncode": code})
            write_json(args.output / "manifest.json", manifest)
            print(f"{label}: {status}", flush=True)
            if status != "complete":
                break
    finally:
        manifest["seconds"] = time.monotonic() - started
        manifest["frozen_after"] = fingerprints()
        manifest["frozen_unchanged"] = before == manifest["frozen_after"]
        manifest["complete"] = (len(manifest["workers"]) == len(labels) and
                                all(w["status"] == "complete" for w in manifest["workers"]))
        write_json(args.output / "manifest.json", manifest)
    if not manifest["frozen_unchanged"]:
        raise RuntimeError("frozen file changed during the pilot; review hashes")
    if not manifest["complete"]:
        raise SystemExit("pilot incomplete; inspect manifest and worker logs")


if __name__ == "__main__":
    main()
