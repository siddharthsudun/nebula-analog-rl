"""Evaluate the measured feasibility library on the held-out 32-spec set.

This is a feasibility diagnostic, not a replacement for the RL policy.  It answers a
high-value question before another long training run: does one nominally valid design
remain valid when the requested target and channel loss are varied over the exact
evaluation distribution?  If so, it is a useful teacher/initialization candidate.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "silq-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

from silq.circuits.ctle import DesignVars
from silq.evaluator import build_evaluator
from silq.specs import DEFAULT_SPEC, hard_pass


def held_out_specs(n: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(0)
    return [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16))) for _ in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--library", default="results/feasibility_band.json")
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--index", type=int, default=None,
                   help="evaluate one library entry; omit to evaluate every entry")
    p.add_argument("--fast", action="store_true",
                   help="use the fast evaluator; final claims should use --no-fast")
    p.add_argument("--out", default="results/robust_library.json")
    args = p.parse_args()

    library = json.loads(Path(args.library).read_text())
    rows = library["both"]
    if args.index is not None:
        rows = [rows[args.index]]

    specs = held_out_specs(args.specs)
    evaluators: dict[float, object] = {}
    results = []
    for li, row in enumerate(rows):
        d = DesignVars(**{k: float(v) for k, v in row["design"].items() if k != "w_dfe"})
        solved = 0
        failures = []
        for si, (target, channel) in enumerate(specs):
            if channel not in evaluators:
                evaluators[channel] = build_evaluator(DEFAULT_SPEC, corner="tt",
                                                       fast=args.fast,
                                                       channel_loss_db=channel)
            spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                       channel_loss_db=channel)
            verdict = evaluators[channel].evaluate(d, vdd=spec.vdd_nominal)
            if not verdict.is_valid:
                failures.append({"spec": si, "guard": verdict.check.value})
                continue
            ok, checks = hard_pass(verdict.unwrap(), spec)
            if ok:
                solved += 1
            else:
                failures.append({"spec": si,
                                 "checks": [k for k, passed in checks.items() if not passed]})
            print(f"library {li:2d} spec {si:2d}: {'SOLVED' if ok else 'failed'}",
                  flush=True)
        results.append({"library_index": li, "solved": solved, "total": len(specs),
                        "design": row["design"], "failures": failures})
        print(f"library {li:2d}: solved {solved}/{len(specs)}", flush=True)

    payload = {"library": args.library, "fast": args.fast, "specs": len(specs),
               "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    best = max(results, key=lambda r: r["solved"], default=None)
    if best:
        print(f"best fixed design: {best['solved']}/{best['total']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
