"""Final characterization + deliverable export.

Takes a sized design and produces the submission artifacts:
  - full-spec measurement across the complete PVT grid (fast=False: real HD3 + noise)
  - a pass/fail table per corner
  - the final sized schematic as a SPICE netlist
  - a JSON report

Usage:
  python -m eqrl.experiments.characterize --design results/summary.json --key "PPO (RL)"
  python -m eqrl.experiments.characterize --design mydesign.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.envs.pvt import corner_grid, evaluate_corners
from eqrl.specs import DEFAULT_SPEC, Spec, hard_pass


def load_design(path: str, key: str | None) -> DesignVars:
    data = json.loads(Path(path).read_text())
    if key and key in data:                       # a summary.json from compare.py
        d = data[key]["best_design"]
    elif "best_design" in data:
        d = data["best_design"]
    else:
        d = data                                  # a bare design dict
    return DesignVars(**{k: d[k] for k in DesignVars().__dict__ if k in d})


def characterize(dv: DesignVars, spec: Spec = DEFAULT_SPEC) -> dict:
    """Full-spec eval across the complete PVT grid. Returns a report dict."""
    results = evaluate_corners(dv, spec, mode="full", fast=False)
    rows, all_pass = [], True
    for (proc, vdd, temp), m in results.items():
        passed, _ = hard_pass(m, spec)
        all_pass &= passed
        rows.append({
            "corner": proc, "vdd": round(vdd, 3), "temp_c": temp,
            "ok": m.ok, "passed": passed,
            "boost_db": round(m.boost_db, 2), "fpk_ghz": round(m.peak_freq_ghz, 2),
            "hd3_db": round(m.hd3_db, 1), "noise_uv": round(m.noise_vrms * 1e6, 1),
            "power_mw": round(m.power_w * 1e3, 2), "area_mm2": round(m.area_mm2, 4),
        })
    return {"design": dv.__dict__, "all_pvt_pass": all_pass, "corners": rows}


def print_table(report: dict) -> None:
    hdr = f"{'corner':6} {'vdd':5} {'T':4} {'boost':6} {'fpk':5} {'hd3':6} {'noise':6} {'pwr':6} {'area':7} {'pass'}"
    print(hdr)
    print("-" * len(hdr))
    for r in report["corners"]:
        print(f"{r['corner']:6} {r['vdd']:<5} {r['temp_c']:<4.0f} "
              f"{r['boost_db']:<6} {r['fpk_ghz']:<5} {r['hd3_db']:<6} "
              f"{r['noise_uv']:<6} {r['power_mw']:<6} {r['area_mm2']:<7} "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    print("-" * len(hdr))
    print("ALL PVT PASS" if report["all_pvt_pass"] else "SOME CORNERS FAIL")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True)
    p.add_argument("--key", default=None, help="key inside a summary.json")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)

    dv = load_design(args.design, args.key)
    report = characterize(dv)
    print_table(report)

    Path(f"{args.outdir}/final_report.json").write_text(json.dumps(report, indent=2))
    Path(f"{args.outdir}/final_schematic.spice").write_text(
        netlist(dv, analysis="ac", models="sky130", corner="tt"))
    print(f"\nreport -> {args.outdir}/final_report.json")
    print(f"schematic -> {args.outdir}/final_schematic.spice")


if __name__ == "__main__":
    main()
