"""Gate `g32_selfcal` against the frozen probe artifact. No new SPICE.

WHY THIS CAN BE DONE OFFLINE. `results/g32_peak_probe.json` already stores, for every axis
on every probed spec, the base measurement and the +h / -h measurements -- real ngspice
results, taken at exactly the step `calibrate_plane` uses. That is precisely the data a
runtime calibration would go and collect. So the calibration can be replayed against the
frozen artifact instead of being re-measured: this script serves the stored points back
through `calibrate_plane`'s own `evaluate` interface and checks that what comes out is what
`g32_peak_report.slopes` -- the function the frozen constants were computed with -- gets
from the same points.

THE GATE (`check`). For each probed spec, the plane calibrated by replay must equal, to
within floating-point tolerance, the per-spec entry `slopes()` produces. Passing means the
new module's arithmetic, its usability filters, its one-sided fallback and its span
convention all agree with the code already in the repo. Failing means a frozen-vs-
calibrated comparison would be measuring two formulas against each other rather than two
planes, and must not be run.

THE DESCRIPTION (`spread`). Having replayed each spec, the per-spec slopes are in hand, so
this also reports how far each one sits from the frozen median that replaced it. That is a
property of an artifact that already exists, stated in the units the solver divides by; it
is not a performance result and nothing here compares solve rates. A frozen-vs-calibrated
comparison of solve rates is a different experiment, it produces a headline number, and it
needs its own preregistration before it is run.

    python -m eqrl.experiments.g32_selfcal_gate --probe results/g32_peak_probe.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE
from eqrl.experiments.g32_peak_report import plane_from_probe, slopes
from eqrl.experiments.g32_selfcal import PROBE_H, calibrate_plane

DIMS = list(ACTION_SPACE.keys())

#: plane key -> the per-spec field `slopes()` reports it from, and which axis it lives on.
KEYS = {
    "d_boost_db_per_unit": ("boost", "d_boost"),
    "boost_axis_d_peak_ghz_per_unit": ("boost", "d_peak"),
    "d_peak_ghz_per_unit": ("peak", "d_peak"),
    "d_ln_peak_per_unit": ("peak", "d_ln_peak"),
    "peak_axis_d_boost_db_per_unit": ("peak", "d_boost"),
}
TOL = 1e-9


def replay(row: dict, h: float):
    """An `evaluate` that serves this spec's stored probe points. Returns (evaluate, x0).

    The artifact stores no design vector, only how far each coordinate moved. That is
    enough: put x0 at 0.5 by default, and at the bound for any coordinate whose stored
    point could not move, which reproduces the stored geometry exactly -- including the one
    clipped point in the frozen artifact. Any x the caller asks for that is not a stored
    probe point is a bug in the replay, not a measurement, so it raises rather than
    silently returning a miss.
    """
    x0 = np.full(len(DIMS), 0.5)
    for name, d in row["dims"].items():
        j = DIMS.index(name)
        if d["points"]["plus"]["moved"] == 0.0:
            x0[j] = 1.0
        elif d["points"]["minus"]["moved"] == 0.0:
            x0[j] = 0.0

    def evaluate(x, _target):
        x = np.asarray(x, dtype=np.float64)
        moved = [j for j in range(len(DIMS)) if abs(x[j] - x0[j]) > 1e-12]
        if len(moved) != 1:
            raise AssertionError("replay expects one moved coordinate, got %r" % moved)
        j = moved[0]
        tag = "plus" if x[j] > x0[j] else "minus"
        q = row["dims"][DIMS[j]]["points"][tag]
        if not q["valid"]:
            return None, -1e9, q.get("guard_check", "invalid")
        return dict(q), 0.0, "ok"

    return evaluate, x0


def per_spec(probe: dict, frozen: dict, h: float) -> list[dict]:
    """Replay every probed spec and pair it with `slopes()`' answer for the same spec."""
    ref, _seen, _dropped = slopes(probe)
    out = []
    for row in probe["rows"]:
        if not row.get("dims"):
            continue
        evaluate, x0 = replay(row, h)
        base = dict(row["base"]) if row["base"].get("valid") else None
        plane, spent, info = calibrate_plane(
            evaluate, x0, row.get("target_boost_db", 0.0), frozen,
            base_rec=base, h=h, two_sided=True)
        want = {}
        for axis, name in (("boost", frozen["boost_axis"]), ("peak", frozen["peak_axis"])):
            hits = [e for e in ref.get(name, []) if e["spec"] == row["spec"]]
            want[axis] = hits[0] if hits else None
        out.append({"spec": row["spec"], "plane": plane, "spent": spent,
                    "info": info, "reference": want})
    return out


def check(records: list[dict], frozen: dict) -> list[str]:
    """Every measured entry must match `slopes()`; every fallback must match `frozen`."""
    bad: list[str] = []
    for r in records:
        for key, (axis, field) in KEYS.items():
            got, ref = r["plane"][key], r["reference"][axis]
            measured = key in r["info"]["measured"]
            if measured:
                if ref is None:
                    bad.append("spec %s: %s was measured but slopes() reports no entry "
                               "for that axis" % (r["spec"], key))
                elif abs(got - ref[field]) > TOL * max(1.0, abs(ref[field])):
                    bad.append("spec %s: %s calibrated %.12g, slopes() %.12g"
                               % (r["spec"], key, got, ref[field]))
            elif got != frozen[key]:
                bad.append("spec %s: %s fell back but is not the frozen value "
                           "(%.12g vs %.12g)" % (r["spec"], key, got, frozen[key]))
    return bad


def spread(records: list[dict], frozen: dict) -> dict:
    """How far each spec's measured slope sits from the constant that replaced it."""
    out = {}
    for key in KEYS:
        vals = [r["plane"][key] for r in records if key in r["info"]["measured"]]
        if not vals:
            out[key] = {"n": 0}
            continue
        f = frozen[key]
        rel = [abs(v - f) / abs(f) for v in vals if f != 0.0]
        out[key] = {
            "n": len(vals), "frozen": f,
            "min": min(vals), "median": statistics.median(vals), "max": max(vals),
            "median_abs_rel_dev": statistics.median(rel) if rel else None,
            "max_abs_rel_dev": max(rel) if rel else None,
            "sign_flips_vs_frozen": sum(1 for v in vals if v * f < 0),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", default="results/g32_peak_probe.json")
    ap.add_argument("--out", default="results/g32_selfcal_gate.json")
    args = ap.parse_args()

    probe = json.loads(open(args.probe, encoding="utf-8").read())
    frozen = plane_from_probe(args.probe)

    h = float(probe.get("probe_h", PROBE_H))
    if h != PROBE_H:
        print("NOTE: artifact probe_h=%g, module PROBE_H=%g; replaying at the artifact's "
              "step so the comparison stays like-for-like." % (h, PROBE_H))

    records = per_spec(probe, frozen, h)
    bad = check(records, frozen)
    sp = spread(records, frozen)

    print("replayed %d specs from %s (spec seed %s)"
          % (len(records), args.probe, probe.get("spec_seed")))
    for r in records:
        print("  spec %-3s evals %d  measured %-2d  fellback %-2d  %s"
              % (r["spec"], r["spent"], len(r["info"]["measured"]),
                 len(r["info"]["fellback"]),
                 ",".join(r["info"]["measured"]) or "-"))

    print("\nper-spec measured slope vs the frozen constant that replaced it")
    for key, s in sp.items():
        if not s["n"]:
            print("  %-32s no spec measured this entry" % key)
            continue
        print("  %-32s n=%d frozen %+9.4f  range [%+9.4f, %+9.4f]  "
              "median |rel dev| %s  sign flips %d"
              % (key, s["n"], s["frozen"], s["min"], s["max"],
                 "n/a" if s["median_abs_rel_dev"] is None
                 else "%5.1f%%" % (100 * s["median_abs_rel_dev"]),
                 s["sign_flips_vs_frozen"]))

    open(args.out, "w", encoding="utf-8").write(json.dumps({
        "probe": args.probe, "spec_seed": probe.get("spec_seed"), "probe_h": h,
        "frozen_plane": frozen, "gate_pass": not bad, "violations": bad,
        "spread": sp,
        "per_spec": [{"spec": r["spec"], "spent": r["spent"],
                      "measured": r["info"]["measured"],
                      "fellback": r["info"]["fellback"],
                      "notes": r["info"]["notes"],
                      "plane": {k: r["plane"][k] for k in KEYS}} for r in records],
    }, indent=2))
    print("\nwrote %s" % args.out)

    if bad:
        print("\nGATE FAILED (%d):" % len(bad))
        for b in bad:
            print("  " + b)
        return 1
    print("\nGATE PASSED: replayed calibration reproduces g32_peak_report.slopes exactly, "
          "and every fallback is the frozen value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
