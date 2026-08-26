"""G3.2a step 2: probe the frozen low-peak set with the high-peak methodology, unchanged.

THE ONE QUESTION. Does the (rs, l_in) plane remain usable below the band, or does the
physics change qualitatively there? G3.2's Stage A repaired 0 of 3 because its plane was
fitted at 1.90-3.19 GHz and applied at 0.567 and 0.952 GHz -- an out-of-domain inference,
verified in results/g32_why_blocked.log, not a step-sizing problem.

Methodology is IDENTICAL to g32_peak_probe by design: same h = 0.05, same central
difference with a one-sided fallback, same six axes, same sweep-edge exclusion, same
log-space slope. If the method changed, a difference between the two regimes could not be
attributed to the regime. The only difference is where the base points sit, which is the
whole point.

PRE-REGISTERED QUESTIONS AND DECISION RULE, written before the run:

  Q1  is l_in's peak sign FIXED across the low-peak set?
  Q2  is that sign the SAME as the high-peak calibration's (negative)?
  Q3  is rs's peak movement still under one grid step on a majority of the set?
  Q4  what is the median log-gain for l_in below the band, against -1.153 above it?
  Q5  does the constant-table null still hold -- does a constant table pick rs and l_in
      on this set, as it did 3/4 and 4/4 above the band?

  THE PLANE REMAINS USABLE BELOW THE BAND iff Q1 is fixed AND Q2 is the same AND Q3 holds.

  A DIFFERENT GAIN MAGNITUDE DOES NOT FAIL THIS TEST. The controller has measured its own
  gain by secant since the G3.2 fix, so it needs the table for the coordinate and the sign
  only. A MIXED OR FLIPPED SIGN DOES fail it, because then no direction can be chosen at
  all and there is nothing for a secant to correct. Q4 and Q5 are reported either way and
  do not enter the rule -- they are recorded here so that they cannot become the criterion
  after the fact.

  If the rule fails, the finding is that there is no single low-dimensional local plane
  spanning the design space, and G3.2 should not be extended further.

Changes no threshold, bound, reward, model, controller, or benchmark criterion. The G3.2
controller and its seed-3 results are untouched: nothing here writes to them.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{P/'shim'};{P/'Library'/'bin'};{os.environ['PATH']}"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass

DIMS = list(ACTION_SPACE.keys())

#: Identical to g32_peak_probe.PROBE_H. Not a parameter to tune -- a constant to match.
PROBE_H = 0.05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="results/g32a_lowpeak_set.json")
    p.add_argument("--out", default="results/g32a_lowpeak_probe.json")
    args = p.parse_args()

    sel = json.load(open(args.set))
    channel = sel["select_channel_db"]
    target = sel["select_target_db"]
    lo, hi = DEFAULT_SPEC.peak_freq_lo_ghz, DEFAULT_SPEC.peak_freq_hi_ghz
    guard = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                            channel_loss_db=channel)
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                               channel_loss_db=channel)
    n_sim = 0

    def evaluate(x):
        nonlocal n_sim
        n_sim += 1
        try:
            v = guard.evaluate(decode_action(np.asarray(x)), vdd=spec.vdd_nominal)
        except Exception as e:
            return {"valid": False, "guard_check": "exception", "reason": repr(e)[:200]}
        if not v.is_valid:
            return {"valid": False, "guard_check": str(getattr(v.check, "value", v.check)),
                    "reason": str(v.reason)[:200]}
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        return {"valid": True, "boost_db": float(m.boost_db),
                "peak_freq_ghz": float(m.peak_freq_ghz),
                "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                "failing": [c for c, g in checks.items() if not g]}

    print("G3.2a LOW-PEAK PROBE | %d frozen designs from %s | h=%.3f | band %.2f-%.2f GHz"
          % (len(sel["designs"]), args.set, PROBE_H, lo, hi), flush=True)
    print("  methodology identical to g32_peak_probe; only the base points differ\n",
          flush=True)

    # The row key is "spec" and not "draw" so that g32_peak_report.slopes() and
    # axis_table() read this artifact with no change at all -- the high- and low-peak
    # tables must be computed by identical code for the comparison to mean anything.
    # Its VALUE is the draw index in the frozen set; there is no spec index here,
    # because a calibration is a property of the circuit, not of any one spec.
    rows: list[dict] = []

    def write(complete: bool) -> None:
        Path(args.out).write_text(json.dumps(
            {"probe_h": PROBE_H, "source_set": args.set,
             "spec_seed": sel["select_seed"], "select_seed": sel["select_seed"],
             "band_ghz": [lo, hi],
             "channel_loss_db": channel, "target_boost_db": target,
             "n_measure_all": n_sim, "complete": complete, "rows": rows}, indent=1))

    for d in sel["designs"]:
        x0 = np.array(d["x"], dtype=np.float64)
        base = evaluate(x0)
        if not base["valid"]:
            # The frozen set was selected on guard-valid evaluations, so this should not
            # happen; if it does it is a reproducibility fact worth recording, not skipping.
            print("  draw %3d  BASE NO LONGER GUARD-VALID (%s)"
                  % (d["draw"], base["guard_check"]), flush=True)
            rows.append({"spec": d["draw"], "selected": d, "base": base, "dims": None})
            write(False)
            continue

        dims = {}
        for j, name in enumerate(DIMS):
            e = np.zeros(len(DIMS)); e[j] = 1.0
            pts = {}
            for tag, s in (("plus", +PROBE_H), ("minus", -PROBE_H)):
                xt = np.clip(x0 + s * e, 0.0, 1.0)
                pts[tag] = {"moved": float(abs(xt[j] - x0[j])), **evaluate(xt)}
            dims[name] = {"points": pts}
        rows.append({"spec": d["draw"], "selected": d, "base": base, "dims": dims})
        print("  draw %3d  base peak %6.4f GHz  boost %6.2f%s"
              % (d["draw"], base["peak_freq_ghz"], base["boost_db"],
                 "" if base["loose_pass"] else "  fails " + ",".join(base["failing"])),
              flush=True)
        write(False)

    write(True)
    print("\n%d measure_all spent (calibration, not a controller arm)" % n_sim, flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
