"""The baseline the problem statement actually names: sweeping the parameter space.

docs/PROBLEM.md quotes the poster: the framework "should take significantly lower time than
sweeping all MOS, R, C, L parameter space". Every baseline in honest_benchmark.py (random,
CMA-ES, TPE) is a stronger opponent than that, which is scientifically right but leaves the
officially named comparator unmeasured.

A full sweep at meaningful resolution is not runnable: six parameters at resolutions where
the change matters (~1 um on width, 0.05 um on length, 5% steps on the log-scaled current
and capacitor, 100 ohm on the resistors) is about 2e9 points, roughly 90 years at the
measured 1.64 s per guarded evaluation. So this measures what CAN be run -- a coarse grid --
and reports two things:

  1. the measured cost per grid point, for honest extrapolation to any resolution
  2. whether a coarse grid finds a design that passes all 8 specs AND is a valid circuit

The second is the real question. If a grid coarse enough to be affordable contains no
solution at all, then the sweep baseline does not merely lose on speed: it fails at every
resolution anyone could afford, and succeeds only at resolutions costing years. That is a
stronger and more honest statement than any speedup multiple.

Success uses the same test as honest_benchmark --require-valid: all 8 hard specs at
fast=False AND a guard-valid verdict. Anything weaker would repeat the asymmetry measured
in results/pass_vs_valid.json, where 24 of 28 spec-passing designs were not real circuits.
"""
import itertools
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

import time
import collections
import dataclasses
import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, DesignVars
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass

KEYS = list(ACTION_SPACE)
#: Points per axis. 4**6 = 4096 evaluations, about 1.9 h at the measured rate. 3 would be
#: too coarse to be a fair representation of a sweep; 5 would be 15625 points (~7 h).
PER_AXIS = int(os.environ.get("SWEEP_PER_AXIS", "4"))
#: Meaningful-resolution point count, for the extrapolation. Derived in the docstring.
FULL_RESOLUTION_POINTS = 2e9
TARGET_BOOST = float(os.environ.get("SWEEP_TARGET", "8.0"))
CHANNEL_LOSS = float(os.environ.get("SWEEP_CHANNEL", "12.0"))
VDD = 1.8

spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=TARGET_BOOST,
                           channel_loss_db=CHANNEL_LOSS)
ev = build_evaluator(spec, corner="tt", fast=False)


def axis(key, n):
    """Grid points on one axis, matching decode_action's log/linear treatment."""
    lo, hi = ACTION_SPACE[key]
    if lo > 0 and hi / lo > 50:
        return list(np.geomspace(lo, hi, n))
    return list(np.linspace(lo, hi, n))


grid = [axis(k, PER_AXIS) for k in KEYS]
total = PER_AXIS ** len(KEYS)
print(f"Grid sweep: {PER_AXIS} points on each of {len(KEYS)} axes = {total} evaluations")
print(f"Target: boost {TARGET_BOOST} dB, channel {CHANNEL_LOSS} dB")
print(f"Success = all 8 specs at fast=False AND guard-valid\n")

counts = collections.Counter()
first_success = None
successes = []
t0 = time.time()

for i, combo in enumerate(itertools.product(*grid)):
    dv = DesignVars(**dict(zip(KEYS, combo)))
    try:
        v = ev.evaluate(dv, vdd=VDD)
    except Exception as e:
        counts[f"EXC:{type(e).__name__}"] += 1
        continue
    if not v.is_valid:
        counts[v.check.value] += 1
        continue
    counts["__valid__"] += 1
    ok, _ = hard_pass(v.unwrap(), spec)
    if ok:
        counts["__spec_pass__"] += 1
        successes.append({"index": i + 1, "design": dv.__dict__})
        if first_success is None:
            first_success = i + 1
            print(f"  FIRST SUCCESS at grid point {i+1}", flush=True)
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{total}  valid {counts['__valid__']}  "
              f"spec-pass {counts['__spec_pass__']}  "
              f"{el/(i+1):.2f} s/pt  eta {(total-i-1)*el/(i+1)/60:.0f} min", flush=True)

elapsed = time.time() - t0
per_point = elapsed / total

print(f"\n{'='*66}")
print(f"evaluated      : {total} grid points in {elapsed/60:.1f} min "
      f"({per_point:.2f} s/point)")
print(f"guard-valid    : {counts['__valid__']} ({counts['__valid__']/total*100:.1f}%)")
print(f"pass all 8     : {counts['__spec_pass__']}")
print(f"first success  : {'grid point ' + str(first_success) if first_success else 'NONE'}")
for k, n in counts.most_common():
    if not k.startswith("__"):
        print(f"    {n:5d} ({n/total*100:5.1f}%)  {k}")

print(f"\nExtrapolation at the same cost per point:")
for n in (4, 6, 10, 20):
    pts = n ** len(KEYS)
    print(f"  {n:2d} points/axis = {pts:>12,} points = {pts*per_point/3600:>10,.1f} h")
print(f"  meaningful resolution ~ {FULL_RESOLUTION_POINTS:,.0f} points = "
      f"{FULL_RESOLUTION_POINTS*per_point/3600/24/365:,.1f} years")

if not first_success:
    print(f"\nA {PER_AXIS}-point grid contains NO design that passes the specs and is a")
    print("valid circuit. The sweep baseline does not lose on speed here -- at any")
    print("resolution that is affordable it does not solve the problem at all.")

Path("results").mkdir(exist_ok=True)
Path("results/sweep_baseline.json").write_text(json.dumps({
    "per_axis": PER_AXIS, "total_points": total,
    "target_boost_db": TARGET_BOOST, "channel_loss_db": CHANNEL_LOSS,
    "elapsed_s": round(elapsed, 1), "s_per_point": round(per_point, 4),
    "valid": counts["__valid__"], "spec_pass": counts["__spec_pass__"],
    "first_success_index": first_success,
    "by_check": dict(counts.most_common()),
    "successes": successes[:20],
    "extrapolation_hours": {str(n): (n ** len(KEYS)) * per_point / 3600
                            for n in (4, 6, 10, 20)},
}, indent=2))
print("\nwrote results/sweep_baseline.json")
