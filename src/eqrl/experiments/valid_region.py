"""Let the data draw the valid region instead of guessing restrictions.

what_binds.py tested hand-picked restrictions one at a time, and two of my guesses were
wrong (R_load projection, i_tail <= 2 mA both changed nothing). This does the opposite:
sample the space, keep whatever comes back valid, and report where the valid designs
actually live in each coordinate.

Sampling is restricted to i_tail <= I_MAX because the unrestricted space is 0.4% valid,
which yields too few valid designs to say anything about the other five variables. That
restriction is the one measured finding, not a guess.

Outputs the marginal range of every variable among valid designs, plus a candidate box,
so the ranges can be chosen from evidence.
"""
import collections
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

import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass

KEYS = list(ACTION_SPACE)
N = len(KEYS)
I_MAX = 0.5e-3           # the one measured restriction (what_binds.py: 0.7% -> 10.7%)
L_MIN = 0.16e-6          # strictly inside the PDK bound, so T2.8 does not mask the rest
SAMPLES = int(os.environ.get("VR_SAMPLES", "700"))
SEED = int(os.environ.get("VR_SEED", "0"))
VDD = 1.8

ev = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False)
rng = np.random.default_rng(SEED)

valid_rows, invalid_rows = [], []
checks = collections.Counter()
spec_passing = 0

for i in range(SAMPLES):
    x = rng.uniform(0.0, 1.0, N)
    dv = decode_action(x)
    dv = dataclasses.replace(dv, i_tail=min(dv.i_tail, I_MAX), l_in=max(dv.l_in, L_MIN))
    try:
        v = ev.evaluate(dv, vdd=VDD)
    except Exception as e:
        checks[f"EXC:{type(e).__name__}"] += 1
        invalid_rows.append(dv)
        continue
    if v.is_valid:
        checks["__valid__"] += 1
        valid_rows.append(dv)
        m = v.unwrap()
        ok, _ = hard_pass(m, DEFAULT_SPEC)
        spec_passing += int(bool(ok))
    else:
        checks[v.check.value] += 1
        invalid_rows.append(dv)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{SAMPLES}  valid so far: {len(valid_rows)}", flush=True)

n = len(valid_rows) + len(invalid_rows)
print(f"\nsamples {n}, VALID {len(valid_rows)} ({len(valid_rows)/n*100:.1f}%), "
      f"of which pass all 8 specs at the default target: {spec_passing}")
for k, c in checks.most_common():
    if k != "__valid__":
        print(f"   {c:4d} ({c/n*100:5.1f}%)  {k}")

if len(valid_rows) < 20:
    print("\nToo few valid designs to describe the region. Raise VR_SAMPLES or lower I_MAX.")
    raise SystemExit(0)


def col(rows, key):
    return np.array([getattr(d, key) for d in rows], dtype=float)


print(f"\nWhere the {len(valid_rows)} VALID designs live, per variable:")
print(f"   {'param':8s} {'declared range':>26}   {'valid p5..p95':>26}  {'shrink':>7}")
proposal = {}
for k in KEYS:
    lo, hi = ACTION_SPACE[k]
    v = col(valid_rows, k)
    p5, p95 = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    declared_span = hi - lo
    shrink = (p95 - p5) / declared_span if declared_span else 1.0
    proposal[k] = [p5, p95]
    print(f"   {k:8s} {lo:11.4g} .. {hi:<11.4g}   {p5:11.4g} .. {p95:<11.4g}  "
          f"{shrink*100:6.1f}%")

print("\nA variable whose valid range still spans most of the declared range is not the "
      "problem;\nthe ones that collapse are where the constraint actually is.")

Path("results").mkdir(exist_ok=True)
Path("results/valid_region.json").write_text(json.dumps({
    "samples": n,
    "valid": len(valid_rows),
    "valid_fraction": round(len(valid_rows) / n, 4),
    "spec_passing_among_valid": spec_passing,
    "i_tail_cap": I_MAX,
    "l_in_floor": L_MIN,
    "by_check": dict(checks.most_common()),
    "valid_p5_p95": {k: [round(a, 12), round(b, 12)] for k, (a, b) in proposal.items()},
    "declared": {k: list(v) for k, v in ACTION_SPACE.items()},
}, indent=2))
print("\nwrote results/valid_region.json")
