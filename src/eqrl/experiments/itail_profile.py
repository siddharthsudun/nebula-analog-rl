"""Validity as a function of tail current, measured directly on a grid.

Two earlier conclusions about i_tail were artifacts of the same mistake. Both what_binds.py
and the first valid_region.py restricted i_tail with `min(i_tail, cap)`. i_tail is
log-scaled over 0.05-20 mA, so most draws exceed any low cap and collapse onto exactly the
cap value. That atom then dominates the sample:

    cap at 2.0 mA -> atom at 2.0 mA, mostly invalid -> "0.7% valid"
    cap at 0.5 mA -> atom at 0.5 mA, mostly valid   -> "10.7% valid"

which reads as "i_tail <= 0.5 mA fixes it", when what it actually shows is "i_tail near
0.5 mA is a good operating point". Sampling properly inside [0.05, 0.5] mA gives about
2.5%, so the region below 0.5 mA is NOT broadly valid either.

This removes the confound by holding i_tail at fixed grid points and randomising only the
other five variables, so each point is an honest conditional validity estimate. No caps,
no clamps, no atoms.
"""
import collections
import json
import os
from pathlib import Path

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, DesignVars
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass

KEYS = list(ACTION_SPACE)
L_MIN = 0.16e-6          # strictly inside the PDK bound so T2.8 does not mask the rest
PER_POINT = int(os.environ.get("IP_PER_POINT", "40"))
VDD = 1.8

#: Log-spaced from the bottom of the declared range to 8x the value the confounded
#: experiments pointed at, so the band can be located rather than assumed.
I_GRID = [0.05e-3, 0.1e-3, 0.2e-3, 0.35e-3, 0.5e-3, 0.75e-3,
          1.0e-3, 1.5e-3, 2.0e-3, 4.0e-3]

ev = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False)
rng = np.random.default_rng(0)


def draw_others(i_tail):
    """Randomise everything except i_tail, using decode_action's own log/linear rule."""
    vals = {"i_tail": i_tail}
    for k in KEYS:
        if k == "i_tail":
            continue
        lo, hi = ACTION_SPACE[k]
        if k == "l_in":
            lo = L_MIN
        x = float(rng.uniform(0.0, 1.0))
        vals[k] = lo * (hi / lo) ** x if (lo > 0 and hi / lo > 50) else lo + (hi - lo) * x
    return DesignVars(**vals)


print(f"Conditional validity vs tail current, {PER_POINT} random designs per point.\n")
print(f"   {'i_tail':>9}  {'valid':>11}  {'spec-pass':>9}   dominant rejections")
rows = []
for it in I_GRID:
    c = collections.Counter()
    passing = 0
    for _ in range(PER_POINT):
        dv = draw_others(it)
        try:
            v = ev.evaluate(dv, vdd=VDD)
        except Exception as e:
            c[f"EXC:{type(e).__name__}"] += 1
            continue
        if v.is_valid:
            c["__valid__"] += 1
            ok, _ = hard_pass(v.unwrap(), DEFAULT_SPEC)
            passing += int(bool(ok))
        else:
            c[v.check.value] += 1
    n = sum(c.values()) or 1
    frac = c["__valid__"] / n
    top = "  ".join(f"{v}x{k.split('_')[0]}" for k, v in c.most_common()[:2]
                    if k != "__valid__")
    print(f"   {it*1e3:7.2f}mA  {c['__valid__']:3d}/{n} {frac*100:5.1f}%  "
          f"{passing:9d}   {top}", flush=True)
    rows.append({"i_tail": it, "n": n, "valid": c["__valid__"],
                 "fraction": round(frac, 4), "spec_passing": passing,
                 "by_check": dict(c.most_common())})

best = max(rows, key=lambda r: r["fraction"])
print(f"\nPeak conditional validity at i_tail = {best['i_tail']*1e3:.2f} mA "
      f"({best['fraction']*100:.1f}%).")
print("Read the shape, not just the peak: a band is a range worth declaring, a single\n"
      "good point surrounded by bad ones means the topology only works at one bias.")

Path("results").mkdir(exist_ok=True)
Path("results/itail_profile.json").write_text(json.dumps(
    {"per_point": PER_POINT, "l_in_floor": L_MIN, "rows": rows}, indent=2))
print("\nwrote results/itail_profile.json")
