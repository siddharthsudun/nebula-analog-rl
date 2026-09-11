"""What fraction of the declared action space is a physically valid circuit?

Training with guards on rejected 97.4% of candidates. That is either the guard being
wrong or the search space being mostly unbuildable, and the two call for opposite
responses. This samples the box uniformly -- no agent, no trajectory -- so the answer is
a property of ACTION_SPACE alone.

Also reports, for the invalid ones, how far outside they are, because "i_tail x r_load
needs 15 V on a 1.8 V rail" is not a near miss the agent could learn its way out of.
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

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC

N = len(ACTION_SPACE)
VDD = 1.8
SAMPLES = 250

rng = np.random.default_rng(0)
ev = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False)

counts = collections.Counter()
drop_of_invalid, drop_of_valid = [], []

for i in range(SAMPLES):
    x = rng.uniform(0.0, 1.0, N)
    dv = decode_action(x)
    # DC drop the loads demand of the supply, before any simulation
    drop = (dv.i_tail / 2.0) * dv.r_load
    try:
        v = ev.evaluate(dv, vdd=VDD)
    except Exception as e:
        counts[f"EXC:{type(e).__name__}"] += 1
        continue
    if v.is_valid:
        counts["__valid__"] += 1
        drop_of_valid.append(drop)
    else:
        counts[v.check.value] += 1
        drop_of_invalid.append(drop)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{SAMPLES} ...", flush=True)

total = sum(counts.values())
valid = counts["__valid__"]
print(f"\nUniform samples of ACTION_SPACE: {total}")
print(f"VALID: {valid}  ({valid/total*100:.1f}%)\n")
for k, n in counts.most_common():
    if k != "__valid__":
        print(f"  {n:4d}  ({n/total*100:5.1f}%)  {k}")

drop_all = np.array(drop_of_invalid + drop_of_valid)
print(f"\nDC load drop (I_leg x R_load) demanded, against a {VDD} V rail:")
print(f"  median over the whole box : {np.median(drop_all):8.2f} V")
print(f"  fraction needing > VDD    : {float((drop_all > VDD).mean())*100:5.1f}%")
if drop_of_valid:
    print(f"  median among VALID        : {np.median(drop_of_valid):8.2f} V")

# How much of the box is excluded by that one constraint alone?
grid = rng.uniform(0.0, 1.0, (200_000, N))
dvs = [decode_action(g) for g in grid[:0]]          # keep decode semantics, avoid cost
lo_i, hi_i = ACTION_SPACE["i_tail"]
lo_r, hi_r = ACTION_SPACE["r_load"]
it = lo_i + grid[:, list(ACTION_SPACE).index("i_tail")] * (hi_i - lo_i)
rl = lo_r + grid[:, list(ACTION_SPACE).index("r_load")] * (hi_r - lo_r)
feas = ((it / 2.0) * rl) <= (VDD - 0.5)
print(f"\nAnalytic (200k points, i_tail x r_load only): "
      f"{feas.mean()*100:.1f}% of the box leaves >= 0.5 V of output headroom.")

Path("results").mkdir(exist_ok=True)
Path("results/space_validity.json").write_text(json.dumps({
    "samples": total,
    "valid": valid,
    "valid_fraction": round(valid / total, 4),
    "by_check": dict(counts.most_common()),
    "median_load_drop_v": float(np.median(drop_all)),
    "fraction_drop_over_vdd": float((drop_all > VDD).mean()),
    "analytic_headroom_fraction": float(feas.mean()),
}, indent=2))
print("\nwrote results/space_validity.json")
