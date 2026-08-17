"""Projecting R_load onto the supply barely helped (0.4% -> 0.5%). So what does bind?

Tests one restriction at a time against the same guard, so the answer is measured rather
than reasoned. Nothing here is applied to the project; it exists to tell us which range
is actually responsible before anyone changes a range.
"""
import collections
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

from eqrl.circuits.ctle import ACTION_SPACE, decode_action, project_feasible
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC

N = len(ACTION_SPACE)
SAMPLES = 150
ev = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False)


def sample(rng, i_max=None, l_min=None, project=False):
    dv = decode_action(rng.uniform(0.0, 1.0, N))
    import dataclasses
    if i_max is not None:
        dv = dataclasses.replace(dv, i_tail=min(dv.i_tail, i_max))
    if l_min is not None:
        dv = dataclasses.replace(dv, l_in=max(dv.l_in, l_min))
    if project:
        dv = project_feasible(dv, vdd=1.8)
    return dv


VARIANTS = [
    ("baseline (declared space)",              dict()),
    ("R_load projected onto supply",           dict(project=True)),
    ("i_tail <= 2 mA",                         dict(i_max=2e-3)),
    ("i_tail <= 2 mA + projection",            dict(i_max=2e-3, project=True)),
    ("i_tail <= 2 mA + l_in >= 0.16 um",       dict(i_max=2e-3, l_min=0.16e-6)),
    ("i_tail <= 0.5 mA + l_in >= 0.16 um",     dict(i_max=0.5e-3, l_min=0.16e-6)),
]

out = {}
for label, kw in VARIANTS:
    rng = np.random.default_rng(0)          # same stream for every variant
    c = collections.Counter()
    for _ in range(SAMPLES):
        dv = sample(rng, **kw)
        try:
            v = ev.evaluate(dv, vdd=1.8)
        except Exception as e:
            c[f"EXC:{type(e).__name__}"] += 1
            continue
        c["__valid__" if v.is_valid else v.check.value] += 1
    t = sum(c.values())
    frac = c["__valid__"] / t
    out[label] = {"valid": c["__valid__"], "n": t, "fraction": round(frac, 4),
                  "by_check": dict(c.most_common())}
    top = "  ".join(f"{n}x{k.split('_')[0]}" for k, n in c.most_common()[:3]
                    if k != "__valid__")
    print(f"{label:36s} {c['__valid__']:3d}/{t}  {frac*100:5.1f}%   {top}", flush=True)

Path("results").mkdir(exist_ok=True)
Path("results/what_binds.json").write_text(json.dumps(out, indent=2))
print("\nwrote results/what_binds.json")
