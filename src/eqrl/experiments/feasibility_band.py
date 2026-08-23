"""Is the specification satisfiable at all, now that a DC-gain floor exists?

Direct measurement of the delivered design (results/solved_design_pvt.json) returns
peak 1.345 GHz -- inside the 1.25-2.5 GHz band -- with dc_gain -7.35 dB. It reaches the
band by attenuating, which is the exploit the DC-gain floor was added to close, so the
design this repository presents as verified does not satisfy the corrected spec.

Meanwhile every valid design the trained policy produces has dc_gain between +1.3 and
+9.3 dB and peaks between 0.45 and 1.2 GHz -- below the band, all eleven of them, none
above. Two populations, each satisfying one constraint and missing the other, is the
signature of a trade-off rather than of a policy that has not learned.

That is a hypothesis, not a finding, and it has a decisive test: sample the action space
and ask whether ANY guard-valid point satisfies the DC-gain floor and the peak-frequency
band together. If some do, the specification is satisfiable and the gap is the policy's.
If none do across a wide sample, the task as specified cannot be solved in this topology
and no amount of training will close it -- which is a result about the problem, and has
to be reported rather than trained around.

Emits the full (dc_gain, peak_freq) scatter either way: the trade-off curve is the
explanation for whichever answer comes back.
"""
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
import json
import time
import numpy as np

from eqrl.circuits.ctle import decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC as S, hard_pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/feasibility_band.json")
    args = p.parse_args()

    ev = build_evaluator(S, corner="tt", fast=True)
    rng = np.random.default_rng(args.seed)

    lo, hi = S.peak_freq_lo_ghz, S.peak_freq_hi_ghz
    floor = S.dc_gain_db_min if S.dc_gain_db_min is not None else 0.0

    pts, n_valid, both, t0 = [], 0, [], time.time()
    for i in range(args.n):
        x = rng.random(6)
        dv = decode_action(x)
        try:
            v = ev.evaluate(dv, vdd=S.vdd_nominal)
        except Exception:
            continue
        if not v.is_valid:
            continue
        n_valid += 1
        m = v.unwrap()
        ok, checks = hard_pass(m, S)
        rec = {"dc_gain_db": round(m.dc_gain_db, 3),
               "peak_freq_ghz": round(m.peak_freq_ghz, 4),
               "boost_db": round(m.boost_db, 3),
               "in_band": bool(lo <= m.peak_freq_ghz <= hi),
               "dc_ok": bool(m.dc_gain_db >= floor),
               "all_pass": bool(ok),
               "failing": [k for k, ok_ in checks.items() if not ok_]}
        pts.append(rec)
        if rec["in_band"] and rec["dc_ok"]:
            rec["design"] = {k: float(val) for k, val in dv.__dict__.items()}
            both.append(rec)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{args.n}  valid {n_valid}  in-band&dc-ok {len(both)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    in_band = [r for r in pts if r["in_band"]]
    dc_ok = [r for r in pts if r["dc_ok"]]
    all_pass = [r for r in pts if r["all_pass"]]

    print(f"\nsampled {args.n}, guard-valid {n_valid}")
    print(f"  peak in {lo}-{hi} GHz          : {len(in_band)}")
    print(f"  dc_gain >= {floor} dB             : {len(dc_ok)}")
    print(f"  BOTH                          : {len(both)}")
    print(f"  all {len(hard_pass(None, S)[1]) if False else 9} checks pass          : {len(all_pass)}")
    if in_band:
        dcs = [r["dc_gain_db"] for r in in_band]
        print(f"\n  dc_gain among in-band designs : min {min(dcs):.2f} max {max(dcs):.2f} dB")
    if dc_ok:
        pk = [r["peak_freq_ghz"] for r in dc_ok]
        print(f"  peak among dc_gain>=0 designs : min {min(pk):.3f} max {max(pk):.3f} GHz")
    if not both:
        print("\nNo sampled point satisfies both. On this evidence the DC-gain floor and\n"
              "the peak-frequency band are in tension in this topology -- report before\n"
              "training further against it.")

    Path("results").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_sampled": args.n, "seed": args.seed, "guard_valid": n_valid,
         "band_ghz": [lo, hi], "dc_gain_floor_db": floor,
         "n_in_band": len(in_band), "n_dc_ok": len(dc_ok), "n_both": len(both),
         "n_all_pass": len(all_pass), "both": both[:50], "scatter": pts}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
