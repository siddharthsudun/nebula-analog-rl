"""A/B: `retarget` against `default`, same budget, same stop tolerance, same specs.

The two modes differ in exactly one thing -- whether a pinned boost axis is allowed to be
swapped for another one -- so any difference here is attributable to that and nothing else.

Two questions, and the second matters more than the first:
  1. Does it fix the pinned case (spec 11.0/12.5, 1.936 dB for zero evaluations)?
  2. Does it REGRESS anything that already worked? The retarget must be inert on every
     spec whose axis is not pinned, and `retarget.fired == False` is the check for that.

Reads only. Writes one scratchpad JSON.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "silq-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from silq import pipeline as pl  # noqa: E402

SPECS = [(11.0, 12.5, 4), (9.0, 12.0, 0), (7.5, 10.0, 1), (10.5, 14.0, 2),
         (6.2, 13.5, 3), (10.0, 11.5, 6), (8.2, 11.0, 5), (9.5, 13.0, 7)]
OUT = Path(sys.argv[1])


def main() -> None:
    from silq.experiments.final_comparison import load_policy
    t = time.perf_counter()
    load_policy(str(REPO / "results" / "seq_clean40k.zip"))
    print("warm-up %.1fs\n" % (time.perf_counter() - t), flush=True)

    print(f"{'spec':>12} {'mode':9} {'wall':>6} {'err':>7} {'evals':>6} {'fired':>6}  reason",
          flush=True)
    rows = []
    for tgt, chan, si in SPECS:
        for mode in ("default", "retarget"):
            t = time.perf_counter()
            # pvt=False: default/retarget must be isolated to exactly one difference
            # (the pinned boost axis). PVT repair is a shared post-verification stage
            # neither mode controls, so leaving it on would add ~80-150s of unrelated
            # noise to the wall-time column this A/B depends on.
            r = pl.design(target_boost_db=tgt, channel_loss_db=chan, spec_index=si,
                          mode=mode, pvt=False)
            dt = time.perf_counter() - t
            v = r.get("verification") or {}
            p, c = r["provenance"], r["cost"]
            rt = (p.get("mode_detail") or {}).get("retarget") or {}
            row = {"target": tgt, "chan": chan, "spec_index": si, "mode": mode,
                   "wall_s": round(dt, 1), "abs_err_db": v.get("abs_err_db"),
                   "status": r["status"], "passed": v.get("passed"),
                   "reached": p["g32_reached_target"], "reason": p["g32_reason"],
                   "stage2_evals": c["stage2_evals"], "measure_all": c["measure_all_total"],
                   "fired": rt.get("fired"), "accepted_axis": rt.get("accepted_axis"),
                   "why_not": rt.get("why_not"), "retarget": rt}
            rows.append(row)
            e = row["abs_err_db"]
            print(f"{tgt:5.1f}/{chan:4.1f} {mode:9} {dt:5.1f}s "
                  f"{'--' if e is None else format(e, '.3f'):>7} "
                  f"{c['stage2_evals']:>6} {str(rt.get('fired')):>6}  {row['reason']}",
                  flush=True)
            OUT.write_text(json.dumps(rows, indent=1))

    print("\n--- verdict ---", flush=True)
    by = {}
    for r in rows:
        by.setdefault((r["target"], r["chan"]), {})[r["mode"]] = r
    better = worse = same = 0
    for spec, m in sorted(by.items()):
        d, g = m.get("default"), m.get("retarget")
        if not d or not g:
            continue
        de = d["abs_err_db"] if d["abs_err_db"] is not None else float("inf")
        ge = g["abs_err_db"] if g["abs_err_db"] is not None else float("inf")
        tag = "same"
        if ge < de - 0.001:
            tag, better = "BETTER", better + 1
        elif ge > de + 0.001:
            tag, worse = "WORSE", worse + 1
        else:
            same += 1
        print(f"  {spec[0]:5.1f}/{spec[1]:4.1f}  default {de:7.3f} -> retarget {ge:7.3f}  "
              f"{tag:6}  fired={g['fired']}  axis={g['accepted_axis']}"
              f"  (+{g['measure_all'] - d['measure_all']} measure_all)", flush=True)
    print(f"\n  better {better}   worse {worse}   unchanged {same}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
