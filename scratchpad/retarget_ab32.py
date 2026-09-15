"""A/B: `retarget` vs `default` at n=32 on a FRESH spec seed, so the improvement can be
quoted as a rate instead of as one demonstration.

Why a fresh seed: the n=8 run that first showed the fix working used hand-picked specs,
one of which (11.0/12.5) was chosen BECAUSE it was the known pinned case. That is a
demonstration, not a measurement -- the fire rate it implies is meaningless. Seed 137 has
never been drawn: seeds 0, 2, 3, 23, 99 and 20260915 all appear in the repo's docs, and
137 does not. Specs come from `make_specs`, the same generator the frozen comparison uses,
so the distribution is unchanged and only the draw is new.

The two modes still differ in exactly ONE thing (whether a pinned boost axis may be
swapped), because `retarget` deliberately inherits Default's budget and stop tolerance.
So three numbers matter and the last two matter most:
  fire rate  -- how often a pinned axis actually occurs in the wild
  better/worse among FIRED specs -- does the swap help when it triggers
  any difference at all among NOT-FIRED specs -- must be exactly zero, or the arm is
    not inert and the whole freeze-safety argument is wrong

Reads only. Writes one scratchpad JSON, incrementally, so a kill mid-run keeps its rows.
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

SPEC_SEED = 137
N_SPECS = 32
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/retarget_ab32.json")


def main() -> None:
    from silq.experiments.final_comparison import load_policy
    from silq.experiments.target_audit import make_specs

    specs = make_specs(N_SPECS, SPEC_SEED)
    t = time.perf_counter()
    load_policy(str(REPO / "results" / "seq_clean40k.zip"))
    print("warm-up %.1fs | spec-seed %d, %d specs\n"
          % (time.perf_counter() - t, SPEC_SEED, len(specs)), flush=True)

    print(f"{'i':>3} {'spec':>12} {'mode':9} {'wall':>6} {'err':>7} {'evals':>6} "
          f"{'fired':>6}  reason", flush=True)
    rows = []
    for i, (tgt, chan) in enumerate(specs):
        for mode in ("default", "retarget"):
            t = time.perf_counter()
            try:
                # pvt=False: default's rows here are joined into mode_bench32.py against
                # accurate/thinking, and this A/B's own wall-time column is read as
                # search cost, not as search + an unrelated ~80-150s PVT repair stage.
                r = pl.design(target_boost_db=tgt, channel_loss_db=chan, spec_index=i,
                              mode=mode, pvt=False)
            except Exception as exc:                      # keep the run going; record it
                rows.append({"i": i, "target": tgt, "chan": chan, "mode": mode,
                             "error": "%s: %s" % (type(exc).__name__, exc)})
                OUT.write_text(json.dumps(rows, indent=1))
                print(f"{i:3d} {tgt:5.1f}/{chan:4.1f} {mode:9}  ERROR {exc}", flush=True)
                continue
            dt = time.perf_counter() - t
            v = r.get("verification") or {}
            p, c = r["provenance"], r["cost"]
            rt = (p.get("mode_detail") or {}).get("retarget") or {}
            row = {"i": i, "target": tgt, "chan": chan, "mode": mode,
                   "wall_s": round(dt, 1), "abs_err_db": v.get("abs_err_db"),
                   "status": r["status"], "passed": v.get("passed"),
                   "reached": p["g32_reached_target"], "reason": p["g32_reason"],
                   "stage2_evals": c["stage2_evals"], "measure_all": c["measure_all_total"],
                   "fired": rt.get("fired"), "accepted_axis": rt.get("accepted_axis"),
                   "why_not": rt.get("why_not"), "retarget": rt}
            rows.append(row)
            OUT.write_text(json.dumps(rows, indent=1))
            e = row["abs_err_db"]
            print(f"{i:3d} {tgt:5.1f}/{chan:4.1f} {mode:9} {dt:5.1f}s "
                  f"{'--' if e is None else format(e, '.3f'):>7} "
                  f"{c['stage2_evals']:>6} {str(rt.get('fired')):>6}  {row['reason']}",
                  flush=True)

    summarize(rows)
    print(f"\nwrote {OUT}", flush=True)


def summarize(rows: list[dict]) -> None:
    INF = float("inf")
    by: dict[int, dict[str, dict]] = {}
    for r in rows:
        if "error" not in r:
            by.setdefault(r["i"], {})[r["mode"]] = r

    print("\n--- per-spec ---", flush=True)
    fired = better = worse = same = 0
    not_fired_diff = []
    for i, m in sorted(by.items()):
        d, g = m.get("default"), m.get("retarget")
        if not d or not g:
            continue
        de = d["abs_err_db"] if d["abs_err_db"] is not None else INF
        ge = g["abs_err_db"] if g["abs_err_db"] is not None else INF
        if ge < de - 0.001:
            tag = "BETTER"; better += 1
        elif ge > de + 0.001:
            tag = "WORSE"; worse += 1
        else:
            tag = "same"; same += 1
        if g["fired"]:
            fired += 1
        elif tag != "same":
            not_fired_diff.append(i)          # this would be a correctness bug, not noise
        print(f"  {i:3d} {d['target']:5.1f}/{d['chan']:4.1f}  {de:8.3f} -> {ge:8.3f}  "
              f"{tag:6}  fired={str(g['fired']):5} axis={g['accepted_axis']} "
              f"(+{g['measure_all'] - d['measure_all']} measure_all)"
              f"  [{'' if g['fired'] else g['why_not'] or ''}]", flush=True)

    n = len(by)
    print(f"\n--- verdict over n={n} on spec-seed {SPEC_SEED} ---", flush=True)
    print(f"  retarget fired on {fired}/{n} specs", flush=True)
    print(f"  better {better}   worse {worse}   unchanged {same}", flush=True)
    if not_fired_diff:
        print(f"  *** NOT INERT: specs {not_fired_diff} differ although the retarget "
              f"never fired. The arm is changing Default's own path -- investigate "
              f"before quoting anything else here. ***", flush=True)
    else:
        print("  inert on every non-fired spec (this is the freeze-safety check)",
              flush=True)


if __name__ == "__main__":
    main()
