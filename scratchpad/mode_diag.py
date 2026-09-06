"""WHY does the search stop where it stops? That single fact decides whether the four
requested mode targets are reachable on the current architecture or not.

g32_solve's stage B is a 1-D line search along plane["boost_axis"]. It can end four ways,
and they have completely different implications:
  reached_target                     -> accuracy is budget-bound  (more r fixes it)
  "converged onto the feasibility wall" -> the GUARD is the binding constraint (mode 4's premise)
  "converged inside the feasible set"   -> the 1-D line cannot represent the target at all
  "budget exhausted"                 -> budget-bound

Also times the real cost units so the <5s and ~60s targets can be checked against a floor
rather than a guess. Reads only; writes one scratchpad JSON.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src")); os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from eqrl import pipeline as pl

SPECS = [(9.0, 12.0, 0), (7.5, 10.0, 1), (10.5, 14.0, 2), (6.2, 13.5, 3), (11.0, 12.5, 4)]
OUT = Path(sys.argv[1])

def main() -> None:
    from eqrl.experiments.final_comparison import load_policy, plane_from_probe
    t = time.perf_counter(); load_policy(str(REPO / "results" / "seq_clean40k.zip"))
    warm = time.perf_counter() - t
    plane = plane_from_probe(pl.PEAK_PROBE)
    print(f"warm-up {warm:.1f}s   boost_axis={plane['boost_axis']}  "
          f"peak_axis={plane['peak_axis']}\n", flush=True)

    rows = []
    for tgt, chan, si in SPECS:
        for mode in ("default", "accurate"):
            t = time.perf_counter()
            r = pl.design(target_boost_db=tgt, channel_loss_db=chan,
                          spec_index=si, mode=mode)
            dt = time.perf_counter() - t
            p, c = r["provenance"], r["cost"]
            v = r.get("verification") or {}
            row = {"target": tgt, "chan": chan, "spec_index": si, "mode": mode,
                   "wall_s": round(dt, 1),
                   "abs_err_db": v.get("abs_err_db"),
                   "reached": p["g32_reached_target"], "reason": p["g32_reason"],
                   "blocked_by": p.get("g32_blocked_by"),
                   "stage2_evals": c["stage2_evals"],
                   "unspent": c["budget_evals_unspent"],
                   "measure_all_total": c["measure_all_total"],
                   "spice_total": c["spice_analyses_total"],
                   "steps": p["g32_steps"]}
            rows.append(row)
            print(f"{tgt:5.1f}/{chan:4.1f} {mode:9} {dt:5.1f}s  "
                  f"err={row['abs_err_db'] if row['abs_err_db'] is None else round(row['abs_err_db'],3):>6}  "
                  f"s2evals={row['stage2_evals']:>2} unspent={row['unspent']:>2}  "
                  f"reached={str(row['reached']):>5}  {row['reason']}", flush=True)
            OUT.write_text(json.dumps({"warmup_s": warm, "plane_boost_axis":
                                       plane["boost_axis"], "rows": rows}, indent=1))
    print(f"\nwrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
