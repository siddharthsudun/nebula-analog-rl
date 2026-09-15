"""Full 32-spec fastest-mode rerun after the second-nearest-neighbor retry fix
(propose_seed_candidates / surrogate_stage1 retry_on_guard_invalid, bounded to
max_seed_evals=4), to get a real, not-inferred accuracy number for the mode-evidence
dashboard. Same spec_seed/N_SPECS as scratchpad/mode_bench5.json so this is directly
comparable to the pre-fix 24/31 baseline. pvt=False: this measures the nominal-stage
search itself, not the separate automatic PVT repair stage.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from eqrl import pipeline as pl
from eqrl.experiments.final_comparison import load_policy
from eqrl.experiments.target_audit import make_specs

SPEC_SEED, N_SPECS = 137, 32
OUT = Path("scratchpad/fastest_full32.json")

specs = make_specs(N_SPECS, SPEC_SEED)
load_policy(str(REPO / "results" / "seq_clean40k.zip"))

rows = []
t0 = time.perf_counter()
for i, (tgt, chan) in enumerate(specs):
    t = time.perf_counter()
    try:
        r = pl.design(target_boost_db=tgt, channel_loss_db=chan, spec_index=i,
                      mode="fastest", pvt=False)
    except Exception as exc:
        rows.append({"i": i, "target": tgt, "chan": chan, "error": "%s: %s" % (type(exc).__name__, exc)})
        OUT.write_text(json.dumps(rows, indent=1))
        print(f"{i:3d}  ERROR {exc}", flush=True)
        continue
    dt = time.perf_counter() - t
    v = r.get("verification") or {}
    row = {"i": i, "target": tgt, "chan": chan, "wall_s": round(dt, 2),
           "passed": v.get("passed"), "status": r.get("status"),
           "abs_err_db": v.get("abs_err_db"),
           "optimizer_evals": r["cost"].get("optimizer_evals"),
           "reason": r["provenance"].get("g32_reason")}
    rows.append(row)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"{i:3d} tgt={tgt:5.1f} chan={chan:4.1f}  passed={row['passed']}  "
          f"oe={row['optimizer_evals']}  wall={dt:.1f}s  reason={row['reason']!r}", flush=True)

solved = [r for r in rows if r.get("passed")]
print(f"\nTOTAL elapsed: {time.perf_counter()-t0:.1f}s")
print(f"SOLVED: {len(solved)}/{len(rows)} = {100*len(solved)/len(rows):.1f}%")
failing = [r["i"] for r in rows if not r.get("passed")]
print(f"FAILING: {failing}")
