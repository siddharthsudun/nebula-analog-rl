"""accurate + thinking over the same 32 specs default already ran (spec-seed 137).

Default's numbers are NOT re-run: `scratchpad/retarget_ab32.json` already holds them for
exactly these specs, from the same code, and re-running them would cost half an hour to
reproduce numbers that are already on disk. They are joined in by spec index.

The question this answers is the one that decides whether `default` and `accurate` are
worth keeping: does `thinking` dominate them, or does it only dominate them on average
while losing on a class of specs? An average is not enough to delete a mode -- a mode
earns deletion only if nothing it does is unique.
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
_NG = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from eqrl import pipeline as pl  # noqa: E402

SPEC_SEED = 137
N_SPECS = 32
MODES = ("accurate", "thinking")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/mode_bench32.json")


def main() -> None:
    from eqrl.experiments.final_comparison import load_policy
    from eqrl.experiments.target_audit import make_specs

    specs = make_specs(N_SPECS, SPEC_SEED)
    load_policy(str(REPO / "results" / "seq_clean40k.zip"))
    rows = []
    for i, (tgt, chan) in enumerate(specs):
        for mode in MODES:
            t = time.perf_counter()
            try:
                r = pl.design(target_boost_db=tgt, channel_loss_db=chan, spec_index=i,
                              mode=mode)
            except Exception as exc:
                rows.append({"i": i, "target": tgt, "chan": chan, "mode": mode,
                             "error": "%s: %s" % (type(exc).__name__, exc)})
                OUT.write_text(json.dumps(rows, indent=1))
                print(f"{i:3d} {mode:9}  ERROR {exc}", flush=True)
                continue
            dt = time.perf_counter() - t
            v = r.get("verification") or {}
            p, c = r["provenance"], r["cost"]
            md = p.get("mode_detail") or {}
            rows.append({"i": i, "target": tgt, "chan": chan, "mode": mode,
                         "wall_s": round(dt, 1), "abs_err_db": v.get("abs_err_db"),
                         "status": r["status"], "passed": v.get("passed"),
                         "reason": p["g32_reason"], "measure_all": c["measure_all_total"],
                         "guidance": r["guidance"]["headline"],
                         "suggest": r["guidance"]["suggest_mode"],
                         "n_ppo": md.get("n_ppo_rollouts"),
                         "n_surrogate": md.get("n_surrogate_starts"),
                         "governor": md.get("governor_stopped")})
            OUT.write_text(json.dumps(rows, indent=1))
            e = v.get("abs_err_db")
            print(f"{i:3d} {tgt:5.1f}/{chan:4.1f} {mode:9} {dt:5.1f}s "
                  f"{'--' if e is None else format(e, '.4f'):>8} "
                  f"{c['measure_all_total']:>4}ma  {r['status']}", flush=True)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
