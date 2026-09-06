"""WHICH constraint walls the precision search? Mode 4 proposes relaxing it, and the
answer decides whether that is a legitimate accuracy/robustness trade or a way to return
something that is not a circuit.

g32_solve records this in info["blocked_by"], which pipeline.design does not surface. So
call g32_solve directly, exactly as pipeline.design does, and read it.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src")); os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from eqrl import pipeline as pl
from eqrl.experiments import final_comparison as fc

SPECS = [(9.0, 12.0, 0), (11.0, 12.5, 4), (8.2, 11.0, 5), (10.0, 11.5, 6)]

def main() -> None:
    plane = fc.plane_from_probe(pl.PEAK_PROBE)
    ladder = fc.rescue_order_from_probe(pl.RESCUE_PROBE)
    policy, env = fc.load_policy(pl.POLICY)
    walls, out = Counter(), []
    for tgt, chan, si in SPECS:
        ev = fc.Evaluation(); evaluate = ev.make_eval(chan)
        xs, s1, _term = fc.stage1_rollout(evaluate, policy, env, si, tgt, chan,
                                          fc.PREREG["k"])
        g2, info, x_f, _rec, left = fc.g32_solve(
            evaluate, xs, s1, tgt, plane, ladder, 30, stop_abs_err_db=0.05)
        jb = fc.DIMS.index(plane["boost_axis"])
        # every step the line search took that was NOT accepted, and why
        bad = [s for s in info["steps"] if not s["feasible"]]
        why = Counter()
        for s in bad:
            why[s["guard_check"] or ("hard_pass:" + ",".join(s["failing"]))] += 1
        rs_at = None if x_f is None else round(float(x_f[jb]), 4)
        walls.update(why)
        print(f"{tgt:5.1f}/{chan:4.1f}  reached={str(info['reached_target']):>5} "
              f"left={left:>2}  rs_norm={rs_at}  reason={info['reason']}")
        print(f"           blocked_by={info['blocked_by']}   rejected steps: {dict(why)}")
        out.append({"target": tgt, "chan": chan, "reached": info["reached_target"],
                    "reason": info["reason"], "blocked_by": info["blocked_by"],
                    "rs_norm_final": rs_at, "budget_left": left,
                    "rejections": dict(why)})
    print("\nALL rejected steps pooled:", dict(walls))
    Path(sys.argv[1]).write_text(json.dumps({"rows": out, "pooled": dict(walls)}, indent=1))

if __name__ == "__main__":
    main()
