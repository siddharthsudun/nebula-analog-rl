"""Spec 11.0/12.5 fails with ZERO stage-2 evaluations and full budget unspent: `rs` is
already pinned at 1.0, so the 1-D line search has nowhere to go. No budget, tolerance or
restart setting can fix that -- only a different axis can. This measures whether one exists.
"""
from __future__ import annotations
import json, os, sys, numpy as np
from pathlib import Path
REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src")); os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"
from eqrl import pipeline as pl
from eqrl.experiments import final_comparison as fc

TGT, CHAN, SI = 11.0, 12.5, 4

def main() -> None:
    plane = fc.plane_from_probe(pl.PEAK_PROBE)
    ladder = fc.rescue_order_from_probe(pl.RESCUE_PROBE)
    policy, env = fc.load_policy(pl.POLICY)
    ev = fc.Evaluation(); evaluate = ev.make_eval(CHAN)
    xs, s1, _t = fc.stage1_rollout(evaluate, policy, env, SI, TGT, CHAN, fc.PREREG["k"])
    _g2, info, x_f, rec_f, _l = fc.g32_solve(evaluate, xs, s1, TGT, plane, ladder, 30,
                                             stop_abs_err_db=0.05)
    print("start: boost %.3f  err %.3f  x = %s" %
          (rec_f["boost_db"], abs(rec_f["boost_db"] - TGT),
           {d: round(float(v), 3) for d, v in zip(fc.DIMS, x_f)}))
    print("DIMS", fc.DIMS, " boost_axis", plane["boost_axis"], "\n")
    base = abs(rec_f["boost_db"] - TGT)
    out = []
    print(f"{'axis':>8} {'step':>6} {'boost':>7} {'err':>7} {'dcgain':>7} {'feasible':>9}")
    for j, name in enumerate(fc.DIMS):
        for step in (-0.20, -0.08, 0.08, 0.20):
            x = np.array(x_f, dtype=np.float64)
            nv = x[j] + step
            if not (0.0 <= nv <= 1.0):
                continue
            x[j] = nv
            rec, _s, g = evaluate(x, TGT)
            if rec is None:
                print(f"{name:>8} {step:>+6.2f} {'--':>7} {'--':>7} {'--':>7} "
                      f"{'GUARD':>9}  {g}")
                out.append({"axis": name, "step": step, "guard": g}); continue
            e = abs(rec["boost_db"] - TGT)
            print(f"{name:>8} {step:>+6.2f} {rec['boost_db']:>7.3f} {e:>7.3f} "
                  f"{rec['dc_gain_db']:>7.2f} {str(rec['loose_pass']):>9}"
                  f"{'   <-- BETTER' if e < base and rec['loose_pass'] else ''}")
            out.append({"axis": name, "step": step, "boost": rec["boost_db"], "err": e,
                        "dc_gain_db": rec["dc_gain_db"], "feasible": rec["loose_pass"]})
    Path(sys.argv[1]).write_text(json.dumps({"base_err": base, "probes": out}, indent=1))

if __name__ == "__main__":
    main()
