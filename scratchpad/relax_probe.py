"""IS MODE 4 ACHIEVABLE? Relax only the DC-gain floor and see whether the precision search
reaches 0.05 dB, and what DC gain it spends to get there.

The wall probe showed T4.10_dc_gain_implausible walls 10 of 12 rejected precision steps.
Physically that is the CTLE trade itself: raising `rs` raises boost by degenerating the
pair, which costs DC gain. Two separate floors both sit at 0 dB and BOTH must move:
  guards.DC_GAIN_DB_MIN   -- the hard guard  (design is rejected outright)
  Spec.dc_gain_db_min     -- the 9th hard_pass check (design is "not feasible")

NOTHING IS EDITED. Both floors are rebound in THIS process only, so the repo, the frozen
constants and every recorded number are untouched. T1 (run integrity) and T2 (device
sanity, incl. saturation) are NOT relaxed and must never be -- those reject things that
are not circuits, and a number measured off one is meaningless.
"""
from __future__ import annotations
import dataclasses, json, os, sys
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src")); os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "silq-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from silq import guards, pipeline as pl
from silq.experiments import final_comparison as fc

SPECS = [(9.0, 12.0, 0), (11.0, 12.5, 4), (10.0, 11.5, 6), (10.5, 14.0, 2), (7.5, 10.0, 1)]
FLOORS = [0.0, -6.0]          # control, then relaxed

def run(tgt, chan, si, floor, plane, ladder, policy, env):
    guards.DC_GAIN_DB_MIN = floor
    fc.DEFAULT_SPEC = dataclasses.replace(fc.DEFAULT_SPEC, dc_gain_db_min=floor)
    ev = fc.Evaluation(); evaluate = ev.make_eval(chan)
    xs, s1, _t = fc.stage1_rollout(evaluate, policy, env, si, tgt, chan, fc.PREREG["k"])
    g2, info, x_f, rec_f, left = fc.g32_solve(evaluate, xs, s1, tgt, plane, ladder, 30,
                                              stop_abs_err_db=0.05)
    arm = fc.summarize([e for e in s1 if e] + g2, tgt, 1.5)
    best = None
    for r in ([e for e in s1 if e] + g2):
        if r["loose_pass"] and (best is None
                                or abs(r["boost_db"] - tgt) < abs(best["boost_db"] - tgt)):
            best = r
    return {"reached": bool(info["reached_target"]), "reason": info["reason"],
            "err": None if best is None else abs(best["boost_db"] - tgt),
            "dc_gain_db": None if best is None else best["dc_gain_db"],
            "evals": len(info["steps"]), "left": left}

def main() -> None:
    base_spec = fc.DEFAULT_SPEC
    plane = fc.plane_from_probe(pl.PEAK_PROBE)
    ladder = fc.rescue_order_from_probe(pl.RESCUE_PROBE)
    policy, env = fc.load_policy(pl.POLICY)
    out = []
    print(f"{'spec':>12} {'floor':>6} {'err':>7} {'dcgain':>8} {'evals':>6} {'reached':>8}  reason")
    for tgt, chan, si in SPECS:
        row = {"target": tgt, "chan": chan, "spec_index": si}
        for floor in FLOORS:
            fc.DEFAULT_SPEC = base_spec        # reset before each rebind
            r = run(tgt, chan, si, floor, plane, ladder, policy, env)
            row[str(floor)] = r
            print(f"{tgt:5.1f}/{chan:4.1f} {floor:>6.1f} "
                  f"{'--' if r['err'] is None else format(r['err'],'.3f'):>7} "
                  f"{'--' if r['dc_gain_db'] is None else format(r['dc_gain_db'],'.2f'):>8} "
                  f"{r['evals']:>6} {str(r['reached']):>8}  {r['reason']}", flush=True)
        out.append(row)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {sys.argv[1]}")

if __name__ == "__main__":
    main()
