"""Where do the blocked handoffs sit relative to the band, and where was the plane fitted?

The G3.2 run repaired 0 of 4 Stage-A specs for a reason the step-sizing fix did not touch.
This regenerates stage 1 for the blocked specs and reports the handoff's peak, boost and
its position along the two repair coordinates, so the claim about WHY can be checked
against measurements rather than inferred from the step trace. Diagnostic only: it changes
nothing and produces no circuits.
"""
from __future__ import annotations

import dataclasses
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

import argparse
import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass
from eqrl.experiments.target_audit import make_specs
from eqrl.experiments.g32_peak_report import plane_from_probe

DIMS = list(ACTION_SPACE.keys())
K = 5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--run", default="results/g32_repair_smoke.json")
    p.add_argument("--peak-probe", default="results/g32_peak_probe.json")
    args = p.parse_args()

    plane = plane_from_probe(args.peak_probe)
    LO, HI = plane["band_ghz"]
    jb, jp = DIMS.index(plane["boost_axis"]), DIMS.index(plane["peak_axis"])
    run = json.load(open(args.run))
    blocked = [r for r in run["rows"] if not r["g32"]["strict_solved_at"]]

    guards: dict[float, object] = {}

    def guard_for(c):
        if c not in guards:
            guards[c] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                        channel_loss_db=c)
        return guards[c]

    def evaluate(x, target, channel):
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception:
            return None
        if not v.is_valid:
            return None
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        return {"boost_db": float(m.boost_db), "peak_freq_ghz": float(m.peak_freq_ghz),
                "loose_pass": bool(ok),
                "failing": [c for c, g in checks.items() if not g]}

    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        xs = [np.array(env._x, dtype=np.float64)]
        tr = [evaluate(env._x, target, channel)]
        while len(tr) < K:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, _t, _u, _ = env.step(a)
            xs.append(np.array(env._x, dtype=np.float64))
            tr.append(evaluate(env._x, target, channel))
        return xs, tr

    print("WHERE THE PLANE WAS FITTED (spec-seed 2 calibration handoffs):")
    cal = json.load(open(args.peak_probe))
    cal_pk = [r["base"]["peak_freq_ghz"] for r in cal["rows"] if r.get("dims")]
    for r in cal["rows"]:
        if r.get("dims"):
            pk = r["base"]["peak_freq_ghz"]
            print("   spec %2d  peak %6.3f GHz   %s" % (
                r["spec"], pk,
                "ABOVE band" if pk > HI else ("below band" if pk < LO else "in band")))
    print("   range %.3f to %.3f GHz\n" % (min(cal_pk), max(cal_pk)))

    print("WHERE IT WAS APPLIED (the blocked seed-3 handoffs):")
    lo_side = 0
    for r in blocked:
        i, tgt, ch = r["spec"], r["target_boost_db"], r["channel_loss_db"]
        xs, tr = stage1(i, tgt, ch)
        h, xh = tr[-1], xs[-1]
        cand = [(len(e["failing"]), j) for j, e in enumerate(tr) if e]
        best = tr[min(cand)[1]] if cand else None
        xbest = xs[min(cand)[1]] if cand else None
        if h is None and best is None:
            print("   spec %2d  every stage-1 design guard-invalid" % i)
            continue
        e, xe = (best, xbest) if best is not None else (h, xh)
        pk = e["peak_freq_ghz"]
        side = "ABOVE band" if pk > HI else ("BELOW band" if pk < LO else "in band")
        lo_side += pk < LO
        print("   spec %2d  peak %6.3f GHz  boost %5.2f  %-10s  %s=%.3f %s=%.3f  fails=%s"
              % (i, pk, e["boost_db"], side, plane["peak_axis"], xe[jp],
                 plane["boost_axis"], xe[jb], ",".join(e["failing"]) or "none"))
    print("\n   %d of %d blocked handoffs sit BELOW the band" % (lo_side, len(blocked)))
    print("   the calibration contains no design below the band, so the sign and gain the"
          "\n   controller uses there are extrapolations, not measurements")


if __name__ == "__main__":
    main()
