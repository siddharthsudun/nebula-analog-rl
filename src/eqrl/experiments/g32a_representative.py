"""G3.2a step 3: are uniform-random low-peak designs the same low-peak designs PPO hands off?

THE VALIDITY QUESTION g32a_select deliberately did not answer. The calibration set was drawn
uniformly, because the plane is a claim about circuit physics in a region and uniform
sampling covers that region without the policy's bias. But the controller only ever meets
low-peak designs that PPO produced. If PPO's low-peak handoffs sit in a corner of the
sub-band region that the uniform draw never visits, then a plane measured on the uniform set
is calibrated in the right FREQUENCY regime and still the wrong PLACE -- which is the same
mistake G3.2 made, one level down.

So this measures where each population actually sits, rather than assuming they coincide.

WHAT IT USES. PPO stage-1 traces on spec-seed 2, specs 8-23. Those sixteen specs are
untouched: the high-peak calibration used seed-2 specs 0-7, and the G3.2 test set is
seed-3 specs 8-17. Using the test set here would tell us about the test set.

WHAT IT REPORTS, and nothing more. Per axis, the interval the frozen calibration set spans,
and what fraction of PPO's low-peak handoffs land inside it; then the same jointly on the
two repair coordinates. There is no pass/fail rule here on purpose. The pre-declared rule
belongs to the probe (g32a_lowpeak_probe), and inventing a second gate after seeing this
would be exactly the kind of after-the-fact criterion this project keeps refusing to write.
Coverage is a caveat on how far the probe's answer reaches, and it is reported as one.

Changes no threshold, bound, reward, model, controller, or benchmark criterion. Read-only
with respect to every frozen artifact; it writes one new file of its own.
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
from eqrl.experiments.g32_peak_report import at_sweep_edge, plane_from_probe

DIMS = list(ACTION_SPACE.keys())

#: Same stage-1 regeneration as everywhere else in G3.x: k=5, deterministic, no reset.
K = 5
SPEC_SEED = 2          # calibration used 0-7 of this seed; the test set is seed 3
SPEC_LO, SPEC_HI = 8, 24


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--set", default="results/g32a_lowpeak_set.json")
    p.add_argument("--peak-probe", default="results/g32_peak_probe.json")
    p.add_argument("--out", default="results/g32a_representative.json")
    args = p.parse_args()

    sel = json.load(open(args.set))
    plane = plane_from_probe(args.peak_probe)
    lo_band = DEFAULT_SPEC.peak_freq_lo_ghz
    floor = sel["sweep_floor_ghz"]
    cal = np.array([d["x"] for d in sel["designs"]], dtype=np.float64)

    specs = make_specs(32, SPEC_SEED)[SPEC_LO:SPEC_HI]
    guards: dict[float, object] = {}
    n_sim = 0

    def guard_for(c):
        if c not in guards:
            guards[c] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                        channel_loss_db=c)
        return guards[c]

    def evaluate(x, target, channel):
        nonlocal n_sim
        n_sim += 1
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(decode_action(np.asarray(x)),
                                            vdd=spec.vdd_nominal)
        except Exception:
            return None
        if not v.is_valid:
            return None
        m = v.unwrap()
        ok, _ = hard_pass(m, spec)
        return {"boost_db": float(m.boost_db), "peak_freq_ghz": float(m.peak_freq_ghz),
                "loose_pass": bool(ok)}

    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    print("G3.2a REPRESENTATIVENESS | PPO stage-1 (k=%d) on spec-seed %d specs %d-%d"
          % (K, SPEC_SEED, SPEC_LO, SPEC_HI - 1), flush=True)
    print("  low-peak means guard-valid with %.4f < peak < %.2f GHz -- the same rule the "
          "calibration\n  set was selected by\n" % (floor, lo_band), flush=True)

    found: list[dict] = []
    n_valid = 0
    for off, (target, channel) in enumerate(specs):
        i = SPEC_LO + off
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        seen = []
        for step in range(K):
            if step:
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, _t, _u, _inf = env.step(a)
            x = np.array(env._x, dtype=np.float64)
            e = evaluate(x, target, channel)
            if e is None:
                continue
            n_valid += 1
            pk = e["peak_freq_ghz"]
            if at_sweep_edge(pk) or not (floor < pk < lo_band):
                continue
            rec = {"spec": i, "step": step, "x": [float(t) for t in x], **e}
            found.append(rec)
            seen.append(rec)
        if seen:
            print("  spec %2d  %d low-peak stage-1 design(s): %s"
                  % (i, len(seen), ", ".join("%.4f GHz" % r["peak_freq_ghz"] for r in seen)),
                  flush=True)

    print("\n  %d guard-valid stage-1 designs across %d specs; %d of them sit below the band"
          % (n_valid, len(specs), len(found)), flush=True)
    if not found:
        print("  PPO produced NO low-peak stage-1 design on this untouched slice. That is "
              "itself a\n  finding: it would mean the sub-band handoffs the G3.2 test set "
              "hit are rare, and it\n  leaves the coverage question unanswerable from this "
              "sample rather than answered.", flush=True)

    ppo = np.array([r["x"] for r in found], dtype=np.float64) if found else np.zeros((0, 6))
    cov: dict[str, dict] = {}
    print("\n  where each population sits, in unit design coordinates:")
    print("  %-8s %-22s %-22s %s" % ("axis", "calibration set (n=%d)" % len(cal),
                                     "PPO low-peak (n=%d)" % len(ppo), "inside"))
    for j, name in enumerate(DIMS):
        c_lo, c_hi = float(cal[:, j].min()), float(cal[:, j].max())
        if len(ppo):
            p_lo, p_hi = float(ppo[:, j].min()), float(ppo[:, j].max())
            n_in = int(((ppo[:, j] >= c_lo) & (ppo[:, j] <= c_hi)).sum())
            rng_s, in_s = "%.3f - %.3f" % (p_lo, p_hi), "%d/%d" % (n_in, len(ppo))
        else:
            p_lo = p_hi = None
            n_in = 0
            rng_s, in_s = "-", "-"
        cov[name] = {"cal_lo": c_lo, "cal_hi": c_hi, "ppo_lo": p_lo, "ppo_hi": p_hi,
                     "n_inside": n_in, "n_ppo": len(ppo)}
        mark = "  <-- repair coordinate" if name in (plane["boost_axis"],
                                                     plane["peak_axis"]) else ""
        print("  %-8s %-22s %-22s %s%s"
              % (name, "%.3f - %.3f" % (c_lo, c_hi), rng_s, in_s, mark))

    jb, jp = DIMS.index(plane["boost_axis"]), DIMS.index(plane["peak_axis"])
    joint = None
    if len(ppo):
        inside = np.ones(len(ppo), dtype=bool)
        for j in (jb, jp):
            inside &= (ppo[:, j] >= cal[:, j].min()) & (ppo[:, j] <= cal[:, j].max())
        joint = int(inside.sum())
        print("\n  jointly on the two repair coordinates (%s, %s): %d of %d PPO low-peak "
              "designs\n  fall inside the calibration set's box"
              % (plane["boost_axis"], plane["peak_axis"], joint, len(ppo)))
        print("  a design outside that box is not necessarily mis-served -- it means the "
              "probe did not\n  measure there, so the plane's behaviour at that point is an "
              "extrapolation the same way\n  the high-peak plane was below the band.")

    Path(args.out).write_text(json.dumps(
        {"spec_seed": SPEC_SEED, "specs": [SPEC_LO, SPEC_HI], "k": K,
         "model": args.model, "calibration_set": args.set,
         "band_lo_ghz": lo_band, "sweep_floor_ghz": floor,
         "n_measure_all": n_sim, "n_guard_valid": n_valid,
         "boost_axis": plane["boost_axis"], "peak_axis": plane["peak_axis"],
         "n_joint_inside": joint, "coverage": cov, "low_peak_designs": found}, indent=1))
    print("\n%d measure_all spent (validity check, not a controller arm)" % n_sim)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
