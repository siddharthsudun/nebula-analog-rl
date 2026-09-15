"""G3.2 primitive 2: can a guard-INVALID handoff be rescued by a single-axis move?

WHY THIS EXISTS. The peak probe answered the coordinate question and produced a second,
unwelcome fact. Every guard rejection in that run was a BIAS failure --
T4.10_dc_gain_implausible 14 times and T2.5_mosfet_not_in_saturation 11 -- and 4 of 8
seed-2 handoffs were rejected outright. The (rs, l_in) plane the peak probe identified is a
peak/boost plane. It has nothing to say about a transistor that is out of saturation, and
stepping in it to fix a bias problem would be a guess dressed as a mechanism.

So Stage A splits in two, and only one half is currently measured:

  Case 2  the handoff is guard-VALID and fails a hard check. Metrics exist, the violated
          check is known, and the peak probe gives measured slopes. Buildable now.
  Case 1  the handoff is guard-INVALID. There are no metrics at all -- no boost, no peak,
          nothing to difference or bracket. The only signal is the guard's own Check.

This file measures Case 1 rather than guessing at it: starting from a guard-invalid handoff,
step one axis at a time, both signs, at two magnitudes, and record which if any returns the
design to guard-valid, and what the guard says when it does not.

THE OUTCOME DECIDES THE ARCHITECTURE, and both answers are useful:

  * if some axis systematically rescues these points, Stage A gets a measured Case-1 branch;
  * if NO single-axis move rescues them, that is the finding, and it explains H1 directly.
    CMA-ES samples all six dimensions at once and recovered a strict solve on seed-3 spec 5
    from exactly such a point. A one-dimensional line search structurally cannot, and no
    amount of choosing the line better would change that. G3.2 would then be the right tool
    for Case 2 and the wrong tool for Case 1, which argues for keeping H1 rather than
    replacing it.

Magnitudes are larger than the peak probe's h=0.05 because a bias failure is not a local
perturbation -- the point is outside the valid set and has to get back in, not be nudged.

Calibration on spec-seed 2, disjoint from the seed-3 specs any controller is tested on.
Spends no per-spec controller budget. Changes no threshold, bound, reward, model, or
benchmark criterion.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "silq-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from silq.circuits.ctle import ACTION_SPACE, decode_action
from silq.evaluator import build_evaluator
from silq.specs import DEFAULT_SPEC, hard_pass
from silq.experiments.target_audit import make_specs

DIMS = list(ACTION_SPACE.keys())

#: Two magnitudes, not one. 0.10 asks whether the valid set is just next door; 0.25 asks
#: whether it is reachable along an axis at all. Both are pre-declared here rather than
#: widened later if the first returns nothing.
MAGS = (0.10, 0.25)
K = 5


def rescue_order_from_probe(path: str) -> list[dict]:
    """The ranked list of single-axis rescue attempts, taken from the committed artifact.

    Ranked by HOW MANY guard-invalid handoffs each (axis, sign) rescued, then by the
    smallest magnitude that did it, so the controller tries the move that worked most
    often and stays as local as it can. The probe stops at the first rescuing magnitude
    per axis and sign, so a (axis, sign) appearing at both 0.10 and 0.25 means 0.10
    sufficed on one handoff and not on another -- both are kept, smaller first, and the
    larger becomes a later attempt rather than a replacement.

    This order is fitted on the guard-invalid handoffs of ONE spec seed and is applied to
    a different one. With only a handful of such handoffs to fit on, it is a ranked
    hypothesis, not an established ordering, and the controller records which entry
    actually fired so the test can say whether the ranking held.
    """
    a = json.load(open(path))
    hits: dict[tuple[str, float, float], set[int]] = {}
    for r in a["rows"]:
        if not r.get("case1"):
            continue
        for t in r["tries"] or []:
            if t["valid"]:
                hits.setdefault((t["axis"], t["sign"], t["mag"]), set()).add(r["spec"])
    # How many distinct handoffs each (axis, sign) rescued, pooled over magnitudes: an
    # axis that works at 0.25 where 0.10 was not enough is still that axis working.
    by_axis_sign: dict[tuple[str, float], set[int]] = {}
    for (axis, sgn, _mag), specs in hits.items():
        by_axis_sign.setdefault((axis, sgn), set()).update(specs)
    order = sorted(hits.items(),
                   key=lambda kv: (-len(by_axis_sign[(kv[0][0], kv[0][1])]), kv[0][2],
                                   kv[0][0], -kv[0][1]))
    return [{"axis": axis, "sign": sgn, "mag": mag, "rescued_specs": sorted(specs)}
            for (axis, sgn, mag), specs in order]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--specs", type=int, default=8)
    p.add_argument("--spec-seed", type=int, default=2)
    p.add_argument("--out", default="results/g32_rescue_probe.json")
    args = p.parse_args()

    specs = make_specs(32, args.spec_seed)[:args.specs]
    guards: dict[float, object] = {}
    n_sim = 0

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        nonlocal n_sim
        n_sim += 1
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception as e:
            return {"valid": False, "guard_check": "exception", "reason": repr(e)[:200]}
        if not v.is_valid:
            return {"valid": False, "guard_check": str(getattr(v.check, "value", v.check)),
                    "reason": str(v.reason)[:200]}
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        return {"valid": True, "boost_db": float(m.boost_db),
                "peak_freq_ghz": float(m.peak_freq_ghz),
                "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                "failing": [c for c, good in checks.items() if not good]}

    from stable_baselines3 import PPO
    from silq.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        """Byte-identical to hybrid_audit.stage1: k evals, NO reset on terminated."""
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        recs = [evaluate(env._x, target, channel)]
        while len(recs) < K:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, _term, _trunc, _ = env.step(a)
            recs.append(evaluate(env._x, target, channel))
        return np.array(env._x, dtype=np.float64), recs

    print("G3.2 RESCUE PROBE | %d specs, spec-seed %d | magnitudes %s"
          % (len(specs), args.spec_seed, list(MAGS)), flush=True)
    print("  asks: from a guard-INVALID handoff, does any single-axis step restore "
          "validity?\n", flush=True)

    rows: list[dict] = []

    def write(complete: bool) -> None:
        Path(args.out).write_text(json.dumps(
            {"mags": list(MAGS), "k": K, "spec_seed": args.spec_seed, "model": args.model,
             "n_measure_all": n_sim, "complete": complete, "rows": rows}, indent=1))

    for i, (target, channel) in enumerate(specs):
        x0, s1 = stage1(i, target, channel)
        base = s1[-1]
        if base["valid"]:
            print("  spec %2d  handoff already guard-valid -- not a Case 1, skipped" % i,
                  flush=True)
            rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                         "base": base, "case1": False, "tries": None})
            write(False)
            continue

        print("  spec %2d  handoff INVALID: %s" % (i, base["guard_check"]), flush=True)
        tries, rescued = [], []
        for j, name in enumerate(DIMS):
            e = np.zeros(len(DIMS)); e[j] = 1.0
            for sgn in (+1.0, -1.0):
                for mag in MAGS:
                    xt = np.clip(x0 + sgn * mag * e, 0.0, 1.0)
                    moved = float(abs(xt[j] - x0[j]))
                    if moved < 1e-9:
                        continue          # clipped at a bound: not a move, not a datum
                    rec = evaluate(xt, target, channel)
                    tries.append({"axis": name, "sign": sgn, "mag": mag, "moved": moved,
                                  **rec})
                    if rec["valid"]:
                        rescued.append((name, sgn, mag, rec))
                        break             # smallest rescuing magnitude on this axis/sign
        rows.append({"spec": i, "target_boost_db": target, "channel_loss_db": channel,
                     "base": base, "case1": True, "tries": tries})
        if rescued:
            for name, sgn, mag, rec in rescued:
                print("      RESCUED by %s %s%.2f -> boost %5.2f peak %.3f  %s"
                      % (name, "+" if sgn > 0 else "-", mag, rec["boost_db"],
                         rec["peak_freq_ghz"],
                         "hard_pass" if rec["loose_pass"] else
                         "fails " + ",".join(rec["failing"])), flush=True)
        else:
            seen: dict[str, int] = {}
            for t in tries:
                seen[t["guard_check"]] = seen.get(t["guard_check"], 0) + 1
            print("      NO single-axis rescue in %d tries; guard said %s"
                  % (len(tries), seen), flush=True)
        write(False)

    write(True)
    print("\n%d measure_all spent (calibration, not a controller arm)" % n_sim, flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
