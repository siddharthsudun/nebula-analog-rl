"""Does G3.2 success separate by DC-gain headroom at the PPO handoff?

WHY THIS NEEDS SIMULATION AT ALL. The intent was to answer this from committed artifacts.
It cannot be: results/g32_repair_smoke.json records boost, peak, feasibility and the guard
check at every step, but dc_gain_db appears nowhere -- not in the handoff record, not in any
of the 41 solver steps. So stage 1 is regenerated here deterministically (PPO k=5, seed
1000+i, no reset), exactly as g32_why_blocked.py already does, and the DC gain is read off
the same designs the frozen run used. The G3.2 outcomes themselves are READ from the frozen
artifact and are not recomputed.

The handoff is not guessed at: the controller's recorded handoff boost is matched against
the regenerated stage-1 designs, and a mismatch is reported rather than papered over.

HEADROOM IS NOT A FITTED QUANTITY. guards.DC_GAIN_DB_MIN is 0.0 dB, so
    headroom = dc_gain_db(handoff) - DC_GAIN_DB_MIN = dc_gain_db(handoff)
which makes this the guard's own bound rather than anything chosen here.

WHAT THIS CAN AND CANNOT ESTABLISH -- stated before the numbers, because the temptation
runs the other way. n = 10 specs, of which 4 failed strict. A threshold search over 10
points with 4 in one class will find a clean split whether or not one exists; that is
degrees of freedom, not evidence. So this file reports the two groups' headroom values and
the overlap between them, and it deliberately DOES NOT fit, report, or suggest a threshold.
It is an explanation of when the constrained solver runs out of room, offered for the
writeup. It is not a router gate, and the seed-3 slice -- looked at repeatedly through
G3.2's development -- could not calibrate one honestly even if a threshold were wanted.

Reads the frozen run; writes one new artifact of its own. Changes no controller, bound,
criterion or model.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from eqrl.circuits.ctle import decode_action
from eqrl.evaluator import build_evaluator
from eqrl.guards import DC_GAIN_DB_MIN
from eqrl.specs import DEFAULT_SPEC, hard_pass

K = 5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--run", default="results/g32_repair_smoke.json")
    p.add_argument("--out", default="results/g32_headroom.json")
    args = p.parse_args()

    run = json.load(open(args.run))
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
            return {"valid": False, "guard_check": str(getattr(v.check, "value", v.check))}
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        return {"valid": True, "boost_db": float(m.boost_db),
                "peak_freq_ghz": float(m.peak_freq_ghz),
                "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                "failing": [c for c, g in checks.items() if not g]}

    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    print("G3.2 DC-GAIN HEADROOM AT THE HANDOFF | %d specs from %s | guard floor %.1f dB"
          % (len(run["rows"]), args.run, DC_GAIN_DB_MIN), flush=True)
    print("  stage 1 regenerated deterministically; dc_gain was never recorded in the run",
          flush=True)
    print("  NO threshold is fitted or reported -- see this file's docstring\n", flush=True)

    rows = []
    for r in run["rows"]:
        i, tgt, ch = r["spec"], r["target_boost_db"], r["channel_loss_db"]
        g = r["g32"]
        env.reset(seed=1000 + i)
        env._target, env._channel = tgt, ch
        obs = env._obs(env._measure(env._x))
        trace = [evaluate(env._x, tgt, ch)]
        while len(trace) < K:
            a, _ = model.predict(obs, deterministic=True)
            obs, _rw, _t, _u, _inf = env.step(a)
            trace.append(evaluate(env._x, tgt, ch))

        # Match the controller's recorded handoff rather than assuming which design it was.
        # A guard-INVALID handoff records boost_db as None: the guard refused the design, so
        # there is no measured DC gain to read and no headroom to speak of. That is its own
        # outcome, not a missing value to be filled in.
        hb = g["handoff"]["boost_db"]
        if hb is None:
            match, handoff = [], None
        else:
            match = [e for e in trace if e and e["valid"] and abs(e["boost_db"] - hb) < 1e-6]
            handoff = match[0] if match else None
        valid = [e for e in trace if e and e["valid"]]
        rec = {
            "spec": i, "target_boost_db": tgt, "channel_loss_db": ch,
            "handoff_matched": bool(match),
            "handoff_guard_invalid": hb is None,
            "handoff_boost_db_recorded": hb,
            "handoff_guard_valid": g["handoff"]["guard_valid"],
            "handoff_dc_gain_db": handoff["dc_gain_db"] if handoff else None,
            "handoff_peak_freq_ghz": handoff["peak_freq_ghz"] if handoff else None,
            "headroom_db": (handoff["dc_gain_db"] - DC_GAIN_DB_MIN) if handoff else None,
            "min_dc_over_valid_stage1": min((e["dc_gain_db"] for e in valid), default=None),
            "max_dc_over_valid_stage1": max((e["dc_gain_db"] for e in valid), default=None),
            "n_stage1_guard_valid": len(valid),
            "strict_solved": g["strict_solved_at"] is not None,
            "loose_solved": g["loose_solved_at"] is not None,
            "best_abs_err": g["best_abs_err"],
            "case": g["solver"]["case"], "wall_hit": g["solver"]["wall_hit"],
            "blocked_by": g["solver"]["blocked_by"],
            "reason": g["solver"]["reason"],
        }
        rows.append(rec)
        print("  spec %2d  headroom %s dB  peak %s GHz  %-11s  %-22s %s"
              % (i,
                 "%6.2f" % rec["headroom_db"] if rec["headroom_db"] is not None else "  n/a",
                 "%6.3f" % rec["handoff_peak_freq_ghz"] if handoff else "  n/a",
                 "STRICT" if rec["strict_solved"] else "not strict",
                 rec["case"][:22],
                 ("blocked: " + rec["blocked_by"]) if rec["blocked_by"] else ""), flush=True)

    Path(args.out).write_text(json.dumps(
        {"run": args.run, "model": args.model, "k": K,
         "dc_gain_db_min": DC_GAIN_DB_MIN, "spec_seed": run["spec_seed"],
         "n_measure_all": n_sim, "rows": rows}, indent=1))

    inval = [r for r in rows if r["handoff_guard_invalid"]]
    if inval:
        print("\n  %d handoff(s) were guard-INVALID, so they have no measured DC gain and no"
              "\n  headroom by definition: spec %s. Excluded from the comparison below rather"
              "\n  than scored as zero -- a guard rejection is a different failure from a "
              "tight budget."
              % (len(inval), ", ".join(str(r["spec"]) for r in inval)))
    unmatched = [r for r in rows
                 if not r["handoff_matched"] and not r["handoff_guard_invalid"]]
    if unmatched:
        print("\n  WARNING: %d handoff(s) could not be matched to a regenerated stage-1 "
              "design;\n  those specs are reported as n/a rather than guessed at: %s"
              % (len(unmatched), ", ".join(str(r["spec"]) for r in unmatched)))

    have = [r for r in rows if r["headroom_db"] is not None]
    ok = sorted(r["headroom_db"] for r in have if r["strict_solved"])
    no = sorted(r["headroom_db"] for r in have if not r["strict_solved"])
    print("\n  headroom at handoff, dB above the guard floor:")
    print("    strict-solved (n=%d):     %s" % (len(ok), ", ".join("%.2f" % v for v in ok)))
    print("    not strict-solved (n=%d): %s" % (len(no), ", ".join("%.2f" % v for v in no)))
    if ok and no:
        # Interval intersection, not "min(ok) <= max(no)". The one-sided form assumes the
        # solved group sits ABOVE the unsolved one; if the separation runs the other way it
        # reports overlap where there is none, which is how it read on the first run.
        overlap = max(min(ok), min(no)) <= min(max(ok), max(no))
        print("\n    ranges %.2f-%.2f vs %.2f-%.2f -- the two groups %s"
              % (min(ok), max(ok), min(no), max(no),
                 "OVERLAP" if overlap else "do not overlap"))
        if not overlap:
            print("    DIRECTION: the %s group has the HIGHER headroom."
                  % ("strict-solved" if min(ok) > max(no) else "NOT strict-solved"))
        print("    n = %d. A split found on this many points, on the slice G3.2 was "
              "developed\n    against, is a hypothesis and not a decision rule. No "
              "threshold is offered." % len(have))

    # Peak position is the other candidate explanation and has to be shown beside headroom,
    # or a separation caused by one would be credited to the other.
    print("\n  the competing explanation, shown so it cannot be quietly ignored --")
    print("    handoff peak below the 1.25 GHz band edge:")
    for tag, sub in (("strict-solved", [r for r in have if r["strict_solved"]]),
                     ("not strict-solved", [r for r in have if not r["strict_solved"]])):
        below = [r for r in sub if r["handoff_peak_freq_ghz"] < 1.25]
        print("      %-18s %d of %d" % (tag, len(below), len(sub)))
    print("\n%d measure_all spent (analysis, not a controller arm)" % n_sim)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
