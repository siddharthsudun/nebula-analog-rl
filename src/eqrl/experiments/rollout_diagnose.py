"""Which check actually blocks the policy? The rollout says 'not solved' and stops there.

policy_rollout answers whether a spec was solved. When the answer is no it records
nothing about WHY, so 30 unsolved specs carry no information beyond their own count --
and 'the policy is not good enough' is not an actionable finding.

This replays the same deterministic rollouts and, for the best design each trajectory
reaches, reports every failing hard_pass check and any guard rejection. Aggregated over
the held-out set it says whether the policy misses on one metric or on many, whether it
is near the boundary or nowhere near it, and whether the blocker is spec compliance or
physical validity -- three different problems with three different fixes.

'Best design' is the step with the fewest failing checks, not the last step: a trajectory
that reaches a near-miss and then wanders away is a different failure from one that never
approaches at all, and scoring only the final step would merge them.
"""
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
import collections
import numpy as np

from eqrl.circuits.ctle import decode_action
from eqrl.evaluator import build_evaluator
from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.specs import DEFAULT_SPEC, hard_pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--out", default="results/rollout_diagnose.json")
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)

    rng = np.random.default_rng(0)          # same seed as policy_rollout: same held-out set
    specs = [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16)))
             for _ in range(args.specs)]
    env = SequentialEqualizerEnv(fast=False, seed=123)
    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def assess(x, target, channel) -> dict:
        """Failing spec checks plus the guard verdict for one design."""
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception as e:
            return {"guard": f"__error__:{type(e).__name__}", "failing": ["__nosim__"],
                    "n_fail": 99}
        if not v.is_valid:
            # An invalid design has no trustworthy measurement, so its spec checks are
            # not reported -- counting them would treat unmeasured as failed.
            # str(): Invalid.check is a Check enum, which neither concatenates into the
            # progress line nor survives json.dumps.
            return {"guard": str(v.check), "failing": [], "n_fail": 98}
        ok, checks = hard_pass(v.unwrap(), spec)
        failing = [k for k, passed in checks.items() if not passed]
        m = v.unwrap()
        return {"guard": None, "failing": failing, "n_fail": 0 if ok else len(failing),
                "boost_db": round(m.boost_db, 2), "dc_gain_db": round(m.dc_gain_db, 2),
                "peak_ghz": round(m.peak_freq_ghz, 2), "eye_v_mv": round(m.eye_v_mv, 1),
                "eye_h_ui": round(m.eye_h_ui, 3)}

    rows, fail_hist, guard_hist = [], collections.Counter(), collections.Counter()
    for i, (target, channel) in enumerate(specs):
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        best = assess(env._x, target, channel)
        for _ in range(env.horizon - 1):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            cur = assess(env._x, target, channel)
            if cur["n_fail"] < best["n_fail"]:
                best = cur
            if term or trunc:
                break
        solved = best["n_fail"] == 0
        rows.append({"spec": i, "target_boost_db": round(target, 2),
                     "channel_loss_db": round(channel, 2), "solved": solved,
                     "best": best})
        if not solved:
            if best["guard"]:
                guard_hist[best["guard"]] += 1
            for k in best["failing"]:
                fail_hist[k] += 1
        print(f"  spec {i:2d}: {'SOLVED' if solved else 'blocked by ' + (best['guard'] or ','.join(best['failing']) or '?')}",
              flush=True)

    n_solved = sum(r["solved"] for r in rows)
    near = [r for r in rows if not r["solved"] and r["best"]["n_fail"] == 1]
    print(f"\nsolved {n_solved}/{len(rows)}")
    print(f"one check away: {len(near)}")
    print("\nfailing spec checks, best design per unsolved spec:")
    for k, n in fail_hist.most_common():
        print(f"  {k:<14} {n}")
    if guard_hist:
        print("\nunsolved specs whose best design never became valid:")
        for k, n in guard_hist.most_common():
            print(f"  {k:<40} {n}")

    Path("results").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "specs": len(rows), "solved": n_solved,
         "one_check_away": len(near),
         "failing_checks": dict(fail_hist.most_common()),
         "never_valid": dict(guard_hist.most_common()), "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
