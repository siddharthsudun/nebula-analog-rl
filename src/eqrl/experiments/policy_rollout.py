"""Can the trained policy actually close a design? The minimum proof the loop works.

Everything else in this repository measures the machinery: that the simulator is fast, that
the guard layer catches bad circuits, that the search space is characterised. None of it
answers the question the problem statement actually asks, which is whether a reinforcement-
learning policy can take a specification it has never seen and produce a sized circuit that
meets it.

This runs that test and nothing else. For each held-out (target boost, channel loss) pair:
roll the policy out deterministically, and after every step check whether the current design
passes all eight hard specs AND survives the guard layer -- the same success test as
honest_benchmark --require-valid, so the number is comparable to the search baselines.

Reported: simulations to first success per spec, and the solve rate. A policy that solves
nothing is a policy that has not learned, whatever its training curve looks like.

Comparison point already measured, same success test, same target: the parameter sweep the
problem statement names as the baseline needed 2,394 evaluations to its first success
(results/sweep_baseline.json).
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
import numpy as np

from eqrl.circuits.ctle import decode_action
from eqrl.evaluator import build_evaluator
from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.specs import DEFAULT_SPEC, hard_pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--specs", type=int, default=8)
    p.add_argument("--out", default="results/policy_rollout.json")
    #: Makes success require hitting the spec's target boost within this tolerance, not
    #: merely landing anywhere in the 3-12 dB range. Off by default so the number stays
    #: comparable to every earlier rollout; on, it measures RETARGETING, which is the
    #: claim the submission actually makes.
    p.add_argument("--boost-tol", type=float, default=None,
                   help="dB tolerance on the target boost; success must also hit the "
                        "requested target (default: off)")
    p.add_argument("--anchor-baseline", action="store_true",
                   help="start each rollout at the verified robust baseline")
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)

    #: Held-out targets. Drawn from the same distribution the env randomises over during
    #: training, but this exact set was never trained on -- the point is retargeting, not
    #: memorisation.
    rng = np.random.default_rng(0)
    specs = [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16)))
             for _ in range(args.specs)]

    anchor = None
    if args.anchor_baseline:
        from eqrl.baselines.robust import robust_design
        anchor = robust_design()
    env = SequentialEqualizerEnv(fast=False, seed=123, anchor_design=anchor,
                                 boost_tol=args.boost_tol)

    #: One guard per spec, built at THAT spec's channel loss. A single guard built once
    #: measures the eye at whatever channel it was constructed with, so every verdict
    #: describes a link the caller is not asking about. That defect made this script
    #: report 0 of 8 solved while direct measurement at each spec's own channel showed
    #: 4 of 6 final designs passing all eight checks.
    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def succeeds(x, target, channel):
        """All 8 specs AND a valid circuit -- honest_benchmark --require-valid's test."""
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel,
                                   boost_target_tol_db=args.boost_tol)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception:
            return False
        if not v.is_valid:
            return False
        ok, _ = hard_pass(v.unwrap(), spec)
        return bool(ok)

    rows = []
    print(f"{len(specs)} held-out specs, deterministic rollout, "
          f"success = all 8 specs AND guard-valid\n")
    for i, (target, channel) in enumerate(specs):
        obs, _ = env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        used = 1
        solved_at = None
        if succeeds(env._x, target, channel):
            solved_at = used
        while solved_at is None and used < env.horizon:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            used += 1
            if succeeds(env._x, target, channel):
                solved_at = used
            if term or trunc:
                break
        rows.append({"target_boost_db": round(target, 2),
                     "channel_loss_db": round(channel, 2),
                     "sims": used, "solved_at": solved_at,
                     "design": decode_action(env._x).__dict__ if solved_at else None})
        print(f"  spec {i}: boost {target:5.2f} dB, channel {channel:5.2f} dB  ->  "
              f"{'SOLVED in ' + str(solved_at) + ' sims' if solved_at else 'not solved'}",
              flush=True)

    solved = [r for r in rows if r["solved_at"]]
    print(f"\nsolved {len(solved)}/{len(rows)}")
    if solved:
        s = sorted(r["solved_at"] for r in solved)
        med = s[len(s) // 2] if len(s) % 2 else (s[len(s)//2 - 1] + s[len(s)//2]) / 2
        print(f"median simulations to first success: {med}")
        print(f"parameter sweep, same success test  : 2394  "
              f"(results/sweep_baseline.json)")
        print(f"ratio vs the officially named baseline: {2394/med:,.0f}x")
    else:
        print("The policy solved nothing. The loop does not close yet; no speedup claim\n"
              "can be made from this checkpoint.")

    Path("results").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "specs": len(rows), "solved": len(solved),
         "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
