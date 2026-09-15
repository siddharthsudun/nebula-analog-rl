"""Does the reward pay the agent to become invalid on purpose?

Three separate attempts to improve this policy have made it worse: halving step_size
(2/32 -> 1/32), extending 40k -> 100k (2/32 -> 1/32, with guard rejections rising from
35.6% to 45.1% and T4.10_dc_gain_implausible tripling to 59% of all failures). A policy
that gets WORSE at producing valid circuits the longer it trains is not underfitting. It
is optimising something, and that something is not the task.

sequential_env.step() computes a per-step reward as an IMPROVEMENT, `score - self._score`,
but assigns `self._score = score` before the guard override replaces the reward. When the
guard rejects a candidate, `_measure` returns Measures(ok=False), `_shaped` scores that
-5.0, and the baseline is left at -5.0. The next valid step is then scored against a
baseline no real design produced, and collects the whole gap as improvement.

If that is right, a valid->invalid->valid excursion pays more than staying valid, and the
agent has every reason to take one. This replays real trajectories under the exact
training configuration and prices those excursions against ordinary valid steps.
"""
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
import json
import statistics
import numpy as np

from silq.envs.sequential_env import SequentialEqualizerEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_long100k.zip")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--out", default="results/reward_audit.json")
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)

    # Exactly the training configuration: guarded, fast, same horizon and step size.
    env = SequentialEqualizerEnv(fast=True, seed=4242, guarded=True)

    after_invalid, normal_valid, invalid_steps = [], [], []
    transitions, rows = 0, []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=5000 + ep)
        prev_invalid = False
        for t in range(env.horizon):
            a, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, info = env.step(a)
            is_invalid = "invalid_check" in info
            rows.append({"ep": ep, "t": t, "reward": round(float(r), 3),
                         "invalid": is_invalid,
                         "check": info.get("invalid_check"),
                         "after_invalid": prev_invalid and not is_invalid})
            if is_invalid:
                invalid_steps.append(float(r))
            elif prev_invalid:
                after_invalid.append(float(r))
                transitions += 1
            else:
                normal_valid.append(float(r))
            prev_invalid = is_invalid
            if term or trunc:
                break

    def stat(xs, name):
        if not xs:
            print(f"  {name:<34} (none)")
            return None
        print(f"  {name:<34} n={len(xs):<4} mean {statistics.mean(xs):+7.3f}  "
              f"max {max(xs):+7.3f}")
        return {"n": len(xs), "mean": statistics.mean(xs), "max": max(xs)}

    print(f"\nmodel {args.model}, {args.episodes} episodes, training config\n")
    a = stat(normal_valid, "valid step, previous also valid")
    b = stat(after_invalid, "valid step, previous INVALID")
    c = stat(invalid_steps, "invalid step (penalty)")

    print()
    if a and b:
        print(f"  a valid step is worth {b['mean'] - a['mean']:+.3f} more when the step "
              f"before it was rejected")
        excursion = (c["mean"] if c else 0.0) + b["mean"]
        print(f"  round trip valid->invalid->valid nets {excursion:+.3f} over two steps, "
              f"versus {2*a['mean']:+.3f} for two ordinary valid steps")
        if excursion > 2 * a["mean"]:
            print("\n  The excursion pays better. The agent is rewarded for entering the\n"
                  "  invalid region and returning, which is what the rising rejection\n"
                  "  rate during extended training looks like from the inside.")

    Path("results").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "episodes": args.episodes, "transitions": transitions,
         "normal_valid": a, "after_invalid": b, "invalid": c, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
