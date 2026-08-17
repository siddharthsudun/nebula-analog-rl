"""Train the sequential design-closing agent across randomized target specs.

The trained policy learns to read the performance-vs-target gap and drive the sizing to
spec in a few steps, for *any* target boost in the range — the generalizing behaviour
that makes RL worthwhile over per-spec search.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.specs import DEFAULT_SPEC


def _invalid_logger(env, path: Path, every: int = 500):
    """Record which guard check rejected each step, and how often.

    A single invalid *rate* is not enough to act on. If most rejections are one check
    firing on one edge of the action space, that is a fact about the search space rather
    than about the designs, and it changes what the run means -- so the breakdown is
    written alongside the model instead of being reconstructed afterwards.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _Log(BaseCallback):
        def __init__(self):
            super().__init__()
            self.counts = collections.Counter()
            self.t0 = time.time()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                self.counts[info.get("invalid_check", "__valid__")] += 1
            if self.num_timesteps % every == 0:
                self._dump()
            return True

        def _dump(self):
            total = sum(self.counts.values()) or 1
            invalid = total - self.counts["__valid__"]
            path.write_text(json.dumps({
                "timesteps": int(self.num_timesteps),
                "elapsed_s": round(time.time() - self.t0, 1),
                "n_sims": getattr(env, "n_sims", None),
                "n_invalid": getattr(env, "n_invalid", None),
                "invalid_rate": round(invalid / total, 4),
                "by_check": dict(self.counts.most_common()),
            }, indent=2))

        def _on_training_end(self) -> None:
            self._dump()

    return _Log()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=40_000)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pvt", action="store_true", help="worst-case V x T robust training")
    p.add_argument("--guarded", action="store_true",
                   help="validate every candidate through the guard layer (Tiers 1-4); "
                        "rejected designs score --invalid-reward instead of a measurement")
    p.add_argument("--fast", dest="fast", action="store_true",
                   help="stub HD3 and noise instead of simulating them (much faster, "
                        "and four metrics stop being measurements)")
    p.add_argument("--no-fast", dest="fast", action="store_false")
    p.set_defaults(fast=False)
    p.add_argument("--feasible-decode", action="store_true",
                   help="EXPERIMENT: project R_load onto what the supply can drive, so "
                        "the agent is handed the nearest buildable design instead of a "
                        "flat penalty. Changes what the search space means.")
    p.add_argument("--out", default="results/seq_agent.zip")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    env = SequentialEqualizerEnv(spec=DEFAULT_SPEC, horizon=args.horizon,
                                 fast=args.fast, seed=args.seed, pvt=args.pvt,
                                 guarded=args.guarded,
                                 feasible_decode=args.feasible_decode)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1,
                n_steps=1024, batch_size=128, gamma=0.95, gae_lambda=0.95,
                ent_coef=0.005, learning_rate=3e-4)
    # checkpoint so a converged model is on disk even if we stop early
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointCallback(save_freq=2048, save_path="results/checkpoints",
                              name_prefix="seq")
    cbs = [ckpt]
    if args.guarded:
        cbs.append(_invalid_logger(env, out.with_name(out.stem + "_invalid.json")))
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False,
                callback=CallbackList(cbs))
    model.save(args.out)
    mins = (time.time() - t0) / 60.0
    # The training debt is the headline number in the amortization comparison, so it is
    # recorded next to the model that incurred it rather than passed in by hand later.
    out.with_name(out.stem + "_train.json").write_text(json.dumps({
        "model": str(out),
        "timesteps": args.timesteps,
        "n_sims": env.n_sims,
        "n_invalid": env.n_invalid,
        "wall_minutes": round(mins, 1),
        "seed": args.seed,
        "fast": args.fast,
        "guarded": args.guarded,
        "pvt": args.pvt,
        "horizon": args.horizon,
        "feasible_decode": args.feasible_decode,
    }, indent=2))
    print(f"saved -> {args.out}  (total sims: {env.n_sims}, "
          f"invalid: {env.n_invalid}, wall: {mins:.1f} min)")


if __name__ == "__main__":
    main()
