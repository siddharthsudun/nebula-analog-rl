"""Train the sequential design-closing agent across randomized target specs.

The trained policy learns to read the performance-vs-target gap and drive the sizing to
spec in a few steps, for *any* target boost in the range — the generalizing behaviour
that makes RL worthwhile over per-spec search.
"""
from __future__ import annotations

import argparse

from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.specs import DEFAULT_SPEC


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=40_000)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pvt", action="store_true", help="worst-case V x T robust training")
    p.add_argument("--out", default="results/seq_agent.zip")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    env = SequentialEqualizerEnv(spec=DEFAULT_SPEC, horizon=args.horizon,
                                 fast=True, seed=args.seed, pvt=args.pvt)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1,
                n_steps=1024, batch_size=128, gamma=0.95, gae_lambda=0.95,
                ent_coef=0.005, learning_rate=3e-4)
    # checkpoint so a converged model is on disk even if we stop early
    ckpt = CheckpointCallback(save_freq=2048, save_path="results/checkpoints",
                              name_prefix="seq")
    model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=ckpt)
    model.save(args.out)
    print(f"saved -> {args.out}  (total sims: {env.n_sims})")


if __name__ == "__main__":
    main()
