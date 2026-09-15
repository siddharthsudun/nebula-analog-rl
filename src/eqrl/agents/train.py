"""Train an RL agent to size the equalizer.

Uses stable-baselines3 (PPO by default). Logs simulations-to-spec so we can plot the
sample-efficiency curve that is the submission's headline result.
"""
from __future__ import annotations

import argparse

from eqrl.envs.equalizer_env import EqualizerEnv
from eqrl.specs import DEFAULT_SPEC


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo", choices=["ppo", "ddpg", "sac"])
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--pvt", action="store_true", help="optimize across PVT corners")
    p.add_argument("--out", default="results/agent.zip")
    args = p.parse_args()

    try:
        from stable_baselines3 import PPO, DDPG, SAC
    except ImportError:
        raise SystemExit("pip install -r requirements.txt (stable-baselines3 missing)")

    env = EqualizerEnv(spec=DEFAULT_SPEC, horizon=args.horizon, pvt=args.pvt)
    Algo = {"ppo": PPO, "ddpg": DDPG, "sac": SAC}[args.algo]
    policy = "MlpPolicy"
    model = Algo(policy, env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"saved -> {args.out}")

    # quick eval: sample the greedy design
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)
    print(f"greedy design reward={reward:.2f} passed={info.get('passed')}")
    print("design:", info.get("design"))


if __name__ == "__main__":
    main()
