"""Why did 40k guarded steps end at 99.9% invalid, worse than random sampling?

Rolls the trained policy out deterministically and records where in the action space it
actually goes, against a uniform-random baseline measured the same way. If the policy has
collapsed onto a corner of the box, the normalised coordinates will sit at 0 or 1.
"""
import collections
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

import numpy as np
from stable_baselines3 import PPO

from silq.circuits.ctle import ACTION_SPACE, decode_action
from silq.envs.sequential_env import SequentialEqualizerEnv

KEYS = list(ACTION_SPACE)
model = PPO.load("results/seq_agent_guarded.zip")
env = SequentialEqualizerEnv(fast=False, seed=7, guarded=True)

EPISODES = 8
checks = collections.Counter()
final_x, all_x = [], []

for ep in range(EPISODES):
    obs, _ = env.reset(seed=100 + ep)
    for t in range(env.horizon):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        checks[info.get("invalid_check", "__valid__")] += 1
        all_x.append(env._x.copy())
        if term or trunc:
            break
    final_x.append(env._x.copy())
    print(f"  episode {ep}: ended t={t+1}, last check={info.get('invalid_check','VALID')}",
          flush=True)

all_x = np.array(all_x)
final_x = np.array(final_x)

print(f"\nsteps taken: {len(all_x)}")
total = sum(checks.values())
print(f"invalid: {(total - checks['__valid__'])/total*100:.1f}%")
for k, n in checks.most_common():
    print(f"   {n:4d}  {k}")

print("\nWhere the policy goes (normalised 0..1; 0 or 1 means pinned at a range edge):")
print(f"   {'param':8s} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}   physical at mean")
for i, k in enumerate(KEYS):
    col = all_x[:, i]
    lo, hi = ACTION_SPACE[k]
    phys = lo + float(col.mean()) * (hi - lo)
    print(f"   {k:8s} {col.mean():7.3f} {col.std():7.3f} {col.min():7.3f} {col.max():7.3f}"
          f"   {phys:.4g}")

frac_edge = float(((all_x < 0.02) | (all_x > 0.98)).mean())
print(f"\nfraction of visited coordinates pinned within 2% of a range edge: {frac_edge*100:.1f}%")

dvs = [decode_action(x) for x in all_x]
drop = np.array([(d.i_tail / 2) * d.r_load for d in dvs])
print(f"DC load drop demanded: median {np.median(drop):.2f} V, "
      f"{float((drop > 1.8).mean())*100:.0f}% exceed the 1.8 V supply")

Path("results").mkdir(exist_ok=True)
Path("results/policy_diagnosis.json").write_text(json.dumps({
    "episodes": EPISODES,
    "steps": len(all_x),
    "invalid_rate": round((total - checks["__valid__"]) / total, 4),
    "by_check": dict(checks.most_common()),
    "mean_normalised": {k: round(float(all_x[:, i].mean()), 4) for i, k in enumerate(KEYS)},
    "fraction_at_edge": round(frac_edge, 4),
    "median_load_drop_v": float(np.median(drop)),
    "fraction_drop_over_vdd": float((drop > 1.8).mean()),
}, indent=2))
print("\nwrote results/policy_diagnosis.json")
