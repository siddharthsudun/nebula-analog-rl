"""Does validity decay WITHIN an episode, before any learning happens?

Both A/B runs are still short of PPO's first update at step 1024, so the acting policy is
untrained. Yet measured validity fell from 10.8% over the first 500 steps to 3.8% over the
next 500. A policy that has not been updated cannot have "collapsed", so the cause is the
environment's own dynamics.

SequentialEqualizerEnv.step does x <- clip(x + 0.18 * action, 0, 1) over a 20-step horizon.
With actions in [-1, 1] that is a random walk whose spread after 20 steps is roughly
0.18 * sqrt(20) = 0.8 of the full box width, so a trajectory starting anywhere ends up
pressed against the edges of [0,1]^6 — and the edges are where the PDK bounds and the
unbuildable corners live.

If validity decays with step index under a random policy, the horizon and step size are
fighting the agent no matter what the reward says, and no amount of reward shaping fixes it.
Measured against step index, averaged over episodes.
"""
import collections
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

import numpy as np

from eqrl.envs.sequential_env import SequentialEqualizerEnv

EPISODES = int(os.environ.get("ED_EPISODES", "10"))
env = SequentialEqualizerEnv(fast=True, seed=3, guarded=True)   # fast: shape, not level
rng = np.random.default_rng(3)

valid_at = collections.Counter()
total_at = collections.Counter()
edge_at = collections.defaultdict(list)

for ep in range(EPISODES):
    env.reset(seed=200 + ep)
    for t in range(env.horizon):
        a = rng.uniform(-1.0, 1.0, env.action_space.shape[0])
        _, _, term, trunc, info = env.step(a)
        total_at[t] += 1
        valid_at[t] += int("invalid_check" not in info)
        x = env._x
        edge_at[t].append(float(((x < 0.02) | (x > 0.98)).mean()))
        if term or trunc:
            break
    print(f"  episode {ep} done", flush=True)

print(f"\n{'step':>5} {'valid':>12}  {'coords at a box edge':>22}")
rows = []
for t in sorted(total_at):
    n, v = total_at[t], valid_at[t]
    e = float(np.mean(edge_at[t]))
    print(f"{t:5d} {v:4d}/{n:<4d} {v/n*100:5.1f}%  {e*100:20.1f}%")
    rows.append({"step": t, "n": n, "valid": v, "fraction": round(v / n, 4),
                 "edge_fraction": round(e, 4)})

early = [r for r in rows if r["step"] < 5]
late = [r for r in rows if r["step"] >= 15]
fe = sum(r["valid"] for r in early) / max(1, sum(r["n"] for r in early))
fl = sum(r["valid"] for r in late) / max(1, sum(r["n"] for r in late))
ee = float(np.mean([r["edge_fraction"] for r in early]))
el = float(np.mean([r["edge_fraction"] for r in late]))
print(f"\nfirst 5 steps : {fe*100:5.1f}% valid, {ee*100:5.1f}% of coords at an edge")
print(f"last 5 steps  : {fl*100:5.1f}% valid, {el*100:5.1f}% of coords at an edge")
print("\nUntrained policy. Any decay here belongs to the env's step size and horizon,\n"
      "not to the reward or the agent.")

Path("results").mkdir(exist_ok=True)
Path("results/episode_drift.json").write_text(json.dumps(
    {"episodes": EPISODES, "step_size": env.step_size, "horizon": env.horizon,
     "rows": rows, "first5_valid": round(fe, 4), "last5_valid": round(fl, 4),
     "first5_edge": round(ee, 4), "last5_edge": round(el, 4)}, indent=2))
print("\nwrote results/episode_drift.json")
