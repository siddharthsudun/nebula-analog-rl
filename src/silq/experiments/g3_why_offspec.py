"""Which hard_pass check do the on-target root-find designs fail? Diagnostic only."""
import dataclasses, json, os
from pathlib import Path
P = Path(os.environ.get("USERPROFILE") or Path.home()) / "silq-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))
import numpy as np
from silq.circuits.ctle import ACTION_SPACE, decode_action
from silq.evaluator import build_evaluator
from silq.specs import DEFAULT_SPEC, hard_pass
from silq.experiments.target_audit import make_specs
from silq.experiments.g3_rootfind import direction_from_probe, PREREG

DIMS = list(ACTION_SPACE.keys())
d_unit, _ = direction_from_probe("results/g3_probe_check.json")
art = json.load(open("results/g3_rootfind_smoke.json"))
specs = make_specs(32, 3)[:8]
guards = {}

def guard_for(c):
    if c not in guards:
        guards[c] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                    channel_loss_db=c)
    return guards[c]

from stable_baselines3 import PPO
from silq.envs.sequential_env import SequentialEqualizerEnv
model = PPO.load("results/seq_clean40k.zip")
env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

def stage1_x(i, t, c):
    env.reset(seed=1000 + i)
    env._target, env._channel = t, c
    obs = env._obs(env._measure(env._x))
    for _ in range(PREREG["k"] - 1):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _t, _tr, _ = env.step(a)
    return np.array(env._x, dtype=np.float64)

print("hard_pass checks on the design the solver landed ON TARGET, per spec")
print("(handoff row shown alongside so the CHANGE is visible)\n")
agg = {}
for r in art["rows"]:
    g = r["g3"]; steps = g["solver"].get("steps") or []
    if not steps:
        continue
    i, tgt, ch = r["spec"], r["target_boost_db"], r["channel_loss_db"]
    best = min((s for s in steps if s["valid"]), key=lambda s: s["abs_err"])
    x0 = stage1_x(i, tgt, ch)
    way = 1.0 if tgt > g["handoff"]["boost_db"] else -1.0
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=tgt, channel_loss_db=ch)
    out = []
    for label, x in (("handoff", x0),
                     ("on-target", np.clip(x0 + best["t"] * way * d_unit, 0, 1))):
        v = guard_for(ch).evaluate(decode_action(x), vdd=spec.vdd_nominal)
        if not v.is_valid:
            out.append((label, None, None)); continue
        m = v.unwrap(); ok, chk = hard_pass(m, spec)
        out.append((label, ok, chk))
    print("spec %2d  target %5.2f  t=%.3f  boost %.2f (err %.2f)"
          % (i, tgt, best["t"], best["boost_db"], best["abs_err"]))
    for label, ok, chk in out:
        if chk is None:
            print("    %-10s GUARD-INVALID" % label); continue
        fails = [k for k, v in chk.items() if not v]
        print("    %-10s hard_pass=%-5s  failing: %s"
              % (label, ok, ", ".join(fails) if fails else "none"))
        if label == "on-target":
            for k in fails:
                agg[k] = agg.get(k, 0) + 1
    print()
print("failing checks across the on-target designs:", agg)
