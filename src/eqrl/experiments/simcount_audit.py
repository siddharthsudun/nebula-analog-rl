"""What one "simulation" actually costs, at three levels of counting.

WHY. Every cost number this project publishes -- "median 4 simulations to solve", the
amortization break-even in `honest_benchmark` -- is denominated in a unit that was never
pinned down. Three different quantities have all been called "a simulation":

  1. OPTIMIZER EVALUATION -- one candidate proposed and scored. This is what a budget
     counts, what `--budget 20` means, and what `loose_solved_at` indexes.
  2. measure_all CALL -- what `SequentialEqualizerEnv.n_sims` increments, and what
     `honest_benchmark`'s docstring means by "1 simulation = 1 candidate evaluation
     (one measure_all)".
  3. SPICE ANALYSIS -- one ac/op/noise/transient run inside libngspice. A single
     `measure_all` issues several, and the guard layer issues more on top.

They are not the same number and the ratio is not 1. Two known gaps:

  * `target_audit`'s PPO arm calls `env.step()`, which measures internally, and THEN
    calls `evaluate()`, which measures again through the guard. One optimizer evaluation
    there is two independent measurement passes. The search baselines measure once.
  * the guard layer runs its own probes (bias, saturation, DC plausibility) beyond the
    measurement itself.

This module measures all three by counting at the chokepoints -- `measures.measure_all`
and `NgspiceServer._analysis`, the single function every SPICE analysis goes through --
rather than reasoning about the call graph.

It does NOT redefine any published number. The historical metric stays what it was; this
gives the conversion factor to state alongside it.

Reads only. Changes no threshold, bound, reward, or benchmark criterion.
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
from eqrl.experiments.target_audit import make_specs
from eqrl.sim import measures as measures_mod
from eqrl.sim.server import NgspiceServer
from eqrl.specs import DEFAULT_SPEC, hard_pass

COUNT = {"measure_all": 0, "analysis": 0}


def install_counters() -> None:
    """Count at the two chokepoints. Patching is the only honest way to do this: the
    call graph runs through the guard layer and libngspice, and any hand-count of it
    would be an assumption, which is the thing being audited."""
    real_measure_all = measures_mod.measure_all
    real_analysis = NgspiceServer._analysis

    def counted_measure_all(*a, **kw):
        COUNT["measure_all"] += 1
        return real_measure_all(*a, **kw)

    def counted_analysis(self, cmd):
        COUNT["analysis"] += 1
        return real_analysis(self, cmd)

    measures_mod.measure_all = counted_measure_all
    NgspiceServer._analysis = counted_analysis
    # The env imports measure_all by name at module import, so rebind it there too.
    import eqrl.envs.sequential_env as se
    se.measure_all = counted_measure_all


def take():
    v = dict(COUNT)
    COUNT["measure_all"] = COUNT["analysis"] = 0
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--specs", type=int, default=3, help="specs to average over")
    p.add_argument("--steps", type=int, default=6, help="PPO steps per spec to count")
    p.add_argument("--out", default="results/simcount_audit.json")
    args = p.parse_args()

    install_counters()
    from stable_baselines3 import PPO
    from eqrl.envs.sequential_env import SequentialEqualizerEnv

    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123)
    specs = make_specs(args.specs)
    guards: dict[float, object] = {}

    def guard_for(channel):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def verify(x, target, channel):
        """target_audit.evaluate, minus the bookkeeping: the guarded verification pass."""
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception:
            return None
        if not v.is_valid:
            return None
        return hard_pass(v.unwrap(), spec)[0]

    rows = []
    for i, (target, channel) in enumerate(specs):
        guard_for(channel)          # build outside the measured region
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        take()
        env._obs(env._measure(env._x))
        reset_cost = take()
        verify(env._x, target, channel)
        verify_cost = take()

        step_costs, ver_costs = [], []
        for _ in range(args.steps):
            a, _ = model.predict(env._obs(env._measure(env._x)), deterministic=True)
            take()                                  # discard the obs re-measure above
            env.step(a)
            step_costs.append(take())
            verify(env._x, target, channel)
            ver_costs.append(take())

        def mean(rs, k):
            return sum(r[k] for r in rs) / max(len(rs), 1)

        rows.append({
            "spec": i, "target_boost_db": target, "channel_loss_db": channel,
            "reset_measure_all": reset_cost["measure_all"],
            "reset_analysis": reset_cost["analysis"],
            "verify_at_reset_measure_all": verify_cost["measure_all"],
            "verify_at_reset_analysis": verify_cost["analysis"],
            "env_step_measure_all": mean(step_costs, "measure_all"),
            "env_step_analysis": mean(step_costs, "analysis"),
            "verify_measure_all": mean(ver_costs, "measure_all"),
            "verify_analysis": mean(ver_costs, "analysis"),
        })
        print("  spec %d: env.step -> %.2f measure_all / %.1f analyses | "
              "verify -> %.2f measure_all / %.1f analyses"
              % (i, rows[-1]["env_step_measure_all"], rows[-1]["env_step_analysis"],
                 rows[-1]["verify_measure_all"], rows[-1]["verify_analysis"]),
              flush=True)

    def avg(k):
        return sum(r[k] for r in rows) / len(rows)

    step_ma, step_an = avg("env_step_measure_all"), avg("env_step_analysis")
    ver_ma, ver_an = avg("verify_measure_all"), avg("verify_analysis")
    summary = {
        "ppo_evaluation_measure_all": step_ma + ver_ma,
        "ppo_evaluation_analyses": step_an + ver_an,
        "search_evaluation_measure_all": ver_ma,
        "search_evaluation_analyses": ver_an,
        "ppo_over_search_measure_all": (step_ma + ver_ma) / max(ver_ma, 1e-9),
        "ppo_over_search_analyses": (step_an + ver_an) / max(ver_an, 1e-9),
    }
    print("\nONE OPTIMIZER EVALUATION COSTS:")
    print("  PPO arm     : %.2f measure_all, %.1f SPICE analyses  "
          "(env.step %.2f + guarded verify %.2f)"
          % (summary["ppo_evaluation_measure_all"], summary["ppo_evaluation_analyses"],
             step_ma, ver_ma))
    print("  search arms : %.2f measure_all, %.1f SPICE analyses"
          % (ver_ma, ver_an))
    print("  ratio       : %.2fx measure_all, %.2fx analyses"
          % (summary["ppo_over_search_measure_all"],
             summary["ppo_over_search_analyses"]))
    Path(args.out).write_text(json.dumps({"model": args.model, "specs": args.specs,
                                          "steps": args.steps, "summary": summary,
                                          "rows": rows}, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
