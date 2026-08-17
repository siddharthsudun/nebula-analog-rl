"""SILQ end-to-end: natural-language spec -> sized, characterized equalizer.

    python -m eqrl.solve "PCIe Gen2 CTLE with about 9 dB of peaking, under 12 mW"

Pipeline: LLM parses the request into a Spec (heuristic fallback with no API key) ->
the trained RL policy sizes a CTLE that meets it -> full characterization incl. the real
channel+DFE eye. This is the "describe it, get a circuit" demo (the bonus deliverable).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.llm.spec_parser import parse_spec
from eqrl.sim.measures import measure_all
from eqrl.specs import hard_pass


def design_for(spec, model_path: str, starts: int = 12):
    """Run the trained policy to find a design meeting `spec` (its target boost)."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = SequentialEqualizerEnv(fast=True, seed=0)
    for s in range(starts):
        obs, _ = env.reset(seed=s)
        env._target = spec.target_boost_db
        obs = env._obs(env._measure(env._x))
        for _ in range(env.horizon):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if info["passed"]:
                return DesignVars(**info["design"]), env.n_sims, s
            if trunc:
                break
    return None, env.n_sims, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("request", nargs="+", help="natural-language spec")
    # No default: there is no policy this can sensibly assume. The old default pointed at
    # results/seq_agent_nominal.zip, which the training script has never written, so the
    # demo failed with a stack trace from deep inside stable-baselines3 rather than saying
    # what was missing. Requiring the path means the caller states which policy they mean.
    p.add_argument("--model", required=True,
                   help="trained policy (.zip) written by eqrl.agents.train_sequential")
    p.add_argument("--out", default="results/solved_design.json")
    args = p.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(
            f"no policy at {args.model}. Train one first:\n"
            "  python -m eqrl.agents.train_sequential --guarded --no-fast "
            "--out results/seq_agent.zip")
    text = " ".join(args.request)

    print(f'  request : "{text}"')
    spec = parse_spec(text)
    print(f'  parsed  : target boost {spec.target_boost_db:.1f} dB, '
          f'power < {spec.power_w_max*1e3:.0f} mW, band {spec.peak_freq_lo_ghz}-'
          f'{spec.peak_freq_hi_ghz} GHz\n')

    dv, sims, start = design_for(spec, args.model)
    if dv is None:
        print(f"  no design met the spec within the search ({sims} sims). "
              "Try a target boost in 4-11 dB.")
        return

    m = measure_all(dv, fast=False)
    ok, checks = hard_pass(m, spec)
    print(f"  SILQ sized a CTLE in {sims} SPICE sims:\n")
    print(f"    W/L        {dv.w_in*1e6:.2f} / {dv.l_in*1e6:.2f} um")
    print(f"    I_tail     {dv.i_tail*1e6:.0f} uA")
    print(f"    Rs / Cs    {dv.rs/1e3:.2f} kOhm / {dv.cs*1e15:.0f} fF")
    print(f"    R_load     {dv.r_load:.0f} Ohm\n")
    print(f"    boost      {m.boost_db:.1f} dB @ {m.peak_freq_ghz:.2f} GHz")
    print(f"    HD3        {m.hd3_db:.0f} dB     noise {m.noise_vrms*1e6:.0f} uVrms")
    print(f"    power      {m.power_w*1e3:.2f} mW    area  {m.area_mm2:.4f} mm^2")
    print(f"    eye        {m.eye_h_ui:.2f} UI / {m.eye_v_mv:.0f} mV\n")
    print(f"  spec: {'ALL CHECKS PASS' if ok else 'FAILS: ' + ', '.join(k for k,v in checks.items() if not v)}")

    Path(args.out).write_text(json.dumps(dv.__dict__, indent=2))
    Path(args.out.replace(".json", ".spice")).write_text(
        netlist(dv, analysis="ac", models="sky130", corner="tt"))
    print(f"\n  saved design -> {args.out} and .spice netlist")


if __name__ == "__main__":
    main()
