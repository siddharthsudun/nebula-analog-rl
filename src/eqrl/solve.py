"""SILQ end-to-end: natural-language spec -> sized, characterized equalizer.

    python -m eqrl.solve "PCIe Gen2 CTLE with about 9 dB of peaking, under 12 mW"

Pipeline: LLM parses the request into a Spec (heuristic fallback with no API key) ->
the trained RL policy sizes a CTLE that meets it -> full characterization incl. the real
channel+DFE eye. This is the "describe it, get a circuit" demo (the bonus deliverable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.evaluator import build_evaluator
from eqrl.envs.sequential_env import SequentialEqualizerEnv
from eqrl.llm.spec_parser import parse_spec
from eqrl.specs import DEFAULT_SPEC, hard_pass


def design_for(spec, model_path: str | None, starts: int = 12):
    """Run the trained policy to find a design meeting `spec` (its target boost)."""
    if model_path is None:
        return None, 0, None
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = SequentialEqualizerEnv(fast=True, seed=0)
    for s in range(starts):
        obs, _ = env.reset(seed=s)
        env._target = spec.target_boost_db
        env._channel = spec.channel_loss_db
        obs = env._obs(env._measure(env._x))
        for _ in range(env.horizon):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if info["passed"]:
                return DesignVars(**info["design"]), env.n_sims, s
            if trunc:
                break
    return None, env.n_sims, None


def validate_design(dv: DesignVars, spec):
    """Validate one candidate through the same guard + hard-pass path as evaluation."""
    evaluator = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                channel_loss_db=spec.channel_loss_db)
    verdict = evaluator.evaluate(dv, vdd=spec.vdd_nominal)
    if not verdict.is_valid:
        return None, False, {"guard": verdict.check.value}
    measures = verdict.unwrap()
    ok, checks = hard_pass(measures, spec)
    return measures, ok, checks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("request", nargs="+", help="natural-language spec")
    p.add_argument("--model", default=None,
                   help="trained policy (.zip) written by eqrl.agents.train_sequential")
    p.add_argument("--baseline-fallback", action="store_true",
                   help="use the verified robust design if the policy is missing or fails")
    p.add_argument("--baseline-only", action="store_true",
                   help="skip PPO and use the verified robust design directly")
    p.add_argument("--out", default="results/solved_design.json")
    args = p.parse_args()
    if args.model and not Path(args.model).exists() and not args.baseline_fallback:
        raise SystemExit(
            f"no policy at {args.model}. Train one first or enable --baseline-fallback:\n"
            "  python -m eqrl.agents.train_sequential --guarded --no-fast "
            "--out results/seq_agent.zip")
    if not args.model and not args.baseline_fallback and not args.baseline_only:
        raise SystemExit("pass --model, or use --baseline-fallback/--baseline-only")
    text = " ".join(args.request)

    print(f'  request : "{text}"')
    spec = parse_spec(text)
    print(f'  parsed  : target boost {spec.target_boost_db:.1f} dB, '
          f'power < {spec.power_w_max*1e3:.0f} mW, band {spec.peak_freq_lo_ghz}-'
          f'{spec.peak_freq_hi_ghz} GHz\n')

    dv = None
    m = None
    ok = False
    checks = {}
    sims = 0
    source = "PPO policy"
    if not args.baseline_only and args.model and Path(args.model).exists():
        candidate, sims, _ = design_for(spec, args.model)
        if candidate is not None:
            m, ok, checks = validate_design(candidate, spec)
            if ok:
                dv = candidate
            else:
                print("  PPO candidate failed full validation; considering fallback.")

    if dv is None and (args.baseline_fallback or args.baseline_only):
        from eqrl.baselines.robust import robust_design
        source = "verified robust baseline"
        dv = robust_design()
        m, ok, checks = validate_design(dv, spec)
        sims += 1

    if dv is None or m is None:
        print(f"  no design met the spec within the search ({sims} sims). "
              "Try --baseline-fallback or a target boost in 5-11 dB.")
        return

    print(f"  SILQ selected the {source} in {sims} SPICE sims:\n")
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
