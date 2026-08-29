"""SILQ end-to-end: natural-language spec -> sized, characterized, verified equalizer.

    python -m eqrl.solve "PCIe Gen2 CTLE with about 9 dB of peaking, under 12 mW"

Pipeline: the LLM parses the request into a Spec (heuristic fallback with no API key) ->
`eqrl.pipeline.design` runs the delivered architecture, PPO for global feasibility search
then G3.2 constrained refinement for specification closure -> the returned design is
re-measured through a fresh guarded evaluator on all ten checks -> the circuit, its
resulting specs, its netlist and its full provenance are written out.

WHAT CHANGED AND WHY. This entry point used to run PPO alone and then fall back to a
fixed hand-verified design, which meant the demo could not reproduce
`results/delivered_circuit.json` and its fallback was the very spec-ignoring fixed design
the project's own `results/target_tracking_clean40k.json` shows is not a solution. The
architecture now lives in one place (`eqrl.pipeline`) and both the benchmark and this
command run it.

THE FALLBACK IS OFF BY DEFAULT AND IS LABELLED WHEN ON. `--allow-fallback` fires only when
PPO -> G3.2 returns nothing guard-valid at all; the result then carries
`status = fallback_fixed_design_not_ai` and `provenance.is_ai_generated = False`, and this
command prints a banner saying so. It is a product escape hatch, never an AI result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eqrl.llm.spec_parser import parse_spec
from eqrl.pipeline import FALLBACK, POLICY, design, describe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("request", nargs="+", help="natural-language spec")
    p.add_argument("--model", default=POLICY,
                   help="trained policy (.zip); default is the frozen delivered policy")
    p.add_argument("--spec-index", type=int, default=0,
                   help="seeds stage 1's environment at 1000 + this; recorded in the "
                        "result so the run reproduces")
    p.add_argument("--channel-loss-db", type=float, default=None,
                   help="channel insertion loss at Nyquist; defaults to the parsed spec")
    p.add_argument("--allow-fallback", action="store_true",
                   help="if PPO -> G3.2 returns nothing guard-valid, return the FIXED "
                        "non-AI baseline design, labelled as such in the result")
    # NOT results/solved_design.json: that is a historical artifact cited by README.md,
    # RESULTS.md and HANDOVER.md ("fails 10 of 45 PVT corners") and must not be silently
    # rewritten by a demo run.
    p.add_argument("--out", default="results/solve_demo.json")
    args = p.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(
            f"no policy at {args.model}. The delivered policy is committed as "
            f"{POLICY}; to train your own:\n"
            "  python -m eqrl.agents.train_sequential --guarded --no-fast "
            "--out results/seq_agent.zip")

    text = " ".join(args.request)
    print(f'  request : "{text}"')
    spec = parse_spec(text)
    channel = args.channel_loss_db if args.channel_loss_db is not None \
        else spec.channel_loss_db
    print(f'  parsed  : target boost {spec.target_boost_db:.2f} dB over a '
          f'{channel:.2f} dB channel, power < {spec.power_w_max*1e3:.0f} mW, '
          f'band {spec.peak_freq_lo_ghz}-{spec.peak_freq_hi_ghz} GHz\n')

    r = design(spec.target_boost_db, channel, model=args.model,
               spec_index=args.spec_index, allow_fallback=args.allow_fallback)

    if r["status"] == FALLBACK:
        print("  " + "!" * 74)
        print("  !! The SILQ architecture produced no guard-valid design for this spec.")
        print("  !! What follows is the FIXED baseline design. It is NOT AI-generated,")
        print("  !! it does not read the requested target, and it must not be reported")
        print("  !! as an output of PPO or G3.2.")
        print("  " + "!" * 74)
    print(describe(r))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    deck = out.with_suffix(".spice")
    if r["netlist"]:
        deck.write_text(r["netlist"])
    print(f"\n  full result + provenance -> {out}")
    if r["netlist"]:
        print(f"  final schematic (netlist) -> {deck}")


if __name__ == "__main__":
    main()
