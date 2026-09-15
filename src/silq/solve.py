"""SILQ end-to-end: natural-language spec -> sized, characterized, verified equalizer.

    python -m silq.solve "PCIe Gen2 CTLE with about 9 dB of peaking, under 12 mW"

Pipeline: the LLM parses the request into a Spec (heuristic fallback with no API key) ->
`silq.pipeline.design` runs the delivered architecture, PPO for global feasibility search
then G3.2 constrained refinement for specification closure -> the returned design is
re-measured through a fresh guarded evaluator on all ten checks -> the circuit, its
resulting specs, its netlist and its full provenance are written out.

WHAT CHANGED AND WHY. This entry point used to run PPO alone and then fall back to a
fixed hand-verified design, which meant the demo could not reproduce
`results/delivered_circuit.json` and its fallback was the very spec-ignoring fixed design
the project's own `results/target_tracking_clean40k.json` shows is not a solution. The
architecture now lives in one place (`silq.pipeline`) and both the benchmark and this
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

from silq.llm.spec_parser import parse_spec_verbose
from silq.pipeline import FALLBACK, POLICY, design, describe


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
            "  python -m silq.agents.train_sequential --guarded --no-fast "
            "--out results/seq_agent.zip")

    text = " ".join(args.request)
    # flush: the refusal below goes to stderr, and an unflushed stdout would print the
    # request line *after* it -- which reads like a crash rather than a rejection.
    print(f'  request : "{text}"', flush=True)
    parsed = parse_spec_verbose(text)
    if not parsed.understood:
        # Refuse rather than run. The parser returns a DEFAULT Spec when it recognises
        # nothing, so proceeding here would print "target boost 9.00 dB" and size a
        # circuit for it, presenting a default as though it had been read out of the
        # request. Sizing the wrong circuit confidently is worse than not sizing one.
        raise SystemExit(
            f"  [{parsed.source}] nothing in that request was recognised as a design "
            "spec.\n  No target was inferred and nothing was simulated. State a target "
            "boost in dB,\n  e.g.  \"PCIe Gen2 CTLE, ~9 dB boost over a 12 dB channel, "
            "under 12 mW\"")
    spec = parsed.spec
    channel = args.channel_loss_db if args.channel_loss_db is not None \
        else spec.channel_loss_db
    print(f'  read    : {", ".join(sorted(parsed.recognised))} '
          f'(everything else below is a default)')
    # An ambiguous word ("gain" is peaking here, but it could have meant DC gain) is
    # resolved rather than refused -- and the resolution is printed, because a judgement
    # the user never sees is not distinguishable from one the parser invented.
    for _field, _why in sorted(parsed.assumptions.items()):
        print(f'  assumed : {_why}')
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
        print(f"  final netlist (SPICE) -> {deck}")

    # A netlist is the schematic only to someone who reads SPICE. `silq.schematic` has
    # rendered the sized circuit since the DFE branch merged, but nothing outside the
    # dashboard called it -- so the command that is meant to be the end-to-end demo
    # emitted a text file a judge cannot read at a glance. Drawing it costs nothing:
    # it is pure string formatting over the design vector, no simulation.
    if r.get("design"):
        try:
            from silq.circuits.ctle import DesignVars
            from silq.schematic import render
            m = (r.get("verification") or {}).get("measures") or {}
            sub = (f"{m['boost_db']:.2f} dB boost @ {m['peak_freq_ghz']:.2f} GHz, "
                   f"{m['power_w'] * 1e3:.2f} mW" if "boost_db" in m else "")
            svg = out.with_suffix(".svg")
            svg.write_text(render(DesignVars(**r["design"]),
                                  title="SILQ CTLE", subtitle=sub), encoding="utf-8")
            print(f"  final schematic (SVG) -> {svg}")
        except Exception as exc:            # noqa: BLE001 - drawing must never fail a run
            # The design and its provenance are already on disk; a rendering bug must not
            # turn a successful sizing run into a non-zero exit.
            print(f"  (schematic not drawn: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
