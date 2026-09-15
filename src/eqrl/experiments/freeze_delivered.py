"""Freeze the delivered SILQ circuit into one self-contained, checksummed manifest.

This module MEASURES NOTHING and OPTIMIZES NOTHING. It reads artifacts that already
exist, checksums them, and writes `results/delivered_circuit.json` — the single file the
site, the writeup and any future reader should quote from, so that no downstream artifact
has to re-derive a number and risk deriving a different one.

What it records, and where each field comes from:

    design            results/pvt_signoff_seed23.json  (the flagship's `design`)
    spec              results/final_comparison_seed23.json  (the held-out spec it was
                      asked for: target boost and channel loss)
    provenance        the arm-B solver trace for that spec — PPO stage 1, then every
                      G3.2 step, with the boost after each one
    pvt               all 45 corners, plus the per-metric worst case across them
    checksums         sha256 of the policy, both source artifacts and the netlist
    commit            the HEAD that produced the sweep

The manifest is regenerated, never hand-edited. If a number in it disagrees with a number
on the site, the manifest is right.

    PYTHONPATH=src python -m eqrl.experiments.freeze_delivered
    PYTHONPATH=src python -m eqrl.experiments.freeze_delivered --verify   # exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.sim.server import AC_DECADE_PTS, AC_FSTART, AC_FSTOP
from eqrl.specs import DEFAULT_SPEC

SIGNOFF = "results/pvt_signoff_seed23.json"
COMPARISON = "results/final_comparison_seed23.json"
POLICY = "results/seq_clean40k.zip"
OUT = "results/delivered_circuit.json"

#: The requirement set, as (metric, lo, hi) with None for "not bounded on that side".
#: Transcribed from specs.Spec so the manifest states the bound beside every number
#: instead of asking the reader to go and look it up.
LIMITS = [
    ("boost_db", None, None),                 # bounded by the target +/- tol, below
    ("peak_freq_ghz", DEFAULT_SPEC.peak_freq_lo_ghz, DEFAULT_SPEC.peak_freq_hi_ghz),
    ("dc_gain_db", DEFAULT_SPEC.dc_gain_db_min, None),
    ("hd3_db", None, DEFAULT_SPEC.hd3_db_max),
    ("noise_vrms", None, DEFAULT_SPEC.noise_vrms_max),
    ("power_w", None, DEFAULT_SPEC.power_w_max),
    ("area_mm2", None, DEFAULT_SPEC.area_mm2_max),
    ("eye_h_ui", DEFAULT_SPEC.eye_h_ui_min, None),
    ("eye_v_mv", DEFAULT_SPEC.eye_v_mv_min, None),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build() -> dict:
    from eqrl.experiments.pvt_signoff import flagship, score

    sweep = json.loads(Path(SIGNOFF).read_text())
    if not sweep.get("complete"):
        raise SystemExit("REFUSING: %s is from an incomplete sweep." % SIGNOFF)
    win, how = flagship(sweep["candidates"])
    if win is None or how != "PVT-clean":
        raise SystemExit(
            "REFUSING: the sign-off did not produce a PVT-clean flagship (%s). "
            "docs/PREREG_PVT_SIGNOFF.md §5 says that branch is reported as the "
            "headline, not frozen as a delivered circuit." % how)

    comp = json.loads(Path(COMPARISON).read_text())
    row = next(r for r in comp["rows"] if r["spec"] == win["spec_index"])
    b = row["b"]
    if b["best_design"] != win["design"]:
        raise SystemExit("REFUSING: the swept design is not the arm-B design for spec %d."
                         % win["spec_index"])

    dv = DesignVars(**win["design"])
    deck = netlist(dv, vdd=DEFAULT_SPEC.vdd_nominal, temp_c=27.0, corner="tt",
                   analysis="ac", models="sky130")
    Path("results/pvt_signoff_flagship.spice").write_text(deck)

    recs = win["corners"]
    worst = {}
    for name, lo, hi in LIMITS:
        vals = [(r[name], k) for k, r in recs.items()]
        worst[name] = {"min": min(vals)[0], "min_at": min(vals)[1],
                       "max": max(vals)[0], "max_at": max(vals)[1],
                       "limit_lo": lo, "limit_hi": hi}
    errs = sorted((abs(r["boost_db"] - row["target_boost_db"]), k)
                  for k, r in recs.items())

    s = score(win)
    return {
        "what": "the delivered SILQ circuit — frozen, not to be re-optimized",
        "generated_by": "eqrl.experiments.freeze_delivered (measures nothing)",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),

        "design": win["design"],
        "device_sizes_note": (
            "w_in/l_in are the input pair (XM1/XM2). The tail mirror devices are sized "
            "by circuits.ctle from i_tail; see the netlist for what is instantiated."),

        "spec": {
            "source": "held-out spec set, make_specs(40, seed=23), index %d"
                      % win["spec_index"],
            "spec_seed": comp["spec_seed"],
            "spec_index": win["spec_index"],
            "target_boost_db": row["target_boost_db"],
            "channel_loss_db": row["channel_loss_db"],
            "boost_tol_db": comp["tol"],
            "requirement_set": "eqrl.specs.Spec defaults + this target/channel; "
                               "dc_gain_db_min=0.0 and boost_target_tol_db=1.5 both ON",
        },

        "provenance": {
            "architecture": "PPO (global feasibility) -> G3.2 constrained refinement",
            "policy": POLICY,
            "policy_sha256": sha256(POLICY),
            "ppo_deterministic": True,
            "g32_base_seed": comp["base_seed"],
            "stage1_ppo": {
                "n_evals": row["stage1"]["n_evals"],
                "best_boost_db": row["stage1"]["best_boost_db"],
                "best_abs_err_db": row["stage1"]["best_abs_err"],
                "handoff_guard_valid": row["handoff"]["guard_valid"],
            },
            "g32_start_source": b["solver"]["start_source"],
            "g32_steps": [{"phase": s_["phase"], "boost_db": s_["boost_db"],
                           "abs_err_db": s_["abs_err"]} for s_ in b["solver"]["steps"]],
            "g32_reason": b["solver"]["reason"],
            "g32_rescue_used": b["solver"]["rescue_used"],
            "g32_repair_used": b["solver"]["repair_used"],
            "total_evals": b["n_evals"],
            "stage2_evals": b["stage2_evals"],
            "measure_all_spent": b["measure_all_spent"],
            "measure_all_budget": comp["prereg"]["budget_measure_all"],
            "hand_tuning": "none — the design is exactly what G3.2 returned",
        },

        "measurement": {
            "instrument": "eqrl.evaluator.build_evaluator(fast=False) — full guard layer, "
                          "real HD3 and noise",
            "ac_sweep": {"decade_pts": AC_DECADE_PTS, "fstart_hz": AC_FSTART,
                         "fstop_hz": AC_FSTOP},
            "ac_sweep_note": (
                "peak_freq_ghz is quantised onto this logarithmic grid. The .ac line "
                "inside the exported netlist is circuits.ctle's default viewing sweep "
                "and is NOT the sweep that produced these numbers."),
            "pdk": "SKY130 (sky130_fd_pr__nfet_01v8), ngspice via libngspice",
        },

        "pvt": {
            "grid": "5 process x VDD +/-5%% x {0, 27, 125} C = 45 corners",
            "processes": sweep["grid"]["procs"],
            "vdds": sweep["grid"]["vdds"],
            "temps_c": sweep["grid"]["temps"],
            "corners_passed": s["n_pass10"],
            "corners_total": 45,
            "all_guard_valid": s["n_guard_valid"] == 45,
            "worst_corner_target_err_db": s["worst_err_db"],
            "worst_corner": errs[-1][1],
            "best_corner": errs[0][1],
            "worst_case_by_metric": worst,
            "tt_nominal": recs["tt|1.800|27"],
            "corners": recs,
        },

        "context": {
            "pool": "all %d strict-solved arm-B candidates were swept; %d is PVT-clean"
                    % (sweep["n_candidates"],
                       sum(1 for c in sweep["candidates"] if score(c)["clean"])),
            "trained_at_corners": False,
            "honest_statement": (
                "This generated candidate passed all 45 tested corners. The optimisation "
                "itself was performed at TT only, so this is an out-of-distribution "
                "measurement of one design, NOT evidence that SILQ produces PVT-robust "
                "designs in general."),
        },

        "checksums": {
            SIGNOFF: sha256(SIGNOFF),
            COMPARISON: sha256(COMPARISON),
            POLICY: sha256(POLICY),
            "results/pvt_signoff_flagship.spice": sha256(
                "results/pvt_signoff_flagship.spice"),
        },
        "netlist": deck,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="rebuild and diff against the committed manifest; exit 1 on drift")
    args = p.parse_args()

    m = build()
    out = Path(OUT)
    if args.verify:
        if not out.exists():
            raise SystemExit("no manifest at %s" % OUT)
        old = json.loads(out.read_text())
        drift = [k for k in m if k != "commit" and old.get(k) != m[k]]
        if drift:
            print("DRIFT in: " + ", ".join(drift))
            raise SystemExit(1)
        print("VERIFIED: %s matches the artifacts it was built from." % OUT)
        return

    out.write_text(json.dumps(m, indent=1))
    print("delivered circuit -> %s" % OUT)
    print("  spec %d   target %.4f dB over a %.2f dB channel"
          % (m["spec"]["spec_index"], m["spec"]["target_boost_db"],
             m["spec"]["channel_loss_db"]))
    print("  TT boost %.4f dB   45-corner worst error %.4f dB   corners passed %d/45"
          % (m["pvt"]["tt_nominal"]["boost_db"], m["pvt"]["worst_corner_target_err_db"],
             m["pvt"]["corners_passed"]))
    print("  policy sha256 %s" % m["provenance"]["policy_sha256"][:16])
    print("  commit %s" % m["commit"][:12])


if __name__ == "__main__":
    main()
