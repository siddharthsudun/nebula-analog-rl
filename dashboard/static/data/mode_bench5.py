"""All five CURRENT modes over one held-out 32-spec draw, in one run.

WHY A FRESH FULL RUN AND NOT A JOIN. `scratchpad/mode_bench32.json` benchmarks `accurate`,
a mode that no longer exists, and `scratchpad/retarget_ab32.json` holds default/retarget
from an earlier tree. Joining rows from three artifacts produced by three different trees
makes a table whose rows are not comparable, and the whole point of this table is a
like-for-like comparison. Every row here comes from one process, one tree, one spec draw.

SPEC SEED 137, not 0. Seed 0 IS the historical set every model-selection decision was made
on (`target_audit.make_specs` docstring); 137 is a clean held-out draw from the identical
distribution, and is the same draw the earlier mode artifacts used, so anyone who wants to
cross-check a single number against them can.

WHAT IS RECORDED FOR THE BASELINE. `cost.optimizer_evals` is the unit `--budget` counts and
the unit `chance_baseline.py`'s B means: one draw = one candidate design evaluated. It is
recorded per spec per mode so each mode can be scored against the chance line for the
budget IT ACTUALLY SPENT, spec by spec, rather than against a single shared line. A mode
that spends 10x the evaluations must clear a 10x-budget bar or the comparison is rigged.
`cost.measure_all_total` is recorded too, for the secondary line in the analysis.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
_NG = Path(os.environ["USERPROFILE"]) / "silq-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from silq import pipeline as pl  # noqa: E402

SPEC_SEED = 137
N_SPECS = 32
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/mode_bench5.json")


def main() -> None:
    from silq.experiments.final_comparison import load_policy
    from silq.experiments.target_audit import make_specs

    modes = list(pl.MODES)
    print("modes under test: %s" % ", ".join(modes), flush=True)
    if "accurate" in modes:
        raise SystemExit("pipeline.MODES still contains 'accurate'; this harness is for "
                         "the post-deletion mode set")

    specs = make_specs(N_SPECS, SPEC_SEED)
    load_policy(str(REPO / "results" / "seq_clean40k.zip"))
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, (tgt, chan) in enumerate(specs):
        for mode in modes:
            t = time.perf_counter()
            try:
                # pvt=False: this benchmark's whole point is comparing modes on their
                # nominal-stage search cost/accuracy. pipeline.design()'s default now
                # runs PVT sizing repair after verification (~80-150s of independent
                # 45-corner SPICE per solved spec) -- leaving it on would fold that
                # shared, mode-independent cost into every row and break the
                # like-for-like comparison this file exists for.
                r = pl.design(target_boost_db=tgt, channel_loss_db=chan, spec_index=i,
                              mode=mode, pvt=False)
            except Exception as exc:
                rows.append({"i": i, "target": tgt, "chan": chan, "mode": mode,
                             "error": "%s: %s" % (type(exc).__name__, exc)})
                OUT.write_text(json.dumps(rows, indent=1))
                print(f"{i:3d} {mode:9}  ERROR {exc}", flush=True)
                continue
            dt = time.perf_counter() - t
            v = r.get("verification") or {}
            p, c, sv = r["provenance"], r["cost"], r["solver"]
            md = p.get("mode_detail") or {}
            rows.append({"i": i, "target": tgt, "chan": chan, "mode": mode,
                         "wall_s": round(dt, 1), "abs_err_db": v.get("abs_err_db"),
                         "status": r["status"], "passed": v.get("passed"),
                         "reason": p["g32_reason"],
                         "optimizer_evals": c["optimizer_evals"],
                         "stage1_evals": c["stage1_evals"],
                         "stage2_evals": c["stage2_evals"],
                         "measure_all": c["measure_all_total"],
                         "guidance": r["guidance"]["headline"],
                         "suggest": r["guidance"]["suggest_mode"],
                         "n_ppo": md.get("n_ppo_rollouts"),
                         "n_surrogate": md.get("n_surrogate_starts"),
                         "governor": md.get("governor_stopped"),
                         # `auto` records its escalation here, not in mode_detail.
                         "auto_escalated": (r.get("auto") or {}).get("escalated"),
                         # _auto() folds the first attempt's measure_all and spice into
                         # the totals but NOT its optimizer_evals -- that one is only
                         # recorded under its own key. Carried through so the analysis
                         # can charge auto for every draw it actually took.
                         "optimizer_evals_prior": c.get("optimizer_evals_prior_attempts", 0),
                         "measure_all_prior": c.get("measure_all_prior_attempts", 0),
                         # The HOUSE chance-matching unit (final_report.py:272-274):
                         # k = min(distinct designs produced, designs that were feasible
                         # at all). Re-evaluating one design is not a second independent
                         # chance at the target, and a chance draw from a pool of achieved
                         # boosts already presumes a valid circuit. Recorded alongside the
                         # raw evaluation count so both conventions can be reported.
                         "n_distinct_boosts": sv["n_distinct_boosts"],
                         "n_loose_pass": sv["n_loose_pass"],
                         "n_valid": sv["n_valid"],
                         "solver_n_evals": sv["n_evals"]})
            OUT.write_text(json.dumps(rows, indent=1))
            e = v.get("abs_err_db")
            print(f"{i:3d} {tgt:5.1f}/{chan:4.1f} {mode:9} {dt:6.1f}s "
                  f"{'--' if e is None else format(e, '.4f'):>8} "
                  f"{c['optimizer_evals']:>4}oe {c['measure_all_total']:>5}ma  "
                  f"{r['status']}", flush=True)
        done = (i + 1) / N_SPECS
        el = time.perf_counter() - t0
        print(f"  -- spec {i+1}/{N_SPECS} done, {el/60:.1f} min elapsed, "
              f"~{el/done/60:.0f} min projected total", flush=True)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
