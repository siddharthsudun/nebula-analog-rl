"""45-corner PVT sign-off for the delivered SILQ circuit.

Preregistered in `docs/PREREG_PVT_SIGNOFF.md`, written before any corner was simulated
and before any per-spec value in the seed-23 artifact was inspected. This module runs
that document and nothing else: it selects no candidate by hand, and every rule it
applies is written there first.

WHAT IT DOES
    pool     arm B (PPO -> G3.2) rows of results/final_comparison_seed23.json whose
             strict_solved_at is not None -- i.e. all nine hard checks plus |boost -
             target| <= 1.5 dB at TT/1.8 V/27 C.  Not sampled, not trimmed.
    sweep    every candidate over all 45 corners: {tt,ss,ff,sf,fs} x {1.71,1.80,1.89} V
             x {0,27,125} C, through build_evaluator(fast=False) -- the GUARD LAYER and
             REAL HD3/noise, not the fast approximations.
    verdict  per corner, `pass9` (the nine checks every published number used) and
             `pass10` (those plus `boost_target` at 1.5 dB, which the delivered circuit
             is claimed to hit).  Both are stored; the flagship rule uses pass10.

WHY NOT experiments/characterize.py
    It sweeps the same grid but calls measure_all directly, so no corner is ever shown
    to the guard, and it writes results/final_report.json -- which is already the
    honest-benchmark final report and unrelated to this work.  Left untouched.

TWO THINGS THE GUARD DOES THAT A SWEEP MUST HANDLE
    Tier 5 checks 16-20 are statements about a SEARCH, and this is not one.  Check 19
    (INVALID rate too low over the last 100 evaluations) cannot fire here: evaluators
    are cached per (process, temperature, channel) and each sees at most three records.
    Check 20 (one design beats 3+ spec targets by >30%) can fire on a single record.  It
    raises SearchHalted, which is caught, RECORDED on the corner, and COUNTED AS A
    FAILURE -- never dropped and never silently passed.  A caught halt is reported in
    the summary by name.

Changes no reward, no PPO hyperparameter, no design bound, no guard threshold, no spec
field, and no recorded number.  Writes only its own artifacts.

    PYTHONPATH=src python -m eqrl.experiments.pvt_signoff
    PYTHONPATH=src python -m eqrl.experiments.pvt_signoff --limit 1   # smoke, 45 corners
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.envs.pvt import corner_grid
from eqrl.evaluator import build_evaluator
from eqrl.guards import SearchHalted
from eqrl.specs import DEFAULT_SPEC, Spec, hard_pass

#: docs/PREREG_PVT_SIGNOFF.md §2-§5, transcribed.  Changing one of these changes what the
#: sign-off means, so they live in one place and are printed at the top of every run.
PREREG = {
    "source": "results/final_comparison_seed23.json",
    "arm": "b",                       # PPO -> G3.2, the delivered architecture
    "pool_rule": "strict_solved_at is not None",
    "grid_mode": "full",              # 5 process x 3 vdd x 3 temp = 45
    "fast": False,                    # real HD3 + noise
    "boost_target_tol_db": 1.5,       # the tenth check, stricter than anything recorded
    "clean_rule": "45/45 guard_valid and pass10",
    "tie_db": 0.01,
}

METRICS = ("dc_gain_db", "peak_gain_db", "boost_db", "peak_freq_ghz", "hd3_db",
           "noise_vrms", "power_w", "area_mm2", "eye_h_ui", "eye_v_mv")


def spec_for(target: float, channel: float, *, with_target_check: bool) -> Spec:
    """The candidate's own spec.  DEFAULT_SPEC's 9 dB target is never substituted."""
    return dataclasses.replace(
        DEFAULT_SPEC, target_boost_db=target, channel_loss_db=channel,
        boost_target_tol_db=(PREREG["boost_target_tol_db"] if with_target_check else None))


def load_pool(path: str, arm: str) -> list[dict]:
    """§2.  Every strict-solved row of the arm, in spec order.  No selection here."""
    data = json.loads(Path(path).read_text())
    pool = []
    for row in data["rows"]:
        res = row[arm]
        if res.get("strict_solved_at") is None:
            continue
        pool.append({
            "spec_index": row["spec"],
            "target_boost_db": row["target_boost_db"],
            "channel_loss_db": row["channel_loss_db"],
            "tt_best_abs_err": res["best_abs_err"],
            "tt_best_boost_db": res["best_boost_db"],
            "design": res["best_design"],
        })
    return pool


def sweep_one(ev, dv: DesignVars, vdd: float, spec10: Spec, spec9: Spec) -> dict:
    """One corner.  Returns the record; never raises for a bad design."""
    rec: dict = {"guard_valid": False, "guard_check": None, "halted": None}
    try:
        v = ev.evaluate(dv, vdd=vdd)
    except SearchHalted as e:
        # A Tier 5 statement about a search, made during a sweep.  Recorded by name and
        # counted as a failing corner -- see the module docstring.
        rec["halted"] = str(e)[:200]
        return rec
    except Exception as e:                                  # noqa: BLE001
        rec["guard_check"] = "exception:%s" % repr(e)[:120]
        return rec
    if not v.is_valid:
        rec["guard_check"] = str(getattr(v.check, "value", v.check))
        return rec
    m = v.unwrap()
    ok10, checks10 = hard_pass(m, spec10)
    ok9, _ = hard_pass(m, spec9)
    rec.update({"guard_valid": True, "pass10": bool(ok10), "pass9": bool(ok9),
                "checks": {k: bool(b) for k, b in checks10.items()},
                "failing": [k for k, b in checks10.items() if not b]})
    rec.update({k: float(getattr(m, k)) for k in METRICS})
    return rec


def run(pool: list[dict], out: Path, log_every: int = 1) -> dict:
    """§3.  Process corner is the OUTER loop: get_server reloads models on a corner
    change (~15 s), so any other ordering pays that 45 times instead of 5."""
    grid = corner_grid(DEFAULT_SPEC, mode="full")
    procs = list(dict.fromkeys(p for p, _, _ in grid))
    vdds = list(dict.fromkeys(v for _, v, _ in grid))
    temps = list(dict.fromkeys(t for _, _, t in grid))
    assert len(procs) * len(vdds) * len(temps) == 45, "the grid is not 45 corners"

    for c in pool:
        c["corners"] = {}
    evs: dict[tuple, object] = {}
    report = {"prereg": PREREG, "grid": {"procs": procs, "vdds": vdds, "temps": temps},
              "n_candidates": len(pool), "n_corners": 45, "complete": False,
              "candidates": pool}
    done = t0 = 0
    t0 = time.time()
    total = len(pool) * 45

    for proc in procs:
        for temp in temps:
            for c in pool:
                key = (proc, temp, c["channel_loss_db"])
                if key not in evs:
                    evs[key] = build_evaluator(
                        spec_for(c["target_boost_db"], c["channel_loss_db"],
                                 with_target_check=True),
                        corner=proc, fast=PREREG["fast"], temp_c=temp,
                        channel_loss_db=c["channel_loss_db"])
                ev = evs[key]
                dv = DesignVars(**c["design"])
                s10 = spec_for(c["target_boost_db"], c["channel_loss_db"],
                               with_target_check=True)
                s9 = spec_for(c["target_boost_db"], c["channel_loss_db"],
                              with_target_check=False)
                for vdd in vdds:
                    r = sweep_one(ev, dv, vdd, s10, s9)
                    c["corners"]["%s|%.3f|%.0f" % (proc, vdd, temp)] = r
                    done += 1
                out.write_text(json.dumps(report, indent=1))
            if log_every:
                el = time.time() - t0
                print("  %s %3.0fC done  %4d/%4d corners  %5.0fs elapsed, ~%5.0fs left"
                      % (proc, temp, done, total, el, el / max(done, 1) * (total - done)),
                      flush=True)
    report["complete"] = True
    out.write_text(json.dumps(report, indent=1))
    return report


# --------------------------------------------------------------------------------------
def score(c: dict) -> dict:
    """§4.  Clean, corners passed, worst-corner target error."""
    recs = list(c["corners"].values())
    passed = [r for r in recs if r.get("guard_valid") and r.get("pass10")]
    errs = [abs(r["boost_db"] - c["target_boost_db"]) for r in passed]
    return {"n_pass10": len(passed),
            "n_pass9": sum(1 for r in recs if r.get("guard_valid") and r.get("pass9")),
            "n_guard_valid": sum(1 for r in recs if r.get("guard_valid")),
            "n_halted": sum(1 for r in recs if r.get("halted")),
            "clean": len(passed) == len(recs) == 45,
            "worst_err_db": max(errs) if errs else None}


def flagship(pool: list[dict]) -> tuple[dict | None, str]:
    """§5, in the order it is written there."""
    for c in pool:
        c["score"] = score(c)
    clean = [c for c in pool if c["score"]["clean"]]
    if clean:
        clean.sort(key=lambda c: (round(c["score"]["worst_err_db"] / PREREG["tie_db"]),
                                  c["tt_best_abs_err"], c["spec_index"]))
        return clean[0], "PVT-clean"
    best = sorted(pool, key=lambda c: (-c["score"]["n_pass10"],
                                       c["score"]["worst_err_db"]
                                       if c["score"]["worst_err_db"] is not None else 1e9,
                                       c["spec_index"]))
    return (best[0] if best else None), "best available, NOT sign-off clean"


def report_text(pool: list[dict]) -> None:
    for c in pool:
        c["score"] = score(c)
    print("\nper candidate (of 45 corners)")
    print("  spec  target  chan   TT err   guard_ok  pass9  pass10  worst_err  clean")
    for c in sorted(pool, key=lambda c: c["spec_index"]):
        s = c["score"]
        print("  %4d  %6.2f  %5.2f  %6.3f   %8d  %5d  %6d  %9s  %s"
              % (c["spec_index"], c["target_boost_db"], c["channel_loss_db"],
                 c["tt_best_abs_err"], s["n_guard_valid"], s["n_pass9"], s["n_pass10"],
                 "-" if s["worst_err_db"] is None else "%.3f" % s["worst_err_db"],
                 "YES" if s["clean"] else ""))

    n_clean = sum(1 for c in pool if c["score"]["clean"])
    n_halt = sum(c["score"]["n_halted"] for c in pool)
    print("\nPVT-clean candidates: %d of %d" % (n_clean, len(pool)))
    if n_halt:
        print("Tier 5 halts caught and counted as failures: %d corners" % n_halt)

    fails: dict[str, int] = {}
    for c in pool:
        for r in c["corners"].values():
            if r.get("guard_valid") and not r.get("pass10"):
                for f in r["failing"]:
                    fails[f] = fails.get(f, 0) + 1
            elif not r.get("guard_valid"):
                k = "guard:" + str(r.get("guard_check") or "halted")
                fails[k] = fails.get(k, 0) + 1
    if fails:
        print("\nwhy corners failed (guard-valid check failures, then guard rejections)")
        for k, n in sorted(fails.items(), key=lambda kv: -kv[1]):
            print("  %-46s %4d" % (k[:46], n))

    win, how = flagship(pool)
    print("\nFLAGSHIP (%s)" % how)
    if win is None:
        print("  none -- the pool is empty")
        return
    s = win["score"]
    print("  spec %d  target %.3f dB  channel %.2f dB" % (
        win["spec_index"], win["target_boost_db"], win["channel_loss_db"]))
    print("  TT boost %.3f dB (err %.3f)   corners passed %d/45   worst-corner err %s"
          % (win["tt_best_boost_db"], win["tt_best_abs_err"], s["n_pass10"],
             "-" if s["worst_err_db"] is None else "%.3f dB" % s["worst_err_db"]))
    print("  design: " + "  ".join("%s=%.6g" % (k, v) for k, v in win["design"].items()))
    if how != "PVT-clean":
        bad = [(k, r) for k, r in win["corners"].items()
               if not (r.get("guard_valid") and r.get("pass10"))]
        print("  failing corners (%d):" % len(bad))
        for k, r in bad:
            why = (",".join(r["failing"]) if r.get("guard_valid")
                   else "guard:" + str(r.get("guard_check") or "halted"))
            print("    %-18s %s" % (k, why))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=PREREG["source"])
    p.add_argument("--arm", default=PREREG["arm"])
    p.add_argument("--out", default="results/pvt_signoff_seed23.json")
    p.add_argument("--limit", type=int, default=0,
                   help="smoke only: sweep the first N candidates of the pool")
    p.add_argument("--report-only", action="store_true",
                   help="re-read --out and re-apply the selection rule, no simulation")
    args = p.parse_args()

    out = Path(args.out)
    if args.report_only:
        pool = json.loads(out.read_text())["candidates"]
        report_text(pool)
        return

    pool = load_pool(args.source, args.arm)
    print("PVT SIGN-OFF | docs/PREREG_PVT_SIGNOFF.md")
    for k, v in PREREG.items():
        print("  %-20s %s" % (k, v))
    print("  pool: %d strict-solved arm-%s candidates x 45 corners = %d evaluations"
          % (len(pool), args.arm, len(pool) * 45))
    if args.limit:
        pool = pool[:args.limit]
        print("  SMOKE: first %d candidate(s) only -- NOT the sign-off run" % len(pool))
    if not pool:
        print("\nPOOL IS EMPTY: no strict-solved candidate in arm %s." % args.arm)
        return

    run(pool, out)
    report_text(pool)
    print("\nartifact -> %s" % out)

    win, how = flagship(pool)
    if win is not None and how == "PVT-clean" and not args.limit:
        dv = DesignVars(**win["design"])
        sp = Path("results/pvt_signoff_flagship.spice")
        sp.write_text(netlist(dv, vdd=DEFAULT_SPEC.vdd_nominal, temp_c=27.0,
                              corner="tt", analysis="ac", models="sky130"))
        print("flagship netlist -> %s" % sp)


if __name__ == "__main__":
    main()
