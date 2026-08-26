"""G3.2b: is (rs, r_load) a working repair plane below the band, or only a moving peak?

G3.2a established two things and left one open. Established: rs is a clean boost coordinate
in BOTH regimes (7/8 sub-grid below the band, 3/4 above), and below the band l_in stops
moving the peak (1/8) while r_load moves it on 8/8 with a fixed negative sign. Left open,
and the only question this file exists to answer:

    Can the (rs, r_load) plane move a below-band design INTO peak_in_band while keeping
    enough boost authority at the landing point to then hit the target?

"Moves the peak reliably" is necessary and not sufficient. r_load also costs boost
(-5.39 dB/unit below the band), and a coordinate that walks the peak into the band while
spending the boost authority needed afterwards is not a repair coordinate. That is the
collateral-damage question, and it cannot be answered from slopes alone -- it needs the
move actually attempted and the landing point actually re-probed.

WHY THIS IS A NEW EXPERIMENT AND NOT A G3.2a AMENDMENT. G3.2a's pre-registered rule
returned PASS on l_in for a reason that turned out to be vacuous (its sign was "fixed"
across the one design of eight where l_in moved the peak at all). That rule stands as the
historical measurement it was. It is not being rewritten to fit r_load, and this file's
criterion is not that rule with the axis swapped -- it is a different question, declared
below before the run, on a set drawn at a seed G3.2a never touched.

WHAT IS DELIBERATELY NOT MEASURED. Only rs and r_load are probed. The other four axes were
measured in G3.2a and re-measuring them here would spend budget re-answering a question
that already has an answer.

PRE-REGISTERED CRITERION, written before the run:

  C1 ENTRY       the plane moves a MAJORITY of the fresh low-peak set into peak_in_band
                 within ENTRY_MAX evaluations.
  C2 AUTHORITY   at the landing point, on a majority of the designs that entered:
                   (a) rs's peak movement is under one grid step -- the same sub-grid test
                       the boost axis has had to pass since G3.2; AND
                   (b) the boost gap to the target is inside rs's MEASURED reach, i.e.
                       |boost_landed - target| <= |d_boost_rs| * (rs travel still available
                       in the needed direction).
                 (b) is deliberately a derived quantity and not an invented dB/unit floor.
                 The previous criterion failed by excluding the quantity that mattered;
                 this one asks the question in the units the question is actually about.
  C3 SOLVE       from the landing point, rs ALONE reaches |boost - target| <= 1.5 dB (the
                 project's established strict tolerance) within ADVANCE_MAX evaluations,
                 still in band, on a majority of the designs that entered.

  The plane is a working repair plane iff C1 AND C2 AND C3.

  A PARTIAL RESULT IS REPORTED AS A PARTIAL RESULT. If C1 passes and C3 fails, the finding
  is that r_load enters the band at a cost the boost axis cannot pay back, which is a
  different and more useful answer than "the plane failed".

THIS IS CALIBRATION-SET PERFORMANCE AND IS NOT A BENCHMARK NUMBER. C3's solve count is
measured on the same eight designs the plane was fitted on, and it is reported only as
evidence that the primitive functions. It is NOT comparable to G3.2's 6/10 or H1's 5/10 on
seed-3 specs, and must never be quoted beside them.

Builds no controller. g32_repair.py, the PPO model, H1, the 20 measure_all budget and the
frozen seed-3 results are untouched: nothing here imports from or writes to them. The step
logic below is local to this file for exactly that reason.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

P = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{P/'shim'};{P/'Library'/'bin'};{os.environ['PATH']}"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass
from eqrl.experiments.g32_peak_report import GRID, at_sweep_edge

DIMS = list(ACTION_SPACE.keys())
J_BOOST = DIMS.index("rs")       # G3.2a: the boost coordinate in both regimes
J_PEAK = DIMS.index("r_load")    # G3.2a: the peak coordinate below the band

PREREG = {
    "probe_h": 0.05,        # identical to g32_peak_probe / g32a_lowpeak_probe
    "step_cap": 0.35,       # identical to g32_repair.PREREG
    "entry_max": 3,
    "advance_max": 3,
    "strict_tol_db": 1.5,   # the project's established strict criterion
    "stop_abs_err_db": 0.25,  # identical to g32_repair.PREREG
    "backoff": 0.5,         # halve on a guard rejection, as g32_repair does
    "min_backoff": 0.125,
}

#: A log-gain smaller than this moved the peak by less than one grid step over the probe
#: span, so it is an upper bound rather than a measurement and cannot size a step. Derived
#: from the sweep resolution, not chosen: log(GRID) / (2 * probe_h).
GAIN_FLOOR = math.log(GRID) / (2 * PREREG["probe_h"])

#: Used ONLY when a design's own r_load gain is under the resolution floor: the low-peak
#: median from results/g32a_lowpeak_probe.json. The secant takes over after the first real
#: move, exactly as in G3.2 after its constant-gain defect was fixed.
FALLBACK_LN_GAIN = -1.153


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="results/g32b_lowpeak_set.json")
    p.add_argument("--out", default="results/g32b_plane.json")
    args = p.parse_args()

    sel = json.load(open(args.set))
    channel, target = sel["select_channel_db"], sel["select_target_db"]
    LO, HI = DEFAULT_SPEC.peak_freq_lo_ghz, DEFAULT_SPEC.peak_freq_hi_ghz
    AIM = (LO * HI) ** 0.5
    guard = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False, channel_loss_db=channel)
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                               channel_loss_db=channel)
    h, cap = PREREG["probe_h"], PREREG["step_cap"]
    n_sim = 0

    def evaluate(x):
        nonlocal n_sim
        n_sim += 1
        try:
            v = guard.evaluate(decode_action(np.asarray(x)), vdd=spec.vdd_nominal)
        except Exception as e:
            return {"valid": False, "guard_check": "exception", "reason": repr(e)[:200]}
        if not v.is_valid:
            return {"valid": False, "guard_check": str(getattr(v.check, "value", v.check)),
                    "reason": str(v.reason)[:200]}
        m = v.unwrap()
        ok, checks = hard_pass(m, spec)
        return {"valid": True, "boost_db": float(m.boost_db),
                "peak_freq_ghz": float(m.peak_freq_ghz),
                "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                "in_band": bool(LO <= float(m.peak_freq_ghz) <= HI),
                "failing": [c for c, g in checks.items() if not g]}

    def probe_axis(x0, j, base):
        """Central difference on one axis, with the same two filters as every G3.x probe."""
        pts = {}
        for tag, s in (("plus", +h), ("minus", -h)):
            xt = np.clip(x0 + s * np.eye(len(DIMS))[j], 0.0, 1.0)
            pts[tag] = {"moved": float(abs(xt[j] - x0[j])), **evaluate(xt)}
        usable = {t: q for t, q in pts.items()
                  if q["valid"] and not at_sweep_edge(q.get("peak_freq_ghz"))}
        out = {"points": pts, "d_ln_peak": None, "d_boost": None, "steps": None}
        if len(usable) == 2:
            pl, mi = usable["plus"], usable["minus"]
            span = pl["moved"] + mi["moved"]
            ratio = pl["peak_freq_ghz"] / mi["peak_freq_ghz"]
            out.update(d_ln_peak=math.log(ratio) / span,
                       d_boost=(pl["boost_db"] - mi["boost_db"]) / span,
                       steps=abs(round(math.log(ratio, GRID))))
        elif len(usable) == 1 and base["valid"] and not at_sweep_edge(base["peak_freq_ghz"]):
            tag, q = next(iter(usable.items()))
            sgn = 1.0 if tag == "plus" else -1.0
            span = max(q["moved"], 1e-12)
            ratio = q["peak_freq_ghz"] / base["peak_freq_ghz"]
            out.update(d_ln_peak=sgn * math.log(ratio) / span,
                       d_boost=sgn * (q["boost_db"] - base["boost_db"]) / span,
                       steps=abs(round(math.log(ratio, GRID))))
        return out

    print("G3.2b (rs, r_load) PLANE | %d designs from %s | select-seed %d | target %.1f dB"
          % (len(sel["designs"]), args.set, sel["select_seed"], target), flush=True)
    print("  band %.2f-%.2f GHz, aim %.4f GHz | entry<=%d, advance<=%d evals | step cap %.2f"
          % (LO, HI, AIM, PREREG["entry_max"], PREREG["advance_max"], cap), flush=True)
    print("  CALIBRATION-SET performance. Not comparable to any seed-3 benchmark number.\n",
          flush=True)

    rows: list[dict] = []

    def write(complete):
        Path(args.out).write_text(json.dumps(
            {"prereg": PREREG, "gain_floor": GAIN_FLOOR,
             "fallback_ln_gain": FALLBACK_LN_GAIN, "source_set": args.set,
             "select_seed": sel["select_seed"], "band_ghz": [LO, HI], "aim_ghz": AIM,
             "target_boost_db": target, "channel_loss_db": channel,
             "boost_axis": DIMS[J_BOOST], "peak_axis": DIMS[J_PEAK],
             "n_measure_all": n_sim, "complete": complete, "rows": rows}, indent=1))

    for d in sel["designs"]:
        x0 = np.array(d["x"], dtype=np.float64)
        rec: dict = {"draw": d["draw"], "steps": []}
        base = evaluate(x0)
        rec["base"] = base
        if not base["valid"]:
            rec["outcome"] = "base no longer guard-valid"
            rows.append(rec); write(False)
            print("  draw %3d  BASE NO LONGER GUARD-VALID" % d["draw"], flush=True)
            continue

        # --- probe the two coordinates, and only those two -------------------------------
        rec["probe"] = {DIMS[J_PEAK]: probe_axis(x0, J_PEAK, base),
                        DIMS[J_BOOST]: probe_axis(x0, J_BOOST, base)}
        pr_pk = rec["probe"][DIMS[J_PEAK]]
        gain = pr_pk["d_ln_peak"]
        if gain is None or abs(gain) < GAIN_FLOOR:
            rec["gain_source"] = ("under the %.3f resolution floor -- using the G3.2a "
                                  "low-peak median" % GAIN_FLOOR)
            gain = FALLBACK_LN_GAIN
        else:
            rec["gain_source"] = "this design's own probe"

        # --- C1: enter the band along r_load, secant-corrected after each real move -------
        x, cur, back = x0.copy(), base, 1.0
        used = 0
        while not cur["in_band"] and used < PREREG["entry_max"]:
            t = float(np.clip(math.log(AIM / cur["peak_freq_ghz"]) / gain, -cap, cap)) * back
            if abs(t) < 1e-9:
                break
            xt = x.copy()
            xt[J_PEAK] = float(np.clip(xt[J_PEAK] + t, 0.0, 1.0))
            moved = xt[J_PEAK] - x[J_PEAK]
            nxt = evaluate(xt)
            used += 1
            rec["steps"].append({"phase": "entry", "t": t, "moved": moved, **nxt})
            if not nxt["valid"] or at_sweep_edge(nxt.get("peak_freq_ghz")):
                back *= PREREG["backoff"]
                rec["entry_blocked_by"] = nxt.get("guard_check") or "sweep edge"
                if back < PREREG["min_backoff"]:
                    break
                continue
            back = 1.0
            if abs(moved) > 1e-9:
                g = math.log(nxt["peak_freq_ghz"] / cur["peak_freq_ghz"]) / moved
                if abs(g) >= GAIN_FLOOR:
                    gain = g
                    rec["secant_gain"] = g
            x, cur = xt, nxt
        rec["entry_evals"] = used
        rec["entered"] = bool(cur["in_band"])
        rec["landing"] = cur
        rec["x_landing"] = [float(v) for v in x]

        if not cur["in_band"]:
            rec["outcome"] = "did not enter the band"
            rows.append(rec); write(False)
            print("  draw %3d  peak %6.4f -> %6.4f GHz  NOT IN BAND after %d eval(s)%s"
                  % (d["draw"], base["peak_freq_ghz"], cur["peak_freq_ghz"], used,
                     "  blocked by " + rec["entry_blocked_by"]
                     if rec.get("entry_blocked_by") else ""), flush=True)
            continue

        # --- C2: what boost authority survives AT the landing point ----------------------
        land_probe = probe_axis(x, J_BOOST, cur)
        rec["landing_probe"] = land_probe
        db = land_probe["d_boost"]
        rec["c2a_peak_sub_grid"] = (land_probe["steps"] == 0)
        gap = target - cur["boost_db"]
        if db is None or abs(db) < 1e-9:
            rec["c2b_reach_ok"], rec["reach_db"] = False, 0.0
        else:
            # Travel still available along rs in the direction that closes the gap.
            need_up = (gap > 0) == (db > 0)
            avail = (1.0 - x[J_BOOST]) if need_up else x[J_BOOST]
            rec["reach_db"] = float(abs(db) * avail)
            rec["c2b_reach_ok"] = bool(abs(gap) <= rec["reach_db"])
        rec["boost_gap_db"] = float(gap)

        # --- C3: does rs alone actually close it, from the point we actually landed on ---
        adv = 0
        while abs(target - cur["boost_db"]) > PREREG["stop_abs_err_db"] \
                and adv < PREREG["advance_max"]:
            slope = db if db not in (None, 0.0) else None
            if slope is None:
                break
            t = float(np.clip((target - cur["boost_db"]) / slope, -cap, cap))
            if abs(t) < 1e-9:
                break
            xt = x.copy()
            xt[J_BOOST] = float(np.clip(xt[J_BOOST] + t, 0.0, 1.0))
            moved = xt[J_BOOST] - x[J_BOOST]
            nxt = evaluate(xt)
            adv += 1
            rec["steps"].append({"phase": "advance", "t": t, "moved": moved, **nxt})
            if not nxt["valid"]:
                rec["advance_blocked_by"] = nxt["guard_check"]
                break
            if abs(moved) > 1e-9:
                db = (nxt["boost_db"] - cur["boost_db"]) / moved
            x, cur = xt, nxt
        rec["advance_evals"] = adv
        rec["final"] = cur
        rec["x_final"] = [float(v) for v in x]
        rec["final_err_db"] = float(abs(target - cur["boost_db"]))
        rec["c3_solved"] = bool(cur["in_band"]
                                and rec["final_err_db"] <= PREREG["strict_tol_db"])
        rec["outcome"] = "entered and solved" if rec["c3_solved"] else "entered, not solved"
        rows.append(rec); write(False)
        print("  draw %3d  peak %6.4f -> %6.4f GHz IN BAND (%d ev) | boost %6.2f -> %6.2f "
              "(%d ev, err %5.2f dB) | rs at landing %+7.2f dB/unit, peak %s | %s"
              % (d["draw"], base["peak_freq_ghz"], rec["landing"]["peak_freq_ghz"], used,
                 rec["landing"]["boost_db"], cur["boost_db"], adv, rec["final_err_db"],
                 land_probe["d_boost"] if land_probe["d_boost"] is not None else float("nan"),
                 "sub-grid" if rec["c2a_peak_sub_grid"] else "MOVED",
                 "SOLVED" if rec["c3_solved"] else "not solved"), flush=True)

    write(True)
    print("\n%d measure_all spent (calibration, not a controller arm)" % n_sim, flush=True)
    print("wrote", args.out, flush=True)

    # --- the pre-declared criterion, applied ---------------------------------------------
    done = [r for r in rows if r.get("entered") is not None]
    ent = [r for r in done if r["entered"]]
    n = len(done)
    c1 = len(ent) * 2 > n
    n_a = sum(1 for r in ent if r["c2a_peak_sub_grid"])
    n_b = sum(1 for r in ent if r["c2b_reach_ok"])
    n_s = sum(1 for r in ent if r["c3_solved"])
    c2 = bool(ent) and n_a * 2 > len(ent) and n_b * 2 > len(ent)
    c3 = bool(ent) and n_s * 2 > len(ent)

    print("\nTHE PRE-DECLARED CRITERION:")
    print("  C1 ENTRY      %d of %d entered peak_in_band within %d evals   -> %s"
          % (len(ent), n, PREREG["entry_max"], "PASS" if c1 else "FAIL"))
    print("  C2 AUTHORITY  at landing: rs peak sub-grid on %d of %d; boost gap inside "
          "measured\n                rs reach on %d of %d                              "
          "     -> %s" % (n_a, len(ent), n_b, len(ent), "PASS" if c2 else "FAIL"))
    print("  C3 SOLVE      rs alone reached |err| <= %.1f dB in band on %d of %d       -> %s"
          % (PREREG["strict_tol_db"], n_s, len(ent), "PASS" if c3 else "FAIL"))
    verdict = ("(rs, r_load) IS A WORKING REPAIR PLANE BELOW THE BAND" if (c1 and c2 and c3)
               else "NOT a working repair plane on this criterion")
    print("\nVERDICT: %s" % verdict)
    if c1 and not c3:
        print("  The informative shape: r_load enters the band, and the boost authority "
              "left at the\n  landing point does not pay the target back. That is a "
              "collateral-damage result,\n  not a peak-control failure.")
    print("  Calibration-set performance. Do NOT quote these counts beside G3.2's or H1's "
          "seed-3\n  numbers -- the plane was fitted on these same eight designs.")


if __name__ == "__main__":
    main()
