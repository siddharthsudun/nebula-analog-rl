"""G3.2a step 4: read the low-peak probe against the pre-declared rule, beside the high one.

Every number here is computed by g32_peak_report.slopes() and axis_table() -- the same code
that produced the high-peak table. This file adds no arithmetic of its own; it only applies
the rule that was written down in g32a_lowpeak_probe's docstring before the probe ran, and
prints the two regimes side by side so a difference is visible rather than argued.

THE RULE, restated here only so it can be read next to its verdict -- it is not being
authored here:

    THE PLANE REMAINS USABLE BELOW THE BAND iff
      Q1  the peak axis's peak sign is FIXED across the low-peak set, AND
      Q2  that sign equals the high-peak calibration's, AND
      Q3  the boost axis's peak movement is under one grid step on a majority of the set.

    A different gain MAGNITUDE does not fail this. The controller has measured its own gain
    by secant since the G3.2 fix, so it needs the coordinate and the sign from the table and
    nothing else. A mixed or flipped SIGN does fail it: no direction can then be chosen, and
    there is nothing for a secant to correct.

    Q4 (the gain magnitude) and Q5 (the constant-table null) are reported and do NOT enter
    the verdict. They are printed so they cannot quietly become the criterion afterwards.

Reads only. Changes no threshold, bound, controller, model, or benchmark criterion.
"""
from __future__ import annotations

import argparse
import json

from silq.experiments.g32_peak_report import (GRID, axis_table, slopes, plane_from_probe)


def regime(path: str) -> dict:
    a = json.load(open(path))
    per_dim, dims_seen, dropped = slopes(a)
    return {"a": a, "per_dim": per_dim, "dims": dims_seen, "dropped": dropped,
            "table": axis_table(per_dim, dims_seen), "plane": plane_from_probe(path)}


def print_table(r: dict, title: str) -> None:
    a, t = r["a"], r["table"]
    print("\n%s | %d base designs | h=%.3f | %d measure_all%s"
          % (title, sum(1 for x in a["rows"] if x.get("dims")), a["probe_h"],
             a["n_measure_all"], "" if a["complete"] else "  (INCOMPLETE RUN)"))
    pk = [x["base"]["peak_freq_ghz"] for x in a["rows"] if x.get("dims")]
    lo, hi = a["band_ghz"]
    print("  base peaks %.3f to %.3f GHz   (band %.2f-%.2f GHz)" % (min(pk), max(pk), lo, hi))
    print("  dropped: %d guard-invalid, %d at a sweep edge; %d surviving slopes moved the "
          "peak\n  less than one %.2f%% grid step -- upper bounds, not zeros"
          % (r["dropped"]["invalid"], r["dropped"]["sweep_edge"], r["dropped"]["sub_grid"],
             (GRID - 1) * 100))
    print("  %-8s %4s  %10s  %10s  %9s  %11s   %s"
          % ("axis", "n", "dpeak", "dboost", "|dpk/dbo|", "d_ln_peak", "peak-sign"))
    for name in r["dims"]:
        if name not in t:
            print("  %-8s %4d  %10s" % (name, 0, "no data"))
            continue
        e = t[name]
        print("  %-8s %4d  %+10.3f  %+10.2f  %9.3f  %+11.3f   %s (%d/%d moved)"
              % (name, e["n"], e["d_peak"], e["d_boost"], e["sel"], e["d_ln_peak"],
                 e["sign"], e["n_moved"], e["n"]))
    p = r["plane"]
    print("  plane chosen here: boost=%s (%.2f dB/unit)   peak=%s (%+.3f ln/unit, "
          "%+.2f dB/unit)"
          % (p["boost_axis"], p["d_boost_db_per_unit"], p["peak_axis"],
             p["d_ln_peak_per_unit"], p["peak_axis_d_boost_db_per_unit"]))


def null_check(r: dict, boost_axis: str, peak_axis: str) -> tuple[int, int, int]:
    """Q5: does one constant table pick the same two coordinates each design's probe does?"""
    per_dim, n, ab, ap = r["per_dim"], 0, 0, 0
    for row in r["a"]["rows"]:
        if not row.get("dims"):
            continue
        here = {name: e for name in r["dims"]
                for e in per_dim[name] if e["spec"] == row["spec"]}
        if len(here) < 2:
            continue
        n += 1
        b = min(here, key=lambda k: (here[k]["sel"], -abs(here[k]["d_boost"])))
        p = max(here, key=lambda k: here[k]["sel"])
        ab += (b == boost_axis)
        ap += (p == peak_axis)
        print("      design %3d  probe picks boost=%-7s peak=%-7s   %s"
              % (row["spec"], b, p,
                 "agrees" if (b == boost_axis and p == peak_axis) else "DIFFERS"))
    return n, ab, ap


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--low", default="results/g32a_lowpeak_probe.json")
    ap_.add_argument("--high", default="results/g32_peak_probe.json")
    args = ap_.parse_args()

    hi_r, lo_r = regime(args.high), regime(args.low)
    print("G3.2a LOW-PEAK CALIBRATION REPORT")
    print("  the two regimes below were measured by the same methodology and are read by "
          "the\n  same code; only where the base designs sit differs")
    print_table(hi_r, "ABOVE THE BAND (G3.2's existing calibration, spec-seed %d)"
                % hi_r["a"]["spec_seed"])
    print_table(lo_r, "BELOW THE BAND (G3.2a's frozen set, select-seed %d)"
                % lo_r["a"]["spec_seed"])

    # The rule asks about G3.2's coordinates specifically. Whether the low-peak regime would
    # have PICKED the same pair on its own is a separate and also interesting fact, printed
    # above and below, but the controller's plane is the high-peak one and that is what has
    # to survive down here.
    hp, lt, ht = hi_r["plane"], lo_r["table"], hi_r["table"]
    b_ax, p_ax = hp["boost_axis"], hp["peak_axis"]
    print("\nTHE PRE-DECLARED RULE, applied to G3.2's coordinates (boost=%s, peak=%s):"
          % (b_ax, p_ax))

    if p_ax not in lt or b_ax not in lt:
        print("  UNDECIDABLE: %s has no measurable slope below the band."
              % (p_ax if p_ax not in lt else b_ax))
        print("\nVERDICT: RULE NOT SATISFIED (the coordinate is not measurable here).")
        return

    q1 = lt[p_ax]["sign"].startswith("fixed")
    hi_sign = ht[p_ax]["sign"]
    q2 = q1 and lt[p_ax]["sign"] == hi_sign
    n_lo = lt[b_ax]["n"]
    n_sub = n_lo - lt[b_ax]["n_moved"]
    q3 = n_sub * 2 > n_lo

    print("  Q1  %s peak sign below the band: %-12s -> %s"
          % (p_ax, lt[p_ax]["sign"], "PASS" if q1 else "FAIL"))
    print("  Q2  same sign as above the band (%s): %-13s -> %s"
          % (hi_sign, "yes" if q2 else "no", "PASS" if q2 else "FAIL"))
    print("  Q3  %s peak under one grid step on a majority: %d of %d -> %s"
          % (b_ax, n_sub, n_lo, "PASS" if q3 else "FAIL"))
    ratio = abs(lt[p_ax]["d_ln_peak"]) / max(abs(ht[p_ax]["d_ln_peak"]), 1e-12)
    print("  Q4  (reported, NOT part of the rule) %s log-gain: %+.3f below vs %+.3f above"
          "  -- ratio %.2fx"
          % (p_ax, lt[p_ax]["d_ln_peak"], ht[p_ax]["d_ln_peak"], ratio))
    print("  Q5  (reported, NOT part of the rule) constant-table null below the band:")
    n, agree_b, agree_p = null_check(lo_r, b_ax, p_ax)
    if n:
        print("      one constant table matches the per-design probe on %d/%d for boost=%s "
              "and %d/%d for peak=%s" % (agree_b, n, b_ax, agree_p, n, p_ax))

    ok = q1 and q2 and q3
    print("\nVERDICT: %s" % ("THE PLANE REMAINS USABLE BELOW THE BAND -- Q1, Q2 and Q3 all "
                             "pass." if ok else
                             "RULE NOT SATISFIED. The plane does not carry below the band."))
    if ok:
        print("  The controller needs the coordinate and the sign from the table and "
              "measures its\n  own gain by secant, so a %.2fx gain difference is inside "
              "what it already handles." % ratio)
    else:
        print("  On the pre-declared reading, there is no single low-dimensional local plane"
              "\n  spanning the design space, and extending G3.2 further is not supported by"
              "\n  this measurement.")
    print("\n  (This is the calibration primitive only. Whether uniform-random low-peak "
          "designs\n  represent the sub-region PPO's low-peak handoffs occupy is a separate "
          "validity\n  question -- g32a_representative.py.)")


if __name__ == "__main__":
    main()
