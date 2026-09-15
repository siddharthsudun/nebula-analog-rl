"""What the peak axis actually spends: DC gain, not boost. Re-read of committed artifacts.

G3.2b's criterion asked whether r_load enters the band while preserving BOOST authority.
It failed, and the traces say it failed for a reason the criterion did not name: on 5 of the
6 designs that never entered, the blocker was the guard check T4.10_dc_gain_implausible.
Every entry step lowers r_load, and DC gain falls monotonically with it in every trace.
Boost was never the currency. DC gain was.

So this asks the question the traces raise, in the units they raise it in: for each axis,
how much peak movement does it buy per dB of DC gain, and per dB of boost, spent? An axis
that raises the peak cheaply in DC terms is a candidate the (boost, peak) factorization
cannot see, because that factorization has no coordinate for the constraint that actually
walls the repair.

NO NEW SIMULATIONS. Every number below comes from results/g32a_lowpeak_probe.json and
results/g32_peak_probe.json, both already committed, both already carrying dc_gain_db at
every probe point. The slopes come from g32_peak_report.slopes() -- the same reader, same
sweep-edge filter, same one-sided fallback -- so these columns are directly comparable to
the tables already reported. Computing them with a private copy of the difference logic
would silently drop the one-sided cases and undercount, which is exactly what a first pass
at this file did.

Reports. Decides nothing, builds nothing, changes no controller, criterion or bound.
"""
from __future__ import annotations

import argparse
import json
import statistics

from eqrl.experiments.g32_peak_report import slopes

AXES = ["w_in", "l_in", "i_tail", "rs", "cs", "r_load"]


def table(path: str, drop: tuple = ()) -> dict[str, dict]:
    a = json.load(open(path))
    per_dim, dims, _ = slopes(a)
    out = {}
    for ax in AXES:
        rec = [e for e in per_dim.get(ax, []) if e["spec"] not in drop]
        if not rec:
            continue
        moved = [e for e in rec if e["steps"] > 0]
        signs = {1 if e["d_peak"] > 0 else -1 for e in moved}
        # Peak movement bought per dB spent, in log-frequency units so the number is
        # comparable across designs sitting at different absolute frequencies. Medians of
        # the per-design ratios, not a ratio of medians: one design with a near-zero
        # denominator would otherwise set the whole column.
        per_dc = [abs(e["d_ln_peak"]) / abs(e["d_dc"]) for e in moved if abs(e["d_dc"]) > 1e-9]
        per_bo = [abs(e["d_ln_peak"]) / abs(e["d_boost"]) for e in moved
                  if abs(e["d_boost"]) > 1e-9]
        out[ax] = {
            "n": len(rec), "n_moved": len(moved),
            "d_ln_peak": statistics.median(e["d_ln_peak"] for e in rec),
            "d_boost": statistics.median(e["d_boost"] for e in rec),
            "d_dc": statistics.median(e["d_dc"] for e in rec),
            "per_dc": statistics.median(per_dc) if per_dc else None,
            "per_boost": statistics.median(per_bo) if per_bo else None,
            "n_dc_free": sum(1 for e in moved if abs(e["d_dc"]) < 1e-9),
            "sign": ("never moved" if not moved else
                     ("fixed %s" % ("+" if signs == {1} else "-") if len(signs) == 1
                      else "MIXED"))}
    return out


def show(t: dict, title: str) -> None:
    print("\n%s" % title)
    print("  %-8s %4s %6s  %10s %10s %9s  %9s %9s   %s"
          % ("axis", "n", "moved", "d_ln_peak", "d_boost", "d_dc_dB",
             "ln/dB_dc", "ln/dB_bo", "peak-sign"))
    for ax in AXES:
        if ax not in t:
            print("  %-8s %4s" % (ax, "  --"))
            continue
        e = t[ax]
        f = lambda v: "%9.3f" % v if v is not None else "        -"
        print("  %-8s %4d %6d  %+10.3f %+10.2f %+9.2f  %s %s   %s%s"
              % (ax, e["n"], e["n_moved"], e["d_ln_peak"], e["d_boost"], e["d_dc"],
                 f(e["per_dc"]), f(e["per_boost"]), e["sign"],
                 "   (%d of %d moves cost NO measurable dc gain)"
                 % (e["n_dc_free"], e["n_moved"]) if e["n_dc_free"] else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--low", default="results/g32a_lowpeak_probe.json")
    p.add_argument("--high", default="results/g32_peak_probe.json")
    p.add_argument("--drop", type=int, nargs="*", default=[82],
                   help="designs to ALSO report without; the full set is always shown first")
    args = p.parse_args()

    print("PEAK MOVEMENT PER dB SPENT | no new simulations; committed probe artifacts only")
    print("  ln/dB_dc and ln/dB_bo are median per-design |d_ln_peak| / |d_(that quantity)|,")
    print("  over the designs where the peak moved by at least one grid step. Higher is")
    print("  cheaper. They are ratios of measured slopes, not a new experiment.")

    show(table(args.low), "BELOW THE BAND (G3.2a frozen set, select-seed 11, all 8)")
    if args.drop:
        show(table(args.low, drop=tuple(args.drop)),
             "BELOW THE BAND, minus design(s) %s -- the near-peakless draw flagged at "
             "selection" % ", ".join(str(d) for d in args.drop))
    show(table(args.high), "ABOVE THE BAND (G3.2 calibration, spec-seed 2)")

    print("\n  Read against G3.2b: r_load is the peak coordinate the (boost, peak) "
          "factorization\n  picks below the band, and it is the most DC-expensive axis "
          "there. That is why 5 of\n  6 non-entering designs died on "
          "T4.10_dc_gain_implausible rather than on boost.")


if __name__ == "__main__":
    main()
