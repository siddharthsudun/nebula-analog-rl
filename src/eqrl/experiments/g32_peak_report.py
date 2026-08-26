"""Read the G3.2 peak probe: is there a usable second coordinate, and is probing needed?

TWO MEASUREMENT FACTS THE RAW ARTIFACT FORCES ON THIS ANALYSIS, both found by reading the
stored points rather than by re-simulating.

1. peak_freq_ghz IS QUANTIZED. It is freq[argmax(mag_db)] on an `ac dec 40` grid, so the
   only peak values that exist are separated by a factor of 10^(1/40) = 1.0593, about 5.9%
   per step. Every peak in the artifact lies exactly on that grid. So "d_peak = 0.000" does
   NOT mean an axis leaves the peak alone; it means the peak moved less than one grid step
   over the probe. This report states such entries as an UPPER BOUND, never as zero. The
   band the controller must satisfy, 1.25 to 2.5 GHz, is 12 grid steps wide, which is coarse
   for measuring a peak and entirely adequate for deciding whether one is inside a 2:1
   band -- which is all Stage A ever asks.

2. SOME PROBE POINTS HAVE NO LOCAL PEAK AT ALL. r_load +h on spec 4 reports 0.001 GHz,
   which is AC_FSTART: the response falls monotonically and argmax lands on the first
   sweep point. Differencing against that gives -26.85 GHz/unit, which would make r_load
   look like far and away the most peak-sensitive axis on the strength of a design that has
   no peak. Points at either sweep edge are excluded from every derivative here and counted
   separately, because a degenerate response is a fact about the design, not a slope.

THE QUESTION. G3.1 failed on 5 of 8 specs for one reason: the single direction it owns
moves boost and peak together, so repairing the peak breaks the boost before the peak comes
back. A second coordinate is only real if some axis moves one much more than the other.

THE NULL, unchanged from the d_boost probe that this project already learned from: a
CONSTANT per-axis table, computed once and shared by every design, against the per-design
probe. If the constant table picks the same coordinates, the controller gets the table for
free and spends no per-spec budget rediscovering it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

#: `ac dec 40` from eqrl.sim.server. One grid step is a factor of 10**(1/40).
DECADE_PTS = 40
GRID = 10.0 ** (1.0 / DECADE_PTS)
AC_FSTART_GHZ = 1e6 / 1e9
AC_FSTOP_GHZ = 1e11 / 1e9


def at_sweep_edge(f: float | None) -> bool:
    """True when argmax landed on the first or last sweep point -- i.e. no interior peak."""
    if f is None:
        return True
    return f <= AC_FSTART_GHZ * GRID or f >= AC_FSTOP_GHZ / GRID


def slopes(a: dict) -> tuple[dict[str, list[dict]], list[str], dict[str, int]]:
    """Per-axis (d_peak, d_boost) estimates with both measurement filters applied.

    Factored out of main() so the controller reads its plane from the SAME code that
    reports it. A second copy of this arithmetic would eventually disagree with the report
    and nobody would notice which of the two was wrong.
    """
    dims_seen: list[str] = []
    per_dim: dict[str, list[dict]] = {}
    dropped = {"invalid": 0, "sweep_edge": 0, "sub_grid": 0}
    for r in a["rows"]:
        if not r.get("dims"):
            continue
        base = r["base"]
        for name, d in r["dims"].items():
            if name not in per_dim:
                per_dim[name] = []
                dims_seen.append(name)
            usable = {}
            for tag in ("plus", "minus"):
                q = d["points"][tag]
                if not q["valid"]:
                    dropped["invalid"] += 1
                elif at_sweep_edge(q.get("peak_freq_ghz")):
                    dropped["sweep_edge"] += 1
                else:
                    usable[tag] = q
            if len(usable) == 2:
                pl, mi = usable["plus"], usable["minus"]
                span = pl["moved"] + mi["moved"]
                dpk = (pl["peak_freq_ghz"] - mi["peak_freq_ghz"]) / span
                dbo = (pl["boost_db"] - mi["boost_db"]) / span
                dlpk = math.log(pl["peak_freq_ghz"] / mi["peak_freq_ghz"]) / span
                steps = abs(round(math.log(pl["peak_freq_ghz"] / mi["peak_freq_ghz"], GRID)))
            elif len(usable) == 1 and not at_sweep_edge(base.get("peak_freq_ghz")):
                tag, q = next(iter(usable.items()))
                sgn = 1.0 if tag == "plus" else -1.0
                span = max(q["moved"], 1e-12)
                dpk = sgn * (q["peak_freq_ghz"] - base["peak_freq_ghz"]) / span
                dbo = sgn * (q["boost_db"] - base["boost_db"]) / span
                dlpk = sgn * math.log(q["peak_freq_ghz"] / base["peak_freq_ghz"]) / span
                steps = abs(round(math.log(q["peak_freq_ghz"] / base["peak_freq_ghz"], GRID)))
            else:
                continue
            if steps == 0:
                dropped["sub_grid"] += 1
            per_dim[name].append({"spec": r["spec"], "d_peak": dpk, "d_boost": dbo,
                                  "d_ln_peak": dlpk, "steps": steps,
                                  "sel": abs(dpk) / max(abs(dbo), 1e-9)})
    return per_dim, dims_seen, dropped


def axis_table(per_dim: dict[str, list[dict]], dims_seen: list[str]) -> dict[str, dict]:
    """Median slope per axis over the specs where it is measurable."""
    table: dict[str, dict] = {}
    for name in dims_seen:
        rec = per_dim[name]
        if not rec:
            continue
        moved = [e for e in rec if e["steps"] > 0]
        signs = {1 if e["d_peak"] > 0 else -1 for e in moved}
        table[name] = {
            "d_peak": statistics.median(e["d_peak"] for e in rec),
            "d_ln_peak": statistics.median(e["d_ln_peak"] for e in rec),
            "d_boost": statistics.median(e["d_boost"] for e in rec),
            "sel": statistics.median(e["sel"] for e in rec),
            "n": len(rec), "n_moved": len(moved),
            "sign": ("never moved" if not moved else
                     ("fixed %s" % ("+" if signs == {1} else "-") if len(signs) == 1
                      else "MIXED"))}
    return table


def plane_from_probe(path: str) -> dict:
    """The two repair coordinates and their measured slopes, with provenance.

    BOOST COORDINATE: the axis that moves boost while disturbing the peak LEAST. This is
    deliberately NOT the six-dimensional composite d_boost that G3 and G3.1 used. That
    composite is exactly what walked the peak out of band on 4 of 4 bracketed specs, and
    this probe says a single axis moves boost harder with peak movement below the
    measurement resolution. Substituting it is the one mechanism change G3.2 makes, and it
    is the change the measurement asked for.

    PEAK COORDINATE: the axis with the most peak movement per unit of boost disturbance.

    The slopes are medians over the specs where they are measurable, used as a CONSTANT
    table, because the probe's own null says a constant table picks the same coordinates
    the per-design probe does -- so per-spec probing would spend budget to learn nothing.
    """
    a = json.load(open(path))
    per_dim, dims_seen, _ = slopes(a)
    t = axis_table(per_dim, dims_seen)
    b = min(t, key=lambda n: (t[n]["sel"], -abs(t[n]["d_boost"])))
    p = max(t, key=lambda n: t[n]["sel"])
    lo, hi = a["band_ghz"]
    return {"boost_axis": b, "peak_axis": p,
            "d_boost_db_per_unit": t[b]["d_boost"],
            "boost_axis_d_peak_ghz_per_unit": t[b]["d_peak"],
            "boost_axis_peak_under_resolution_on": "%d of %d specs"
            % (t[b]["n"] - t[b]["n_moved"], t[b]["n"]),
            "d_peak_ghz_per_unit": t[p]["d_peak"],
            "d_ln_peak_per_unit": t[p]["d_ln_peak"],
            "peak_axis_d_boost_db_per_unit": t[p]["d_boost"],
            "band_ghz": [lo, hi], "peak_aim_ghz": (lo * hi) ** 0.5,
            "grid_step_pct": (GRID - 1) * 100, "source": path,
            "n_specs_measured": t[p]["n"], "spec_seed": a["spec_seed"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default="results/g32_peak_probe.json")
    args = p.parse_args()
    a = json.load(open(args.probe))
    lo, hi = a["band_ghz"]

    print("G3.2 PEAK PROBE REPORT | spec-seed %d | h=%.3f | %d measure_all%s"
          % (a["spec_seed"], a["probe_h"], a["n_measure_all"],
             "" if a["complete"] else "  (INCOMPLETE RUN)"))
    print("  peak resolution: ac dec %d -> one grid step is %.2f%%; band %.2f-%.2f GHz is "
          "%.1f steps wide" % (DECADE_PTS, (GRID - 1) * 100, lo, hi, math.log(hi / lo, GRID)))

    per_dim, dims_seen, dropped = slopes(a)
    print("\n  points dropped: %d guard-invalid, %d at a sweep edge (no interior peak)"
          % (dropped["invalid"], dropped["sweep_edge"]))
    print("  %d surviving slope estimates moved the peak by LESS THAN ONE grid step -- "
          "those are\n  upper bounds of %.2f%%, not zeros"
          % (dropped["sub_grid"], (GRID - 1) * 100))

    print("\n  per-axis, median over the specs where the slope is measurable")
    print("  %-8s %4s  %10s  %10s  %9s   %s"
          % ("axis", "n", "dpeak", "dboost", "|dpk/dbo|", "peak-sign"))
    table = axis_table(per_dim, dims_seen)
    for name in dims_seen:
        if name not in table:
            print("  %-8s %4d  %10s" % (name, 0, "no data"))
            continue
        t = table[name]
        print("  %-8s %4d  %+10.3f  %+10.2f  %9.3f   %s (%d/%d moved)"
              % (name, t["n"], t["d_peak"], t["d_boost"], t["sel"], t["sign"],
                 t["n_moved"], t["n"]))

    # Two coordinates, chosen by opposite criteria: one that moves boost while disturbing
    # the peak least, and one that moves the peak most per unit of boost disturbance.
    boost_axis = min(table, key=lambda n: (table[n]["sel"], -abs(table[n]["d_boost"])))
    peak_axis = max(table, key=lambda n: table[n]["sel"])
    print("\n  boost coordinate (moves boost, least peak disturbance): %s"
          "   %.2f dB/unit, peak %s"
          % (boost_axis, table[boost_axis]["d_boost"],
             "under one grid step on %d of %d specs"
             % (table[boost_axis]["n"] - table[boost_axis]["n_moved"],
                table[boost_axis]["n"])))
    print("  peak  coordinate (most peak per unit boost):             %s"
          "   %+.3f GHz/unit, boost %+.2f dB/unit"
          % (peak_axis, table[peak_axis]["d_peak"], table[peak_axis]["d_boost"]))

    print("\n  THE NULL -- per-design probe against one constant table:")
    agree_b = agree_p = n = 0
    for r in a["rows"]:
        if not r.get("dims"):
            continue
        here = {name: e for name in dims_seen
                for e in per_dim[name] if e["spec"] == r["spec"]}
        if len(here) < 2:
            continue
        n += 1
        b_here = min(here, key=lambda k: (here[k]["sel"], -abs(here[k]["d_boost"])))
        p_here = max(here, key=lambda k: here[k]["sel"])
        agree_b += (b_here == boost_axis)
        agree_p += (p_here == peak_axis)
        print("      spec %2d  probe picks boost=%-7s peak=%-7s   %s"
              % (r["spec"], b_here, p_here,
                 "agrees" if (b_here == boost_axis and p_here == peak_axis)
                 else "DIFFERS from the constant table"))
    if n:
        print("    the constant table matches the per-design probe on %d/%d specs for the "
              "boost axis\n    and %d/%d for the peak axis" % (agree_b, n, agree_p, n))

    # The second question the same run answers: what does the guard actually reject, and
    # which hard check actually refuses. Stage A cannot be designed without both.
    reasons: dict[str, int] = {}
    hp: dict[str, int] = {}
    for r in a["rows"]:
        pool = list(r["stage1"]) + [q for d in (r.get("dims") or {}).values()
                                    for q in d["points"].values()]
        for q in pool:
            if q["valid"]:
                for c in q["failing"]:
                    hp[c] = hp.get(c, 0) + 1
            else:
                reasons[q["guard_check"]] = reasons.get(q["guard_check"], 0) + 1
    print("\n  guard rejections across every point in this run (stage 1 + probes):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("      %-40s %d" % (k, v))
    print("  hard_pass failures among the guard-valid points:")
    for k, v in sorted(hp.items(), key=lambda kv: -kv[1]):
        print("      %-40s %d" % (k, v))


if __name__ == "__main__":
    main()
