"""Can sigma0 = 0.05 actually move an H1 handoff? Measured from real SPICE, not a corpus.

WHY THIS EXISTS. The concern that stalled week 1 was that stage 2's sampling radius might
be too local to close the gap between where PPO hands off and where the target is. The
first attempt to answer that read the surrogate corpus, which turned out to be mislabelled
(docs/REPRODUCE.md section 19), and every one of those numbers was withdrawn. This answers
the same question from the only source that cannot be wrong about it: the boosts stage 2
actually measured, through the same guarded fast=False evaluator every arm uses.

THE TWO QUANTITIES, per spec:

    GAP    |handoff boost - target|          how far stage 2 has to move
    REACH  max |stage 2 boost - handoff|     how far it did move, over its r evaluations

REACH is the DISPLACEMENT ACHIEVED, not the displacement toward the target. That is
deliberate: a radius that cannot move boost at all is a geometry problem, while a radius
that moves it the wrong way is a search problem, and conflating them is what sent the
first investigation into the corpus. Both are reported.

WHAT WOULD CONDEMN sigma0 = 0.05. Not "stage 2 failed to solve" -- that can happen for
many reasons. The specific finding that would justify a prospective amendment is
REACH < GAP on most specs: stage 2 physically cannot travel the required distance, so no
amount of better search inside that radius would help.

Reads one hybrid_audit artifact. Runs no simulation and loads no policy.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="results/h1_smoke_reach.json")
    p.add_argument("--arm", default="h1")
    args = p.parse_args()

    d = json.load(open(args.artifact))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d

    print("H1 STAGE-2 REACH vs REQUIRED GAP -- every number below is SPICE-measured\n")
    print("spec  target  handoff | gap    reach  moved? | stage1 err  stage2 err  better?")
    gaps, reaches, improved, movable = [], [], [], []
    for r in rows:
        a = r[args.arm]
        t, h = r["target_boost_db"], a["handoff"]["boost_db"]
        s2 = a["stage2"]["all_valid_boosts"]
        if h is None or not s2:
            print("  %2d   %5.2f   %-6s | %s" % (r["spec"], t, "-" if h is None else
                  "%.2f" % h, "no valid handoff or no valid stage-2 evaluation"))
            continue
        gap = abs(h - t)
        reach = float(np.max(np.abs(np.array(s2) - h)))
        e1, e2 = a["stage1"]["best_abs_err"], a["stage2"]["best_abs_err"]
        gaps.append(gap)
        reaches.append(reach)
        movable.append(reach >= gap)
        # "Better" is judged only where stage 1 had a loose pass to improve on; a stage 2
        # that finds the first loose pass is a different kind of win and is not counted
        # here as a refinement.
        if e1 is not None:
            improved.append(e2 is not None and e2 < e1)
        print("  %2d   %5.2f   %5.2f  | %5.2f  %5.2f  %-6s | %-10s  %-10s  %s"
              % (r["spec"], t, h, gap, reach, "YES" if reach >= gap else "no",
                 "-" if e1 is None else "%.2f" % e1,
                 "-" if e2 is None else "%.2f" % e2,
                 "-" if e1 is None else ("YES" if improved[-1] else "no")))

    if not gaps:
        print("\nno spec produced both a valid handoff and a valid stage-2 evaluation.")
        return
    g, rc = np.array(gaps), np.array(reaches)
    print("\n  median gap   %5.2f dB     median reach  %5.2f dB" % (np.median(g),
                                                                    np.median(rc)))
    print("  reach >= gap on %d of %d specs" % (sum(movable), len(movable)))
    if improved:
        print("  stage 2 improved on stage 1 in %d of %d specs that had one to improve on"
              % (sum(improved), len(improved)))

    print()
    if sum(movable) <= len(movable) // 3:
        print("  READS AS A GEOMETRY PROBLEM. Stage 2 could not travel the required")
        print("  distance on most specs, so the radius -- not the search -- is binding.")
        print("  That is the finding that would justify amending sigma0 prospectively.")
    elif np.median(rc) < np.median(g):
        print("  MIXED. Reach covers the gap on some specs and not others; the radius is")
        print("  marginal rather than plainly wrong.")
    else:
        print("  THE RADIUS IS NOT THE BINDING CONSTRAINT. Stage 2 can travel the")
        print("  required distance; if it still misses, that is a search or objective")
        print("  question and sigma0 is not the thing to change.")
    print("\n  n is small by construction -- this is a smoke test, not the experiment.")


if __name__ == "__main__":
    main()
