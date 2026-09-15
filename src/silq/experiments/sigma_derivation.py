"""Derive stage 2's sigma0 from a pre-declared reach criterion. No SPICE. No new data.

WHY THIS EXISTS. The registered sigma0 = 0.05 was chosen by eye, before anyone had
measured how far boost can actually move inside a local neighbourhood of this design
space. It turns out it cannot move nearly far enough: see `step_reach.py`. Rather than
replace one eyeballed number with another, this derives the value from a rule declared
in advance, using ONLY (a) the recorded SPICE corpus and (b) the frozen PPO artifacts.
Both existed before the hybrid arm did. No seed-2 or seed-3 hybrid result is read here,
and the smoke test is not read here.

THE RULE (docs/REPRODUCE.md section 19):

    sigma* = the smallest sigma on the grid for which
             REACH_p( sigma )  >=  GAP_q
    where
      REACH_p(sigma) = the p-th percentile, over starting points, of the largest
                       |change in boost| that ONE unscreened CMA-ES generation of
                       `popsize` draws from N(x0, sigma^2 I) attains;
      GAP_q          = the q-th percentile of the frozen PPO handoff-gap distribution.

    p = 50 and q = 75 are fixed below and are the only free choices; the sensitivity
    of sigma* to both is printed in full so the choice is visible rather than buried.

WHY p = 50 AND NOT 90. The percentile is taken over STARTING POINTS, and stage 2 does not
get to choose its starting point -- it inherits whatever design PPO happens to hand it.
Sizing the neighbourhood at the 90th percentile of reach would size it for the luckiest
one handoff in ten and leave the median one just as stuck as sigma0 = 0.05 leaves it.
The median is the honest centre of the distribution the arm will actually face.

WHY REACH IS SIMULATED, NOT MEASURED AS A BALL MAXIMUM. `step_reach.py` reports the max
|delta boost| over every corpus point inside a ball, which is an upper bound no sampler
could realise: CMA-ES gets `popsize` draws, not the continuum. Here the generation is
simulated exactly as `hybrid_audit.stage2` draws it -- popsize points from an isotropic
Gaussian, clipped to the unit cube -- and scored with the surrogate. That makes REACH the
quantity the arm can actually attain rather than the one the geometry permits.

WHY THE UNSCREENED ARM SETS THE SIZE. sigma* is derived from H1's sampling, not H2's.
H2's screen chooses `popsize` out of 20x popsize draws, so its reach is strictly greater
at the same sigma. Sizing on H1 guarantees neither arm is crippled by the radius, and
keeps the radius from being tuned to flatter the arm under test.

THE SURROGATE IS RANKING, NOT JUDGING, HERE TOO. It supplies the reach curve. Nothing in
the experiment's outcome is decided by it; every reported design is still verified by the
same guarded fast=False evaluator every other arm uses.

k = 5, r = 10 and the 2k + r <= 20 budget parity are NOT touched. The finding was that the
refinement neighbourhood is the wrong SIZE, not that the arm needs more evaluations.
"""

from __future__ import annotations

# =============================================================================
# VOID ANALYSIS -- DO NOT USE, DO NOT CITE, DO NOT RUN
# =============================================================================
# This script ran against the pre-fix surrogate corpus, whose `boost_db` labels were
# the REAL PART of the complex transfer function rather than 20*log10|H|. See
# docs/REPRODUCE.md section 19. Every number it produced is void, including:
#
#   - the reach curve and every REACH_p(sigma) figure
#   - the finding that no sigma on the grid satisfies the rule
#   - the pooled handoff-gap comparison drawn from it
#
# It is kept, disabled, for provenance: it is the record of how the wrong conclusion
# was reached and of what the mislabelling looked like from the inside. It refuses to
# execute rather than merely warning, because a comment at the top of a runnable file
# is not a safeguard -- someone six months from now would run it and cite the output.
#
# Re-running it against the corrected corpus is NOT a matter of deleting this guard.
# The conclusions would have to be re-derived and re-reviewed from scratch, and any
# such analysis needs its own decision, not a resurrection of this file.
# =============================================================================
raise RuntimeError(
    "VOID ANALYSIS: this script used the pre-fix corpus parser, which read the real "
    "part of H as magnitude in dB. Its outputs are invalid and must not be used. "
    "See docs/REPRODUCE.md section 19.")

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from silq.surrogate import Surrogate, load_corpus

#: PRE-DECLARED. The two free choices in the rule, fixed here before sigma* is read off.
REACH_PCTILE = 50      # over starting points -- see the module docstring
GAP_PCTILE = 75        # of the frozen PPO handoff-gap distribution

#: Stage 1 hands off the design the policy holds after k evaluations. The frozen
#: `target_audit` runs recorded longer trajectories, so the handoff proxy is the k-th
#: valid boost, NOT the best over the whole run -- the best would be a design stage 1
#: never actually hands over, and would understate the gap.
HANDOFF_EVAL = 5

#: cma's default for n = 6. Duplicated rather than imported so this file runs without cma.
POPSIZE = 9

SIGMA_GRID = np.round(np.arange(0.02, 0.51, 0.01), 3)

def handoff_gaps(pattern: str = "results/target_audit*.json") -> dict[str, np.ndarray]:
    """|boost - target| at the k-th evaluation, per frozen PPO artifact.

    Reads the frozen artifacts only. Specs where the policy produced fewer than k valid
    designs contribute their LAST valid boost -- dropping them would silently restrict the
    distribution to the specs the policy found easy, which is the direction that would
    make the derived sigma too small.
    """
    out: dict[str, np.ndarray] = {}
    for f in sorted(glob.glob(pattern)):
        rows = json.load(open(f))
        rows = rows["rows"] if isinstance(rows, dict) and "rows" in rows else rows
        gaps = []
        for r in rows:
            if "ppo" not in r:
                continue
            b = r["ppo"].get("all_valid_boosts") or []
            if not b:
                continue
            gaps.append(abs(b[min(HANDOFF_EVAL, len(b)) - 1] - r["target_boost_db"]))
        if gaps:
            out[Path(f).stem] = np.array(gaps)
    return out

def reach_curve(sur: Surrogate, centers: np.ndarray, sigmas: np.ndarray,
                popsize: int, rng: np.random.Generator) -> dict[float, np.ndarray]:
    """For each sigma, the per-starting-point reach of ONE unscreened generation.

    Drawn exactly as `hybrid_audit.stage2` draws it: x0 + sigma * N(0, I), clipped to the
    unit cube by the [0, 1] bounds. The clip is applied here too -- omitting it would
    credit the sampler with displacement it cannot keep, and near a bound that is most of
    the displacement.
    """
    base, _ = sur.predict_boost(centers)
    n, d = centers.shape
    out = {}
    for s in sigmas:
        draws = np.clip(centers[:, None, :] + s * rng.standard_normal((n, popsize, d)),
                        0.0, 1.0)
        pred, _ = sur.predict_boost(draws.reshape(-1, d))
        out[float(s)] = np.abs(pred.reshape(n, popsize) - base[:, None]).max(axis=1)
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="results/surrogate_corpus.npz")
    p.add_argument("--centers", type=int, default=3000)
    p.add_argument("--popsize", type=int, default=POPSIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/sigma_derivation.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    c = load_corpus(args.corpus)
    # Full metric matrix, not just the boost column: `predict_boost` selects the column
    # itself, and handing it a 1-D Y would silently index the wrong axis.
    X, Y = c["X"], c["Y"]
    sur = Surrogate(X, Y, k=5)
    print(f"corpus {len(X)} records; reach simulated with popsize {args.popsize}, "
          f"{args.centers} starting points\n")

    gaps = handoff_gaps()
    print("FROZEN PPO HANDOFF GAPS  |boost at evaluation %d - target|" % HANDOFF_EVAL)
    for name, g in gaps.items():
        print("  %-42s n=%2d  p50 %5.2f  p75 %5.2f  p90 %5.2f dB"
              % (name, len(g), np.percentile(g, 50), np.percentile(g, 75),
                 np.percentile(g, 90)))
    pooled = np.concatenate(list(gaps.values()))
    print("  %-42s n=%2d  p50 %5.2f  p75 %5.2f  p90 %5.2f dB  <- POOLED, the rule uses this"
          % ("ALL", len(pooled), np.percentile(pooled, 50), np.percentile(pooled, 75),
             np.percentile(pooled, 90)))
    gap_target = float(np.percentile(pooled, GAP_PCTILE))

    centers = X[rng.choice(len(X), args.centers, replace=False)]
    curve = reach_curve(sur, centers, SIGMA_GRID, args.popsize, rng)

    print("\nREACH OF ONE UNSCREENED GENERATION  (dB of boost displacement)")
    print("  sigma   p25    p50    p75    p90     <- percentile over starting points")
    for s in SIGMA_GRID:
        if round(s * 100) % 4 and s != 0.05:
            continue
        r = curve[float(s)]
        print("  %.2f   %5.2f  %5.2f  %5.2f  %5.2f%s"
              % (s, *[np.percentile(r, q) for q in (25, 50, 75, 90)],
                 "   <- AS REGISTERED" if s == 0.05 else ""))

    # -- the rule.
    sigma_star = None
    for s in SIGMA_GRID:
        if np.percentile(curve[float(s)], REACH_PCTILE) >= gap_target:
            sigma_star = float(s)
            break

    print("\n" + "=" * 74)
    print("RULE  smallest sigma with  REACH_p%d(sigma) >= GAP_p%d"
          % (REACH_PCTILE, GAP_PCTILE))
    print("  GAP_p%d  = %.2f dB  (frozen PPO, %d specs across %d artifacts)"
          % (GAP_PCTILE, gap_target, len(pooled), len(gaps)))
    if sigma_star is None:
        print("  NO sigma on the grid up to %.2f reaches it." % SIGMA_GRID[-1])
    else:
        print("  sigma* = %.2f   (REACH_p%d = %.2f dB)"
              % (sigma_star, REACH_PCTILE,
                 np.percentile(curve[sigma_star], REACH_PCTILE)))
        print("  registered sigma0 = 0.05 gives REACH_p%d = %.2f dB -- short by %.2f dB"
              % (REACH_PCTILE, np.percentile(curve[0.05], REACH_PCTILE),
                 gap_target - np.percentile(curve[0.05], REACH_PCTILE)))
    print("=" * 74)

    print("\nSENSITIVITY  sigma* under other choices of the two free percentiles")
    print("        GAP_p50  GAP_p75  GAP_p90")
    sens = {}
    for pr in (25, 50, 75, 90):
        line = []
        for q in (50, 75, 90):
            gt = float(np.percentile(pooled, q))
            hit = next((float(s) for s in SIGMA_GRID
                        if np.percentile(curve[float(s)], pr) >= gt), None)
            sens[f"reach_p{pr}_gap_p{q}"] = hit
            line.append("  >%.2f " % SIGMA_GRID[-1] if hit is None else "   %.2f  " % hit)
        print("REACH_p%-2d %s%s" % (pr, "".join(line),
                                    "   <- THE RULE" if pr == REACH_PCTILE else ""))

    Path(args.out).write_text(json.dumps({
        "rule": {"reach_pctile": REACH_PCTILE, "gap_pctile": GAP_PCTILE,
                 "handoff_eval": HANDOFF_EVAL, "popsize": args.popsize,
                 "centers": args.centers, "seed": args.seed},
        "gap_pctiles": {k: {str(q): float(np.percentile(v, q)) for q in (50, 75, 90)}
                        for k, v in gaps.items()},
        "gap_target_db": gap_target,
        "reach_curve": {str(s): {str(q): float(np.percentile(curve[float(s)], q))
                                 for q in (25, 50, 75, 90)} for s in SIGMA_GRID},
        "sigma_star": sigma_star,
        "sensitivity": sens,
    }, indent=1))
    print("\nwrote", args.out)

if __name__ == "__main__":
    main()
