"""Empirical chance baseline for target tracking: what score does IGNORING the spec get?

WHY THIS IS THE NUMBER THAT MATTERS. A strict target test scores a method on whether the
achieved boost lands within +/-tol of the requested boost. Because the achievable boosts
and the requested targets both live in roughly the same few-dB band, a method that never
reads the spec still scores well above zero. Quoting "10/32" without that reference point
implies retargeting where there may be none -- and when measured, that is exactly what
happened: the 22 Aug policy scored 10/32, against a chance expectation of 9.6.

METHOD. The pool is the set of boosts the system has DEMONSTRABLY achieved on real
simulations (default: results/target_tracking_clean40k.json, the 26 re-simulated designs
from the loose rollout). For each of the 32 held-out targets, the chance of a single
spec-blind draw landing in tolerance is the fraction of the pool within +/-tol. Summing
over targets gives the expected solves for one draw per spec.

This is deliberately EMPIRICAL rather than a parametric fit: it asks what a method that
samples from our own achieved-performance distribution would score, which is the honest
null hypothesis for "does the policy use the target".

The one-draw figure is a LOWER bound on chance. A method with a budget of B evaluations
per spec gets B chances, so its spec-blind expectation is higher still -- reported here as
the 1 - (1-p)^B curve. A policy scoring at or below the B-draw line is not retargeting.

Reads only. Changes nothing.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np


def make_targets_exact(n: int) -> list[float]:
    """The same held-out targets policy_rollout and target_audit use: default_rng(0),
    uniform(5, 11) for the target then uniform(8, 16) for the channel, per spec. The draw
    ORDER matters -- consuming the channel draw keeps the stream aligned, so target i here
    is target i there. Verified against the recorded targets in the rollout artifacts."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        t = float(rng.uniform(5, 11))
        _ = float(rng.uniform(8, 16))     # channel draw, consumed to keep the stream aligned
        out.append(t)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="results/target_tracking_clean40k.json",
                   help="JSON list of records carrying a 'boost' field")
    p.add_argument("--specs", type=int, default=32)
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--budgets", default="1,4,20",
                   help="evaluation budgets to report the spec-blind expectation for")
    p.add_argument("--out", default="results/chance_baseline.json")
    args = p.parse_args()

    recs = json.loads(Path(args.pool).read_text())
    pool = [r["boost"] for r in recs if r.get("boost") is not None]
    if len(pool) < 5:
        raise SystemExit("pool has only %d usable boosts; refusing to report a chance "
                         "rate from that" % len(pool))
    targets = make_targets_exact(args.specs)

    ps = [sum(abs(b - t) <= args.tol for b in pool) / len(pool) for t in targets]
    budgets = [int(x) for x in args.budgets.split(",")]

    print("EMPIRICAL CHANCE BASELINE  (spec-blind draws from our own achieved boosts)")
    print("  pool            : %d achieved boosts from %s" % (len(pool), args.pool))
    print("  pool range      : %.2f - %.2f dB   median %.2f" % (min(pool), max(pool),
                                                                st.median(pool)))
    print("  targets         : %d, range %.2f - %.2f dB" % (len(targets), min(targets),
                                                            max(targets)))
    print("  tolerance       : +/-%.1f dB" % args.tol)
    print()
    print("  per-target hit probability of ONE spec-blind draw:")
    print("    mean %.3f   median %.3f   min %.3f   max %.3f"
          % (st.mean(ps), st.median(ps), min(ps), max(ps)))
    print()
    print("  expected solves out of %d, by evaluation budget:" % len(targets))
    exp = {}
    for b in budgets:
        e = sum(1.0 - (1.0 - q) ** b for q in ps)
        exp[b] = e
        print("    budget %3d : %5.1f / %d   (%.1f%%)" % (b, e, len(targets),
                                                          100 * e / len(targets)))
    print()
    print("  A method scoring at or below its budget's line is not using the spec.")

    Path(args.out).write_text(json.dumps(
        {"pool_source": args.pool, "pool_n": len(pool), "tol_db": args.tol,
         "n_targets": len(targets), "per_target_p": ps,
         "expected_solves_by_budget": {str(k): v for k, v in exp.items()}}, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
