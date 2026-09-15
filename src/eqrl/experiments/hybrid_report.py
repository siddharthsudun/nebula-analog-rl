"""The H1 result, reported as a design trajectory rather than a scoreboard row.

WHY NOT JUST "PPO = X, H1 = Y". Strategy A's claim is not that one optimizer outscores
another. It is that the two stages do DIFFERENT JOBS: the policy gets into the feasible
region quickly, and a bounded local search then spends a controlled budget finishing the
circuit. A single headline number cannot distinguish that from "the second optimizer was
better", so this reports each spec as a path --

    PPO handoff  ->  CMA-ES refinement  ->  best verified circuit

-- with the boost, the target error, whether hard validity survived, and what it cost at
every step. If the story is real it should be legible spec by spec, not only in an average.

THE PRIMARY QUESTION IS STILL THE CHANCE LINE. Section 8 measured a 6/32 -> 16/32 strict
gain that was pure coverage: more feasible designs produce more accidental target hits. So
the headline test is unchanged and is computed here with the SAME pool, the SAME definition
of k (distinct designs, capped by loose passes) and the SAME permutation as
`final_report.py`, so the numbers are directly comparable to every arm already reported.

WHAT THIS CANNOT ANSWER. `H1 vs frozen PPO at equal budget` needs a `target_audit` run on
the same spec seed, and none exists for seeds 2 or 3. What IS available, and is arguably
the better test of the Strategy A hypothesis, is the PAIRED within-arm comparison: stage 1
IS a PPO rollout, so stage-1-alone against the full arm asks whether refinement improved
the policy's own handoff, on the same spec, from the same starting point.

Reads hybrid_audit artifacts. Runs no simulation, loads no policy, changes no threshold.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

POOL = "results/target_tracking_clean40k.json"


def terciles(targets: list[float]) -> list[int]:
    """Index 0/1/2 per spec by target boost, cut at the tercile boundaries of THIS set."""
    lo, hi = np.quantile(targets, [1 / 3, 2 / 3])
    return [0 if t <= lo else (1 if t <= hi else 2) for t in targets]


def chance_line(targets, arms_k, observed, tol, rng, n_sim=2000):
    """The per-method matched chance line, identical in definition to final_report.py."""
    pool = [r["boost"] for r in json.loads(Path(POOL).read_text())
            if r.get("boost") is not None]
    ps = [sum(abs(b - t) <= tol for b in pool) / len(pool) for t in targets]
    exp = sum(1 - (1 - q) ** k for q, k in zip(ps, arms_k))
    sims = [sum(rng.random() < (1 - (1 - q) ** k) for q, k in zip(ps, arms_k))
            for _ in range(n_sim)]
    return exp, float(np.mean([v >= observed for v in sims])), len(pool)


def report(path: str, arm: str, tol: float, rng) -> dict | None:
    d = json.loads(Path(path).read_text())
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    if not rows:
        print(f"{path}: no specs recorded yet.")
        return None
    seed = d.get("spec_seed", "?") if isinstance(d, dict) else "?"

    print("=" * 100)
    print("H1 TRAJECTORY REPORT   %s   spec-seed %s   %d specs" % (path, seed, len(rows)))
    print("=" * 100)
    # The handoff column reports PASS/valid/NO rather than a bare yes/no because the three
    # states mean different things and get confused constantly: PASS is a design that
    # already satisfies the hard spec, `valid` is one that simulates cleanly but fails a
    # spec check, and NO is guard-rejected. Only PASS designs are eligible to be anyone's
    # "best", so without this distinction a handoff at 4.84 dB that never passed reads as
    # a refinement that went backwards.
    print("\n            |-------- PPO handoff --------|--- after refinement ---|")
    print("spec  target | boost   err   handoff-state | boost   err   solved  | "
          "measure_all")
    tgt = []
    for r in rows:
        a, t = r[arm], r["target_boost_db"]
        tgt.append(t)
        hb = a["handoff"]["boost_db"]
        fb, fe = a["best_boost_db"], a["best_abs_err"]
        print("  %2d   %5.2f | %-6s %-5s %-13s | %-6s %-5s %-7s | %.0f"
              % (r["spec"], t,
                 "-" if hb is None else "%.2f" % hb,
                 "-" if hb is None else "%.2f" % abs(hb - t),
                 "PASS" if a["handoff"].get("loose_pass")
                 else ("valid" if a["handoff"]["guard_valid"] else "NO"),
                 "-" if fb is None else "%.2f" % fb,
                 "-" if fe is None else "%.2f" % fe,
                 "strict" if a["strict_solved_at"] else
                 ("loose" if a["loose_solved_at"] else "-"),
                 a["measure_all_spent"]))

    n = len(rows)
    strict = sum(1 for r in rows if r[arm]["strict_solved_at"])
    loose = sum(1 for r in rows if r[arm]["loose_solved_at"])
    s1_strict = sum(1 for r in rows if r[arm]["stage1"]["strict_solved_at"])
    s1_loose = sum(1 for r in rows if r[arm]["stage1"]["loose_solved_at"])
    ks = [min(len({round(b, 4) for b in r[arm]["all_valid_boosts"]}),
              r[arm]["n_loose_pass"]) for r in rows]

    print("\n--- PRIMARY: does it beat its own matched chance line? " + "-" * 44)
    exp, pv, npool = chance_line(tgt, ks, strict, tol, rng)
    print("  strict %d/%d   mean distinct k %.2f   chance line %.1f/%d   "
          "p(>=observed) %.4f%s"
          % (strict, n, st.mean(ks), exp, n, pv,
             "   <- ABOVE CHANCE" if pv < 0.05 else "   <- ON ITS LINE"))
    print("  (pool of %d demonstrably-achieved boosts, same as every other arm)" % npool)

    print("\n--- SECONDARY " + "-" * 85)
    print("  hard-spec compliance   stage 1 alone %2d/%d loose  ->  full arm %2d/%d loose"
          % (s1_loose, n, loose, n))
    print("  strict target matching stage 1 alone %2d/%d       ->  full arm %2d/%d"
          % (s1_strict, n, strict, n))

    # Paired: did refinement improve the policy's own handoff, spec by spec?
    both = [(r[arm]["stage1"]["best_abs_err"], r[arm]["best_abs_err"]) for r in rows
            if r[arm]["stage1"]["best_abs_err"] is not None]
    if both:
        better = sum(1 for a, b in both if b is not None and b < a - 1e-9)
        same = sum(1 for a, b in both if b is not None and abs(b - a) <= 1e-9)
        print("  target error, paired    improved on %d of %d specs where stage 1 had a "
              "loose pass (%d unchanged)" % (better, len(both), same))
        print("                          median err  stage 1 %.2f dB  ->  full arm %.2f dB"
              % (st.median([a for a, _ in both]),
                 st.median([b if b is not None else a for a, b in both])))

    # Recovery: the handoff was invalid or not loose-passing, and the arm solved anyway.
    no_pass = sum(1 for r in rows if not r[arm]["handoff"].get("loose_pass"))
    print("  handoff quality         %d of %d handoffs did not themselves "
          "pass the hard spec" % (no_pass, n))
    inval = [r for r in rows if not r[arm]["handoff"]["guard_valid"]]
    if inval:
        rec_l = sum(1 for r in inval if r[arm]["stage2"]["loose_solved_at"])
        rec_s = sum(1 for r in inval if r[arm]["stage2"]["strict_solved_at"])
        print("  recovery                %d handoffs were guard-INVALID; stage 2 still "
              "reached %d loose, %d strict" % (len(inval), rec_l, rec_s))

    spent = sum(r[arm]["measure_all_spent"] for r in rows)
    print("  cost                    %.0f measure_all total, %.1f per spec, "
          "%.1f per strict solve"
          % (spent, spent / n, spent / strict if strict else float("nan")))

    print("\n--- BY TARGET TERCILE " + "-" * 77)
    tc = terciles(tgt)
    print("  tercile  n   target range      strict   loose   median final err")
    for k in (0, 1, 2):
        idx = [i for i, c in enumerate(tc) if c == k]
        if not idx:
            continue
        errs = [rows[i][arm]["best_abs_err"] for i in idx
                if rows[i][arm]["best_abs_err"] is not None]
        print("     %d    %2d   %5.2f - %5.2f dB   %2d/%-2d    %2d/%-2d   %s"
              % (k + 1, len(idx), min(tgt[i] for i in idx), max(tgt[i] for i in idx),
                 sum(1 for i in idx if rows[i][arm]["strict_solved_at"]), len(idx),
                 sum(1 for i in idx if rows[i][arm]["loose_solved_at"]), len(idx),
                 "-" if not errs else "%.2f dB" % st.median(errs)))

    return {"spec_seed": seed, "n": n, "strict": strict, "loose": loose,
            "stage1_strict": s1_strict, "stage1_loose": s1_loose,
            "chance_expected": exp, "chance_p": pv, "mean_distinct_k": st.mean(ks),
            "measure_all_total": spent}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", nargs="+",
                   default=["results/hybrid_audit_h1_seed2.json",
                            "results/hybrid_audit_h1_seed3.json"])
    p.add_argument("--arm", default="h1")
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--out", default="results/hybrid_report.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = []
    for a in args.artifacts:
        if not Path(a).exists():
            print(f"{a}: not present yet, skipped.\n")
            continue
        r = report(a, args.arm, args.tol, rng)
        if r:
            out.append(r)
        print()

    if len(out) > 1:
        print("=" * 100)
        print("REPLICATION -- the criterion is that the finding holds on BOTH spec sets")
        print("=" * 100)
        for r in out:
            print("  seed %-3s strict %2d/%-2d  chance %.1f  p %.4f   %s"
                  % (r["spec_seed"], r["strict"], r["n"], r["chance_expected"],
                     r["chance_p"], "above" if r["chance_p"] < 0.05 else "ON ITS LINE"))
        if all(r["chance_p"] < 0.05 for r in out):
            print("\n  H1 clears its matched chance line on every spec set tested.")
        elif any(r["chance_p"] < 0.05 for r in out):
            print("\n  SPLIT RESULT. It clears on one spec set and not the other, which is")
            print("  the pattern section 8 already saw once. It does not replicate.")
        else:
            print("\n  H1 sits ON its matched chance line. Whatever strict solves it has")
            print("  are explained by how many distinct feasible designs it found, not by")
            print("  aiming them at the target.")

    if out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
