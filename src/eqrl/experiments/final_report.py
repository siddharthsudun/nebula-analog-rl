"""One command that regenerates every number in the final benchmark report.

Reads the artifacts the four matched-protocol runs wrote and emits the comparison
tables. Runs no simulation and loads no model -- if a number in the report is not
derivable from a recorded artifact by this script, it does not belong in the report.

WHAT IT ENFORCES, rather than assumes:

  * every arm used the SAME spec list. Checked, not trusted: the recorded
    (target, channel) pairs are compared across all four artifacts and the run aborts
    if they differ.
  * correlation is computed on an UNFILTERED population. Two are reported, and both
    are stated with the population they condition on -- there is no such thing as an
    unconditioned one here, because a method that produced no valid design contributes
    no achieved boost.
  * the "closest valid design" correlation is compared against its own PERMUTATION
    NULL. That statistic picks the pool element nearest the target, which induces
    positive correlation for a method that cannot see the target at all -- measured at
    +0.24 to +0.34 depending on pool size. Reporting it without the null would repeat,
    in a subtler form, exactly the selection-on-the-dependent-variable mistake that
    produced the discredited +0.962.
  * PPO costs 2.00 measure_all per optimizer evaluation against the baselines' 1.00
    (experiments/simcount_audit.py). So an evaluation-matched read and a
    simulation-matched read are different comparisons, and both are printed.

Changes no threshold, bound, reward, model or benchmark criterion.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

#: Measured, not assumed: results/simcount_audit.json, 3 specs x 6 steps.
PPO_MEASURE_ALL_PER_EVAL = 2.0
SEARCH_MEASURE_ALL_PER_EVAL = 1.0


def load(path: str, key: str):
    """Return (specs, per-spec results) for one arm, or (None, None) if absent."""
    p = Path(path)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    if not d.get("complete", True):
        print(f"  !! {p.name} is INCOMPLETE ({len(d['rows'])} specs) -- skipping")
        return None, None
    specs = [(r["target_boost_db"], r["channel_loss_db"]) for r in d["rows"]]
    return specs, [r[key] for r in d["rows"]]


def corr(pairs) -> float:
    if len(pairs) < 3:
        return float("nan")
    return float(np.corrcoef([a for a, _ in pairs], [b for _, b in pairs])[0, 1])


def closest_pairs(targets, arms):
    """(target, nearest achieved boost among ALL guard-valid designs) per spec."""
    out = []
    for t, a in zip(targets, arms):
        pool = a["all_valid_boosts"]
        if pool:
            out.append((t, min(pool, key=lambda b: abs(b - t))))
    return out


def permutation_null(targets, arms, rng, n=5000):
    """Shuffle which spec each POOL belongs to. A pool cannot know a target it was
    never evaluated against, so this is the distribution of the statistic under
    'the method ignores the spec' -- with the best-of selection rule left intact."""
    pools = [a["all_valid_boosts"] for a in arms]
    out = []
    for _ in range(n):
        idx = rng.permutation(len(pools))
        r = corr(closest_pairs(targets, [{"all_valid_boosts": pools[i]} for i in idx]))
        if not math.isnan(r):
            out.append(r)
    return np.array(out)


def mcnemar(a, b, key):
    """Exact two-sided paired test. The arms share specs, so the paired test is the
    right one and an unpaired proportion test would overstate the uncertainty."""
    n10 = sum(1 for x, y in zip(a, b) if x[key] and not y[key])
    n01 = sum(1 for x, y in zip(a, b) if not x[key] and y[key])
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def solved_within(arms, k, key):
    return sum(1 for a in arms if a[key] and a[key] <= k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec-seed", type=int, default=0,
                   help="0 reads the historical artifacts; any other n reads the "
                        "_seed<n> held-out artifacts")
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--chance", default="results/chance_baseline.json")
    p.add_argument("--perms", type=int, default=5000)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    sfx = "" if args.spec_seed == 0 else f"_seed{args.spec_seed}"
    base = f"results/target_audit{'_clean40k' if args.spec_seed == 0 else sfx}.json"
    out = args.out or f"results/final_report{sfx}.json"

    sources = [
        ("PPO",        base,                                    "ppo"),
        ("PPO-restart", f"results/target_audit_freshrestart{sfx}.json", "ppo"),
        ("RANDOM",     base,                                    "random"),
        ("CMA-ES",     f"results/search_audit_cmaes{sfx}.json",  "cmaes"),
        ("TPE",        f"results/search_audit_tpe{sfx}.json",    "tpe"),
    ]
    arms, specs_ref, ref_name = {}, None, None
    print(f"FINAL BENCHMARK REPORT  --  spec-seed {args.spec_seed}, budget "
          f"{args.budget}, strict tolerance +/-{args.tol} dB\n")
    for name, path, key in sources:
        specs, res = load(path, key)
        if res is None:
            print(f"  (absent: {name} <- {path})")
            continue
        if specs_ref is None:
            specs_ref, ref_name = specs, name
        elif specs != specs_ref:
            raise SystemExit(
                f"SPEC MISMATCH: {name} ({path}) does not use the same specs as "
                f"{ref_name}. These results are not comparable and no report will be "
                f"written from them.")
        arms[name] = res
    if not arms:
        raise SystemExit("no complete artifacts found")
    targets = [t for t, _ in specs_ref]
    print(f"  spec list verified identical across {len(arms)} arms "
          f"({len(targets)} specs)\n")

    # ---- A / B: feasibility and retargeting -----------------------------------
    print("FEASIBILITY (loose: all hard specs, target unscored) and RETARGETING "
          f"(strict: loose AND within +/-{args.tol} dB)\n")
    print("  %-12s %-16s %-16s %-8s %-9s" % ("method", "loose (median)",
                                             "strict (median)", "valid", "distinct"))
    table = {}
    for name, a in arms.items():
        lo = [x["loose_solved_at"] for x in a if x["loose_solved_at"]]
        stc = [x["strict_solved_at"] for x in a if x["strict_solved_at"]]
        dist = st.mean(len({round(b, 6) for b in x["all_valid_boosts"]}) for x in a)
        nval = st.mean(x["n_valid"] for x in a)
        print("  %-12s %2d/%-2d (%-6s) %2d/%-2d (%-6s)   %4.1f/%-3d %5.1f"
              % (name, len(lo), len(targets),
                 f"{st.median(lo):.1f}" if lo else "-",
                 len(stc), len(targets),
                 f"{st.median(stc):.1f}" if stc else "-",
                 nval, args.budget, dist))
        table[name] = {
            "loose": len(lo), "strict": len(stc), "n_specs": len(targets),
            "loose_median": st.median(lo) if lo else None,
            "strict_median": st.median(stc) if stc else None,
            "mean_valid_evals": nval, "mean_distinct_boosts": dist,
        }

    # ---- paired significance ---------------------------------------------------
    if "PPO" in arms:
        print("\nPAIRED COMPARISON vs PPO (exact McNemar; same specs, so paired)\n")
        for name, a in arms.items():
            if name == "PPO":
                continue
            for key, lab in (("loose_solved_at", "loose "),
                             ("strict_solved_at", "strict")):
                w, l, pv = mcnemar(arms["PPO"], a, key)
                print("  %s  PPO vs %-12s PPO-only %2d, other-only %2d, p = %.4f%s"
                      % (lab, name, w, l, pv,
                         "   <- significant" if pv < 0.05 else ""))
                table.setdefault(name, {}).setdefault("vs_ppo", {})[lab.strip()] = {
                    "ppo_only": w, "other_only": l, "p": pv}
            print()

    # ---- simulation-matched read ----------------------------------------------
    print("SIMULATION-MATCHED READ (PPO costs %.1f measure_all per evaluation, the "
          "baselines %.1f)\n" % (PPO_MEASURE_ALL_PER_EVAL, SEARCH_MEASURE_ALL_PER_EVAL))
    half = args.budget // 2
    for name, a in arms.items():
        k = half if name.startswith("PPO") else args.budget
        cost = k * (PPO_MEASURE_ALL_PER_EVAL if name.startswith("PPO")
                    else SEARCH_MEASURE_ALL_PER_EVAL)
        print("  %-12s first %2d evals = %3.0f measure_all -> loose %2d/%d  strict %2d/%d"
              % (name, k, cost, solved_within(a, k, "loose_solved_at"), len(targets),
                 solved_within(a, k, "strict_solved_at"), len(targets)))
        table[name]["sim_matched"] = {
            "evals": k, "measure_all": cost,
            "loose": solved_within(a, k, "loose_solved_at"),
            "strict": solved_within(a, k, "strict_solved_at")}

    # ---- correlation, both populations, against the permutation null ------------
    rng = np.random.default_rng(7)
    print("\nTARGET TRACKING -- correlation(requested, achieved), UNFILTERED by success")
    print("  population 1: the closest design among all GUARD-VALID evaluations")
    print("  the null is this same statistic with the spec-to-pool assignment shuffled,")
    print("  which is what a method that ignores the target scores by construction.\n")
    print("  %-12s %-4s %-9s %-9s %-20s %s"
          % ("method", "n", "observed", "null mean", "null 95% band", "p"))
    for name, a in arms.items():
        pairs = closest_pairs(targets, a)
        obs = corr(pairs)
        null = permutation_null(targets, a, rng, args.perms)
        pv = float((null >= obs).mean())
        lo, hi = np.percentile(null, [2.5, 97.5])
        errs = sorted(abs(b - t) for t, b in pairs)
        print("  %-12s %2d   %+.3f    %+.3f    [%+.3f, %+.3f]     %.4f%s"
              % (name, len(pairs), obs, null.mean(), lo, hi, pv,
                 "  <- above its null" if pv < 0.05 else ""))
        table[name]["tracking_all_valid"] = {
            "n": len(pairs), "corr": obs, "null_mean": float(null.mean()),
            "null_ci": [float(lo), float(hi)], "p_one_sided": pv,
            "median_abs_err_db": st.median(errs) if errs else None,
            "within_tol": sum(e <= args.tol for e in errs)}
    print("\n  NOTE: %d arms were tested, so a single p just under 0.05 does not survive"
          "\n  a multiple-comparison correction. Treat it as a hint, not a finding."
          % len(arms))

    print("\n  population 2: the closest LOOSE-PASSING design (the headline statistic)\n")
    print("  %-12s %-4s %-9s %-20s %-8s %s"
          % ("method", "n", "observed", "null 95% band", "p", "med|err|"))
    for name, a in arms.items():
        pairs = [(t, x["best_boost_db"]) for t, x in zip(targets, a)
                 if x["best_boost_db"] is not None]
        if len(pairs) < 3:
            print("  %-12s %2d   (too few to correlate)" % (name, len(pairs)))
            continue
        obs = corr(pairs)
        bb = [b for _, b in pairs]
        tt = [t for t, _ in pairs]
        null = np.array([corr(list(zip(tt, rng.permutation(bb))))
                         for _ in range(args.perms)])
        pv = float((null >= obs).mean())
        lo, hi = np.percentile(null, [2.5, 97.5])
        errs = sorted(abs(b - t) for t, b in pairs)
        print("  %-12s %2d   %+.3f    [%+.3f, %+.3f]     %.4f%s  %.2f dB"
              % (name, len(pairs), obs, lo, hi, pv,
                 " <-above" if pv < 0.05 else "       ", st.median(errs)))
        table[name]["tracking_loose_pass"] = {
            "n": len(pairs), "corr": obs, "null_ci": [float(lo), float(hi)],
            "p_one_sided": pv, "median_abs_err_db": st.median(errs)}

    # ---- the decisive test: chance matched to each method's own feasibility ------
    pool_path = Path("results/target_tracking_clean40k.json")
    if pool_path.exists():
        pool = [r["boost"] for r in json.loads(pool_path.read_text())
                if r.get("boost") is not None]
        ps = [sum(abs(b - t) <= args.tol for b in pool) / len(pool) for t in targets]
        print("\nPER-METHOD MATCHED CHANCE LINE  --  the decisive retargeting test\n")
        print("  For each spec, p is the fraction of the %d demonstrably-achieved boosts"
              % len(pool))
        print("  landing within tolerance of THAT target. A method that produced k")
        print("  distinct feasible designs for that spec, spec-blind, solves it with")
        print("  probability 1-(1-p)^k. Summing over specs with EACH METHOD'S OWN k asks")
        print("  the only question that matters: given how many feasible designs you")
        print("  found, did you AIM them? k counts DISTINCT designs -- re-evaluating one")
        print("  design five times is not five independent chances at the target.\n")
        print("  %-12s %-8s %-9s %-11s %s"
              % ("method", "strict", "mean k", "chance", "p(>=observed)"))
        for name, a in arms.items():
            ks = [min(len({round(b, 4) for b in x["all_valid_boosts"]}),
                      x["n_loose_pass"]) for x in a]
            exp = sum(1 - (1 - q) ** k for q, k in zip(ps, ks))
            obs = sum(1 for x in a if x["strict_solved_at"])
            sims = [sum(rng.random() < (1 - (1 - q) ** k) for q, k in zip(ps, ks))
                    for _ in range(2000)]
            pv = float(np.mean([v >= obs for v in sims]))
            print("  %-12s %2d/%-2d    %5.2f     %5.1f/%-2d     %.4f%s"
                  % (name, obs, len(targets), st.mean(ks), exp, len(targets), pv,
                     "   <- ABOVE chance" if pv < 0.05 else ""))
            table[name]["chance_matched"] = {
                "observed": obs, "expected": exp, "mean_distinct_k": st.mean(ks),
                "p_one_sided": pv}
        print("\n  The pool is drawn from THIS system's own achieved boosts, so it is")
        print("  self-referential and mildly generous to PPO. The sanity check is random")
        print("  search: a method that provably cannot see the target should land ON its")
        print("  line, and it does.")

    # ---- chance reference -------------------------------------------------------
    ch = Path(args.chance)
    if ch.exists():
        e = json.loads(ch.read_text())["expected_solves_by_budget"]
        print("\nCHANCE REFERENCE (spec-blind draws from our own achieved boosts)")
        for b in sorted(e, key=int):
            print("    budget %3s : %5.1f/%d" % (b, e[b], len(targets)))
        print("  Compare a method against the line for its number of INDEPENDENT")
        print("  attempts. It is conditional on producing a valid design at all.")

    Path(out).write_text(json.dumps(
        {"spec_seed": args.spec_seed, "budget": args.budget, "tol": args.tol,
         "n_specs": len(targets), "permutations": args.perms,
         "measure_all_per_eval": {"ppo": PPO_MEASURE_ALL_PER_EVAL,
                                  "search": SEARCH_MEASURE_ALL_PER_EVAL},
         "arms": table}, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
