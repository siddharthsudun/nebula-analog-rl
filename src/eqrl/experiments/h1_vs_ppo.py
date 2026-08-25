"""The head-to-head table: frozen PPO, PPO+H1, and the matched chance line, per spec seed.

WHY THIS EXISTS. `hybrid_report` answers "did refinement improve the policy's own handoff"
from inside the arm. That is the paired comparison and it is informative, but it cannot say
what adding H1 buys against the policy run on its own budget. This puts frozen PPO, the
random arm, and H1 in one table on the SAME 32 specs, with the SAME strict criterion and
the SAME per-method chance line, so the delta from adding H1 is directly visible.

THE BUDGET HAS TWO HONEST READS and section 13 pre-declares both, so both are printed:

    evaluation-matched   budget 20   20 PPO evaluations = 40 measure_all
    simulation-matched   budget 10   10 PPO evaluations = 20 measure_all, what H1 spends

They are different comparisons, not a better and a worse one. PPO costs 2.00 measure_all
per evaluation against a search arm's 1.00 (section 13, measured), so an evaluation-matched
read hands PPO twice the simulator. The simulation-matched read is the parity comparison
against H1; the evaluation-matched read is the one comparable to the seed 0/1 numbers
already published. Reporting only whichever is kinder is exactly the failure this repo
keeps pre-registering against.

THE CHANCE LINE IS THE POINT. A method that finds more distinct feasible designs hits more
targets by accident, so raw strict counts across methods are not comparable to each other
without it. k is computed identically for every arm: distinct rounded boosts, capped by the
number of loose passes.

Reads artifacts only. Runs no simulation, loads no policy, changes no threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POOL = "results/target_tracking_clean40k.json"


def pool_boosts() -> list[float]:
    return [r["boost"] for r in json.loads(Path(POOL).read_text())
            if r.get("boost") is not None]


def chance_line(targets, arms_k, observed, tol, rng, n_sim=2000):
    """Identical in definition to final_report.py and hybrid_report.py."""
    pool = pool_boosts()
    ps = [sum(abs(b - t) <= tol for b in pool) / len(pool) for t in targets]
    exp = sum(1 - (1 - q) ** k for q, k in zip(ps, arms_k))
    sims = [sum(rng.random() < (1 - (1 - q) ** k) for q, k in zip(ps, arms_k))
            for _ in range(n_sim)]
    return exp, float(np.mean([v >= observed for v in sims]))


def k_of(t: dict) -> int:
    """distinct rounded boosts, capped by loose passes -- the same rule for every arm."""
    return min(len({round(b, 4) for b in t["all_valid_boosts"]}), t["n_loose_pass"])


def score(rows, key, tol, rng):
    """One arm's strict count, loose count, median final error and chance line."""
    ts = [r["target_boost_db"] for r in rows]
    arms = [r[key] for r in rows]
    strict = sum(1 for a in arms if a["strict_solved_at"])
    loose = sum(1 for a in arms if a["loose_solved_at"])
    errs = [a["best_abs_err"] for a in arms if a["best_abs_err"] is not None]
    ks = [k_of(a) for a in arms]
    exp, pv = chance_line(ts, ks, strict, tol, rng)
    return {"strict": strict, "loose": loose, "n": len(rows),
            "median_err": float(np.median(errs)) if errs else None,
            "mean_k": float(np.mean(ks)), "chance": exp, "p": pv}


def load(path: str):
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--out", default="results/h1_vs_ppo.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = {}
    for s in args.seeds:
        entries = []
        # (label, artifact, arm key, measure_all per spec)
        want = [("frozen PPO  (20 evals, 40 sim)", f"results/target_audit_seed{s}_b20.json",
                 "ppo", 40),
                ("frozen PPO  (10 evals, 20 sim)", f"results/target_audit_seed{s}_b10.json",
                 "ppo", 20),
                ("random      (20 evals, 20 sim)", f"results/target_audit_seed{s}_b20.json",
                 "random", 20),
                ("PPO + H1    (15 evals, 20 sim)", f"results/hybrid_audit_h1_seed{s}.json",
                 "h1", 20)]
        for label, path, key, cost in want:
            rows = load(path)
            if rows is None:
                print(f"  (missing {path}, skipped)")
                continue
            r = score(rows, key, args.tol, rng)
            r["method"], r["measure_all_per_spec"] = label.strip(), cost
            entries.append(r)

        if not entries:
            continue
        print("=" * 92)
        print("HEAD-TO-HEAD   spec-seed %d   %d specs   strict = |achieved - target| "
              "<= %.1f dB" % (s, entries[0]["n"], args.tol))
        print("=" * 92)
        print("method                            sim/spec  strict  chance   p      "
              "loose   median err")
        for e in entries:
            print("  %-30s  %4d      %2d/%-2d   %5.1f   %.4f  %2d/%-2d   %s"
                  % (e["method"], e["measure_all_per_spec"], e["strict"], e["n"],
                     e["chance"], e["p"], e["loose"], e["n"],
                     "-" if e["median_err"] is None else "%.2f dB" % e["median_err"]))
        print("\n  above chance at p < 0.05: %s"
              % (", ".join(e["method"] for e in entries if e["p"] < 0.05) or "NONE"))
        out[str(s)] = entries
        print()

    if out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
