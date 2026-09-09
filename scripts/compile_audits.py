"""Compile every target-audit pass into one comparison table.

WHY THIS EXISTS. The multi-seed audits produce eight files across two protocols, and
comparing them by eye is exactly how a favourable protocol gets quoted and an unfavourable
one gets forgotten. This prints both protocols for every seed, always, in one table.

NAMING, because the repo has a trap in it. `target_audit_seed1.json` is the seed-0 MODEL
evaluated on spec-seed 1 -- a held-out spec draw, not a seed-1 model. Results for the
seed-N POLICY are named `target_audit_modelseedN*.json`. This script reads only the
`modelseed` files plus the two seed-0 baselines, and says which is which in its output.

THE TWO PROTOCOLS, both reported because a one-protocol comparison is weaker:
  default        - the deterministic policy may replay one trajectory after the env
                   terminates, so it under-counts DISTINCT designs.
  fresh-restarts - re-seeds each attempt; the fairer count.

Incomplete files are included and flagged, so this can be run while audits are still in
flight. Percentages for a partial file are over the rows present, never over 32.

Read-only. Nothing here writes to results/ or touches a frozen number.

    PYTHONPATH=src python scripts/compile_audits.py
    PYTHONPATH=src python scripts/compile_audits.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"

# label -> (filename, protocol). Seed 0 is the frozen headline both other seeds compare to.
PASSES = [
    ("seed 0", "target_audit_clean40k.json", "default"),
    ("seed 0", "target_audit_freshrestart.json", "fresh"),
    ("seed 1", "target_audit_modelseed1.json", "default"),
    ("seed 1", "target_audit_modelseed1_freshrestart.json", "fresh"),
    ("seed 2", "target_audit_modelseed2.json", "default"),
    ("seed 2", "target_audit_modelseed2_freshrestart.json", "fresh"),
    ("seed 3", "target_audit_modelseed3.json", "default"),
    ("seed 3", "target_audit_modelseed3_freshrestart.json", "fresh"),
]


def med(xs):
    """Median that ignores None. `best_abs_err` is None whenever a spec produced no
    guard-valid design at all, and statistics.median raises on a None rather than
    skipping it -- which is how an earlier version of this died mid-table."""
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load(fname: str):
    p = RES / fname
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None  # audits rewrite in place; a read can land mid-write


def stats(d: dict, arm: str = "ppo") -> dict:
    """Per-pass tally for one arm. `arm` is "ppo" or "random"; both are recorded by
    target_audit under an identical budget and an identical validity test, so the
    random column is the only thing that makes the ppo column mean anything."""
    rows = d.get("rows") or []
    n = len(rows)
    ppo = [r[arm] for r in rows]
    loose = [p for p in ppo if p["loose_solved_at"] is not None]
    strict = [p for p in ppo if p["strict_solved_at"] is not None]
    # A spec with no guard-valid design at all is a different failure from one that was
    # valid but off-target, so they are counted separately rather than lumped as "failed".
    no_valid = [p for p in ppo if not p["n_valid"]]
    return {
        "n": n,
        # The frozen seed-0 headline predates the `complete` flag, so absence of the
        # key is not evidence of an unfinished run. Completeness is rows == specs.
        "complete": bool(d.get("complete")) or (n > 0 and n == d.get("specs")),
        "loose": len(loose),
        "strict": len(strict),
        "no_valid": len(no_valid),
        "median_evals_to_strict": med([p["strict_solved_at"] for p in strict]),
        "median_abs_err": med([p["best_abs_err"] for p in ppo]),
        "median_abs_err_strict": med([p["best_abs_err"] for p in strict]),
        "budget": d.get("budget"),
        "tol": d.get("tol"),
        "spec_seed": d.get("spec_seed"),
        "model": d.get("model"),
    }


def pct(a: int, b: int) -> str:
    return "  -  " if not b else "%5.1f%%" % (100.0 * a / b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    args = ap.parse_args()

    table = []
    for seed, fname, proto in PASSES:
        d = load(fname)
        if d is None:
            table.append({"seed": seed, "proto": proto, "file": fname, "missing": True})
            continue
        s = stats(d)
        s["rand"] = stats(d, "random")
        s.update(seed=seed, proto=proto, file=fname, missing=False)
        table.append(s)

    print()
    print("TARGET AUDITS -- every seed, both protocols")
    print("strict = a guard-valid design within tol dB of target; loose = guard-valid only")
    print()
    hdr = ("%-7s %-8s %-5s %-13s %-13s %-8s %-9s %-11s %s"
           % ("seed", "protocol", "n", "loose", "strict", "no-valid",
              "med err", "RAND strict", ""))
    print(hdr)
    print("-" * len(hdr))
    for t in table:
        if t.get("missing"):
            print("%-7s %-8s  not started" % (t["seed"], t["proto"]))
            continue
        flag = "" if t["complete"] else "  << IN FLIGHT, partial"
        print("%-7s %-8s %-5d %-13s %-13s %-8s %-9s %-11s%s" % (
            t["seed"], t["proto"], t["n"],
            "%2d  %s" % (t["loose"], pct(t["loose"], t["n"])),
            "%2d  %s" % (t["strict"], pct(t["strict"], t["n"])),
            "%d" % t["no_valid"],
            "-" if t["median_abs_err"] is None else "%.2f dB" % t["median_abs_err"],
            "%2d  %s" % (t["rand"]["strict"], pct(t["rand"]["strict"], t["n"])),
            flag))

    # ---- regression check against the seed-0 baseline, per protocol ------------------
    base = {t["proto"]: t for t in table
            if t.get("seed") == "seed 0" and not t.get("missing")}
    print()
    print("CHANGE vs seed 0, same protocol  (strict rate, percentage points)")
    print("only complete passes are compared; a partial rate is not a rate")
    print()
    for t in table:
        if t.get("missing") or t["seed"] == "seed 0":
            continue
        b = base.get(t["proto"])
        if b is None or not t["complete"] or not b["complete"]:
            print("  %-7s %-8s  pending" % (t["seed"], t["proto"]))
            continue
        d = 100.0 * t["strict"] / t["n"] - 100.0 * b["strict"] / b["n"]
        verdict = "BETTER" if d > 3 else ("WORSE" if d < -3 else "within noise")
        print("  %-7s %-8s  %+6.1f pp   %s" % (t["seed"], t["proto"], d, verdict))

    # ---- matched-spec comparison ----------------------------------------------------
    # A partial pass MUST NOT be compared against a complete one. The audit walks specs in
    # a fixed order and the early specs are markedly harder: on the first 9, every seed
    # scores ~5-6/9 loose, against ~26/32 over the full set. Comparing seed 3's first 11
    # rows to seed 0's 32 therefore invents a deficit that is not there. This section
    # truncates every pass to the shortest one, so all seeds face identical targets.
    for proto in ("default", "fresh"):
        group = [t for t in table if not t.get("missing") and t["proto"] == proto]
        if len(group) < 2:
            continue
        n = min(t["n"] for t in group)
        if n == 0:
            continue
        print()
        print("MATCHED on the first %d specs, %s protocol (identical targets per seed)"
              % (n, proto))
        print("  %-7s %-10s %-10s %s" % ("seed", "loose", "strict", "median n_valid"))
        for t in group:
            d = load(t["file"])
            ppo = [r["ppo"] for r in (d.get("rows") or [])[:n]]
            L = sum(1 for p_ in ppo if p_["loose_solved_at"] is not None)
            S = sum(1 for p_ in ppo if p_["strict_solved_at"] is not None)
            nv = sorted(p_["n_valid"] for p_ in ppo)
            print("  %-7s %-10s %-10s %d" % (
                t["seed"], "%d/%d" % (L, n), "%d/%d" % (S, n), nv[len(nv) // 2]))

    # ---- per-spec difficulty, pooled across seeds -------------------------------------
    # "Our policy is weak on spec 7" and "spec 7 is unreachable under the frozen guard"
    # look identical in a per-seed table and demand opposite responses. Pooling across
    # independently-trained seeds separates them: a spec no seed ever reaches is a
    # property of the spec, not of any policy, and no amount of retraining will move it.
    for proto in ("default", "fresh"):
        group = [t for t in table if not t.get("missing") and t["proto"] == proto]
        if len(group) < 2:
            continue
        n = min(t["n"] for t in group)
        if n == 0:
            continue
        loaded = [(t["seed"], load(t["file"])) for t in group]
        never_loose, never_strict, rand_wins = [], [], []
        for i in range(n):
            L = S = RS = 0
            for _seed, d in loaded:
                r = (d.get("rows") or [])[i]
                L += r["ppo"]["loose_solved_at"] is not None
                S += r["ppo"]["strict_solved_at"] is not None
                RS += r["random"]["strict_solved_at"] is not None
            if L == 0:
                never_loose.append(i)
            if S == 0:
                never_strict.append(i)
            if RS > S:
                rand_wins.append(i)
        print()
        print("PER-SPEC, pooled over %d seeds, first %d specs, %s protocol"
              % (len(loaded), n, proto))
        print("  no seed reached guard-valid at all : %2d specs  %s"
              % (len(never_loose), never_loose or "-"))
        print("  no seed reached target tolerance   : %2d specs  %s"
              % (len(never_strict), never_strict or "-"))
        print("  random beat PPO (more seeds strict): %2d specs  %s"
              % (len(rand_wins), rand_wins or "-"))

    # ---- replay waste, the mechanism behind the protocol gap --------------------------
    # The default protocol lets the deterministic policy re-walk one trajectory after the
    # episode terminates, so most of its "evaluations" re-measure a design it already
    # measured. Counting DISTINCT boost values against total valid ones prices that
    # directly, and explains the whole default-vs-fresh strict gap without appeal to
    # anything about the policy itself.
    print()
    print("REPLAY WASTE  (duplicate guard-valid evaluations, PPO arm)")
    print("  %-7s %-8s %8s %9s %s" % ("seed", "protocol", "valid", "distinct", "wasted"))
    for t in table:
        if t.get("missing"):
            continue
        d = load(t["file"])
        if d is None:
            continue
        tot = uniq = 0
        for r in d.get("rows") or []:
            b = [round(x, 6) for x in r["ppo"]["all_valid_boosts"]]
            tot += len(b); uniq += len(set(b))
        if not tot:
            continue
        print("  %-7s %-8s %8d %9d %5.0f%%%s"
              % (t["seed"], t["proto"], tot, uniq, 100.0 * (tot - uniq) / tot,
                 "" if t["complete"] else "   (partial)"))

    # ---- boost-margin regimes ---------------------------------------------------------
    # margin = channel_loss - target_boost. A negative margin asks the equalizer for more
    # boost than the channel loses. The policy is near-perfect on positive margin and
    # near-useless on negative; splitting the rate this way keeps a 3-spec effect from
    # being averaged into invisibility -- or, equally, from being quoted as a headline.
    for proto in ("default", "fresh"):
        group = [t for t in table if not t.get("missing") and t["proto"] == proto
                 and t["complete"]]
        if not group:
            continue
        print()
        print("BOOST MARGIN  (%s protocol, %d complete passes) -- RAW POLICY, no G3.2"
              % (proto, len(group)))
        for lo, hi, name in ((None, 0.0, "margin < 0  (target > channel loss)"),
                             (0.0, None, "margin >= 0")):
            pL = pN = rL = 0
            for t in group:
                d = load(t["file"])
                for r in d.get("rows") or []:
                    m = r["channel_loss_db"] - r["target_boost_db"]
                    if (lo is None and m >= hi) or (hi is None and m < lo):
                        continue
                    pN += 1
                    pL += r["ppo"]["loose_solved_at"] is not None
                    rL += r["random"]["loose_solved_at"] is not None
            if pN:
                print("  %-38s n=%3d | PPO loose %5.1f%%   RANDOM loose %5.1f%%"
                      % (name, pN, 100.0 * pL / pN, 100.0 * rL / pN))

    # ---- is any seed actually different? ----------------------------------------------
    # Eyeballing a 32-job rate is how you talk yourself into a regression that is not
    # there -- it happened twice while this analysis was being written. Each seed is
    # tested against the OTHER seeds pooled, with Fisher's exact test, on matched jobs
    # only. Fisher rather than a normal approximation because these counts are small.
    # Reported for both loose and strict; a difference that fails here is reported as
    # "not separable", never quietly as a trend.
    try:
        from scipy.stats import fisher_exact
    except Exception:
        fisher_exact = None
    if fisher_exact is not None:
        for proto in ("default", "fresh"):
            group = [t for t in table if not t.get("missing") and t["proto"] == proto]
            if len(group) < 3:
                continue
            n = min(t["n"] for t in group)
            if n < 8:
                continue
            counts = {}
            for t in group:
                d = load(t["file"])
                ppo = [r["ppo"] for r in (d.get("rows") or [])[:n]]
                counts[t["seed"]] = (
                    sum(1 for p_ in ppo if p_["loose_solved_at"] is not None),
                    sum(1 for p_ in ppo if p_["strict_solved_at"] is not None))
            partial = any(not t["complete"] for t in group)
            print()
            print("SEED DIFFERENCE TEST -- %s protocol, matched first %d jobs%s"
                  % (proto, n, "   (PARTIAL DATA -- provisional)" if partial else ""))
            print("  each seed vs the other seeds pooled, Fisher exact, two-sided")
            print("  CAVEAT: %d tests are run here (%d seeds x 2 metrics). One p<0.05 among"
                  % (2 * len(counts), len(counts)))
            print("  them is ordinary chance -- Bonferroni-corrected the threshold is %.4f."
                  % (0.05 / (2 * len(counts))))
            print("  Seed 3 is the exception: it was flagged as suspect from its TRAINING")
            print("  curve before this audit ran, so for seed 3 this is a confirmatory test")
            print("  of a prior hypothesis, not one of eight fishing expeditions.")
            for seed, (L, S) in counts.items():
                others = [v for k, v in counts.items() if k != seed]
                oL = sum(v[0] for v in others); oS = sum(v[1] for v in others)
                oN = n * len(others)
                for label, a, b in (("loose ", L, oL), ("strict", S, oS)):
                    ob = oN if label == "loose " else oN
                    _, pv = fisher_exact([[a, n - a], [b, ob - b]])
                    verdict = ("DIFFERENT" if pv < 0.05 else
                               "not separable" if pv > 0.10 else "borderline")
                    print("    %-7s %s  %2d/%-3d vs %3d/%-3d pooled   p=%.3f   %s"
                          % (seed, label, a, n, b, ob, pv, verdict))

    done = sum(1 for t in table if not t.get("missing") and t["complete"])
    print()
    print("%d of %d passes complete." % (done, len(PASSES)))

    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
