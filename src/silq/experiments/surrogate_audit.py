"""Does the surrogate clear the gate that was pre-registered before it was built?

THE GATE (docs/REPRODUCE.md section 16, committed before this ran):

    on a CHRONOLOGICAL split of the corpus --
      (a) boost_db MAE <= 0.5 dB overall,                                     AND
      (b) >= 90% of held-out designs within 1.5 dB of truth IN THE SPARSEST
          DECILE by distance to the nearest training design.

    Clear it and the surrogate ships into the H2 arm. Miss it and H2 is CUT and the week
    proceeds with H1 only. THE GATE DOES NOT MOVE. It was written down at a level a
    preliminary probe had already cleared, precisely so that seeing the full-corpus
    number could not be an argument for relaxing it.

WHY CHRONOLOGICAL AND NOT I.I.D. The corpus is mostly PPO design trajectories, which walk
in steps of 0.18 through a unit cube. An i.i.d. split therefore puts a held-out design's
own one-step neighbour into the training set, and the model gets graded on interpolating
between two points it has already seen. That is not a hard question and the score is not
a real one. Sorting by run id sorts by UTC timestamp, so training on the earliest 80%
and testing on the latest 20% holds out whole runs -- different policies, different
experiments, different regions.

Both splits are printed. The i.i.d. number is reported ONLY as the contrast that shows
how much the leak was worth; the gate is judged on the chronological one.

THE DISTANCE STRATIFICATION IS THE POINT OF (b). An aggregate MAE over a corpus this
clustered is dominated by dense regions and says nothing about the sparse ones -- which
are exactly where a search would ask. Bucketing the error by distance-to-nearest-training-
design answers the question that matters: how wrong is it where it knows least.

Also reports COVERAGE -- how far a uniformly random design sits from anything recorded --
because a model that is accurate only where nothing will ever ask is not useful.

TWO TESTS, AND THE GATE IS ONLY ONE OF THEM. A held-out split grades the model against
labels made by the same parser that made its training labels, so it measures INTERNAL
CONSISTENCY and is blind to any error the two share. That is not hypothetical here: the
first version of this corpus read the real part of H as though it were magnitude in dB,
and this gate passed it at 0.166 dB MAE while it was predicting a quantity with no
physical meaning. So the audit runs both, in this order, and neither is optional:

    PARSER CORRECTNESS   independent SPICE ground truth  ->  metrics_from_acx
                         `assert_matches_feasibility_band`, raises on mismatch. Uses
                         designs whose metrics were recorded by `sim.measures.peaking`
                         itself, so nothing in `surrogate.py` produced the answer key.

    MODEL ACCURACY       corpus labels -> model fit -> held-out corpus labels
                         the pre-registered gate below.

The first must pass before the second means anything at all. A model can only be as
correct as its labels, and no split of those labels can tell you whether they are right.

Runs no simulation, loads no policy, changes no threshold, bound, reward, criterion or
frozen checkpoint. Reads `results/raw` and writes one JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from silq.surrogate import (METRICS, Surrogate, assert_encoding_matches_ctle,
                            assert_matches_feasibility_band, build_corpus, load_corpus,
                            save_corpus)

#: PRE-REGISTERED. Read from docs/REPRODUCE.md section 16 and duplicated here so the
#: script fails loudly rather than silently drifting from the committed protocol.
GATE_MAE_DB = 0.5
GATE_SPARSEST_DECILE_WITHIN = 0.90
STRICT_TOL_DB = 1.5


def evaluate(sur: Surrogate, Xte: np.ndarray, Yte: np.ndarray) -> dict:
    pred, dist = sur.predict(Xte)
    err = pred - Yte
    out: dict = {"n_test": int(len(Xte)), "per_metric": {}}
    for j, name in enumerate(METRICS):
        var = float(Yte[:, j].var())
        out["per_metric"][name] = {
            "mae": float(np.abs(err[:, j]).mean()),
            "median_abs_err": float(np.median(np.abs(err[:, j]))),
            "r2": float(1.0 - (err[:, j] ** 2).mean() / var) if var > 0 else float("nan"),
            "sd_of_truth": float(np.sqrt(var)),
        }
    b = METRICS.index("boost_db")
    # Deciles by distance to the nearest training design, split BY RANK rather than by
    # value. Distances tie heavily -- a corpus of design trajectories revisits points --
    # and value-based quantile edges then collapse several deciles into one bucket, which
    # both mislabels them and hides the sparse tail the gate is about. Equal-count blocks
    # over the sorted order are what "decile" is supposed to mean. The LAST one is the
    # sparsest and is the one the gate is written against.
    order = np.argsort(dist, kind="stable")
    strata = []
    for i, block in enumerate(np.array_split(order, 10)):
        if not len(block):
            continue
        strata.append({
            "decile": i + 1, "d_lo": float(dist[block].min()),
            "d_hi": float(dist[block].max()), "n": int(len(block)),
            "boost_mae": float(np.abs(err[block, b]).mean()),
            "within_1p5db": float(np.mean(np.abs(err[block, b]) <= STRICT_TOL_DB)),
        })
    out["boost_by_distance_decile"] = strata
    out["boost_within_1p5db_overall"] = float(np.mean(np.abs(err[:, b]) <= STRICT_TOL_DB))
    out["boost_within_0p5db_overall"] = float(np.mean(np.abs(err[:, b]) <= 0.5))
    out["sparsest_decile"] = strata[-1] if strata else None
    return out


def show(tag: str, r: dict) -> None:
    print(f"\n{tag}   (test n = {r['n_test']})")
    for name, s in r["per_metric"].items():
        print("   %-14s MAE %6.3f   median %6.3f   R2 %6.3f   (sd of truth %5.2f)"
              % (name, s["mae"], s["median_abs_err"], s["r2"], s["sd_of_truth"]))
    print("   boost error by distance to nearest training design:")
    for s in r["boost_by_distance_decile"]:
        mark = "   <- SPARSEST, the gate is judged here" if s["decile"] == 10 else ""
        print("     decile %2d  d in [%.3f, %.3f]  n=%5d   MAE %6.3f   within 1.5 dB %5.1f%%%s"
              % (s["decile"], s["d_lo"], s["d_hi"], s["n"], s["boost_mae"],
                 100 * s["within_1p5db"], mark))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--cache", default="results/surrogate_corpus.npz",
                   help="parsed corpus; built from --raw if absent")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap records, for smoke tests")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--max-test", type=int, default=20000,
                   help="subsample the test set for the kNN query; 20k is ample for a "
                        "mean and keeps the audit to minutes. Subsampling is uniform "
                        "over the held-out block, never selective.")
    p.add_argument("--out", default="results/surrogate_audit.json")
    args = p.parse_args()

    print("checking the surrogate's coordinates against ctle.encode_action ...", flush=True)
    err = assert_encoding_matches_ctle()
    print(f"  bitwise identical after the float32 cast; float64 residual {err:.3e} "
          f"(= float32 eps, which is the cast and not a disagreement)  OK", flush=True)

    cache = Path(args.cache)
    if args.rebuild or not cache.exists():
        print(f"\nparsing the recorded SPICE corpus under {args.raw} "
              f"(no simulation is run) ...", flush=True)
        corpus = build_corpus(args.raw, limit=args.limit)
        if args.limit is None:
            save_corpus(corpus, cache)
            print(f"  cached -> {cache}", flush=True)
    else:
        corpus = load_corpus(cache)
        print(f"\nloaded cached corpus {cache}", flush=True)

    X, Y, rid = corpus["X"], corpus["Y"], corpus["run_id"]
    n = len(X)
    print(f"corpus: {n} records, {len(set(r[:8] for r in rid))} distinct run days")
    results: dict = {"corpus_records": int(n), "k": args.k,
                     "gate": {"mae_db": GATE_MAE_DB,
                              "sparsest_decile_within_1p5db": GATE_SPARSEST_DECILE_WITHIN}}

    # MANDATORY, and deliberately not behind a flag. This is the only check in the file
    # whose answer key was not produced by `surrogate.py`, so it is the only one that can
    # catch a mislabelled corpus. It raises; a failure here stops the audit rather than
    # being reported as a poor score, because every number below it would be meaningless.
    print("\nchecking the parser against independent SPICE ground truth ...", flush=True)
    gt = assert_matches_feasibility_band(root=args.raw, corpus_path=args.cache)
    n_gt = gt.pop("n_checked")
    for name, e in gt.items():
        print("   %-14s max error %.5f dB" % (name, e))
    print(f"  matches `sim.measures.peaking` on {n_gt} uniformly drawn designs  OK",
          flush=True)
    results["ground_truth_check"] = {"n_checked": n_gt, "max_abs_err_db": gt}

    rng = np.random.default_rng(0)
    n_test = int(args.test_frac * n)

    # -- the contrast: i.i.d. split. Reported to show the size of the leak, NOT to judge.
    perm = rng.permutation(n)
    tr, te = perm[n_test:], perm[:n_test]
    te = te[rng.permutation(len(te))[:args.max_test]]
    iid = evaluate(Surrogate(X[tr], Y[tr], k=args.k), X[te], Y[te])
    show("IID SPLIT  (leaky -- shown for contrast only, the gate is NOT judged here)", iid)
    results["iid_split"] = iid

    # -- the real one: hold out the most recent runs entirely.
    order = np.argsort(rid, kind="stable")
    cut = n - n_test
    tr, te = order[:cut], order[cut:]
    te = te[rng.permutation(len(te))[:args.max_test]]
    chrono = evaluate(Surrogate(X[tr], Y[tr], k=args.k), X[te], Y[te])
    show("CHRONOLOGICAL SPLIT  (held-out era -- THE GATE IS JUDGED ON THIS)", chrono)
    results["chronological_split"] = chrono
    results["chronological_train_last_run"] = str(rid[tr[-1]])
    results["chronological_test_first_run"] = str(rid[te.min()])

    # -- coverage: does the corpus reach where a search would actually ask?
    sur_full = Surrogate(X[order[:cut]], Y[order[:cut]], k=args.k)
    _, d_uniform = sur_full.predict(rng.random((4000, X.shape[1])))
    cov = {"median": float(np.median(d_uniform)), "p90": float(np.quantile(d_uniform, 0.9)),
           "max": float(d_uniform.max()), "unit_cube_diagonal": float(np.sqrt(X.shape[1]))}
    results["coverage_uniform_query"] = cov
    print("\nCOVERAGE  distance from a UNIFORMLY RANDOM design to the nearest recorded one")
    print("   median %.3f   90th pct %.3f   max %.3f   (unit cube diagonal %.2f)"
          % (cov["median"], cov["p90"], cov["max"], cov["unit_cube_diagonal"]))

    # -- the verdict.
    mae = chrono["per_metric"]["boost_db"]["mae"]
    sparse = chrono["sparsest_decile"]["within_1p5db"]
    pass_a = mae <= GATE_MAE_DB
    pass_b = sparse >= GATE_SPARSEST_DECILE_WITHIN
    results["gate_result"] = {"boost_mae_db": mae, "sparsest_decile_within_1p5db": sparse,
                              "pass_a_mae": bool(pass_a), "pass_b_sparsest": bool(pass_b),
                              "passed": bool(pass_a and pass_b)}
    print("\n" + "=" * 72)
    print("PRE-REGISTERED GATE  (docs/REPRODUCE.md section 16, committed before this ran)")
    print("  (a) boost MAE %.3f dB  <=  %.2f dB           %s"
          % (mae, GATE_MAE_DB, "PASS" if pass_a else "FAIL"))
    print("  (b) sparsest decile within 1.5 dB: %.1f%%  >=  %.0f%%   %s"
          % (100 * sparse, 100 * GATE_SPARSEST_DECILE_WITHIN, "PASS" if pass_b else "FAIL"))
    if pass_a and pass_b:
        print("\n  GATE CLEARED -- the surrogate ships into the H2 arm.")
    else:
        print("\n  GATE MISSED -- H2 IS CUT. The week proceeds with H1 only.")
        print("  The gate does not move. Do not re-run this with a looser threshold.")
    print("=" * 72)

    Path(args.out).write_text(json.dumps(results, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
