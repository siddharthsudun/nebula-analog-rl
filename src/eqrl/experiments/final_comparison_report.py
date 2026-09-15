"""Reads results/final_comparison_seed23.json and reports exactly what §5 + amendment 1 ask.

ORDER IS DELIBERATE. The primary outcomes come first -- median absolute target error, cost,
reduction from the handoff, and each arm's own matched chance line -- and the solve counts
come after them, under a heading that says what they are. The seed-3 slice was 6/10 vs 5/10
and that one-spec gap is not a result at n = 40 either; §4.1 fixed that in advance and this
file is not allowed to quietly promote it.

The chance line and the terciles are imported from hybrid_report rather than rewritten, so
they are the same pool, the same k, and the same permutation every earlier arm was scored
against. A private copy would be a second definition to keep in sync, and the numbers would
stop being comparable the first time one drifted.

Runs no simulation, loads no policy, changes no threshold.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from eqrl.experiments.hybrid_report import chance_line, terciles

NAMES = {"a": "A  PPO -> H1", "b": "B  PPO -> G3.2", "c": "C  PPO -> G3.2 -> H1"}


def kvals(rows, a):
    """Distinct designs, capped by loose passes -- the definition every arm is scored on."""
    return [min(len({round(b, 4) for b in r[a]["all_valid_boosts"]}), r[a]["n_loose_pass"])
            for r in rows]


def med(vals):
    v = [x for x in vals if x is not None]
    return st.median(v) if v else None


def f(x, w=6, p=2):
    return ("%*s" % (w, "-")) if x is None else ("%*.*f" % (w, p, x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/final_comparison_seed23.json")
    p.add_argument("--seed", type=int, default=20260826, help="permutation seed, reporting only")
    args = p.parse_args()

    d = json.loads(Path(args.run).read_text())
    rows, arms, tol = d["rows"], d["arms"], d["tol"]
    n = len(rows)
    tgt = [r["target_boost_db"] for r in rows]
    rng = np.random.default_rng(args.seed)

    print("=" * 92)
    print("FINAL THREE-ARM COMPARISON   spec-seed %s   %d specs   %s"
          % (d["spec_seed"], n, "COMPLETE" if d["complete"] else "*** PARTIAL ***"))
    print("  budget %g measure_all per spec, every arm.  strict = hard_pass and |err| <= "
          "%.1f dB" % (d["prereg"]["budget_measure_all"], tol))
    print("=" * 92)

    # ---- per spec ------------------------------------------------------------------
    print("\nPER SPEC   err = |achieved - requested|, over the whole arm; ev = stage-2 "
          "evaluations")
    print("            handoff |" + "|".join("  %-20s" % NAMES[a][:20] for a in arms))
    print("spec  tgt   boost   |" + "|".join("   err   ev  solved   " for _ in arms))
    for r in rows:
        hb = r["handoff"]["boost_db"]
        cells = []
        for a in arms:
            e = r[a]
            cells.append(" %s %4d  %-7s" % (
                f(e["best_abs_err"], 6), e["stage2_evals"],
                "strict" if e["strict_solved_at"] else
                ("loose" if e["loose_solved_at"] else "-")))
        print("  %2d %5.2f  %s |%s" % (r["spec"], r["target_boost_db"], f(hb), "|".join(cells)))

    # ---- PRIMARY -------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("PRIMARY OUTCOMES")
    print("=" * 92)
    print("\n  %-22s %10s %10s %10s %10s %10s"
          % ("arm", "median err", "mean ev", "max ev", "mean m_all", "max m_all"))
    for a in arms:
        errs = [r[a]["best_abs_err"] for r in rows]
        ev = [r[a]["stage2_evals"] for r in rows]
        ma = [r[a]["measure_all_spent"] for r in rows]
        print("  %-22s %10s %10.2f %10d %10.2f %10.1f"
              % (NAMES[a], f(med(errs), 10, 3), st.mean(ev), max(ev), st.mean(ma), max(ma)))
    nodesign = {a: sum(1 for r in rows if r[a]["best_abs_err"] is None) for a in arms}
    if any(nodesign.values()):
        print("\n  specs with NO hard_pass design at all (excluded from median err): %s"
              % ", ".join("%s %d" % (a.upper(), v) for a, v in nodesign.items()))

    # ---- 3. reduction from the handoff, §5.1 ---------------------------------------
    print("\n  TARGET-ERROR REDUCTION FROM THE PPO HANDOFF   (|err_handoff| - |err_final|,"
          " dB)")
    have = [r for r in rows if r["handoff"]["boost_db"] is not None]
    print("    computed over the %d of %d specs with a guard-valid handoff; %d excluded "
          "and never imputed" % (len(have), n, n - len(have)))
    for a in arms:
        red = []
        for r in have:
            fin = r[a]["best_abs_err"]
            if fin is None:
                continue
            red.append(abs(r["handoff"]["boost_db"] - r["target_boost_db"]) - fin)
        print("    %-22s median %+7.3f   improved %2d/%2d   worsened %d"
              % (NAMES[a], med(red) or 0.0, sum(1 for v in red if v > 0), len(red),
                 sum(1 for v in red if v < 0)))

    # ---- 4. the matched chance line -------------------------------------------------
    print("\n  MATCHED CHANCE LINE   k = min(distinct rounded boosts, loose passes), pooled")
    print("  across stage 1 and ALL of stage 2 including any fallback portion.")
    print("    %-22s %8s %8s %12s %10s" % ("arm", "strict", "mean k", "chance line", "p(>=obs)"))
    for a in arms:
        ks = kvals(rows, a)
        strict = sum(1 for r in rows if r[a]["strict_solved_at"])
        exp, pv, npool = chance_line(tgt, ks, strict, tol, rng)
        print("    %-22s %5d/%2d %8.2f %9.1f/%2d %10.4f%s"
              % (NAMES[a], strict, n, st.mean(ks), exp, n, pv,
                 "   ABOVE CHANCE" if pv < 0.05 else "   on its line"))
    print("    (pool of %d demonstrably-achieved boosts, the same pool as every other arm)"
          % npool)

    # ---- SECONDARY ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("SECONDARY OUTCOMES -- descriptive. §4.1 fixed in advance that n = 40 cannot")
    print("resolve a few-spec difference in solve count, and no claim is made from one.")
    print("=" * 92)
    print("\n  %-22s %10s %10s %10s %10s"
          % ("arm", "strict", "loose", "mean k", "mean valid"))
    for a in arms:
        print("  %-22s %7d/%-2d %7d/%-2d %10.2f %10.2f"
              % (NAMES[a],
                 sum(1 for r in rows if r[a]["strict_solved_at"]), n,
                 sum(1 for r in rows if r[a]["loose_solved_at"]), n,
                 st.mean(kvals(rows, a)),
                 st.mean(r[a]["n_valid"] for r in rows)))

    # ---- paired, spec by spec -------------------------------------------------------
    print("\n  PAIRED, SPEC BY SPEC   (same specs, same handoff, so differences are the "
          "arms')")
    for x, y in [(u, v) for u in arms for v in arms if u < v]:
        both = [(r[x]["best_abs_err"], r[y]["best_abs_err"]) for r in rows
                if r[x]["best_abs_err"] is not None and r[y]["best_abs_err"] is not None]
        if not both:
            continue
        dif = [u - v for u, v in both]
        print("    %s vs %s   on %d specs:  %s closer on %d, %s closer on %d, tied %d"
              "   median diff %+0.3f dB"
              % (x.upper(), y.upper(), len(both), y.upper(),
                 sum(1 for v in dif if v > 1e-9), x.upper(),
                 sum(1 for v in dif if v < -1e-9),
                 sum(1 for v in dif if abs(v) <= 1e-9), st.median(dif)))

    # ---- terciles, amendment 1 §1 ----------------------------------------------------
    tc = terciles(tgt)
    lo, hi = np.quantile(tgt, [1 / 3, 2 / 3])
    print("\n  BY TARGET TERCILE   cut at the 1/3 and 2/3 quantiles of THIS set's targets:"
          " %.2f, %.2f dB" % (lo, hi))
    print("    %-22s %-24s %-24s %s"
          % ("arm", "low tercile", "mid tercile", "high tercile"))
    for a in arms:
        cells = []
        for g in (0, 1, 2):
            sub = [r for r, t in zip(rows, tc) if t == g]
            m = med([r[a]["best_abs_err"] for r in sub])
            s = sum(1 for r in sub if r[a]["strict_solved_at"])
            cells.append("med %s  strict %d/%-2d" % (f(m, 5, 2), s, len(sub)))
        print("    %-22s %-24s %-24s %s" % (NAMES[a], *cells))

    # ---- ARM C ----------------------------------------------------------------------
    if "c" in arms and "b" in arms:
        print("\n" + "=" * 92)
        print("ARM C -- THE HYBRID'S ACTUAL VALUE PROPOSITION")
        print("=" * 92)
        fired = [r for r in rows if r["c"]["fallback"]["fired"]]
        solved_first = [r for r in rows if r["c"]["solver"]["reached_target"]]
        nobudget = [r for r in rows if not r["c"]["solver"]["reached_target"]
                    and not r["c"]["fallback"]["fired"]]
        print("\n  G3.2 reached its own 0.25 dB target, fallback never fired:  %2d/%d"
              % (len(solved_first), n))
        print("  G3.2 stopped short and the fallback FIRED:                   %2d/%d"
              % (len(fired), n))
        print("  G3.2 stopped short with no budget left to hand over:         %2d/%d"
              % (len(nobudget), n))
        if fired:
            ev = [r["c"]["fallback"]["evals_used"] for r in fired]
            print("\n  the fallback received %d-%d evaluations (mean %.1f) out of the same "
                  "10-call pool" % (min(ev), max(ev), st.mean(ev)))

            rec_strict = [r for r in fired
                          if r["c"]["strict_solved_at"] and not r["b"]["strict_solved_at"]]
            lost = [r for r in fired
                    if r["b"]["strict_solved_at"] and not r["c"]["strict_solved_at"]]
            gains = []
            for r in fired:
                b, c = r["b"]["best_abs_err"], r["c"]["best_abs_err"]
                if b is not None and c is not None:
                    gains.append(b - c)
                elif b is None and c is not None:
                    gains.append(None)      # found a first feasible design; not a dB gain
            newfeas = sum(1 for r in fired
                          if r["b"]["best_abs_err"] is None
                          and r["c"]["best_abs_err"] is not None)
            print("\n  WHEN G3.2 FAILED, DID H1 RECOVER IT?")
            print("    strict-recovered by the fallback:              %2d of %d fired"
                  % (len(rec_strict), len(fired)))
            print("    first hard_pass design found by the fallback:  %2d of %d fired"
                  % (newfeas, len(fired)))
            g = [v for v in gains if v is not None]
            if g:
                print("    target-error improvement on the rest:  median %+0.3f dB, "
                      "improved %d, unchanged %d"
                      % (st.median(g), sum(1 for v in g if v > 1e-9),
                         sum(1 for v in g if abs(v) <= 1e-9)))
            if lost:
                print("    *** %d spec(s) strict under B but NOT under C: %s. C keeps the "
                      "best design\n        across both stages, so this should be "
                      "impossible -- investigate before reporting."
                      % (len(lost), ", ".join(str(r["spec"]) for r in lost)))

            print("\n  WHY THE FALLBACK FIRED   (diagnostic only -- §3.2: never a control "
                  "input)")
            why: dict[tuple, int] = {}
            for r in fired:
                s = r["c"]["solver"]
                why[(s["reason"], s["blocked_by"], s["wall_hit"])] = \
                    why.get((s["reason"], s["blocked_by"], s["wall_hit"]), 0) + 1
            for (reason, blocked, wall), cnt in sorted(why.items(), key=lambda e: -e[1]):
                print("    %2d x  %-38s wall_hit %-5s  %s"
                      % (cnt, (reason or "-")[:38], wall, blocked or ""))
            print("\n  per-spec: %s"
                  % ", ".join("%d(%dev)" % (r["spec"], r["c"]["fallback"]["evals_used"])
                              for r in fired))
            src: dict[str, int] = {}
            for r in fired:
                s = r["c"]["fallback"]["x0_source"]
                src[s] = src.get(s, 0) + 1
            print("  fallback start point (amendment 1 §4): %s"
                  % ", ".join("%s x%d" % (k_, v) for k_, v in src.items()))

    print("\n" + "=" * 92)
    print("read %s -- %d simulations were run to produce it" % (args.run, d["n_simulations_run"]))


if __name__ == "__main__":
    main()
