"""Is the reach plateau physics, or is it the surrogate smoothing?

`sigma_derivation` found that the median starting point cannot move boost by more than
~2.5 dB in one 9-draw generation NO MATTER HOW LARGE sigma GETS -- the curve flattens
from sigma 0.24 onward. Two very different things produce that shape:

  (1) PHYSICS. Boost is a shallow function of the design over most of the cube, and the
      designs that reach high boost are rare. Then no local sampler helps and the hybrid
      arm's whole premise is wrong.

  (2) SMOOTHING. kNN with k = 5 averages its neighbours, which compresses extremes. At
      large sigma the draws land far from any single recorded design and the prediction
      regresses toward the local mean, so the MEASURED reach shrinks even though the TRUE
      reach does not.

These are distinguished by comparing what the surrogate predicts against what the corpus
actually recorded, over the same designs. If the recorded boosts span a much wider range
than the predictions do, the plateau is (2) and the derivation understates sigma*.

Also checks the question the plateau raises about the specs themselves: are the requested
targets inside the range this circuit is recorded to achieve at all? A gap that no design
in 374k SPICE runs has ever closed is not a refinement problem.

No SPICE. No policy. Reads the corpus and the spec generator only.
"""

from __future__ import annotations

# =============================================================================
# VOID ANALYSIS -- DO NOT USE, DO NOT CITE, DO NOT RUN
# =============================================================================
# This script ran against the pre-fix surrogate corpus, whose `boost_db` labels were
# the REAL PART of the complex transfer function rather than 20*log10|H|. See
# docs/REPRODUCE.md section 19. Every number it produced is void, including:
#
#   - the 0.945-0.984 spread-compression ratios
#   - the conclusion that the reach plateau was physics rather than smoothing
#   - the corpus boost percentiles (p50 2.00 dB, p99 7.63 dB)
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

import numpy as np

from silq.experiments.hybrid_audit import make_specs
from silq.surrogate import METRICS, Surrogate, load_corpus

B = METRICS.index("boost_db")

def main() -> None:
    c = load_corpus("results/surrogate_corpus.npz")
    X, Y = c["X"], c["Y"]
    truth = Y[:, B]
    rng = np.random.default_rng(0)

    print("RECORDED boost over %d SPICE runs" % len(truth))
    for q in (0, 1, 25, 50, 75, 99, 100):
        print("   p%-3d %6.2f dB" % (q, np.percentile(truth, q)))

    # -- (1) vs (2): predict held-out points and compare the SPREADS, not the errors.
    # A compressing model can have a small MAE and still flatten the tails, which is
    # exactly the failure that would fake a reach plateau.
    idx = rng.choice(len(X), 20000, replace=False)
    mask = np.ones(len(X), bool)
    mask[idx] = False
    sur = Surrogate(X[mask], Y[mask], k=5)
    pred, dist = sur.predict_boost(X[idx])
    act = truth[idx]
    print("\nSPREAD COMPRESSION  (held-out, k = 5)")
    print("   recorded   sd %5.2f   p1 %6.2f   p99 %6.2f   range %5.2f"
          % (act.std(), np.percentile(act, 1), np.percentile(act, 99),
             np.percentile(act, 99) - np.percentile(act, 1)))
    print("   predicted  sd %5.2f   p1 %6.2f   p99 %6.2f   range %5.2f"
          % (pred.std(), np.percentile(pred, 1), np.percentile(pred, 99),
             np.percentile(pred, 99) - np.percentile(pred, 1)))
    print("   ratio of predicted spread to recorded spread: %.3f"
          % (pred.std() / act.std()))
    print("   (1.00 = no compression. Well below 1.00 means the reach curve is a "
          "floor,\n    not a ceiling, and sigma* is UNDERSTATED by the derivation.)")

    # Compression grows with distance, so report it where the large-sigma draws land.
    print("\n   by distance to nearest recorded design:")
    order = np.argsort(dist, kind="stable")
    for i, block in enumerate(np.array_split(order, 5)):
        print("     quintile %d  d in [%.3f, %.3f]  sd ratio %.3f   MAE %5.3f dB"
              % (i + 1, dist[block].min(), dist[block].max(),
                 pred[block].std() / act[block].std(),
                 np.abs(pred[block] - act[block]).mean()))

    # -- are the requested targets even in range?
    print("\nREQUESTED TARGETS vs RECORDED ACHIEVABLE BOOST")
    for seed in (0, 1, 2, 3):
        t = np.array([tg for tg, _ in make_specs(32, seed)])
        above = float(np.mean(t > np.percentile(truth, 99)))
        print("   spec seed %d  targets p50 %5.2f  min %5.2f  max %5.2f | "
              "%3.0f%% above the 99th percentile of anything ever recorded"
              % (seed, np.median(t), t.min(), t.max(), 100 * above))

if __name__ == "__main__":
    main()
