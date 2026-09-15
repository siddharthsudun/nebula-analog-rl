"""Does the surrogate hold up OFF the corpus distribution, not just off the corpus sample?

WHY THIS IS A DIFFERENT QUESTION FROM THE GATE. `surrogate_audit` held out the most recent
20% of runs -- a different era, but still PPO trajectories, still the same kind of design.
It cleared the gate at 0.166 dB. `reach_by_region` then asked the model about UNIFORMLY
RANDOM designs, which is a different distribution entirely, and got an answer that
contradicts recorded SPICE: the surrogate puts uniform-cube boost at p90 = 2.67 dB, while
`feasibility_band.json` -- 2000 uniform designs actually simulated -- records p90 = 9.61 dB
among the 460 that are guard-valid.

One of those is wrong, and it decides whether the reach plateau is a fact about the design
space or an artifact of asking a corpus-shaped model an off-corpus question. kNN regresses
toward the local mean, and the corpus mean boost is 2.00 dB, so under-prediction in
unvisited high-boost regions is exactly the failure to expect.

THE TEST. `feasibility_band` records both the design and the SPICE-measured boost for the
subset of its scatter that carries a design. Predict those with the surrogate and compare.
These designs were drawn uniformly, not by any policy, so they are a genuine off-corpus
probe with ground truth attached.

NOTE ON WHAT THIS SUBSET IS. The records carrying a design are the in-band ones, so they
are not a random slice of the uniform sample and their boost distribution should not be
read as representative. That does not weaken the test being run here, which is per-design
predicted-vs-measured, not distributional.

No SPICE. No policy. Reads two existing artifacts.
"""
from __future__ import annotations

import json

import numpy as np

from silq.surrogate import KEYS, Surrogate, load_corpus, normalize

TOL = 1.5


def main() -> None:
    c = load_corpus("results/surrogate_corpus.npz")
    sur = Surrogate(c["X"], c["Y"], k=5)

    fb = json.load(open("results/feasibility_band.json"))
    recs = [r for r in fb["scatter"] if "design" in r]
    Xq = normalize(np.array([[r["design"][k] for k in KEYS] for r in recs]))
    act = np.array([r["boost_db"] for r in recs])
    pred, dist = sur.predict_boost(Xq)
    err = pred - act

    print("OFF-CORPUS PROBE  %d uniformly drawn designs with SPICE ground truth\n"
          % len(recs))
    print("  measured boost   p50 %5.2f  p90 %5.2f  max %5.2f dB"
          % (np.percentile(act, 50), np.percentile(act, 90), act.max()))
    print("  predicted boost  p50 %5.2f  p90 %5.2f  max %5.2f dB"
          % (np.percentile(pred, 50), np.percentile(pred, 90), pred.max()))
    print("\n  MAE %5.3f dB   median |err| %5.3f   within %.1f dB %5.1f%%"
          % (np.abs(err).mean(), np.median(np.abs(err)), TOL,
             100 * np.mean(np.abs(err) <= TOL)))
    print("  BIAS (predicted - measured)  mean %+5.3f  median %+5.3f dB"
          % (err.mean(), np.median(err)))
    print("  distance to nearest corpus design  median %.3f  max %.3f"
          % (np.median(dist), dist.max()))

    # A mean bias near zero can still hide the failure that matters here: systematically
    # pulling the HIGH designs down toward the corpus mean while leaving the low ones
    # alone. Split by measured boost to see it.
    print("\n  bias by measured boost -- the tell for regression toward the corpus mean")
    edges = [0, 2, 4, 6, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (act >= lo) & (act < hi)
        if m.sum() < 3:
            continue
        print("    measured in [%3.0f, %3.0f)  n=%3d   mean bias %+6.3f dB   MAE %5.3f"
              % (lo, hi, m.sum(), err[m].mean(), np.abs(err[m]).mean()))

    hi = act >= 6
    if hi.sum() >= 3:
        print("\n  On the %d designs measured at >= 6 dB -- the region the targets live in --"
              % hi.sum())
        print("  the surrogate predicts a mean of %.2f dB against a measured mean of %.2f."
              % (pred[hi].mean(), act[hi].mean()))


if __name__ == "__main__":
    main()
