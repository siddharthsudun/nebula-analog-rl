"""Are the requested targets inside the boost this circuit can actually produce?

THE QUESTION THIS ANSWERS. `sigma_derivation` found that no sampling radius on the grid
lets one CMA-ES generation close the median PPO handoff gap, and `reach_ceiling` showed
that plateau is real rather than surrogate smoothing (spread ratio 0.945-0.984 at every
distance). That points somewhere other than the refiner. The specs are drawn
`uniform(5, 11)` dB of boost, and the recorded corpus has a 99th percentile of 7.63 dB.
If the target distribution sits above the achievable one, then most of the "gap" the
refiner is being asked to close does not close for anyone, at any radius, with any method.

TWO INDEPENDENT ESTIMATES OF ACHIEVABLE BOOST, because the obvious one is biased:

  CORPUS         374k recorded SPICE runs. BIASED: it is mostly PPO trajectories, so it
                 reports where the policy went, not what the circuit can do.
  UNIFORM SAMPLE `results/feasibility_band.json` -- 2000 designs drawn uniformly at
                 random, 460 guard-valid, every one a real SPICE run. UNBIASED by
                 policy visitation. This is the estimate that answers the question.

If both put the target distribution in the far upper tail, the finding is about the
BENCHMARK, not the arm -- and the target distribution is not something to change here.

No SPICE. No policy. Reads two existing artifacts.
"""

from __future__ import annotations

# =============================================================================
# VOID ANALYSIS -- DO NOT USE, DO NOT CITE, DO NOT RUN
# =============================================================================
# This script ran against the pre-fix surrogate corpus, whose `boost_db` labels were
# the REAL PART of the complex transfer function rather than 20*log10|H|. See
# docs/REPRODUCE.md section 19. Every number it produced is void, including:
#
#   - the claim that 56-66% of targets sit above the corpus p99
#   - the per-spec matched chance probabilities computed on the corpus
#   - NOTE: this file's UNIFORM-SAMPLE rows came from real SPICE and were sound; they are void here only because the artifact interleaved them with corpus rows
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

import json

import numpy as np

from silq.experiments.target_audit import make_specs
from silq.surrogate import METRICS, load_corpus

QS = (50, 75, 90, 95, 99, 100)

def line(tag: str, b: np.ndarray) -> None:
    print("  %-30s n=%6d  " % (tag, len(b))
          + "  ".join("p%-3d %5.2f" % (q, np.percentile(b, q)) for q in QS))

def main() -> None:
    print("ACHIEVABLE BOOST (dB)\n")
    corpus = load_corpus("results/surrogate_corpus.npz")["Y"][:, METRICS.index("boost_db")]
    line("corpus (PPO-biased)", corpus)

    fb = json.load(open("results/feasibility_band.json"))
    uni = np.array([r["boost_db"] for r in fb["scatter"]])
    line("uniform sample, guard-valid", uni)
    allpass = np.array([r["boost_db"] for r in fb["scatter"] if r.get("all_pass")])
    if len(allpass):
        line("uniform sample, all-pass", allpass)

    print("\nREQUESTED TARGETS  uniform(5, 11) dB -- target_audit.make_specs\n")
    ref = {"corpus": corpus, "uniform guard-valid": uni}
    for seed in (0, 1, 2, 3):
        t = np.array([tg for tg, _ in make_specs(32, seed)])
        parts = []
        for name, b in ref.items():
            parts.append("%3.0f%% above %s p99" % (100 * np.mean(t > np.percentile(b, 99)),
                                                   name))
        print("  spec seed %d  targets p50 %5.2f  |  %s" % (seed, np.median(t),
                                                            "   ".join(parts)))

    # The strict criterion is +/-1.5 dB, so what matters is not whether the target is
    # exceeded but whether ANY recorded design lands inside the tolerance band around it.
    print("\nFRACTION OF RECORDED DESIGNS INSIDE THE +/-1.5 dB STRICT BAND OF EACH TARGET")
    print("  (this is the per-spec matched chance probability p, computed on each source)")
    for seed in (2, 3):
        t = np.array([tg for tg, _ in make_specs(32, seed)])
        for name, b in ref.items():
            p = np.array([np.mean(np.abs(b - tg) <= 1.5) for tg in t])
            print("  seed %d  %-22s p: median %.4f  min %.4f  max %.4f  "
                  "| %d of 32 specs have p < 0.01"
                  % (seed, name, np.median(p), p.min(), p.max(), int((p < 0.01).sum())))

if __name__ == "__main__":
    main()
