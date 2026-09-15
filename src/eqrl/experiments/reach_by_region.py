"""Is the reach plateau a property of the DESIGN SPACE, or of WHERE PPO SITS?

`sigma_derivation` measured reach from centres drawn out of the recorded corpus. The
corpus is mostly PPO trajectories, and `target_reachability` has since shown that those
trajectories occupy a low-boost pocket: recorded boost has median 2.00 dB, while a
uniform guard-valid sample of the same design space has median 1.13 but p75 5.55, p90
9.61 and p95 11.60 dB. So the plateau may simply be that boost is flat WHERE PPO IS, not
that it is flat everywhere.

That distinction decides what the amendment should do. If reach is just as flat from
uniform centres, the refiner is hopeless and sigma is irrelevant. If reach is much larger
from uniform centres, then the binding problem is the HANDOFF LOCATION -- stage 1 delivers
stage 2 into a region where the requested boost cannot be reached -- and no choice of
sigma repairs that either, but for a completely different reason.

Same estimator as `sigma_derivation.reach_curve`, only the centres change.

No SPICE. No policy.
"""

from __future__ import annotations

# =============================================================================
# VOID ANALYSIS -- DO NOT USE, DO NOT CITE, DO NOT RUN
# =============================================================================
# This script ran against the pre-fix surrogate corpus, whose `boost_db` labels were
# the REAL PART of the complex transfer function rather than 20*log10|H|. See
# docs/REPRODUCE.md section 19. Every number it produced is void, including:
#
#   - every median-reach figure, for all three centre sources
#   - the landing probabilities against uniform(5, 11) targets
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

from eqrl.experiments.sigma_derivation import SIGMA_GRID, POPSIZE, reach_curve
from eqrl.surrogate import Surrogate, load_corpus, normalize, KEYS

SHOW = (0.05, 0.08, 0.12, 0.20, 0.30, 0.40, 0.50)

def main() -> None:
    rng = np.random.default_rng(0)
    c = load_corpus("results/surrogate_corpus.npz")
    X, Y = c["X"], c["Y"]
    sur = Surrogate(X, Y, k=5)

    fb = json.load(open("results/feasibility_band.json"))
    # Not every scatter record carries its design; those are dropped rather than
    # reconstructed, since a guessed design would put a centre somewhere nothing was
    # actually simulated.
    uni_valid = normalize(np.array([[r["design"][k] for k in KEYS]
                                    for r in fb["scatter"] if "design" in r]))

    sources = {
        "corpus (where PPO sits)": X[rng.choice(len(X), 3000, replace=False)],
        "uniform, guard-valid": uni_valid,
        "uniform over the cube": rng.random((3000, X.shape[1])),
    }

    base_all = {n: sur.predict_boost(v)[0] for n, v in sources.items()}
    print("STARTING-POINT BOOST by centre source")
    for n, b in base_all.items():
        print("  %-26s n=%5d  p25 %5.2f  p50 %5.2f  p75 %5.2f  p90 %5.2f dB"
              % (n, len(b), *[np.percentile(b, q) for q in (25, 50, 75, 90)]))

    print("\nMEDIAN REACH of one unscreened %d-draw generation (dB)" % POPSIZE)
    print("  sigma  " + "".join("%-26s" % n for n in sources))
    curves = {n: reach_curve(sur, v, np.array(SHOW), POPSIZE, rng)
              for n, v in sources.items()}
    for s in SHOW:
        row = "".join("%-26.2f" % np.percentile(curves[n][float(s)], 50) for n in sources)
        print("  %.2f   %s%s" % (s, row, "   <- AS REGISTERED" if s == 0.05 else ""))

    # The number that actually matters: from each centre, can ONE generation put boost
    # inside the +/-1.5 dB strict band of a target drawn from the benchmark's own
    # uniform(5, 11)? Reach in the abstract is not the question; landing on target is.
    print("\nP(one generation lands within 1.5 dB of a uniform(5, 11) target)")
    tg = rng.uniform(5, 11, 4000)
    for s in SHOW:
        parts = []
        for n, v in sources.items():
            d = v.shape[1]
            draws = np.clip(v[:, None, :] + s * rng.standard_normal((len(v), POPSIZE, d)),
                            0.0, 1.0)
            pred, _ = sur.predict_boost(draws.reshape(-1, d))
            pred = pred.reshape(len(v), POPSIZE)
            t = tg[:len(v)][:, None]
            parts.append("%-26.3f" % np.mean((np.abs(pred - t) <= 1.5).any(axis=1)))
        print("  %.2f   %s" % (s, "".join(parts)))

if __name__ == "__main__":
    main()
