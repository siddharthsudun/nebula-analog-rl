"""How far can boost move inside one CMA-ES step of sigma0 = 0.05?

No SPICE, no policy, no protocol constant is touched. This only measures a PROPERTY OF
THE RECORDED CORPUS: within the ball CMA-ES actually samples on generation 1, how much
does achieved boost vary? If the answer is a fraction of a dB, then r = 10 evaluations at
sigma0 = 0.05 cannot close a 1.7 dB gap no matter how good the ranker is, and the H1/H2
comparison would be measuring screening inside a region with nothing to find.
"""
# =============================================================================
# VOID ANALYSIS -- DO NOT USE, DO NOT CITE, DO NOT RUN
# =============================================================================
# This script ran against the pre-fix surrogate corpus, whose `boost_db` labels were
# the REAL PART of the complex transfer function rather than 20*log10|H|. See
# docs/REPRODUCE.md section 19. Every number it produced is void, including:
#
#   - the 0.54 dB median reach within one sigma0*sqrt(6) ball
#   - the claim that boost is flat inside a local neighbourhood
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
from scipy.spatial import cKDTree
from eqrl.surrogate import load_corpus, METRICS

c = load_corpus("results/surrogate_corpus.npz")
X, Y = c["X"], c["Y"][:, METRICS.index("boost_db")]
tree = cKDTree(X)
rng = np.random.default_rng(0)

# CMA-ES samples x0 + sigma0 * N(0, I) in 6 dims: per-axis sd 0.05, so the expected
# euclidean step is sigma0 * sqrt(6) ~ 0.122, and ~95% of a generation lands inside
# roughly twice that.
for r in (0.061, 0.122, 0.245):
    idx = rng.choice(len(X), 2000, replace=False)
    spread, reach = [], []
    for i in idx:
        nb = tree.query_ball_point(X[i], r)
        if len(nb) < 5:
            continue
        b = Y[nb]
        spread.append(b.std())
        reach.append(np.abs(b - Y[i]).max())
    print("radius %.3f (%.1f x sigma0*sqrt(6))  n=%4d  "
          "median sd of boost in ball %.3f dB   median MAX reach %.3f dB"
          % (r, r / (0.05 * 6 ** 0.5), len(spread),
             float(np.median(spread)), float(np.median(reach))))
