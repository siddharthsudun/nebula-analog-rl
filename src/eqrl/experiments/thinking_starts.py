"""Diverse starting designs for Thinking mode, proposed by the surrogate corpus.

WHY THIS EXISTS, AND WHY IT IS NOT WHAT THE SPEC LITERALLY ASKED FOR.
`docs/DESIGN_MODES_V2.md` section 3 item 2 says "screen restarts with the surrogate
instead of blind seed offsets". Taken literally that is impossible: a Thinking restart is
a PPO rollout seeded at `1000 + spec_index`, and the surrogate ranks *designs*, not RNG
seeds -- there is no way to know what a seed will produce without paying the five real
evaluations that running it costs. Ranking seeds would be ranking noise.

What the surrogate CAN do is propose starting DESIGNS, which is exactly what
`fastest_hedge.propose_seed` already does for Fastest. So a screened restart here is a
corpus-proposed start rather than a screened PPO seed: one real guarded evaluation instead
of PPO's five, at a point the corpus has ground-truth reason to believe is near target.

WHY DIVERSITY IS THE POINT AND NEAREST-TO-TARGET IS NOT. `propose_seed` returns the single
closest corpus row, so calling it n times returns the same design n times. A restart that
starts where the last one started explores nothing. The n=32 measurement in section 6 of
that doc is what motivates this: 8 of 32 specs were unsolved because stage 1 and the
repair ladder never reached the feasible set at all -- 6 "never reached the feasible set",
2 "no guard-valid design at all". Those are start-quality failures, and more budget spent
from the same bad start does not fix a bad start. So the candidates below are chosen to be
mutually far apart in the normalized action space, subject to being plausible.

THE CORPUS STILL NEVER JUDGES. Every design proposed here is evaluated by the same
guarded, independent evaluator every other mode uses, and can be rejected by it. The
corpus decides only where to look. `eqrl.surrogate`: "IT IS A RANKER, NEVER A JUDGE."
"""
from __future__ import annotations

import numpy as np


def plausible_mask(surrogate, spec) -> np.ndarray:
    """Corpus rows the surrogate has GROUND TRUTH reason to think are worth evaluating.

    Only the three AC metrics the corpus actually records are used -- `dc_gain_db`,
    `boost_db`, `peak_freq_ghz` -- and they are the row's own measured values from a SPICE
    run already paid for, not a kNN prediction. The other `loose_pass` checks (eye height
    and width, noise, HD3, power, area) are channel- and guard-dependent, the corpus does
    not carry them, and that is precisely why each proposal below still costs one real
    evaluation. Degrades to the whole corpus rather than raising if the band is too tight
    to admit anything -- a wider candidate set is strictly better than none.
    """
    Y = surrogate.Y
    dc_gain, peak = Y[:, 0], Y[:, 2]
    mask = (dc_gain >= spec.dc_gain_db_min) & (peak >= spec.peak_freq_lo_ghz) & (
        peak <= spec.peak_freq_hi_ghz)
    return mask if mask.any() else np.ones(len(Y), dtype=bool)


def diverse_seeds(target: float, surrogate, spec, n: int, *, pool: int = 512,
                  exclude: np.ndarray | None = None,
                  min_separation: float = 0.15) -> list[np.ndarray]:
    """Up to `n` starting designs: near `target` in boost, far apart from each other.

    Two-step, and both steps matter. First take the `pool` plausible rows whose OWN
    recorded boost is nearest `target` -- that is the ranking, and it is the one question
    the corpus's `Y` genuinely answers. Then pick from that pool by greedy farthest-point
    in the normalized action space, which is what makes these restarts rather than
    duplicates.

    `min_separation` is a floor on that farthest-point distance, so a corpus that is dense
    in one region returns FEWER seeds instead of n near-identical ones. Returning three
    real restarts is honest; returning eight copies of one design and calling it eight
    restarts is not. `exclude` seeds the distance computation with designs already tried
    (the PPO rollouts' own starts), so a corpus proposal is not allowed to duplicate a
    start this run has already paid for.

    The returned vectors are rows of `surrogate.X`, i.e. already in the normalized action
    space `evaluate` expects -- the same representation `fastest_hedge.propose_seed`
    returns.
    """
    if n <= 0:
        return []
    X, Y = surrogate.X, surrogate.Y
    mask = plausible_mask(surrogate, spec)
    err = np.where(mask, np.abs(Y[:, 1] - target), np.inf)
    order = np.argsort(err)[:pool]
    order = order[np.isfinite(err[order])]
    if order.size == 0:
        return []

    cand = X[order]
    chosen: list[np.ndarray] = []
    # Distance to the nearest ALREADY-COMMITTED design: the excluded starts first, then
    # each pick as it is made. Starting from the excluded set is what stops seed 1 from
    # landing on top of the PPO start.
    if exclude is not None and len(exclude):
        ex = np.atleast_2d(np.asarray(exclude, dtype=np.float64))
        dist = np.linalg.norm(cand[:, None, :] - ex[None, :, :], axis=2).min(axis=1)
    else:
        dist = np.full(len(cand), np.inf)

    for _ in range(n):
        j = int(np.argmax(dist))
        if not np.isfinite(dist[j]):
            j = 0                      # nothing excluded yet: take the nearest-to-target
        elif dist[j] < min_separation:
            break                      # everything left is a near-duplicate; stop early
        chosen.append(np.asarray(cand[j], dtype=np.float64))
        dist = np.minimum(dist, np.linalg.norm(cand - cand[j], axis=1))
    return chosen
