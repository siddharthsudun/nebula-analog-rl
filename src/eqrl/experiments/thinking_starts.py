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

#: Cache for `corpus_areas_mm2`, keyed by `id(surrogate.X)`. The area of a corpus row never
#: changes -- it is a pure function of the design -- and computing it for 374k rows is the
#: only part of acceptance filtering that is not already a numpy comparison.
#:
#: THE VALUE IS `(Xn, area)`, NOT `area`, AND THE FIRST ELEMENT IS LOAD-BEARING. An `id()`
#: key is only stable while the object it names is alive; CPython reuses the address of a
#: freed object. Storing the array alongside its own key makes the key unfreeable, so a
#: stale hit is impossible by construction rather than by luck. It is luck today: the only
#: thing keeping `surrogate.X` alive is `fastest_hedge._ASSET_CACHE`, a process-lifetime
#: dict in another module that was not written to underwrite this invariant, and that would
#: stop underwriting it the moment it gained eviction or an LRU bound.
#:
#: WHY THAT MATTERED MORE THAN A NORMAL CACHE BUG: the proof in `corpus_areas_mm2` that the
#: vectorized area matches `DesignVars.area_mm2` runs BELOW the early return. A cache hit
#: skips it. So the single scenario the assertion exists to catch -- this function returning
#: areas that belong to a different set of designs -- was the one scenario in which the
#: assertion could not fire. Found by silq-main re-deriving the module at HEAD, 07 Sep 2026.
#:
#: THE COST, STATED RATHER THAN GLOSSED: pinning means an entry retains its corpus `X` for
#: the process lifetime, so a caller that loads corpora transiently now leaks ~18 MB each
#: (374588 x 6 float64) instead of ~3 MB. On the real path that is zero extra retention --
#: `_ASSET_CACHE` already holds the same array for the process lifetime, and this entry
#: holds a second reference to it, not a second copy. A bounded cache trading a correctness
#: hazard for memory would be the wrong trade here; there is one corpus.
_AREA_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def corpus_areas_mm2(surrogate) -> np.ndarray:
    """Analytic area for every corpus row, vectorized, PROVEN equal to the scalar model.

    WHY VECTORIZED AND WHY PROVEN. `DesignVars.area_mm2` is the authority, but calling it
    once per row builds 374k dataclasses on every design() call. So the arithmetic is
    re-derived in numpy here -- and then checked against the authority on a sample, raising
    rather than warning, exactly as `surrogate.assert_encoding_matches_ctle` does for the
    coordinate system. A silently divergent area model would filter out the wrong corpus
    rows and there is no downstream metric that would reveal it.

    Area is analytic and costs no simulation, which is what makes it usable as a FILTER at
    all. Power is not: `sim.measures` computes it as vdd * supply_current from a real SPICE
    run and explicitly rejects the vdd * i_tail shortcut, so power is deliberately absent
    from this module. An unlabelled proxy would be a different measurement from the one the
    verifier applies.
    """
    key = id(surrogate.X)
    hit = _AREA_CACHE.get(key)
    if hit is not None:
        # The cached Xn IS the pinned key: identical by construction, not merely equal.
        assert hit[0] is surrogate.X
        return hit[1]

    from eqrl.circuits.ctle import (ACTION_SPACE, MIRROR_DENSITY_A_PER_UM, MIRROR_L_UM,
                                    MIRROR_W_MAX_UM, MIRROR_W_MIN_UM, DesignVars)
    from eqrl.surrogate import KEYS, _HI, _LO, _LOG

    Xn = surrogate.X
    raw = np.empty_like(Xn)
    for j in range(Xn.shape[1]):
        if _LOG[j]:
            raw[:, j] = np.exp(np.log(_LO[j])
                               + Xn[:, j] * (np.log(_HI[j]) - np.log(_LO[j])))
        else:
            raw[:, j] = _LO[j] + Xn[:, j] * (_HI[j] - _LO[j])
    col = {k: raw[:, i] for i, k in enumerate(KEYS)}

    # ctle.DesignVars.area_mm2's own terms, in the same order, with mirror_sizing's
    # max/ceil written as their numpy equivalents.
    MIM = 2e-15 / 1e-12
    total_um = np.maximum((col["i_tail"] / 2.0) / MIRROR_DENSITY_A_PER_UM, MIRROR_W_MIN_UM)
    fingers = np.maximum(1, np.ceil(total_um / MIRROR_W_MAX_UM))
    wb_um = total_um / fingers
    area = ((2 * col["w_in"] * col["l_in"])
            + 3 * (wb_um * fingers * 1e-6) * (MIRROR_L_UM * 1e-6)
            + col["cs"] / MIM
            + ((col["rs"] + 2 * col["r_load"]) / 320.0) * (1e-6) ** 2
            + 500e-12) * 1e6

    # Prove it, do not assume it. Bitwise-close on a spread-out sample of real corpus rows.
    idx = np.linspace(0, len(Xn) - 1, min(512, len(Xn))).astype(int)
    truth = np.array([DesignVars(**{k: col[k][i] for k in KEYS}).area_mm2() for i in idx])
    if not np.allclose(area[idx], truth, rtol=1e-12, atol=0.0):
        worst = float(np.abs(area[idx] - truth).max())
        raise AssertionError(
            "thinking_starts.corpus_areas_mm2 disagrees with ctle.DesignVars.area_mm2 by "
            f"up to {worst:.3e} mm^2 on {len(idx)} sampled corpus rows. The vectorized "
            "copy has drifted from the authority; re-derive it rather than loosening this "
            "check, because a wrong area model filters the wrong rows invisibly.")
    _ = ACTION_SPACE           # imported to document where the ranges this decodes live
    _AREA_CACHE[key] = (Xn, area)   # tuple pins the key alive; see _AREA_CACHE's comment
    return area


def acceptance_mask(surrogate, spec) -> np.ndarray:
    """`plausible_mask` PLUS the two acceptance checks the corpus can also answer.

    WHAT THIS ADDS, AND IT IS ONLY TWO THINGS. `plausible_mask` already screens
    `dc_gain_db_min` and the peak-frequency band. Of the eleven `REQUIREMENT_FIELDS`, the
    corpus can additionally answer `boost_db_min`/`boost_db_max` (its own recorded boost)
    and `area_mm2_max` (analytic). The remaining checks -- noise, HD3, both eye metrics,
    and power -- are channel- or guard-dependent, or cost a SPICE run, and are NOT
    approximated here.

    IT IS INERT ON THE BENCHMARK, WHICH IS NOT THE SAME AS INERT AT THE DEFAULTS.
    `scratchpad/g32_acceptance_viability.py` scores it on the 32 spec draws at seed 137: at
    the default spec it removes ZERO rows from the 512-row near-target pool and changes ZERO
    of 32 selected seed sets -- bit-identical, not a small effect. `area` cannot bite at all
    at the default ceiling, because the analytic supremum over the WHOLE action space is
    0.002227 mm^2 against a 0.05 mm^2 budget.

    BUT `boost_range` CAN BITE AT THE DEFAULTS, NEAR THE EDGES, and an earlier draft of this
    docstring wrongly generalized the 0/32 to "inert at the defaults" full stop. The pool is
    the 512 rows nearest the target IN BOOST, so a target near a range boundary pulls in rows
    from the far side of it -- at an 11.5 dB target, 68 of the 512 sit above the 12 dB
    ceiling -- and this mask removes exactly those.

    SCANNED, NOT BISECTED, because the effect is NOT a clean threshold and a bisection
    reported one that does not exist. At 0.01 dB resolution:

        3.00-3.04 dB    differs (5 of 61 sampled)
        3.05-11.13 dB   IDENTICAL everywhere (61 + 144 samples, no exceptions)
        11.14-12.00 dB  differs at 76 of 121 samples -- ragged, not contiguous: 11.15-11.18,
                        11.20, 11.21, 11.27, 11.28, 11.30, 11.32 and 11.87 are identical

    The raggedness is the honest part. Whether a removed row CHANGES THE SELECTION depends
    on whether farthest-point would have picked it, which is local corpus geometry -- so
    "rows were removed" does not imply "the seeds moved". At a 3.05 dB target the mask drops
    76 pool rows and the chosen seeds are unchanged. Quoting a threshold would imply a
    monotonicity that was measured not to hold.

    The benchmark's 32 draws span 5.01-10.99 dB, wholly inside the quiet band, which is
    precisely why the sweep could not see any of this.

    So at the defaults this is a no-op ON THE BENCHMARK, and near the edges of the boost
    range it is a small correctness improvement: it stops restarts being proposed from
    designs whose own recorded boost is outside the range the run will accept. It acts more
    broadly when a user tightens the boost range, or pushes the area ceiling below the
    corpus median of 0.00103 mm^2.

    Degrades to `plausible_mask` rather than returning nothing if the tightened band admits
    no row, for the same reason `plausible_mask` degrades to the whole corpus: a wider
    candidate set is strictly better than none, and the guarded evaluator still judges every
    proposal.
    """
    base = plausible_mask(surrogate, spec)
    Y = surrogate.Y
    mask = base & (Y[:, 1] >= spec.boost_db_min) & (Y[:, 1] <= spec.boost_db_max)
    if spec.area_mm2_max is not None:
        mask = mask & (corpus_areas_mm2(surrogate) < spec.area_mm2_max)
    return mask if mask.any() else base


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
                  min_separation: float = 0.15,
                  acceptance: bool = False) -> list[np.ndarray]:
    """Up to `n` starting designs: near `target` in boost, far apart from each other.

    `acceptance` is the `g32_acceptance` mode's only hook, and it is a CALL PARAMETER that
    defaults to False -- never a module-global that a caller rebinds. The web server runs
    one process across concurrent sessions, so a global would let one user's tightened
    requirements silently select another user's restarts. Off, this function is
    byte-identical to its pre-mode behaviour.

    When on, the FILTER APPLIES TO THE POOL AND THE FARTHEST-POINT SELECTION THEN RUNS
    INSIDE IT. That order is the whole design and it is not interchangeable. Filtering to a
    single best-scoring row would discard diversity, and diversity is the demonstrated
    mechanism: the same frozen policy solves 4 of 11 specs at one rollout and 11 of 11 with
    restarts, with zero corpus starts used. An acceptance filter that narrowed the pool
    before diversifying would be trading away the thing that is working in exchange for a
    constraint the guarded evaluator was going to check anyway.

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
    mask = acceptance_mask(surrogate, spec) if acceptance else plausible_mask(surrogate, spec)
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
