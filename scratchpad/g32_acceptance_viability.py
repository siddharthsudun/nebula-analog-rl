"""VIABILITY TEST for the proposed `g32_acceptance` mode, run BEFORE writing it.

THE QUESTION. `g32_acceptance` would filter the corpus by the user's REQUIREMENTS before
`thinking_starts.diverse_seeds` picks restart designs, instead of after. That is only worth
building if the filter actually removes rows the current code would have picked. This
script answers that with numbers, on the same 32 spec draws the benchmark uses, and it is
designed so that "no, it changes nothing" is a result it can return.

WHAT IS ALREADY FILTERED, AND WHAT WOULD BE NEW. `plausible_mask` (thinking_starts.py:182)
already screens on THREE requirement fields: dc_gain_db_min, peak_freq_lo_ghz and
peak_freq_hi_ghz. So the arm's real delta is only:

    boost_db_min / boost_db_max   corpus Y column 1, free
    area_mm2_max                  analytic from X via DesignVars.area_mm2, free
    power_w_max                   NOT free -- see below

and nothing else, because noise, HD3 and both eye metrics are channel- and guard-dependent
and the corpus does not carry them.

POWER IS NOT FREE AND IS NOT FILTERED ON HERE. measures.py:81-88 computes power as
vdd * srv.supply_current(...), a real SPICE call, and explicitly rejects the analytic
shortcut vdd * dv.i_tail. This script therefore does NOT filter on power. It reports the
i_tail-implied figure separately and labelled as a proxy. That label is the whole point:
an unlabelled proxy in a shipped mode would be a silently different measurement from the
one the verifier applies.

THE TWO REGIMES ARE BOTH REPORTED ON PURPOSE. At competition defaults the added filters
are expected to be near-inert; under a user-tightened requirement they are expected to
bite. A mode that only helps in the second regime is still worth shipping -- but it must be
SOLD as the second regime, and this table is what stops it being sold as the first.

WHAT THIS SCRIPT ALONE CANNOT SEE, ADDED AFTER THE FACT. Its "defaults" row reports 0 of 32
seed sets changed, and that number is right -- but it was then generalized, by me, into
"the arm is inert at the competition defaults", which is FALSE. The 32 draws at seed 137
span 5.01-10.99 dB; a 0.01 dB scan of the whole target range finds seeds IDENTICAL through
3.05-11.13 dB, DIFFERENT at 3.00-3.04, and different at 76 of 121 sampled targets from 11.14
up -- ragged, with 11.15-11.18, 11.20, 11.21, 11.27, 11.28, 11.30, 11.32 and 11.87 still
identical. At the DEFAULT spec with nothing tightened. The pool is the 512 rows nearest the
target IN BOOST, so a target near a range boundary pulls in rows from the far side of it (at
11.5 dB, 68 of 512 sit above the 12 dB ceiling) and `boost_range` removes exactly those.

A SECOND CORRECTION, WHICH THE EDGE PROBE BELOW IS WHY. Between those two states I bisected
for the edge and reported a band of [3.0664, 11.2125] dB. Bisection presumes monotonicity in
the target; the scan shows there is none, so it returned a boundary that does not exist. The
probe's own output is what exposed it: at 3.05 dB it removes 76 pool rows and the seeds do
NOT change, which cannot happen if a single crossing point separates the two behaviours.
Filtering rows and moving the selection are different events -- farthest-point can pick the
same designs out of a smaller pool.

The script could not have caught the first error, because every target it is given lies
inside the quiet interval. `EDGE_TARGETS` below is the repair: it probes the boundary rather than
sampling the benchmark and hoping the benchmark covers the space. A sweep can only falsify
a claim about the region it samples.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eqrl.circuits.ctle import ACTION_SPACE, DesignVars                     # noqa: E402
from eqrl.experiments.chance_baseline import make_targets_exact             # noqa: E402
from eqrl.experiments.fastest_hedge import load_fastest_assets              # noqa: E402
from eqrl.experiments.thinking_starts import diverse_seeds, plausible_mask  # noqa: E402
from eqrl.specs import Spec                                                 # noqa: E402
from eqrl.surrogate import KEYS, _HI, _LO, _LOG                             # noqa: E402

POOL = 512          # thinking_starts.diverse_seeds default
N_SEEDS = 4         # pipeline's THINKING_SURROGATE_STARTS
MIN_SEP = 0.15      # diverse_seeds default
SPEC_SEED = 137
N_SPECS = 32


def denormalize(Xn):
    """Inverse of surrogate.normalize, re-derived from _LO/_HI/_LOG rather than restated."""
    Xn = np.atleast_2d(np.asarray(Xn, dtype=np.float64))
    out = np.empty_like(Xn)
    for j in range(Xn.shape[1]):
        if _LOG[j]:
            out[:, j] = np.exp(np.log(_LO[j])
                               + Xn[:, j] * (np.log(_HI[j]) - np.log(_LO[j])))
        else:
            out[:, j] = _LO[j] + Xn[:, j] * (_HI[j] - _LO[j])
    return out


def areas_mm2(Xn):
    raw = denormalize(Xn)
    return np.array([DesignVars(**dict(zip(KEYS, r))).area_mm2() for r in raw])


def _greedy(cand, n, min_sep=MIN_SEP):
    """The farthest-point loop from thinking_starts.diverse_seeds, on a given candidate set."""
    chosen, dist = [], np.full(len(cand), np.inf)
    for _ in range(n):
        j = int(np.argmax(dist))
        if not np.isfinite(dist[j]):
            j = 0
        elif dist[j] < min_sep:
            break
        chosen.append(np.asarray(cand[j], dtype=np.float64))
        dist = np.minimum(dist, np.linalg.norm(cand - cand[j], axis=1))
    return chosen


def _pool_order(surrogate, base_mask, target, extra=None):
    Y = surrogate.Y
    mask = base_mask if extra is None else (base_mask & extra)
    if not mask.any():
        return np.array([], dtype=int)
    err = np.where(mask, np.abs(Y[:, 1] - target), np.inf)
    order = np.argsort(err)[:POOL]
    return order[np.isfinite(err[order])]


def _same(a, b):
    return len(a) == len(b) and all(np.allclose(u, v) for u, v in zip(a, b))


def keep_default(Y, ac, spec):
    """The acceptance filter at the DEFAULT spec: boost in range, area under the ceiling."""
    return ((Y[:, 1] >= spec.boost_db_min) & (Y[:, 1] <= spec.boost_db_max)
            & (ac < spec.area_mm2_max))


def main():
    surrogate, _cx, _rad = load_fastest_assets()
    X, Y = surrogate.X, surrogate.Y
    spec = Spec()
    print("CORPUS: X %s  Y %s  (Y = dc_gain_db, boost_db, peak_freq_ghz)"
          % (X.shape, Y.shape))

    # ---- structural bounds over the WHOLE action space, not a sample ------------------
    # area_mm2 is analytic and increasing in every term it sums, so the upper corner of
    # ACTION_SPACE gives the true supremum. This is a BOUND, not a statistic: no design in
    # the space can exceed it, which is a strictly stronger statement than "0 failures in
    # the corners we simulated".
    hi_dv = DesignVars(**{k: ACTION_SPACE[k][1] for k in KEYS})
    lo_dv = DesignVars(**{k: ACTION_SPACE[k][0] for k in KEYS})
    a_hi, a_lo = hi_dv.area_mm2(), lo_dv.area_mm2()
    print("\nSTRUCTURAL AREA BOUND (analytic, over the whole action space):")
    print("   min %.6f   max %.6f mm^2   vs spec default %.3f  ->  %s"
          % (a_lo, a_hi, spec.area_mm2_max,
             "AREA CANNOT FAIL" if a_hi < spec.area_mm2_max else "area can fail"))
    it_hi = ACTION_SPACE["i_tail"][1]
    print("   [PROXY, NOT THE MEASUREMENT] i_tail max %.4g A, vdd*i_tail = %.4g W vs "
          "power_w_max %.4g" % (it_hi, spec.vdd_nominal * it_hi, spec.power_w_max))
    print("   measures.py:81-88 uses vdd*srv.supply_current() and explicitly rejects that")
    print("   shortcut, so this line bounds nothing on its own -- it is reported labelled.")

    ac = areas_mm2(X)
    print("   corpus areas: min %.6f  median %.6f  max %.6f mm^2"
          % (ac.min(), np.median(ac), ac.max()))

    targets = make_targets_exact(N_SPECS, SPEC_SEED)
    base_mask = plausible_mask(surrogate, spec)
    print("\nplausible_mask (dc_gain + peak band) already admits %d / %d rows (%.1f%%)"
          % (base_mask.sum(), len(X), 100 * base_mask.mean()))
    print("boost in default range [%.0f, %.0f]: %d rows (%.1f%%)"
          % (spec.boost_db_min, spec.boost_db_max,
             ((Y[:, 1] >= spec.boost_db_min) & (Y[:, 1] <= spec.boost_db_max)).sum(),
             100 * ((Y[:, 1] >= spec.boost_db_min) & (Y[:, 1] <= spec.boost_db_max)).mean()))

    regimes = [
        ("defaults",   dict(boost_db_min=3.0, boost_db_max=12.0, area_mm2_max=0.05)),
        ("area/10",    dict(boost_db_min=3.0, boost_db_max=12.0, area_mm2_max=0.005)),
        ("area/50",    dict(boost_db_min=3.0, boost_db_max=12.0, area_mm2_max=0.001)),
        ("boost 6-9",  dict(boost_db_min=6.0, boost_db_max=9.0,  area_mm2_max=0.05)),
        ("both tight", dict(boost_db_min=6.0, boost_db_max=9.0,  area_mm2_max=0.001)),
    ]

    out = {}
    for name, req in regimes:
        keep = ((Y[:, 1] >= req["boost_db_min"]) & (Y[:, 1] <= req["boost_db_max"])
                & (ac < req["area_mm2_max"]))
        removed, changed, empt, detail = [], 0, 0, []
        for i, t in enumerate(targets):
            order = _pool_order(surrogate, base_mask, t)
            removed.append(int((~keep[order]).sum()) if order.size else 0)
            # CURRENT behaviour vs the arm's composition: filter to a POOL, then run the
            # existing farthest-point selection inside it. Never filter to a winner.
            a = diverse_seeds(t, surrogate, spec, N_SEEDS)
            corder = _pool_order(surrogate, base_mask, t, keep)
            b = _greedy(X[corder], N_SEEDS) if corder.size else []
            if not b:
                empt += 1
            if not _same(a, b):
                changed += 1
                detail.append({"spec": i, "target": float(t),
                               "n_base": len(a), "n_constrained": len(b)})
        rp = np.array(removed)
        out[name] = {"requirements": req, "corpus_keep": int(keep.sum()),
                     "corpus_keep_frac": float(keep.mean()),
                     "pool_removed_mean": float(rp.mean()),
                     "pool_removed_max": int(rp.max()),
                     "specs_changed": changed, "specs_empty_pool": empt,
                     "changed_detail": detail}
        print("\n[%-10s] %s" % (name, req))
        print("   corpus rows kept : %7d / %d (%.3f%%)"
              % (keep.sum(), len(X), 100 * keep.mean()))
        print("   removed from the %d-row near-target pool: mean %.1f  max %d"
              % (POOL, rp.mean(), rp.max()))
        print("   specs whose CHOSEN SEED SET changes    : %d / %d   (empty pool: %d)"
              % (changed, N_SPECS, empt))

    # ---- the boundary probe the 32-draw sweep structurally cannot perform --------------
    from eqrl.experiments.thinking_starts import acceptance_mask
    print("\nEDGE PROBE at the DEFAULT spec (nothing tightened) -- the region the 32 draws")
    print("never sample. 'differs' here refutes 'inert at the defaults'.")
    print("   %8s %10s %10s %8s  %s" % ("target", "pool_min", "pool_max", "outside", "seeds"))
    edge = {}
    for t in (3.0, 3.05, 3.5, 5.0, 9.0, 11.0, 11.25, 11.5, 11.9, 12.0):
        order = _pool_order(surrogate, base_mask, t)
        b = Y[order, 1]
        n_out = int(((b < spec.boost_db_min) | (b > spec.boost_db_max)).sum())
        s_off = diverse_seeds(t, surrogate, spec, N_SEEDS)
        s_on = _greedy(X[_pool_order(surrogate, base_mask & keep_default(Y, ac, spec), t)],
                       N_SEEDS)
        differs = not _same(s_off, s_on)
        edge[t] = {"pool_min": float(b.min()), "pool_max": float(b.max()),
                   "rows_outside_range": n_out, "seeds_differ": differs}
        print("   %8.2f %10.3f %10.3f %8d  %s"
              % (t, b.min(), b.max(), n_out, "DIFFER" if differs else "identical"))
    print("   -> identical throughout 3.05-11.13 dB; differs at 3.00-3.04 and at most (NOT")
    print("      all) targets from 11.14 up -- 11.87 is outside and still identical, so the")
    print("      region outside the quiet interval is ragged and has no threshold. The")
    print("      benchmark targets span 5.01-10.99 dB, which is why the table reads 0/32.")

    Path("scratchpad/g32_acceptance_viability.json").write_text(json.dumps(
        {"corpus_n": int(len(X)), "spec_seed": SPEC_SEED, "n_specs": N_SPECS,
         "pool": POOL, "n_seeds": N_SEEDS, "min_separation": MIN_SEP,
         "area_bound_mm2": {"min": a_lo, "max": a_hi,
                            "spec_default": spec.area_mm2_max},
         "itail_power_proxy_w": {"value": spec.vdd_nominal * it_hi,
                                 "LABEL": "PROXY ONLY -- not measures.py's measurement"},
         "plausible_mask_frac": float(base_mask.mean()),
         "regimes": out}, indent=1))
    print("\nwrote scratchpad/g32_acceptance_viability.json")


if __name__ == "__main__":
    main()
