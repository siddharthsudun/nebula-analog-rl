"""Fastest mode's one new mechanism: a single surrogate-guided jump before the frozen G3.2.

WHY THIS IS A NEW FILE AND NOT A CHANGE TO `g32_repair.py` OR `final_comparison.g32_solve`.
Those are transcribed, frozen controllers -- see `final_comparison.py`'s module docstring.
This is a genuinely new mechanism, so it gets its own clearly-labelled file rather than
being folded into either frozen contract. `g32_solve` itself is called UNCHANGED below.

WHY THIS IS NOT H2. `docs/REPRODUCE.md` section 19.6 cut H2 (CMA-ES refinement with
surrogate-ranked candidate pre-screening) because its pre-registered accuracy gate failed
specifically in the sparse decile by distance to the nearest training design -- 81.2% of
held-out designs within 1.5 dB there, against a required 90%, with boost error rising to
0.917 dB MAE in that decile (0.199 dB in the densest). H2 trusted a RANKING over many
un-verified CMA-ES proposals scattered across that sparse region by construction.

This mechanism differs in exactly the ways that remove the failure mode the gate caught:

  1. It proposes exactly ONE candidate, along the single axis G3.2's own probe measured as
     the cleanest boost mover (`plane["boost_axis"]`) -- not a cloud of CMA-ES proposals.
  2. It is accepted only when its nearest-neighbour distance in the corpus is at or below
     the corpus's OWN median self-distance -- the dense half of the space, where accuracy
     sits far closer to the corrected gate's 0.456 dB overall MAE than to the sparse
     decile's 0.917 dB. If nothing on the grid clears that bar, no jump is taken and this
     mode costs exactly what a plain G3.2 call with a small budget would.
  3. Whatever candidate IS proposed is spent on a real evaluation before anything
     downstream trusts it. The surrogate never decides a result -- it only decides where to
     spend one evaluation that `g32_solve` would otherwise have spent on its own first
     bisection step. A bad guess costs exactly that one evaluation; `g32_solve` corrects it
     exactly as it would correct its own first step, because the candidate is folded into
     `s1trace` and picked up by `g32_solve`'s own "already feasible, closest to target"
     scan -- no branch inside `g32_solve` is touched, duplicated, or special-cased.

The corpus is a ranker here in the same sense `docs/REPRODUCE.md` section 16 always meant:
it never appears in a pass/fail decision, and the design it proposes is verified by the
same guarded, independent evaluator every other arm uses.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from eqrl.experiments import final_comparison as fc
from eqrl.experiments.final_comparison import DIMS, PREREG

#: Mirrors `eqrl.surrogate._CORPUS_DEFAULT`. Not imported directly because that name is
#: private to that module; the value is a committed results file, not a tunable.
DEFAULT_CORPUS_PATH = "results/surrogate_corpus.npz"

_ASSET_CACHE: dict[str, tuple[Any, np.ndarray, float]] = {}


def _corpus_safety_radius(corpus_X: np.ndarray) -> float:
    """The corpus's own median nearest-neighbour self-distance.

    Not a hand-picked number: section 19.6 reports accuracy by DECILE of distance to the
    nearest training design, not by a raw distance value, so the cutoff is derived from the
    same corpus at call time. The median self-distance is the boundary of the dense half --
    comfortably inside the deciles the corrected gate measured near its 0.456 dB overall
    MAE rather than the sparse tail's 0.917 dB.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(corpus_X)
    d, _ = tree.query(corpus_X, k=2, workers=-1)
    return float(np.median(d[:, 1]))


def load_fastest_assets(corpus_path: str | None = None) -> tuple[Any, np.ndarray, float]:
    """(surrogate, corpus_X, safety_radius), built once per process and cached.

    Building the kNN tree and the safety radius costs real time over ~375k records, and
    Fastest mode exists to save time -- so this is not repeated on every `design()` call.
    """
    from eqrl.surrogate import Surrogate, load_corpus

    path = corpus_path or DEFAULT_CORPUS_PATH
    if path not in _ASSET_CACHE:
        corpus = load_corpus(path)
        surrogate = Surrogate(corpus["X"], corpus["Y"])
        radius = _corpus_safety_radius(corpus["X"])
        _ASSET_CACHE[path] = (surrogate, corpus["X"], radius)
    return _ASSET_CACHE[path]


def propose_jump(x_f, target: float, plane: dict, surrogate, safety_radius: float,
                 *, step_cap: float = PREREG["step_cap"], grid: int = 41):
    """One candidate design along `plane["boost_axis"]`, or `None` if nothing is safe.

    Grid search, not gradient search: the surrogate is a kNN lookup table, not a
    differentiable model, and a `grid`-point scan of one axis costs nothing next to a
    single SPICE evaluation.
    """
    jb = DIMS.index(plane["boost_axis"])
    x_f = np.asarray(x_f, dtype=np.float64)
    ts = np.linspace(-step_cap, step_cap, grid)
    cand = np.tile(x_f, (grid, 1))
    cand[:, jb] = np.clip(x_f[jb] + ts, 0.0, 1.0)

    pred_boost, dist = surrogate.predict_boost(cand)
    ok = dist <= safety_radius
    if not ok.any():
        return None
    err = np.where(ok, np.abs(pred_boost - target), np.inf)
    j = int(np.argmin(err))
    if not np.isfinite(err[j]) or np.allclose(cand[j], x_f):
        return None
    return cand[j]


def fastest_stage2(evaluate, xs, s1trace, target: float, plane: dict, ladder, budget: int,
                   *, surrogate=None, corpus_X=None, safety_radius: float | None = None):
    """Fastest mode's stage 2. See module docstring for why this differs from H2.

    Spends at most one evaluation deciding where to jump, then hands off to the UNCHANGED,
    frozen `g32_solve` with whatever budget remains. Returns exactly what `g32_solve`
    returns -- `(trace, info, x_f, rec_f, budget_left)` -- with one extra key, `info["hedge"]`,
    recording what the hedge attempted and why, for the provenance record.
    """
    xs = list(xs)
    s1trace = list(s1trace)
    hedge: dict[str, Any] = {"attempted": False, "evaluated": False, "accepted": False,
                             "reason": None}

    feasible = [(abs(e["boost_db"] - target), j) for j, e in enumerate(s1trace)
               if e and e["loose_pass"]]
    if not feasible:
        hedge["reason"] = "stage 1 produced no feasible point to jump from"
    elif surrogate is None or corpus_X is None or safety_radius is None or budget <= 0:
        hedge["reason"] = "surrogate unavailable or no budget"
    else:
        x_f = xs[min(feasible)[1]]
        hedge["attempted"] = True
        cand = propose_jump(x_f, target, plane, surrogate, safety_radius)
        if cand is None:
            hedge["reason"] = "no candidate cleared the corpus safety radius"
        else:
            rec, _score, _gcheck = evaluate(cand, target)
            hedge["evaluated"] = True
            xs.append(np.asarray(cand, dtype=np.float64))
            s1trace.append(rec)
            budget -= 1
            hedge["accepted"] = bool(rec and rec["loose_pass"])
            hedge["reason"] = ("accepted" if hedge["accepted"] else
                               "candidate not guard/spec valid; G3.2 will still see and "
                               "may reject or repair it")

    # Called through the module object, not a name bound at import time: `server.py`'s
    # live-narration wrapper monkey-patches `final_comparison.g32_solve` as a module
    # attribute for the duration of a run, and only a lookup through `fc.` at call time
    # sees that patch -- a name imported directly here would keep pointing at the
    # original, unpatched function forever.
    trace, info, x_out, rec_out, left = fc.g32_solve(evaluate, xs, s1trace, target, plane,
                                                      ladder, budget)
    info["hedge"] = hedge
    return trace, info, x_out, rec_out, left
