"""A cheap scout over the SPICE runs this project has already paid for.

WHAT IT IS. A regressor from a normalized design vector to the three metrics that define
the requested spec:

    normalized 6-D design  ->  (dc_gain_db, boost_db, peak_freq_ghz)

WHY IT COSTS NOTHING TO BUILD. `results/raw` holds every SPICE run this project has ever
issued -- 415,092 of them, 374,590 carrying an `acx.data`. All three metrics above come
out of the AC run alone (`sim.measures.peaking` reads `mag[0]`, `mag[argmax] - mag[0]`,
and `freq[argmax]`), and `meta.json` records the design that produced it. So the training
set is recoverable by PARSING FILES ALREADY ON DISK. Not one new simulation is run here.

    `acx.data` IS COMPLEX, NOT MAGNITUDE. `NgspiceServer.ac_complex` writes
    `[freq, real, imag]` -- wrdata's layout for a complex vector -- so the magnitude has
    to be RECONSTRUCTED as 20*log10|real + j*imag|. The first version of this module read
    column 1 directly as mag_db, which is the real part. That produced small, plausible,
    entirely fictitious boosts (corpus median 2.00 dB against a true median near 3 dB, and
    high-boost designs flattened by 5+ dB), and NO held-out split could detect it: train
    and test shared the same wrong labelling, so the pre-registered gate passed at 0.166
    dB while predicting the wrong quantity. What caught it was checking against a source
    with INDEPENDENT ground truth -- see `assert_matches_feasibility_band` below.

    `results/raw` carries no `ac.data`; the AC sweep that `peaking` uses was never
    persisted. So the corpus is built from `acx.data`, whose sweep stops at 24 GHz rather
    than AC_FSTOP = 100 GHz. On the 51 designs with independent ground truth that
    difference is worth nothing at all: all three metrics reproduce to a max error of
    0.0005 dB. It is recorded here because a design peaking above 24 GHz would be
    understated, and that is a real if unobserved limit.

WHAT IT IS NOT. It predicts AC metrics only. It does not predict guard validity, eye
height or width, input-referred noise, or HD3 -- and guard validity is the binding
constraint, not boost: a uniform sample of the design space is 23% guard-valid and 1.2%
all-pass (results/feasibility_band.json). Its coverage is also biased toward the regions
PPO visited, because that is what generated the corpus.

    IT IS A RANKER, NEVER A JUDGE.

No solve, pass, or success in any reported number may be decided by this model. It exists
to ORDER candidates so that the real evaluator spends its budget on better ones. Every
design that reaches a result is verified by the same guarded `fast=False` evaluator every
other arm uses. See docs/REPRODUCE.md section 16 for the pre-registered acceptance gate.

COORDINATES. Designs are normalized by `ctle.encode_action`, so a distance in surrogate
space is the same distance the policy's own action space uses. The vectorized path here
re-derives ctle's `lo > 0 and hi / lo > 50` log-scale rule rather than hard-coding which
axes it selects -- and `assert_encoding_matches_ctle` checks the two agree to 1e-12 on
random designs, because a silently divergent coordinate system would make every distance,
every neighbour and every prediction wrong in a way no metric would reveal.

    For the record, that rule selects only `w_in` (100x) and `cs` (200x). `i_tail` (20x),
    `rs` (50x) and `r_load` (50x) are LINEAR -- 50 is not greater than 50.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from silq.circuits.ctle import ACTION_SPACE, DesignVars, encode_action

#: Field order of the normalized vector. `ACTION_SPACE` is ordered, and `encode_action`
#: iterates it in the same order, so this IS the action-vector layout.
KEYS: list[str] = list(ACTION_SPACE)

#: The three metrics an AC run determines. Ordered; index 1 (`boost_db`) is the one the
#: strict criterion is written against.
METRICS: tuple[str, ...] = ("dc_gain_db", "boost_db", "peak_freq_ghz")

_LO = np.array([ACTION_SPACE[k][0] for k in KEYS], dtype=np.float64)
_HI = np.array([ACTION_SPACE[k][1] for k in KEYS], dtype=np.float64)
#: ctle.decode_action's own rule, re-derived rather than restated.
_LOG = (_LO > 0) & (_HI / np.maximum(_LO, 1e-300) > 50)


def normalize(raw: np.ndarray) -> np.ndarray:
    """Physical design values -> [0, 1]^6, vectorized `ctle.encode_action`.

    `raw` is (n, 6) in SI units, columns in `KEYS` order. Clipped to [0, 1] exactly as
    `encode_action` clips, so an out-of-range record lands on the boundary instead of
    silently extending the space.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    out = np.empty_like(raw)
    lin = ~_LOG
    out[:, lin] = (raw[:, lin] - _LO[lin]) / (_HI[lin] - _LO[lin])
    v = np.maximum(raw[:, _LOG], _LO[_LOG])
    out[:, _LOG] = np.log(v / _LO[_LOG]) / np.log(_HI[_LOG] / _LO[_LOG])
    return np.clip(out, 0.0, 1.0)


def assert_encoding_matches_ctle(n: int = 2000, seed: int = 0) -> float:
    """Check `normalize` against `ctle.encode_action` and return the float64 residual.

    THE TEST IS BITWISE EQUALITY AFTER CASTING TO float32, not a tolerance. `encode_action`
    returns float32 (the policy's action dtype) while this module works in float64, so a
    raw comparison shows ~3e-8 -- float32 epsilon -- on every axis. That is the cast, not a
    disagreement, and the way to prove it is to cast and demand the bits match, which is
    strictly stronger than picking a tolerance loose enough to pass.

    Raises rather than warns. A silently divergent coordinate system would make every
    distance, neighbour and prediction wrong in a way no accuracy metric would reveal.
    """
    rng = np.random.default_rng(seed)
    raw = np.empty((n, len(KEYS)))
    for j, k in enumerate(KEYS):
        lo, hi = ACTION_SPACE[k]
        raw[:, j] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    mine = normalize(raw)
    theirs = np.array([encode_action(DesignVars(**dict(zip(KEYS, r)))) for r in raw])
    if theirs.dtype != np.float32:
        raise AssertionError(
            f"ctle.encode_action now returns {theirs.dtype}, not float32. This check "
            "was written around that cast; re-derive it rather than relaxing it.")
    if not (mine.astype(np.float32) == theirs).all():
        n_bad = int((mine.astype(np.float32) != theirs).sum())
        raise AssertionError(
            f"surrogate.normalize disagrees with ctle.encode_action on {n_bad} of "
            f"{mine.size} components after the float32 cast, by up to "
            f"{np.abs(mine - theirs).max():.3e}. The surrogate's coordinate system must "
            "be the policy's or every distance, neighbour and prediction it produces is "
            "wrong. Refusing to continue.")
    return float(np.abs(mine - theirs).max())


def metrics_from_acx(txt: str) -> tuple[float, float, float] | None:
    """(dc_gain_db, boost_db, peak_freq_ghz) from one `acx.data`.

    `acx.data` is `[freq, real, imag]` -- wrdata's layout for the complex vector
    `v(outp)-v(outn)`, written by `NgspiceServer.ac_complex`. THE MAGNITUDE IS NOT IN THE
    FILE and must be reconstructed; reading column 1 as mag_db reads the real part, which
    is a different quantity that looks enough like a small dB figure to pass unnoticed.

    Given the magnitude, this then mirrors `sim.measures.peaking` exactly: DC is the first
    sample, the peak is the argmax, boost is their difference. Verified against 51 designs
    with independent SPICE ground truth to a maximum error of 0.0005 dB on all three
    metrics -- see `assert_matches_feasibility_band`.

    Returns None for a file that is empty, truncated, or not three columns, and for a
    non-positive magnitude that would make the log undefined. A partial record is dropped,
    never padded into a plausible-looking one.
    """
    try:
        a = np.array(txt.split(), dtype=np.float64)
    except ValueError:
        return None
    if a.size < 30 or a.size % 3:
        return None
    a = a.reshape(-1, 3)
    freq = a[:, 0]
    absH = np.abs(a[:, 1] + 1j * a[:, 2])
    if not np.isfinite(freq).all() or not np.isfinite(absH).all() or (absH <= 0).any():
        return None
    mag = 20.0 * np.log10(absH)
    i = int(np.argmax(mag))
    return float(mag[0]), float(mag[i] - mag[0]), float(freq[i] / 1e9)


def assert_matches_feasibility_band(root: str | Path = "results/raw",
                                    band: str | Path = "results/feasibility_band.json",
                                    corpus_path: str | Path | None = None,
                                    tol_db: float = 0.01) -> dict[str, float]:
    """Check the parser against a source of INDEPENDENT ground truth. Raises on mismatch.

    THE GATE IN `surrogate_audit` CANNOT DO THIS. A held-out split grades the model on
    labels produced by the same parser that made the training labels, so any error shared
    by both is invisible to it -- and a corpus-wide mislabelling is exactly that kind of
    error. It passed at 0.166 dB MAE while the labels were the real part of H.

    `feasibility_band.json` is the antidote: 2000 uniformly drawn designs run through the
    real evaluator, with `dc_gain_db`, `boost_db` and `peak_freq_ghz` recorded by
    `sim.measures.peaking` itself rather than by anything in this module. The subset that
    also records its design gives per-design ground truth this parser must reproduce.

    Uniform draws also put the check where the corpus is thinnest and where the requested
    targets actually live, which is where the mislabelling did its worst damage.
    """
    from scipy.spatial import cKDTree

    root = Path(root)
    fb = json.loads(Path(band).read_text())
    recs = [r for r in fb["scatter"] if "design" in r]
    if not recs:
        raise AssertionError(f"{band} records no designs; cannot validate the parser.")
    corpus = load_corpus(corpus_path or _CORPUS_DEFAULT)
    tree = cKDTree(corpus["X"])
    Xq = normalize(np.array([[r["design"][k] for k in KEYS] for r in recs]))
    d, idx = tree.query(Xq, k=1, workers=-1)
    hit = d < 1e-9
    if hit.sum() < 10:
        raise AssertionError(
            f"only {int(hit.sum())} of {len(recs)} ground-truth designs are present in "
            "the corpus; too few to validate the parser against.")

    want = np.array([[r["dc_gain_db"], r["boost_db"], r["peak_freq_ghz"]]
                     for r, h in zip(recs, hit) if h])
    got = []
    for j in idx[hit]:
        m = metrics_from_acx((root / str(corpus["run_id"][j]) / "acx.data").read_text())
        if m is None:
            raise AssertionError(f"run {corpus['run_id'][j]} failed to parse during "
                                 "ground-truth validation.")
        got.append(m)
    err = np.abs(np.array(got) - want)
    out = {name: float(err[:, i].max()) for i, name in enumerate(METRICS)}
    out["n_checked"] = int(hit.sum())
    bad = {k: v for k, v in out.items() if k in METRICS and v > tol_db}
    if bad:
        raise AssertionError(
            "the corpus parser disagrees with SPICE ground truth on "
            + ", ".join(f"{k} by up to {v:.4f}" for k, v in bad.items())
            + f" over {int(hit.sum())} uniformly drawn designs. The corpus is mislabelled; "
              "no gate on a held-out split can detect this, because training and test "
              "labels share the error. Refusing to continue.")
    return out


def build_corpus(root: str | Path = "results/raw", limit: int | None = None,
                 progress_every: int = 25000) -> dict[str, np.ndarray]:
    """Parse every recorded AC run into (X, Y, run_id).

    Run ids are the artifact directory names, which begin with a UTC timestamp, so
    sorting by run id sorts chronologically. That is what makes the held-out-era split in
    `surrogate_audit` possible -- and an i.i.d. split is not an acceptable substitute,
    because the corpus is mostly PPO trajectories and a design's own one-step neighbour
    would land in the training set.
    """
    root = Path(root)
    raws: list[list[float]] = []
    ys: list[tuple[float, float, float]] = []
    rids: list[str] = []
    seen = skipped_no_ac = skipped_bad = 0
    with os.scandir(root) as it:
        names = sorted(e.name for e in it if e.is_dir())
    for name in names:
        if limit is not None and len(raws) >= limit:
            break
        seen += 1
        if progress_every and seen % progress_every == 0:
            print(f"  scanned {seen:>7}/{len(names)}  kept {len(raws):>7}", flush=True)
        acx = root / name / "acx.data"
        if not acx.exists():
            skipped_no_ac += 1
            continue
        try:
            m = json.loads((root / name / "meta.json").read_text())
            dv = m["design"]
            row = [float(dv[k]) for k in KEYS]
        except (OSError, ValueError, KeyError, TypeError):
            skipped_bad += 1
            continue
        try:
            y = metrics_from_acx(acx.read_text())
        except OSError:
            y = None
        if y is None:
            skipped_bad += 1
            continue
        raws.append(row)
        ys.append(y)
        rids.append(name)
    print(f"  scanned {seen}, kept {len(raws)}, no acx {skipped_no_ac}, "
          f"unusable {skipped_bad}", flush=True)
    return {"X": normalize(np.array(raws)), "Y": np.array(ys, dtype=np.float64),
            "run_id": np.array(rids)}


_CORPUS_DEFAULT = "results/surrogate_corpus.npz"


def save_corpus(corpus: dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **corpus)


def load_corpus(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


class Surrogate:
    """k-nearest-neighbour regressor over the corpus.

    kNN ON PURPOSE, not for want of anything fancier. It is the model whose behaviour the
    downstream uses actually need: retrieval-from-memory and local ranking are both
    literally nearest-neighbour queries, so the accuracy this reports IS the accuracy
    those mechanisms would get. It also cannot extrapolate confidently into empty space --
    it returns the neighbourhood it has -- and it reports the distance to that
    neighbourhood, which is the honest uncertainty signal a fitted model would have to be
    coaxed into producing.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, k: int = 5):
        from scipy.spatial import cKDTree
        if len(X) < k:
            raise ValueError(f"corpus of {len(X)} is smaller than k={k}")
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.Y = np.ascontiguousarray(Y, dtype=np.float64)
        self.k = int(k)
        self._tree = cKDTree(self.X)

    def predict(self, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(prediction (n, 3), distance to nearest training design (n,)).

        The distance is returned, not hidden, because it is the model's own statement of
        how far outside its evidence the question was. A caller ranking candidates should
        be able to see when it is ranking on nothing.
        """
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float64))
        d, idx = self._tree.query(Xq, k=self.k, workers=-1)
        if self.k == 1:
            d, idx = d[:, None], idx[:, None]
        return self.Y[idx].mean(axis=1), d[:, 0]

    def predict_boost(self, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Just `boost_db` and the neighbour distance -- what the H2 ranker asks for."""
        p, d = self.predict(Xq)
        return p[:, METRICS.index("boost_db")], d
