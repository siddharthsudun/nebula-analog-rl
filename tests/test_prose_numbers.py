"""Structural constants quoted in prose must equal what the code computes.

WHY: SEVEN STALE-NUMBER DEFECTS IN ONE DAY, NONE CAUGHT BY CI
--------------------------------------------------------------
On 07 Sep 2026 this project found seven numbers that were asserted in prose and false in
fact. The worst was the CTLE area worst case, published as **0.012 mm²** when the true
analytic supremum is **0.002227** — a 5.4× overstatement, live simultaneously in a
module docstring, the README, and both built sites. Its cause is the general one: the
docstring's arithmetic was correct *when written*, at a 20 mA tail current, and nobody
recomputed it when `ACTION_SPACE` narrowed `i_tail` to 1 mA. A number in prose is an
assertion with no test behind it, and it survives every edit to the code it describes.

`tests/test_citations.py` does this for `file:line` pointers. This file does it for the
handful of numbers that are DERIVED — recomputable from the code in milliseconds, with no
simulator. Measured results (median errors, pass counts) are NOT in scope: those come from
runs and belong to their preregistered artifacts. The distinction is the point. A derived
constant that disagrees with its own code is always a bug; a measured one may simply be old.

WHAT IS CHECKED, AND WHY EACH EARNS ITS PLACE
-----------------------------------------------
1. the area supremum, and the *premise* that makes it a supremum rather than a sample;
2. the 22× margin against the area budget, which is a quotient of two quoted numbers;
3. the corpus size, quoted in three docs and used to size a cache comment;
4. `hard_pass`'s check count, which the docs quote as eight and nine in different places
   and which is exactly the ambiguity `pass_vs_valid` shipped a wrong artifact over.

Each assertion recomputes the value and then greps every prose site for it, so adding a
new mention of a number that has since moved fails here rather than in a judge's reading.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from eqrl.circuits.ctle import ACTION_SPACE, DesignVars
from eqrl.specs import DEFAULT_SPEC, hard_pass

ROOT = Path(__file__).resolve().parents[1]

#: Every file that quotes one of these constants in prose. Listed explicitly rather than
#: globbed: the test must fail when a NEW file quotes a stale number, and it can only do
#: that if adding the file here is part of writing the claim.
PROSE_SITES = ("README.md", "docs/PREREG_G32_ACCEPTANCE.md", "docs/PROBLEM.md",
               "docs/DESIGN_MODES_V2.md", "docs/REPRODUCE.md",
               "docs/RESULTS_INFERENCE_MODES.md", "web/index.html", "site/index.html",
               "src/eqrl/circuits/ctle.py", "src/eqrl/experiments/thinking_starts.py")

#: The area budget the specification sets. Quoted as "0.05 mm²" wherever the margin is.
AREA_BUDGET_MM2 = 0.05


def _prose() -> dict[str, str]:
    out = {}
    for rel in PROSE_SITES:
        p = ROOT / rel
        if p.is_file():
            # nbsp: the built sites write "0.002227&nbsp;mm²", so normalize before search.
            out[rel] = p.read_text(encoding="utf-8", errors="replace").replace("&nbsp;", " ")
    return out


def _upper_corner() -> DesignVars:
    return DesignVars(**{k: hi for k, (_lo, hi) in ACTION_SPACE.items()})


def test_area_supremum_is_what_the_prose_says():
    """The headline number, recomputed from ACTION_SPACE rather than trusted."""
    sup = _upper_corner().area_mm2()
    assert f"{sup:.6f}" == "0.002227", (
        f"the analytic area supremum is now {sup:.6f} mm^2, not 0.002227. Every site in "
        "PROSE_SITES quotes the old value and is now false. Recompute, then update them "
        "together -- this is exactly how 0.012 survived for weeks.")

    quoted = [rel for rel, txt in _prose().items() if "0.002227" in txt]
    assert len(quoted) >= 6, (
        f"only {len(quoted)} prose sites quote the supremum; it was 8 on 07 Sep 2026 "
        f"({quoted}). A site that stopped quoting it may have reverted to a stale figure.")

    stale = {rel: m for rel, txt in _prose().items()
             # The number 0.012/0.0113 may appear only where it is EXPLICITLY retracted.
             for m in re.findall(r"0\.0113|0\.012 ?mm", txt)
             if not re.search(r"(used to say|predated|earlier|replaces|instead of|not)"
                              r"[^\n]{0,80}" + re.escape(m), txt)}
    assert not stale, (
        f"a retracted area figure is quoted as live in {sorted(stale)}. 0.012 and 0.0113 "
        "are the pre-1mA values; they may appear only as explicitly-labelled corrections.")


def test_the_upper_corner_really_is_the_supremum():
    """The PREMISE, not just the number.

    The prose does not merely report 0.002227 -- it claims the value is a *bound over the
    whole space*, because "every term of `area_mm2` is increasing in its own variable".
    That claim is what makes it stronger than the 200k-sample maximum quoted beside it, and
    it is the part a reader is being asked to accept. So it is tested directly: from many
    random interior points, raising any single variable must never DECREASE the area.

    Without this, the test above would pin a number whose meaning had silently changed --
    a `min()` introduced into the mirror sizing would leave 0.002227 correct as an
    evaluation and false as a supremum.
    """
    rng = np.random.default_rng(0)
    keys = list(ACTION_SPACE)
    sup = _upper_corner().area_mm2()

    for _ in range(200):
        base = {k: rng.uniform(lo, hi) for k, (lo, hi) in ACTION_SPACE.items()}
        a0 = DesignVars(**base).area_mm2()
        assert a0 <= sup + 1e-15, (
            f"an interior point has area {a0:.9f} > the claimed supremum {sup:.9f}. The "
            "upper corner is not the maximum and the 'bound over the whole space' claim "
            "in README/web/site is false.")
        for k in keys:
            up = dict(base)
            up[k] = rng.uniform(base[k], ACTION_SPACE[k][1])
            assert DesignVars(**up).area_mm2() >= a0 - 1e-15, (
                f"raising `{k}` lowered the area. `area_mm2` is no longer increasing in "
                "each variable, so the upper corner is NOT a supremum and every site "
                "calling 0.002227 an analytic bound must be reworded to 'sampled max'.")


def test_area_margin_quotient():
    """22x is a quotient of two quoted numbers, so it can rot when either moves."""
    margin = AREA_BUDGET_MM2 / _upper_corner().area_mm2()
    assert 22.0 <= margin < 23.0, (
        f"the area margin is now {margin:.1f}x, and README/web/site/PREREG all say 22x.")
    for rel, txt in _prose().items():
        if re.search(r"\d+\s*[x×]\s*margin", txt):
            assert re.search(r"22\s*[x×]\s*margin", txt), (
                f"{rel} quotes an area margin that is not 22x; the computed value is "
                f"{margin:.1f}x.")


def test_corpus_size_quoted_in_docs():
    """Three docs and a cache comment quote the corpus size. It is one number."""
    npz = ROOT / "results" / "surrogate_corpus.npz"
    if not npz.is_file():
        pytest.skip("surrogate_corpus.npz not present; this checks prose against the asset")
    with np.load(npz) as d:
        n = int(d["X"].shape[0])
    assert n == 374588, (
        f"the corpus now holds {n} designs, not 374588. Every doc quoting the old size is "
        "stale, and `thinking_starts`'s retention comment sizes itself on it.")
    # A 374,xxx figure inside a sentence about the RAW ARCHIVE is a different quantity --
    # runs on disk carrying an `acx.data`, counted before the rebuild. Found by this test
    # firing on docs/REPRODUCE.md, where both were called "the corpus" and so looked like
    # a contradiction. The exemption is narrow on purpose: it requires the archive named
    # in the same sentence, which is exactly the disambiguation that was missing.
    for rel, txt in _prose().items():
        for sentence in re.split(r"(?<=[.)])\s", txt):
            if re.search(r"raw archive|results/raw|acx\.data", sentence, re.I):
                continue
            for m in re.findall(r"\b374[, ]?\d{3}\b", sentence):
                assert m.replace(",", "").replace(" ", "") == str(n), (
                    f"{rel} quotes a surrogate-corpus size of {m}, but the asset holds "
                    f"{n}. If this is the raw-archive count instead, name the archive in "
                    "the same sentence -- the two were conflated once already.")


def test_hard_pass_check_count():
    """Eight published checks, nine at this repo's defaults. Both numbers are load-bearing.

    `pass_vs_valid` shipped an artifact scored on EIGHT while its code had drifted to NINE,
    because `DEFAULT_SPEC.dc_gain_db_min` was flipped to 0.0 after the artifact was written.
    The counts are quoted throughout `docs/PASS_VS_VALID.md`, so they are pinned here.
    """
    import dataclasses

    class _M:                                    # a measurement that passes everything
        boost_db = 6.0; dc_gain_db = 3.0; peak_freq_ghz = 5.0
        power_w = 1e-4; area_mm2 = 1e-4; noise_vrms = 1e-9
        hd3_db = -60.0; eye_h_ui = 0.9; eye_v_mv = 300.0; ok = True

    eight = dataclasses.replace(DEFAULT_SPEC, dc_gain_db_min=None)
    _ok8, checks8 = hard_pass(_M(), eight)
    _ok9, checks9 = hard_pass(_M(), DEFAULT_SPEC)

    assert len(checks8) == 8, (
        f"the published spec now emits {len(checks8)} checks, not eight: "
        f"{sorted(checks8)}. docs/PASS_VS_VALID.md quotes eight throughout.")
    assert len(checks9) == 9 and "dc_gain" in checks9, (
        f"this repo's default spec now emits {len(checks9)} checks, not nine: "
        f"{sorted(checks9)}. The ninth is `dc_gain`, and PASS_VS_VALID says so.")
    assert DEFAULT_SPEC.dc_gain_db_min == 0.0, (
        "DEFAULT_SPEC.dc_gain_db_min is no longer 0.0, so 'nine at our defaults' is false.")


# ---------------------------------------------------------------------------------------
# 5. The g32_acceptance quiet interval, and the two constants it silently depends on.
# ---------------------------------------------------------------------------------------

#: The interval `g32_acceptance` is documented as provably inert over, at the default spec.
#: It is quoted in three places. Two are code comments, in `pipeline.py` and in
#: `thinking_starts.py`, where a stale number misleads a maintainer. The third is prose
#: written to become UI copy, where a stale number misleads a judge, and that is what
#: earns this a test rather than another comment.
#: docs/PREREG_G32_ACCEPTANCE.md:267 reads "for a target between 3.05 and 11.13 dB".
QUIET_LO_DB, QUIET_HI_DB = 3.05, 11.13

#: The scan resolution the interval was measured at. The endpoints are only meaningful to
#: this precision: "3.05 is inside and 3.04 is outside" is a claim about a 0.01 dB grid.
QUIET_STEP_DB = 0.01

#: THE INTERVAL IS NOT A PROPERTY OF THE MODE. It is a property of the mode AT THESE TWO
#: NUMBERS, and neither of them is where a reader would look for it: one is a module
#: constant in another file, the other is a DEFAULT ARGUMENT of `diverse_seeds`. Measured
#: 07 Sep 2026 by contiguous 0.01 dB scans over 3.00-12.00 dB, assuming no monotonicity:
#:
#:      n=4  pool=512    [3.05, 11.13]   <- shipped
#:      n=8  pool=512    [3.06, 11.13]
#:      n=16 pool=512    [3.07, 11.13]
#:      n=2  pool=512    [3.01,  3.01]   <- NO CONTIGUOUS INTERVAL EXISTS
#:      n=4  pool=256    [3.04, 11.42]
#:      n=4  pool=1024   [3.04,  3.04]   <- NO CONTIGUOUS INTERVAL EXISTS
#:
#: At n=2 and at pool=1024 the claim does not merely move, it STOPS HAVING A SHAPE: there
#: is no contiguous band to quote, so a sentence of the form "between X and Y this cannot
#: change the result" has no true instance. That is why this test checks the constants as
#: well as the endpoints. An endpoints-only test would pass at pool=1024 for the two points
#: it sampled while the sentence it guards had become unwriteable.
EXPECTED_SURROGATE_STARTS = 4
EXPECTED_SEED_POOL = 512


def _acceptance_seeds_identical(target_db: float, surrogate, *, pool: int) -> bool:
    """Do the acceptance-filtered restarts equal the unfiltered ones at this target?"""
    from eqrl.experiments.thinking_starts import diverse_seeds
    from eqrl.pipeline import spec_for

    spec = spec_for(target_db, DEFAULT_SPEC.channel_loss_db, 1.5)
    kw = dict(pool=pool)
    plain = diverse_seeds(target_db, surrogate, spec, EXPECTED_SURROGATE_STARTS, **kw)
    filt = diverse_seeds(target_db, surrogate, spec, EXPECTED_SURROGATE_STARTS,
                         acceptance=True, **kw)
    return len(plain) == len(filt) and all(
        np.array_equal(a, b) for a, b in zip(plain, filt))


def test_quiet_interval_endpoints_are_where_the_prose_says():
    """The documented interval must be quiet inside and NOT quiet one step outside.

    Both halves are load-bearing. Checking only that the inside is quiet would pass for an
    interval quoted far too narrow; checking only the outside would pass for one quoted far
    too wide. The step below and above the endpoints is what pins them.
    """
    try:
        from eqrl.experiments.fastest_hedge import load_fastest_assets
        surrogate, _corpus_x, _radius = load_fastest_assets()
    except Exception as exc:                      # noqa: BLE001 -- asset, not logic
        pytest.skip(f"surrogate corpus unavailable ({exc}); this checks prose against it")

    inside = [QUIET_LO_DB, QUIET_HI_DB]
    outside = [round(QUIET_LO_DB - QUIET_STEP_DB, 2), round(QUIET_HI_DB + QUIET_STEP_DB, 2)]

    wrong_in = [t for t in inside
                if not _acceptance_seeds_identical(t, surrogate, pool=EXPECTED_SEED_POOL)]
    assert not wrong_in, (
        f"g32_acceptance CHANGES the restart seeds at {wrong_in} dB, which the docs place "
        f"INSIDE the quiet interval [{QUIET_LO_DB}, {QUIET_HI_DB}]. Every site quoting that "
        "interval is now asserting inertness where the code is not inert. Re-scan and "
        "re-quote; do not widen this test.")

    wrong_out = [t for t in outside
                 if _acceptance_seeds_identical(t, surrogate, pool=EXPECTED_SEED_POOL)]
    assert not wrong_out, (
        f"the seeds are unchanged at {wrong_out} dB, one 0.01 dB step OUTSIDE the quoted "
        f"interval [{QUIET_LO_DB}, {QUIET_HI_DB}], so the interval as quoted is narrower "
        "than the truth. This is the benign direction -- the published claim stays true -- "
        "but the endpoint is no longer the measured one, so re-scan and re-quote it.")


def test_quiet_interval_is_contiguous_not_just_its_endpoints():
    """Sampled across the interval, because the region ABOVE it is ragged, not clean.

    The measured shape has isolated identical points scattered through 11.14-12.00 dB
    (11.20, 11.87 and others). That is precisely why a bisection reported a threshold that
    does not exist, and why the first two versions of this claim were retracted. An
    endpoints-only test cannot distinguish a genuinely contiguous quiet interval from two
    lucky endpoints with holes between them.
    """
    try:
        from eqrl.experiments.fastest_hedge import load_fastest_assets
        surrogate, _corpus_x, _radius = load_fastest_assets()
    except Exception as exc:                      # noqa: BLE001 -- asset, not logic
        pytest.skip(f"surrogate corpus unavailable ({exc}); this checks prose against it")

    # 0.5 dB sampling, not the 0.01 dB the endpoints were measured at: a full contiguous
    # scan is ~1800 calls and belongs in the preregistered artifact, not in a unit test.
    # This samples densely enough to catch a hole big enough to matter to a UI sentence.
    targets = np.round(np.arange(QUIET_LO_DB, QUIET_HI_DB + 1e-9, 0.5), 2)
    holes = [float(t) for t in targets
             if not _acceptance_seeds_identical(float(t), surrogate,
                                                pool=EXPECTED_SEED_POOL)]
    assert not holes, (
        f"the quiet interval [{QUIET_LO_DB}, {QUIET_HI_DB}] has holes at {holes} dB. It is "
        "not an interval, and no sentence of the form 'between X and Y this cannot change "
        "the result' is true of it. The honest claim becomes a set, not a range.")


def test_the_quiet_interval_still_has_the_two_constants_it_was_measured_at():
    """The endpoints above are only the endpoints AT n=4 AND pool=512.

    This is the test that catches the failure the other two cannot see. `pool` is a DEFAULT
    ARGUMENT, so a caller changing it does not touch any line that quotes the interval, and
    at pool=1024 no contiguous interval exists at all -- the claim stops having a shape
    rather than moving. Nothing else in this tree ties those two numbers to the prose.
    """
    import inspect

    from eqrl.experiments.thinking_starts import diverse_seeds
    from eqrl.pipeline import THINKING_SURROGATE_STARTS

    assert THINKING_SURROGATE_STARTS == EXPECTED_SURROGATE_STARTS, (
        f"THINKING_SURROGATE_STARTS is now {THINKING_SURROGATE_STARTS}, not "
        f"{EXPECTED_SURROGATE_STARTS}. The quiet interval "
        f"[{QUIET_LO_DB}, {QUIET_HI_DB}] was measured at 4 restarts and moves with this "
        "number (3.06 at n=8, 3.07 at n=16, and NO contiguous interval at all at n=2). "
        "Re-scan before re-quoting it anywhere, including in UI copy.")

    pool_default = inspect.signature(diverse_seeds).parameters["pool"].default
    assert pool_default == EXPECTED_SEED_POOL, (
        f"diverse_seeds' `pool` default is now {pool_default}, not {EXPECTED_SEED_POOL}. "
        f"The quiet interval [{QUIET_LO_DB}, {QUIET_HI_DB}] was measured at 512 and does "
        "not survive the change: at pool=1024 the identical-seed targets are scattered "
        "rather than contiguous, so there is no interval to quote. This default is the "
        "least visible dependency the published claim has -- it is an argument default in "
        "a different module from every sentence that quotes it.")
