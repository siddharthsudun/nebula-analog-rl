"""The cost counter itself: does it count everything, and does it clean up after itself.

No SPICE. `counting()` wraps whatever is bound at the two chokepoints, so a fake bound in
its place is measured exactly like the real thing -- which is the property being tested.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from eqrl.simcount import MEASURE_ALL_BINDINGS, counting

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fakes(monkeypatch):
    """Bind a trivial callable at every measure_all site and report how often it ran."""
    hits = {"n": 0}

    def fake_measure_all(*a, **kw):
        hits["n"] += 1
        return "measured"

    mods = []
    for mod_name, attr in MEASURE_ALL_BINDINGS:
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, attr, fake_measure_all, raising=True)
        mods.append((mod, attr))
    return hits, mods, fake_measure_all


def test_it_counts_every_binding_not_just_the_canonical_one(fakes):
    """sequential_env holds its own reference; missing it undercounts every env.step."""
    hits, mods, _ = fakes
    with counting() as c:
        for mod, attr in mods:
            getattr(mod, attr)()
    assert c["measure_all"] == len(mods) == 2
    assert hits["n"] == 2


def test_the_originals_are_restored_on_the_way_out(fakes):
    _hits, mods, fake = fakes
    with counting():
        for mod, attr in mods:
            assert getattr(mod, attr) is not fake, "should be wrapped inside the block"
    for mod, attr in mods:
        assert getattr(mod, attr) is fake, "%s.%s was not restored" % (mod.__name__, attr)


def test_the_originals_are_restored_even_when_the_block_raises(fakes):
    _hits, mods, fake = fakes
    with pytest.raises(ValueError):
        with counting():
            raise ValueError("boom")
    for mod, attr in mods:
        assert getattr(mod, attr) is fake


def test_counting_nests_and_both_levels_see_every_call(fakes):
    """`pipeline.design` counts per phase; a test wraps the whole call to check the sum.

    That only works if an outer counter sees the measurements an inner counter also sees.
    """
    _hits, mods, _ = fakes
    mod, attr = mods[0]
    with counting() as outer:
        getattr(mod, attr)()
        with counting() as inner:
            getattr(mod, attr)()
            getattr(mod, attr)()
        getattr(mod, attr)()
    assert inner["measure_all"] == 2
    assert outer["measure_all"] == 4


def test_a_count_is_final_once_the_block_exits(fakes):
    _hits, mods, _ = fakes
    mod, attr = mods[0]
    with counting() as c:
        getattr(mod, attr)()
    after = dict(c)
    getattr(mod, attr)()
    assert dict(c) == after


def test_it_patches_the_same_places_the_published_audit_patches():
    """`simcount_audit` produced REPRODUCE.md section 13's conversion factor. If this
    module counted at different places, the demo's cost and the published factor would be
    measuring two different things.

    Read as source text rather than imported: `simcount_audit` performs the ngspice/PDK
    environment bootstrap at import, and this assertion is about code, not environment.
    """
    src = (ROOT / "src/eqrl/experiments/simcount_audit.py").read_text(encoding="utf-8")
    body = src[src.index("def install_counters"):src.index("def take")]

    # Left-hand sides of the assignments that install the counters.
    targets = set(re.findall(r"^\s*([\w.]+)\s*=\s*counted_\w+", body, re.M))
    assert targets == {"measures_mod.measure_all", "NgspiceServer._analysis",
                       "se.measure_all"}, targets

    # `measures_mod` and `se` are aliases; confirm what they alias, then that this module
    # names the same two modules.
    assert "from eqrl.sim import measures as measures_mod" in src
    assert "import eqrl.envs.sequential_env as se" in body
    assert {m for m, _a in MEASURE_ALL_BINDINGS} == {"eqrl.sim.measures",
                                                     "eqrl.envs.sequential_env"}
    assert {a for _m, a in MEASURE_ALL_BINDINGS} == {"measure_all"}
