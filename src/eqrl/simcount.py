"""Count what a run actually costs the simulator, at the two chokepoints REPRODUCE.md
section 13 defines.

WHY THIS EXISTS. Section 13's whole claim is that this project counts simulations honestly,
in units it has pinned down: `measures.measure_all` calls and `NgspiceServer._analysis`
runs. `eqrl.experiments.simcount_audit` established the conversion factor by patching those
two functions, because the call graph runs through the guard layer and libngspice and any
hand-count of it is an assumption -- which is the thing being audited. But that audit is a
one-off script with process-global counters, so nothing that ships a cost number to a user
could use it, and `eqrl.pipeline` ended up reporting a hand-derived figure instead.

This is the same measurement as a context manager: it installs the counters, restores the
original functions on the way out, and can be nested. It is the mechanism, not a second
opinion -- `tests/test_simcount.py` pins it to the same three patch points
`simcount_audit.install_counters` uses, so the two cannot drift into disagreeing.

    with counting() as c:
        r = design(8.92, 14.83)
    c["measure_all"], c["analysis"]

WHAT GETS PATCHED, AND WHY THREE NAMES FOR TWO CHOKEPOINTS. `eqrl.evaluator` reaches
measure_all through the module object (`_m.measure_all`), so patching
`eqrl.sim.measures.measure_all` catches it. `eqrl.envs.sequential_env` did
`from eqrl.sim.measures import measure_all` at import time and therefore holds its own
reference, which has to be rebound separately or every env.step() goes uncounted. That
second binding is exactly the kind of thing a hand-count misses.

Counting is not free of side conditions: it patches module globals, so it is for
measurement and reporting, not for concurrent use from multiple threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

#: The module attributes that must be rebound for a count to be complete. Kept as data so
#: a test can compare this list against `simcount_audit.install_counters` rather than
#: trusting that two hand-written patch sets stayed in step.
MEASURE_ALL_BINDINGS = (
    ("eqrl.sim.measures", "measure_all"),      # what eqrl.evaluator resolves through
    ("eqrl.envs.sequential_env", "measure_all"),   # imported by name at module import
)


@contextmanager
def counting() -> Iterator[dict]:
    """Count measure_all calls and SPICE analyses inside the block.

    Yields a dict that is updated live and is final once the block exits. The original
    functions are restored even if the block raises, so a failed run cannot leave the
    process counting into a dead dictionary.
    """
    import importlib

    from eqrl.sim.server import NgspiceServer

    count = {"measure_all": 0, "analysis": 0}

    # Whatever is bound right now is what gets wrapped -- which composes correctly with
    # `guards.seal_direct_access`, in either order: seal-then-count measures the sealed
    # function, count-then-seal leaves the seal wrapping the counter. Both count once.
    saved: list[tuple[object, str, object]] = []
    originals = {}
    for mod_name, attr in MEASURE_ALL_BINDINGS:
        mod = importlib.import_module(mod_name)
        originals[(mod_name, attr)] = getattr(mod, attr)
        saved.append((mod, attr, getattr(mod, attr)))

    # One counter per distinct underlying function, so a name bound twice to the same
    # object is not counted twice.
    def wrap(fn):
        def counted(*a, **kw):
            count["measure_all"] += 1
            return fn(*a, **kw)
        counted.__wrapped__ = fn
        return counted

    base = originals[("eqrl.sim.measures", "measure_all")]
    counted_measure_all = wrap(base)
    for mod, attr, current in saved:
        # Rebind every holder to ONE counter. A holder that had drifted to a different
        # function is left wrapped in its own counter rather than silently replaced,
        # because replacing it would change behaviour instead of measuring it.
        setattr(mod, attr, counted_measure_all if current is base else wrap(current))

    real_analysis = NgspiceServer._analysis

    def counted_analysis(self, cmd):
        count["analysis"] += 1
        return real_analysis(self, cmd)

    NgspiceServer._analysis = counted_analysis
    try:
        yield count
    finally:
        for mod, attr, original in saved:
            setattr(mod, attr, original)
        NgspiceServer._analysis = real_analysis
