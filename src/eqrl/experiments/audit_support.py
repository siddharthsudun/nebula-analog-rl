"""Prospective audit helpers. No simulator imports or work at module import.

Use only in a dedicated, serial audit process: the cost meter wraps process globals.
It never changes a reward, threshold, model, or controller. Limits are ceilings, not
predictions of elapsed time. Existing files are never replaced.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path


class AuditBudgetExceeded(BaseException):
    """Cannot be swallowed by the frozen solvers' candidate-error handlers."""


def fingerprint(path):
    p = Path(path)
    return {"path": p.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "mtime_ns": p.stat().st_mtime_ns, "bytes": p.stat().st_size}


def write_new(path, data):
    """Exclusive create. A crash may leave a partial file, never overwrite evidence."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


@contextmanager
def bounded_calls(max_measure_all, max_analyses):
    """Count actual entries, stopping BEFORE a call would exceed either ceiling.

    Counts include failures, resets and internal verification; separate independent
    verification belongs in its own block/process. Nested simcount.counting wrappers
    call through these wrappers. Direct measure_all aliases are also covered.
    A wall-clock deadline requires the parent subprocess timeout, since a native
    simulator call can hang without returning to Python.
    """
    if max_measure_all < 0 or max_analyses < 1:
        raise ValueError("invalid audit budget")
    from eqrl.sim.server import NgspiceServer
    from eqrl.simcount import MEASURE_ALL_BINDINGS
    counter = {"measure_all": 0, "analysis": 0}
    saved = []
    wrappers = {}

    def wrapped(fn):
        def call(*args, **kwargs):
            if counter["measure_all"] >= max_measure_all:
                raise AuditBudgetExceeded("measure_all ceiling reached")
            counter["measure_all"] += 1
            return fn(*args, **kwargs)
        return call

    for name, attr in MEASURE_ALL_BINDINGS:
        module = importlib.import_module(name)
        fn = getattr(module, attr)
        saved.append((module, attr, fn))
        if id(fn) not in wrappers:
            wrappers[id(fn)] = wrapped(fn)
        setattr(module, attr, wrappers[id(fn)])
    analysis = NgspiceServer._analysis

    def counted_analysis(self, cmd):
        if counter["analysis"] >= max_analyses:
            raise AuditBudgetExceeded("SPICE analysis ceiling reached")
        counter["analysis"] += 1
        return analysis(self, cmd)

    NgspiceServer._analysis = counted_analysis
    try:
        yield counter
    finally:
        NgspiceServer._analysis = analysis
        for module, attr, fn in saved:
            setattr(module, attr, fn)
