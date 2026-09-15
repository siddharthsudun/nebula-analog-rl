"""The property that makes every other check meaningful: same input, same verdict.

The guard layer was built to catch bad simulations. Nothing in it asserted that evaluating
the same design twice returns the same answer, and that omission hid the most consequential
defect in the project: `raw_eval` probed the operating point WITHOUT priming the server
with the candidate, so Tier 2 read whatever parameters were left over -- the deck defaults
on a fresh server, the previous candidate on a warm one -- and compared them against this
candidate's requested values.

Measured before the fix: the same design evaluated twice in a row changed verdict on 9 of
14 designs. On a fresh server it was stable but uniformly wrong.

These tests are deliberately about *consistency*, not about which verdict is correct. A
test that pinned specific verdicts would need updating whenever a threshold moved; these
stay valid for any thresholds, and they fail loudly if evaluation ever becomes
order-dependent again.
"""
from __future__ import annotations

import pytest

from silq.circuits.ctle import DesignVars

pytestmark = pytest.mark.skipif(
    not __import__("silq.circuits.pdk", fromlist=["available"]).available(),
    reason="needs the SKY130 PDK and a working ngspice",
)


def _verdict(ev, dv):
    v = ev.evaluate(dv, vdd=1.8)
    return "__valid__" if v.is_valid else v.check.value


@pytest.fixture(scope="module")
def ev():
    from silq.evaluator import build_evaluator
    from silq.specs import DEFAULT_SPEC
    return build_evaluator(DEFAULT_SPEC, corner="tt", fast=True)


#: Three designs spread across the space. `l_in` is kept strictly inside the PDK bound so
#: T2.8 does not mask what Tier 2 says about the operating point.
A = DesignVars(w_in=12e-6, l_in=0.30e-6, i_tail=0.20e-3, rs=2000.0, cs=300e-15,
               r_load=1200.0)
B = DesignVars(w_in=60e-6, l_in=0.50e-6, i_tail=0.45e-3, rs=900.0, cs=1.2e-12,
               r_load=3000.0)
FAR = DesignVars(w_in=95e-6, l_in=0.95e-6, i_tail=0.06e-3, rs=4800.0, cs=1.9e-12,
                 r_load=4700.0)


@pytest.mark.parametrize("dv", [A, B, FAR], ids=["A", "B", "FAR"])
def test_same_design_twice_gives_the_same_verdict(ev, dv):
    assert _verdict(ev, dv) == _verdict(ev, dv)


@pytest.mark.parametrize("dv", [A, B], ids=["A", "B"])
def test_verdict_survives_an_unrelated_evaluation_in_between(ev, dv):
    """The harder case: state left by a very different design must not leak in."""
    first = _verdict(ev, dv)
    _verdict(ev, FAR)
    assert _verdict(ev, dv) == first


def test_verdict_does_not_depend_on_evaluation_order(ev):
    a_then_b = [_verdict(ev, A), _verdict(ev, B)]
    b_then_a = [_verdict(ev, B), _verdict(ev, A)]
    assert a_then_b == [b_then_a[1], b_then_a[0]]


def test_the_probed_operating_point_belongs_to_the_design_evaluated(ev):
    """Directly pins the defect rather than its symptom.

    Tier 2.6 compares delivered tail current against `dv.i_tail`. If the probe describes
    some other design, the delivered current bears no relation to what was requested. Two
    designs an order of magnitude apart in tail current must not report the same one.
    """
    from silq.evaluator import make_raw_eval
    from silq.guards import ArtifactStore
    import tempfile
    from pathlib import Path

    raw = make_raw_eval(corner="tt", fast=True)
    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(Path(d) / "raw")
        lo = DesignVars(w_in=12e-6, l_in=0.30e-6, i_tail=0.06e-3, rs=2000.0,
                        cs=300e-15, r_load=1200.0)
        hi = DesignVars(w_in=12e-6, l_in=0.30e-6, i_tail=0.50e-3, rs=2000.0,
                        cs=300e-15, r_load=1200.0)
        _, _, op_lo = raw(lo, artifacts=store.new_run(), vdd=1.8)
        _, _, op_hi = raw(hi, artifacts=store.new_run(), vdd=1.8)

    i_lo = sum(op_lo.tail_currents.values())
    i_hi = sum(op_hi.tail_currents.values())
    # An 8x request ratio must show up as a clearly larger delivered current.
    assert i_hi > 2.0 * i_lo, (
        f"probed tail current barely moved ({i_lo:.6g} A -> {i_hi:.6g} A) while the "
        f"request went {lo.i_tail:.6g} -> {hi.i_tail:.6g} A. The operating point does "
        "not belong to the design being evaluated."
    )


def test_probe_reflects_the_requested_current_not_the_deck_default(ev):
    """The fresh-server symptom: probing the deck's own defaults.

    The server deck defaults to i_tail = 2 mA. A candidate asking for 0.1 mA must not
    come back with the default's delivered current, which is what made T2.6 fire on
    nearly everything.
    """
    from silq.evaluator import make_raw_eval
    from silq.guards import ArtifactStore
    import tempfile
    from pathlib import Path

    raw = make_raw_eval(corner="tt", fast=True)
    dv = DesignVars(w_in=12e-6, l_in=0.30e-6, i_tail=0.10e-3, rs=2000.0,
                    cs=300e-15, r_load=1200.0)
    with tempfile.TemporaryDirectory() as d:
        store = ArtifactStore(Path(d) / "raw")
        _, _, op = raw(dv, artifacts=store.new_run(), vdd=1.8)
    delivered = sum(op.tail_currents.values())
    assert delivered < 0.5e-3, (
        f"delivered tail current {delivered:.6g} A is nowhere near the requested "
        f"{dv.i_tail:.6g} A and is consistent with the deck default of 2 mA"
    )
