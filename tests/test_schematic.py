"""The schematic renderer, which `silq.solve` now calls on every successful run.

`solve.py` wraps the call in try/except so that a drawing bug can never fail a sizing
run whose design and provenance are already written to disk. That is the right choice
there and it is exactly why these tests exist: without them a broken renderer degrades
silently to a printed warning, and the first person to notice is someone watching a demo.

No PDK and no simulator -- `render` is string formatting over the design vector -- so
this runs everywhere, including CI.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from silq.circuits.ctle import DesignVars
from silq.schematic import render

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def svg() -> str:
    # The values from a real solve, so the engineering-notation formatting is exercised
    # on numbers that actually occur rather than on the dataclass defaults.
    return render(
        DesignVars(w_in=22.94e-6, l_in=0.3108e-6, i_tail=641.0e-6,
                   rs=3.44e3, cs=224.3e-15, r_load=2968.4),
        title="SILQ CTLE", subtitle="9.81 dB boost @ 1.42 GHz, 1.70 mW")


def test_is_well_formed_xml(svg):
    # A malformed SVG renders as nothing in a browser and raises nowhere in Python, so
    # parsing it is the only check that catches an unbalanced tag.
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("viewBox"), "no viewBox: the drawing will not scale to its container"


def test_draws_an_actual_circuit(svg):
    """Guard against the failure that matters: a valid, empty SVG."""
    root = ET.fromstring(svg)
    counts = {}
    for el in root.iter():
        counts[el.tag.replace(SVG_NS, "")] = counts.get(el.tag.replace(SVG_NS, ""), 0) + 1
    # A differential CTLE needs two supply rails, both signal paths, and the tail mirror.
    assert counts.get("line", 0) >= 20, f"too few wires to be a schematic: {counts}"
    assert counts.get("polyline", 0) >= 10, f"no routed nets: {counts}"
    assert counts.get("text", 0) >= 20, f"unlabelled schematic: {counts}"


@pytest.mark.parametrize("needle,why", [
    ("22.94", "input-pair width missing"),
    ("3.44", "degeneration resistor missing"),
    ("224", "degeneration capacitor missing"),
    ("641", "tail current missing"),
    ("1.70 mW", "subtitle not rendered"),
])
def test_the_sized_values_reach_the_drawing(svg, needle, why):
    """The drawing must show THIS design, not the topology's defaults.

    A schematic that renders the same picture whatever the agent produced is decoration;
    the whole point of emitting it per-run is that the numbers on it are the numbers the
    search chose.
    """
    assert needle in svg, why


def test_dfe_is_declared_as_not_analog(svg):
    """The 1-tap DFE is RX DSP applied after the SPICE measurement, not a circuit node.

    The audit's sharpest criticism of the eye numbers is that a DFE contribution can be
    read as though it came out of the netlist. The schematic answers that on its own
    face, and that label is load-bearing -- if it is ever dropped, the picture starts
    implying a claim the simulation does not support.
    """
    assert "not an analog node" in svg
    assert "DFE" in svg


def test_w_dfe_is_shown_when_the_tap_is_used(svg):
    # w_dfe = 0 in the fixture; the label must still state it rather than hide the tap.
    assert "w_dfe" in svg
    on = render(DesignVars(w_dfe=0.25))
    assert "0.25" in on, "a non-zero DFE tap must be visible in the drawing"
