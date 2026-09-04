"""The 1-tap DFE receiver stage (circuits/dfe.py).

A DFE is a sampled, decision-driven block with no small-signal `.ac` response, so it is a
separate transient stage after the CTLE. These tests pin its behaviour: the tap cancels
the first post-cursor, the eye peaks when the tap matches it, and the exported netlist
actually contains the summing node.
"""
import shutil

import numpy as np
import pytest

from eqrl.circuits.dfe import (dfe_stage_eye, dfe_testbench_netlist, optimal_tap)


def _ngspice():
    return shutil.which("ngspice") is not None


def test_netlist_contains_the_dfe_summing_node():
    """The exported schematic must literally contain the 1-tap DFE (a summer that
    subtracts tap * previous decision) — the topology Astera asked for."""
    deck, bits = dfe_testbench_netlist(c0=0.5, c1=0.3, tap=0.3)
    assert "Eout outn 0 poly(2) rxn 0 nbd 0" in deck, "no DFE summing node in the netlist"
    assert "Erx rxn 0 poly(2)" in deck, "no received-signal (main + post-cursor) node"
    assert len(bits) > 0


@pytest.mark.skipif(not _ngspice(), reason="ngspice not installed")
def test_optimal_tap_maximizes_the_eye():
    """A 1-tap DFE opens the eye most when its tap equals the first post-cursor c1 —
    the value an adaptive (LMS) DFE converges to."""
    c0, c1 = 0.5, 0.28
    taps = np.linspace(0.0, 2.0 * c1, 9)
    eyes = [dfe_stage_eye(t, c0=c0, c1=c1) for t in taps]
    best = taps[int(np.argmax(eyes))]
    assert abs(best - optimal_tap(c1)) <= (taps[1] - taps[0]) + 1e-12, (
        f"eye peaked at tap={best:.3f}, expected ~c1={c1}"
    )


@pytest.mark.skipif(not _ngspice(), reason="ngspice not installed")
def test_dfe_improves_the_eye_over_no_dfe():
    """Turning the DFE on (tap=c1) must open the eye vs leaving it off (tap=0)."""
    c0, c1 = 0.5, 0.3
    off = dfe_stage_eye(0.0, c0=c0, c1=c1)
    on = dfe_stage_eye(c1, c0=c0, c1=c1)
    assert on > off, f"DFE did not help: off={off:.3f} on={on:.3f}"


@pytest.mark.skipif(not _ngspice(), reason="ngspice not installed")
def test_dfe_handles_an_inverting_ctle():
    """Real CTLEs invert (c0 < 0). The eye height must still come out positive and the
    DFE must still help, because the tap sign is carried by c1."""
    c0, c1 = -0.5, -0.28
    off = dfe_stage_eye(0.0, c0=c0, c1=c1)
    on = dfe_stage_eye(c1, c0=c0, c1=c1)
    assert off > 0 and on > off, f"inverting-link DFE failed: off={off:.3f} on={on:.3f}"
