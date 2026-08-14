"""Single-parameter sweeps must agree with analog theory.

A pipeline can pass every integrity check, reproduce a reference point, and still be
wired to the wrong node — the failure then shows up as a *direction* error: turning a
knob moves the metric the wrong way. These sweeps are the cheapest way to catch it.

    Rs  up  ->  peaking up        boost = 1 + gm.Rs/2
    I   up  ->  bandwidth up      higher gm pushes the degeneration pole out
    RL  up  ->  DC gain up        A0 = gm.RL / (1 + gm.Rs/2), linear in RL

Each sweep asserts the trend, not just the endpoints: a pipeline that happens to get
the two ends right while behaving non-monotonically in between is still broken.
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from eqrl.circuits.ctle import DesignVars

N_POINTS = 7
MONOTONIC_SLACK = 0.02       # tolerated fraction of the total span per backward step
MIN_TOTAL_CHANGE = 0.10      # endpoint change must be at least 10% of |first| to count


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def assert_increasing(xs, ys, *, what: str, knob: str):
    _assert_direction(xs, ys, +1, what=what, knob=knob)


def assert_decreasing(xs, ys, *, what: str, knob: str):
    _assert_direction(xs, ys, -1, what=what, knob=knob)


def _assert_direction(xs, ys, sign: int, *, what: str, knob: str):
    ys = np.asarray(ys, dtype=float)
    xs = np.asarray(xs, dtype=float)
    assert np.all(np.isfinite(ys)), f"{what} contains non-finite values: {ys}"

    trace = "\n".join(f"    {knob}={x:<12.6g} {what}={y:.6g}" for x, y in zip(xs, ys))
    direction = "increase" if sign > 0 else "decrease"

    # 1. overall trend
    slope = float(np.polyfit(np.arange(len(ys), dtype=float), ys, 1)[0])
    assert slope * sign > 0, (
        f"THEORY VIOLATION: increasing {knob} should {direction} {what}, but the fitted "
        f"slope is {slope:+.6g} per step.\n{trace}"
    )

    # 2. endpoint change is real, not noise
    span = abs(ys[-1] - ys[0])
    scale = max(abs(ys[0]), 1e-12)
    assert span >= scale * MIN_TOTAL_CHANGE, (
        f"{knob} barely moved {what} ({span:.6g}, < {MIN_TOTAL_CHANGE * 100:.0f}% of "
        f"{scale:.6g}). Either the parameter is not reaching the netlist or the metric "
        f"is not measuring what we think.\n{trace}"
    )

    # 3. no meaningful reversals along the way
    total = abs(ys[-1] - ys[0]) or 1.0
    steps = np.diff(ys) * sign
    worst = float(np.min(steps))
    assert worst >= -MONOTONIC_SLACK * total, (
        f"{what} reverses direction during the {knob} sweep (worst backward step "
        f"{worst:+.6g}, tolerance {-MONOTONIC_SLACK * total:+.6g}).\n{trace}"
    )


def bandwidth_ghz(freq: np.ndarray, mag_db: np.ndarray) -> float:
    """Upper -3 dB point relative to the DC level, searching above the peak.

    Raises if the sweep never reaches -3 dB. Returning the last frequency instead
    would silently report 'bandwidth == sweep edge' for every design, which makes the
    metric look constant and hides the real defect (too narrow an AC sweep).
    """
    dc = float(mag_db[0])
    i = int(np.argmax(mag_db))
    below = np.where(mag_db[i:] <= dc - 3.0)[0]
    if not below.size:
        raise AssertionError(
            f"AC sweep never falls to DC-3dB ({dc - 3.0:.2f} dB); it ends at "
            f"{mag_db[-1]:.2f} dB at {freq[-1] / 1e9:.1f} GHz. Bandwidth is not "
            "measurable over this frequency range — widen the sweep."
        )
    return float(freq[i + below[0]] / 1e9)


AC_WIDE = "ac dec 30 1e6 1e13\nlet vdb = db(v(outp)-v(outn))\nwrdata $OUT vdb"
"""A wider AC control than the production `ngspice_runner.ac` (1e6..1e10).

Bandwidth for this topology lands around 57 GHz, so the production sweep cannot see
it. These tests drive the same netlist + simulator + parser through the existing
`run()` entry point with a wider span, so the pipeline is still what is under test.
"""


# ---------------------------------------------------------------------------
# Layer 1 — the closed form. Runs everywhere, no tooling required.
# ---------------------------------------------------------------------------

def _analytic(dv: DesignVars, n: int = 20001, fmax_hz: float = 1e13):
    """fmax_hz defaults far above the 10 GHz production sweep on purpose: the -3 dB
    point of this topology is ~57 GHz at nominal bias, so a 10 GHz ceiling pins every
    bandwidth reading to the sweep edge and the metric looks constant."""
    from tests.test_reference import analytic_response, C_LOAD, VOV
    f = np.logspace(6, math.log10(fmax_hz), n)
    db = 20 * np.log10(np.abs(analytic_response(dv.i_tail / VOV, dv.rs, dv.cs,
                                                dv.r_load, C_LOAD, f)))
    return f, db


class TestTheoryItself:
    """Guards the reference model. If these fail, the expectations below are wrong."""

    def test_rs_increases_peaking(self):
        rs = np.linspace(200.0, 3000.0, N_POINTS)
        boost = []
        for v in rs:
            f, db = _analytic(DesignVars(rs=float(v)))
            boost.append(float(db.max() - db[0]))
        assert_increasing(rs, boost, what="peaking_db", knob="rs")

    def test_itail_increases_bandwidth(self):
        it = np.linspace(0.5e-3, 4e-3, N_POINTS)
        bw = []
        for v in it:
            f, db = _analytic(DesignVars(i_tail=float(v)))
            bw.append(bandwidth_ghz(f, db))
        assert_increasing(it, bw, what="bandwidth_ghz", knob="i_tail")

    def test_rload_increases_dc_gain(self):
        rl = np.linspace(200.0, 3000.0, N_POINTS)
        dc = []
        for v in rl:
            f, db = _analytic(DesignVars(r_load=float(v)))
            dc.append(float(db[0]))
        assert_increasing(rl, dc, what="dc_gain_db", knob="r_load")

    def test_rload_decreases_bandwidth(self):
        """Sanity on the other side of the same knob: bigger RL, lower load pole."""
        rl = np.linspace(400.0, 3000.0, N_POINTS)
        bw = []
        for v in rl:
            f, db = _analytic(DesignVars(r_load=float(v)))
            bw.append(bandwidth_ghz(f, db))
        assert_decreasing(rl, bw, what="bandwidth_ghz", knob="r_load")


# ---------------------------------------------------------------------------
# Layer 2 — the real pipeline
# ---------------------------------------------------------------------------

def _ngspice() -> bool:
    return shutil.which("ngspice") is not None


def _pdk() -> bool:
    try:
        from eqrl.circuits import pdk
        return pdk.available()
    except (ImportError, FileNotFoundError):
        return False


def _sweep_pipeline(field: str, values, models: str, *, wide: bool = False):
    """Run the real netlist + ngspice + parser once per value.

    wide=True uses AC_WIDE so bandwidth is reachable; everything else is identical to
    the production path.
    """
    from eqrl.circuits.ctle import netlist
    from eqrl.sim.ngspice_runner import ac, run

    out = []
    for v in values:
        dv = DesignVars(**{field: float(v)})
        deck = netlist(dv, analysis="none", models=models)
        if wide:
            res = run(deck, control=AC_WIDE)
            arr = np.atleast_2d(res["data"])
            freq, mag = arr[:, 0], arr[:, 1]
        else:
            r = ac(deck)
            freq, mag = r["freq"], r["mag_db"]
        assert freq.size > 50, f"AC sweep for {field}={v:g} returned {freq.size} points"
        entry = {
            "dc_gain_db": float(mag[0]),
            "peaking_db": float(mag.max() - mag[0]),
            "peak_freq_ghz": float(freq[int(np.argmax(mag))] / 1e9),
        }
        if wide:
            entry["bandwidth_ghz"] = bandwidth_ghz(freq, mag)
        out.append(entry)
    return out


@pytest.mark.skipif(not _ngspice(), reason="ngspice not installed")
class TestPipelineBehavioural:
    """Needs ngspice only — runs in CI without the PDK."""

    def test_rs_increases_peaking(self):
        rs = np.linspace(200.0, 3000.0, N_POINTS)
        res = _sweep_pipeline("rs", rs, "behavioral")
        assert_increasing(rs, [r["peaking_db"] for r in res],
                          what="peaking_db", knob="rs")

    def test_itail_increases_bandwidth(self):
        it = np.linspace(0.5e-3, 4e-3, N_POINTS)
        res = _sweep_pipeline("i_tail", it, "behavioral", wide=True)
        assert_increasing(it, [r["bandwidth_ghz"] for r in res],
                          what="bandwidth_ghz", knob="i_tail")

    def test_production_ac_sweep_can_see_bandwidth(self):
        """The production sweep stops at 10 GHz; -3 dB lands near 57 GHz. Until the
        sweep is widened, no bandwidth number from `ngspice_runner.ac` or
        `NgspiceServer.ac` is a bandwidth number."""
        from eqrl.circuits.ctle import netlist
        from eqrl.sim.ngspice_runner import ac
        r = ac(netlist(DesignVars(l_in=0.3e-6), analysis="none", models="behavioral"))
        bandwidth_ghz(r["freq"], r["mag_db"])   # raises if the sweep cannot reach it

    def test_rload_increases_dc_gain(self):
        rl = np.linspace(200.0, 3000.0, N_POINTS)
        res = _sweep_pipeline("r_load", rl, "behavioral")
        assert_increasing(rl, [r["dc_gain_db"] for r in res],
                          what="dc_gain_db", knob="r_load")

    def test_cs_decreases_peak_frequency(self):
        """The zero sits at 1/(2.pi.Rs.Cs): more Cs, lower peak."""
        cs = np.linspace(50e-15, 800e-15, N_POINTS)
        res = _sweep_pipeline("cs", cs, "behavioral")
        assert_decreasing(cs, [r["peak_freq_ghz"] for r in res],
                          what="peak_freq_ghz", knob="cs")


@pytest.mark.skipif(not (_ngspice() and _pdk()), reason="ngspice + SKY130 PDK required")
class TestPipelineSky130:
    """The transistor-level netlist must obey the same directions.

    If behavioural passes and SKY130 fails, the defect is in the transistor deck —
    bias point, device sizing, or the tail sources — not in the measurement code.
    """

    def test_rs_increases_peaking(self):
        rs = np.linspace(200.0, 3000.0, N_POINTS)
        res = _sweep_pipeline("rs", rs, "sky130")
        assert_increasing(rs, [r["peaking_db"] for r in res],
                          what="peaking_db", knob="rs")

    def test_itail_increases_bandwidth(self):
        it = np.linspace(0.5e-3, 4e-3, N_POINTS)
        res = _sweep_pipeline("i_tail", it, "sky130", wide=True)
        assert_increasing(it, [r["bandwidth_ghz"] for r in res],
                          what="bandwidth_ghz", knob="i_tail")

    def test_rload_increases_dc_gain(self):
        rl = np.linspace(200.0, 3000.0, N_POINTS)
        res = _sweep_pipeline("r_load", rl, "sky130")
        assert_increasing(rl, [r["dc_gain_db"] for r in res],
                          what="dc_gain_db", knob="r_load")


@pytest.mark.skipif(not _ngspice(), reason="ngspice not installed")
def test_w_dfe_is_not_a_dead_parameter():
    """w_dfe is in ACTION_SPACE and in the design vector. If it reaches nothing in the
    netlist, the agent is optimizing a knob wired to no circuit — the search will look
    healthy while one seventh of its budget does nothing.

    This test documents the expectation. It is currently expected to FAIL, because the
    1-tap DFE is absent from both the sky130 deck and the param deck.
    """
    res = _sweep_pipeline("w_dfe", np.linspace(0.0, 0.5, 4), "behavioral")
    metrics = {k: {round(r[k], 9) for r in res} for k in res[0]}
    changed = [k for k, vals in metrics.items() if len(vals) > 1]
    assert changed, (
        "w_dfe changed no measured quantity across its full range. It is a design "
        "variable connected to nothing — either wire the 1-tap DFE into the netlist or "
        "remove it from ACTION_SPACE."
    )
