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


def bandwidth_ghz(freq: np.ndarray, mag_db: np.ndarray, ref: str = "peak") -> float:
    """Upper -3 dB point, searching above the peak.

    ref="peak" : 3 dB below the maximum. The only definition that works for an
                 ATTENUATING equalizer, where the DC level is the floor rather than a
                 plateau — the measured SKY130 stage has DC gain -10.6 dB and peaks at
                 -0.35 dB, so it never falls 3 dB below DC at any frequency.
    ref="dc"   : 3 dB below the DC level. Valid only when the stage has DC gain.

    Raises if the sweep never reaches -3 dB, rather than returning the sweep edge —
    reporting 'bandwidth == last frequency' for every design makes the metric look
    constant and hides the real defect.
    """
    i = int(np.argmax(mag_db))
    level = float(mag_db[i]) if ref == "peak" else float(mag_db[0])
    below = np.where(mag_db[i:] <= level - 3.0)[0]
    if not below.size:
        raise AssertionError(
            f"AC sweep never falls to {ref}-3dB ({level - 3.0:.2f} dB); it ends at "
            f"{mag_db[-1]:.2f} dB at {freq[-1] / 1e9:.1f} GHz. Bandwidth is not "
            f"measurable over this range with ref={ref!r}."
        )
    return float(freq[i + below[0]] / 1e9)


def bias_valid_itail(r_load: float, vdd: float = 1.8, headroom_v: float = 0.4) -> float:
    """Largest tail current the supply can sustain through a given load.

    MEASURED consequence of ignoring this: at i_tail=4 mA into the default 1 kohm load,
    each leg drops 2.0 V across a 1.8 V rail. The stage stops amplifying entirely —
    there is no peak anywhere, just gate-drain feedthrough climbing monotonically from
    -15.9 dB at 1 GHz to -9.5 dB at 10 THz. argmax then lands on the last sweep point
    and every derived metric is meaningless.
    """
    return 2.0 * (vdd - headroom_v) / r_load


def bias_valid_rload(i_tail: float, vdd: float = 1.8, headroom_v: float = 0.4) -> float:
    """Largest load resistor that still leaves the output inside the rails.

    Each leg carries i_tail/2 through R_load, so the DC drop is (i_tail/2)*R_load. Once
    that approaches VDD the output node is pinned at the bottom rail and the input
    device leaves saturation — the stage stops being an amplifier. Sweeps that ignore
    this are not testing the circuit, they are testing a collapsed bias point.
    """
    return (vdd - headroom_v) / (i_tail / 2.0)


AC_WIDE = "ac dec 30 1e6 1e11\nlet vdb = db(v(outp)-v(outn))\nwrdata $OUT vdb"
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


def peak_at_edge(freq: np.ndarray, mag_db: np.ndarray, tol: int = 1) -> bool:
    """A 'peak' on the last sweep sample is a stopped sweep, not a resonance."""
    return int(np.argmax(mag_db)) >= len(mag_db) - 1 - tol


def _sweep_pipeline(field: str, values, models: str, *, wide: bool = False,
                    raw: bool = False):
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
            if raw:
                out.append((freq, mag))
                continue
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

    def test_itail_increases_bandwidth_while_the_bias_survives(self):
        """MEASURED: 0.5 mA -> -3dB at 8.91 GHz; 2.0 mA -> 12.59 GHz. Bandwidth rises
        with current exactly as theory says — but only up to the point where the leg
        current can still fit across the supply. Past ~2.8 mA into the default 1 kohm
        load the stage collapses (see the companion test below)."""
        r_load = DesignVars().r_load
        it = np.linspace(0.5e-3, bias_valid_itail(r_load), N_POINTS)
        res = _sweep_pipeline("i_tail", it, "sky130", wide=True)
        assert_increasing(it, [r["bandwidth_ghz"] for r in res],
                          what="bandwidth_ghz", knob="i_tail")

    def test_current_mirror_prevents_the_overcurrent_collapse(self):
        """The current mirror removes an entire class of degenerate designs.

        HISTORY: with ideal tail sources this test asserted the OPPOSITE. Requesting
        4 mA into a 1 kohm load meant forcing 2 V across a 1.8 V supply; an ideal source
        obliged, the output pinned at the rail, the input devices left saturation, and
        the response degenerated into pure gate-drain feedthrough that climbed
        monotonically to the end of the sweep. 'Peak frequency' became an artifact of
        where the sweep stopped.

        A real mirror cannot do that. When the drain voltage falls, the mirror device
        leaves saturation and delivers less current — the circuit self-limits instead of
        collapsing. Measured after the fix: DC gain +9.43 dB with a clean roll-off,
        where the same request previously produced a dead stage.

        This is a second, unadvertised benefit of the bias fix: it shrinks the invalid
        region of the search space rather than merely making PVT more honest.
        """
        r_load = DesignVars().r_load
        hard_it = bias_valid_itail(r_load) * 1.6
        assert (hard_it / 2) * r_load > 1.8, "sanity: an ideal source would exceed VDD here"

        freq, mag = _sweep_pipeline("i_tail", [hard_it], "sky130", wide=True, raw=True)[0]
        assert not peak_at_edge(freq, mag), (
            "response still peaks at the last sweep sample — the stage is behaving like "
            "feedthrough, so the mirror is not limiting as expected"
        )
        i = int(np.argmax(mag))
        assert mag[i] > mag[-1] + 3.0, (
            f"expected a real peak above the high-frequency floor, got peak {mag[i]:.2f} dB "
            f"vs {mag[-1]:.2f} dB at the sweep end"
        )

    def test_rload_increases_dc_gain_while_the_bias_survives(self):
        """Swept only over loads the supply can actually sustain.

        MEASURED: with the 2 mA default tail, DC gain rises to +2.8 dB at 1133 ohm,
        turns over by 1600 ohm, and collapses to -39.7 dB by 3000 ohm. That is not a
        gain curve — past ~1400 ohm the 1 mA leg current drops more than VDD across the
        load, the output pins at the bottom rail and the input device leaves saturation.
        Theory only describes the stage while it is biased, so the sweep stops there.
        `test_rload_beyond_the_supply_collapses_the_stage` covers the other side.
        """
        i_tail = DesignVars().i_tail
        rl_max = bias_valid_rload(i_tail)
        rl = np.linspace(200.0, rl_max, N_POINTS)
        res = _sweep_pipeline("r_load", rl, "sky130")
        assert_increasing(rl, [r["dc_gain_db"] for r in res],
                          what="dc_gain_db", knob="r_load")

    def test_current_mirror_softens_the_large_load_cliff(self):
        """Same fix, reached from the load axis.

        HISTORY: with ideal tail sources, DC gain rose to +2.8 dB at 1133 ohm, turned
        over by 1600, and fell off a cliff to -39.7 dB by 3000 — the bias had collapsed,
        but the simulator still returned a plausible-looking number that the optimizer
        read as an ordinary bad reward.

        With the mirror the degradation is graceful: the mirror device runs out of
        headroom and reduces its current rather than pinning the output. The stage gets
        worse, which is correct, instead of dying while still reporting a number.
        """
        i_tail = DesignVars().i_tail
        rl_max = bias_valid_rload(i_tail)
        ok = _sweep_pipeline("r_load", [rl_max * 0.8], "sky130")[0]
        far = _sweep_pipeline("r_load", [rl_max * 2.5], "sky130")[0]
        assert far["dc_gain_db"] > ok["dc_gain_db"] - 30.0, (
            f"gain fell off a cliff ({ok['dc_gain_db']:.2f} -> {far['dc_gain_db']:.2f} dB) — "
            "that is the collapse signature the mirror was supposed to remove"
        )


@pytest.mark.skipif(not _ngspice(), reason="ngspice not installed")
@pytest.mark.xfail(strict=True, reason=(
    "CONFIRMED BY SIMULATION: w_dfe changes no measured quantity. It appears in "
    "ACTION_SPACE and in every exported design, but dv_to_params() omits it and "
    "neither param_deck() nor _core_sky130() contains a DFE element, so one seventh "
    "of the action space is wired to nothing. strict=True so this flips to a failure "
    "the moment the DFE is actually connected — at which point delete the marker."))
def test_w_dfe_is_not_a_dead_parameter():
    """w_dfe is in ACTION_SPACE and in the design vector. If it reaches nothing in the
    netlist, the agent is optimizing a knob wired to no circuit — the search will look
    healthy while one seventh of its budget does nothing.
    """
    res = _sweep_pipeline("w_dfe", np.linspace(0.0, 0.5, 4), "behavioral")
    metrics = {k: {round(r[k], 9) for r in res} for k in res[0]}
    changed = [k for k, vals in metrics.items() if len(vals) > 1]
    assert changed, (
        "w_dfe changed no measured quantity across its full range. It is a design "
        "variable connected to nothing — either wire the 1-tap DFE into the netlist or "
        "remove it from ACTION_SPACE."
    )
