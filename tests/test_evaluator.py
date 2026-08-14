"""Wiring tests: the composed evaluator must not lose a check on the way through.

Run with fake servers, so they exercise the composition today rather than waiting on
the toolchain.
"""
from __future__ import annotations

import io

import pytest

from eqrl.circuits.ctle import DesignVars
from eqrl.guards import (
    ArtifactStore, Check, DeviceOP, GuardConfigError, Invalid, OperatingPoint,
    SearchMonitor, Valid,
)
from eqrl.sim.measures import Measures
from eqrl.sim.ngspice_runner import NgspiceError
from eqrl.specs import DEFAULT_SPEC


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeNg:
    def __init__(self, stdout="ngspice ok", stderr=""):
        self.stdout, self.stderr = stdout, stderr

    def exec_command(self, cmd):
        return ""


class FakeNgNoCapture:
    """A backend that swallows its diagnostics — check 2 could never fire on it."""

    def exec_command(self, cmd):
        return ""


class FakeServer:
    """Mimics NgspiceServer, including the scratch dir it writes .data files into."""

    def __init__(self, ng=None, corner="tt", scratch=None):
        self._ng = ng if ng is not None else FakeNg()
        self.corner = corner
        self._dir = scratch
        if scratch is not None:
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "ac.data").write_text("1.0 2.0\n2.0 3.0\n")

    def set_corner(self, corner):
        self.corner = corner


def sane_op():
    return OperatingPoint(
        devices=(DeviceOP("XM1", vds=0.6, vdsat=0.2), DeviceOP("XM2", vds=0.6, vdsat=0.2)),
        node_voltages={"outp": 1.2, "outn": 1.2},
        tail_currents={"Itp": 1e-3, "Itn": 1e-3},
    )


def marginal() -> Measures:
    """Valid under Tier 4 and not close enough to spec to trip check 20."""
    return Measures(dc_gain_db=5.0, peak_gain_db=14.0, boost_db=9.0, peak_freq_ghz=2.0,
                    hd3_db=-35.0, noise_vrms=1.2e-3, power_w=11e-3, area_mm2=0.040,
                    eye_h_ui=0.45, eye_v_mv=120.0, ok=True)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """build_evaluator with the simulator faked out."""
    from eqrl import evaluator as ev_mod
    from eqrl.sim import measures as m_mod

    monkeypatch.setattr(ev_mod, "probe_operating_point", lambda srv, **kw: sane_op())
    monkeypatch.setattr(m_mod, "measure_all", lambda dv, **kw: marginal())
    # The real netlist builder needs the SKY130 .lib on disk; stand in for it so these
    # wiring tests run before the PDK is installed. `test_missing_pdk_stops_the_run`
    # below covers the un-patched path.
    monkeypatch.setattr(ev_mod, "netlist",
                        lambda dv, **kw: "* fake deck\nXM1 outp inp sp 0 "
                                         "sky130_fd_pr__nfet_01v8\n.end\n")

    def make(**kw):
        return ev_mod.build_evaluator(
            DEFAULT_SPEC,
            store=ArtifactStore(tmp_path / "raw"),
            server_factory=lambda c: FakeServer(corner=c, scratch=tmp_path / f"srv_{c}"),
            **kw,
        )
    return make


DV = DesignVars(l_in=0.30e-6, i_tail=2e-3, r_load=1e3)


# ---------------------------------------------------------------------------

class TestBackendGate:

    def test_refuses_a_backend_that_cannot_surface_diagnostics(self, tmp_path):
        """Refusing at setup beats returning an evaluator whose check 2 is a no-op."""
        from eqrl.evaluator import build_evaluator
        with pytest.raises(GuardConfigError, match="check 2"):
            build_evaluator(DEFAULT_SPEC,
                            store=ArtifactStore(tmp_path / "raw"),
                            server_factory=lambda c: FakeServer(FakeNgNoCapture()))

    def test_can_be_overridden_deliberately(self, tmp_path):
        from eqrl.evaluator import build_evaluator
        ev = build_evaluator(DEFAULT_SPEC, store=ArtifactStore(tmp_path / "raw"),
                             server_factory=lambda c: FakeServer(FakeNgNoCapture()),
                             check_backend=False)
        assert ev is not None

    def test_missing_ngspice_handle_is_rejected(self, tmp_path):
        from eqrl.evaluator import build_evaluator

        class Bare:
            pass

        with pytest.raises(GuardConfigError):
            build_evaluator(DEFAULT_SPEC, store=ArtifactStore(tmp_path / "raw"),
                            server_factory=lambda c: Bare())


class TestComposition:

    def test_happy_path_is_valid(self, wired):
        v = wired().evaluate(DV, vdd=1.8)
        assert isinstance(v, Valid)
        assert v.unwrap().boost_db == 9.0

    def test_artifacts_carry_the_netlist_and_params(self, wired):
        v = wired().evaluate(DV, vdd=1.8)
        deck = (v.artifact_dir / "netlist.cir").read_text()
        assert "sky130_fd_pr__nfet_01v8" in deck
        import json
        meta = json.loads((v.artifact_dir / "meta.json").read_text())
        assert meta["corner"] == "tt" and meta["exit_code"] == 0

    def test_simulator_data_is_copied_out_of_the_doomed_scratch_dir(self, wired):
        """NgspiceServer's TemporaryDirectory dies with the server. If the .data files
        are not adopted, the raw output is gone and check 3 has nothing to inspect."""
        v = wired().evaluate(DV, vdd=1.8)
        assert (v.artifact_dir / "ac.data").exists()
        assert (v.artifact_dir / "ac.data").read_text().startswith("1.0 2.0")

    def test_simulator_error_becomes_invalid_with_nonzero_exit(self, wired, monkeypatch):
        from eqrl import evaluator as ev_mod
        monkeypatch.setattr(ev_mod, "probe_operating_point",
                            lambda srv, **kw: (_ for _ in ()).throw(
                                NgspiceError("non-convergent: op")))
        v = wired().evaluate(DV, vdd=1.8)
        assert isinstance(v, Invalid)
        assert "non-convergent" in v.reason

    def test_measure_all_ok_false_is_a_run_failure_not_a_metric(self, wired, monkeypatch):
        """A zeroed Measures must not reach Tier 4, where it would read as merely
        'implausible' rather than 'the simulation did not run'."""
        from eqrl.sim import measures as m_mod
        monkeypatch.setattr(m_mod, "measure_all", lambda dv, **kw: Measures(ok=False))
        v = wired().evaluate(DV, vdd=1.8)
        assert isinstance(v, Invalid)
        assert "ok=False" in v.reason

    def test_tier2_still_fires_through_the_wiring(self, wired, monkeypatch):
        from eqrl import evaluator as ev_mod
        triode = OperatingPoint(
            devices=(DeviceOP("XM2", vds=0.01, vdsat=0.30),),
            node_voltages={"outp": 1.2}, tail_currents={"Itp": 1e-3, "Itn": 1e-3})
        monkeypatch.setattr(ev_mod, "probe_operating_point", lambda srv, **kw: triode)
        v = wired().evaluate(DV, vdd=1.8)
        assert isinstance(v, Invalid) and v.check is Check.T2_NOT_SATURATED
        assert "XM2" in v.reason

    def test_tier4_still_fires_through_the_wiring(self, wired, monkeypatch):
        from eqrl.sim import measures as m_mod
        monkeypatch.setattr(m_mod, "measure_all",
                            lambda dv, **kw: Measures(**{**marginal().as_dict(),
                                                         "boost_db": 99.0}))
        v = wired().evaluate(DV, vdd=1.8)
        assert isinstance(v, Invalid) and v.check is Check.T4_PEAKING

    def test_operating_point_can_be_disabled_only_explicitly(self, wired, monkeypatch):
        from eqrl import evaluator as ev_mod
        monkeypatch.setattr(ev_mod, "probe_operating_point",
                            lambda srv, **kw: pytest.fail("should not be probed"))
        ev = wired(with_operating_point=False)
        assert isinstance(ev.evaluate(DV, vdd=1.8), Valid)

    def test_monitor_is_fed(self, wired, tmp_path):
        mon = SearchMonitor(DEFAULT_SPEC, halt_path=tmp_path / "HALT.txt",
                            stream=io.StringIO())
        ev = wired(monitor=mon)
        for _ in range(3):
            ev.evaluate(DV, vdd=1.8)
        assert len(mon.records) == 3
        assert all(r.valid for r in mon.records)


class TestSetupFailuresStopTheRun:

    def test_missing_pdk_stops_the_run_rather_than_failing_one_candidate(
            self, monkeypatch, tmp_path):
        """A deck we cannot record is a run we cannot reproduce. This must NOT be
        downgraded to a per-candidate INVALID, or a whole sweep would quietly report
        'every design failed' when the real problem is an unset PDK_ROOT."""
        from eqrl import evaluator as ev_mod
        from eqrl.sim import measures as m_mod

        monkeypatch.setattr(ev_mod, "probe_operating_point", lambda srv, **kw: sane_op())
        monkeypatch.setattr(m_mod, "measure_all", lambda dv, **kw: marginal())
        monkeypatch.setattr(ev_mod, "netlist", lambda dv, **kw: (_ for _ in ()).throw(
            FileNotFoundError("sky130.lib.spice not found under PDK_ROOT=/nope")))

        ev = ev_mod.build_evaluator(DEFAULT_SPEC, store=ArtifactStore(tmp_path / "raw"),
                                    server_factory=lambda c: FakeServer(corner=c))
        with pytest.raises(GuardConfigError, match="could not be recorded"):
            ev.evaluate(DV, vdd=1.8)


class TestCornerWiring:

    def test_corner_probe_is_attached(self, wired):
        assert wired().corner_probe is not None

    def test_dead_corner_include_is_caught_through_the_wiring(self, wired, monkeypatch):
        ev = wired()
        ev.corner_probe = lambda corner: 0.45          # every corner identical
        r = ev.verify_corners(("tt", "ss"))
        assert isinstance(r, Invalid) and r.check is Check.T3_CORNER_IDENTICAL

    def test_live_corners_pass(self, wired):
        ev = wired()
        ev.corner_probe = {"tt": 0.45, "ss": 0.52}.__getitem__
        assert ev.verify_corners(("tt", "ss")) is None
