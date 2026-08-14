"""Operating-point parsing — testable today, before ngspice exists.

This is where a silent wrong answer would be cheapest to introduce: a regex that
quietly matches nothing produces an empty operating point, and an empty operating
point read as "nothing wrong" disables all of Tier 2. Every test below is about
making absence loud.
"""
from __future__ import annotations

import pytest

from eqrl.guards import Check, OperatingPoint, check_circuit_sanity
from eqrl.sim.probe import (
    CTLE_INSTANCES, CTLE_NODES, CTLE_TAIL_BRANCHES, DEVICE_PARAMS, NFET, ProbeError,
    build_operating_point, device_ref, op_commands, parse_assignments,
    probe_operating_point,
)

# A realistic ngspice transcript for the two-device CTLE at a healthy bias.
GOOD_TRANSCRIPT = """
@m.xm1.msky130_fd_pr__nfet_01v8[vds] = 6.01234e-01
@m.xm1.msky130_fd_pr__nfet_01v8[vdsat] = 1.98000e-01
@m.xm1.msky130_fd_pr__nfet_01v8[vgs] = 9.00000e-01
@m.xm1.msky130_fd_pr__nfet_01v8[vth] = 4.51000e-01
@m.xm1.msky130_fd_pr__nfet_01v8[id] = 1.00000e-03
@m.xm2.msky130_fd_pr__nfet_01v8[vds] = 6.01234e-01
@m.xm2.msky130_fd_pr__nfet_01v8[vdsat] = 1.98000e-01
@m.xm2.msky130_fd_pr__nfet_01v8[vgs] = 9.00000e-01
@m.xm2.msky130_fd_pr__nfet_01v8[vth] = 4.51000e-01
@m.xm2.msky130_fd_pr__nfet_01v8[id] = 1.00000e-03
v(outp) = 1.19876e+00
v(outn) = 1.19876e+00
v(sp) = 4.50000e-01
v(sn) = 4.50000e-01
i(itp) = 1.00000e-03
i(itn) = 1.00000e-03
"""


def _build(text: str = GOOD_TRANSCRIPT) -> OperatingPoint:
    return build_operating_point(parse_assignments(text), CTLE_INSTANCES, CTLE_NODES,
                                 CTLE_TAIL_BRANCHES)


class TestReferences:

    def test_device_ref_targets_the_mosfet_inside_the_subckt(self):
        assert device_ref("XM1") == f"@m.xm1.m{NFET}"

    def test_op_commands_cover_every_required_quantity(self):
        cmds = op_commands(CTLE_INSTANCES, CTLE_NODES, CTLE_TAIL_BRANCHES)
        assert cmds[0] == "op"
        for inst in CTLE_INSTANCES:
            for p in DEVICE_PARAMS:
                assert f"print {device_ref(inst)}[{p}]" in cmds
        for n in CTLE_NODES:
            assert f"print v({n})" in cmds
        for b in CTLE_TAIL_BRANCHES:
            assert f"print i({b})" in cmds


class TestParsing:

    def test_parses_a_healthy_transcript(self):
        op = _build()
        assert len(op.devices) == 2
        assert op.devices[0].vds == pytest.approx(0.601234)
        assert op.devices[0].vdsat == pytest.approx(0.198)
        assert op.node_voltages["outp"] == pytest.approx(1.19876)
        assert op.tail_currents["Itp"] == pytest.approx(1e-3)

    @pytest.mark.parametrize("line,key,val", [
        ("x = 1.5", "x", 1.5),
        ("  y  =  -2.5e-3  ", "y", -2.5e-3),
        ("v(outp) = ( 1.2 )", "v(outp)", 1.2),
        ("V(OUTP) = 1.2", "v(outp)", 1.2),
    ])
    def test_assignment_forms(self, line, key, val):
        assert parse_assignments(line)[key] == pytest.approx(val)

    @pytest.mark.parametrize("noise", [
        "", "ngspice 42 -> op", "no matching data", "Warning: singular matrix",
        "@m.xm1.msky130_fd_pr__nfet_01v8[vds] =",         # truncated
    ])
    def test_non_assignments_are_ignored_not_guessed(self, noise):
        assert parse_assignments(noise) == {}

    def test_result_is_healthy_under_tier2(self, tmp_path):
        """The end the parser exists for: a good transcript must survive check 5-7."""
        from eqrl.guards import ArtifactStore
        from eqrl.circuits.ctle import DesignVars
        store = ArtifactStore(tmp_path / "raw")
        a = store.new_run()
        a.exit_code = 0
        dv = DesignVars(l_in=0.3e-6, i_tail=2e-3)
        assert check_circuit_sanity(_build(), dv, a, vdd=1.8) is None


class TestAbsenceIsLoud:
    """Every one of these would otherwise yield a partial operating point that Tier 2
    reads as healthy."""

    def test_missing_vdsat_raises(self):
        text = GOOD_TRANSCRIPT.replace(
            "@m.xm1.msky130_fd_pr__nfet_01v8[vdsat] = 1.98000e-01\n", "")
        with pytest.raises(ProbeError, match="vdsat"):
            _build(text)

    def test_missing_vds_raises(self):
        text = GOOD_TRANSCRIPT.replace(
            "@m.xm2.msky130_fd_pr__nfet_01v8[vds] = 6.01234e-01\n", "")
        with pytest.raises(ProbeError, match="XM2"):
            _build(text)

    def test_missing_node_raises(self):
        with pytest.raises(ProbeError, match=r"v\(sp\)"):
            _build(GOOD_TRANSCRIPT.replace("v(sp) = 4.50000e-01\n", ""))

    def test_missing_branch_raises(self):
        with pytest.raises(ProbeError, match=r"i\(itn\)"):
            _build(GOOD_TRANSCRIPT.replace("i(itn) = 1.00000e-03\n", ""))

    def test_completely_empty_transcript_raises(self):
        with pytest.raises(ProbeError):
            _build("ngspice done, no output\n")

    def test_optional_params_may_be_absent(self):
        """vgs/vth/id are diagnostics, not gating — they become NaN, and the device is
        still evaluated on vds/vdsat."""
        text = GOOD_TRANSCRIPT
        for p in ("vgs", "vth", "id"):
            text = "\n".join(l for l in text.splitlines() if f"[{p}]" not in l)
        op = build_operating_point(parse_assignments(text), CTLE_INSTANCES, CTLE_NODES,
                                   CTLE_TAIL_BRANCHES)
        assert op.devices[0].vds == pytest.approx(0.601234)
        assert op.devices[0].vgs != op.devices[0].vgs      # NaN


class TestDriver:

    class _FakeNg:
        def __init__(self, transcript: str):
            self._lines = transcript.strip().splitlines()
            self.commands: list[str] = []

        def exec_command(self, cmd: str):
            self.commands.append(cmd)
            if cmd == "op":
                return ""
            return self._lines.pop(0) if self._lines else ""

    class _FakeServer:
        def __init__(self, transcript: str):
            self._ng = TestDriver._FakeNg(transcript)

    def test_driver_issues_op_first_and_parses(self):
        srv = self._FakeServer(GOOD_TRANSCRIPT)
        op = probe_operating_point(srv)
        assert srv._ng.commands[0] == "op"
        assert len(op.devices) == 2
        assert op.devices[1].name == "XM2"

    def test_silent_simulator_raises_rather_than_returning_empty(self):
        srv = self._FakeServer("")
        with pytest.raises(ProbeError, match="no operating-point output"):
            probe_operating_point(srv)

    def test_server_without_ngspice_handle_raises(self):
        class NoHandle:
            pass
        with pytest.raises(ProbeError):
            probe_operating_point(NoHandle())
