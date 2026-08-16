"""The resident libngspice server: does it agree with the subprocess path?

The server exists purely for speed. That makes it dangerous: if it ever disagrees with
the subprocess path, every training run is optimizing against a different circuit than
the one the final characterization measures, and nothing would say so.

Skipped unless PySpice + the ngspice shared library are both usable.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from eqrl.circuits.ctle import DesignVars, netlist

# Same design as results/final_report.json, so failures are directly comparable
# to the committed PVT table.
DESIGN = DesignVars(w_in=11.83e-6, l_in=0.15e-6, i_tail=1e-4, rs=5000.0,
                    cs=2.217e-13, r_load=1171.2)


def _server_available() -> bool:
    try:
        from eqrl.circuits import pdk
        if not pdk.available():
            return False
        from eqrl.sim.server import NgspiceServer  # noqa: F401
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.new_instance()
        return True
    except Exception:                              # noqa: BLE001 - availability probe
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(), reason="PySpice + libngspice not usable here")


@pytest.fixture(scope="module")
def srv():
    from eqrl.sim.server import NgspiceServer
    s = NgspiceServer("tt")
    yield s
    s.close()


def _peak(mag, freq):
    i = int(np.argmax(mag))
    return float(mag[0]), float(mag[i] - mag[0]), float(freq[i] / 1e9)


class TestAgreesWithSubprocess:

    def test_dc_gain_and_boost_match(self, srv):
        """The two paths must measure the same circuit.

        Peak FREQUENCY is allowed to differ slightly: the server sweeps `dec 40` and the
        runner `dec 50`, so argmax lands on neighbouring grid points. Gain and boost are
        grid-independent and must agree tightly.
        """
        from eqrl.sim.ngspice_runner import ac as sub_ac

        r = srv.ac(DESIGN)
        dc_s, boost_s, fpk_s = _peak(r["mag_db"], r["freq"])

        q = sub_ac(netlist(DESIGN, analysis="none", models="sky130", corner="tt"))
        dc_p, boost_p, fpk_p = _peak(q["mag_db"], q["freq"])

        assert dc_s == pytest.approx(dc_p, abs=0.05), (
            f"resident server and subprocess disagree on DC gain: "
            f"{dc_s:.3f} vs {dc_p:.3f} dB — they are not simulating the same circuit"
        )
        assert boost_s == pytest.approx(boost_p, abs=0.05), (
            f"boost mismatch: {boost_s:.3f} vs {boost_p:.3f} dB"
        )
        assert fpk_s == pytest.approx(fpk_p, rel=0.10), (
            f"peak frequency mismatch beyond grid resolution: {fpk_s:.3f} vs {fpk_p:.3f} GHz"
        )

    @pytest.mark.xfail(strict=True, reason=(
        "results/final_report.json IS STALE. It was measured with ideal tail current "
        "sources; the circuit now has a real current mirror, which costs 3-4 dB of boost "
        "on this design (tt 10.22 -> 6.58, ss 9.90 -> 5.74, ff 10.43 -> 7.30) and pushes "
        "the ss peak to 1.15 GHz, BELOW the 1.25 GHz spec floor. The committed "
        "'45/45 corners pass' result therefore no longer holds for this design. "
        "strict=True so this turns red the moment the report is regenerated — at which "
        "point update the expected value and delete this marker."))
    def test_reproduces_the_committed_pvt_row(self, srv):
        """tt / 1.8 V / 27 C in results/final_report.json reports boost 10.22 dB."""
        r = srv.ac(DESIGN)
        _, boost, _ = _peak(r["mag_db"], r["freq"])
        assert boost == pytest.approx(10.22, abs=0.15), (
            f"boost {boost:.2f} dB does not reproduce the committed 10.22 dB"
        )


class TestCornersReallyChange:
    """Tier 3, check 9, against the real simulator rather than a stub."""

    def test_process_corners_give_different_results(self, srv):
        out = {}
        for c in ("tt", "ss", "ff"):
            srv.set_corner(c)
            r = srv.ac(DESIGN)
            out[c] = _peak(r["mag_db"], r["freq"])[1]
        srv.set_corner("tt")
        assert out["ss"] != pytest.approx(out["tt"], abs=1e-6), (
            "ss and tt returned identical boost — the corner .lib is not taking effect, "
            "so a 'PVT sweep' would be five runs of typical wearing different labels"
        )
        assert out["ff"] != pytest.approx(out["tt"], abs=1e-6)
        assert out["ss"] < out["tt"] < out["ff"], (
            f"expected boost to order ss < tt < ff, got {out}"
        )

    def test_guard_layer_corner_check_passes_on_real_models(self, srv):
        from eqrl.guards import check_corner_integrity

        def probe(corner: str) -> float:
            srv.set_corner(corner)
            r = srv.ac(DESIGN)
            return float(r["mag_db"][0])          # DC gain shifts with the corner

        assert check_corner_integrity(probe, ("tt", "ss", "ff")) is None
        srv.set_corner("tt")


class TestResidencyIsWhatMakesItFast:

    def test_repeated_evals_do_not_reload_models(self, srv):
        """A server that silently re-parsed the PDK per call would still be correct,
        just slow — and the whole justification for its existence would be gone."""
        import time

        srv.ac(DESIGN)                             # warm
        t0 = time.perf_counter()
        for cs in np.linspace(1.5e-13, 3e-13, 5):
            srv.ac(DesignVars(**{**DESIGN.__dict__, "cs": float(cs)}))
        per_eval = (time.perf_counter() - t0) / 5
        assert per_eval < 1.0, (
            f"{per_eval*1e3:.0f} ms/eval — the server is re-parsing models per call; "
            "model loading alone costs seconds, so residency has been lost"
        )
