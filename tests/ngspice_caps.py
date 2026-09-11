"""Can the ngspice on THIS machine actually run the DFE testbench?

`shutil.which("ngspice")` says an executable exists; it does not say the executable
understands the decks this repo writes. The DFE testbench is built from POLY-form
controlled sources (`Erx rxn 0 poly(2) ...`), which ngspice implements by translating into
an internal XSPICE `a` device. On Ubuntu's ngspice 42 that translation emits the device
without its model and the run dies before any analysis:

    MIF-ERROR - unable to find definition of model a$poly$erx
    Simulation interrupted due to error!

The same deck runs on the ngspice 41 (Windows) and 47 (macOS) builds SETUP.md pins, which
is where every recorded DFE number comes from. So this is a property of the installed
binary, not of the netlist writer.

The probe RUNS THE REAL TESTBENCH rather than a reduced stand-in. A hand-written `poly(2)`
probe deck was tried first and was worse than useless: ngspice 42 solves it happily and
still cannot run the testbench, so the probe reported "supported" and the four tests failed
anyway. The only honest question is the one the tests themselves ask.

Note what is deliberately NOT guarded: `test_netlist_contains_the_dfe_summing_node` asserts
the writer emits the summing node and needs no simulator at all. If this probe ever returns
False on every machine, the DFE tests go quiet everywhere and that text test is the only
thing still pinning the topology -- which is why it stays outside the guard.
"""
from __future__ import annotations

import functools
import shutil


def dfe_testbench_supported() -> bool:
    """True if the ngspice on PATH can simulate the 1-tap DFE testbench.

    The cache is keyed on the RESOLVED executable, never on "did it work the first time
    anyone asked". PATH is not constant across a pytest session: several
    `eqrl.experiments.*` modules prepend the bundled ngspice shim at import time, so a
    module collected early can see no ngspice while one collected later sees a working
    binary. Caching a bare False from the first call turned this into a session-wide
    "unsupported" and skipped a test that passes here -- the failure it exists to prevent,
    inverted.
    """
    exe = shutil.which("ngspice")
    return False if exe is None else _probe(exe)


@functools.lru_cache(maxsize=None)
def _probe(exe: str) -> bool:
    """One real transient run through one specific binary. Cached per binary."""
    try:
        from eqrl.circuits.dfe import dfe_stage_eye

        dfe_stage_eye(0.3, c0=0.5, c1=0.3)
    except Exception:           # noqa: BLE001 - any failure here means "cannot measure"
        return False
    return True
