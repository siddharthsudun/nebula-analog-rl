"""What the ngspice on THIS machine can actually run.

`shutil.which("ngspice")` says an executable exists; it does not say the executable
understands the decks this repo writes. The DFE testbench is built from POLY-form
controlled sources (`Erx rxn 0 poly(2) ...`), which ngspice implements by translating
into an internal XSPICE `a` device. On Ubuntu's ngspice 42 that translation emits the
device without its model and the whole run dies before any analysis:

    MIF-ERROR - unable to find definition of model a$poly$erx
    Simulation interrupted due to error!

The same deck runs on the ngspice 41 (Windows) and 47 (macOS) builds SETUP.md pins, which
is where every recorded DFE number comes from. So this is a property of the installed
binary, not of the netlist writer, and a test that needs POLY has to be able to ask.

This is deliberately a RUNTIME probe rather than a version check: the point is whether the
feature works here, and a version table would go stale the moment a distro backports.
"""
from __future__ import annotations

import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

#: A DC probe whose only interesting feature is the POLY source. It is deliberately
#: `poly(2)` with two controlling pairs -- the exact form `dfe_testbench_netlist` writes
#: for `Erx` and `Eout` -- because the failure being detected is in ngspice's translation
#: of that construct, and a simpler `poly(1)` might well translate cleanly on a binary
#: where the real deck still dies.
_PROBE = """* poly(2) capability probe
V1 a 0 DC 1
V2 c 0 DC 1
E1 b 0 poly(2) a 0 c 0 0 1 1
R1 b 0 1k
.op
.end
"""


def poly_sources_supported() -> bool:
    """True if the ngspice on PATH can run a POLY-form controlled source.

    The cache is keyed on the RESOLVED executable, never on "was it supported the first
    time anyone asked". PATH is not constant across a pytest session: several
    `eqrl.experiments.*` modules prepend the bundled ngspice shim to it at import time, so
    a module collected early can see no ngspice at all while one collected later sees a
    working binary. Caching a bare False on the first call made this probe report "cannot
    run POLY" for the whole session and skipped a test that passes here -- the failure mode
    it exists to prevent, inverted.
    """
    exe = shutil.which("ngspice")
    return False if exe is None else _poly_probe(exe)


@functools.lru_cache(maxsize=None)
def _poly_probe(exe: str) -> bool:
    """Run the probe deck through one specific ngspice binary. Cached per binary."""
    with tempfile.TemporaryDirectory() as d:
        deck = Path(d) / "poly_probe.cir"
        deck.write_text(_PROBE, encoding="utf-8")
        try:
            r = subprocess.run([exe, "-b", str(deck)], capture_output=True, text=True,
                               timeout=60)
        except (OSError, subprocess.SubprocessError):
            return False
    blob = (r.stdout or "") + (r.stderr or "")
    return "MIF-ERROR" not in blob and "Simulation interrupted" not in blob
