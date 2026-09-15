"""Locate the SKY130 ngspice model library.

Resolves the corner-aware `sky130.lib.spice` that volare/open_pdks installs. Set
`PDK_ROOT` (default ~/pdk) to point at the install. Corners map to the `.lib` sections
inside that file: tt / ss / ff / sf / fs.
"""
from __future__ import annotations

import os
from pathlib import Path

# process corner name -> section label inside sky130.lib.spice
CORNER_SECTION = {
    "tt": "tt", "ss": "ss", "ff": "ff", "sf": "sf", "fs": "fs",
}


def pdk_root() -> Path:
    return Path(os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))


def sky130_lib() -> Path:
    """Path to sky130.lib.spice, or raise with a helpful message."""
    root = pdk_root()
    candidates = [
        root / "sky130A" / "libs.tech" / "ngspice" / "sky130.lib.spice",
        root / "volare" / "sky130" / "versions",  # searched below
    ]
    p = candidates[0]
    if p.exists():
        return p
    # fall back to a glob in case the layout differs
    hits = list(root.rglob("sky130.lib.spice"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"sky130.lib.spice not found under PDK_ROOT={root}. "
        "Run: volare enable --pdk sky130 <version>  (see SETUP.md)."
    )


def lib_include(corner: str = "tt") -> str:
    """A `.lib \"...\" <corner>` line for the requested process corner."""
    sec = CORNER_SECTION[corner]
    return f'.lib "{sky130_lib()}" {sec}'


def available() -> bool:
    try:
        sky130_lib()
        return True
    except FileNotFoundError:
        return False
