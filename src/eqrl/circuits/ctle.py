"""Parametric CTLE (+ 1-tap DFE stub) netlist generator.

Turns a design-variable dict into a SPICE netlist string. Phase 0 uses ideal/behavioral
devices so the loop closes fast; swap `.model` lines for SKY130 device models in Phase 1.

Topology: differential pair with source-degeneration Rs||Cs, resistive load R_load.
The Rs*Cs zero produces the HF peaking that boosts the Nyquist band.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DesignVars:
    """The RL action, decoded into physical device values (SI units)."""
    w_in: float = 20e-6     # input pair width  [m]
    l_in: float = 0.15e-6   # input pair length [m]
    i_tail: float = 2e-3    # tail current      [A]
    rs: float = 1e3         # degeneration R    [ohm]
    cs: float = 200e-15     # degeneration C    [F]
    r_load: float = 1e3     # load R            [ohm]
    w_dfe: float = 0.0      # 1-tap DFE weight  [fraction of UI]

    def area_mm2(self) -> float:
        """Crude analytic area: transistors + a fixed passive/routing budget."""
        tx = 2 * self.w_in * self.l_in            # two input devices [m^2]
        passive = 2e-9                            # ~2000 um^2 for Rs/Cs/load/bias
        return (tx + passive) * 1e6               # m^2 -> mm^2


# Order of the action vector <-> DesignVars fields. Ranges are (lo, hi) in SI.
ACTION_SPACE = {
    "w_in":   (1e-6, 100e-6),
    "l_in":   (0.15e-6, 1e-6),
    "i_tail": (0.1e-3, 5e-3),
    "rs":     (100.0, 5e3),
    "cs":     (10e-15, 2e-12),
    "r_load": (100.0, 5e3),
    "w_dfe":  (0.0, 0.5),
}


def decode_action(a) -> DesignVars:
    """Map a normalized action in [0,1]^7 (or [-1,1]) to physical DesignVars.

    Accepts either [0,1] or [-1,1]; log-scales the wide-range knobs.
    """
    import math

    keys = list(ACTION_SPACE.keys())
    vals = {}
    for i, k in enumerate(keys):
        lo, hi = ACTION_SPACE[k]
        x = float(a[i])
        x = (x + 1) / 2 if x < 0 else x          # accept [-1,1]
        x = min(max(x, 0.0), 1.0)
        if lo > 0 and hi / lo > 50:              # log-scale wide, strictly-positive ranges
            v = lo * (hi / lo) ** x
        else:
            v = lo + (hi - lo) * x
        vals[k] = v
    return DesignVars(**vals)


C_LOAD = 30e-15   # next-stage input cap the CTLE drives [F]
VCM = 0.9         # input common-mode [V]
AC_AMP = 0.5      # AC drive amplitude per side (differential AC=1)
TRAN_AMP = 0.05   # 100 MHz transient amplitude per side for HD3 [V]

# default backend now that the SKY130 PDK is installed; "behavioral" is the fast fallback
DEFAULT_MODELS = "sky130"

_ANALYSIS = {
    "none":  "",     # analysis supplied by a .control block in the runner
    "ac":    ".ac dec 50 1e6 10e9",
    "op":    ".op",
    "noise": ".noise v(outp) vinp dec 20 10e6 5e9",
    "tran":  ".tran 2p 40n",
}


def netlist(dv: DesignVars, *, vdd: float = 1.8, temp_c: float = 27.0,
            corner: str = "tt", analysis: str = "ac", models: str | None = None) -> str:
    """Build a CTLE testbench netlist.

    models="sky130" -> real sky130_fd_pr transistors + corner .lib (PVT-aware).
    models="behavioral" -> VCCS transconductors (fast, PDK-free fallback).
    analysis: "none"/"ac"/"op"/"noise"/"tran".
    """
    models = models or DEFAULT_MODELS
    src = (f"Vcm cm 0 {VCM}\n"
           f"Vinp inp cm AC {AC_AMP} SIN(0 {TRAN_AMP} 100e6)\n"
           f"Vinn inn cm AC -{AC_AMP} SIN(0 -{TRAN_AMP} 100e6)")

    if models == "sky130":
        core = _core_sky130(dv, vdd, corner, src)
    elif models == "behavioral":
        core = _core_behavioral(dv, vdd, src)
    else:
        raise ValueError(f"unknown models={models!r}")

    header = f"* CTLE  corner={corner} vdd={vdd} temp={temp_c}C analysis={analysis} models={models}\n"
    header += f".temp {temp_c}\n.param vdd={vdd}\nVdd vdd 0 {{vdd}}\n"
    return header + src + "\n" + core + _ANALYSIS[analysis] + "\n.end\n"


def param_deck(corner: str = "tt") -> str:
    """A parametric, control-block-free SKY130 CTLE deck for the persistent server.

    All design variables are `.param`s (w,l in um; itail,rs,cs,rl; vddp). The server
    re-evaluates candidates with `alterparam ...; reset` — models load only once.
    """
    from eqrl.circuits.pdk import lib_include
    return (
        f"* CTLE param deck (server) corner={corner}\n"
        f"{lib_include(corner)}\n"
        ".temp 27\n"
        ".param w=20 l=0.15 itail=2m rs=1k cs=1p rl=800 vddp=1.8\n"
        "Vdd vdd 0 {vddp}\n"
        "Vcm cm 0 'vddp/2'\n"
        f"Vinp inp cm AC {AC_AMP} SIN(0 {TRAN_AMP} 100e6)\n"
        f"Vinn inn cm AC -{AC_AMP} SIN(0 -{TRAN_AMP} 100e6)\n"
        "XM1 outp inp sp 0 sky130_fd_pr__nfet_01v8 L={l} W={w} nf=1 m=1\n"
        "XM2 outn inn sn 0 sky130_fd_pr__nfet_01v8 L={l} W={w} nf=1 m=1\n"
        "Itp sp 0 'itail/2'\n"
        "Itn sn 0 'itail/2'\n"
        "Rs sp sn {rs}\n"
        "Cs sp sn {cs}\n"
        "Rlp vdd outp {rl}\n"
        "Rln vdd outn {rl}\n"
        f"Clp outp 0 {C_LOAD}\n"
        f"Cln outn 0 {C_LOAD}\n"
        ".end\n"
    )


def dv_to_params(dv: DesignVars) -> dict[str, float]:
    """Design variables -> deck `.param` values (W/L clamped to sky130 minimums, um)."""
    return {
        "w": max(dv.w_in * 1e6, 0.42),
        "l": max(dv.l_in * 1e6, 0.15),
        "itail": dv.i_tail,
        "rs": dv.rs,
        "cs": dv.cs,
        "rl": dv.r_load,
    }


def _core_sky130(dv: DesignVars, vdd: float, corner: str, src: str) -> str:
    """Transistor-level source-degenerated CTLE on the SKY130 PDK.

    M1/M2 are nfet_01v8 input devices; each source is sunk by an ideal tail current
    (i_tail/2). Rs||Cs bridges the sources (the degeneration that makes the peaking
    zero). Resistive loads set the DC gain. Bulk = ground.
    """
    from eqrl.circuits.pdk import lib_include

    w_um = max(dv.w_in * 1e6, 0.42)     # sky130 nfet min width
    l_um = max(dv.l_in * 1e6, 0.15)     # sky130 nfet min length
    i_leg = dv.i_tail / 2.0
    return (
        f"{lib_include(corner)}\n"
        f"XM1 outp inp sp 0 sky130_fd_pr__nfet_01v8 L={l_um:.4f} W={w_um:.4f} nf=1 m=1\n"
        f"XM2 outn inn sn 0 sky130_fd_pr__nfet_01v8 L={l_um:.4f} W={w_um:.4f} nf=1 m=1\n"
        f"Itp sp 0 {i_leg}\n"
        f"Itn sn 0 {i_leg}\n"
        f"Rs sp sn {dv.rs}\n"
        f"Cs sp sn {dv.cs}\n"
        f"Rlp vdd outp {dv.r_load}\n"
        f"Rln vdd outn {dv.r_load}\n"
        f"Clp outp 0 {C_LOAD}\n"
        f"Cln outn 0 {C_LOAD}\n"
    )


def _core_behavioral(dv: DesignVars, vdd: float, src: str) -> str:
    """VCCS-based CTLE (no PDK). gm ~ i_tail / Vov, Vov ~ 0.15 V."""
    gm = dv.i_tail / 0.15
    return (
        f"G1 outp sp inp sp {gm}\n"
        f"G2 outn sn inn sn {gm}\n"
        f"Rtp sp 0 50k\nRtn sn 0 50k\n"
        f"Rs sp sn {dv.rs}\nCs sp sn {dv.cs}\n"
        f"Rlp vdd outp {dv.r_load}\nRln vdd outn {dv.r_load}\n"
        f"Clp outp 0 {C_LOAD}\nCln outn 0 {C_LOAD}\n"
    )
