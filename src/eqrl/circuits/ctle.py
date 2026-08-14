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
        if hi / max(lo, 1e-18) > 50:             # log-scale wide ranges
            v = lo * (hi / lo) ** x
        else:
            v = lo + (hi - lo) * x
        vals[k] = v
    return DesignVars(**vals)


def netlist(dv: DesignVars, *, vdd: float = 1.8, temp_c: float = 27.0,
            corner: str = "tt", analysis: str = "ac") -> str:
    """Build a CTLE testbench netlist for the requested analysis.

    analysis: "ac" (peaking), "op" (power), "noise", "tran" (HD3/eye).
    NOTE: Phase 0 uses a behavioral gm; replace M1/M2 with SKY130 models in Phase 1.
    """
    gm = dv.i_tail / 0.15            # rough gm estimate for behavioral phase
    ana = {
        "ac":    ".ac dec 50 1e6 10e9",
        "op":    ".op",
        "noise": ".noise v(outp,outn) vin dec 20 10e6 5e9",
        "tran":  ".tran 1p 40n",
    }[analysis]

    return f"""* CTLE testbench  corner={corner} vdd={vdd} temp={temp_c}C  analysis={analysis}
.temp {temp_c}
.param vdd={vdd}
Vdd vdd 0 {{vdd}}
Vcm cm 0 {vdd/2}

* differential input source (AC + transient capable)
Vin  inp inn AC 1 SIN(0 0.05 100e6)
Rcm_p inp cm 1e9
Rcm_n inn cm 1e9

* --- behavioral diff pair with source degeneration (Phase 0) ---
Gp outp 0 inp inn {gm}
Gn outn 0 inn inp {gm}
Rs  sp sn {dv.rs}
Cs  sp sn {dv.cs}
Rlp vdd outp {dv.r_load}
Rln vdd outn {dv.r_load}
Cl_p outp 0 30f
Cl_n outn 0 30f

{ana}
.end
"""
