"""Parametric CTLE (+ 1-tap DFE stub) netlist generator.

Turns a design-variable dict into a SPICE netlist string. Phase 0 uses ideal/behavioral
devices so the loop closes fast; swap `.model` lines for SKY130 device models in Phase 1.

Topology: differential pair with source-degeneration Rs||Cs, resistive load R_load.
The Rs*Cs zero produces the HF peaking that boosts the Nyquist band.
"""
from __future__ import annotations

import dataclasses
import math
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
        """Analytic silicon area: every device the netlist actually instantiates.

        The previous model counted the two input devices and then a flat 2000 um^2 for
        everything else, which made area very nearly a constant -- the input pair spans
        6 um^2 at the default sizing, so essentially the whole number was the constant.
        It was also blind to the current mirror, which is by far the largest structure
        here: at 20 mA the three mirror devices draw 1600 um of width at 2 um length,
        9600 um^2, roughly five times that entire fixed budget.

        Now each contribution is computed from the design variables:

          input pair   2 * W_in * L_in
          mirror       3 * W_total * MIRROR_L_UM      (reference + two outputs)
          Cs           cs / MIM_DENSITY               sky130 MIM, ~2 fF/um^2
          Rs, R_load   (rs + 2*r_load) squares of poly at RES_SHEET_OHM_SQ,
                       drawn RES_WIDTH_UM wide for matching
          overhead     ROUTING_UM2                    routing, guard rings, taps

        HONEST LIMIT: this makes area design-dependent rather than constant, but it does
        not make the area spec reachable. Worst case over the whole action space is about
        0.012 mm^2 against a 0.05 mm^2 budget, so `area` still cannot fail. Whether that
        budget is the right one is a spec question, not a modelling one.
        """
        MIM_DENSITY_F_PER_M2 = 2e-15 / 1e-12       # 2 fF/um^2 -> F/m^2
        RES_SHEET_OHM_SQ = 320.0                   # sky130 p+ poly precision resistor
        RES_WIDTH_UM = 1.0                         # drawn width, for matching
        ROUTING_M2 = 500e-12                       # ~500 um^2 routing/guard ring/taps

        tx = 2 * self.w_in * self.l_in
        wb_um, fingers = mirror_sizing(self.i_tail)
        mirror = 3 * (wb_um * fingers * 1e-6) * (MIRROR_L_UM * 1e-6)
        cap = self.cs / MIM_DENSITY_F_PER_M2
        squares = (self.rs + 2 * self.r_load) / RES_SHEET_OHM_SQ
        res = squares * (RES_WIDTH_UM * 1e-6) ** 2
        return (tx + mirror + cap + res + ROUTING_M2) * 1e6    # m^2 -> mm^2


# Order of the action vector <-> DesignVars fields. Ranges are (lo, hi) in SI.
#
# w_dfe is NOT here: the 1-tap DFE is a receiver-DSP block applied at the slicer and
# adapted to the post-cursor in the eye engine, not an analog knob — a phantom parameter
# would be dishonest.
#
# i_tail's upper bound was 20 mA, on the reasoning that the 15 mW power budget should be
# reachable and violable so the agent has to trade power against boost. That reasoning was
# wrong, because the bias collapses long before the power budget binds. Conditional
# validity, measured with 30 random designs at each fixed tail current
# (experiments/itail_profile.py, results/itail_profile.json):
#
#     0.05 mA  16.7%     0.35 mA  30.0%     0.75 mA   6.7%     1.5 mA  0.0%
#     0.10 mA  23.3%     0.50 mA  23.3%     1.00 mA  13.3%     2.0 mA  0.0%
#     0.20 mA  20.0%                                           4.0 mA  0.0%
#
# Zero valid designs in 90 samples above 1 mA, every one rejected for the same reason:
# T2.5, the input pair or the mirror out of saturation. The mechanism is the one documented
# at VCM_VDD_RATIO -- more tail current means more Vgs on the input pair, which pulls the
# tail node down, which is exactly the headroom the mirror needs. Above ~1 mA there is no
# combination of the other five parameters that recovers it.
#
# So the old range spent roughly 60% of its log-measure on a region where nothing can
# succeed. 1 mA is the measured edge, kept rather than trimmed further because 0.05-1 mA is
# a broad workable band (7-30%) and narrowing to the 0.35 mA peak would be fitting the
# search space to a single sample.
#
# For the record, docs/PROBLEM.md's own first-cut table said 0.1-5 mA; the code said
# 0.05-20 mA. Neither was measured. This is.
ACTION_SPACE = {
    "w_in":   (1e-6, 100e-6),
    "l_in":   (0.15e-6, 1e-6),
    "i_tail": (0.05e-3, 1.0e-3),
    "rs":     (100.0, 5e3),
    "cs":     (10e-15, 2e-12),
    "r_load": (100.0, 5e3),
}


def decode_action(a, domain: str = "unit") -> DesignVars:
    """Map a normalized action to physical DesignVars.

    domain="unit" : a is in [0, 1]^7   (the internal representation)
    domain="pm1"  : a is in [-1, 1]^7  (a gym Box(-1,1) policy output)

    The domain is EXPLICIT and must be. The previous version tried to infer it with
    `x = (x + 1) / 2 if x < 0 else x`, which is not a coordinate change but a branch:
    it mapped negative inputs onto [0, 0.5] while leaving positive inputs on [0, 1].
    That made the mapping discontinuous and non-monotonic at zero —

        a = -0.01  ->  w_in =  9.77 um
        a = +0.00  ->  w_in =  1.00 um     (a cliff, mid-range)

    — which is exactly where a tanh-squashed policy puts most of its probability mass.
    Inference is impossible in principle anyway: 0.3 is a valid point in both domains
    and means different things in each, so the caller has to say.

    Wide, strictly-positive ranges are log-scaled so the search resolves small values.
    """
    keys = list(ACTION_SPACE.keys())
    if domain == "pm1":
        x_norm = [(float(v) + 1.0) / 2.0 for v in a]      # linear over the WHOLE range
    elif domain == "unit":
        x_norm = [float(v) for v in a]
    else:
        raise ValueError(f"unknown domain {domain!r}; use 'unit' or 'pm1'")

    vals = {}
    for i, k in enumerate(keys):
        lo, hi = ACTION_SPACE[k]
        x = min(max(x_norm[i], 0.0), 1.0)
        if lo > 0 and hi / lo > 50:              # log-scale wide, strictly-positive ranges
            v = lo * (hi / lo) ** x
        else:
            v = lo + (hi - lo) * x
        vals[k] = v
    return DesignVars(**vals)


C_LOAD = 30e-15   # next-stage input cap the CTLE drives [F]
#: Input common-mode, as a fraction of VDD.
#:
#: MEASURED, not chosen by preference. At the previous 0.5*VDD the source node sat at
#: ~0.105 V while the tail mirror device needs ~0.40 V of Vdsat, so the mirror was in
#: TRIODE at every bias point and delivered only 16-46% of the requested current:
#:
#:     requested/leg   delivered   XMtp headroom
#:        250 uA        116 uA       -274 mV
#:       1000 uA        326 uA       -295 mV
#:       4000 uA        649 uA       -325 mV
#:
#: A triode device is a resistor, not a current source, so `i_tail` was a badly
#: compressed non-linear knob and the mirror provided none of the PVT behaviour it was
#: added for. Widening the mirror alone does not fix it (at 0.5*VDD even a 100 um device
#: is still -84 mV short) — the headroom has to come from the common-mode.
#:
#: The tail node sits at sp = VCM - Vgs1, so VCM trades mirror headroom against input-pair
#: headroom one-for-one. Raising it saturates the mirror over more of the space; the same
#: move pushes the input pair out of saturation at a lower load resistance, because the
#: output has to stay above sp + Vdsat1 and sp has just gone up.
#:
#: Both effects were measured. Tier 2 coverage over a 33-point (i_tail, l_in, r_load) grid
#: at four PVT corners favours a high common-mode:
#:
#:      VCM        tt 1.80 27C   ss 1.71 125C   ss 1.71 0C   ff 1.89 0C   total
#:      0.80*VDD      28/33          24/33         27/33        27/33    106/132
#:      0.84*VDD      30/33          26/33         27/33        28/33    111/132
#:
#: The monotonicity suite -- the encoded physics contract, which sweeps r_load far enough
#: to reach the input pair's limit -- favours the opposite:
#:
#:      VCM        0.72      0.76      0.80      0.84
#:      failures    0/14      2/14      2/14      3/14
#:
#: The grid above tops out at r_load = 2 kohm and so never visits the region where the
#: input pair binds, which is why it disagrees. The physics contract is the stronger
#: signal: it is what asserts the simulator is measuring the circuit we think it is, and
#: 0.72 is the only value where all of it holds. Coverage lost at high i_tail is coverage
#: the guard correctly reports as invalid, which is a worse trade than a design space with
#: a non-monotone gain landscape for the agent to learn.
#:
#: At 0.72 both devices are saturated and the mirror delivers 88-96% of the request from
#: 0.05 mA to 2 mA at both 1.8 V and 1.71 V. Above ~5 mA the binding constraint becomes
#: the load drop (I*R vs VDD), which is real physics for the agent to respect.
VCM_VDD_RATIO = 0.72
VCM = 1.8 * VCM_VDD_RATIO   # input common-mode at nominal VDD [V]
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
        # fail fast: skip dynamic gmin/source stepping so non-convergent designs return
        # in ~ms (agent learns to avoid them) instead of grinding for seconds.
        ".options gminsteps=0 srcsteps=0 itl1=100\n"
        # temperature via an alterparam'd option (shared-mode `set temp` is ignored).
        ".param w=20 l=0.15 itail=2m rs=1k cs=1p rl=800 vddp=1.8 tempc=27\n"
        f".param wb=80 lb={MIRROR_L_UM} mb=2\n"
        ".options temp={tempc}\n"
        "Vdd vdd 0 {vddp}\n"
        f"Vcm cm 0 'vddp*{VCM_VDD_RATIO}'\n"
        f"Vinp inp cm AC {AC_AMP} SIN(0 {TRAN_AMP} 100e6)\n"
        f"Vinn inn cm AC -{AC_AMP} SIN(0 -{TRAN_AMP} 100e6)\n"
        "XM1 outp inp sp 0 sky130_fd_pr__nfet_01v8 L={l} W={w} nf=1 m=1\n"
        "XM2 outn inn sn 0 sky130_fd_pr__nfet_01v8 L={l} W={w} nf=1 m=1\n"
        + _MIRROR
        + "Rs sp sn {rs}\n"
        "Cs sp sn {cs}\n"
        "Rlp vdd outp {rl}\n"
        "Rln vdd outn {rl}\n"
        f"Clp outp 0 {C_LOAD}\n"
        f"Cln outn 0 {C_LOAD}\n"
        ".end\n"
    )


# NMOS current-mirror tail bias: a trimmed reference (itail/2) sets Vgs on a
# diode-connected device; two matched mirrors sink the leg currents. Unlike an ideal
# source, the delivered current drifts with process, VDD and temperature (Vds mismatch +
# channel-length modulation), so PVT can fail the way real analog does.
_MIRROR = (
    "Iref vdd nbias 'itail/2'\n"
    "XMref nbias nbias 0 0 sky130_fd_pr__nfet_01v8 L={lb} W={wb} nf=1 m={mb}\n"
    "XMtp sp nbias 0 0 sky130_fd_pr__nfet_01v8 L={lb} W={wb} nf=1 m={mb}\n"
    "XMtn sn nbias 0 0 sky130_fd_pr__nfet_01v8 L={lb} W={wb} nf=1 m={mb}\n"
)


#: Mirror channel length. Not a style choice — mirror accuracy is set by channel-length
#: modulation, because XMref sits near Vgs (~0.9 V) while XMtp sits at the tail node
#: (~0.3 V) and Id drifts with that 0.6 V difference. Measured worst-case delivery error
#: over i_tail in [0.05, 20] mA and VDD in {1.8, 1.71}:
#:     L = 0.5 um -> 34.0%    L = 1 um -> 8.3%    L = 2 um -> 3.7%    L = 4 um -> 1.2%
#: The guard requires 10% (TAIL_CURRENT_TOLERANCE), so 0.5 um could not meet it at any
#: common-mode. 2 um clears it with margin at a quarter of the area of 4 um.
MIRROR_L_UM = 2.0

#: Total mirror width per ampere of leg current, at MIRROR_L_UM. Anchored to a measured
#: point rather than a rule of thumb: at 1 mA/leg, 160 um total gave +160 mV of saturation
#: headroom and 2.6% current error, while 80 um gave only +64 mV and 40 um went into
#: triode at -68 mV. 1 mA / 160 um = 6.25 uA/um.
MIRROR_DENSITY_A_PER_UM = 6.25e-6

#: sky130_fd_pr__nfet_01v8 is binned to a finite per-device width. Measured against
#: ngspice 41: W = 100 um solves, W = 101 um aborts with "could not find a valid
#: modelname" — and on the resident-server path that abort carries no exit code, so it
#: is the kind of failure that reads as a clean run producing no numbers. The previous
#: clamp of 200 um was therefore silently unbuildable above ~10 mA. Real layout draws a
#: wide device as parallel fingers, so that is what we do; 90 um leaves bin margin.
MIRROR_W_MAX_UM = 90.0
MIRROR_W_MIN_UM = 0.42               # sky130 nfet minimum drawn width


def mirror_sizing(i_tail: float) -> tuple[float, int]:
    """Mirror geometry for a requested tail current: (width per finger um, finger count).

    Splitting into fingers rather than clamping matters: a clamp quietly starves the
    mirror (the requested current is never delivered and the guard sees a bias error),
    whereas exceeding the bin limit is not simulable at all.
    """
    total_um = max((i_tail / 2.0) / MIRROR_DENSITY_A_PER_UM, MIRROR_W_MIN_UM)
    fingers = max(1, math.ceil(total_um / MIRROR_W_MAX_UM))
    return total_um / fingers, fingers


def project_feasible(dv: DesignVars, vdd: float = 1.8,
                     out_headroom_v: float = 0.55) -> DesignVars:
    """Clamp R_load so the DC load drop cannot exceed what the supply can provide.

    OPT-IN. Nothing calls this by default, because it changes what the search space means
    and therefore what every published RL number measures.

    Each leg carries i_tail/2 through R_load, so the output sits at VDD - (i_tail/2)*R_load.
    Most of the declared action space asks for more than the whole supply across the load
    -- at the corner of the box, 20 mA through 5 kohm is 50 V on a 1.8 V rail -- and
    `space_validity.py` measures only 0.4% of the box as a buildable circuit. An agent
    given a mostly-impossible space, and a flat penalty for every impossible design, has
    no gradient to follow; the measured consequence is in HANDOVER, section 6.

    This projects rather than rejects: the action keeps its full range and is mapped onto
    the largest load the chosen tail current can actually drive. The agent is not told
    "no", it is handed the nearest buildable design, so every step still returns a real
    measurement to learn from.

    `out_headroom_v` is the output voltage reserved for the input pair (its Vds must clear
    the tail node plus Vdsat); 0.55 V is just above the 0.504 V measured in
    tests/test_monotonicity._headroom_v at the nominal bias.
    """
    max_r = max((vdd - out_headroom_v) / max(dv.i_tail / 2.0, 1e-12), ACTION_SPACE["r_load"][0])
    if dv.r_load <= max_r:
        return dv
    return dataclasses.replace(dv, r_load=float(max_r))


def dv_to_params(dv: DesignVars) -> dict[str, float]:
    """Design variables -> deck `.param` values (W/L clamped to sky130 minimums, um)."""
    wb, mb = mirror_sizing(dv.i_tail)
    return {
        "w": max(dv.w_in * 1e6, 0.42),
        "l": max(dv.l_in * 1e6, 0.15),
        "itail": dv.i_tail,
        "rs": dv.rs,
        "cs": dv.cs,
        "rl": dv.r_load,
        "wb": wb,
        "lb": MIRROR_L_UM,
        "mb": mb,
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
    wb, mb = mirror_sizing(dv.i_tail)
    return (
        f"{lib_include(corner)}\n"
        f"XM1 outp inp sp 0 sky130_fd_pr__nfet_01v8 L={l_um:.4f} W={w_um:.4f} nf=1 m=1\n"
        f"XM2 outn inn sn 0 sky130_fd_pr__nfet_01v8 L={l_um:.4f} W={w_um:.4f} nf=1 m=1\n"
        f"Iref vdd nbias {i_leg}\n"
        f"XMref nbias nbias 0 0 sky130_fd_pr__nfet_01v8 L={MIRROR_L_UM} W={wb:.4f} nf=1 m={mb}\n"
        f"XMtp sp nbias 0 0 sky130_fd_pr__nfet_01v8 L={MIRROR_L_UM} W={wb:.4f} nf=1 m={mb}\n"
        f"XMtn sn nbias 0 0 sky130_fd_pr__nfet_01v8 L={MIRROR_L_UM} W={wb:.4f} nf=1 m={mb}\n"
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
