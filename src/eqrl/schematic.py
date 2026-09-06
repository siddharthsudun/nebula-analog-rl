"""Render the delivered CTLE as a labelled schematic (SVG).

This DRAWS what `eqrl.circuits.ctle.netlist` INSTANTIATES -- it is not a decorative
diagram. Every device, every connection and every annotated value is taken from the same
DesignVars the simulator was handed, so the picture cannot drift from the circuit that was
actually measured. The mirror geometry is not hard-coded either: it comes from
`ctle.mirror_sizing(i_tail)`, the same function the netlist builder calls.

Topology (matches `ctle.param_deck` line for line):

    XM1/XM2   differential input pair, drains -> outp/outn, sources -> sp/sn
    Rs || Cs  source degeneration between sp and sn  (the Rs*Cs zero IS the HF peaking)
    Rlp/Rln   resistive loads, vdd -> outp/outn
    Clp/Cln   fixed load capacitance, outp/outn -> gnd
    Iref + XMref/XMtp/XMtn   NMOS current-mirror tail (a real mirror, not an ideal source)

The six values the RL agent chose are drawn in the accent colour; everything else --
topology, the fixed load caps, the mirror length -- is fixed by the design and drawn muted.
That split is the point of the figure: it shows exactly what was searched and what was not.

The 1-tap DFE is drawn as a DOWNSTREAM, DASHED block deliberately. It is a receiver-DSP
tap applied at the slicer in the eye engine, not an analog node in this netlist, and
w_dfe = 0 in the delivered design. Drawing it as an analog component would be a lie.
"""
from __future__ import annotations

from eqrl.circuits.ctle import C_LOAD, MIRROR_L_UM, DesignVars, mirror_sizing

# Layout constants. The drawing is a fixed topology, so these are laid out by hand on a
# grid rather than by a solver -- shared baselines are what make it read as deliberate.
W, H = 940, 800
VDD_Y = 100           # supply rail
GND_Y = 730           # ground rail
XL, XR = 340, 610     # the two signal legs (left = p, right = n)
LOAD_Y = 175          # load resistor body
OUT_Y = 250           # output node row
PAIR_Y = 330          # input pair device centre
DEG_Y = 440           # Rs || Cs degeneration row
TAIL_Y = 600          # tail mirror device centre
REF_X = 110           # mirror reference branch
TAP_DX = 110          # how far the output taps sit outside each leg


def _eng(value: float, unit: str, digits: int = 3) -> str:
    """Format a value in engineering units (so 1.8162e-13 F reads as '181.6 fF')."""
    if value == 0:
        return f"0 {unit}"
    for scale, prefix in ((1e-15, "f"), (1e-12, "p"), (1e-9, "n"),
                          (1e-6, "u"), (1e-3, "m"), (1.0, ""), (1e3, "k"), (1e6, "M")):
        if abs(value) < scale * 1000:
            return f"{value / scale:.{digits}g} {prefix}{unit}"
    return f"{value:.{digits}g} {unit}"


def _nmos(cx: float, cy: float, *, mirror: bool = False) -> str:
    """One NMOS symbol. Returns SVG with drain pin up, source pin down, gate to the side.

    Pin coordinates are given by `nmos_pins` so the wiring code never guesses where the
    terminals are.
    """
    s = -1 if mirror else 1
    gate_x = cx - 16 * s
    chan_x = cx - 6 * s
    lead_x = cx + 16 * s
    p = []
    # gate plate + gate lead
    p.append(f'<line x1="{gate_x}" y1="{cy - 20}" x2="{gate_x}" y2="{cy + 20}"/>')
    p.append(f'<line x1="{gate_x}" y1="{cy}" x2="{cx - 42 * s}" y2="{cy}"/>')
    # channel
    p.append(f'<line x1="{chan_x}" y1="{cy - 20}" x2="{chan_x}" y2="{cy + 20}"/>')
    # drain lead (up) and source lead (down)
    p.append(f'<line x1="{chan_x}" y1="{cy - 20}" x2="{lead_x}" y2="{cy - 20}"/>')
    p.append(f'<line x1="{lead_x}" y1="{cy - 20}" x2="{lead_x}" y2="{cy - 40}"/>')
    p.append(f'<line x1="{chan_x}" y1="{cy + 20}" x2="{lead_x}" y2="{cy + 20}"/>')
    p.append(f'<line x1="{lead_x}" y1="{cy + 20}" x2="{lead_x}" y2="{cy + 40}"/>')
    # bulk/source arrow (NMOS: arrow points into the channel)
    ax = (chan_x + lead_x) / 2
    p.append(f'<polygon points="{ax - 5 * s},{cy + 15} {ax + 5 * s},{cy + 20} '
             f'{ax - 5 * s},{cy + 25}" class="fill"/>')
    return "".join(p)


def _nmos_pins(cx: float, cy: float, *, mirror: bool = False) -> dict:
    s = -1 if mirror else 1
    return {"d": (cx + 16 * s, cy - 40), "s": (cx + 16 * s, cy + 40), "g": (cx - 42 * s, cy)}


def _res(cx: float, cy: float, label: str, value: str, *, accent: bool,
         horizontal: bool = False) -> str:
    """Resistor: IEC box body, with the label above/left and the value below/right."""
    cls = "val accent" if accent else "val"
    if horizontal:
        body = f'<rect x="{cx - 26}" y="{cy - 10}" width="52" height="20" class="body"/>'
        txt = (f'<text x="{cx}" y="{cy - 18}" class="lbl" text-anchor="middle">{label}</text>'
               f'<text x="{cx}" y="{cy + 30}" class="{cls}" text-anchor="middle">{value}</text>')
    else:
        body = f'<rect x="{cx - 10}" y="{cy - 26}" width="20" height="52" class="body"/>'
        txt = (f'<text x="{cx + 18}" y="{cy - 6}" class="lbl">{label}</text>'
               f'<text x="{cx + 18}" y="{cy + 12}" class="{cls}">{value}</text>')
    return body + txt


def _cap(cx: float, cy: float, label: str, value: str, *, accent: bool,
         horizontal: bool = False, label_left: bool = False) -> str:
    cls = "val accent" if accent else "val"
    if horizontal:
        p = (f'<line x1="{cx - 5}" y1="{cy - 16}" x2="{cx - 5}" y2="{cy + 16}"/>'
             f'<line x1="{cx + 5}" y1="{cy - 16}" x2="{cx + 5}" y2="{cy + 16}"/>')
        txt = (f'<text x="{cx}" y="{cy - 24}" class="lbl" text-anchor="middle">{label}</text>'
               f'<text x="{cx}" y="{cy + 34}" class="{cls}" text-anchor="middle">{value}</text>')
    else:
        p = (f'<line x1="{cx - 16}" y1="{cy - 5}" x2="{cx + 16}" y2="{cy - 5}"/>'
             f'<line x1="{cx - 16}" y1="{cy + 5}" x2="{cx + 16}" y2="{cy + 5}"/>')
        # The label sits on the side away from the input-pair gate leads, which reach
        # outward into this same band.
        tx = cx - 22 if label_left else cx + 22
        anchor = ' text-anchor="end"' if label_left else ""
        txt = (f'<text x="{tx}" y="{cy - 2}" class="lbl"{anchor}>{label}</text>'
               f'<text x="{tx}" y="{cy + 15}" class="{cls}"{anchor}>{value}</text>')
    return p + txt


def _dot(x: float, y: float) -> str:
    return f'<circle cx="{x}" cy="{y}" r="4" class="fill"/>'


def _gnd(x: float, y: float) -> str:
    """Local ground symbol. Used for the load caps so their return path does not have to
    run a long wire down across the bias bus and create a misleading crossing."""
    return (f'<line x1="{x - 14}" y1="{y}" x2="{x + 14}" y2="{y}"/>'
            f'<line x1="{x - 8}" y1="{y + 6}" x2="{x + 8}" y2="{y + 6}"/>'
            f'<line x1="{x - 3}" y1="{y + 12}" x2="{x + 3}" y2="{y + 12}"/>')


def _wire(*pts) -> str:
    d = " ".join(f"{x},{y}" for x, y in pts)
    return f'<polyline points="{d}"/>'


def render(dv: DesignVars, *, title: str = "Delivered CTLE",
           subtitle: str = "") -> str:
    """Return a standalone, theme-aware SVG of the sized circuit."""
    wb_um, fingers = mirror_sizing(dv.i_tail)
    w_um, l_um = dv.w_in * 1e6, dv.l_in * 1e6

    g: list[str] = []

    # ---- rails ------------------------------------------------------------------------
    g.append(f'<line x1="70" y1="{VDD_Y}" x2="{W - 70}" y2="{VDD_Y}" class="rail"/>')
    g.append(f'<text x="70" y="{VDD_Y - 12}" class="rail-lbl">VDD = 1.8 V</text>')
    g.append(f'<line x1="70" y1="{GND_Y}" x2="{W - 70}" y2="{GND_Y}" class="rail"/>')
    g.append(f'<text x="70" y="{GND_Y + 24}" class="rail-lbl">GND</text>')

    # ---- load resistors + output nodes -------------------------------------------------
    rload = _eng(dv.r_load, "Ω")
    for x, name in ((XL, "Rlp"), (XR, "Rln")):
        g.append(_wire((x, VDD_Y), (x, LOAD_Y - 26)))
        g.append(_res(x, LOAD_Y, name, rload, accent=True))
        g.append(_wire((x, LOAD_Y + 26), (x, OUT_Y)))
        g.append(_dot(x, VDD_Y))

    # output taps + fixed load caps (returned to a local ground, not the bottom rail)
    cload = _eng(C_LOAD, "F")
    for x, out, cx_off in ((XL, "outp", -TAP_DX), (XR, "outn", TAP_DX)):
        tap = x + cx_off
        left = cx_off < 0
        g.append(_wire((x, OUT_Y), (tap, OUT_Y)))
        g.append(_dot(x, OUT_Y))
        g.append(f'<text x="{tap + (-10 if left else 10)}" y="{OUT_Y - 14}" '
                 f'class="node" text-anchor="{"end" if left else "start"}">{out}</text>')
        g.append(_wire((tap, OUT_Y), (tap, OUT_Y + 46)))
        g.append(_cap(tap, OUT_Y + 60, "Clp" if left else "Cln", cload, accent=False,
                      label_left=left))
        g.append(_wire((tap, OUT_Y + 70), (tap, OUT_Y + 92)))
        g.append(_gnd(tap, OUT_Y + 92))

    # ---- input pair --------------------------------------------------------------------
    dev_lbl = f"W = {w_um:.2f} um   L = {l_um:.3f} um"
    for x, gate, dev, mir in ((XL, "inp", "XM1", False), (XR, "inn", "XM2", True)):
        cx = x + (16 if not mir else -16)   # so the drain/source pins land on the leg
        pins = _nmos_pins(cx, PAIR_Y, mirror=mir)
        g.append(f'<g class="dev">{_nmos(cx, PAIR_Y, mirror=mir)}</g>')
        g.append(_wire((pins["d"][0], OUT_Y), pins["d"]))
        g.append(_wire(pins["s"], (pins["s"][0], DEG_Y)))
        gx, gy = pins["g"]
        # Keep the gate lead short: it points outward into the same band as the output
        # tap wire, so a long lead would drive the label across that wire.
        g.append(_wire((gx, gy), (gx + (-20 if not mir else 20), gy)))
        anchor = "end" if not mir else "start"
        g.append(f'<text x="{gx + (-28 if not mir else 28)}" y="{gy + 5}" '
                 f'class="node" text-anchor="{anchor}">{gate}</text>')
        g.append(f'<text x="{cx + (44 if not mir else -44)}" y="{PAIR_Y - 8}" '
                 f'class="lbl" text-anchor="{"start" if not mir else "end"}">{dev}</text>')
    g.append(f'<text x="{(XL + XR) / 2}" y="{PAIR_Y - 34}" class="val accent" '
             f'text-anchor="middle">{dev_lbl}</text>')
    g.append(f'<text x="{(XL + XR) / 2}" y="{PAIR_Y - 52}" class="lbl" '
             f'text-anchor="middle">input pair (sky130_fd_pr__nfet_01v8)</text>')

    # ---- source degeneration Rs || Cs --------------------------------------------------
    g.append(f'<text x="{XL - 34}" y="{DEG_Y - 10}" class="node" text-anchor="end">sp</text>')
    g.append(f'<text x="{XR + 34}" y="{DEG_Y - 10}" class="node">sn</text>')
    g.append(_dot(XL, DEG_Y))
    g.append(_dot(XR, DEG_Y))
    mid = (XL + XR) / 2
    # Rs branch (upper) and Cs branch (lower), genuinely in parallel between sp and sn
    g.append(_wire((XL, DEG_Y), (XL, DEG_Y - 30), (mid - 26, DEG_Y - 30)))
    g.append(_wire((mid + 26, DEG_Y - 30), (XR, DEG_Y - 30), (XR, DEG_Y)))
    g.append(_res(mid, DEG_Y - 30, "Rs", _eng(dv.rs, "Ω"), accent=True, horizontal=True))
    g.append(_wire((XL, DEG_Y), (XL, DEG_Y + 42), (mid - 5, DEG_Y + 42)))
    g.append(_wire((mid + 5, DEG_Y + 42), (XR, DEG_Y + 42), (XR, DEG_Y)))
    g.append(_cap(mid, DEG_Y + 42, "Cs", _eng(dv.cs, "F"), accent=True, horizontal=True))
    g.append(f'<text x="{mid}" y="{DEG_Y + 92}" class="hint" text-anchor="middle">'
             f'Rs·Cs zero sets the HF peaking</text>')

    # ---- tail current mirror -----------------------------------------------------------
    mir_lbl = f"W = {wb_um:.2f} um x {fingers}   L = {MIRROR_L_UM:g} um"
    tail_pins = {}
    for x, dev, mir in ((XL, "XMtp", False), (XR, "XMtn", True)):
        cx = x + (16 if not mir else -16)
        pins = _nmos_pins(cx, TAIL_Y, mirror=mir)
        tail_pins[dev] = pins
        g.append(f'<g class="dev">{_nmos(cx, TAIL_Y, mirror=mir)}</g>')
        g.append(_wire((pins["d"][0], DEG_Y), pins["d"]))
        g.append(_wire(pins["s"], (pins["s"][0], GND_Y)))
        g.append(_dot(pins["s"][0], GND_Y))
        g.append(f'<text x="{cx + (44 if not mir else -44)}" y="{TAIL_Y + 4}" '
                 f'class="lbl" text-anchor="{"start" if not mir else "end"}">{dev}</text>')

    # reference leg: Iref from the rail into a diode-connected XMref
    ref_pins = _nmos_pins(REF_X + 16, TAIL_Y)
    g.append(f'<g class="dev">{_nmos(REF_X + 16, TAIL_Y)}</g>')
    g.append(_wire(ref_pins["s"], (ref_pins["s"][0], GND_Y)))
    g.append(_dot(ref_pins["s"][0], GND_Y))
    ix, iy = ref_pins["d"][0], TAIL_Y - 110
    g.append(_wire((ix, VDD_Y), (ix, iy - 22)))
    g.append(_dot(ix, VDD_Y))
    g.append(f'<circle cx="{ix}" cy="{iy}" r="22" class="body"/>')
    g.append(f'<line x1="{ix}" y1="{iy - 10}" x2="{ix}" y2="{iy + 10}"/>')
    g.append(f'<polygon points="{ix - 5},{iy + 2} {ix + 5},{iy + 2} {ix},{iy + 12}" '
             f'class="fill"/>')
    g.append(_wire((ix, iy + 22), ref_pins["d"]))
    g.append(f'<text x="{ix - 30}" y="{iy - 6}" class="lbl" text-anchor="end">Iref</text>')
    g.append(f'<text x="{ix - 30}" y="{iy + 12}" class="val accent" text-anchor="end">'
             f'{_eng(dv.i_tail / 2, "A")}</text>')

    # nbias: diode connection + the gate bus to both tail devices
    nb_y = TAIL_Y + 78
    gx_ref = ref_pins["g"]
    g.append(_wire(gx_ref, (gx_ref[0] - 18, gx_ref[1]), (gx_ref[0] - 18, nb_y)))
    g.append(_wire((gx_ref[0] - 18, nb_y), (tail_pins["XMtn"]["g"][0], nb_y)))
    # diode-connect XMref: gate tied to its own drain
    g.append(_wire((gx_ref[0] - 18, gx_ref[1] - 60), (gx_ref[0] - 18, gx_ref[1])))
    g.append(_wire((gx_ref[0] - 18, gx_ref[1] - 60), (ref_pins["d"][0], gx_ref[1] - 60)))
    g.append(_dot(ref_pins["d"][0], gx_ref[1] - 60))
    for dev in ("XMtp", "XMtn"):
        gxp = tail_pins[dev]["g"]
        g.append(_wire(gxp, (gxp[0], nb_y)))
        g.append(_dot(gxp[0], nb_y))
    g.append(f'<text x="{REF_X - 16}" y="{nb_y - 8}" class="node">nbias</text>')
    # Tail annotation sits in the free right margin: the space between the legs is crossed
    # by both tail drain wires, so centred text there would collide with them.
    ann_x = XR + 108
    g.append(f'<text x="{ann_x}" y="{TAIL_Y - 22}" class="lbl">NMOS current-mirror tail</text>')
    g.append(f'<text x="{ann_x}" y="{TAIL_Y - 2}" class="val accent">'
             f'I_tail = {_eng(dv.i_tail, "A")}</text>')
    g.append(f'<text x="{ann_x}" y="{TAIL_Y + 16}" class="hint">{mir_lbl}</text>')
    g.append(f'<text x="{ann_x}" y="{TAIL_Y + 32}" class="hint">'
             f'drifts with PVT, unlike an ideal source</text>')

    # ---- 1-tap DFE: downstream, dashed, explicitly NOT an analog node -------------------
    dfe_x, dfe_y = W - 85, OUT_Y - 40
    g.append(f'<rect x="{dfe_x - 65}" y="{dfe_y - 30}" width="130" height="62" '
             f'class="ghost"/>')
    g.append(f'<text x="{dfe_x}" y="{dfe_y - 10}" class="lbl" text-anchor="middle">'
             f'1-tap DFE</text>')
    g.append(f'<text x="{dfe_x}" y="{dfe_y + 6}" class="hint" text-anchor="middle">'
             f'RX DSP at the slicer</text>')
    g.append(f'<text x="{dfe_x}" y="{dfe_y + 24}" class="val" text-anchor="middle">'
             f'w_dfe = {dv.w_dfe:g}</text>')
    g.append(f'<text x="{dfe_x}" y="{dfe_y + 50}" class="hint" text-anchor="middle">'
             f'not an analog node</text>')

    header = (f'<text x="40" y="34" class="title">{title}</text>'
              + (f'<text x="40" y="54" class="sub">{subtitle}</text>' if subtitle else ""))
    legend = (f'<text x="{W - 40}" y="34" class="legend-accent" text-anchor="end">'
              f'■ sized by the RL agent</text>'
              f'<text x="{W - 40}" y="52" class="legend-fixed" text-anchor="end">'
              f'■ fixed by the topology</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     role="img" aria-label="Schematic of the delivered CTLE with RL-chosen device values"
     class="silq-schematic">
  <style>
    .silq-schematic {{ --ink:#1c2230; --muted:#6b7688; --accent:#c2410c; --line:#3a4356;
                       --ghost:#9aa5b6; font-family: ui-sans-serif, system-ui, sans-serif; }}
    @media (prefers-color-scheme: dark) {{
      .silq-schematic {{ --ink:#e6eaf2; --muted:#98a3b6; --accent:#fb923c; --line:#c3ccdc;
                         --ghost:#7b8698; }}
    }}
    .silq-schematic line, .silq-schematic polyline, .silq-schematic circle,
    .silq-schematic rect, .silq-schematic polygon
        {{ fill:none; stroke:var(--line); stroke-width:1.7;
           stroke-linecap:round; stroke-linejoin:round; }}
    .silq-schematic .fill {{ fill:var(--line); stroke:none; }}
    .silq-schematic .body {{ fill:none; stroke:var(--line); stroke-width:1.7; }}
    .silq-schematic .rail {{ stroke-width:2.6; }}
    .silq-schematic .ghost {{ stroke:var(--ghost); stroke-dasharray:5 4; stroke-width:1.5; }}
    .silq-schematic text {{ stroke:none; }}
    .silq-schematic .title {{ fill:var(--ink); font-size:19px; font-weight:650; }}
    .silq-schematic .sub, .silq-schematic .hint {{ fill:var(--muted); font-size:11.5px; }}
    .silq-schematic .lbl {{ fill:var(--ink); font-size:12.5px; font-weight:600; }}
    .silq-schematic .node {{ fill:var(--muted); font-size:12px; font-style:italic; }}
    .silq-schematic .rail-lbl {{ fill:var(--muted); font-size:12px; font-weight:600; }}
    .silq-schematic .val {{ fill:var(--muted); font-size:12px;
                            font-variant-numeric:tabular-nums; }}
    .silq-schematic .val.accent {{ fill:var(--accent); font-weight:650; }}
    .silq-schematic .legend-accent {{ fill:var(--accent); font-size:11.5px; font-weight:600; }}
    .silq-schematic .legend-fixed {{ fill:var(--muted); font-size:11.5px; font-weight:600; }}
  </style>
  {header}{legend}
  {''.join(g)}
</svg>"""


def render_delivered(path: str = "results/delivered_circuit.json") -> str:
    """Render the frozen delivered circuit straight from its manifest."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text())
    dv = DesignVars(**d["design"])
    spec = d.get("spec", {})
    sub = ("target {t:.2f} dB boost over a {c:.2f} dB channel  |  "
           "sized in {n} SPICE evaluations  |  frozen, not re-optimised").format(
        t=spec.get("target_boost_db", float("nan")),
        c=spec.get("channel_loss_db", float("nan")),
        n=d.get("provenance", {}).get("total_evals", "?"))
    return render(dv, title="Delivered CTLE, SILQ", subtitle=sub)


def main() -> None:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Render the delivered circuit as an SVG.")
    p.add_argument("--out", default="results/delivered_schematic.svg")
    p.add_argument("--circuit", default="results/delivered_circuit.json")
    a = p.parse_args()
    Path(a.out).write_text(render_delivered(a.circuit), encoding="utf-8")
    print(f"schematic -> {a.out}")


if __name__ == "__main__":
    main()
