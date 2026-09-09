"""Behavioural 1-tap decision-feedback equalizer (DFE) — the receiver stage that follows
the CTLE, as an actual simulatable netlist.

WHY THIS IS A SEPARATE STAGE, NOT PART OF THE CTLE `.ac` DECK
------------------------------------------------------------
A DFE is a *sampled, decision-driven* block: at each symbol it subtracts
`tap * previous_decision`. It is fundamentally nonlinear and clocked, so it has **no**
small-signal `.ac` transfer function and must never be placed in the CTLE's `.ac`
characterization deck — doing so would corrupt the linear CTLE measurement (boost, peak,
noise). That is exactly why the CTLE is sized in SPICE (`.ac`) and the DFE lives here, in
a transient testbench. Astera's topology ("1-Stage CTLE ... + 1-Tap DFE") is realised as
CTLE(transistors) -> this DFE stage.

WHAT THE TESTBENCH IS
---------------------
A minimal, convergent transient model of the receiver slicer path:

    received  rx(t)  = c0·b(t) + c1·b(t-UI)        ; main cursor + first post-cursor (ISI)
    equalized out(t) = rx(t)  - tap·b̂(t-UI)         ; 1-tap DFE subtracts the post-cursor

where (c0, c1) are the *real* cursors of this design's channel+CTLE pulse response
(eye.pulse_cursors), so the ISI the DFE cancels is the ISI this circuit actually produces.
The feedback uses the known transmitted bit for the previous decision — the standard
design-time simplification (correct-decision assumption; error propagation is a separate
analysis). Built from linear `E poly` sources so ngspice converges every time and no PDK
is needed.

THIS MODULE IS NOT ON THE SCORING PATH — READ THIS BEFORE QUOTING IT
--------------------------------------------------------------------
Nothing in the measurement, training, PVT or reporting path imports this module. It is a
hand-run characterisation and schematic-export utility, reached only through its own CLI
(`main` below). Every published number was measured with a DIFFERENT 1-tap DFE: the one
inside the eye engine, where `compute_eye` (`sim/eye.py`) adapts a tap to the measured
first post-cursor and clips it, on by default. `measure_all` takes that default, so every
reward, every one of the PVT corners and every eye number is a post-DFE eye.

The consequence for `dv.w_dfe`: it is read on no scoring path, so its value influences no
published number. Inside `measure_dfe` below it picks a tap fraction for that
characterisation, and that is its only effect anywhere in the repo. That deadness is
deliberate, not an oversight — the reason `w_dfe` is excluded from the action vector is
given at its exclusion comment (`ctle.py:97-99`), and `tests/test_dfe_is_off_the_scoring_path.py`
pins both facts so they cannot change silently.
"""
from __future__ import annotations

import numpy as np

from eqrl.circuits.ctle import DesignVars
from eqrl.sim.ngspice_runner import run

UI_DEFAULT = 200e-12          # 5 Gb/s
SPS_DEFAULT = 20              # samples per UI in the transient
NBITS_DEFAULT = 128


def _pattern(n_bits: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n_bits) * 2 - 1     # ±1


def _pwl(bits: np.ndarray, ui: float, level: float) -> str:
    pts = []
    for k, v in enumerate(bits):
        t0 = k * ui
        pts.append((t0, v * level))
        pts.append((t0 + ui - 1e-13, v * level))
    return " ".join(f"{t:.6e} {a:.5f}" for t, a in pts)


def dfe_testbench_netlist(c0: float, c1: float, tap: float, *, ui: float = UI_DEFAULT,
                          sps: int = SPS_DEFAULT, n_bits: int = NBITS_DEFAULT,
                          seed: int = 0) -> tuple[str, np.ndarray]:
    """Emit the transient 1-tap-DFE testbench deck (the exportable receiver schematic).

    Returns (netlist, transmitted_bits). Node `outn` is the equalized slicer input.
    """
    bits = _pattern(n_bits, seed)
    bd = np.r_[0, bits[:-1]]                       # previous bit (1-UI delayed decision)
    level = 1.0                                    # ±1 symbol; cursors carry the volts
    deck = (
        "* Receiver 1-tap DFE stage (transient). Node outn = equalized slicer input.\n"
        f"* cursors: c0={c0:.6g} V (main), c1={c1:.6g} V (first post-cursor); tap={tap:.6g}\n"
        f"Vb  nb  0 PWL({_pwl(bits, ui, level)})\n"
        f"Vbd nbd 0 PWL({_pwl(bd, ui, level)})\n"
        f"* received signal = main cursor + first post-cursor (the ISI to cancel)\n"
        f"Erx rxn 0 poly(2) nb 0 nbd 0 0 {c0:.6g} {c1:.6g}\n"
        f"* 1-tap DFE: subtract tap * previous decision (summing node)\n"
        f"Eout outn 0 poly(2) rxn 0 nbd 0 0 1 {-tap:.6g}\n"
        ".end\n"
    )
    return deck, bits


def _eye_height(deck: str, bits: np.ndarray, *, ui: float, sps: int, sign: float = 1.0,
                warm: int = 6) -> float:
    """Run the deck, sample `outn` at each UI centre, return eye height (V).

    `sign` normalises for an inverting CTLE (c0 < 0) so the eye height is positive.
    """
    control = (f"tran {ui/sps:.4e} {len(bits)*ui:.4e}\n"
               "let vout = v(outn)\n"
               "wrdata $OUT vout")
    res = run(deck, control=control)
    arr = np.atleast_2d(res["data"])
    t, v = arr[:, 0], arr[:, 1] * sign
    centres = (np.arange(warm, len(bits)) + 0.5) * ui
    idx = np.searchsorted(t, centres)
    idx = idx[idx < len(t)]
    s = v[idx]
    dec = bits[warm:warm + len(s)]
    ones, zeros = s[dec > 0], s[dec < 0]
    if not len(ones) or not len(zeros):
        return 0.0
    return float(ones.min() - zeros.max())


def optimal_tap(c1: float) -> float:
    """The tap an adaptive (LMS) 1-tap DFE converges to: it cancels the first post-cursor."""
    return float(c1)


def dfe_stage_eye(tap: float, *, c0: float = 0.5, c1: float = 0.28,
                  ui: float = UI_DEFAULT, sps: int = SPS_DEFAULT,
                  n_bits: int = NBITS_DEFAULT, seed: int = 0) -> float:
    """Eye height (V) at the slicer for a given DFE tap and (c0, c1). Sweeping `tap`
    changes this — it peaks at tap ≈ c1 (full post-cursor cancellation)."""
    deck, bits = dfe_testbench_netlist(c0, c1, tap, ui=ui, sps=sps, n_bits=n_bits, seed=seed)
    sign = 1.0 if c0 >= 0 else -1.0
    return _eye_height(deck, bits, ui=ui, sps=sps, sign=sign)


def measure_dfe(dv: DesignVars, freq_ac: np.ndarray, H_ctle: np.ndarray, *,
                channel_loss_db: float = 12.0, seed: int = 0) -> dict:
    """Characterise the 1-tap DFE for a real design: extract the channel+CTLE cursors,
    and measure the eye at the slicer with no tap, at the adaptive optimum, and at the
    user's pinned tap fraction `dv.w_dfe`.

    NO CALLER ON THE SCORING PATH. This function is a hand-run characterisation utility;
    its only caller in the repo is `export_dfe_schematic` below, which is itself reached
    only from this module's CLI. Nothing in measurement, training, PVT or reporting calls
    either. So `dv.w_dfe` steers `eye_h_at_wdfe_v` in the dict returned here and nothing
    else: no reward, no corner, no published number depends on its value. The DFE that
    the scored eye actually runs is the adaptive one in `compute_eye` (`sim/eye.py`),
    which is always on and adapts its tap to the measured post-cursor — this testbench
    does not feed it. `w_dfe` is deliberately not an action variable; the reason is at
    its exclusion comment (`ctle.py:97-99`).

    `dv.w_dfe` is a FRACTION of the optimal tap (0 = no tap applied in THIS testbench,
    which is not the same thing as the scored eye's DFE being off; 1 = full post-cursor
    cancellation; >1 = over-correct). The sign is carried by c1, so the knob is sign-safe.
    """
    from eqrl.sim.eye import pulse_cursors

    c0, c1 = pulse_cursors(freq_ac, H_ctle, channel_loss_db=channel_loss_db)
    tap_opt = optimal_tap(c1)                       # what an adaptive (LMS) DFE converges to
    frac = float(dv.w_dfe)
    tap_wdfe = frac * tap_opt
    eye_off = dfe_stage_eye(0.0, c0=c0, c1=c1, seed=seed)
    eye_adapt = dfe_stage_eye(tap_opt, c0=c0, c1=c1, seed=seed)
    eye_wdfe = dfe_stage_eye(tap_wdfe, c0=c0, c1=c1, seed=seed)
    return {
        "c0_v": c0, "c1_v": c1,
        "tap_optimal": tap_opt, "w_dfe_fraction": frac, "tap_applied": tap_wdfe,
        "eye_h_nodfe_v": eye_off,
        "eye_h_adaptive_v": eye_adapt,
        "eye_h_at_wdfe_v": eye_wdfe,
        "dfe_gain_v": eye_adapt - eye_off,          # what the adaptive DFE recovers
    }


def export_dfe_schematic(dv: DesignVars, freq_ac: np.ndarray, H_ctle: np.ndarray,
                         path: str, *, channel_loss_db: float = 12.0) -> dict:
    """Write the design's 1-tap DFE receiver-stage netlist (the schematic Astera's topology
    asks for, after the CTLE) and return its DFE characterisation. Uses the design's real
    channel+CTLE cursors and the adaptive optimum tap."""
    from pathlib import Path
    from eqrl.sim.eye import pulse_cursors

    c0, c1 = pulse_cursors(freq_ac, H_ctle, channel_loss_db=channel_loss_db)
    deck, _ = dfe_testbench_netlist(c0, c1, optimal_tap(c1))
    Path(path).write_text(deck)
    return measure_dfe(dv, freq_ac, H_ctle, channel_loss_db=channel_loss_db)


def main() -> None:
    """CLI: emit the 1-tap DFE receiver-stage schematic for a saved design.

        python -m eqrl.circuits.dfe --design results/delivered_circuit.json \
            --channel 14.8 --out results/delivered_dfe_stage.spice
    """
    import argparse
    import json

    from eqrl.sim.server import get_server

    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True, help="design JSON (bare dv or {'design': ...})")
    p.add_argument("--channel", type=float, default=12.0, help="channel loss at Nyquist (dB)")
    p.add_argument("--out", default="results/dfe_stage.spice")
    args = p.parse_args()

    data = json.loads(open(args.design).read())
    d = data.get("design", data)
    dv = DesignVars(**{k: d[k] for k in DesignVars().__dict__ if k in d})
    H = get_server("tt").ac_complex(dv)
    r = export_dfe_schematic(dv, H["freq"], H["H"], args.out, channel_loss_db=args.channel)
    print(f"DFE stage schematic -> {args.out}")
    print(f"  first post-cursor c1 = {r['c1_v']*1e3:.1f} mV, adaptive tap = "
          f"{r['tap_optimal']*1e3:.1f} mV")
    print(f"  eye at slicer: DFE off {r['eye_h_nodfe_v']*1e3:.0f} mV -> "
          f"adaptive DFE {r['eye_h_adaptive_v']*1e3:.0f} mV "
          f"(+{r['dfe_gain_v']*1e3:.0f} mV recovered)")


if __name__ == "__main__":
    main()
