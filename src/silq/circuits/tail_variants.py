"""Experimental tail implementations; never selected by the shipped pipeline.

Wide-swing cascode: ECE5211 Chapter 6 part 2, Figure 12 (n=1),
https://www.d.umn.edu/~htang/ECE5211_doc_files/ECE5211_files/Chapter6_part2.pdf
This improves compliance relative to a conventional cascode, not necessarily to
the shipped simple mirror. All bulk terminals remain grounded as in SKY130.
The bias device is smaller than W/4 to provide margin and allow for body effect.
Reference currents are ideal trimmed sources, as in the baseline; neither circuit
models an autonomous reference generator or constitutes layout sign-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from silq.circuits import ctle


@dataclass(frozen=True)
class TailConfig:
    kind: str = "simple"
    width_scale: float = 1.0
    bias_width_ratio: float = 0.18

    def __post_init__(self):
        if self.kind not in {"simple", "wide_swing"}:
            raise ValueError(f"unknown tail kind: {self.kind}")
        if not math.isfinite(self.width_scale) or not 1 <= self.width_scale <= 16:
            raise ValueError("width_scale must be finite and in [1, 16]")
        if not math.isfinite(self.bias_width_ratio) or not 0.05 <= self.bias_width_ratio <= 0.25:
            raise ValueError("bias_width_ratio must be finite and in [0.05, 0.25]")


def split_width(total_um: float) -> tuple[float, int]:
    total_um = max(total_um, ctle.MIRROR_W_MIN_UM)
    fingers = max(1, math.ceil(total_um / ctle.MIRROR_W_MAX_UM))
    return total_um / fingers, fingers


def geometry(dv: ctle.DesignVars, config: TailConfig) -> dict[str, float]:
    w, m = ctle.mirror_sizing(dv.i_tail)
    wt, mt = split_width(w * m * config.width_scale)
    wq, mq = split_width(w * m * config.width_scale * config.bias_width_ratio)
    return {"wt": wt, "mt": mt, "wq": wq, "mq": mq}


@dataclass
class TailDesign(ctle.DesignVars):
    tail: TailConfig = field(default_factory=TailConfig)

    def area_mm2(self) -> float:
        w, m = ctle.mirror_sizing(self.i_tail)
        g = geometry(self, self.tail)
        # Simple: three devices. Wide swing: two devices per reference/output
        # branch (six total), plus the diode-connected cascode-bias transistor.
        width = (3 if self.tail.kind == "simple" else 6) * g["wt"] * g["mt"]
        if self.tail.kind == "wide_swing":
            width += g["wq"] * g["mq"]
        return super().area_mm2() + (width - 3 * w * m) * ctle.MIRROR_L_UM * 1e-6


def instances(config: TailConfig) -> tuple[str, ...]:
    base = ("XM1", "XM2", "XMref", "XMtp", "XMtn")
    return base if config.kind == "simple" else base + ("XMrc", "XMcp", "XMcn", "XMcb")


def nodes(config: TailConfig) -> tuple[str, ...]:
    base = ("outp", "outn", "sp", "sn", "nbias", "cm")
    return base if config.kind == "simple" else base + ("nref", "ntp", "ntn", "ncas")


def tail_devices(config: TailConfig) -> tuple[str, ...]:
    # Read current at the CTLE source nodes, including cascode body-current effects.
    return ("XMtp", "XMtn") if config.kind == "simple" else ("XMcp", "XMcn")


def param_deck(corner: str, config: TailConfig) -> str:
    deck = ctle.param_deck(corner)
    marker = f"Vcm cm 0 'vddp*{ctle.VCM_VDD_RATIO}'"
    if deck.count(marker) != 1 or deck.count(ctle._MIRROR) != 1:
        raise ValueError("baseline deck changed; review the experimental adapter")
    deck = deck.replace(marker, "Vcm cm 0 {vddp*vcm_ratio}")
    deck = deck.replace(".end\n", ".param vcm_ratio=0.72 wt=80 mt=2 wq=14.4 mq=2\n.end\n")
    model = "sky130_fd_pr__nfet_01v8 L={lb} W={wt} nf=1 m={mt}"
    if config.kind == "simple":
        mirror = ctle._MIRROR.replace("W={wb}", "W={wt}").replace("m={mb}", "m={mt}")
    else:
        mirror = (
            "Iref vdd nbias 'itail/2'\n"
            f"XMrc nbias ncas nref 0 {model}\n"
            f"XMref nref nbias 0 0 {model}\n"
            f"XMcp sp ncas ntp 0 {model}\n"
            f"XMtp ntp nbias 0 0 {model}\n"
            f"XMcn sn ncas ntn 0 {model}\n"
            f"XMtn ntn nbias 0 0 {model}\n"
            "Ibcas vdd ncas 'itail/2'\n"
            "XMcb ncas ncas 0 0 sky130_fd_pr__nfet_01v8 L={lb} W={wq} nf=1 m={mq}\n"
        )
    return deck.replace(ctle._MIRROR, mirror)


def parameters(dv: TailDesign, *, vdd: float, temp_c: float, vcm_ratio: float) -> dict[str, float]:
    if not math.isfinite(vcm_ratio) or not 0 < vcm_ratio < 1:
        raise ValueError("vcm_ratio must be finite and between zero and one")
    return dict(ctle.dv_to_params(dv), **geometry(dv, dv.tail),
                vddp=vdd, tempc=temp_c, vcm_ratio=vcm_ratio)


def snapshot(dv: TailDesign, *, corner: str, vdd: float, temp_c: float,
             vcm_ratio: float, analysis: str = "op", **_kw) -> str:
    deck = param_deck(corner, dv.tail)
    # Same six-significant-digit parameter values sent to the resident simulator.
    assignments = "\n".join(f".param {k}={v:.6g}" for k, v in
                            parameters(dv, vdd=vdd, temp_c=temp_c, vcm_ratio=vcm_ratio).items())
    return deck.replace(".end\n", assignments + "\n" + ctle._ANALYSIS[analysis] + "\n.end\n")
