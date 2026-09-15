"""Bounded SNR/BER characterization for the separately versioned eye measurement.

This script is deliberately outside every reward, guard and frozen report path.  It
acquires one complex CTLE response and one 10 MHz--5 GHz SPICE input-referred noise value
for a selected circuit, then writes a new JSON/PNG pair.  It never updates an existing
result.  Default cost is exactly two SPICE analyses plus 81 behavioural evaluations:
three noiseless references and three equalization conditions x 26 SNR points, each at
2,048 bits.  Every behavioural evaluation builds phase-training, DFE-training and
evaluation patterns; ``--n-bits`` is bounded to 8,192 so it cannot silently turn into a
long sweep.

The axis is *amplitude SNR at the slicer*, ``20 log10(A / sigma)``.  ``A`` is the absolute
main-cursor voltage for that response; each point scales a sigma derived from the measured
input-referred noise so this is a characterization axis, not a claim about a new measured
noise source.  Semi-analytic BER is ``mean(Q(margin / sigma))`` on noiseless sampled
margins and assumes correct DFE feedback.  The noisy Monte-Carlo trace feeds the receiver's
own decisions back, so it captures error propagation at observable BER but cannot resolve
rare events below roughly ``1 / n_bits``.

The scalar SPICE ``inoise_total`` is not a noise PSD.  We refer it to the slicer through
the peak magnitude of the same measured CTLE response only under a flat/equivalent
input-noise-spectrum assumption.  It cannot establish a PSD-shaped output sigma, channel
thermal noise, jitter, or a silicon BER.  A future statistical-eye implementation can
replace the scalar referral with a measured output PSD without changing this artifact schema.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eqrl.circuits.ctle import DesignVars
from eqrl.sim.eye import EyeResultV2, compute_eye_v2
from eqrl.sim.server import get_server


SNR_START_DB = -5.0
SNR_STOP_DB = 20.0
DEFAULT_STEP_DB = 1.0
DEFAULT_BITS = 2048
MAX_BITS = 8192
DEFAULT_OUT = Path("results/snr_ber_sweep_v1.json")
DEFAULT_PLOT = Path("results/snr_ber_sweep_v1.png")


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    dfe_taps: int
    H_ctle: np.ndarray


def _q(x: np.ndarray) -> np.ndarray:
    """Gaussian Q-function without making SciPy a runtime requirement for this script."""
    root_two = math.sqrt(2.0)
    return np.fromiter((0.5 * math.erfc(float(v) / root_two) for v in x),
                       dtype=float, count=x.size)


def _snr_grid(step_db: float, start_db: float = SNR_START_DB,
              stop_db: float = SNR_STOP_DB) -> np.ndarray:
    """Endpoints are inclusive and must both land on the grid.

    The default window is the requested -5..20 dB. It is a parameter rather than a
    constant because that window does not contain every crossing it is used to read:
    the delivered design's CTLE-on curve reaches BER 1e-12 at roughly 21 dB, so the
    default run reports a null `ctle_on` crossing -- absent from the axis, not
    unattainable. Widening the axis is a stated choice recorded in the artifact's
    `axis.range_db`, never a silent change to the default.
    """
    if not np.isfinite(step_db) or step_db <= 0.0:
        raise ValueError("--snr-step-db must be a positive finite number")
    if not (np.isfinite(start_db) and np.isfinite(stop_db)) or stop_db <= start_db:
        raise ValueError("--snr-stop-db must be finite and greater than --snr-start-db")
    grid = np.arange(start_db, stop_db + 0.5 * step_db, step_db)
    grid = grid[grid <= stop_db + 1e-12]
    if grid.size > 101:
        raise ValueError("the requested SNR grid has more than 101 points")
    if grid[0] != start_db or abs(grid[-1] - stop_db) > 1e-9:
        raise ValueError(
            f"--snr-step-db must include both {start_db:g} dB and {stop_db:g} dB endpoints")
    return grid


def _load_delivered(path: Path) -> tuple[DesignVars, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DesignVars(**data["design"]), float(data["spec"]["channel_loss_db"])


def _ac_and_input_noise(dv: DesignVars, *, corner: str, vdd: float, temp_c: float) -> tuple[dict, float]:
    """Acquire the response and the scalar noise source once, before all DSP sweeps."""
    srv = get_server(corner)
    srv.set_corner(corner)
    response = srv.ac_complex(dv, vdd=vdd, temp_c=temp_c)
    input_noise_vrms = float(srv.noise_total(dv, vdd=vdd, temp_c=temp_c))
    if not np.isfinite(input_noise_vrms) or input_noise_vrms <= 0.0:
        raise ValueError("SPICE input-referred noise must be positive and finite")
    return response, input_noise_vrms


def _crossing_snr_db(snr_db: np.ndarray, ber: np.ndarray, target: float) -> float | None:
    """Log-BER interpolation of a descending curve's crossing with ``target``."""
    if target <= 0.0 or not np.isfinite(target):
        raise ValueError("equalization BER target must be positive and finite")
    for i in range(len(snr_db) - 1):
        lo, hi = ber[i], ber[i + 1]
        if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > 0.0):
            continue
        if lo >= target >= hi:
            y0, y1, yt = math.log10(lo), math.log10(hi), math.log10(target)
            if y0 == y1:
                return float(snr_db[i])
            alpha = (yt - y0) / (y1 - y0)
            return float(snr_db[i] + alpha * (snr_db[i + 1] - snr_db[i]))
    return None


def _semi_analytic(result: EyeResultV2, sigma_v: float) -> float:
    """Correct-feedback approximation from the noiseless sampled-margin distribution."""
    return float(np.mean(_q(result.margins_v / sigma_v)))


def _condition_data(condition: Condition, freq: np.ndarray, *, channel_loss_db: float,
                    n_bits: int, seed: int, snr_db: np.ndarray, input_noise_vrms: float
                    ) -> dict:
    """Build one semi-analytic and one independent noisy-MC trace for a link condition."""
    reference = compute_eye_v2(
        freq, condition.H_ctle, channel_loss_db=channel_loss_db, n_bits=n_bits,
        dfe_taps=condition.dfe_taps, seed=seed, noise_sigma_v=0.0)
    amplitude_v = abs(reference.c0)
    if amplitude_v <= np.finfo(float).eps:
        raise ValueError(f"{condition.label} has no non-zero main cursor")

    # ``noise_total`` is input-referred at the CTLE input.  This peak-gain referral is the
    # same one behind the contextual ~45 dB estimate for the delivered link.  It is an
    # equivalent broad-band estimate, not a measured output-noise spectrum.
    peak_voltage_gain = float(np.max(np.abs(condition.H_ctle)))
    sigma_reference_v = input_noise_vrms * peak_voltage_gain
    opening_amplitude_v = max(reference.signed_opening_v, 0.0) / 2.0
    operating_snr_db = (20.0 * math.log10(opening_amplitude_v / sigma_reference_v)
                        if opening_amplitude_v > 0.0 and sigma_reference_v > 0.0 else None)

    semi_ber: list[float] = []
    mc_ber: list[float] = []
    mc_errors: list[int] = []
    mc_count: list[int] = []
    noise_scale: list[float] = []
    for point, requested_snr_db in enumerate(snr_db):
        sigma_v = amplitude_v / (10.0 ** (requested_snr_db / 20.0))
        semi_ber.append(_semi_analytic(reference, sigma_v))
        noisy = compute_eye_v2(
            freq, condition.H_ctle, channel_loss_db=channel_loss_db, n_bits=n_bits,
            dfe_taps=condition.dfe_taps, seed=seed + 1000 + point,
            noise_sigma_v=sigma_v)
        mc_ber.append(noisy.ber)
        mc_errors.append(noisy.errors)
        mc_count.append(noisy.count)
        noise_scale.append(sigma_v / sigma_reference_v)

    return {
        "key": condition.key,
        "label": condition.label,
        "dfe_taps": condition.dfe_taps,
        "main_cursor_amplitude_v": amplitude_v,
        "noiseless_signed_opening_v": reference.signed_opening_v,
        "noiseless_errors": reference.errors,
        "noiseless_count": reference.count,
        "correct_feedback_assumption": {
            "statement": "Semi-analytic BER assumes all prior DFE feedback decisions are correct.",
            "noiseless_receiver_trace_has_zero_errors": reference.errors == 0,
            "scope": "When false, retain the curve only as a conditional approximation; use the noisy MC trace for observed decision feedback.",
        },
        "noise_referral": {
            "input_noise_vrms": input_noise_vrms,
            "peak_voltage_gain": peak_voltage_gain,
            "equivalent_output_sigma_vrms": sigma_reference_v,
            "assumption": "Input-referred scalar noise is treated as flat/equivalent through the peak CTLE response; it is not an output PSD.",
        },
        "contextual_operating_snr_db": operating_snr_db,
        "points": [
            {
                "snr_db": float(db),
                "sigma_vrms": amplitude_v / (10.0 ** (float(db) / 20.0)),
                "scale_of_measured_equivalent_sigma": float(scale),
                "semi_analytic_ber_correct_feedback": float(analytic),
                "noisy_mc_ber_receiver_feedback": float(observed),
                "noisy_mc_errors": int(errors),
                "noisy_mc_count": int(count),
                "noisy_mc_zero_resolution_upper_scale": 1.0 / count,
            }
            for db, scale, analytic, observed, errors, count in zip(
                snr_db, noise_scale, semi_ber, mc_ber, mc_errors, mc_count, strict=True)
        ],
    }


def _plot(curves: list[dict], out: Path, *, equalization_ber: float,
          start_db: float = SNR_START_DB, stop_db: float = SNR_STOP_DB) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    for curve in curves:
        points = curve["points"]
        x = np.array([p["snr_db"] for p in points])
        semi = np.array([p["semi_analytic_ber_correct_feedback"] for p in points])
        observed = np.array([p["noisy_mc_ber_receiver_feedback"] for p in points])
        observed[observed == 0.0] = np.nan  # a zero count is below MC resolution, not zero BER
        (line,) = ax.semilogy(x, semi, label=f"{curve['label']} semi-analytic (correct feedback)")
        ax.semilogy(x, observed, "o", color=line.get_color(), ms=3.5,
                    label=f"{curve['label']} noisy MC (receiver feedback)")
    ax.axhline(equalization_ber, color="#555", lw=0.9, ls="--",
               label=f"equalization reference BER = {equalization_ber:.0e}")
    dfe_curve = next(c for c in curves if c["key"] == "ctle_dfe")
    context = dfe_curve["contextual_operating_snr_db"]
    if context is not None:
        if start_db <= context <= stop_db:
            ax.axvline(context, color="#333", lw=0.9, ls=":")
            ax.text(context, 0.45, f"measured-noise context {context:.1f} dB",
                    rotation=90, va="center", ha="right", fontsize=8)
        else:
            ax.annotate(f"measured-noise context {context:.1f} dB lies beyond this axis",
                        xy=(stop_db, 0.45), xytext=(stop_db - 10.0, 0.15),
                        arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=8)
    ax.set(xlim=(start_db, stop_db), ylim=(1e-14, 1.0),
           xlabel="Slicer amplitude SNR (dB): 20 log10(A / sigma)",
           ylabel="BER", title="Behavioural SNR/BER characterization, not silicon sign-off")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delivered", type=Path, default=Path("results/delivered_circuit.json"))
    p.add_argument("--corner", default="tt")
    p.add_argument("--vdd", type=float, default=1.8)
    p.add_argument("--temp-c", type=float, default=27.0)
    p.add_argument("--n-bits", type=int, default=DEFAULT_BITS)
    p.add_argument("--snr-step-db", type=float, default=DEFAULT_STEP_DB)
    p.add_argument("--snr-start-db", type=float, default=SNR_START_DB)
    p.add_argument("--snr-stop-db", type=float, default=SNR_STOP_DB)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--equalization-ber", type=float, default=1e-12)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    if not 16 <= args.n_bits <= MAX_BITS:
        raise SystemExit(f"--n-bits must be in [16, {MAX_BITS}]")
    if not np.isfinite(args.equalization_ber) or args.equalization_ber <= 0.0:
        raise SystemExit("--equalization-ber must be positive and finite")
    if (args.out.exists() or args.plot.exists()) and not args.overwrite:
        raise SystemExit("refusing to overwrite an artifact; pass --overwrite deliberately")
    snr_db = _snr_grid(args.snr_step_db, args.snr_start_db, args.snr_stop_db)
    dv, channel_loss_db = _load_delivered(args.delivered)
    response, input_noise_vrms = _ac_and_input_noise(
        dv, corner=args.corner, vdd=args.vdd, temp_c=args.temp_c)
    freq, H_ctle = response["freq"], response["H"]
    conditions = (
        Condition("ctle_off", "CTLE off", 0, np.ones_like(H_ctle)),
        Condition("ctle_on", "CTLE on", 0, H_ctle),
        Condition("ctle_dfe", "CTLE + DFE", 1, H_ctle),
    )
    curves = [
        _condition_data(c, freq, channel_loss_db=channel_loss_db, n_bits=args.n_bits,
                        seed=args.seed + i * 100_000, snr_db=snr_db,
                        input_noise_vrms=input_noise_vrms)
        for i, c in enumerate(conditions)
    ]
    crossings = {
        c["key"]: _crossing_snr_db(
            snr_db,
            np.array([p["semi_analytic_ber_correct_feedback"] for p in c["points"]]),
            args.equalization_ber)
        for c in curves
    }
    report = {
        "schema": "eqrl.snr_ber_sweep.v1",
        "status": "generated by an unguarded, bounded characterization script; not a reward or sign-off artifact",
        "axis": {
            "name": "slicer_amplitude_snr_db",
            "definition": "20 log10(A / sigma)",
            "range_db": [args.snr_start_db, args.snr_stop_db],
            "range_is_default": ([args.snr_start_db, args.snr_stop_db]
                                 == [SNR_START_DB, SNR_STOP_DB]),
            "points": [float(x) for x in snr_db],
        },
        "cost_contract": {
            "spice_analyses": "one ac_complex response plus one input-referred .noise analysis",
            "behavioural_evaluations": len(curves) * (1 + len(snr_db)),
            "pattern_waveforms_per_behavioural_evaluation": 3,
            "bits_per_trace": args.n_bits,
            "maximum_bits_per_trace": MAX_BITS,
        },
        "noise_limitations": [
            "SPICE inoise_total is a scalar input-referred 10 MHz--5 GHz value, not a PSD.",
            "The output referral uses peak CTLE gain under a flat/equivalent-spectrum assumption.",
            "The link channel is not used as a noise transfer because the measured source is CTLE input-referred; channel thermal noise is absent.",
            "The injected MC noise is white additive slicer noise; it excludes jitter and clock recovery.",
        ],
        "selected_design": {"values": dv.__dict__, "channel_loss_db": channel_loss_db,
                            "corner": args.corner, "vdd": args.vdd, "temp_c": args.temp_c},
        "equalization_reference": {
            "ber": args.equalization_ber,
            "semi_analytic_snr_at_ber_db": crossings,
            "ctle_on_minus_off_db": (
                crossings["ctle_off"] - crossings["ctle_on"]
                if crossings["ctle_off"] is not None and crossings["ctle_on"] is not None else None),
            "ctle_dfe_minus_ctle_on_db": (
                crossings["ctle_on"] - crossings["ctle_dfe"]
                if crossings["ctle_on"] is not None and crossings["ctle_dfe"] is not None else None),
            "interpretation": "Positive shift means less slicer amplitude SNR is required at the stated BER under the correct-feedback approximation.",
        },
        "curves": curves,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(curves, args.plot, equalization_ber=args.equalization_ber,
          start_db=args.snr_start_db, stop_db=args.snr_stop_db)
    print(f"wrote {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
