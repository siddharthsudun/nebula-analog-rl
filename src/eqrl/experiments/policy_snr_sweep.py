"""Evaluation-only SNR robustness sweep over the TRAINED POLICY.

This is a stress test bolted onto the side of the submission. It is NOT part of the
Astera Labs problem statement, which specifies no link SNR and no BER target, and it
changes nothing about the frozen objective, reward, guard layer or benchmark. The main
benchmark numbers are produced elsewhere and are untouched by this file.

WHAT IS AND IS NOT SNR-DEPENDENT HERE -- read this before reading the table
--------------------------------------------------------------------------
The scored eye on the training/benchmark path is `sim.eye.compute_eye` (v1), which is
frozen and takes no noise argument, and the policy observation (18 dims) carries no
noise or SNR term. The policy therefore CANNOT see link noise, and the design it emits
for a given (target, channel) is bit-identical at every SNR. That is a property of the
architecture, not of this experiment.

Consequently the following are SNR-INVARIANT BY CONSTRUCTION and are reported once,
outside the per-SNR table, rather than repeated as six identical rows dressed up as a
robustness result:

    achieved boost, requested target boost, absolute boost error, peak frequency,
    valid-design count, simulator failure count, evaluation count, reward/score,
    and the frozen v1 strict-pass rate.

The following genuinely move with SNR, because they are re-measured under injected
noise, and they are what the per-SNR table contains:

    BER (empirical and semi-analytic), eye height, eye width, and a strict/loose
    pass rate recomputed on the noise-degraded eye.

MEASUREMENT-PATH CAVEAT
-----------------------
Noise is only available in `compute_eye_v2`, which is a different measurement contract
from the frozen v1 scorer (explicit finite causal FIR, linear convolution, full-support
startup exclusion). A v2 pass rate is therefore NOT numerically comparable to the frozen
v1 benchmark number. This script reports the noiseless v2 point as the reference row of
its own curve, and prints the frozen v1 rate separately as the anchor. Comparing a v2
row against the v1 anchor is an error; comparing v2 rows against each other is the
experiment.

NOISE MODEL
-----------
Zero-mean additive white Gaussian noise on the slicer decision samples, injected inside
`compute_eye_v2` (`sim/eye.py`, `_decision_feedback_v2`), post-channel and post-CTLE,
before decision feedback. Per the requested definition:

    sigma_v = signal_rms / 10 ** (SNR_dB / 20)

with `signal_rms` taken as the absolute main-cursor voltage |c0| of the same design's
noiseless response, which is the amplitude the slicer actually decides against. This is
the same referral `experiments/snr_ber_sweep.py` uses, so the two studies are on one
scale. It is a flat-spectrum slicer-referred model: it contains no measured noise PSD,
no channel thermal noise, and no jitter or clock-recovery effects.

Every draw is reproducible: the eye seed is explicit and per (spec, SNR) point, and the
noise stream is a `SeedSequence` child inside `compute_eye_v2` that is independent of
the phase-selection and DFE-adaptation streams.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

P = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{P/'shim'};{P/'Library'/'bin'};{os.environ['PATH']}"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse  # noqa: E402
import csv  # noqa: E402

import numpy as np  # noqa: E402

from eqrl.circuits.ctle import DesignVars, decode_action  # noqa: E402
from eqrl.sim.eye import EyeResultV2, compute_eye_v2  # noqa: E402
from eqrl.specs import DEFAULT_SPEC, hard_pass  # noqa: E402

# `eqrl.envs` pulls in gymnasium and `eqrl.sim.server` opens ngspice; both are imported
# inside the functions that need them so the noise mathematics in this module stays
# importable (and testable) without a simulator or an RL stack present.

#: The six evaluation points the study is specified at. Not a training curriculum.
SNR_POINTS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

#: Default canonical policy. `results/delivered_circuit.json` names this checkpoint as
#: the one the delivered circuit was produced from.
DEFAULT_MODEL = "results/seq_clean40k.zip"

#: Held-out spec seed. Identical to `experiments/policy_rollout.py` so the rollout half
#: of this study lands on the same specs as the existing rollout artifact.
SPEC_SEED = 0
TARGET_RANGE = (5.0, 11.0)
CHANNEL_RANGE = (8.0, 16.0)

#: Base seed for the eye/noise draws. The per-point seed is derived from this and the
#: spec index so that every (spec, SNR) cell is independently reproducible while the
#: noiseless reference for a spec shares the spec's stream.
DEFAULT_EYE_SEED = 20260909

#: 8192 bits gives ~8.1e3 scored symbols, so the smallest resolvable non-zero empirical
#: BER is ~1.2e-4. Above roughly 12 dB the empirical BER floors at exactly zero and the
#: semi-analytic column is the only one carrying information. That is a resolution
#: limit, not a measurement of an error-free link, and it is recorded as such.
DEFAULT_N_BITS = 8192

DEFAULT_OUT = "results/policy_snr_sweep_v1.json"
SCHEMA = "eqrl.policy_snr_sweep.v1"


# ---------------------------------------------------------------------------
# noise helpers


def sigma_for_snr(signal_rms_v: float, snr_db: float) -> float:
    """sigma_v = signal_rms / 10 ** (SNR_dB / 20). The requested definition, verbatim."""
    if not np.isfinite(signal_rms_v) or signal_rms_v <= 0.0:
        raise ValueError("signal_rms_v must be positive and finite")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    return float(signal_rms_v) / (10.0 ** (float(snr_db) / 20.0))


def _q(x: np.ndarray) -> np.ndarray:
    """Gaussian tail Q(x) = 0.5 * erfc(x / sqrt(2)), vectorised without scipy."""
    return 0.5 * np.array([math.erfc(v / math.sqrt(2.0)) for v in np.atleast_1d(x)])


def semi_analytic_ber(reference: EyeResultV2, sigma_v: float) -> float:
    """Correct-feedback BER from the NOISELESS sampled-margin distribution.

    Assumes every fed-back decision was right, so it is optimistic exactly where error
    propagation matters (low SNR) and is the trustworthy column exactly where the
    empirical count runs out of symbols (high SNR).
    """
    if sigma_v <= 0.0:
        return 0.0
    return float(np.mean(_q(reference.margins_v / sigma_v)))


def measured_snr_db(reference: EyeResultV2, noisy: EyeResultV2,
                    signal_rms_v: float) -> tuple[float | None, float | None, int]:
    """Empirically recover the injected sigma, to verify requested vs delivered SNR.

    `eye_matrix` is the polarity-normalised, DFE-corrected sample matrix for the scored
    rows, and the two runs share a seed, so they share their bit pattern, sampling phase
    and DFE tap. Their difference is therefore the injected noise -- EXCEPT on rows whose
    preceding decision disagreed between the runs, where the one-tap feedback subtracts a
    different value and contaminates the residual. Those rows are dropped: a row is kept
    only when neither run made an error on the row before it. The first scored row is
    dropped too, because its predecessor lies outside the scored window.

    Returns (measured_snr_db, measured_sigma_v, n_samples_used).
    """
    if reference.eye_matrix.shape != noisy.eye_matrix.shape:
        raise ValueError("reference and noisy runs disagree on scored shape")
    clean_row = ~((reference.margins_v < 0.0) | (noisy.margins_v < 0.0))
    keep = np.zeros(clean_row.size, dtype=bool)
    keep[1:] = clean_row[:-1]
    n = int(keep.sum()) * int(reference.eye_matrix.shape[1])
    if keep.sum() < 32:
        return None, None, n
    residual = noisy.eye_matrix[keep] - reference.eye_matrix[keep]
    sigma = float(np.std(residual))
    if sigma <= 0.0:
        return None, 0.0, n
    return float(20.0 * math.log10(signal_rms_v / sigma)), sigma, n


# ---------------------------------------------------------------------------
# statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarise(values: list[float]) -> dict:
    """mean / median / std / n / 95% CI of the mean for a continuous metric."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "ci95_lo": None, "ci95_hi": None}
    mean = float(a.mean())
    sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
    return {"n": int(a.size), "mean": mean, "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "ci95_lo": mean - 1.96 * sem, "ci95_hi": mean + 1.96 * sem}


def spearman(x: list[float], y: list[float]) -> float | None:
    """Rank correlation without scipy. Ties get average ranks."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        # average ranks within tie groups
        for val in np.unique(v):
            idx = np.flatnonzero(v == val)
            if idx.size > 1:
                r[idx] = r[idx].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return None if denom == 0.0 else float((ra * rb).sum() / denom)


# ---------------------------------------------------------------------------
# stage A: SNR-independent policy rollouts


def held_out_specs(n: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(SPEC_SEED)
    return [(float(rng.uniform(*TARGET_RANGE)), float(rng.uniform(*CHANNEL_RANGE)))
            for _ in range(n)]


def run_rollouts(model, env, specs, guard_for, boost_tol) -> list[dict]:
    """Deterministic rollout per spec. Identical in structure to policy_rollout.main.

    Returns one record per spec carrying the delivered design and every SNR-invariant
    measurement. `delivered` is the design at first success when the policy solves, and
    the final-step design when it does not -- what the policy would actually ship.
    """
    rows = []
    for i, (target, channel) in enumerate(specs):
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel,
                                   boost_target_tol_db=boost_tol)

        def evaluate(x):
            dv = decode_action(np.asarray(x))
            try:
                v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
            except Exception as exc:                       # simulator failure
                return dv, None, False, repr(exc)[:120]
            if not v.is_valid:
                return dv, None, False, str(getattr(v.check, "value", v.check))
            return dv, v.unwrap(), True, None

        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))

        used, n_valid, n_failed = 1, 0, 0
        solved_at, delivered = None, None
        dv, m, valid, why = evaluate(env._x)
        n_valid += int(valid)
        n_failed += int(not valid)
        best = (dv, m) if valid else None
        if valid and hard_pass(m, spec)[0]:
            solved_at, delivered = used, (dv, m)

        while solved_at is None and used < env.horizon:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            used += 1
            dv, m, valid, why = evaluate(env._x)
            n_valid += int(valid)
            n_failed += int(not valid)
            if valid:
                best = (dv, m)
                if hard_pass(m, spec)[0]:
                    solved_at, delivered = used, (dv, m)
            if term or trunc:
                break

        if delivered is None:
            delivered = best                                # last valid design
        strict_ok, checks = (hard_pass(delivered[1], spec) if delivered else (False, {}))

        rec = {
            "index": i,
            "target_boost_db": target,
            "channel_loss_db": channel,
            "evaluations": used,
            "n_valid": n_valid,
            "n_sim_failed": n_failed,
            "solved_at": solved_at,
            "delivered_from": ("first_success" if solved_at else
                               ("last_valid" if delivered else None)),
            "has_design": delivered is not None,
            "last_invalid_reason": why if delivered is None else None,
        }
        if delivered is not None:
            dv, m = delivered
            rec.update({
                "design": dataclasses.asdict(dv),
                "boost_db": float(m.boost_db),
                "abs_boost_err_db": float(abs(m.boost_db - target)),
                "peak_freq_ghz": float(m.peak_freq_ghz),
                "dc_gain_db": float(m.dc_gain_db),
                "noise_vrms": float(m.noise_vrms),
                "power_w": float(m.power_w),
                "eye_h_ui_v1": float(m.eye_h_ui),
                "eye_v_mv_v1": float(m.eye_v_mv),
                "strict_pass_v1": bool(strict_ok),
                "failing_v1": [c for c, good in checks.items() if not good],
                #: Every `hard_pass` check EXCEPT the two eye checks. Those two are the
                #: only ones noise can move, so the per-SNR strict rate is this flag
                #: AND-ed with the v2 eye measured at that SNR -- one definition of the
                #: eye per curve, never a v1 eye check silently AND-ed with a v2 one.
                "non_eye_checks_pass": all(v for c, v in checks.items()
                                           if c not in ("eye_h", "eye_v")),
            })
        rows.append(rec)
        state = f"solved@{solved_at}" if solved_at else "unsolved"
        boost = "no valid design" if delivered is None else f"boost {rec['boost_db']:5.2f} dB"
        print(f"  spec {i:2d}: target {target:5.2f} dB  channel {channel:5.2f} dB  ->  "
              f"{state:12s} {boost}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# stage B: per-design SNR evaluation


def sweep_design(dv, *, channel_loss_db: float, corner: str, vdd: float, temp_c: float,
                 n_bits: int, eye_seed: int, snr_points) -> dict:
    """Measure one delivered design at every SNR point, plus a noiseless reference."""
    from eqrl.sim.server import get_server

    srv = get_server(corner)
    srv.set_corner(corner)
    response = srv.ac_complex(dv, vdd=vdd, temp_c=temp_c)
    freq, H = response["freq"], response["H"]

    common = dict(channel_loss_db=channel_loss_db, n_bits=n_bits, seed=eye_seed)
    reference = compute_eye_v2(freq, H, noise_sigma_v=0.0, **common)
    signal_rms_v = float(abs(reference.c0))
    if not np.isfinite(signal_rms_v) or signal_rms_v <= 0.0:
        raise ValueError("main-cursor amplitude is not positive; cannot referr noise")

    points = []
    for snr_db in snr_points:
        sigma_v = sigma_for_snr(signal_rms_v, snr_db)
        res = compute_eye_v2(freq, H, noise_sigma_v=sigma_v, **common)
        meas_db, meas_sigma, n_used = measured_snr_db(reference, res, signal_rms_v)
        points.append({
            "snr_db": float(snr_db),
            "sigma_v": sigma_v,
            "measured_snr_db": meas_db,
            "measured_sigma_v": meas_sigma,
            "measured_n_samples": n_used,
            "snr_error_db": None if meas_db is None else float(meas_db - snr_db),
            "ber_empirical": float(res.ber),
            "ber_errors": int(res.errors),
            "ber_symbols": int(res.count),
            "ber_semi_analytic": semi_analytic_ber(reference, sigma_v),
            "eye_h_ui": float(res.width_ui),
            "eye_v_mv": float(res.height_v * 1e3),
            "signed_opening_mv": float(res.signed_opening_v * 1e3),
            "dfe_tap_v": float(res.dfe_tap),
        })

    return {
        "signal_rms_v": signal_rms_v,
        "eye_seed": eye_seed,
        "n_bits": n_bits,
        "reference_noiseless": {
            "eye_h_ui": float(reference.width_ui),
            "eye_v_mv": float(reference.height_v * 1e3),
            "ber_empirical": float(reference.ber),
            "c0_v": float(reference.c0),
            "dfe_tap_v": float(reference.dfe_tap),
        },
        "points": points,
    }


# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluation-only SNR robustness sweep over a trained policy. "
                    "An extension experiment; not part of the Astera problem statement.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"policy checkpoint (default: {DEFAULT_MODEL})")
    p.add_argument("--specs", type=int, default=16,
                   help="held-out spec count (default: 16)")
    p.add_argument("--snr-db", type=float, nargs="+", default=list(SNR_POINTS_DB),
                   help="SNR evaluation points in dB (default: -5 0 5 10 15 20)")
    p.add_argument("--n-bits", type=int, default=DEFAULT_N_BITS,
                   help=f"symbols per eye measurement (default: {DEFAULT_N_BITS})")
    p.add_argument("--eye-seed", type=int, default=DEFAULT_EYE_SEED,
                   help=f"base seed for eye/noise draws (default: {DEFAULT_EYE_SEED})")
    p.add_argument("--corner", default="tt")
    p.add_argument("--vdd", type=float, default=1.8)
    p.add_argument("--temp-c", type=float, default=27.0)
    p.add_argument("--boost-tol", type=float, default=None,
                   help="dB tolerance on the requested target; adds the target check to "
                        "strict pass (default: off, matching policy_rollout)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--csv", default=None, help="default: --out with a .csv suffix")
    p.add_argument("--plot", default=None, help="default: --out with a .png suffix")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-only", action="store_true",
                   help="re-aggregate, re-plot and re-print from the per-spec records "
                        "already in --out; runs no simulation and loads no checkpoint")
    return p


def main() -> None:
    args = _parser().parse_args()
    out = Path(args.out)
    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")

    if args.report_only:
        prior = json.loads(out.read_text(encoding="utf-8"))
        if prior.get("schema") != SCHEMA:
            raise SystemExit(f"{out} is not a {SCHEMA} artifact")
        # the measurements are whatever the run recorded; only the derived layer is
        # rebuilt, and it is rebuilt against that run's own settings, not this CLI's
        for name in ("eye_seed", "n_bits", "corner", "vdd", "temp_c"):
            setattr(args, name, prior.get(name if name != "eye_seed" else "eye_seed_base",
                                          getattr(args, name)))
        args.model, args.boost_tol = prior["model"], prior["boost_tol_db"]
        payload = _aggregate(prior["per_spec"], prior["snr_points_db"], args)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_csv(csv_path, prior["per_spec"])
        _plot(payload, plot_path)
        _report(payload)
        print(f"\nrewrote {out}\nrewrote {csv_path}\nrewrote {plot_path}")
        return

    for path in (out, csv_path, plot_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace it")

    snr_points = [float(v) for v in args.snr_db]
    if len(set(snr_points)) != len(snr_points):
        raise SystemExit("--snr-db must not repeat a point")

    from stable_baselines3 import PPO

    from eqrl.envs.sequential_env import SequentialEqualizerEnv
    from eqrl.evaluator import build_evaluator

    model = PPO.load(args.model)

    specs = held_out_specs(args.specs)
    env = SequentialEqualizerEnv(fast=False, seed=123, boost_tol=args.boost_tol)

    guards: dict[float, object] = {}

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner=args.corner,
                                              fast=False, channel_loss_db=channel)
        return guards[channel]

    print(f"model    : {args.model}")
    print(f"specs    : {len(specs)} held-out (seed {SPEC_SEED})")
    print(f"snr pts  : {snr_points} dB")
    print("stage A  : rollouts (SNR-independent by construction)\n")
    rollouts = run_rollouts(model, env, specs, guard_for, args.boost_tol)

    print("\nstage B  : per-design SNR evaluation")
    per_spec = []
    for rec in rollouts:
        if not rec["has_design"]:
            print(f"  spec {rec['index']:2d}: no valid design; skipped", flush=True)
            per_spec.append({**rec, "snr": None})
            continue
        dv = DesignVars(**rec["design"])
        sweep = sweep_design(dv, channel_loss_db=rec["channel_loss_db"],
                             corner=args.corner, vdd=args.vdd, temp_c=args.temp_c,
                             n_bits=args.n_bits,
                             eye_seed=args.eye_seed + 1000 * rec["index"],
                             snr_points=snr_points)
        per_spec.append({**rec, "snr": sweep})
        print(f"  spec {rec['index']:2d}: |c0| {sweep['signal_rms_v']*1e3:6.2f} mV  "
              f"BER @-5dB {sweep['points'][0]['ber_empirical']:.3e}  "
              f"@20dB {sweep['points'][-1]['ber_empirical']:.3e}", flush=True)

    payload = _aggregate(per_spec, snr_points, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, per_spec)
    _plot(payload, plot_path)
    _report(payload)
    print(f"\nwrote {out}\nwrote {csv_path}\nwrote {plot_path}")


def _aggregate(per_spec: list[dict], snr_points: list[float], args) -> dict:
    scored = [r for r in per_spec if r.get("snr")]
    invariant = {
        "note": "SNR-invariant by construction: the frozen scorer is noiseless and the "
                "18-dim observation carries no noise term, so the policy emits the same "
                "design at every SNR. Reported once, not repeated per SNR point.",
        "n_specs": len(per_spec),
        "n_with_design": len(scored),
        "strict_pass_v1": wilson(sum(1 for r in scored if r.get("strict_pass_v1")),
                                 len(per_spec))[0],
        "solve_rate": wilson(sum(1 for r in per_spec if r.get("solved_at")),
                             len(per_spec))[0],
        "abs_boost_err_db": summarise([r["abs_boost_err_db"] for r in scored]),
        "boost_db": summarise([r["boost_db"] for r in scored]),
        "evaluations": summarise([float(r["evaluations"]) for r in per_spec]),
        "n_valid": summarise([float(r["n_valid"]) for r in per_spec]),
        "n_sim_failed": summarise([float(r["n_sim_failed"]) for r in per_spec]),
        "eye_h_ui_v1": summarise([r["eye_h_ui_v1"] for r in scored]),
        "eye_v_mv_v1": summarise([r["eye_v_mv_v1"] for r in scored]),
    }

    def rates(pts: list[dict]) -> dict:
        """Strict/loose rate over the same denominator, given one eye per spec.

        `strict` is every non-eye `hard_pass` check from the design's own frozen
        measurement AND-ed with the eye supplied here; `loose` drops everything but the
        boost range and that eye. Only the eye differs between rows, so a rate change
        across the curve is attributable to the eye and nothing else.
        """
        strict, loose = 0, 0
        for r, pt in zip(scored, pts):
            spec = dataclasses.replace(DEFAULT_SPEC,
                                       target_boost_db=r["target_boost_db"],
                                       channel_loss_db=r["channel_loss_db"],
                                       boost_target_tol_db=args.boost_tol)
            ok_eye = (pt["eye_h_ui"] >= spec.eye_h_ui_min
                      and pt["eye_v_mv"] >= spec.eye_v_mv_min)
            strict += int(bool(r["non_eye_checks_pass"]) and ok_eye)
            loose += int((spec.boost_db_min <= r["boost_db"] <= spec.boost_db_max)
                         and ok_eye)
        sp, sp_lo, sp_hi = wilson(strict, len(per_spec))
        lp, lp_lo, lp_hi = wilson(loose, len(per_spec))
        return {"strict_pass_rate": sp, "strict_ci95": [sp_lo, sp_hi],
                "strict_k": strict, "loose_pass_rate": lp,
                "loose_ci95": [lp_lo, lp_hi], "loose_k": loose, "n": len(per_spec)}

    #: The zero-noise v2 anchor. Any gap between this and the frozen v1 rate is the
    #: v1->v2 measurement-contract change, NOT noise; subtracting it is the only way to
    #: read the curve's high-SNR end honestly.
    ref_pts = [r["snr"]["reference_noiseless"] for r in scored]
    noiseless_v2 = {
        "note": "zero-noise v2 reference. The difference between this rate and the "
                "frozen v1 strict rate is the measurement-contract change; the "
                "difference between this and a per-SNR row is the noise.",
        **rates(ref_pts),
        "eye_h_ui": summarise([p["eye_h_ui"] for p in ref_pts]),
        "eye_v_mv": summarise([p["eye_v_mv"] for p in ref_pts]),
        "ber_empirical": summarise([p["ber_empirical"] for p in ref_pts]),
    }

    curve = []
    for j, snr_db in enumerate(snr_points):
        pts = [r["snr"]["points"][j] for r in scored]
        curve.append({
            "snr_db": snr_db,
            **rates(pts),
            "ber_empirical": summarise([p["ber_empirical"] for p in pts]),
            "ber_semi_analytic": summarise([p["ber_semi_analytic"] for p in pts]),
            "eye_h_ui": summarise([p["eye_h_ui"] for p in pts]),
            "eye_v_mv": summarise([p["eye_v_mv"] for p in pts]),
            "measured_snr_db": summarise([p["measured_snr_db"] for p in pts]),
            "snr_error_db": summarise([p["snr_error_db"] for p in pts]),
            "corr_boost_vs_ber": spearman([r["boost_db"] for r in scored],
                                          [p["ber_semi_analytic"] for p in pts]),
            "corr_boost_vs_eye_v": spearman([r["boost_db"] for r in scored],
                                            [p["eye_v_mv"] for p in pts]),
            "corr_abs_err_vs_ber": spearman([r["abs_boost_err_db"] for r in scored],
                                            [p["ber_semi_analytic"] for p in pts]),
        })

    return {
        "schema": SCHEMA,
        "reachability": _reachability(scored),
        "what": "Evaluation-only SNR robustness sweep of a trained PPO policy. An "
                "EXTENSION experiment; the Astera Labs problem statement specifies no "
                "link SNR and no BER target, and no frozen artifact is affected.",
        "model": args.model,
        "spec_seed": SPEC_SEED,
        "target_range_db": list(TARGET_RANGE),
        "channel_range_db": list(CHANNEL_RANGE),
        "eye_seed_base": args.eye_seed,
        "n_bits": args.n_bits,
        "corner": args.corner, "vdd": args.vdd, "temp_c": args.temp_c,
        "boost_tol_db": args.boost_tol,
        "snr_points_db": snr_points,
        "noise_model": {
            "equation": "sigma_v = signal_rms / 10 ** (SNR_dB / 20)",
            "signal_rms": "absolute main-cursor voltage |c0| of the noiseless response",
            "distribution": "zero-mean white Gaussian, iid per sample",
            "injection_point": "sim/eye.py:_decision_feedback_v2 -- slicer decision "
                               "samples, post-channel, post-CTLE, pre-decision-feedback",
            "measurement_engine": "compute_eye_v2",
            "limitations": [
                "flat-spectrum slicer-referred noise; no measured noise PSD",
                "no channel thermal noise and no transmitter noise",
                "no jitter, package or clock-recovery modelling",
                "empirical BER floors at zero once it falls below ~1/scored-symbols; "
                "the semi-analytic column carries the high-SNR information",
                "v2 eye values are not numerically comparable to the frozen v1 scorer",
            ],
        },
        "snr_invariant": invariant,
        "noiseless_v2_reference": noiseless_v2,
        "curve": curve,
        "per_spec": per_spec,
    }


def _reachability(scored: list[dict]) -> dict:
    """Is a low-SNR failure a POLICY failure, or is the check unreachable by any design?

    `sim.eye._truth_opening` returns `positive.min() - negative.max()`: a WORST-CASE
    opening over every scored symbol, not an average. Under iid Gaussian noise both
    extremes walk outward, so the opening shrinks by roughly `k * sigma` where `k` is
    twice the expected extreme of ~N/2 standard normals -- about 7 at this symbol count.
    The measured `k_fit` per SNR point checks that claim against the data rather than
    assuming it.

    The consequence is a hard ceiling. `opening_0` cannot exceed `2 * |c0|` (zero ISI,
    every symbol landing exactly on its cursor), so the `eye_v_mv_min` check is
    unsatisfiable BY ANY DESIGN once `2 * |c0| - k * sigma < eye_v_mv_min`. Below that
    SNR no amount of retraining can lift the pass rate, because the failure is the
    metric's definition meeting Gaussian tails, not a policy that chose badly.
    """
    k_assumed = 7.0
    floor_mv = DEFAULT_SPEC.eye_v_mv_min
    ratios, thresholds, k_fit = [], [], {}
    for r in scored:
        c0_mv = r["snr"]["signal_rms_v"] * 1e3
        op0 = r["snr"]["reference_noiseless"]["eye_v_mv"]
        ratios.append(op0 / c0_mv)
        if op0 > floor_mv:
            thresholds.append(20.0 * math.log10(k_assumed * c0_mv / (op0 - floor_mv)))
        for pt in r["snr"]["points"]:
            sigma_mv = pt["sigma_v"] * 1e3
            k_fit.setdefault(pt["snr_db"], []).append(
                (op0 - pt["signed_opening_mv"]) / sigma_mv)

    def ceiling(open_ratio: float, c0_mv: float) -> float | None:
        num = open_ratio * c0_mv - floor_mv
        return None if num <= 0 else 20.0 * math.log10(k_assumed * c0_mv / num)

    c0_typ = float(np.median([r["snr"]["signal_rms_v"] * 1e3 for r in scored]))
    return {
        "note": "the eye opening is a worst-case (min-max) statistic, so it shrinks by "
                "~k*sigma; below the ceiling SNR the eye_v check is unsatisfiable by "
                "ANY design and a low pass rate is not a policy failure",
        "k_assumed": k_assumed,
        "k_fit_by_snr_db": {str(s): summarise(v)["mean"] for s, v in k_fit.items()},
        "eye_v_mv_min": floor_mv,
        "opening_over_c0": summarise(ratios),
        "opening_over_c0_physical_max": 2.0,
        "median_c0_mv": c0_typ,
        "policy_closure_snr_db": summarise(thresholds),
        "best_case_closure_snr_db": ceiling(2.0, c0_typ),
        "best_observed_closure_snr_db": ceiling(max(ratios), c0_typ),
    }


def _write_csv(path: Path, per_spec: list[dict]) -> None:
    cols = ["index", "target_boost_db", "channel_loss_db", "boost_db",
            "abs_boost_err_db", "solved_at", "evaluations", "n_valid", "n_sim_failed",
            "strict_pass_v1", "eye_h_ui_v1", "eye_v_mv_v1", "signal_rms_v", "snr_db",
            "sigma_v", "measured_snr_db", "snr_error_db", "ber_empirical",
            "ber_semi_analytic", "eye_h_ui", "eye_v_mv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in per_spec:
            base = {k: r.get(k) for k in cols if k in r}
            if not r.get("snr"):
                w.writerow(base)
                continue
            base["signal_rms_v"] = r["snr"]["signal_rms_v"]
            for pt in r["snr"]["points"]:
                w.writerow({**base, **{k: pt.get(k) for k in
                                       ("snr_db", "sigma_v", "measured_snr_db",
                                        "snr_error_db", "ber_empirical",
                                        "ber_semi_analytic", "eye_h_ui", "eye_v_mv")}})


def _plot(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = payload["curve"]
    x = [c["snr_db"] for c in curve]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0][0]
    a.errorbar(x, [c["strict_pass_rate"] for c in curve],
               yerr=[[c["strict_pass_rate"] - c["strict_ci95"][0] for c in curve],
                     [c["strict_ci95"][1] - c["strict_pass_rate"] for c in curve]],
               marker="o", capsize=3, label="strict (all checks)")
    a.plot(x, [c["loose_pass_rate"] for c in curve], marker="s", ls="--",
           label="loose (boost range + eye)")
    ref = payload["noiseless_v2_reference"]
    a.axhline(ref["strict_pass_rate"], color="seagreen", ls=":",
              label=f"v2 noiseless anchor {ref['strict_pass_rate']:.2f}")
    a.axhline(payload["snr_invariant"]["strict_pass_v1"], color="grey", ls="-.",
              label=f"frozen v1 rate {payload['snr_invariant']['strict_pass_v1']:.2f} "
                    f"(different contract)")
    a.set_ylim(-0.05, 1.05)
    a.set_ylabel("pass rate")
    a.set_title("Pass rate vs SNR (eye re-measured under noise)")
    a.legend(fontsize=8)

    a = ax[0][1]
    emp = [c["ber_empirical"]["mean"] for c in curve]
    sem = [c["ber_semi_analytic"]["mean"] for c in curve]
    a.semilogy(x, [max(v, 1e-12) if v is not None else np.nan for v in emp],
               marker="o", label="empirical (floors at 0)")
    a.semilogy(x, [max(v, 1e-30) if v is not None else np.nan for v in sem],
               marker="^", ls="--", label="semi-analytic")
    a.set_ylabel("mean BER")
    a.set_title("BER vs SNR")
    a.legend(fontsize=8)

    a = ax[1][0]
    a.errorbar(x, [c["eye_v_mv"]["mean"] for c in curve],
               yerr=[[c["eye_v_mv"]["mean"] - c["eye_v_mv"]["ci95_lo"] for c in curve],
                     [c["eye_v_mv"]["ci95_hi"] - c["eye_v_mv"]["mean"] for c in curve]],
               marker="o", capsize=3, label="eye height (mV)")
    a.axhline(DEFAULT_SPEC.eye_v_mv_min, color="crimson", ls=":",
              label=f"spec min {DEFAULT_SPEC.eye_v_mv_min:g} mV")
    a.set_xlabel("requested SNR (dB)")
    a.set_ylabel("eye height (mV)")
    a.set_title("Eye height vs SNR")
    a.legend(fontsize=8)

    a = ax[1][1]
    inv = payload["snr_invariant"]["abs_boost_err_db"]["mean"]
    a.plot(x, [inv] * len(x), marker="o", color="grey")
    a.set_xlabel("requested SNR (dB)")
    a.set_ylabel("|boost error| (dB)")
    a.set_title("Absolute boost error vs SNR\n(FLAT: SNR-invariant by construction)",
                fontsize=10)
    a.set_ylim(0, max(0.5, (inv or 0) * 2.5))

    for row in ax:
        for a in row:
            a.grid(alpha=0.3)
    fig.suptitle("Evaluation-only SNR robustness sweep -- extension experiment, "
                 "not part of the Astera problem statement", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _report(payload: dict) -> None:
    inv = payload["snr_invariant"]
    print("\nSNR-INVARIANT BY CONSTRUCTION (reported once)")
    print(f"  specs                     : {inv['n_specs']}  "
          f"({inv['n_with_design']} produced a valid design)")
    print(f"  solve rate                : {inv['solve_rate']:.3f}")
    print(f"  strict pass (frozen v1)   : {inv['strict_pass_v1']:.3f}")
    print(f"  mean |boost error|        : {inv['abs_boost_err_db']['mean']:.3f} dB")
    print(f"  mean evaluations          : {inv['evaluations']['mean']:.1f}")
    print(f"  mean simulator failures   : {inv['n_sim_failed']['mean']:.2f}")
    ref = payload["noiseless_v2_reference"]
    print(f"  strict pass (v2, no noise) : {ref['strict_pass_rate']:.3f}  "
          f"<- the curve's own anchor; its gap to the v1 rate above is the "
          f"measurement-contract change, not noise")
    print("\nPER-SNR (eye re-measured under injected noise; v2 contract)")
    print(f"  {'SNR':>6} {'meas':>7} {'err':>6} {'strict':>7} {'loose':>7} "
          f"{'BER emp':>10} {'BER s-a':>10} {'eye mV':>8} {'eye UI':>7}")
    for c in payload["curve"]:
        ms = c["measured_snr_db"]["mean"]
        er = c["snr_error_db"]["mean"]
        print(f"  {c['snr_db']:6.1f} {('%7.2f' % ms) if ms is not None else '      -'} "
              f"{('%6.2f' % er) if er is not None else '     -'} "
              f"{c['strict_pass_rate']:7.3f} {c['loose_pass_rate']:7.3f} "
              f"{c['ber_empirical']['mean']:10.3e} {c['ber_semi_analytic']['mean']:10.3e} "
              f"{c['eye_v_mv']['mean']:8.2f} {c['eye_h_ui']['mean']:7.3f}")

    rch = payload["reachability"]
    print("\nIS A LOW-SNR FAILURE FIXABLE BY RETRAINING?")
    print(f"  eye opening is min-max, so it loses ~k*sigma; k fitted = "
          f"{ {s: round(v, 2) for s, v in rch['k_fit_by_snr_db'].items()} }")
    print(f"  delivered opening / |c0|   : {rch['opening_over_c0']['mean']:.3f} "
          f"(physical max {rch['opening_over_c0_physical_max']:.1f})")
    print(f"  this policy closes below   : "
          f"{rch['policy_closure_snr_db']['mean']:.2f} dB "
          f"(range {rch['policy_closure_snr_db']['ci95_lo']:.2f}"
          f"-{rch['policy_closure_snr_db']['ci95_hi']:.2f})")
    print(f"  ANY design closes below    : {rch['best_case_closure_snr_db']:.2f} dB "
          f"<- below this the eye_v check is unsatisfiable, so the pass rate is a "
          f"property of the metric, not of the policy")


if __name__ == "__main__":
    main()
