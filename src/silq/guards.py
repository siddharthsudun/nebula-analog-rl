"""guards.py — the validation layer between raw SPICE output and every downstream metric.

THE THREAT MODEL
----------------
A broken simulation that still returns *plausible* numbers is worse than one that
crashes. The agent optimizes toward the artifact, the reward curve looks healthy, and
the defect surfaces weeks later at characterization. Every check here exists to turn a
silent wrong answer into a loud, named failure.

CONTRACT
--------
Every evaluation returns exactly one of:
    Valid(metrics=..., run_id=..., artifact_dir=...)
    Invalid(check=..., reason=..., run_id=..., artifact_dir=..., violation=...)  # NO .metrics

`Invalid` deliberately has no `metrics` attribute. A caller that forgets to branch gets
an AttributeError, not a plausible zero. There is no third state and no default.

TIERS
-----
1  run integrity      — exit code, stderr text, output file, transient completeness
2  circuit sanity     — .op: saturation headroom, tail current, rails, PDK bounds
3  corner integrity   — proves the corner .lib actually changed the models
4  physical plausibility — numbers that mean the *measurement* broke
5  search pathology   — log-level; these HALT the run

Tiers 1–4 mark a single candidate INVALID. Tier 5 raises SearchHalted and writes
HALT.txt — it is a statement about the experiment, not about one design.

THRESHOLDS ARE NOT TUNABLE FROM HERE
------------------------------------
Every bound below is a named module constant. They are set by the project owners.
If a check fires, that is a finding to report — not a number to widen.

VIOLATION MAGNITUDE IS NOT A THRESHOLD
--------------------------------------
`Invalid.violation` reports HOW FAR outside the bound a rejected design sits. It is
reported, never consulted: no check reads it, and no verdict changes because of it. It
exists so a caller that has to *score* a rejection can tell a device missing saturation
by 10 mV from one demanding 50 V across a 1.8 V supply. See the field's own docstring.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import uuid
from abc import ABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from silq.circuits.ctle import ACTION_SPACE, DesignVars
from silq.specs import Spec

# =============================================================================
# THRESHOLDS — owned by the project, not by this module's callers.
# =============================================================================

# -- Tier 2
SATURATION_HEADROOM_V = 0.050        # Vds must exceed Vdsat by at least this
TAIL_CURRENT_TOLERANCE = 0.10        # actual tail current within 10% of requested
RAIL_MARGIN_V = 0.0                  # node voltages must lie within [0 - m, vdd + m]

# SKY130 sky130_fd_pr__nfet_01v8 geometry bounds (metres).
PDK_BOUNDS: dict[str, tuple[float, float]] = {
    "w_in": (0.42e-6, 100e-6),
    "l_in": (0.15e-6, 100e-6),
}
PDK_BOUND_RELTOL = 1e-9              # "exactly at" means within this relative distance

# -- Tier 3
CORNER_PROBE_MIN_DELTA = 1e-4        # tt vs ss probe must differ by at least this (V)

# -- Tier 4
DC_GAIN_DB_MAX = 60.0
DC_GAIN_DB_MIN = 0.0                 # below this, with a topology that should have gain
PEAKING_DB_MAX = 20.0
POWER_W_MIN = 0.1e-3
POWER_W_MAX = 100e-3
NOISE_VRMS_MIN = 0.01e-3             # 0.01 mVrms — below this the .noise parse broke
HD3_DB_FLOOR = -70.0                 # "better" than this is not a real measurement
EYE_H_UI_MAX = 1.0                   # one unit interval, by definition

# -- Tier 5
EDGE_PIN_FRACTION = 0.02             # within 2% of a range edge counts as pinned
EDGE_PIN_MAX_CONSECUTIVE = 20
DUPLICATE_METRIC_DECIMALS = 6
TREND_WINDOW = 50
INVALID_WINDOW = 100
INVALID_RATE_MAX = 0.30
INVALID_RATE_MIN = 0.01
SPEC_BEAT_FRACTION = 0.30            # beating a target by >30%
SPEC_BEAT_COUNT = 3                  # ...on 3+ targets simultaneously

# stderr / stdout text that means the solver gave up.
_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"no\s+convergence", "no convergence"),
    (r"convergence\s+(failure|problem|not\s+achieved)", "convergence failure"),
    (r"timestep\s+too\s+small", "timestep too small"),
    (r"singular\s+matrix", "singular matrix"),
    (r"matrix\s+is\s+singular", "singular matrix"),
    (r"iteration\s+limit\s+reached", "iteration limit reached"),
    (r"doAnalyses:\s*(TRAN|DC|AC)", "analysis aborted"),
    (r"fatal\s+error", "fatal error"),
    (r"can'?t\s+find\s+(model|subcircuit)", "missing model/subcircuit"),
    (r"unknown\s+subckt", "missing subcircuit"),
    (r"could\s+not\s+find\s+include\s+file", "missing include file"),
    # Device geometry outside what the PDK model is binned for. sky130's nfet_01v8
    # is characterised to W <= 100 um per device; W = 101 matches no bin and ngspice
    # reports exactly this, on stderr, with no exit code at all on the resident path.
    # Measured against ngspice 41 — without it a device the PDK refuses to model
    # reads as a clean run that simply produced no numbers.
    (r"could\s+not\s+find\s+a\s+valid\s+model", "device geometry outside PDK model bins"),
    (r"simulation\s+interrupted", "simulation interrupted"),
)
_COMPILED_FAILURES = tuple((re.compile(p, re.I), label) for p, label in _FAILURE_PATTERNS)

#: Exception types that mean "this candidate produced no measurement" rather than "the
#: harness is broken". A candidate that fails to solve is a verdict, not a crash.
#:
#: PySpice raises NgSpiceCommandError, which derives from NameError and so is not covered
#: by OSError/ValueError/RuntimeError. Measured over 33 designs at four PVT corners, 3-6
#: designs per corner raise it, so leaving it uncaught ends a training run within minutes
#: — and ends it with a traceback rather than a logged, attributable Invalid.
#:
#: Resolved lazily and cached: guards.py must import without a simulator installed, and
#: silq.sim.probe imports this module, so a module-level import would be circular.
_SIM_EXC_CACHE: tuple[type[BaseException], ...] | None = None


def _simulator_exceptions() -> tuple[type[BaseException], ...]:
    global _SIM_EXC_CACHE
    if _SIM_EXC_CACHE is None:
        types: list[type[BaseException]] = [OSError, ValueError, RuntimeError]
        try:                                     # ProbeError is already a RuntimeError
            from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError
            types.append(NgSpiceCommandError)
        except ImportError:
            pass
        _SIM_EXC_CACHE = tuple(types)
    return _SIM_EXC_CACHE


# =============================================================================
# Exceptions
# =============================================================================

class GuardError(Exception):
    """Base class for guard-layer errors. Never caught broadly inside this module."""


class GuardConfigError(GuardError):
    """The guard layer itself is misconfigured (bad paths, missing probe hooks)."""


class ArtifactError(GuardError):
    """Raw artifacts could not be written. Fatal: an unlogged run is an invalid run."""


class SearchHalted(BaseException):
    """A Tier-5 pathology fired. The run must stop.

    Deliberately derived from BaseException, not Exception, so that an `except
    Exception` anywhere in a training loop cannot swallow it. This mirrors how
    KeyboardInterrupt and SystemExit behave, and it is intentional: Tier 5 means the
    experiment is producing garbage and must not continue.
    """

    def __init__(self, check: "Check", detail: str, halt_path: Path):
        super().__init__(f"{check.value}: {detail}  (see {halt_path})")
        self.check = check
        self.detail = detail
        self.halt_path = halt_path


# =============================================================================
# Check registry — every INVALID names one of these.
# =============================================================================

class Check(Enum):
    # Tier 1 — run integrity
    T1_EXIT_CODE = "T1.1_simulator_exit_code_nonzero"
    T1_STDERR_FAILURE = "T1.2_solver_failure_text_in_output"
    T1_OUTPUT_MISSING = "T1.3_output_file_missing_or_empty"
    T1_TRANSIENT_TRUNCATED = "T1.4_transient_did_not_reach_tstop"

    # Tier 2 — circuit sanity
    T2_NOT_SATURATED = "T2.5_mosfet_not_in_saturation"
    T2_TAIL_CURRENT = "T2.6_tail_current_wrong_or_zero"
    T2_NODE_OUT_OF_RAILS = "T2.7_node_voltage_outside_supply_rails"
    T2_AT_PDK_BOUND = "T2.8_device_parameter_at_pdk_bound"

    # Tier 3 — corner integrity
    T3_CORNER_IDENTICAL = "T3.9_corner_models_identical_include_failed"

    # Tier 4 — physical plausibility
    T4_DC_GAIN = "T4.10_dc_gain_implausible"
    T4_PEAKING = "T4.11_peaking_implausible"
    T4_POWER = "T4.12_power_implausible"
    T4_NOISE = "T4.13_input_noise_implausibly_low"
    T4_HD3 = "T4.14_hd3_implausibly_good"
    T4_EYE = "T4.15_eye_zero_or_above_theoretical_max"

    # Tier 5 — search pathology (HALT)
    T5_PARAM_PINNED = "T5.16_parameter_pinned_at_range_edge"
    T5_DUPLICATE_METRICS = "T5.17_identical_metrics_from_distinct_designs"
    T5_REWARD_EYE_DIVERGE = "T5.18_reward_up_while_eye_height_down"
    T5_INVALID_RATE = "T5.19_invalid_rate_out_of_band"
    T5_TOO_GOOD = "T5.20_beats_multiple_hostile_specs_simultaneously"


TIER_OF: dict[Check, int] = {
    **{c: 1 for c in (Check.T1_EXIT_CODE, Check.T1_STDERR_FAILURE,
                      Check.T1_OUTPUT_MISSING, Check.T1_TRANSIENT_TRUNCATED)},
    **{c: 2 for c in (Check.T2_NOT_SATURATED, Check.T2_TAIL_CURRENT,
                      Check.T2_NODE_OUT_OF_RAILS, Check.T2_AT_PDK_BOUND)},
    Check.T3_CORNER_IDENTICAL: 3,
    **{c: 4 for c in (Check.T4_DC_GAIN, Check.T4_PEAKING, Check.T4_POWER,
                      Check.T4_NOISE, Check.T4_HD3, Check.T4_EYE)},
    **{c: 5 for c in (Check.T5_PARAM_PINNED, Check.T5_DUPLICATE_METRICS,
                      Check.T5_REWARD_EYE_DIVERGE, Check.T5_INVALID_RATE,
                      Check.T5_TOO_GOOD)},
}


# =============================================================================
# Artifacts — every run is written to disk and never deleted.
# =============================================================================

@dataclass
class RunArtifacts:
    """The complete raw record of one simulator invocation."""

    run_id: str
    directory: Path
    netlist: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    data_files: dict[str, Path] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # transient bookkeeping for Tier 1.4
    requested_tstop: float | None = None
    reached_tstop: float | None = None

    def write(self) -> None:
        """Persist everything. Called for EVERY run, pass or fail."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "netlist.cir").write_text(self.netlist, encoding="utf-8")
            (self.directory / "stdout.txt").write_text(self.stdout, encoding="utf-8")
            (self.directory / "stderr.txt").write_text(self.stderr, encoding="utf-8")
            (self.directory / "meta.json").write_text(
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "exit_code": self.exit_code,
                        "requested_tstop": self.requested_tstop,
                        "reached_tstop": self.reached_tstop,
                        "data_files": {k: str(v) for k, v in self.data_files.items()},
                        **self.meta,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            raise ArtifactError(
                f"could not write artifacts for run {self.run_id} to {self.directory}: {e}"
            ) from e

    def adopt(self, path: Path, name: str | None = None) -> Path:
        """Copy a simulator output file into the run directory so it survives."""
        name = name or path.name
        dest = self.directory / name
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shutil.copy2(path, dest)
                self.data_files[name] = dest
        except OSError as e:
            raise ArtifactError(f"could not adopt {path} into {self.directory}: {e}") from e
        return dest


class ArtifactStore:
    """Creates and owns per-run artifact directories. Never deletes anything."""

    def __init__(self, root: Path | str = Path("results/raw")):
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ArtifactError(f"cannot create artifact root {self.root}: {e}") from e
        self._counter = 0

    def new_run(self, **meta: Any) -> RunArtifacts:
        self._counter += 1
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{self._counter:06d}_{uuid.uuid4().hex[:8]}"
        return RunArtifacts(run_id=run_id, directory=self.root / run_id, meta=dict(meta))


# =============================================================================
# Verdicts
# =============================================================================

class Verdict(ABC):
    """Sealed result type. Exactly two inhabitants: Valid and Invalid."""

    run_id: str
    artifact_dir: Path

    @property
    def is_valid(self) -> bool:
        return isinstance(self, Valid)


@dataclass(frozen=True)
class Valid(Verdict):
    metrics: Any                      # silq.sim.measures.Measures
    run_id: str
    artifact_dir: Path

    def unwrap(self) -> Any:
        return self.metrics


@dataclass(frozen=True)
class Invalid(Verdict):
    """A failed evaluation.

    Note the absence of a `metrics` attribute. This is load-bearing: code that reads
    `.metrics` without checking `.is_valid` raises AttributeError instead of silently
    consuming a zeroed struct.
    """

    check: Check
    reason: str
    run_id: str
    artifact_dir: Path

    #: How far past the failing bound this design sits, made dimensionless by dividing
    #: the overshoot by that bound's own scale. 0.0 means "exactly on the bound"; 1.0
    #: means "past it by the width of the bound itself"; 9.0 for a tail current that
    #: delivers nothing against a 10% tolerance. None means NO distance exists for this
    #: failure — the solver never returned, the .op probe produced nothing, the corner
    #: include did not take effect. A consumer must treat None as the worst case, not
    #: as zero: it is missing information, not a small violation.
    #:
    #: Reported, never consulted. Adding it changes no verdict; the same designs are
    #: rejected for the same reasons. It exists because a FLAT rejection penalty gives a
    #: search no gradient out of the infeasible region, and that is not a hypothetical:
    #: over a 43,009-simulation PPO run, 99.90% of candidates were rejected and the
    #: invalid rate got WORSE with training (97.4% at step 500 -> 99.90% at step 40k),
    #: ending below uniform random sampling of the same box (99.6% invalid). With every
    #: rejection scored identically the value function is flat and the entropy bonus
    #: walks the policy mean out to the corners of the action box.
    violation: float | None = None

    @property
    def tier(self) -> int:
        return TIER_OF[self.check]

    def unwrap(self) -> Any:
        raise GuardError(f"{self.check.value}: {self.reason} (raw output: {self.artifact_dir})")

    def __str__(self) -> str:
        return (f"INVALID[tier {self.tier}] {self.check.value}: {self.reason} "
                f"| run={self.run_id} | raw={self.artifact_dir}")


def _invalid(check: Check, reason: str, art: RunArtifacts,
             violation: float | None = None) -> Invalid:
    return Invalid(check=check, reason=reason, run_id=art.run_id,
                   artifact_dir=art.directory, violation=violation)


def _violation(excess: float, scale: float) -> float | None:
    """Overshoot past a bound, in units of that bound's scale. See `Invalid.violation`.

    Returns None — "no distance" — rather than a number whenever the quantity is not
    computable: a non-finite input (the solve produced NaN, so the design's distance from
    the bound is unknown, not zero) or a non-positive scale (dividing by a bound of zero
    does not produce a relative distance). Clamped at 0 because a caller reaching this
    function has already decided the check failed; a tiny negative from floating-point
    rounding at the bound must read as "on the bound", not as a valid design.
    """
    if not (np.isfinite(excess) and np.isfinite(scale)) or scale <= 0:
        return None
    return max(float(excess) / float(scale), 0.0)


# =============================================================================
# TIER 1 — run integrity. Runs before anything is parsed.
# =============================================================================

def check_run_integrity(art: RunArtifacts, *, expect_output: bool = True) -> Invalid | None:
    """Checks 1–4. Returns Invalid on the first failure, else None."""
    # 1. exit code. An unrecorded exit code is NOT a pass — a check whose input is
    #    missing must fail, or every caller that forgets to set it silently disables it.
    if art.exit_code is None:
        return _invalid(Check.T1_EXIT_CODE,
                        "simulator exit code was never recorded; run integrity cannot be "
                        "confirmed. The runner must set RunArtifacts.exit_code.", art)
    if art.exit_code != 0:
        return _invalid(Check.T1_EXIT_CODE,
                        f"simulator exited with code {art.exit_code}", art)

    # 2. solver failure text (scan BOTH streams — ngspice reports to either)
    blob = f"{art.stderr}\n{art.stdout}"
    for pattern, label in _COMPILED_FAILURES:
        hit = pattern.search(blob)
        if hit:
            line = _containing_line(blob, hit.start())
            return _invalid(Check.T1_STDERR_FAILURE,
                            f"solver reported '{label}' -> {line!r}", art)

    # 3. output file present and non-empty
    if expect_output:
        if not art.data_files:
            return _invalid(Check.T1_OUTPUT_MISSING,
                            "no simulator output file was produced", art)
        for name, path in art.data_files.items():
            p = Path(path)
            if not p.exists():
                return _invalid(Check.T1_OUTPUT_MISSING,
                                f"expected output '{name}' does not exist at {p}", art)
            if p.stat().st_size == 0:
                return _invalid(Check.T1_OUTPUT_MISSING,
                                f"expected output '{name}' is empty ({p})", art)

    # 4. transient reached the requested stop time
    if art.requested_tstop is not None:
        if art.reached_tstop is None:
            return _invalid(Check.T1_TRANSIENT_TRUNCATED,
                            f"transient requested tstop={art.requested_tstop:g}s but no "
                            "final timepoint was recorded", art)
        # 0.1% tolerance for the final timestep landing just short.
        if art.reached_tstop < art.requested_tstop * 0.999:
            return _invalid(
                Check.T1_TRANSIENT_TRUNCATED,
                f"transient truncated: reached {art.reached_tstop:g}s of requested "
                f"{art.requested_tstop:g}s "
                f"({100 * art.reached_tstop / art.requested_tstop:.1f}%)", art)
    return None


def _containing_line(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    return text[start: end if end != -1 else len(text)].strip()[:200]


# =============================================================================
# TIER 2 — circuit sanity, from the operating point.
# =============================================================================

@dataclass(frozen=True)
class DeviceOP:
    """One MOSFET's operating point. All SI units."""

    name: str
    vds: float
    vdsat: float
    vgs: float = float("nan")
    vth: float = float("nan")
    id: float = float("nan")

    @property
    def headroom(self) -> float:
        return self.vds - self.vdsat

    @property
    def saturated(self) -> bool:
        return self.headroom >= SATURATION_HEADROOM_V


@dataclass(frozen=True)
class OperatingPoint:
    """Parsed .op result: signal-path devices, node voltages, tail currents."""

    devices: tuple[DeviceOP, ...] = ()
    node_voltages: Mapping[str, float] = field(default_factory=dict)
    tail_currents: Mapping[str, float] = field(default_factory=dict)


def check_circuit_sanity(op: OperatingPoint, dv: DesignVars, art: RunArtifacts,
                         *, vdd: float, requested_tail_a: float | None = None
                         ) -> Invalid | None:
    """Checks 5–8. Returns Invalid on the first failure, else None."""
    # 5. every signal-path MOSFET saturated with headroom, named if not
    if not op.devices:
        return _invalid(Check.T2_NOT_SATURATED,
                        "operating point contained no signal-path devices — the .op "
                        "probe returned nothing, so saturation cannot be confirmed", art)
    unsaturated = [d for d in op.devices if not d.saturated]
    if unsaturated:
        detail = "; ".join(
            f"{d.name}: Vds={d.vds:.4f}V Vdsat={d.vdsat:.4f}V "
            f"headroom={d.headroom * 1e3:+.1f}mV (need >= {SATURATION_HEADROOM_V * 1e3:.0f}mV)"
            for d in unsaturated
        )
        # Distance is set by the device that misses saturation by the MOST: fixing any
        # lesser one still leaves this design invalid, so the worst offender is what the
        # design has to travel to become buildable. A non-finite Vds/Vdsat lands here
        # too (nan >= 0.05 is False), and its distance is unknown, not large — _violation
        # turns that into None.
        shortfalls = [SATURATION_HEADROOM_V - d.headroom for d in unsaturated]
        worst = max(shortfalls) if all(np.isfinite(s) for s in shortfalls) else float("nan")
        return _invalid(Check.T2_NOT_SATURATED,
                        f"{len(unsaturated)} device(s) not in saturation — {detail}", art,
                        violation=_violation(worst, SATURATION_HEADROOM_V))

    # 6. tail current nonzero and within tolerance of request
    requested = dv.i_tail if requested_tail_a is None else requested_tail_a
    if not op.tail_currents:
        return _invalid(Check.T2_TAIL_CURRENT,
                        "no tail-current branches were probed; cannot confirm bias", art)
    total = float(sum(abs(v) for v in op.tail_currents.values()))
    if total == 0.0:
        # Delivering nothing is a 100% deviation from any positive request. Against a
        # non-positive request there is no fraction to take, so no distance exists.
        zero_err = _violation(1.0 - TAIL_CURRENT_TOLERANCE, TAIL_CURRENT_TOLERANCE)
        return _invalid(Check.T2_TAIL_CURRENT,
                        f"tail current is exactly zero (requested {requested:g} A) — "
                        f"branches {dict(op.tail_currents)}", art,
                        violation=zero_err if requested > 0 else None)
    if requested > 0:
        err = abs(total - requested) / requested
        if err > TAIL_CURRENT_TOLERANCE:
            return _invalid(
                Check.T2_TAIL_CURRENT,
                f"tail current {total:g} A deviates {err * 100:.1f}% from requested "
                f"{requested:g} A (limit {TAIL_CURRENT_TOLERANCE * 100:.0f}%) — "
                f"branches {dict(op.tail_currents)}", art,
                violation=_violation(err - TAIL_CURRENT_TOLERANCE, TAIL_CURRENT_TOLERANCE))

    # 7. node voltages inside the rails
    if not op.node_voltages:
        return _invalid(Check.T2_NODE_OUT_OF_RAILS,
                        "no node voltages were probed; cannot confirm the circuit is "
                        "biased inside the rails", art)
    lo, hi = 0.0 - RAIL_MARGIN_V, vdd + RAIL_MARGIN_V
    for node, v in op.node_voltages.items():
        if not np.isfinite(v):
            return _invalid(Check.T2_NODE_OUT_OF_RAILS,
                            f"node '{node}' voltage is {v} — the .op did not solve", art)
        if not (lo <= v <= hi):
            # Scale is the rail span itself. A node 50 mV outside a 1.8 V supply is a
            # bias that is nearly right; the corner of this action space asks for 50 V
            # across the load, which is not a circuit. Those must not score alike.
            return _invalid(Check.T2_NODE_OUT_OF_RAILS,
                            f"node '{node}' at {v:.4f}V is outside rails "
                            f"[{lo:.3f}, {hi:.3f}]V", art,
                            violation=_violation(max(lo - v, v - hi), hi - lo))

    # 8. no device parameter sitting exactly on — or outside — a PDK bound.
    #    'Outside' is included deliberately: a zero-width device is not "exactly at"
    #    0.42 um, so a bounds-equality test alone would wave it through.
    for pname, (pmin, pmax) in PDK_BOUNDS.items():
        val = getattr(dv, pname, None)
        if val is None:
            continue
        if val < pmin or val > pmax:
            excess, scale = (pmin - val, pmin) if val < pmin else (val - pmax, pmax)
            return _invalid(
                Check.T2_AT_PDK_BOUND,
                f"design parameter '{pname}' = {val:g} is outside the SKY130 legal range "
                f"[{pmin:g}, {pmax:g}]; the device does not physically exist and any "
                "model output for it is extrapolation", art,
                violation=_violation(excess, scale))
        for bound, which in ((pmin, "min"), (pmax, "max")):
            if bound != 0 and abs(val - bound) <= abs(bound) * PDK_BOUND_RELTOL:
                # ON the bound, not past it: distance zero, the smallest violation there
                # is. Still invalid — the model is extrapolating — but a design one part
                # in 1e9 from legal is as close to buildable as an invalid design gets.
                return _invalid(
                    Check.T2_AT_PDK_BOUND,
                    f"design parameter '{pname}' = {val:g} sits exactly on the SKY130 "
                    f"{which} bound ({bound:g}); the optimizer is against the process "
                    "limit and the model may be extrapolating", art, violation=0.0)
    return None


# =============================================================================
# TIER 3 — corner integrity.
# =============================================================================

class CornerProbe(Protocol):
    """Returns a scalar that MUST differ between process corners.

    The natural probe is the threshold voltage of a fixed reference device at a fixed
    bias, read from the .op of an identical netlist under each corner's .lib.
    """

    def __call__(self, corner: str) -> float: ...


def check_corner_integrity(probe: CornerProbe, corners: Sequence[str] = ("tt", "ss"),
                           *, artifact_dir: Path | None = None,
                           run_id: str = "corner-probe") -> Invalid | None:
    """Check 9. Proves the corner `.lib` include actually changed the models.

    Without this, a silently-failed include means every 'corner' is typical and the
    PVT sweep is five identical runs wearing different labels.
    """
    if len(corners) < 2:
        raise GuardConfigError("corner integrity needs at least two corners to compare")

    readings: dict[str, float] = {}
    for c in corners:
        try:
            reading = float(probe(c))
            # NaN must not reach the comparison below: abs(nan - nan) < tol is False,
            # so a probe returning NaN for every corner would PASS the difference test.
            if not np.isfinite(reading):
                return Invalid(
                    check=Check.T3_CORNER_IDENTICAL,
                    reason=(f"corner probe for '{c}' returned {reading}, not a model "
                            "parameter. The corner cannot be verified."),
                    run_id=run_id,
                    artifact_dir=artifact_dir or Path("results/raw"),
                )
            readings[c] = reading
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # Specific, expected failure modes. Anything else propagates untouched.
            raise GuardConfigError(
                f"corner probe failed for corner '{c}': {type(e).__name__}: {e}"
            ) from e

    names = list(readings)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            delta = abs(readings[a] - readings[b])
            if delta < CORNER_PROBE_MIN_DELTA:
                return Invalid(
                    check=Check.T3_CORNER_IDENTICAL,
                    reason=(f"corners '{a}' and '{b}' returned effectively identical model "
                            f"probes ({readings[a]:.6g} vs {readings[b]:.6g}, "
                            f"delta={delta:.3g} < {CORNER_PROBE_MIN_DELTA:.3g}). The corner "
                            ".lib include is not taking effect — every PVT corner is "
                            "silently running typical models."),
                    run_id=run_id,
                    artifact_dir=artifact_dir or Path("results/raw"),
                )
    return None


# =============================================================================
# TIER 4 — physical plausibility. These mean the MEASUREMENT broke.
# =============================================================================

def theoretical_eye_v_max_mv(dv: DesignVars, vdd: float = 1.8) -> float:
    """Max differential output swing this topology can produce, in mV.

    Steering the whole tail current into one load drops that side by I_tail * R_load
    while the other side sits at VDD, so the differential output (outp - outn) travels
    from -I_tail*R_load to +I_tail*R_load: a peak-to-peak of 2 * I_tail * R_load. The
    eye height in `measures.eye` is differential (see sim/eye.py), so the ceiling must
    be the differential figure. The earlier single-ended form was a factor of two too
    tight and could reject a legitimately large eye.

    The drop is also capped by the supply: a load node cannot be pulled below ground,
    so I_tail * R_load can never exceed VDD however large the product is asked to be.
    Without that cap the ceiling for a 20 mA / 5 kohm request evaluates to 100 V, which
    no measurement could ever exceed -- the check had no teeth exactly where a broken
    eye measurement is most likely.

    Note this uses the *requested* tail current. The mirror delivers slightly less than
    requested, so this stays a true upper bound; the deviation itself is Tier 2.6's job,
    not this one's.
    """
    return 2.0 * min(abs(dv.i_tail * dv.r_load), abs(vdd)) * 1e3


def check_physical_plausibility(m: Any, dv: DesignVars, art: RunArtifacts,
                                *, topology_has_gain: bool = True,
                                vdd: float = 1.8) -> Invalid | None:
    """Checks 10–15. Returns Invalid on the first failure, else None."""
    # Any NaN/inf anywhere is a broken parse, full stop.
    for fname, val in m.as_dict().items():
        if isinstance(val, float) and not np.isfinite(val):
            return _invalid(Check.T4_DC_GAIN if fname.startswith("dc") else Check.T4_PEAKING,
                            f"metric '{fname}' is {val} — non-finite value from the parser",
                            art)

    # 10. DC gain
    if m.dc_gain_db > DC_GAIN_DB_MAX:
        return _invalid(Check.T4_DC_GAIN,
                        f"DC gain {m.dc_gain_db:.2f} dB exceeds {DC_GAIN_DB_MAX:.0f} dB — "
                        "a single degenerated pair cannot do this", art,
                        violation=_violation(m.dc_gain_db - DC_GAIN_DB_MAX, DC_GAIN_DB_MAX))
    if topology_has_gain and m.dc_gain_db < DC_GAIN_DB_MIN:
        # DC_GAIN_DB_MIN is 0 dB, so there is no relative distance to *it*; the scale
        # that does exist is the width of the plausible band, 0 to 60 dB.
        return _invalid(Check.T4_DC_GAIN,
                        f"DC gain {m.dc_gain_db:.2f} dB is below {DC_GAIN_DB_MIN:.0f} dB "
                        "for a topology that should have gain", art,
                        violation=_violation(DC_GAIN_DB_MIN - m.dc_gain_db,
                                             DC_GAIN_DB_MAX - DC_GAIN_DB_MIN))

    # 11. peaking
    if m.boost_db > PEAKING_DB_MAX:
        return _invalid(Check.T4_PEAKING,
                        f"peaking {m.boost_db:.2f} dB exceeds {PEAKING_DB_MAX:.0f} dB", art,
                        violation=_violation(m.boost_db - PEAKING_DB_MAX, PEAKING_DB_MAX))

    # 12. power
    if m.power_w < POWER_W_MIN:
        return _invalid(Check.T4_POWER,
                        f"power {m.power_w * 1e3:.4f} mW is below "
                        f"{POWER_W_MIN * 1e3:.2f} mW — the bias is not being applied", art,
                        violation=_violation(POWER_W_MIN - m.power_w, POWER_W_MIN))
    if m.power_w > POWER_W_MAX:
        return _invalid(Check.T4_POWER,
                        f"power {m.power_w * 1e3:.2f} mW exceeds "
                        f"{POWER_W_MAX * 1e3:.0f} mW", art,
                        violation=_violation(m.power_w - POWER_W_MAX, POWER_W_MAX))

    # 13. input-referred noise
    if m.noise_vrms < NOISE_VRMS_MIN:
        return _invalid(Check.T4_NOISE,
                        f"input-referred noise {m.noise_vrms * 1e3:.5f} mVrms is below "
                        f"{NOISE_VRMS_MIN * 1e3:.2f} mVrms — the .noise integration "
                        "returned nothing", art,
                        violation=_violation(NOISE_VRMS_MIN - m.noise_vrms, NOISE_VRMS_MIN))

    # 14. HD3 too good
    if m.hd3_db < HD3_DB_FLOOR:
        return _invalid(Check.T4_HD3,
                        f"HD3 {m.hd3_db:.2f} dB is better than {HD3_DB_FLOOR:.0f} dB — "
                        "the FFT found no third harmonic, which means no signal", art,
                        violation=_violation(HD3_DB_FLOOR - m.hd3_db, abs(HD3_DB_FLOOR)))

    # 15. eye
    if m.eye_h_ui == 0.0 or m.eye_v_mv == 0.0:
        return _invalid(Check.T4_EYE,
                        f"eye is exactly zero (h={m.eye_h_ui} UI, v={m.eye_v_mv} mV)", art)
    if m.eye_h_ui > EYE_H_UI_MAX:
        return _invalid(Check.T4_EYE,
                        f"eye width {m.eye_h_ui:.3f} UI exceeds one unit interval", art,
                        violation=_violation(m.eye_h_ui - EYE_H_UI_MAX, EYE_H_UI_MAX))
    v_max = theoretical_eye_v_max_mv(dv, vdd)
    if v_max <= 0:
        # Skipping the ceiling because the ceiling is zero would let ANY eye height
        # through on a design that cannot produce signal at all.
        return _invalid(Check.T4_EYE,
                        f"theoretical maximum eye height is {v_max:.3f} mV "
                        f"(I_tail={dv.i_tail:g} A, R_load={dv.r_load:g} ohm) — this "
                        f"design cannot produce an eye, yet {m.eye_v_mv:.1f} mV was "
                        "reported", art)
    if m.eye_v_mv > v_max:
        return _invalid(Check.T4_EYE,
                        f"eye height {m.eye_v_mv:.1f} mV exceeds the theoretical maximum "
                        f"{v_max:.1f} mV (2 x min(I_tail x R_load, VDD), "
                        f"I_tail={dv.i_tail:g} A, R_load={dv.r_load:g} ohm, "
                        f"VDD={vdd:g} V)", art,
                        violation=_violation(m.eye_v_mv - v_max, v_max))
    return None


# =============================================================================
# TIER 5 — search pathology. These HALT the run.
# =============================================================================

@dataclass
class EvalRecord:
    """One entry in the experiment log."""

    run_id: str
    params: dict[str, float]                 # physical design variables
    accepted: bool                           # did the search accept this candidate
    valid: bool
    reward: float | None = None
    metrics: dict[str, float] | None = None  # None when invalid
    check: str | None = None                 # the failing check, when invalid
    artifact_dir: str | None = None


class SearchMonitor:
    """Checks 16–20 over the experiment log. Raises SearchHalted; writes HALT.txt."""

    def __init__(self, spec: Spec, *, halt_path: Path | str = Path("HALT.txt"),
                 action_space: Mapping[str, tuple[float, float]] | None = None,
                 stream=None):
        self.spec = spec
        self.halt_path = Path(halt_path)
        self.space = dict(action_space or ACTION_SPACE)
        self.records: list[EvalRecord] = []
        self._stream = stream if stream is not None else sys.stdout
        self._edge_streak: dict[str, int] = {k: 0 for k in self.space}
        self._metric_index: dict[tuple, tuple[str, dict]] = {}

    # -- ingestion ---------------------------------------------------------
    def record(self, rec: EvalRecord) -> None:
        """Append one evaluation and run every Tier-5 check. May raise SearchHalted."""
        self.records.append(rec)
        self._check_16_pinned(rec)
        self._check_17_duplicates(rec)
        self._check_18_reward_eye_divergence()
        self._check_19_invalid_rate()
        self._check_20_too_good(rec)

    # -- 16 ----------------------------------------------------------------
    def _check_16_pinned(self, rec: EvalRecord) -> None:
        if not rec.accepted:
            return
        for name, (lo, hi) in self.space.items():
            if name not in rec.params:
                continue
            span = hi - lo
            if span <= 0:
                continue
            v = rec.params[name]
            at_edge = (v - lo) <= span * EDGE_PIN_FRACTION or (hi - v) <= span * EDGE_PIN_FRACTION
            self._edge_streak[name] = self._edge_streak[name] + 1 if at_edge else 0
            if self._edge_streak[name] > EDGE_PIN_MAX_CONSECUTIVE:
                edge = "lower" if (v - lo) <= span * EDGE_PIN_FRACTION else "upper"
                self._halt(
                    Check.T5_PARAM_PINNED,
                    f"parameter '{name}' has sat within {EDGE_PIN_FRACTION * 100:.0f}% of its "
                    f"{edge} range edge for {self._edge_streak[name]} consecutive accepted "
                    f"designs. Current value: {name}={v:g} (range [{lo:g}, {hi:g}]). Either "
                    "the range is wrong or this parameter has no effect on the objective.",
                    rec)

    # -- 17 ----------------------------------------------------------------
    def _check_17_duplicates(self, rec: EvalRecord) -> None:
        if not rec.valid or rec.metrics is None:
            return
        key = tuple(round(float(rec.metrics[k]), DUPLICATE_METRIC_DECIMALS)
                    for k in sorted(rec.metrics))
        prev = self._metric_index.get(key)
        if prev is not None:
            prev_run, prev_params = prev
            if _params_differ(prev_params, rec.params):
                self._halt(
                    Check.T5_DUPLICATE_METRICS,
                    f"two distinct parameter vectors produced identical metrics to "
                    f"{DUPLICATE_METRIC_DECIMALS} decimal places. The parser is returning a "
                    f"constant.\n  run A: {prev_run}  params={prev_params}\n"
                    f"  run B: {rec.run_id}  params={rec.params}", rec)
        else:
            self._metric_index[key] = (rec.run_id, dict(rec.params))

        # Same intent, finer resolution: a single metric frozen across many distinct
        # designs is the same defect and shows up long before the full vector collides.
        self._check_17b_constant_metric()

    def _check_17b_constant_metric(self) -> None:
        valid = [r for r in self.records if r.valid and r.metrics]
        if len(valid) < TREND_WINDOW:
            return
        window = valid[-TREND_WINDOW:]
        distinct_params = {tuple(sorted(r.params.items())) for r in window}
        if len(distinct_params) < 2:
            return
        for metric in window[0].metrics:
            vals = {round(float(r.metrics[metric]), DUPLICATE_METRIC_DECIMALS)
                    for r in window if metric in r.metrics}
            if len(vals) == 1:
                self._halt(
                    Check.T5_DUPLICATE_METRICS,
                    f"metric '{metric}' returned the identical value "
                    f"{next(iter(vals))!r} across {len(window)} evaluations spanning "
                    f"{len(distinct_params)} distinct parameter vectors. That metric is a "
                    "constant, not a measurement.", window[-1])

    # -- 18 ----------------------------------------------------------------
    def _check_18_reward_eye_divergence(self) -> None:
        usable = [r for r in self.records
                  if r.valid and r.reward is not None and r.metrics
                  and "eye_h_ui" in r.metrics]
        if len(usable) < TREND_WINDOW:
            return
        w = usable[-TREND_WINDOW:]
        x = np.arange(len(w), dtype=float)
        reward_slope = float(np.polyfit(x, [r.reward for r in w], 1)[0])
        eye_slope = float(np.polyfit(x, [r.metrics["eye_h_ui"] for r in w], 1)[0])
        if reward_slope > 0 and eye_slope < 0:
            self._halt(
                Check.T5_REWARD_EYE_DIVERGE,
                f"over the last {TREND_WINDOW} evaluations reward is improving "
                f"(slope {reward_slope:+.4g}/eval) while eye height is degrading "
                f"(slope {eye_slope:+.4g} UI/eval). The reward is being optimized against "
                "something that is not signal integrity.", w[-1])

    # -- 19 ----------------------------------------------------------------
    def _check_19_invalid_rate(self) -> None:
        if len(self.records) < INVALID_WINDOW:
            return
        w = self.records[-INVALID_WINDOW:]
        rate = sum(1 for r in w if not r.valid) / len(w)
        if rate > INVALID_RATE_MAX:
            self._halt(Check.T5_INVALID_RATE,
                       f"INVALID rate {rate * 100:.1f}% over the last {INVALID_WINDOW} "
                       f"evaluations exceeds {INVALID_RATE_MAX * 100:.0f}%. The search is "
                       "spending its budget in a region the simulator cannot solve.", w[-1])
        if rate < INVALID_RATE_MIN:
            self._halt(Check.T5_INVALID_RATE,
                       f"INVALID rate {rate * 100:.1f}% over the last {INVALID_WINDOW} "
                       f"evaluations is below {INVALID_RATE_MIN * 100:.0f}%. A run this "
                       "clean almost always means the guards are not firing at all — "
                       "verify the validation layer is actually wired in.", w[-1])

    # -- 20 ----------------------------------------------------------------
    def _spec_beats(self, m: Mapping[str, float]) -> list[str]:
        """Which spec targets are beaten by more than SPEC_BEAT_FRACTION."""
        s, beats = self.spec, []
        upper = (("hd3_db", s.hd3_db_max), ("noise_vrms", s.noise_vrms_max),
                 ("power_w", s.power_w_max), ("area_mm2", s.area_mm2_max))
        for key, limit in upper:
            if key not in m:
                continue
            if key == "hd3_db":
                # dB limit is negative; "beating by 30%" = 30% more margin below it.
                if m[key] <= limit - abs(limit) * SPEC_BEAT_FRACTION:
                    beats.append(f"{key}={m[key]:.4g} vs limit {limit:.4g}")
            elif limit > 0 and m[key] <= limit * (1 - SPEC_BEAT_FRACTION):
                beats.append(f"{key}={m[key]:.4g} vs limit {limit:.4g}")
        lower = (("eye_h_ui", s.eye_h_ui_min), ("eye_v_mv", s.eye_v_mv_min))
        for key, limit in lower:
            if key in m and limit > 0 and m[key] >= limit * (1 + SPEC_BEAT_FRACTION):
                beats.append(f"{key}={m[key]:.4g} vs minimum {limit:.4g}")
        return beats

    def _check_20_too_good(self, rec: EvalRecord) -> None:
        if not rec.valid or rec.metrics is None:
            return
        beats = self._spec_beats(rec.metrics)
        if len(beats) >= SPEC_BEAT_COUNT:
            self._halt(
                Check.T5_TOO_GOOD,
                f"design simultaneously beats {len(beats)} spec targets by more than "
                f"{SPEC_BEAT_FRACTION * 100:.0f}%: " + "; ".join(beats) +
                ". These specs trade against each other; winning all of them at once "
                "means the measurements are not real.", rec)

    # -- halting -----------------------------------------------------------
    def _halt(self, check: Check, detail: str, rec: EvalRecord) -> None:
        body = (
            "HALT — Tier 5 search pathology detected\n"
            "=======================================\n\n"
            f"trigger      : {check.value}\n"
            f"detail       : {detail}\n\n"
            f"run id       : {rec.run_id}\n"
            f"raw output   : {rec.artifact_dir}\n"
            f"parameters   : {json.dumps(rec.params, indent=2, default=str)}\n"
            f"metrics      : {json.dumps(rec.metrics, indent=2, default=str)}\n"
            f"reward       : {rec.reward}\n"
            f"evaluations  : {len(self.records)}\n"
            f"timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Optimization has been stopped. Do not restart until this is explained.\n"
        )
        try:
            self.halt_path.parent.mkdir(parents=True, exist_ok=True)
            self.halt_path.write_text(body, encoding="utf-8")
        except OSError as e:
            raise ArtifactError(f"could not write {self.halt_path}: {e}") from e

        banner = "!" * 78
        print(f"\n{banner}\n{body}{banner}\n", file=self._stream, flush=True)
        raise SearchHalted(check, detail, self.halt_path)


def _params_differ(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    if set(a) != set(b):
        return True
    return any(float(a[k]) != float(b[k]) for k in a)


# =============================================================================
# Orchestration — the single entry point everything else must use.
# =============================================================================

class GuardedEvaluator:
    """Wraps a raw measurement callable so no metric escapes unvalidated.

    `raw_eval(dv, **kw) -> (Measures, RunArtifacts, OperatingPoint | None)`
    """

    def __init__(self, raw_eval, spec: Spec, store: ArtifactStore | None = None,
                 monitor: SearchMonitor | None = None, *, topology_has_gain: bool = True,
                 require_operating_point: bool = True,
                 corner_probe: CornerProbe | None = None):
        self.raw_eval = raw_eval
        self.spec = spec
        self.store = store or ArtifactStore()
        self.monitor = monitor
        self.topology_has_gain = topology_has_gain
        # If a caller supplies no .op, Tier 2 has nothing to check. Defaulting that to
        # "pass" would silently disable four checks, so it defaults to "reject".
        self.require_operating_point = require_operating_point
        self.corner_probe = corner_probe
        self._corners_verified: set[tuple[str, ...]] = set()

    # -- Tier 3 wiring -----------------------------------------------------
    def verify_corners(self, corners: Sequence[str] = ("tt", "ss"),
                       *, force: bool = False) -> Invalid | None:
        """Run check 9. Call at startup and at the top of every corner batch.

        Results are cached per corner set; pass force=True to re-probe (e.g. after the
        PDK path or PDK_ROOT changes underneath a long run).
        """
        if self.corner_probe is None:
            raise GuardConfigError(
                "no corner_probe was configured, so check 9 (corner integrity) cannot "
                "run. Supply one, or state explicitly that this run is single-corner."
            )
        key = tuple(corners)
        if not force and key in self._corners_verified:
            return None
        failure = check_corner_integrity(self.corner_probe, corners,
                                         artifact_dir=self.store.root)
        if failure is None:
            self._corners_verified.add(key)
        return failure

    def evaluate(self, dv: DesignVars, *, vdd: float = 1.8, accepted: bool = True,
                 reward: float | None = None, **kw) -> Verdict:
        art = self.store.new_run(design=asdict(dv), vdd=vdd, **kw)
        verdict: Verdict
        try:
            try:
                m, art_out, op = self.raw_eval(dv, artifacts=art, vdd=vdd, **kw)
                if art_out is not None:
                    art = art_out
            except _simulator_exceptions() as e:
                # Specific, anticipated failures only. Everything else propagates
                # untouched — but the artifacts are still written by the finally below,
                # because an unlogged run is a run we cannot investigate.
                art.stderr += f"\n[guarded] {type(e).__name__}: {e}\n"
                verdict = _invalid(Check.T1_STDERR_FAILURE, f"{type(e).__name__}: {e}", art)
            else:
                verdict = self._apply_tiers(m, dv, op, art, vdd)
        finally:
            art.write()   # ALWAYS: pass, fail, or unexpected exception in flight

        self._log(verdict, dv, accepted, reward)
        return verdict

    def _apply_tiers(self, m, dv, op, art, vdd) -> Verdict:
        failure = check_run_integrity(art)
        if failure is not None:
            return failure
        if op is None:
            if self.require_operating_point:
                return _invalid(
                    Check.T2_NOT_SATURATED,
                    "no operating point was supplied, so Tier 2 (saturation, tail "
                    "current, rails, PDK bounds) could not run. Provide an "
                    "OperatingPoint, or construct the evaluator with "
                    "require_operating_point=False and accept that four checks are off.",
                    art)
        else:
            failure = check_circuit_sanity(op, dv, art, vdd=vdd)
            if failure is not None:
                return failure
        failure = check_physical_plausibility(
            m, dv, art, topology_has_gain=self.topology_has_gain, vdd=vdd)
        if failure is not None:
            return failure
        return Valid(metrics=m, run_id=art.run_id, artifact_dir=art.directory)

    def _log(self, verdict: Verdict, dv: DesignVars, accepted: bool,
             reward: float | None) -> None:
        if self.monitor is None:
            return
        valid = verdict.is_valid
        self.monitor.record(EvalRecord(
            run_id=verdict.run_id,
            params=asdict(dv),
            accepted=accepted and valid,
            valid=valid,
            reward=reward,
            metrics=verdict.metrics.as_dict() if valid else None,
            check=None if valid else verdict.check.value,
            artifact_dir=str(verdict.artifact_dir),
        ))


# =============================================================================
# Bypass prevention
# =============================================================================

def seal_direct_access() -> None:
    """Make `silq.sim.measures.measure_all` raise if called outside this layer.

    Call once at the top of any training or experiment entry point. Enforces the
    'nothing may bypass the guards' rule at runtime rather than by convention.
    """
    from silq.sim import measures as _measures

    if getattr(_measures, "_guard_sealed", False):
        return
    original = _measures.measure_all

    def _sealed(*a, **kw):
        if not kw.pop("_via_guards", False):
            raise GuardError(
                "measure_all() was called directly. Every measurement must go through "
                "silq.guards.GuardedEvaluator so Tier 1-4 validation cannot be skipped."
            )
        return original(*a, **kw)

    _sealed.__wrapped__ = original          # type: ignore[attr-defined]
    _measures.measure_all = _sealed         # type: ignore[assignment]
    _measures._guard_sealed = True          # type: ignore[attr-defined]


__all__ = [
    "Check", "TIER_OF", "GuardError", "GuardConfigError", "ArtifactError", "SearchHalted",
    "RunArtifacts", "ArtifactStore", "Verdict", "Valid", "Invalid",
    "DeviceOP", "OperatingPoint",
    "check_run_integrity", "check_circuit_sanity", "check_corner_integrity",
    "check_physical_plausibility", "theoretical_eye_v_max_mv",
    "EvalRecord", "SearchMonitor", "GuardedEvaluator", "seal_direct_access",
]
