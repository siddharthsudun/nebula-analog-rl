"""The one entry point the RL envs and baselines should call.

Composes: resident ngspice server -> .op probe -> measurements -> guards.
Returns a `Verdict`, never a bare `Measures`.

    ev = build_evaluator(DEFAULT_SPEC, corner="tt")
    ev.verify_corners(("tt", "ss"))          # check 9, at startup
    v = ev.evaluate(design_vars)
    if v.is_valid:
        reward = compute_reward(v.unwrap(), spec)
    else:
        reward = INVALID_PENALTY               # v.check names what failed

ON THE RESIDENT-SERVER BACKEND
------------------------------
Tier 1 checks 1 and 2 assume a subprocess: an exit code and a stderr stream.
libngspice has neither. Rather than let those two checks quietly become no-ops, this
module resolves the situation ONCE, at construction:

  * exit code  -> synthesised. A clean command sequence is 0; an NgspiceError is 1.
    This is a faithful mapping, since NgSpiceServer already converts the library's
    command errors into NgspiceError.
  * stderr     -> must be capturable. If the backend cannot hand back its diagnostic
    text, `build_evaluator` raises at setup instead of returning an evaluator whose
    check 2 can never fire.

Failing loudly at startup is the point. A guard layer that silently drops two checks
on the fast path is the exact failure mode this project is trying to avoid.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Sequence

from eqrl.circuits.ctle import DesignVars, dv_to_params, netlist
from eqrl.guards import (
    ArtifactStore, GuardConfigError, GuardedEvaluator, OperatingPoint, RunArtifacts,
    SearchMonitor,
)
from eqrl.sim.ngspice_runner import NgspiceError
from eqrl.sim.probe import ProbeError, probe_operating_point
from eqrl.specs import DEFAULT_SPEC, Spec

#: Attribute names PySpice builds have used for the captured diagnostic stream.
_STDERR_ATTRS = ("stderr", "_stderr", "error_output", "_error_output")
_STDOUT_ATTRS = ("stdout", "_stdout", "output", "_ngspice_output")


def _read_stream(ng: Any, names: Sequence[str]) -> str | None:
    for name in names:
        val = getattr(ng, name, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            return "\n".join(str(x) for x in val)
        return str(val)
    return None


def _assert_capturable(server: Any) -> None:
    ng = getattr(server, "_ng", None)
    if ng is None:
        raise GuardConfigError(
            "the simulator backend exposes no ngspice handle, so Tier 1 check 2 "
            "(solver failure text) cannot run."
        )
    if _read_stream(ng, _STDERR_ATTRS) is None and _read_stream(ng, _STDOUT_ATTRS) is None:
        raise GuardConfigError(
            "this libngspice backend does not expose its diagnostic output, so Tier 1 "
            "check 2 (convergence / timestep / singular-matrix text) could never fire.\n"
            "Fix one of:\n"
            "  * register a send_char callback on NgSpiceShared and buffer it on the "
            "server, exposing it as `.stdout` / `.stderr`; or\n"
            "  * run validation passes through eqrl.sim.ngspice_runner (subprocess), "
            "which gives real streams and a real exit code.\n"
            "Refusing to build an evaluator whose check 2 is a no-op."
        )


def _adopt_server_data(server: Any, artifacts: RunArtifacts) -> None:
    """Copy the server's scratch .data files into the run's artifact directory."""
    from pathlib import Path

    src = getattr(server, "_dir", None)
    if src is None:
        return
    try:
        for f in sorted(Path(src).glob("*.data")):
            if f.is_file():
                artifacts.adopt(f)
    except OSError as e:                       # disk problems must not be silent
        artifacts.stderr += f"\n[guards] could not preserve simulator data: {e}\n"


def make_raw_eval(*, corner: str = "tt", fast: bool = True,
                  temp_c: float = 27.0, channel_loss_db: float = 12.0,
                  server_factory: Callable[[str], Any] | None = None,
                  with_operating_point: bool = True,
                  netlist_builder: Callable | None = None,
                  operating_point_probe: Callable | None = None):
    """Build the `raw_eval` callable GuardedEvaluator expects.

    Returns `(Measures, RunArtifacts, OperatingPoint | None)`.
    Experimental topologies must supply both their actual netlist builder and a
    probe covering every instantiated device. Default callers retain the CTLE
    builder and probe; all results still pass through the unchanged guard layer.
    """
    from eqrl.sim.server import get_server
    if (netlist_builder is None) != (operating_point_probe is None):
        raise GuardConfigError("experimental netlist and operating-point probe must be supplied together")
    factory = server_factory or get_server
    make_netlist = netlist_builder or netlist
    read_op = operating_point_probe or probe_operating_point

    def raw_eval(dv: DesignVars, *, artifacts: RunArtifacts, vdd: float = 1.8, **kw):
        srv = factory(corner)
        try:
            artifacts.netlist = make_netlist(dv, vdd=vdd, temp_c=temp_c, corner=corner,
                                        analysis="op", models="sky130")
        except FileNotFoundError as e:
            # A missing PDK is a broken setup, not a bad candidate. Raising
            # GuardConfigError (not OSError) means GuardedEvaluator does NOT convert
            # it into a per-candidate INVALID — it stops the run, which is correct:
            # a deck we cannot record is a run we cannot reproduce or audit.
            raise GuardConfigError(
                f"cannot build the SKY130 deck, so this run could not be recorded: {e}"
            ) from e
        artifacts.meta.update(corner=corner, temp_c=temp_c, fast=fast,
                              params=dv_to_params(dv))

        op: OperatingPoint | None = None
        try:
            if with_operating_point:
                # PRIME FIRST. probe_operating_point() runs `.op` on whatever parameters
                # the server currently holds; it does not know about `dv`. Without this
                # line Tier 2 read the wrong circuit entirely:
                #
                #   fresh server -> the deck's own .param defaults (i_tail = 2 mA, ...)
                #   warm server  -> the PREVIOUS candidate, primed by its measure_all
                #
                # and then compared that operating point against THIS candidate's
                # requested values. Measured consequence: the same design evaluated twice
                # in a row changed verdict 9 times in 14, because the second evaluation
                # probed the parameters the first one left behind. On a fresh server it
                # was stable but uniformly wrong -- T2.6 fired on almost everything,
                # comparing the default deck's ~1.87 mA delivered tail current against a
                # candidate asking for ~0.2 mA.
                #
                # Every Tier 2 verdict (saturation, tail current, rails) recorded before
                # this fix was a statement about a different design. T2.8 was unaffected;
                # it reads `dv` directly rather than the operating point.
                srv._prime(dv, vdd, temp_c)
                op = read_op(srv)
            from eqrl.sim import measures as _m
            # channel_loss_db must be threaded through. The eye metrics depend on it,
            # and Tier 4.15 checks them, so a guard built for one channel silently judges
            # every design against that channel however the caller varies the spec.
            # Measured consequence: policy_rollout reported 0 of 8 held-out specs solved
            # while direct measurement at each spec's own channel showed 4 of 6 designs
            # passing all eight checks. The verdicts were about a link nobody was asking
            # about.
            m = _m.measure_all(dv, vdd=vdd, temp_c=temp_c, corner=corner, fast=fast,
                               channel_loss_db=channel_loss_db,
                               srv=srv, _via_guards=True)
            artifacts.exit_code = 0
        except (NgspiceError, ProbeError) as e:
            # A failed command sequence is this backend's equivalent of a nonzero exit.
            artifacts.exit_code = 1
            artifacts.stderr += f"\n{type(e).__name__}: {e}\n"
            raise RuntimeError(str(e)) from e
        finally:
            ng = getattr(srv, "_ng", None)
            if ng is not None:
                artifacts.stdout += _read_stream(ng, _STDOUT_ATTRS) or ""
                artifacts.stderr += _read_stream(ng, _STDERR_ATTRS) or ""
            # NgspiceServer writes its .data files into a TemporaryDirectory that is
            # destroyed with the server. Copy them into this run's artifact directory
            # or the raw output is gone — which is both a lost audit trail and a
            # check-3 failure. Runs in `finally` so a failed run keeps its partial data.
            _adopt_server_data(srv, artifacts)

        if not m.ok:
            # measure_all's own failure flag. Surfaced as a run failure rather than
            # letting a zeroed Measures reach Tier 4, where most fields would look
            # merely 'implausible' instead of 'the simulation did not run'.
            artifacts.exit_code = 1
            raise RuntimeError("measure_all reported ok=False (simulation did not solve)")

        return m, artifacts, op

    return raw_eval


def build_evaluator(spec: Spec = DEFAULT_SPEC, *, corner: str = "tt", fast: bool = True,
                    channel_loss_db: float | None = None,
                    temp_c: float = 27.0,
                    store: ArtifactStore | None = None,
                    monitor: SearchMonitor | None = None,
                    server_factory: Callable[[str], Any] | None = None,
                    with_operating_point: bool = True,
                    check_backend: bool = True,
                    netlist_builder: Callable | None = None,
                    operating_point_probe: Callable | None = None) -> GuardedEvaluator:
    """Assemble a fully-wired GuardedEvaluator.

    Raises GuardConfigError at setup if the backend cannot support Tier 1 check 2 —
    see the module docstring. Pass check_backend=False only with a deliberate,
    written-down decision to run with that check disabled.
    netlist_builder and operating_point_probe support explicit experimental
    circuit adapters without replacing global functions or the default server.
    """
    from eqrl.sim.probe import make_corner_probe
    from eqrl.sim.server import get_server
    factory = server_factory or get_server

    if check_backend:
        _assert_capturable(factory(corner))

    return GuardedEvaluator(
        make_raw_eval(corner=corner, fast=fast, temp_c=temp_c,
                      channel_loss_db=(spec.channel_loss_db if channel_loss_db is None
                                       else channel_loss_db),
                      server_factory=factory,
                      with_operating_point=with_operating_point,
                      netlist_builder=netlist_builder,
                      operating_point_probe=operating_point_probe),
        spec,
        store or ArtifactStore(),
        monitor,
        require_operating_point=with_operating_point,
        corner_probe=make_corner_probe(factory),
    )


__all__ = ["build_evaluator", "make_raw_eval"]
