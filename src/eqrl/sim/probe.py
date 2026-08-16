"""Operating-point extraction — the data Tier 2 and check 9 need.

Two halves, split deliberately:

  * PURE PARSING (no simulator). `parse_assignments` and `build_operating_point` turn
    ngspice text into an `OperatingPoint`. Fully unit-testable today, which matters
    because this is where a silent wrong answer would be cheapest to introduce.
  * SIMULATOR DRIVERS. `probe_operating_point` and `make_corner_probe` issue the
    commands. These cannot run until ngspice + SKY130 are installed.

SKY130 devices are subcircuits, so an instance `XM1 ... sky130_fd_pr__nfet_01v8` is
reached in ngspice as `@m.xm1.msky130_fd_pr__nfet_01v8[vds]` — the internal MOSFET
inside the subckt, not the X-instance itself.

A missing value is never defaulted. If a requested parameter does not come back, this
module raises, because a partial operating point read as complete is exactly how a
device in triode gets waved through.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from eqrl.guards import DeviceOP, OperatingPoint

NFET = "sky130_fd_pr__nfet_01v8"

#: Parameters pulled per device. All four query forms verified against ngspice 41 with
#: the SKY130 models loaded — `@m.xm1.msky130_fd_pr__nfet_01v8[vds]` and friends return
#: real values; `print i(<current source>)` does NOT work and is not used.
#: vds/vdsat gate the saturation check; id carries the delivered tail current, so it is
#: required too — without it the bias cannot be confirmed.
DEVICE_PARAMS: tuple[str, ...] = ("vds", "vdsat", "vgs", "vth", "id")
REQUIRED_DEVICE_PARAMS: tuple[str, ...] = ("vds", "vdsat", "id")

# ngspice prints scalars as `name = value`; values may be plain, exponential, or
# carry an engineering suffix. Vectors print as `name = ( v )` in some builds.
_ASSIGN = re.compile(
    r"^\s*([^\s=]+)\s*=\s*\(?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)?\s*$"
)


class ProbeError(RuntimeError):
    """The operating point could not be read. Never downgraded to a default."""


# ---------------------------------------------------------------------------
# Pure parsing
# ---------------------------------------------------------------------------

def device_ref(instance: str, model: str = NFET) -> str:
    """ngspice handle for the MOSFET inside a SKY130 subcircuit instance."""
    return f"@m.{instance.lower()}.m{model}"


def op_commands(instances: Sequence[str], nodes: Sequence[str],
                model: str = NFET) -> list[str]:
    """The `.control` body that dumps everything Tier 2 needs from one `.op`.

    No branch-current queries: ngspice rejects `print i(iref)` for a current source, and
    the tail current is read from the mirror devices' drain current instead.
    """
    cmds = ["op"]
    for inst in instances:
        ref = device_ref(inst, model)
        cmds += [f"print {ref}[{p}]" for p in DEVICE_PARAMS]
    cmds += [f"print v({n})" for n in nodes]
    return cmds


def parse_assignments(text: str) -> dict[str, float]:
    """Collect every `name = value` line ngspice printed.

    Keys are lowercased and stripped of whitespace. Later assignments win, which
    matches ngspice re-printing a value after a `reset`.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _ASSIGN.match(line)
        if not m:
            continue
        key, val = m.group(1).strip().lower(), m.group(2)
        try:
            out[key] = float(val)
        except ValueError:                       # pragma: no cover - regex guarantees it
            raise ProbeError(f"unparseable value on line: {line!r}") from None
    return out


def build_operating_point(assignments: Mapping[str, float],
                          instances: Sequence[str],
                          nodes: Sequence[str],
                          tail_devices: Sequence[str],
                          model: str = NFET) -> OperatingPoint:
    """Assemble an OperatingPoint, raising on anything missing.

    Deliberately strict: an absent vds or vdsat means check 5 cannot be evaluated,
    and a check that cannot be evaluated must not report success.
    """
    devices: list[DeviceOP] = []
    for inst in instances:
        ref = device_ref(inst, model).lower()
        vals: dict[str, float] = {}
        for p in DEVICE_PARAMS:
            key = f"{ref}[{p}]"
            if key in assignments:
                vals[p] = assignments[key]
            elif p in REQUIRED_DEVICE_PARAMS:
                raise ProbeError(
                    f"device '{inst}': required operating-point parameter '{p}' was not "
                    f"returned by ngspice (looked for {key!r}). Cannot confirm saturation."
                )
            else:
                vals[p] = float("nan")
        devices.append(DeviceOP(name=inst, **vals))

    node_v: dict[str, float] = {}
    for n in nodes:
        key = f"v({n.lower()})"
        if key not in assignments:
            raise ProbeError(f"node voltage {key!r} was not returned by ngspice")
        node_v[n] = assignments[key]

    # Tail current is the mirror devices' drain current. Reading it from the device
    # rather than a branch is not a convenience: ngspice rejects `print i(iref)` for a
    # current source, and the Itp/Itn branches were deleted when the mirror landed.
    by_name = {d.name: d for d in devices}
    tail_i: dict[str, float] = {}
    for t in tail_devices:
        d = by_name.get(t)
        if d is None:
            raise ProbeError(
                f"tail device '{t}' was not among the probed instances {list(by_name)} — "
                "the netlist and the probe disagree about what the circuit contains"
            )
        if not np.isfinite(d.id):
            raise ProbeError(
                f"tail device '{t}' returned no drain current, so the delivered bias "
                "cannot be confirmed"
            )
        tail_i[t] = float(d.id)

    return OperatingPoint(devices=tuple(devices), node_voltages=node_v,
                          tail_currents=tail_i)


# ---------------------------------------------------------------------------
# Simulator drivers (need ngspice + SKY130)
# ---------------------------------------------------------------------------

#: Every MOSFET in the testbench, not just the signal path.
#:
#: XMref/XMtp/XMtn are the current mirror that replaced the ideal tail sources. They are
#: included deliberately: the mirror output devices are the ones that actually run out of
#: headroom, and a saturation check blind to them would have reported a healthy circuit
#: while the mirror sat in triode delivering a third of its requested current.
CTLE_INSTANCES = ("XM1", "XM2", "XMref", "XMtp", "XMtn")

#: nbias is the mirror's gate node — if it collapses, the whole bias is gone.
CTLE_NODES = ("outp", "outn", "sp", "sn", "nbias")

#: Tail current now flows through the mirror devices, so it is read from their drain
#: current rather than a branch. `print i(iref)` is not accepted by ngspice for a
#: current source (verified against the simulator), and Itp/Itn no longer exist.
CTLE_TAIL_DEVICES = ("XMtp", "XMtn")


def probe_operating_point(server, *, instances: Sequence[str] = CTLE_INSTANCES,
                          nodes: Sequence[str] = CTLE_NODES,
                          tail_devices: Sequence[str] = CTLE_TAIL_DEVICES,
                          model: str = NFET) -> OperatingPoint:
    """Run `.op` on a resident NgspiceServer and return the parsed operating point.

    `server` must expose `_ng.exec_command(str)` and return its output — the shape
    NgspiceServer already has. Kept duck-typed so tests can inject a fake.
    """
    try:
        captured: list[str] = []
        for cmd in op_commands(instances, nodes, model):
            result = server._ng.exec_command(cmd)
            if result:
                captured.append("\n".join(result) if isinstance(result, (list, tuple))
                                else str(result))
    except AttributeError as e:
        raise ProbeError(f"server does not expose an ngspice handle: {e}") from e

    text = "\n".join(captured)
    if not text.strip():
        raise ProbeError(
            "ngspice returned no operating-point output. The .op did not solve, or this "
            "libngspice build does not echo `print` results — capture stdout explicitly."
        )
    return build_operating_point(parse_assignments(text), instances, nodes,
                                 tail_devices, model)


def make_corner_probe(server_factory, *, instance: str = "XM1", param: str = "vth",
                      model: str = NFET):
    """Build the check-9 probe: threshold voltage of a fixed device, per corner.

    vth at a fixed bias genuinely differs between tt and ss in SKY130, so a corner
    include that silently failed shows up as two identical readings.

    `server_factory(corner)` must return a server bound to that corner —
    `eqrl.sim.server.get_server` satisfies this.
    """
    ref = device_ref(instance, model)

    def probe(corner: str) -> float:
        srv = server_factory(corner)
        out = srv._ng.exec_command("op")
        val = srv._ng.exec_command(f"print {ref}[{param}]")
        parsed = parse_assignments("\n".join(str(x) for x in (out, val) if x))
        key = f"{ref.lower()}[{param}]"
        if key not in parsed:
            raise ProbeError(
                f"corner '{corner}': could not read {key!r}. Check 9 cannot run, so the "
                "corner include cannot be verified."
            )
        return parsed[key]

    return probe
