#!/usr/bin/env python3
"""Check that a fresh clone can actually run this repo.

    python scripts/smoke_test.py

Prints one PASS / FAIL / WARN line per check and exits non-zero if anything failed.
Target runtime is well under 30 s: exactly one SPICE evaluation is performed, with
`fast=True`, which skips the transient-HD3 and `.noise` analyses.

This exists because the ways a clone breaks are boring and fatal: a dependency missing
from requirements.txt, PDK_ROOT unset, the GUI ngspice build on PATH instead of the
console one, a documented command pointing at a file nobody ever produced. Each of those
looks like "the project is broken" to somebody with five minutes to spend on it.

Nothing here imports a third-party package at module level, so the import check itself
can report a missing dependency instead of crashing on it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))          # same trick as conftest.py

_FAILED: list[str] = []
_WARNED: list[str] = []


def _emit(status: str, name: str, detail: str = "") -> None:
    line = f"[{status:4}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line, flush=True)


def ok(name: str, detail: str = "") -> None:
    _emit("PASS", name, detail)


def fail(name: str, detail: str = "") -> None:
    _emit("FAIL", name, detail)
    _FAILED.append(name)


def warn(name: str, detail: str = "") -> None:
    """Something a judge should know about but which does not stop the repo running."""
    _emit("WARN", name, detail)
    _WARNED.append(name)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)), flush=True)


# ---------------------------------------------------------------------------
# 1. third-party imports
# ---------------------------------------------------------------------------

#: import name -> what breaks without it
REQUIRED_IMPORTS = {
    "numpy": "everything",
    "matplotlib": "experiments/*.py plots",
    "gymnasium": "the RL environment",
    "stable_baselines3": "PPO/DDPG training",
    "torch": "stable-baselines3 policies",
    "optuna": "the Bayesian baseline",
    "cma": "the CMA-ES baseline (honest_benchmark)",
    "PySpice": "the resident libngspice server -- the whole speed argument",
    "pytest": "the test suite",
}

#: not required: absence is a documented, degraded-but-working path
OPTIONAL_IMPORTS = {
    "anthropic": "LLM spec parser falls back to a keyword heuristic",
}


def check_imports() -> None:
    section("third-party imports")
    import importlib

    for mod, why in REQUIRED_IMPORTS.items():
        try:
            m = importlib.import_module(mod)
        except Exception as exc:                       # noqa: BLE001 - report anything
            fail(f"import {mod}", f"{type(exc).__name__}: {exc} (needed for: {why})")
            continue
        ver = getattr(m, "__version__", "?")
        ok(f"import {mod}", str(ver))

    for mod, why in OPTIONAL_IMPORTS.items():
        try:
            importlib.import_module(mod)
            ok(f"import {mod} (optional)")
        except Exception:                              # noqa: BLE001
            _emit("SKIP", f"import {mod} (optional)", f"not installed; {why}")


def check_silq_imports() -> None:
    section("project imports")
    import importlib

    for mod in ("silq.specs", "silq.circuits.ctle", "silq.circuits.pdk",
                "silq.guards", "silq.evaluator", "silq.sim.measures",
                "silq.sim.ngspice_runner", "silq.sim.server", "silq.envs.equalizer_env",
                "silq.envs.sequential_env", "silq.llm.spec_parser"):
        try:
            importlib.import_module(mod)
            ok(f"import {mod}")
        except Exception as exc:                       # noqa: BLE001
            fail(f"import {mod}", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 2. PDK
# ---------------------------------------------------------------------------

def check_pdk() -> bool:
    """True if the SKY130 corner library is usable, so the sim check can be skipped."""
    section("SKY130 PDK")
    if not os.environ.get("PDK_ROOT"):
        warn("PDK_ROOT set", "unset; pdk.py will fall back to ~/pdk")
    else:
        ok("PDK_ROOT set", os.environ["PDK_ROOT"])

    try:
        from silq.circuits import pdk
    except Exception as exc:                           # noqa: BLE001
        fail("import silq.circuits.pdk", str(exc))
        return False

    try:
        lib = pdk.sky130_lib()
    except FileNotFoundError as exc:
        fail("sky130.lib.spice resolves", str(exc))
        return False
    ok("sky130.lib.spice resolves", str(lib))

    # Every corner the PVT grid uses must be a real section in that file, and the model
    # files each section .includes must be on disk. A partial PDK extraction (common:
    # the tarballs are per-library) resolves the .lib but explodes inside ngspice.
    text = lib.read_text(errors="ignore")
    missing_sections = [c for c in pdk.CORNER_SECTION.values()
                        if f".lib {c}" not in text]
    if missing_sections:
        fail("corner sections present", f"missing: {', '.join(missing_sections)}")
    else:
        ok("corner sections present", " ".join(sorted(set(pdk.CORNER_SECTION.values()))))

    includes = [ln.split('"')[1] for ln in text.splitlines()
                if ln.strip().startswith(".include") and '"' in ln]
    absent = sorted({inc for inc in includes if not (lib.parent / inc).exists()})
    if absent:
        fail("model files .included by the corners exist",
             f"{len(absent)} missing, e.g. {absent[0]}")
    else:
        ok("model files .included by the corners exist", f"{len(set(includes))} files")

    for corner in ("tt", "ss", "ff", "sf", "fs"):
        pdk.lib_include(corner)          # raises KeyError on an unknown corner
    ok("pdk.lib_include() builds a .lib line for all 5 corners")
    return True


# ---------------------------------------------------------------------------
# 3. ngspice, both paths
# ---------------------------------------------------------------------------

def check_ngspice_exe() -> None:
    section("ngspice executable (subprocess path)")
    exe = shutil.which("ngspice")
    if exe is None:
        fail("ngspice on PATH",
             "not found. macOS: brew install ngspice. Windows: see SETUP.md -- and note "
             "the conda-forge ngspice.exe is the GUI build, which hangs under -b; the "
             "shim must expose ngspice_con.exe as ngspice.")
        return
    try:
        # -v exits 1 on some builds after printing the banner, so the return code is
        # not the signal; the banner text is.
        p = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=30)
        banner = (p.stdout + p.stderr).strip().splitlines()
        ver = next((ln for ln in banner if "ngspice" in ln.lower()), banner[0] if banner else "")
    except Exception as exc:                           # noqa: BLE001
        fail("ngspice -v runs", f"{type(exc).__name__}: {exc}")
        return
    ok("ngspice on PATH", f"{exe} :: {ver.strip()[:70]}")


def check_libngspice() -> bool:
    """True if the resident server can start."""
    section("libngspice (resident server path -- the fast one)")
    try:
        from silq.sim.server import _tolerant_shared_instance
    except Exception as exc:                           # noqa: BLE001
        fail("import silq.sim.server", str(exc))
        return False
    try:
        # Claim shared-instance id 0 through the project's own tolerant subclass. Probing
        # with NgSpiceShared.new_instance() instead would hand id 0 to the intolerant base
        # class for the rest of the process -- see the comment in tests/test_server_resident.py.
        _tolerant_shared_instance()
    except Exception as exc:                           # noqa: BLE001
        fail("libngspice loads",
             f"{type(exc).__name__}: {exc}. Set NGSPICE_LIBRARY_PATH (Windows: "
             r"...\Library\bin\ngspice{}.dll -- the {} is a PySpice placeholder, keep it).")
        return False
    ok("libngspice loads via PySpice NgSpiceShared")
    return True


# ---------------------------------------------------------------------------
# 4. one end-to-end evaluation
# ---------------------------------------------------------------------------

def check_one_evaluation() -> None:
    """Exactly one measure_all(), at fast=True. This is the only simulator call here."""
    section("end-to-end evaluation (1 SPICE call)")
    import math
    import time

    try:
        from silq.circuits.ctle import DesignVars
        from silq.sim.measures import measure_all
    except Exception as exc:                           # noqa: BLE001
        fail("import the measurement path", str(exc))
        return

    dv = DesignVars()          # defaults: 20 um / 0.15 um, 2 mA, 1 k, 200 fF, 1 k
    t0 = time.perf_counter()
    try:
        m = measure_all(dv, fast=True)
    except Exception as exc:                           # noqa: BLE001
        fail("measure_all(DesignVars(), fast=True)", f"{type(exc).__name__}: {exc}")
        return
    dt = time.perf_counter() - t0

    if not m.ok:
        fail("measure_all returns ok=True",
             "the simulator ran but the design did not solve (Measures.ok=False)")
        return

    # fast=True substitutes constants for hd3_db and noise_vrms, so those two are not
    # evidence of anything here; they are still checked for finiteness.
    bad = [k for k, v in m.as_dict().items()
           if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        fail("all metrics finite", f"non-finite: {', '.join(bad)}")
        return

    ok("measure_all(fast=True) returns finite metrics", f"{dt * 1e3:.0f} ms")
    print(f"       boost {m.boost_db:.2f} dB @ {m.peak_freq_ghz:.2f} GHz, "
          f"dc {m.dc_gain_db:.2f} dB, power {m.power_w * 1e3:.2f} mW, "
          f"area {m.area_mm2:.4f} mm2, eye {m.eye_h_ui:.2f} UI / {m.eye_v_mv:.0f} mV")

    if abs(m.boost_db) < 1e-9 and abs(m.dc_gain_db) < 1e-9:
        fail("evaluation produced a non-trivial result",
             "every AC metric is exactly zero, which means the sweep returned nothing")


# ---------------------------------------------------------------------------
# 5. file paths the documented CLIs default to
# ---------------------------------------------------------------------------

#: Input paths a CLI defaults to that are *produced by another command*, not shipped.
#: A missing entry here is the failure this check exists to catch: a default pointing at
#: a file that neither exists nor has a documented producer is a five-minute dead end
#: for whoever clones the repo.
PRODUCED_BY = {
    "results/seq_agent.zip":
        "python -m silq.agents.train_sequential --guarded  (writes this path by default; "
        "long -- real SPICE on every step)",
}

#: argparse dests that name an output, not an input. These only need a writable parent.
OUTPUT_DESTS = {"--out", "--outdir", "--outfile", "--o"}

CLI_FILES = [
    "src/silq/solve.py",
    "src/silq/experiments/honest_benchmark.py",
    "src/silq/experiments/generalization.py",
    "src/silq/experiments/compare.py",
]


def check_cli_defaults() -> None:
    """Scan the CLIs for string defaults that look like paths and check each one.

    Scanned rather than imported: the parsers are built inside main(), and importing
    these modules drags in stable-baselines3 for no benefit here.
    """
    section("default paths in the documented CLIs")
    import re

    pat = re.compile(
        r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*?default\s*=\s*"([^"]+)"',
        re.DOTALL,
    )
    seen = 0
    for rel in CLI_FILES:
        path = ROOT / rel
        if not path.exists():
            fail(f"{rel} exists", "documented CLI module is missing")
            continue
        src = path.read_text(errors="ignore")
        for flag, default in pat.findall(src):
            # only interested in defaults that name a path
            if (flag not in OUTPUT_DESTS and "/" not in default
                    and not default.endswith((".json", ".zip"))):
                continue
            seen += 1
            target = ROOT / default
            label = f"{rel} {flag}={default}"
            if flag in OUTPUT_DESTS:
                parent = target if target.suffix == "" else target.parent
                if parent.is_dir():
                    ok(label, "output location exists")
                else:
                    fail(label, f"output directory {parent} does not exist")
            elif target.exists():
                ok(label, "present")
            elif default in PRODUCED_BY:
                warn(label, f"absent by design; produce it with: {PRODUCED_BY[default]}")
            else:
                fail(label,
                     "default input file does not exist and no producing command is "
                     "documented -- either ship it, change the default, or add it to "
                     "PRODUCED_BY in this script")
    if seen == 0:
        fail("CLI default scan", "found no path-like defaults; the regex has drifted")

    # solve.py --model is required=True with no default. Assert that stays true, because
    # a default here would silently point at a checkpoint that does not load.
    solve = (ROOT / "src/silq/solve.py").read_text(errors="ignore")
    if '"--model", required=True' in solve.replace("\n", " ").replace("  ", " "):
        ok("src/silq/solve.py --model has no default", "must be passed explicitly")
    else:
        warn("src/silq/solve.py --model has no default",
             "could not confirm by inspection; check it by hand")


# ---------------------------------------------------------------------------

def main() -> int:
    print("silq smoke test")
    print(f"python {sys.version.split()[0]} :: {sys.executable}")
    print(f"repo   {ROOT}")

    check_imports()
    check_silq_imports()
    have_pdk = check_pdk()
    check_ngspice_exe()
    have_lib = check_libngspice()

    if have_pdk and have_lib:
        check_one_evaluation()
    else:
        section("end-to-end evaluation (1 SPICE call)")
        _emit("SKIP", "measure_all(fast=True)",
              "needs both the SKY130 PDK and libngspice; see the failures above")

    check_cli_defaults()

    print()
    if _FAILED:
        print(f"FAILED ({len(_FAILED)}): " + ", ".join(_FAILED))
        return 1
    if _WARNED:
        print(f"OK with {len(_WARNED)} warning(s): " + ", ".join(_WARNED))
        return 0
    print("OK -- all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
