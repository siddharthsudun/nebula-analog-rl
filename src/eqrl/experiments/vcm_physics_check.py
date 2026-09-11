"""Run the existing SKY130 physics tests with BOTH VCM entry points set.

At most 48 AC subprocess evaluations total (24 per VCM) and 300 seconds.
The tests also run their existing operating-point headroom probes. These are
direction checks, not full HD3/noise characterization or PVT acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from eqrl.experiments.tail_mirror_pilot import ROOT, fingerprints, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, choices=(0.68, 0.72))
    args = parser.parse_args()
    if args.ratio is not None:
        import pytest
        from eqrl.circuits import ctle
        from eqrl.sim import ngspice_runner
        ctle.VCM_VDD_RATIO = args.ratio
        ctle.VCM = 1.8 * args.ratio
        original_run = ngspice_runner.run
        count = 0

        def counted(deck, **kw):
            nonlocal count
            if count >= 24:
                raise RuntimeError("AC evaluation limit reached")
            count += 1
            expected = f"Vcm cm 0 {ctle.VCM}\n"
            if expected not in deck:
                raise RuntimeError("test deck did not receive the requested VCM")
            return original_run(deck, **kw)

        ngspice_runner.run = counted
        label = str(args.ratio)
        code = pytest.main([
            str(ROOT / "tests/test_monotonicity.py") + "::TestPipelineSky130",
            "-q", "--tb=short", f"--junitxml={args.output / (label + '.xml')}",
            f"--basetemp={args.output / ('pytest-' + label)}",
        ])
        write_json(args.output / (label + ".json"),
                   {"ratio": args.ratio, "ac_evaluations": count, "pytest_exit_code": int(code)})
        raise SystemExit(code)
    args.output.mkdir(parents=True, exist_ok=False)
    before = fingerprints()
    started = time.monotonic()
    rows = []
    for ratio in (0.72, 0.68):
        remaining = 300 - (time.monotonic() - started)
        if remaining <= 0:
            break
        with (args.output / f"{ratio}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([
                sys.executable, "-m", "eqrl.experiments.vcm_physics_check",
                "--output", str(args.output.resolve()), "--ratio", str(ratio),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # Kill this worker and any current ngspice subprocess on Windows.
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   capture_output=True, check=False)
                else:
                    proc.kill()
                proc.wait()
                code = "wall_limit"
        rows.append({"ratio": ratio, "returncode": code})
        print(json.dumps(rows[-1]), flush=True)
    write_json(args.output / "manifest.json", {
        "rows": rows, "seconds": time.monotonic() - started,
        "ac_limit": 48, "wall_limit_seconds": 300,
        "frozen_before": before, "frozen_after": fingerprints(),
        "frozen_unchanged": before == fingerprints(),
    })
    if len(rows) != 2 or any(r["returncode"] != 0 for r in rows):
        raise SystemExit("physics checks did not all pass; see saved logs")


if __name__ == "__main__":
    main()
