"""What is the resident simulator actually worth? Measure it, commit the number.

The repo claims "a measured 350x speedup" in five places. That figure compares an AC
evaluation on the resident server against a full ngspice process launch, which includes
starting the binary and re-parsing the SKY130 model library -- work the resident server
does once at startup and the subprocess path repeats every time. It is a real cost, but
quoting it as the speedup of an *evaluation* overstates what the architecture buys during
a search, where the interesting comparison is per-candidate throughput.

This measures both honestly and writes them down, so neither the claim nor the correction
is unsourced:

  resident  : time per AC evaluation on a warm NgspiceServer (alterparam + reset + ac)
  subprocess: time per AC evaluation launching ngspice per candidate, the way a naive
              driver would, including model reload

Run it and quote the number it prints. Do not quote a ratio this script did not produce.
"""
import json
import os
import statistics
import time
from pathlib import Path

P = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{P/'shim'};{P/'Library'/'bin'};{os.environ['PATH']}"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import numpy as np

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.circuits.pdk import lib_include            # noqa: F401  (import check)
from eqrl.sim.ngspice_runner import run as sub_run
from eqrl.sim.server import NgspiceServer

N = int(os.environ.get("SPEEDUP_N", "12"))
rng = np.random.default_rng(0)


def designs(n):
    """Distinct, comfortably-biased candidates, so neither path is timed on a design
    that fails to converge (which would make the faster path look better for the wrong
    reason)."""
    out = []
    for _ in range(n):
        out.append(DesignVars(
            w_in=float(rng.uniform(5e-6, 40e-6)), l_in=0.3e-6,
            i_tail=float(rng.uniform(0.1e-3, 0.4e-3)),
            rs=float(rng.uniform(500, 4000)), cs=float(rng.uniform(50e-15, 800e-15)),
            r_load=float(rng.uniform(400, 2000))))
    return out


AC_CTRL = "ac dec 30 1e6 1e10\nlet vdb = db(v(outp)-v(outn))\nwrdata $OUT vdb"

print(f"Timing {N} AC evaluations on each path.\n")

srv = NgspiceServer("tt")
srv.ac(designs(1)[0])                                   # warm up, excluded
res_t = []
for dv in designs(N):
    t0 = time.perf_counter()
    srv.ac(dv)
    res_t.append(time.perf_counter() - t0)

sub_t = []
for dv in designs(N):
    deck = netlist(dv, corner="tt", analysis="none", models="sky130").replace(".end", "")
    t0 = time.perf_counter()
    try:
        sub_run(deck, control=AC_CTRL, require_output=False, timeout=120)
    except Exception as e:
        print(f"  subprocess eval failed ({type(e).__name__}), skipping")
        continue
    sub_t.append(time.perf_counter() - t0)

r_med, s_med = statistics.median(res_t), statistics.median(sub_t)
print(f"resident   : median {r_med*1e3:8.1f} ms  (n={len(res_t)}, "
      f"min {min(res_t)*1e3:.1f}, max {max(res_t)*1e3:.1f})")
print(f"subprocess : median {s_med*1e3:8.1f} ms  (n={len(sub_t)}, "
      f"min {min(sub_t)*1e3:.1f}, max {max(sub_t)*1e3:.1f})")
print(f"\nspeedup per evaluation: {s_med/r_med:.1f}x")
print("\nThis is the number to quote. It is the per-candidate throughput gain during a\n"
      "search, which is what the architecture is for.")

Path("results").mkdir(exist_ok=True)
Path("results/speedup.json").write_text(json.dumps({
    "n": N,
    "resident_median_s": r_med, "subprocess_median_s": s_med,
    "resident_all_s": res_t, "subprocess_all_s": sub_t,
    "speedup_per_evaluation": round(s_med / r_med, 2),
}, indent=2))
print("\nwrote results/speedup.json")
