"""Local dashboard backend -- FastAPI, not Streamlit.

Serves the hand-built static frontend in dashboard/ and a small JSON API over the
same eqrl entry points the CLI uses: GuardedEvaluator.evaluate() (guard demo),
results/*.json (explorer), and eqrl.pipeline.design() (live pipeline -- inference
only, see that module's own docstring: it loads the frozen PPO checkpoint and never
trains or touches a reward, hyperparameter, design bound, or guard threshold).

Local-only dev tooling. Not part of RESULTS.md or the deployed site (web/, site/) --
nothing in src/ or scripts/smoke_test.py imports this file.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Same three-line ngspice/PDK bootstrap every eqrl/experiments/*.py module performs
# at import time. eqrl.sim.server only locates the PySpice DLL, not PDK_ROOT, so any
# entry point outside eqrl.experiments has to do this itself.
_NGSPICE = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NGSPICE / 'shim'};{_NGSPICE / 'Library' / 'bin'};{os.environ['PATH']}"

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from eqrl.circuits.ctle import DesignVars  # noqa: E402
from eqrl.evaluator import build_evaluator  # noqa: E402
from eqrl.pipeline import CLOSED_NOT_VERIFIED, FALLBACK, SOLVED, UNSOLVED, describe, design  # noqa: E402
from eqrl.specs import DEFAULT_SPEC, hard_pass  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
DASHBOARD_DIR = REPO_ROOT / "dashboard"

# key, human label, SI scale (human = SI / scale), decimals, input step
DESIGN_FIELDS: list[tuple[str, str, float, int, float]] = [
    ("w_in", "W_in (µm)", 1e-6, 3, 0.001),
    ("l_in", "L_in (µm)", 1e-6, 4, 0.0001),
    ("i_tail", "I_tail (µA)", 1e-6, 1, 0.1),
    ("rs", "R_s (kΩ)", 1e3, 3, 0.001),
    ("cs", "C_s (fF)", 1e-15, 1, 0.1),
    ("r_load", "R_load (Ω)", 1.0, 1, 0.1),
    ("w_dfe", "w_dfe (frac. UI)", 1.0, 3, 0.001),
]

CURATED: dict[str, dict[str, str]] = {
    "delivered_circuit.json": {
        "label": "Delivered circuit (flagship)",
        "blurb": "The final PPO → G3.2 design, independently re-verified across "
                 "all 45 PVT corners (5 process × 3 VDD × 3 temperature).",
    },
    "pass_vs_valid.json": {
        "label": "Pass vs. valid — the 86% finding",
        "blurb": "Of 28 designs that passed all 8 hard specs under CMA-ES, 24 (86%) "
                 "were rejected by the guard layer — mostly for not actually "
                 "being amplifiers.",
    },
    "target_tracking_clean40k.json": {
        "label": "Target tracking (retargeting correlation)",
        "blurb": "Does the achieved boost track the boost that was REQUESTED, or "
                 "just land anywhere in the legal 3–12 dB range? Correlation "
                 "across 26 held-out specs.",
    },
    "g32_selfcal_bench.json": {
        "label": "G3.2 self-calibration benchmark",
        "blurb": "Stage-2 constrained refinement, benchmarked over 10 specs.",
    },
    "final_comparison_seed23.json": {
        "label": "Final comparison (seed 23)",
        "blurb": "The full arm-by-arm search-method comparison this project's "
                 "headline numbers are drawn from. Large file.",
    },
    "reward_audit_clean40k.json": {
        "label": "Reward audit (clean 40k)",
        "blurb": "Reward-vs-metric sanity audit for the frozen seq_clean40k policy.",
    },
    "chance_baseline.json": {
        "label": "Chance baseline",
        "blurb": "What uniform-random sampling of the design box achieves, for "
                 "scale.",
    },
    "generalization.json": {
        "label": "Generalization",
        "blurb": "How the policy performs on specs it was not trained around.",
    },
}
FEATURED_ORDER = list(CURATED.keys())

app = FastAPI(title="SILQ Dashboard")

_evaluator_cache: dict[tuple[str, bool, float], Any] = {}


def get_evaluator(corner: str = "tt", fast: bool = True, channel_loss_db: float = 12.0):
    key = (corner, fast, channel_loss_db)
    if key not in _evaluator_cache:
        _evaluator_cache[key] = build_evaluator(
            DEFAULT_SPEC, corner=corner, fast=fast, channel_loss_db=channel_loss_db)
    return _evaluator_cache[key]


def _human_fields(si: dict[str, float]) -> dict[str, float]:
    return {key: round(si[key] / scale, dec) for key, _, scale, dec, _ in DESIGN_FIELDS}


def _to_si(human: dict[str, float]) -> dict[str, float]:
    return {key: human[key] * scale for key, _, scale, _, _ in DESIGN_FIELDS}


def _load_results_json(name: str) -> Any | None:
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _safe_results_path(name: str) -> Path:
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".json"):
        raise HTTPException(400, "invalid artifact name")
    p = (RESULTS_DIR / name).resolve()
    if p.parent != RESULTS_DIR.resolve() or not p.is_file():
        raise HTTPException(404, "artifact not found")
    return p


# -- Guard layer --------------------------------------------------------------

class EvaluateRequest(BaseModel):
    fields: dict[str, float]
    corner: str = "tt"
    vdd: float = 1.8


class CornerCheckRequest(BaseModel):
    corner: str = "tt"


@app.get("/api/guard/presets")
def guard_presets():
    presets = [{
        "id": "defaults", "label": "Defaults",
        "description": "eqrl.circuits.ctle.DesignVars() as written",
        "fields": _human_fields(vars(DesignVars())),
    }]

    delivered = _load_results_json("delivered_circuit.json")
    if delivered:
        presets.append({
            "id": "delivered", "label": "Delivered circuit",
            "description": "The flagship PPO → G3.2 design from "
                           "results/delivered_circuit.json",
            "fields": _human_fields(delivered["design"]),
        })

    pvv = _load_results_json("pass_vs_valid.json")
    if pvv:
        by_reason: dict[str, dict] = {}
        for d in pvv["designs"]:
            by_reason.setdefault(d["guard_reason"], d)
        if "T4.10_dc_gain_implausible" in by_reason:
            presets.append({
                "id": "dc_attenuator", "label": "Passed spec, DC attenuator",
                "description": "One of the 24/28 designs from results/pass_vs_valid.json "
                               "that passed all 8 hard specs and was still rejected "
                               "— here for negative DC gain.",
                "fields": _human_fields(by_reason["T4.10_dc_gain_implausible"]["design"]),
            })
        if "T2.5_mosfet_not_in_saturation" in by_reason:
            presets.append({
                "id": "out_of_saturation", "label": "Passed spec, out of saturation",
                "description": "Another of that same 24/28 — here for the input "
                               "pair leaving saturation.",
                "fields": _human_fields(by_reason["T2.5_mosfet_not_in_saturation"]["design"]),
            })

    return {
        "corners": ["tt", "ss", "ff", "sf", "fs"],
        "vdd": {"min": 1.71, "max": 1.89, "nominal": DEFAULT_SPEC.vdd_nominal,
                "tolerance": DEFAULT_SPEC.vdd_tolerance},
        "field_meta": [{"key": k, "label": label, "decimals": dec, "step": step}
                       for k, label, _, dec, step in DESIGN_FIELDS],
        "presets": presets,
    }


@app.post("/api/guard/evaluate")
def guard_evaluate(req: EvaluateRequest):
    dv = DesignVars(**_to_si(req.fields))
    evaluator = get_evaluator(corner=req.corner)
    verdict = evaluator.evaluate(dv, vdd=req.vdd)

    if verdict.is_valid:
        m = verdict.unwrap()
        ok, checks = hard_pass(m, DEFAULT_SPEC)
        return {
            "valid": True,
            "metrics": m.as_dict(),
            "hard_pass": {"ok": ok, "checks": checks},
            "run_id": verdict.run_id,
            "artifact_dir": str(verdict.artifact_dir),
        }
    return {
        "valid": False,
        "tier": verdict.tier,
        "check": verdict.check.value,
        "reason": verdict.reason,
        "violation": verdict.violation,
        "reached_tiers": [1, 2, 4],
        "run_id": verdict.run_id,
        "artifact_dir": str(verdict.artifact_dir),
    }


@app.post("/api/guard/verify-corners")
def guard_verify_corners(req: CornerCheckRequest):
    evaluator = get_evaluator(corner=req.corner)
    failure = evaluator.verify_corners(("tt", "ss"), force=True)
    if failure is None:
        return {"ok": True}
    return {"ok": False, "reason": failure.reason, "check": failure.check.value}


# -- Results explorer ----------------------------------------------------------

@app.get("/api/results")
def list_results():
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.name)
    rows = []
    for p in files:
        meta = CURATED.get(p.name, {})
        rows.append({
            "name": p.name,
            "featured": p.name in CURATED,
            "label": meta.get("label", p.name),
            "blurb": meta.get("blurb", ""),
            "size_kb": round(p.stat().st_size / 1024, 1),
        })
    rows.sort(key=lambda r: (FEATURED_ORDER.index(r["name"]) if r["name"] in FEATURED_ORDER
                             else len(FEATURED_ORDER), r["name"]))
    return rows


@app.get("/api/results/{name}")
def get_result(name: str):
    return FileResponse(_safe_results_path(name), media_type="application/json")


# -- Design time ----------------------------------------------------------------

@app.get("/api/design-time")
def design_time():
    """Assemble the design-time story from the recorded artifacts. Reads only; the numbers
    are whatever the result files say.

    The framing here is deliberate. The defensible claim is about EVALUATION COUNT at a
    matched budget -- which is what the problem statement actually asks for ("fewer search
    spaces, lowest design time"). The raw strict SOLVE COUNT is NOT a defensible claim:
    the preregistered chance-matched control puts PPO-restart's 16 strict solves against
    an expectation of 16.9 (p=0.75). That negative is returned alongside the positive
    result rather than dropped, because dropping it is the thing that would not survive a
    judge reading the JSON.
    """
    report = _load_results_json("final_report.json")
    sweep = _load_results_json("sweep_baseline.json")
    speed = _load_results_json("speedup.json")
    delivered = _load_results_json("delivered_circuit.json")
    if not (report and sweep and speed and delivered):
        raise HTTPException(status_code=404, detail="design-time artifacts are missing")

    order = ["PPO-restart", "PPO", "CMA-ES", "TPE", "RANDOM"]
    arms = []
    for name in order:
        a = report["arms"].get(name)
        if not a:
            continue
        vs = a.get("vs_ppo", {})
        arms.append({
            "name": name,
            "is_silq": name.startswith("PPO"),
            "loose": a["loose"], "strict": a["strict"], "n_specs": a["n_specs"],
            "loose_median": a["loose_median"], "strict_median": a["strict_median"],
            "sim_matched": a.get("sim_matched"),
            "vs_ppo_loose_p": (vs.get("loose") or {}).get("p"),
            "vs_ppo_strict_p": (vs.get("strict") or {}).get("p"),
            "chance": a.get("chance_matched"),
        })

    prov = delivered.get("provenance", {})
    per_eval = speed["resident_median_s"]
    return {
        "protocol": {
            "n_specs": report["n_specs"], "budget": report["budget"],
            "tol_db": report["tol"], "permutations": report["permutations"],
            "spec_seed": report["spec_seed"],
            "note": "held-out spec set; every arm gets the same evaluation budget.",
        },
        "delivered": {
            "total_evals": prov.get("total_evals"),
            "stage1_ppo_evals": (prov.get("stage1_ppo") or {}).get("n_evals"),
            "stage2_evals": prov.get("stage2_evals"),
            "measure_all_spent": prov.get("measure_all_spent"),
            "measure_all_budget": prov.get("measure_all_budget"),
            "unit_warning": "total_evals and measure_all are DIFFERENT units. Do not add or "
                            "compare them in one sentence -- see eqrl.experiments.simcount_audit.",
            "wall_clock_s_at_resident_rate": round(prov.get("total_evals", 0) * per_eval, 3),
        },
        "arms": arms,
        "sweep": {
            "per_axis": sweep["per_axis"], "total_points": sweep["total_points"],
            "elapsed_s": sweep["elapsed_s"], "elapsed_h": round(sweep["elapsed_s"] / 3600, 2),
            "s_per_point": sweep["s_per_point"],
            "valid": sweep["valid"], "spec_pass": sweep["spec_pass"],
            "first_success_index": sweep["first_success_index"],
            "extrapolation_hours": sweep["extrapolation_hours"],
            "note": "This is the COARSEST possible grid -- 4 points per axis. It is not a "
                    "near-optimal search; it passed spec on %d of %d points."
                    % (sweep["spec_pass"], sweep["total_points"]),
        },
        "per_eval": {
            "resident_median_s": speed["resident_median_s"],
            "subprocess_median_s": speed["subprocess_median_s"],
            "speedup": speed["speedup_per_evaluation"],
            "note": "Resident libngspice server vs one subprocess per evaluation. This is an "
                    "engineering speedup of the simulator loop, independent of the search.",
        },
        "caveats": [
            "PPO training is a ONE-TIME cost that must be amortized over future specs. "
            "The 6-evaluation figure is inference-time design cost, not total cost. The "
            "break-even curve is computed by eqrl.experiments.honest_benchmark.",
            "The strict solve COUNT is at the chance-matched line (see the chance column). "
            "The evaluation-count result is the claim; the solve count is not.",
            "The sweep baseline ran at TT only, like the optimization. Neither number is a "
            "PVT-robustness claim.",
        ],
    }


# -- Schematic ------------------------------------------------------------------

class SchematicRequest(BaseModel):
    fields: dict[str, float] | None = None      # human units, as the guard view uses
    title: str = "CTLE candidate"
    subtitle: str = ""


@app.get("/api/schematic")
def schematic_delivered():
    """The frozen delivered circuit, drawn from its own manifest."""
    from eqrl.schematic import render_delivered
    return Response(content=render_delivered(str(RESULTS_DIR / "delivered_circuit.json")),
                    media_type="image/svg+xml")


@app.post("/api/schematic")
def schematic_custom(req: SchematicRequest):
    """Draw an arbitrary candidate -- so the schematic tracks whatever is in the guard
    fields, rather than only ever showing the delivered design."""
    from eqrl.circuits.ctle import DesignVars
    from eqrl.schematic import render

    dv = DesignVars(**_to_si(req.fields)) if req.fields else DesignVars()
    return Response(content=render(dv, title=req.title, subtitle=req.subtitle),
                    media_type="image/svg+xml")


# -- Live pipeline --------------------------------------------------------------

class PipelineRunRequest(BaseModel):
    target_boost_db: float
    channel_loss_db: float = DEFAULT_SPEC.channel_loss_db
    spec_index: int = 0
    allow_fallback: bool = False


class ParseSpecRequest(BaseModel):
    text: str


@app.get("/api/pipeline/defaults")
def pipeline_defaults():
    return {
        "target_boost_db": DEFAULT_SPEC.target_boost_db,
        "boost_db_min": DEFAULT_SPEC.boost_db_min,
        "boost_db_max": DEFAULT_SPEC.boost_db_max,
        "channel_loss_db": DEFAULT_SPEC.channel_loss_db,
        "boost_tol_db": DEFAULT_SPEC.boost_tol_db,
        "statuses": {"solved": SOLVED, "closed_not_verified": CLOSED_NOT_VERIFIED,
                     "unsolved": UNSOLVED, "fallback": FALLBACK},
    }


@app.post("/api/pipeline/parse-spec")
def pipeline_parse_spec(req: ParseSpecRequest):
    """Natural language -> target fields, or an explicit refusal.

    An unparseable request MUST NOT come back as a spec. `parse_spec` returns a default
    Spec when it recognises nothing, and echoing those defaults into the sliders would
    show the user a confident 9 dB / 12 dB target that their words had no part in
    producing -- a fabricated interpretation. So this reports what was actually
    recognised and 422s when that is empty.
    """
    from eqrl.llm.spec_parser import parse_spec_verbose

    from eqrl.llm.spec_parser import UNITS

    r = parse_spec_verbose(req.text)
    if not r.understood:
        raise HTTPException(
            status_code=422,
            detail=(f"[{r.source}] nothing in that request was recognised as a design "
                    "spec, so no target was inferred and the fields were left alone. "
                    "State a target boost in dB (e.g. \"PCIe Gen2 CTLE, ~9 dB boost "
                    "over a 12 dB channel, under 12 mW\")."),
        )

    # Which parsed fields actually steer this run. `pipeline.design()` takes exactly two
    # arguments; every other spec field is enforced by hard_pass/the guard at its
    # DEFAULT_SPEC value, which this entry point cannot override. Reporting all thirteen
    # as "parsed" without that distinction would be the same lie as inventing a target:
    # the user would read a 5 mW power line back and assume the search honoured it.
    APPLIED = {"target_boost_db", "channel_loss_db"}
    rows = []
    for key in sorted(r.recognised):
        asked = r.recognised[key]
        run_value = asked if key in APPLIED else getattr(DEFAULT_SPEC, key, None)
        unit, scale = UNITS.get(key, ("", 1.0))
        rows.append({
            "field": key,
            "unit": unit,
            # `asked`/`run_value` stay in the Spec's own SI units; `*_disp` are the same
            # numbers scaled for the unit label, so no interface has to know that power
            # is stored in watts but shown in milliwatts.
            "asked": asked,
            "asked_disp": asked * scale,
            "applied": key in APPLIED,
            "run_value": run_value,
            "run_disp": None if run_value is None else run_value * scale,
            # A field the run cannot honour AND whose default disagrees with what was
            # asked for. This is the only case where the delivered circuit is scored
            # against something other than the request.
            "conflict": key not in APPLIED and run_value is not None
                        and abs(float(run_value) - float(asked)) > 1e-12,
        })
    return {
        "target_boost_db": r.spec.target_boost_db,
        "channel_loss_db": r.spec.channel_loss_db,
        "recognised": sorted(r.recognised),
        "fields": rows,
        "source": r.source,
        "applied_note": ("pipeline.design() is frozen to two inputs: target boost and "
                        "channel loss. Every other constraint is enforced at the "
                        "benchmarked default, not at the value you gave."),
    }


@app.post("/api/pipeline/run")
def pipeline_run(req: PipelineRunRequest):
    result = design(req.target_boost_db, req.channel_loss_db,
                    spec_index=req.spec_index, allow_fallback=req.allow_fallback)
    result["describe_text"] = describe(result)
    return json.loads(json.dumps(result, default=str))


# -- Static frontend -------------------------------------------------------------

@app.middleware("http")
async def _no_store_frontend(request, call_next):
    """Never let a browser cache the dashboard.

    This is a local demo server whose assets are edited between reloads. A cached app.js
    silently serves a stale UI -- which looks exactly like a broken feature, and is the
    kind of thing that would ruin a screen recording.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(DASHBOARD_DIR / "index.html")
