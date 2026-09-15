"""Measured nondominated circuit choices. Corpus metrics only propose candidates."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

OBJECTIVES = ("power_w", "noise_vrms", "area_mm2", "target_error_db")


def design_key(design):
    return json.dumps(design, sort_keys=True, separators=(",", ":"))


def objectives(item, target):
    m = item["verification"]["measures"]
    values = (m["power_w"], m["noise_vrms"], m["area_mm2"], abs(m["boost_db"] - target))
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Nonfinite Pareto objective")
    return values


def frontier(items, target):
    unique = {}
    for item in items:
        v = item.get("verification") or {}
        if v.get("passed") is True and v.get("guard_valid") is True:
            try:
                objectives(item, target)
            except (KeyError, TypeError, ValueError):
                continue
            unique.setdefault(design_key(item["design"]), item)
    rows = list(unique.values())
    values = [objectives(item, target) for item in rows]
    def dominates(a, b):
        return all(x <= y for x, y in zip(a,b)) and any(x < y for x,y in zip(a,b))
    return [item for i,item in enumerate(rows) if not any(dominates(v,values[i]) for j,v in enumerate(values) if i != j)]


def choose(items, target, cap=3):
    remaining = list(frontier(items, target))
    selected = []
    for axis, label in [(0,"Low power"),(1,"Low noise"),(2,"Small area")]:
        if not remaining or len(selected) >= cap:
            break
        item = min(remaining, key=lambda r:(objectives(r,target)[axis],objectives(r,target)[3]))
        remaining.remove(item)
        selected.append(dict(item, label=label, optimized_quantity=OBJECTIVES[axis], direction="minimize"))
    return selected


def proposals(result, spec, cap):
    from silq.experiments.fastest_hedge import load_fastest_assets
    from silq.circuits.ctle import decode_action, encode_action, DesignVars
    surrogate, _, _ = load_fastest_assets()
    order = np.argsort(abs(surrogate.Y[:,1] - spec.target_boost_db), kind="stable")
    order = [i for i in order if abs(surrogate.Y[i,1] - spec.target_boost_db) <= spec.boost_target_tol_db][:240]
    designs = [asdict(decode_action(surrogate.X[i])) for i in order]
    # Interleave boost-nearest seeds and distinct sizing extremes. These are proposal
    # heuristics, never substituted for measured power/noise/area or guard evidence.
    rankings = [designs, sorted(designs,key=lambda d:d["i_tail"]),
                sorted(designs,key=lambda d:-d["w_in"]),
                sorted(designs,key=lambda d:d["w_in"]*d["l_in"])]
    chosen = result.get("design")
    local = []
    if chosen:
        # Deliberately explore competing quantities around the already valid circuit.
        # These sizing moves are proposals only; every result is freshly guarded.
        for factor in (.92, 1.08, .84, 1.16):
            for field in ("i_tail", "w_in", "l_in", "r_load", "rs", "cs"):
                proposal = dict(chosen)
                proposal[field] *= factor
                proposal = asdict(decode_action(encode_action(DesignVars(**proposal))))
                proposal["w_dfe"] = chosen.get("w_dfe",0.0)
                local.append(proposal)
    seen = {design_key(chosen)} if chosen else set()
    out = []
    traces = [e["design"] for e in result.pop("_pareto_trace",[]) if e and e.get("design") and e.get("loose_pass")]
    for d in local[:18] + traces[:4] + [d for group in zip(*rankings) for d in group]:
        key = design_key(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(id="pareto_"+hashlib.sha256(key.encode()).hexdigest()[:12], origin="measured Pareto proposal", design=d))
        if len(out) >= cap:
            break
    return out


def verification_from_row(row):
    return dict(guard_valid=bool(row["guard_valid"]), passed=bool(row.get("passed")),
                checks=row.get("checks"), measures=row.get("measures"),
                abs_err_db=row.get("target_error_db"), guard_check=row.get("guard_rejection"),
                failing=[k for k,v in (row.get("checks") or {}).items() if not v],
                artifact_dir=row.get("artifact_dir"))


def attach(result, spec, pool, output, deadline, cap=12):
    candidates = proposals(result, spec, cap)
    items = []
    if result.get("design") and result.get("verification"):
        items.append(dict(design=result["design"], verification=result["verification"], origin="nominal winner"))
    tried = []
    if candidates and time.monotonic() < deadline:
        rows, counts = pool.evaluate("search", candidates, [("tt",spec.vdd_nominal,27.0)], spec, Path(output)/"pareto_raw", deadline)
        cost = result.setdefault("cost",{})
        for key, destination in [("measure_all","measure_all_total"),("analysis","spice_analyses_total")]:
            cost[destination] = cost.get(destination,0) + counts[key]
        cost["measure_all_alternatives"] = counts["measure_all"]
        cost["spice_analyses_alternatives"] = counts["analysis"]
        for candidate in candidates:
            v = verification_from_row(rows[candidate["id"]][0])
            tried.append(dict(id=candidate["id"], passed=v["passed"]))
            items.append(dict(design=candidate["design"], verification=v, origin=candidate["origin"]))
    result["pareto"] = dict(objectives=list(OBJECTIVES), scope="Nominal TT measurements at this request; PVT status is per circuit.",
        evaluated=len(tried), valid=sum(x["verification"].get("passed",False) for x in items),
        tried=tried, measured_items=items)
    finalize(result, spec)


def seed_primary(result, spec):
    result.setdefault("pareto", dict(objectives=list(OBJECTIVES), evaluated=0,
        measured_items=[], scope="Nominal TT tradeoffs for this request. Check PVT on each circuit separately."))
    finalize(result,spec)


def finalize(result, spec):
    block = result.get("pareto")
    if block is None:
        return
    items = list(block.pop("measured_items", block.get("frontier",[])))
    primary = None
    if result.get("design") and (result.get("verification") or {}).get("passed"):
        primary = dict(design=result["design"], verification=result["verification"], origin="primary pipeline circuit",
            label="Primary result", optimized_quantity="target_error_db", direction="minimize", is_primary=True)
        items.append(primary)
    front = frontier(items, spec.target_boost_db)
    block["frontier"] = front
    front_keys = {design_key(i["design"]) for i in front}
    # Published choices retain identities, order and measurements even when a later
    # batch dominates one. Mark that fact rather than moving the scientist's selection.
    published = list(block.get("items",[]))
    if not published and primary is not None:
        published.append(primary)
    known = {design_key(i["design"]) for i in published}
    for item in choose(front,spec.target_boost_db):
        key = design_key(item["design"])
        if key not in known:
            published.append(item);known.add(key)
    for item in published:
        key=design_key(item["design"])
        item["id"]="choice_"+hashlib.sha256(key.encode()).hexdigest()[:16]
        same=key==design_key(result.get("design"))
        item["is_primary"]=same
        item["on_frontier"]=key in front_keys
        # The primary inherits the run's own PVT verdict, so it has to inherit the GRID
        # that verdict came from too. Fastest certifies 3 corners; every other mode
        # certifies 45. Without these two fields the card falls back to its 45 default
        # and shows a three-corner pass as "45 / 45 corners" -- the one way a short grid
        # can do real damage. An alternative circuit has had no sweep of its own at all.
        run_pvt=result.get("pvt") or {}
        accepted=bool(same and run_pvt.get("accepted"))
        item["pvt"]=dict(accepted=accepted,
            status="verified" if accepted else "not_run_for_this_circuit",
            **(dict(corners_checked=run_pvt.get("corners_checked"),grid_label=run_pvt.get("grid_label"))
               if accepted and run_pvt.get("grid_label") else {}))
        item["objectives"]=dict(zip(OBJECTIVES,objectives(item,spec.target_boost_db)))
        from silq.circuits.ctle import DesignVars
        from silq.pvt_refinement import recorded_netlist
        item["netlist"]=recorded_netlist(DesignVars(**item["design"]),vdd=spec.vdd_nominal,temp_c=27,corner="tt")
    block.update(items=published,frontier_size=len(front),requested=3,complete=len(published)>=3,
        note="Distinct measured tradeoffs; earlier choices stay available for review." if len(published)>=3 else
             f"{len(published)} circuit(s) available; no duplicates or dominated fillers. More measurements may add choices.")
