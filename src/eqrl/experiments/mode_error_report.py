"""New statistics beside the frozen reports, never a replacement for them.

Cluster bootstrap samples specification IDs with all modes/policy seeds kept paired.
Intervals are conditional on the three supplied checkpoints, NOT population-level
training-seed uncertainty. The historical achieved-boost pool is resampled too for
the parametric chance-reference interval. No simulator imports are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eqrl.experiments.audit_support import write_new


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not a.size:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None, "max": None}
    q1, median, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    return {"n": int(a.size), "median": float(median), "q1": float(q1),
            "q3": float(q3), "iqr": float(q3 - q1), "max": float(a.max())}


def paired_cluster_interval(values, seed=202609092, draws=5000):
    """One row per spec, one column per checkpoint. Keep columns together."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not np.all(np.isfinite(a)):
        raise ValueError("non-empty finite spec x checkpoint matrix required")
    rng = np.random.default_rng(seed)
    sims = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(draws)])
    return {"estimate": float(a.mean()), "ci95": np.quantile(sims, [.025, .975]).tolist(),
            "lower_one_sided_bonferroni6": float(np.quantile(sims, .05 / 6)),
            "n_spec_clusters": len(a), "n_checkpoints": a.shape[1],
            "scope": "conditional on supplied checkpoints; specification bootstrap"}


def conditional_chance(targets, budgets, pool, *, seed=202609093, draws=5000, tol=1.5):
    """Descriptive independent-attempt reference, NOT a permutation test.

    k is the observed total deployment measure_all budget for a mode/spec. This
    explicitly charges one hypothetical *already feasible* draw per full measurement,
    not frozen final_report's min(distinct rounded boosts,n_loose_pass) convention.
    Both are assumption-dependent references; neither proves equivalence to chance.
    """
    t, k, b = (np.asarray(v, dtype=float) for v in (targets, budgets, pool))
    if t.ndim != 1 or k.shape != t.shape or b.ndim != 1 or not len(t) or not len(b):
        raise ValueError("non-empty aligned targets/budgets and a boost pool required")
    if not all(np.all(np.isfinite(v)) for v in (t, k, b)) or np.any(k < 0):
        raise ValueError("finite targets/pool and nonnegative budgets required")
    def expectation(ts, ks, bs):
        q = (np.abs(ts[:, None] - bs[None, :]) <= tol).mean(axis=1)
        return float(np.mean(1 - (1 - q) ** ks))
    rng = np.random.default_rng(seed)
    simulations = []
    for _ in range(draws):
        ix = rng.integers(0, len(t), len(t))
        sampled_pool = b[rng.integers(0, len(b), len(b))]
        simulations.append(expectation(t[ix], k[ix], sampled_pool))
    return {"expected_rate": expectation(t, k, b),
            "ci95": np.quantile(simulations, [.025, .975]).tolist(),
            "n_specs": len(t), "pool_size": len(b), "tol_db": tol,
            "budget_unit": "observed deployment measure_all calls, includes internal verification",
            "assumption": "independent already-feasible draws; not an empirical optimizer baseline",
            "not_equivalence_test": True}


def flatten(row):
    search = row["search"]
    job = search["job"]
    result = search.get("result") or {}
    external = row.get("independent_verification") or {}
    verified = external.get("result") or {}
    measures = verified.get("measures")
    delivered = result.get("design") is not None
    internal_cost = result.get("cost") or {}
    return {**{k: job[k] for k in ("mode", "model", "condition", "spec_index", "target", "channel")},
            "delivered": delivered,
            "verification_failed": delivered and not bool(verified.get("passed")),
            "verification_unavailable": delivered and not external.get("complete", False),
            "strict": delivered and bool(verified.get("passed")),
            "abs_target_error_db": abs(measures["boost_db"] - job["target"]) if measures else None,
            "optimizer_evals": (internal_cost["optimizer_evals"] +
                                internal_cost.get("optimizer_evals_prior_attempts", 0))
                               if "optimizer_evals" in internal_cost else None,
            "measure_all": (search.get("actual_cost") or {}).get("measure_all"),
            "spice_analyses": (search.get("actual_cost") or {}).get("analysis"),
            "external_verification_cost": external.get("actual_cost"),
            "search_error": search.get("error"), "budget_ceiling": job["budget"]}


def report(data, pool):
    rows = [flatten(row) for row in data["rows"]]
    manifest = data["manifest"]
    modes, models = manifest["modes"], manifest["models"]
    index = {(r["mode"], r["model"], r["condition"], r["spec_index"]): r for r in rows}
    out = {"source_scope": "new prospective protocol; no frozen result is revised",
           "input_drift": data["input_drift"], "per_spec": rows, "summaries": [],
           "paired_controls": [], "training_costs": manifest["training"]}
    for mode in modes:
        for model in models:
            group = [r for r in rows if r["mode"] == mode and r["model"] == model
                     and r["condition"] == "informed"]
            delivered = sum(r["delivered"] for r in group)
            failures = sum(r["verification_failed"] for r in group)
            known = [r for r in group if r["measure_all"] is not None]
            train = next(t for t in manifest["training"] if Path(t["model"]).name == Path(model).name)
            deployment = sum(r["measure_all"] for r in known)
            ext_cost = sum((r["external_verification_cost"] or {}).get("measure_all", 0) for r in group)
            out["summaries"].append({"mode": mode, "model": model, "n_specs": len(group),
                "target_error_db": distribution([r["abs_target_error_db"] for r in group]),
                "target_error_undefined": sum(r["abs_target_error_db"] is None for r in group),
                "delivered": delivered, "verification_failures": failures,
                "verification_failure_rate": failures / delivered if delivered else None,
                "verification_unavailable": sum(r["verification_unavailable"] for r in group),
                "optimizer_evals_per_spec": distribution([r["optimizer_evals"] for r in group]),
                "measure_all_per_spec": distribution([r["measure_all"] for r in group]),
                "chance": conditional_chance([r["target"] for r in known],
                                              [r["measure_all"] for r in known], pool) if known else None,
                "chance_unknown_budget_exclusions": len(group) - len(known),
                "cost": {"deployment_measure_all_known": deployment,
                         "independent_audit_measure_all_known": ext_cost,
                         "historical_policy_training_n_sims": train["n_sims"],
                         "deployment_plus_policy_training_known": deployment + train["n_sims"],
                         "corpus_creation_cost": None,
                         "total_cost_complete": False,
                         "note": "fastest uses no PPO; training charged only as system investment sensitivity; corpus cost unaccounted"}})
        for condition in ("target_blind_8db", "target_shuffled"):
            differences, excluded = [], []
            for i in range(manifest["n_specs"]):
                block = []
                for model in models:
                    a = index.get((mode, model, "informed", i))
                    b = index.get((mode, model, condition, i))
                    if a is None or b is None or a["measure_all"] is None or b["measure_all"] is None:
                        break
                    block.append(float(a["strict"]) - float(b["strict"]))
                if len(block) == len(models):
                    differences.append(block)
                else:
                    excluded.append(i)
            interval = paired_cluster_interval(differences) if differences else None
            out["paired_controls"].append({"mode": mode, "control": condition,
                "strict_rate_difference": interval, "excluded_specs": excluded,
                "supports_advantage": bool(interval and not excluded and not data["input_drift"]
                                            and interval["lower_one_sided_bonferroni6"] > 0),
                "not_equivalence_test": True})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pool", default="results/target_tracking_clean40k.json")
    a = p.parse_args()
    data = json.loads(Path(a.input).read_text(encoding="utf-8"))
    pool = [r["boost"] for r in json.loads(Path(a.pool).read_text(encoding="utf-8"))
            if r.get("boost") is not None]
    write_new(a.out, report(data, pool))


if __name__ == "__main__":
    main()
