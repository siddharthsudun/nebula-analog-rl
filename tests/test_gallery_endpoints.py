"""Contracts for the candidate-gallery endpoints.

The gallery is the only simulator-backed route added for the multi-candidate view, so it
carries the same obligations as `/api/guard/evaluate`: it must take the ONE module-level
`_run_lock` without blocking, refuse a second concurrent caller with the structured 409,
and release the lock on every exit path including `SearchHalted`, which derives from
`BaseException` and therefore slips past a bare `except Exception`.

These call the route functions directly rather than through FastAPI's lifespan: the
lifespan starts a real background warm-up against ngspice, while everything here must
prove request-level behaviour against fakes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import server as dashboard_server
from silq.guards import Check, SearchHalted


def _catalog(n: int) -> dict[str, dict]:
    """A minimal stand-in for the artifact-backed catalog, with n distinct candidates."""
    return {
        f"c{i}": {
            "id": f"c{i}",
            "label": f"Candidate {i}",
            "kind": "held_out",
            "design": {"w_in": 20e-6, "l_in": 0.15e-6, "i_tail": 500e-6,
                       "rs": 1e3, "cs": 200e-15, "r_load": 1e3, "w_dfe": 0.0},
            "target_boost_db": 8.0,
            "channel_loss_db": 12.0,
            # The serializer requires both provenance fields: the gallery's whole point is
            # that a historical selection is never presented as a fresh measurement.
            "historical_context": f"stub candidate {i}",
            "binding_constraint": {"label": "stub", "detail": "stub",
                                   "scope": "additional_design_requirement"},
        }
        for i in range(n)
    }


def _request(**kw) -> object:
    return dashboard_server.GalleryEvaluateRequest(**kw)


@pytest.fixture
def stub_catalog(monkeypatch):
    monkeypatch.setattr(dashboard_server, "_gallery_catalog", lambda: _catalog(8))
    return _catalog(8)


# -- validation: these must all fail BEFORE the simulator lock is taken ----------------

@pytest.mark.parametrize("payload, reason", [
    ({"candidate_ids": []}, "empty selection"),
    ({"candidate_ids": [f"c{i}" for i in range(7)]}, "over the 6-candidate cap"),
    ({"candidate_ids": ["c0", "c0"]}, "duplicate id"),
    ({"candidate_ids": ["nope"]}, "unknown id"),
])
def test_invalid_selections_are_refused_with_422(stub_catalog, payload, reason):
    response = dashboard_server.candidate_gallery_evaluate(_request(**payload))
    assert response.status_code == 422, reason
    assert json.loads(response.body)["error_code"] == "invalid_request"


def test_mixing_catalogued_and_live_candidates_is_refused(stub_catalog):
    response = dashboard_server.candidate_gallery_evaluate(
        _request(candidate_ids=["c0"], live_candidates=[{"design": {}}]))
    assert response.status_code == 422
    assert json.loads(response.body)["error_code"] == "invalid_request"


def test_a_rejected_selection_never_reaches_the_simulator(stub_catalog, monkeypatch):
    """Validation must run before the lock, or a bad request can stall a good one."""
    def explode(_candidate):
        raise AssertionError("an invalid selection reached the evaluator")

    monkeypatch.setattr(dashboard_server, "_evaluate_gallery_candidate", explode)
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())
    response = dashboard_server.candidate_gallery_evaluate(_request(candidate_ids=["nope"]))
    assert response.status_code == 422
    # The lock must still be free: a refused request may not leave it held.
    assert dashboard_server._run_lock.acquire(blocking=False)
    dashboard_server._run_lock.release()


# -- lock discipline -------------------------------------------------------------------

def test_concurrent_gallery_evaluation_returns_structured_pipeline_busy(stub_catalog,
                                                                       monkeypatch):
    """The gallery shares one simulator lock with the guard and pipeline routes."""
    entered, release = threading.Event(), threading.Event()

    def blocking_evaluate(candidate):
        entered.set()
        assert release.wait(timeout=2.0), "the test never released the fake evaluator"
        return {"id": candidate["id"], "guard_valid": True}

    monkeypatch.setattr(dashboard_server, "_evaluate_gallery_candidate", blocking_evaluate)
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())

    first: list[object] = []
    worker = threading.Thread(
        target=lambda: first.append(
            dashboard_server.candidate_gallery_evaluate(_request(candidate_ids=["c0"]))),
        daemon=True)
    worker.start()
    assert entered.wait(timeout=2.0), "the first gallery request never entered the fake"

    try:
        blocked = dashboard_server.candidate_gallery_evaluate(_request(candidate_ids=["c1"]))
        assert blocked.status_code == 409
        body = json.loads(blocked.body)
        assert body["error_code"] == "pipeline_busy"
        assert body["error_number"] == 202
    finally:
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive(), "the first gallery request did not finish after release"


def test_gallery_shares_the_lock_with_the_guard_endpoint(stub_catalog, monkeypatch):
    """Holding the lock outside the gallery must still produce the busy response.

    This is the regression that the duplicate `_run_lock` binding would have reopened:
    two bindings meant the gallery could hold a *different* lock from the guard route and
    both would run against the one non-reentrant resident ngspice process.
    """
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())
    monkeypatch.setattr(dashboard_server, "_evaluate_gallery_candidate",
                        lambda c: {"id": c["id"]})
    assert dashboard_server._run_lock.acquire(blocking=False)
    try:
        response = dashboard_server.candidate_gallery_evaluate(_request(candidate_ids=["c0"]))
        assert response.status_code == 409
        assert json.loads(response.body)["error_code"] == "pipeline_busy"
    finally:
        dashboard_server._run_lock.release()


def test_search_halted_releases_the_gallery_lock(stub_catalog, monkeypatch, tmp_path):
    """SearchHalted is a BaseException; the route's finally must still release."""
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())

    def halt(_candidate):
        raise SearchHalted(Check.T5_INVALID_RATE, "test halt", tmp_path / "HALT.txt")

    monkeypatch.setattr(dashboard_server, "_evaluate_gallery_candidate", halt)
    response = dashboard_server.candidate_gallery_evaluate(_request(candidate_ids=["c0"]))
    assert response.status_code == 409
    assert json.loads(response.body)["error_code"] == "search_halted"
    assert dashboard_server._run_lock.acquire(blocking=False), (
        "SearchHalted escaped the inner try but the gallery route did not release `_run_lock`")
    dashboard_server._run_lock.release()


def test_an_unexpected_evaluator_error_still_releases_the_lock(stub_catalog, monkeypatch):
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())
    monkeypatch.setattr(dashboard_server, "_evaluate_gallery_candidate",
                        lambda _c: (_ for _ in ()).throw(RuntimeError("boom")))
    response = dashboard_server.candidate_gallery_evaluate(_request(candidate_ids=["c0"]))
    assert response.status_code >= 400
    assert dashboard_server._run_lock.acquire(blocking=False), (
        "an unexpected evaluator error left `_run_lock` held")
    dashboard_server._run_lock.release()


# -- listing route ---------------------------------------------------------------------

def test_listing_route_never_measures_and_is_not_cacheable(stub_catalog, monkeypatch):
    """GET must be artifact-only: measurement happens on the explicit POST."""
    monkeypatch.setattr(dashboard_server, "_evaluate_gallery_candidate",
                        lambda _c: pytest.fail("the listing route measured a candidate"))
    response = dashboard_server.candidate_gallery()
    body = json.loads(response.body)
    assert len(body["candidates"]) == 8
    assert body["max_candidates"] == dashboard_server._GALLERY_MAX_CANDIDATES
    # Simulator-derived state must not be cached by an intermediary or the browser.
    assert response.headers["Cache-Control"] == "no-store"


def test_the_candidate_cap_matches_the_documented_gallery_size():
    """The UI promises 4-6 tiles; the server must not silently allow more."""
    assert dashboard_server._GALLERY_MAX_CANDIDATES == 6


# -- the real, artifact-backed catalog --------------------------------------------------

def test_every_real_catalog_entry_carries_its_provenance():
    """Artifact-only (no simulator): the serializer requires both provenance fields.

    A catalog entry that omitted either one would not fail quietly -- it would 500 the
    listing route -- but the reason these are mandatory is editorial, not defensive: the
    gallery shows historical *selections* beside a fresh measurement, and a tile with no
    `historical_context` would read as a current result for a design that was rejected.
    """
    catalog = dashboard_server._gallery_catalog()
    assert catalog, "the artifact-backed catalog is empty"
    assert "delivered" in catalog
    kinds = {c["kind"] for c in catalog.values()}
    assert "historical_rejection" in kinds, (
        "the comparison set must keep at least one real guard rejection visible")
    for ident, candidate in catalog.items():
        assert candidate["historical_context"].strip(), ident
        constraint = candidate["binding_constraint"]
        assert constraint["label"].strip() and constraint["detail"].strip(), ident
        assert constraint["scope"] in {"additional_design_requirement", "circuit_sanity"}, ident
        # No entry may carry a cached metric: the POST route re-measures.
        assert not ({"metrics", "boost_db", "guard_valid"} & set(candidate)), (
            f"{ident} smuggles a stale measurement into the selection metadata")


def test_the_real_catalog_fits_inside_the_selection_cap():
    """The UI offers the whole catalog; a catalog bigger than the cap is unselectable."""
    assert len(dashboard_server._gallery_catalog()) <= dashboard_server._GALLERY_MAX_CANDIDATES
