"""Every failure this server can return is labelled the same way.

The taxonomy in `server.ERROR_CODES` promises a code, a stable number, a short title, a
detail and a next step on every failure path. It kept that promise only where the handler
could RETURN `_api_error`. Five checks sat in helpers and mid-endpoint, had to raise, and
raised a bare `HTTPException` -- which FastAPI renders as `{"detail": "artifact not
found"}`. The dashboard's renderer looks for `title`, finds none, and falls through to
`"Failed to load: artifact not found"`: a string that names neither what is missing nor
how to make it exist, and carries no number anyone could quote in a bug report.

So what these pin is not one message but the CLOSURE: no failure path outside the
taxonomy, checked at the source level, because the next endpoint added is where a
`raise HTTPException` would come back.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server as dashboard_server
from server import ApiError, ERROR_CODES

SERVER_SRC = Path("server.py").read_text(encoding="utf-8")
REQUIRED_KEYS = {"error_code", "error_number", "label", "title", "detail", "hint"}


def test_no_failure_path_escapes_the_taxonomy():
    """A bare HTTPException is an unlabelled error by construction.

    It cannot carry a code, a number or a next step -- FastAPI serialises only `detail`.
    `ApiError` exists so a site that must raise still gets the full body, so the source
    rule is simply that nothing raises the bare one. Do NOT relax this by allowing a
    bare raise "just for an internal case": every one of the five this replaced was
    reachable from the dashboard.
    """
    offenders = [line.strip() for line in SERVER_SRC.splitlines()
                 if "raise HTTPException(" in line]
    assert not offenders, (
        "these raise a bare HTTPException, which reaches the user as {'detail': ...} "
        "with no code, no number and no next step -- raise ApiError instead: "
        + " | ".join(offenders))


def test_every_error_code_has_its_own_number():
    """The number is what a person quotes. Two failures sharing one are indistinguishable."""
    assert len(set(ERROR_CODES.values())) == len(ERROR_CODES), ERROR_CODES


def test_every_declared_code_is_actually_emitted():
    """A code nobody raises is a taxonomy entry that documents nothing."""
    unused = [c for c in ERROR_CODES if f'"{c}"' not in SERVER_SRC.split("ERROR_CODES")[2]]
    assert not unused, f"declared but never emitted: {unused}"


def test_the_raisable_error_carries_the_same_body_as_the_returned_one():
    """ApiError and _api_error must not drift into two shapes."""
    returned = json.loads(bytes(dashboard_server._api_error(
        404, "artifact_missing", "t", "d", "h").body))
    raised = ApiError(404, "artifact_missing", "t", "d", "h").body
    assert returned == raised
    assert REQUIRED_KEYS <= set(raised)
    assert raised["error_number"] == ERROR_CODES["artifact_missing"]
    assert raised["label"] == f"Error {ERROR_CODES['artifact_missing']}"


def test_an_unknown_code_still_gets_a_body_rather_than_a_crash():
    """A typo'd code must degrade to 500, not raise inside the error path itself."""
    body = ApiError(500, "not_a_real_code", "t", "d", "h").body
    assert body["error_number"] == 500 and REQUIRED_KEYS <= set(body)


@pytest.fixture()
def empty_results(tmp_path, monkeypatch):
    """A checkout whose results/ has none of the artifacts the read-only routes want."""
    monkeypatch.setattr(dashboard_server, "RESULTS_DIR", tmp_path)
    return tmp_path


@pytest.mark.parametrize("route, expected", [
    (lambda: dashboard_server.design_time(),
     ["final_report.json", "sweep_baseline.json", "speedup.json", "delivered_circuit.json"]),
    (lambda: dashboard_server.model_performance(),
     ["surrogate_audit.json", "final_report.json"]),
    (lambda: dashboard_server.candidate_gallery(),
     ["delivered_circuit.json", "pass_vs_valid.json"]),
])
def test_a_missing_artifact_names_the_file_and_says_what_to_do(empty_results, route, expected):
    """"artifacts are missing" is not a diagnosis; WHICH file is missing is.

    The three routes read four, two and two artifacts. Told only that some are absent, a
    reader has to open the endpoint's source to find out which -- which is the work the
    message existed to save.
    """
    with pytest.raises(ApiError) as excinfo:
        route()
    body = excinfo.value.body
    assert body["error_code"] == "artifact_missing"
    assert excinfo.value.status_code == 404
    assert REQUIRED_KEYS <= set(body)
    for name in expected:
        assert name in body["detail"], f"{name} missing from: {body['detail']}"
    assert body["hint"].strip(), "a 404 with no next step is half an error message"


def test_a_partially_generated_checkout_names_only_what_is_absent(empty_results):
    """Listing files that ARE present would send the reader to regenerate the wrong one."""
    # Non-empty: the endpoint's guard is a truthiness test, so an empty object counts
    # as absent -- correctly, but it would not exercise the "names only what is absent"
    # behaviour this pins.
    (empty_results / "final_report.json").write_text('{"solved": 1}', encoding="utf-8")
    with pytest.raises(ApiError) as excinfo:
        dashboard_server.model_performance()
    detail = excinfo.value.body["detail"]
    assert "surrogate_audit.json" in detail
    assert "final_report.json" not in detail


@pytest.mark.parametrize("name, code, status", [
    ("../secrets.json", "invalid_request", 400),
    ("results/nested.json", "invalid_request", 400),
    ("not_json.txt", "invalid_request", 400),
    ("absent.json", "artifact_missing", 404),
])
def test_the_artifact_path_check_labels_its_refusals(empty_results, name, code, status):
    with pytest.raises(ApiError) as excinfo:
        dashboard_server._safe_results_path(name)
    assert excinfo.value.body["error_code"] == code
    assert excinfo.value.status_code == status
    assert REQUIRED_KEYS <= set(excinfo.value.body)


def test_the_handler_renders_the_body_it_was_given():
    """Registering the handler is the half that is easy to forget; without it FastAPI
    falls back to the HTTPException handler and serialises `detail` alone."""
    import asyncio
    exc = ApiError(404, "artifact_missing", "t", "d", "h")
    response = asyncio.run(dashboard_server._api_error_handler(None, exc))
    assert response.status_code == 404
    assert json.loads(bytes(response.body)) == exc.body
    assert dashboard_server.ApiError in dashboard_server.app.exception_handlers


# -- the 201 slot: a checkout with no checkpoint ------------------------------------------

def test_a_missing_checkpoint_is_an_installation_state_not_a_server_bug(tmp_path, monkeypatch):
    """Without this the run reached PPO.load and came back as "Unexpected server error /
    FileNotFoundError", whose next step is "Retry" -- advice that cannot ever work.
    """
    monkeypatch.setattr(dashboard_server, "REPO_ROOT", tmp_path)
    response = dashboard_server._policy_missing_error("thinking")
    body = json.loads(bytes(response.body))
    assert response.status_code == 503
    assert body["error_code"] == "policy_missing"
    assert body["error_number"] == ERROR_CODES["policy_missing"]
    assert REQUIRED_KEYS <= set(body)
    assert "thinking" in body["detail"]
    assert "retry" not in body["hint"].lower() or "cannot" in body["hint"].lower()


@pytest.mark.parametrize("mode", ["thinking", "default", "retarget", "g32_acceptance"])
def test_a_policy_backed_mode_is_refused_before_it_searches(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(dashboard_server, "REPO_ROOT", tmp_path)
    response = dashboard_server.pipeline_run(
        dashboard_server.PipelineRunRequest(target_boost_db=9.0, channel_loss_db=12.0, mode=mode))
    assert response.status_code == 503
    assert json.loads(bytes(response.body))["error_code"] == "policy_missing"


@pytest.mark.parametrize("mode", ["fastest", "auto"])
def test_the_two_modes_that_can_run_without_a_checkpoint_are_not_refused(tmp_path, monkeypatch, mode):
    """`fastest` never loads the policy (pipeline.py:829) and `auto` starts with it.

    Refusing them would take away capability a policy-less checkout really has, so the
    pre-flight must let them past -- they fail later, or do not fail at all. This asserts
    only that the refusal is not the answer; what comes back next needs a simulator.
    """
    monkeypatch.setattr(dashboard_server, "REPO_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(dashboard_server, "_policy_missing_error",
                        lambda m: calls.append(m) or "refused")
    monkeypatch.setattr(dashboard_server, "ready", lambda: False, raising=False)
    try:
        dashboard_server.pipeline_run(
            dashboard_server.PipelineRunRequest(target_boost_db=9.0, channel_loss_db=12.0, mode=mode))
    except Exception:
        pass  # anything past the pre-flight is this test's "not refused"
    assert calls == [], f"{mode} was refused for a missing policy it does not need"
