"""Contract for `/api/snr-robustness`, the extension panel's only data source.

This route is presentation-only -- it projects `results/policy_snr_sweep_v1.json` down to
what the dashboard draws and computes nothing. That makes most of it uninteresting to
test. Three things are not:

1. **The two anchors must both survive.** The study's entire attribution argument is that
   the same v2 scorer at zero noise lands on the frozen v1 benchmark rate, so the drop
   along the curve is noise rather than the change of measurement engine. A refactor that
   drops either anchor turns the panel into an unsupported claim, silently.
2. **The SNR-invariant metrics must NOT appear per curve point.** The policy emits a
   bit-identical design at every SNR, so solve rate, boost error and evaluation count
   cannot move with noise. Repeating them down the per-SNR table would dress an
   architectural constant up as a robustness result -- the one presentation mistake this
   panel exists to avoid.
3. **A missing artifact must degrade to the structured 404**, because the sweep is an
   extension: a clone that never ran it has to see "not generated", not a stack trace.

The route function is called directly rather than through a TestClient, matching
`tests/test_gallery_endpoints.py`: the app's lifespan starts a real ngspice warm-up, and
nothing here needs a simulator.
"""
from __future__ import annotations

import json

import pytest

import server as dashboard_server


@pytest.fixture()
def payload():
    doc = dashboard_server._load_results_json(dashboard_server.SNR_ARTIFACT)
    if doc is None:
        pytest.skip(f"results/{dashboard_server.SNR_ARTIFACT} not generated in this tree")
    return dashboard_server.snr_robustness()


def test_the_curve_covers_every_swept_snr_point_in_order(payload):
    xs = [p["snr_db"] for p in payload["curve"]]
    assert xs == sorted(payload["snr_points_db"]), (
        "the panel plots curve rows against the x axis in the order it receives them, so a "
        "reordered or short curve draws a wrong line rather than raising")


def test_both_noiseless_anchors_are_reported(payload):
    """The drop is attributable to noise only because these two agree."""
    for key in ("strict_pass_v1", "strict_pass_v2_noiseless"):
        assert isinstance(payload[key], float), (
            f"{key} is missing. The panel's caption claims the frozen v1 benchmark rate and "
            "the zero-noise v2 rate coincide; without both numbers that claim is unbacked.")


def test_no_curve_point_carries_an_snr_invariant_metric(payload):
    """Guard against the one misrepresentation this experiment can commit."""
    forbidden = ("boost", "solve", "eval", "n_valid", "reward")
    for point in payload["curve"]:
        offending = [k for k in point if any(f in k for f in forbidden)
                     and not k.startswith("corr_")]
        assert not offending, (
            f"curve point at {point['snr_db']} dB carries {offending}. These are "
            "SNR-invariant by construction -- the frozen scorer is noiseless and the "
            "observation has no noise term -- so tabulating them per SNR point presents an "
            "architectural constant as a robustness result. They belong in the "
            "'invariant' block, which is rendered once.")


def test_the_invariant_block_holds_them_instead(payload):
    assert set(payload["invariant"]) == {"abs_boost_err_db", "evaluations", "solve_rate"}


def test_the_ceiling_is_the_best_case_not_the_observed_case(payload):
    """`unreachable_below_db` is the shaded band's edge, and the band's caption says NO
    design can pass to its left. That is only true if it is computed from the physical
    ceiling on the eye opening, which no real design reaches -- so it must sit below the
    SNR at which the policy's actual designs close."""
    assert payload["unreachable_below_db"] < payload["policy_closure_db"], (
        "the shaded region claims infeasibility for any design in any parameter space; if "
        "its edge were the policy's own closure SNR the claim would be far too strong")


def test_a_missing_artifact_returns_the_structured_404(monkeypatch):
    monkeypatch.setattr(dashboard_server, "_load_results_json", lambda name: None)
    res = dashboard_server.snr_robustness()
    assert res.status_code == 404
    body = json.loads(res.body)
    assert body["error_code"] == "artifact_missing"
    assert "policy_snr_sweep" in body["hint"], (
        "the hint is the only place a reader learns how to generate the missing artifact")
