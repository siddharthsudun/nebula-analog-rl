"""Pareto selection must be genuinely multi-objective, and honest about what it checked.

The feature it replaces ranked "alternatives" by boost error alone and then showed the
final circuit's PVT verdict beside all of them. Both of those are what these tests pin
down: selection has to trade real measured quantities off against each other, and a
circuit that was never PVT-checked has to say so.
"""
from __future__ import annotations

import itertools

import pytest

from silq import pareto
from silq.circuits import pdk

#: `pareto.finalize` renders a SKY130 netlist for every choice it returns -- that is
#: exactly what test_every_choice_carries_its_objectives_and_a_netlist pins -- so the
#: tests that call it need the PDK on disk. Everything above them is pure selection
#: arithmetic over synthetic measurements and runs anywhere, which is why this marker is
#: on the finalize tests only and not on the module.
needs_pdk = pytest.mark.skipif(
    not pdk.available(),
    reason="pareto.finalize renders a SKY130 netlist for every choice")

#: Distinct circuits need distinct design keys, or `frontier`'s dedupe collapses them and
#: a test passes for the wrong reason. Untagged items get a fresh sizing each call.
_UNIQUE = itertools.count(1000)


#: A real, buildable sizing. `finalize` renders a netlist for every choice it returns, so
#: these have to be genuine DesignVars fields, not stand-ins.
BASE_DESIGN = {"w_in": 5.5784369934553225e-05, "l_in": 3.9211895465850825e-07,
               "i_tail": 0.0007122746556997299, "rs": 3287.4510782957077,
               "cs": 1.8162099438699198e-13, "r_load": 2450.2967834472656, "w_dfe": 0.0}


def item(power, noise, area, boost, *, passed=True, guard_valid=True, tag=None):
    """One measured circuit, in the shape `frontier`/`choose` consume.

    `tag` perturbs a real sizing field so distinct circuits get distinct design keys
    without leaving the buildable design space.
    """
    design = dict(BASE_DESIGN)
    design["rs"] = BASE_DESIGN["rs"] + (abs(hash(tag)) % 997 if tag is not None else next(_UNIQUE))
    return {
        "design": design,
        "verification": {
            "passed": passed, "guard_valid": guard_valid,
            "measures": {"power_w": power, "noise_vrms": noise,
                         "area_mm2": area, "boost_db": boost},
        },
    }


class TestFrontier:
    def test_dominated_circuit_is_excluded(self):
        best = item(1e-3, 1e-6, 0.01, 8.0)
        worse = item(2e-3, 2e-6, 0.02, 8.0)      # worse on every axis
        got = pareto.frontier([best, worse], 8.0)
        assert [x["verification"]["measures"]["power_w"] for x in got] == [1e-3]

    def test_tradeoff_circuits_both_survive(self):
        low_power = item(1e-3, 5e-6, 0.02, 8.0)
        low_noise = item(3e-3, 1e-6, 0.02, 8.0)
        got = pareto.frontier([low_power, low_noise], 8.0)
        assert len(got) == 2, "neither circuit dominates the other; both are on the front"

    def test_failed_and_guard_invalid_circuits_never_enter(self):
        ok = item(1e-3, 1e-6, 0.01, 8.0)
        failed = item(1e-9, 1e-9, 1e-9, 8.0, passed=False)
        invalid = item(1e-9, 1e-9, 1e-9, 8.0, guard_valid=False)
        got = pareto.frontier([ok, failed, invalid], 8.0)
        assert len(got) == 1, "an unverified circuit must not be offered as a choice"

    def test_duplicate_designs_collapse(self):
        a = item(1e-3, 1e-6, 0.01, 8.0)
        got = pareto.frontier([a, dict(a)], 8.0)
        assert len(got) == 1

    def test_nonfinite_measurement_is_dropped_not_ranked(self):
        ok = item(1e-3, 1e-6, 0.01, 8.0)
        bad = item(float("nan"), 1e-6, 0.01, 8.0, tag="nan")
        assert len(pareto.frontier([ok, bad], 8.0)) == 1


class TestChoose:
    def test_each_choice_optimizes_a_different_quantity(self):
        items = [
            item(1e-3, 9e-6, 0.09, 8.0, tag="p"),   # cheapest power
            item(9e-3, 1e-6, 0.09, 8.0, tag="n"),   # quietest
            item(9e-3, 9e-6, 0.01, 8.0, tag="a"),   # smallest
        ]
        got = pareto.choose(items, 8.0)
        assert [c["optimized_quantity"] for c in got] == ["power_w", "noise_vrms", "area_mm2"]
        assert [c["label"] for c in got] == ["Low power", "Low noise", "Small area"]
        assert all(c["direction"] == "minimize" for c in got)

    def test_winner_on_each_axis_is_actually_the_best_on_that_axis(self):
        items = [
            item(1e-3, 9e-6, 0.09, 8.0),
            item(9e-3, 1e-6, 0.09, 8.0),
            item(9e-3, 9e-6, 0.01, 8.0),
        ]
        got = pareto.choose(items, 8.0)
        by_axis = {c["optimized_quantity"]: c["verification"]["measures"] for c in got}
        assert by_axis["power_w"]["power_w"] == 1e-3
        assert by_axis["noise_vrms"]["noise_vrms"] == 1e-6
        assert by_axis["area_mm2"]["area_mm2"] == 0.01

    def test_no_duplicate_circuits_when_one_design_wins_everything(self):
        """A single dominating circuit yields ONE choice, not three copies of it."""
        got = pareto.choose([item(1e-3, 1e-6, 0.01, 8.0)], 8.0)
        assert len(got) == 1
        keys = {pareto.design_key(c["design"]) for c in got}
        assert len(keys) == len(got)

    def test_selection_never_exceeds_the_cap(self):
        items = [item(1e-3 * i, 1e-6 * (9 - i), 0.01 * i, 8.0, tag=str(i))
                 for i in range(1, 8)]
        assert len(pareto.choose(items, 8.0)) <= 3

    def test_target_error_is_an_objective_not_the_ranking(self):
        """The old behaviour ranked purely by boost error. A circuit that is slightly
        further off target but far cheaper must still be selectable."""
        on_target_expensive = item(9e-3, 9e-6, 0.09, 8.0)
        off_target_cheap = item(1e-3, 1e-6, 0.01, 8.4)
        got = pareto.choose([on_target_expensive, off_target_cheap], 8.0)
        designs = {pareto.design_key(c["design"]) for c in got}
        assert pareto.design_key(off_target_cheap["design"]) in designs


@needs_pdk
class TestFinalizeHonesty:
    def _spec(self):
        from silq.specs import DEFAULT_SPEC
        import dataclasses
        return dataclasses.replace(DEFAULT_SPEC, target_boost_db=8.0)

    def test_alternatives_are_not_given_the_primary_circuits_pvt_verdict(self):
        spec = self._spec()
        primary = item(1e-3, 9e-6, 0.09, 8.0, tag="primary")
        other = item(9e-3, 1e-6, 0.02, 8.0, tag="other")
        result = {
            "design": primary["design"], "verification": primary["verification"],
            "pvt": {"accepted": True, "status": "verified"},
            "pareto": {"objectives": list(pareto.OBJECTIVES),
                       "measured_items": [primary, other]},
        }
        pareto.finalize(result, spec)
        by_key = {pareto.design_key(c["design"]): c for c in result["pareto"]["items"]}
        chosen_primary = by_key[pareto.design_key(primary["design"])]
        chosen_other = by_key[pareto.design_key(other["design"])]
        assert chosen_primary["pvt"]["accepted"] is True
        assert chosen_other["pvt"]["accepted"] is False, \
            "an alternative must not inherit the primary circuit's PVT pass"
        assert chosen_other["pvt"]["status"] == "not_run_for_this_circuit"

    def test_incomplete_set_is_flagged_rather_than_padded(self):
        spec = self._spec()
        only = item(1e-3, 1e-6, 0.01, 8.0, tag="only")
        result = {
            "design": only["design"], "verification": only["verification"],
            "pvt": {"accepted": False, "status": "pending"},
            "pareto": {"objectives": list(pareto.OBJECTIVES), "measured_items": [only]},
        }
        pareto.finalize(result, spec)
        block = result["pareto"]
        assert block["complete"] is False
        assert len(block["items"]) < 3
        assert "no duplicates or dominated fillers" in block["note"]

    def test_every_choice_carries_its_objectives_and_a_netlist(self):
        spec = self._spec()
        items = [item(1e-3, 9e-6, 0.09, 8.0, tag="p"),
                 item(9e-3, 1e-6, 0.09, 8.0, tag="n")]
        result = {
            "design": items[0]["design"], "verification": items[0]["verification"],
            "pvt": {"accepted": False, "status": "pending"},
            "pareto": {"objectives": list(pareto.OBJECTIVES), "measured_items": list(items)},
        }
        pareto.finalize(result, spec)
        for choice in result["pareto"]["items"]:
            assert set(choice["objectives"]) == set(pareto.OBJECTIVES)
            assert choice["id"].startswith("choice_")
            assert choice["netlist"].strip(), "each circuit needs its own recorded netlist"


def test_objectives_reject_a_negative_measurement():
    with pytest.raises(ValueError):
        pareto.objectives(item(-1.0, 1e-6, 0.01, 8.0), 8.0)


@needs_pdk
def test_stream_ids_reference_and_published_choices_are_stable():
    from silq.pipeline import spec_for
    spec=spec_for(8,12,1.5)
    primary=item(.005,.0005,.02,8,tag="primary")
    a=item(.003,.0007,.02,8,tag="a")
    result=dict(design=primary['design'],verification=primary['verification'],pareto=dict(measured_items=[a]))
    pareto.finalize(result,spec)
    before={pareto.design_key(i['design']):i['id'] for i in result['pareto']['items']}
    assert result['pareto']['items'][0]['design']==primary['design']
    better=item(.002,.0004,.01,8,tag="better")
    result['pareto']['measured_items']=[a,better]
    pareto.finalize(result,spec)
    after={pareto.design_key(i['design']):i['id'] for i in result['pareto']['items']}
    assert all(after[k]==v for k,v in before.items())
    assert result['pareto']['items'][0]['design']==primary['design']
    assert result['pareto']['items'][0]['on_frontier'] is False
    assert len(after)==len(result['pareto']['items'])
