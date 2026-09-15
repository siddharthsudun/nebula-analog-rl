"""Prevent silent VCM, transistor-probe, area, and deck-provenance mistakes."""
from dataclasses import replace
import re

import pytest

from silq.circuits import ctle
from silq.circuits.tail_variants import (
    TailConfig, TailDesign, geometry, instances, nodes, param_deck, parameters, snapshot, tail_devices,
)


@pytest.fixture(autouse=True)
def fake_pdk(monkeypatch):
    from silq.circuits import pdk
    monkeypatch.setattr(pdk, "lib_include", lambda c: f"* PDK corner {c}")


@pytest.mark.parametrize("config", [TailConfig(), TailConfig("wide_swing", 1), TailConfig("wide_swing", 4)])
def test_every_transistor_is_probed_and_every_added_node_is_observed(config):
    deck = param_deck("tt", config)
    devices = re.findall(r"^(XM\w+)\s", deck, re.M)
    assert set(devices) == set(instances(config))
    for name, source_node in zip(tail_devices(config), ("sp", "sn")):
        assert re.search(rf"^{name} {source_node} ", deck, re.M)
    for line in deck.splitlines():
        if line.startswith("XM"):
            for node in line.split()[1:5]:
                assert node in set(nodes(config)) | {"0", "inp", "inn"}


def test_actual_simulator_parameters_and_snapshot_both_carry_vcm():
    dv = TailDesign(i_tail=0.7e-3, tail=TailConfig("wide_swing", 4))
    deck = snapshot(dv, corner="tt", vdd=1.71, temp_c=125, vcm_ratio=0.68)
    p = parameters(dv, vdd=1.71, temp_c=125, vcm_ratio=0.68)
    assert "Vcm cm 0 {vddp*vcm_ratio}" in deck
    for k, v in p.items():
        assert f".param {k}={v:.6g}\n" in deck
    assert p["vcm_ratio"] == 0.68
    assert "XMcp sp ncas ntp 0" in deck
    assert "Ibcas vdd ncas 'itail/2'" in deck


def test_area_counts_all_added_transistor_width_and_default_is_unchanged():
    dv = TailDesign(i_tail=0.7e-3)
    assert dv.area_mm2() == ctle.DesignVars(i_tail=0.7e-3).area_mm2()
    wide = replace(dv, tail=TailConfig("wide_swing", 4))
    g = geometry(wide, wide.tail)
    w, m = ctle.mirror_sizing(dv.i_tail)
    expected_extra = (6*g["wt"]*g["mt"] + g["wq"]*g["mq"] - 3*w*m)*ctle.MIRROR_L_UM*1e-6
    assert wide.area_mm2() - dv.area_mm2() == pytest.approx(expected_extra)
    assert max(g["wt"], g["wq"]) <= ctle.MIRROR_W_MAX_UM


def test_experimental_deck_does_not_mutate_shipped_vcm_or_deck():
    original = ctle.param_deck("tt")
    constants = (ctle.VCM, ctle.VCM_VDD_RATIO)
    param_deck("tt", TailConfig("wide_swing", 4))
    assert ctle.param_deck("tt") == original
    assert constants == (ctle.VCM, ctle.VCM_VDD_RATIO)


@pytest.mark.parametrize("value", [float("nan"), 0, 1, -0.1])
def test_invalid_vcm_is_rejected(value):
    with pytest.raises(ValueError):
        parameters(TailDesign(), vdd=1.8, temp_c=27, vcm_ratio=value)


def test_pilot_writes_numpy_check_flags_as_json_booleans(tmp_path):
    import json
    import numpy as np
    from silq.experiments.tail_mirror_pilot import write_json
    path = tmp_path / "row.json"
    write_json(path, {"passed": np.bool_(True)})
    assert json.loads(path.read_text()) == {"passed": True}
