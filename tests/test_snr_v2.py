import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from silq.snr_spec import SNRRequest
from silq.sim.snr_evaluation import signal_calibration, channel_unit_rms, evaluate_snr


def measured(**kw):
    return dict(mode='measured', value_vrms=.005, signal_reference='tx_vpp',
                signal_value_v=.8, bandwidth_hz=[1e7, 5e9], **kw)


@pytest.mark.parametrize('data', [
    {'mode': 'measured', 'value_vrms': .005},
    {'mode': 'estimated', 'budget_vrms': .005},
    dict(measured(), budget_vrms=.01), dict(measured(), value_vrms=-1),
    dict(measured(), value_vrms=float('nan')), dict(measured(), signal_value_v=True),
    dict(measured(), bandwidth_hz=[5e9, 1e7]), dict(measured(), bandwidth_hz=[0, 5e9]),
    dict(measured(), bandwidth_hz=[1e7, 6e9]), dict(measured(), assumed_fields=['noise']),
    {'mode': 'unknown', 'low_vrms': .005}, dict(measured(), bogus=3),
])
def test_invalid_contracts_refused(data):
    with pytest.raises(ValueError):
        SNRRequest.from_dict(data)


def test_unknown_provenance_and_roundtrip():
    request = SNRRequest.from_dict({'mode': 'unknown'})
    assert request.provenance == dict(noise='assumed', signal='assumed', bandwidth='assumed')
    assert SNRRequest.from_dict(request.to_dict()) == request
    edited = SNRRequest.from_dict(dict(request.to_dict(), low_vrms=.002, high_vrms=.04))
    assert edited.provenance['noise'] == 'assumed'
    supplied = SNRRequest.from_dict(dict(mode='unknown', signal_reference='ctle_input_vrms', signal_value_v=.2))
    assert supplied.provenance['signal'] == 'supplied'
    assert supplied.provenance['bandwidth'] == 'assumed'


def test_budget_and_range_cover_interiors():
    data = measured(); data.pop('value_vrms'); data.update(mode='estimated', budget_vrms=.04)
    request = SNRRequest.from_dict(data)
    assert request.points() == (0., .01, .02, .03, .04)
    assert SNRRequest.from_dict(request.to_dict()) == request


def test_both_signal_references_normalize_equivalently():
    tx = SNRRequest.from_dict(measured())
    swing, rms = signal_calibration(tx, 12.)
    rx = SNRRequest.from_dict(dict(measured(), signal_reference='ctle_input_vrms', signal_value_v=rms))
    assert signal_calibration(rx, 12.) == pytest.approx((swing, rms))
    assert channel_unit_rms(12., (1e7, 2.5e9)) != channel_unit_rms(12., (1e7, 5e9))
    assert len(tx.observation(swing)) == 8
    assert tx.observation(swing) != replace(tx, bandwidth_hz=(1e7, 2.5e9)).observation(swing)


def test_interior_failure_gates_result_and_band_reaches_noise(monkeypatch):
    import silq.sim.snr_evaluation as scorer
    from silq.sim.measures import Measures
    from silq.specs import DEFAULT_SPEC
    data = measured(); data.pop('value_vrms'); data.update(mode='estimated', low_vrms=0., high_vrms=.04, bandwidth_hz=[1e8, 3e9])
    request = SNRRequest.from_dict(data)
    srv = SimpleNamespace(ac_complex=lambda *a, **k: dict(freq=np.array([1e6, 24e9]), H=np.ones(2)))
    bands = []
    monkeypatch.setattr(scorer, 'differential_output_noise', lambda *a, **kw: bands.append(kw['bandwidth_hz']) or .001)
    def eye(*a, **kw):
        bad = .019 < kw['noise_sigma_v'] < .021
        return SimpleNamespace(height_v=0. if bad else .3, width_ui=0. if bad else .8,
            errors=int(bad), count=480, ber=float(bad)/480, sample_phase=7, dfe_tap=.01)
    monkeypatch.setattr(scorer, 'compute_eye_v2', eye)
    monkeypatch.setattr('silq.envs.sequential_env._shaped', lambda m, *a: (m.eye_v_mv, m.eye_v_mv > 100))
    _, _, passed, detail = evaluate_snr(srv, None, Measures(), DEFAULT_SPEC, request)
    assert not passed and detail['worst_point_index'] == 2
    assert detail['points'][0]['passed'] and detail['points'][-1]['passed']
    assert bands == [(1e8, 3e9)]
    assert len({p['sample_phase'] for p in detail['points']}) == 1
    json.dumps(detail, allow_nan=False)


def test_nominal_pass_cannot_hide_unknown_failure(monkeypatch):
    import silq.sim.snr_evaluation as scorer
    from silq.snr_pipeline import attach_noise_result
    from silq.circuits.ctle import decode_action
    from dataclasses import asdict
    from silq.sim.measures import Measures
    monkeypatch.setattr('silq.sim.server.get_server', lambda *a: object())
    monkeypatch.setattr(scorer, 'evaluate_snr', lambda *a, **k: (None, 0., False,
        {'status': 'measured', 'passed': False}))
    result = {'status': 'solved', 'design': asdict(decode_action(np.ones(6)*.5)),
        'verification': {'guard_valid': True, 'passed': True, 'measures': Measures().as_dict()}}
    attach_noise_result(result, SNRRequest.from_dict({'mode':'unknown'}), target=9., channel=12.)
    assert result['status'] == 'noise_not_verified'
    assert result['nominal_status'] == 'solved' and result['overall_conditional']
    assert not result['overall_passed']


def test_api_rejects_incomplete_noise_before_search(monkeypatch):
    import server
    monkeypatch.setattr(server, 'design', lambda *a, **k: pytest.fail('must not search'))
    result = server.pipeline_run(server.PipelineRunRequest(target_boost_db=9.,
        noise_request={'mode': 'measured', 'value_vrms': .005}))
    assert result.status_code == 422
    assert json.loads(result.body)['error_number'] == 103
