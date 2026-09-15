import numpy as np
import pytest

from eqrl.llm.snr_parser import parse_noise_intent, resolve_snr_request
from eqrl.llm.spec_parser import parse_spec_verbose


def test_noise_is_strictly_opt_in():
    assert not parse_noise_intent('9 dB boost', {'mode': 'unknown'})['enabled']
    assert not parse_noise_intent('SNR 20 dB but ignore SNR')['enabled']
    parsed = parse_spec_verbose('9 dB boost, external noise 5 mV RMS', backend='off')
    assert 'noise_vrms_max' not in parsed.recognised
    assert parsed.noise['request']['value_vrms'] == .005
    intrinsic = parse_spec_verbose('noise under 1 mV', backend='off')
    assert intrinsic.recognised['noise_vrms_max'] == .001


def test_direct_snr_uses_explicit_reference_and_band():
    parsed = parse_noise_intent('input SNR 20 dB, input signal 200 mV RMS, band 10 MHz to 5 GHz')
    request = resolve_snr_request(parsed['request'], 12.)
    assert request.low_vrms == pytest.approx(.02)
    assert request.signal_reference == 'ctle_input_vrms'
    assert request.bandwidth_hz == (1e7, 5e9)
    with pytest.raises(ValueError):
        resolve_snr_request(parse_noise_intent('SNR 20 dB')['request'], 12.)


def test_unsupported_output_target_cannot_execute_as_measured_input():
    parsed = parse_noise_intent('output SNR 20 dB, 1 Vpp, band 10 MHz to 5 GHz')
    assert parsed['warnings']
    with pytest.raises(ValueError):
        resolve_snr_request(parsed['request'], 12.)


def test_llm_wrapper_retains_nested_snr_without_polluting_nominal_fields(monkeypatch):
    import eqrl.llm.spec_parser as parser
    monkeypatch.setattr(parser, '_pick_backend', lambda *a: 'api')
    monkeypatch.setattr(parser, '_llm_fields', lambda *a, **k: ({'_noise_request': {
        'mode': 'measured', 'input_snr_db': 23., 'signal_reference': 'tx_vpp',
        'signal_value_v': .8, 'bandwidth_hz': [1e7, 5e9]}}, {}))
    parsed = parser.parse_spec_verbose('consider SNR', backend='auto')
    assert parsed.noise['request']['input_snr_db'] == 23.
    assert '_noise_request' not in parsed.recognised


def test_full_budget_and_qualification_are_distinct():
    from eqrl.agents.train_noise_pilot import _build_parser, _validated_config
    from eqrl.experiments.noise_holdout import qualification_cases, cases
    config = _validated_config(_build_parser().parse_args(['--run-dir','unused',
        '--profile','full','--timesteps','40960','--wall-seconds','64785','--warm-start-frozen']))
    assert config['timesteps_effective'] == 40960
    assert config['rollouts_effective'] == 40
    assert config['warm_start_frozen'] and not config['fresh_policy']
    holdout = qualification_cases()
    assert len(holdout) == 24
    assert {r['seed'] for r in holdout}.isdisjoint({r['seed'] for r in cases()})
    assert qualification_cases() == holdout


def test_nominal_transfer_preserves_predictions_with_new_columns_zero():
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from eqrl.agents.train_noise_pilot import transfer_nominal_policy
    class Dummy(gym.Env):
        def __init__(self, width):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
    source = PPO('MlpPolicy', Dummy(18), n_steps=8, batch_size=8, seed=1)
    target = PPO('MlpPolicy', Dummy(26), n_steps=8, batch_size=8, seed=2)
    before = {k:v.clone() for k,v in source.policy.state_dict().items()}
    transfer_nominal_policy(source, target)
    obs = np.random.default_rng(77).normal(size=(5, 26)).astype(np.float32)
    np.testing.assert_allclose(source.predict(obs[:,:18], deterministic=True)[0],
                               target.predict(obs, deterministic=True)[0], atol=1e-7)
    assert all(torch.equal(v, source.policy.state_dict()[k]) for k,v in before.items())
    for name, value in target.policy.state_dict().items():
        if value.ndim == 2 and value.shape[1] == 26:
            assert torch.count_nonzero(value[:,18:]) == 0


def test_nominal_api_never_invokes_snr_when_omitted(monkeypatch):
    """A run that asks for no SNR must not touch the SNR path at all.

    Rewritten for the resident-worker dispatch: `pipeline_run` no longer calls `design`
    in-process, it hands ONE payload to `eqrl.runtime`. So the thing to assert is that the
    dispatched payload carries `noise_request=None` and that exactly one dispatch happens.

    Patching `ready` is not decoration. An unready runtime answers 503 *and*
    `_runtime_unavailable()` schedules a warm-up, which takes `_run_lock` on a background
    thread and holds it for the length of a real prepare(). That is correct in the server
    -- a request landing mid-warm-up gets an honest 409 instead of racing the one resident
    libngspice process -- but in a test it leaves the lock held for whatever runs next,
    which is what made `test_runtime_watchdog` fail only when this module preceded it.
    """
    import server
    from eqrl import runtime
    import eqrl.llm.snr_parser as noise_parser
    import eqrl.snr_pipeline as noise_pipeline
    monkeypatch.setattr(noise_parser, 'resolve_snr_request', lambda *a: pytest.fail('SNR parsing while off'))
    monkeypatch.setattr(noise_pipeline, 'attach_noise_result', lambda *a, **k: pytest.fail('SNR scoring while off'))
    payloads = []
    class Worker:
        def run(self, payload, limit, emit):
            payloads.append(payload)
            return {'status': 'solved', 'verification': {'passed': True}}
    monkeypatch.setattr(runtime, 'ready', lambda: True)
    monkeypatch.setattr(runtime, 'get_runtime', lambda: Worker())
    result = server.pipeline_run(server.PipelineRunRequest(target_boost_db=9.))
    assert result['status'] == 'solved'
    assert 'noise_evaluation' not in result
    assert len(payloads) == 1
    assert payloads[0]['noise_request'] is None
    assert not server._run_lock.locked(), "the run lock must be released before returning"


@pytest.mark.parametrize('text', [
    'signal-to-noise ratio of 25 dB',
    'signal to noise ratio of 25 dB',
    'signal-to-noise of 25 dB',
    'input SNR of 25 dB',
    '25 dB input SNR',
    'SNR = 25 dB',
])
def test_every_way_of_saying_snr_carries_its_number_too(text):
    """The INTENT test and the VALUE rules must accept the same vocabulary.

    They did not: INTENT matched "signal-to-noise" but the value rules anchored on the
    three letters "snr", so the spelled-out phrasing switched SNR ON and dropped the
    figure. The user then saw an SNR panel demanding a measured input SNR they had just
    typed -- the worst of both readings, and indistinguishable from the number being
    unreadable."""
    parsed = parse_noise_intent(text)
    assert parsed['enabled']
    assert parsed['request']['input_snr_db'] == pytest.approx(25.)
    assert parsed['request']['mode'] == 'measured'


@pytest.mark.parametrize('text', [
    'signal-to-noise ratio above 20 dB',
    'target signal-to-noise ratio 20 dB',
    'output signal-to-noise ratio of 20 dB',
])
def test_an_output_snr_target_is_refused_in_words_as_well_as_letters(text):
    """The mirror of the test above: widening the vocabulary must widen the REFUSAL with
    it. An output-SNR target read as a measured input SNR is not a near miss -- it is a
    different quantity, and executing it would score the run against a number the user
    never measured."""
    parsed = parse_noise_intent(text)
    assert any('unsupported' in w.lower() for w in parsed['warnings'])
    assert 'input_snr_db' not in parsed['request']
