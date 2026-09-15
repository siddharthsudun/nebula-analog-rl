import pytest
from eqrl.agents.train_noise_pilot import _build_parser, _validated_config, _env_counters, _noise_fields


def test_default_budget_is_exactly_five_rollouts():
    args = _build_parser().parse_args(['--run-dir', 'unused-test-output'])
    config = _validated_config(args)
    assert config['timesteps_effective'] == 5120
    assert config['rollouts_effective'] == 5
    assert config['fast'] is False and config['guarded'] is True


def test_rounding_cannot_exceed_step_cap():
    args = _build_parser().parse_args(['--run-dir', 'unused', '--n-envs', '3'])
    with pytest.raises(ValueError, match='rounding'):
        _validated_config(args)


def test_counters_only_query_real_worker_attributes_and_keep_numbers():
    class Env:
        def get_attr(self, name):
            assert name in {'n_sims', 'n_invalid', 'n_noise_failures', 'n_noise_evaluations'}
            return [2, 3]
    result = _env_counters(Env())
    assert result['n_sims'] == 5
    assert isinstance(result['n_noise_evaluations'], int)


def test_noise_progress_uses_actual_environment_info():
    mode, outcome, passed = _noise_fields({'noise': {'mode': 'range'},
        'noise_evaluation': {'status': 'measured'}, 'passed': True})
    assert (mode, outcome, passed) == ('range', 'measured', True)


def test_worker_factory_can_be_serialized_without_native_handles(tmp_path):
    import cloudpickle
    from eqrl.agents.train_noise_pilot import _worker_factory
    factory = _worker_factory({'seed': 7, 'horizon': 20}, 0, tmp_path)
    restored = cloudpickle.loads(cloudpickle.dumps(factory))
    assert callable(restored)
