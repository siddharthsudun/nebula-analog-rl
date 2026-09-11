"""Real four-process startup, signal/band checks, and a tiny optimizer wiring check."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    from eqrl.agents.train_noise_pilot import _configure_ngspice, _set_single_threaded, _worker_factory, _write_json, _make_callback
    _set_single_threaded()
    _configure_ngspice()
    import numpy as np
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3 import PPO
    from eqrl.circuits.ctle import DesignVars, encode_action
    from eqrl.sim.snr_evaluation import channel_unit_rms
    from eqrl.experiments.noise_holdout import cases, evaluate_policy
    config = dict(seed=91371, horizon=20, timesteps_requested=8, timesteps_effective=8, wall_seconds=180)
    vec = None
    checks = []
    started = time.monotonic()
    _write_json(run_dir/'trainer_heartbeat.json', {'phase':'starting', 'updated_at_unix':time.time()})
    try:
        vec = SubprocVecEnv([_worker_factory(config, rank, run_dir) for rank in range(4)], start_method='spawn')
        data = json.loads((Path(__file__).resolve().parents[3]/'results'/'delivered_circuit.json').read_text())
        dv = DesignVars(**{k:v for k,v in data['design'].items() if k in {f.name for f in fields(DesignVars)}})
        for key, val in dict(_anchor_x=encode_action(dv), anchor_noise=0.,
                             target_range=(data['spec']['target_boost_db'],)*2,
                             channel_range=(data['spec']['channel_loss_db'],)*2).items():
            vec.set_attr(key, val)
        _write_json(run_dir/'trainer_heartbeat.json', {'phase':'training', 'updated_at_unix':time.time()})
        for mode in ('measured', 'estimated', 'unknown'):
            for reference in ('tx_vpp', 'ctle_input_vrms'):
                band = (2e7, 3e9)
                value = .8 if reference == 'tx_vpp' else .8*channel_unit_rms(data['spec']['channel_loss_db'], band)
                req = dict(mode=mode, signal_reference=reference, signal_value_v=value, bandwidth_hz=band)
                if mode == 'measured':
                    req['value_vrms'] = .003
                else:
                    req.update(low_vrms=.001, high_vrms=.03)
                results = vec.env_method('reset', seed=91371, options={'noise_request':req})
                for obs, info in results:
                    assert obs.shape == (26,) and np.isfinite(obs).all()
                    detail = info['noise_evaluation']
                    assert detail['status'] == 'measured', detail
                    assert detail['receiver_output_noise_vrms'] > 0
                    assert detail['bandwidth_hz'] == list(band)
                    assert abs(detail['equivalent_tx_vpp']-.8) < 1e-10
                    assert len(detail['points']) == (1 if mode == 'measured' else 5)
                    assert len({p['sample_phase'] for p in detail['points']}) == 1
                    assert len({p['dfe_tap'] for p in detail['points']}) == 1
                checks.append({'request':req, 'evaluation':results[0][1]['noise_evaluation']})
                _write_json(run_dir/'trainer_heartbeat.json', {'phase':'training', 'updated_at_unix':time.time()})
        # The paired references must generate identical scored eyes under fixed streams.
        for i in range(0, len(checks), 2):
            a, b = checks[i]['evaluation'], checks[i+1]['evaluation']
            assert np.allclose([p['eye_v_mv'] for p in a['points']], [p['eye_v_mv'] for p in b['points']])
        model = PPO('MlpPolicy', vec, n_steps=2, batch_size=8, n_epochs=1, seed=7, verbose=0)
        callback = _make_callback(run_dir, config, time.monotonic()+120)
        model.learn(total_timesteps=8, callback=callback)
        assert model._n_updates == 1
        model.save(str(run_dir/'optimizer_smoke.zip'))
        restored = PPO.load(str(run_dir/'optimizer_smoke.zip'))
        assert restored.observation_space.shape == (26,)
        _write_json(run_dir/'trainer_heartbeat.json', {'phase':'validating', 'updated_at_unix':time.time()})
        subset = cases()[:1]
        result = evaluate_policy(restored, subset, run_dir, label='smoke', deadline=time.monotonic()+120)
        assert result['complete'] and result['evaluations'] <= 5
        status = {'status':'completed', 'checks':checks, 'optimizer_updates':model._n_updates,
                  'worker_pids':[json.loads((run_dir/f'worker_{i}.json').read_text())['pid'] for i in range(4)],
                  'wall_seconds':time.monotonic()-started}
        assert len(set(status['worker_pids'])) == 4
        _write_json(run_dir/'status.json', status)
        print(json.dumps({'status':'passed', 'worker_pids':status['worker_pids'], 'wall_seconds':status['wall_seconds']}))
    finally:
        _write_json(run_dir/'trainer_heartbeat.json', {'phase':'closing', 'updated_at_unix':time.time()})
        if vec is not None:
            vec.close()


if __name__ == '__main__':
    main()
