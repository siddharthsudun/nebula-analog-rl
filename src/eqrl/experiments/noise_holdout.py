"""Deterministic paired monitoring set. Not a production qualification benchmark."""
from dataclasses import fields
import json
import time
import numpy as np

from eqrl.snr_spec import SNRRequest
from eqrl.sim.snr_evaluation import channel_unit_rms


def cases():
    result = []
    for mode in ('measured', 'estimated', 'unknown'):
        for reference in ('tx_vpp', 'ctle_input_vrms'):
            for i in range(2):
                channel, target = (9.3, 6.7) if i == 0 else (14.7, 9.4)
                band = (2e7, 3e9) if i == 0 else (8e7, 4e9)
                swing = .63 if i == 0 else .91
                value = swing if reference == 'tx_vpp' else swing*channel_unit_rms(channel, band)
                request = dict(mode=mode, signal_reference=reference, signal_value_v=value, bandwidth_hz=band)
                if mode == 'measured':
                    request['value_vrms'] = .0037 if i == 0 else .027
                elif mode == 'estimated':
                    if i == 0:
                        request.update(low_vrms=.0023, high_vrms=.034)
                    else:
                        request['budget_vrms'] = .034
                result.append(dict(id=len(result), target=target, channel=channel,
                    seed=730100+i, noise=SNRRequest.from_dict(request).to_dict()))
    return result


def summarize(rows):
    total = sum(r['evaluations'] for r in rows)
    invalid = sum(r['invalid'] for r in rows)
    return {'cases': len(rows), 'strict_passes': sum(r['strict_pass'] for r in rows),
            'noisy_eye_passes': sum(r.get('noisy_eye_pass', False) for r in rows),
            'invalid_rate': invalid/total if total else 0., 'evaluations': total,
            'mode_results': {mode: {'cases': sum(r['mode'] == mode for r in rows),
                'strict_passes': sum(r['strict_pass'] for r in rows if r['mode'] == mode),
                'noisy_eye_passes': sum(r.get('noisy_eye_pass', False) for r in rows if r['mode'] == mode),
                'invalid': sum(r['invalid'] for r in rows if r['mode'] == mode),
                'evaluations': sum(r['evaluations'] for r in rows if r['mode'] == mode)}
                for mode in ('measured', 'estimated', 'unknown')}, 'rows': rows}


def qualification_cases():
    """Fresh frozen 24-case set, excluded from training/checkpoint selection."""
    import numpy as np
    rng = np.random.default_rng(2026091017)
    result = []
    for mode in ('measured', 'estimated', 'unknown'):
        for reference in ('tx_vpp', 'ctle_input_vrms'):
            for _ in range(4):
                target, channel = float(rng.uniform(5, 11)), float(rng.uniform(8, 16))
                band = (float(rng.uniform(1e7, 1e8)), float(rng.uniform(2.5e9, 5e9)))
                swing = float(rng.uniform(.5, 1))
                request = dict(mode=mode, signal_reference=reference,
                    signal_value_v=swing if reference == 'tx_vpp' else swing*channel_unit_rms(channel, band),
                    bandwidth_hz=band)
                if mode == 'measured':
                    request['value_vrms'] = float(10**rng.uniform(-3, np.log10(.05)))
                elif mode == 'estimated':
                    request['low_vrms'], request['high_vrms'] = sorted(map(float, 10**rng.uniform(-3, np.log10(.05), 2)))
                result.append(dict(id=len(result), target=target, channel=channel,
                    seed=2026191000+len(result), noise=SNRRequest.from_dict(request).to_dict()))
    return result


def regressed(current, initial):
    return (initial['strict_passes']-current['strict_passes'] >= 2
            or current['invalid_rate']-initial['invalid_rate'] >= .20-1e-12)


def evaluate_policy(model, manifest, run_dir, *, label, deadline, legacy=False, fixed=False):
    from eqrl.agents.train_noise_pilot import _write_json
    from eqrl.envs.noise_env import NoiseEqualizerEnv
    from eqrl.circuits.ctle import DesignVars, encode_action
    rows = []
    env = NoiseEqualizerEnv(horizon=4, seed=123, artifact_root=str(run_dir/'raw'/'validation'))
    env.heartbeat_path = run_dir/'validation_worker.json'
    fixed_x = None
    if fixed:
        data = json.loads((run_dir.parents[1]/'results'/'delivered_circuit.json').read_text())
        names = {f.name for f in fields(DesignVars)}
        fixed_x = encode_action(DesignVars(**{k:v for k,v in data['design'].items() if k in names}))
    try:
        for case in manifest:
            if time.monotonic() >= deadline:
                raise TimeoutError('wall_budget during validation')
            env.target_range = (case['target'], case['target'])
            env.channel_range = (case['channel'], case['channel'])
            env._anchor_x = fixed_x
            env.anchor_noise = 0.
            observation, _ = env.reset(seed=case['seed'], options={'noise_request': case['noise']})
            row = {'id': case['id'], 'mode': case['noise']['mode'], 'strict_pass': False, 'noisy_eye_pass': False,
                   'invalid': 0, 'evaluations': 0, 'candidates': []}
            for step in range(1 if fixed else 5):
                if time.monotonic() >= deadline:
                    raise TimeoutError('wall_budget during validation')
                if step:
                    action, _ = model.predict(observation[:18] if legacy else observation, deterministic=True)
                    observation, _, _, _, info = env.step(action)
                    valid, passed = info['sim_ok'], info['passed']
                    measures = info['measures']
                else:
                    valid, passed = env._noise_score is not None, env._noise_passed
                    measures = None
                row['evaluations'] += 1
                row['invalid'] += int(not valid)
                row['strict_pass'] |= bool(passed)
                points = env.noise_details.get('points', [])
                row['noisy_eye_pass'] |= bool(valid and points and all(p['noisy_eye_passed'] for p in points))
                row['candidates'].append({'valid': bool(valid), 'passed': bool(passed),
                    'noise_evaluation': env.noise_details, 'measures': measures})
                _write_json(run_dir/'trainer_heartbeat.json', {'phase': 'validating', 'label': label,
                    'case': case['id'], 'candidate': step, 'updated_at_unix': time.time()})
                # Do not keep acting on a terminal episode. The budget is a maximum.
                if passed:
                    break
            rows.append(row)
            _write_json(run_dir/f'validation_{label}.json', dict(summarize(rows), complete=False))
    finally:
        env.close()
    result = dict(summarize(rows), complete=True, budget_per_case=5,
                  fixed_circuit_evaluated_once=fixed)
    _write_json(run_dir/f'validation_{label}.json', result)
    return result
