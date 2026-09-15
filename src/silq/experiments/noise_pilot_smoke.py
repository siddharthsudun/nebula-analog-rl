"""Bounded real-SPICE preflight for the separately versioned noise pilot."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out)
    # Claim a fresh artifact before spending any simulation time.
    with path.open('x', encoding='utf-8') as handle:
        json.dump({'status': 'running'}, handle)
    from silq.agents.train_noise_pilot import _configure_ngspice, _set_single_threaded
    _configure_ngspice()
    _set_single_threaded()
    import numpy as np
    from silq.circuits.ctle import DesignVars
    from silq.envs.noise_env import NoiseEqualizerEnv
    start = time.monotonic()
    payload = {'status': 'running', 'checks': []}
    env = None
    try:
        data = json.loads(Path('results/delivered_circuit.json').read_text())
        names = {f.name for f in fields(DesignVars)}
        dv = DesignVars(**{key: value for key, value in data['design'].items() if key in names})
        target = data['spec']['target_boost_db']
        channel = data['spec']['channel_loss_db']
        env = NoiseEqualizerEnv(anchor_design=dv, target_range=(target, target),
            channel_range=(channel, channel), horizon=1, seed=91371)
        requests = [
            {'mode': 'specific', 'value_vrms': 0},
            {'mode': 'specific', 'value_vrms': .05},
            {'mode': 'range', 'low_vrms': .001, 'high_vrms': .05},
            {'mode': 'unknown'},
        ]
        for request in requests:
            observation, info = env.reset(seed=91371, options={'noise_request': request})
            assert env.observation_space.contains(observation)
            detail = info['noise_evaluation']
            assert detail['status'] == 'measured', detail
            assert detail['receiver_output_noise_vrms'] > 0
            payload['checks'].append({'request': request, 'evaluation': detail})
        quiet = payload['checks'][0]['evaluation']['points'][0]
        loud = payload['checks'][1]['evaluation']['points'][0]
        assert loud['output_noise_vrms'] > quiet['output_noise_vrms']
        assert loud['sample_phase'] == quiet['sample_phase']
        assert loud['dfe_tap'] == quiet['dfe_tap']
        assert loud['eye_v_mv'] < quiet['eye_v_mv']
        _, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(reward)
        assert truncated
        payload['step'] = {'reward': reward, 'terminated': terminated, 'truncated': truncated,
            'passed': info['passed'], 'sim_ok': info['sim_ok']}
        payload['status'] = 'passed'
    except Exception as exc:
        payload['status'] = 'failed'
        payload['error'] = repr(exc)
        raise
    finally:
        if env is not None:
            env.close()
        payload['wall_seconds'] = time.monotonic() - start
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'wall_seconds': payload['wall_seconds']}))


if __name__ == '__main__':
    main()
