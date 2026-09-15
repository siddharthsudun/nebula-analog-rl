"""Own one bounded pilot process tree; never touch other sessions' processes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def health_reason(run_dir, *, now, started, deadline, n_envs=4, stall_seconds=300):
    if now >= deadline:
        return 'hard_deadline_stopped'
    trainer = load_json(run_dir / 'trainer_heartbeat.json')
    phase = trainer.get('phase', 'starting')
    if now - trainer.get('updated_at_unix', started) > stall_seconds:
        return 'trainer_stalled'
    if phase == 'training':
        for rank in range(n_envs):
            worker = load_json(run_dir / f'worker_{rank}.json')
            if now - worker.get('updated_at_unix', started) > stall_seconds:
                return f'worker_{rank}_stalled'
    if phase == 'validating':
        worker = load_json(run_dir / 'validation_worker.json')
        if now - max(worker.get('updated_at_unix', started), trainer.get('updated_at_unix', started)) > stall_seconds:
            return 'validation_stalled'
    regression = load_json(run_dir / 'regression.json')
    if (regression.get('consecutive_regressions', 0) >= 2 and phase not in ('closing',)
            and now-regression.get('recorded_at_unix', started) > 30):
        return 'regression_stopped'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--control-dir', required=True)
    parser.add_argument('--wall-seconds', type=int, default=7200)
    parser.add_argument('--preflight', action='store_true', help='run real four-worker verification instead of the pilot')
    parser.add_argument('--full', action='store_true', help='40960 steps, optional policy only, frozen-policy transfer')
    args = parser.parse_args()
    cap = 64800 if args.full else 7200
    if not 30 <= args.wall_seconds <= cap:
        parser.error(f'wall budget must be 30 to {cap} seconds')
    control = Path(args.control_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        parser.error('run directory must be fresh')
    control.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'eqrl.agents.train_noise_pilot',
        '--run-dir', str(run_dir), '--timesteps', '5120', '--n-envs', '4',
        '--wall-seconds', str(args.wall_seconds - 15)]
    if args.preflight:
        command = [sys.executable, '-m', 'eqrl.experiments.noise_v2_preflight', '--run-dir', str(run_dir)]
    elif args.full:
        command = [sys.executable, '-m', 'eqrl.agents.train_noise_pilot',
            '--run-dir', str(run_dir), '--timesteps', '40960', '--n-envs', '4',
            '--profile', 'full', '--warm-start-frozen', '--wall-seconds', str(args.wall_seconds-15)]
    started = time.time()
    status_path = control / 'supervisor.json'
    def record(payload):
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temp.replace(status_path)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with (control / 'stdout.log').open('x', encoding='utf-8') as stdout, (control / 'stderr.log').open('x', encoding='utf-8') as stderr:
        proc = subprocess.Popen(command, cwd=repo, stdout=stdout, stderr=stderr,
            creationflags=flags, env=os.environ.copy())
        payload = {'status': 'running', 'pid': proc.pid, 'supervisor_pid': os.getpid(),
            'started_at_unix': started, 'deadline_unix': started + args.wall_seconds,
            'run_dir': str(run_dir), 'command': command, 'poll_seconds': 15, 'stall_seconds': 300}
        record(payload)
        while proc.poll() is None:
            now = time.time()
            reason = health_reason(run_dir, now=now, started=started, deadline=started+args.wall_seconds)
            payload.update(updated_at_unix=now, elapsed_seconds=now-started,
                           trainer=load_json(run_dir/'trainer_heartbeat.json'),
                           progress=load_json(run_dir/'progress.json'))
            if reason:
                # Only the owned child and descendants, never name-matched sessions.
                killed = subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    capture_output=True, text=True, creationflags=flags, timeout=20)
                payload.update(status=reason if killed.returncode == 0 else 'termination_failed',
                    termination_exit_code=killed.returncode, termination_output=killed.stdout+killed.stderr)
                final = load_json(run_dir/'status.json')
                final.update(status=payload['status'], completed=False, stop_reason=reason,
                    actual_timesteps=payload.get('progress', {}).get('actual_timesteps', 0),
                    supervised=True)
                if run_dir.is_dir():
                    (run_dir/'status.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
                break
            record(payload)
            try:
                proc.wait(timeout=min(15, max(.1, started+args.wall_seconds-time.time())))
            except subprocess.TimeoutExpired:
                pass
        if payload['status'] == 'running':
            final = load_json(run_dir/'status.json')
            payload.update(status=final.get('status', 'failed'), exit_code=proc.returncode)
        payload.update(elapsed_seconds=time.time()-started, updated_at_unix=time.time())
        record(payload)


if __name__ == '__main__':
    main()
