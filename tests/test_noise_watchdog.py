import importlib.util
import json
from pathlib import Path
from silq.experiments.noise_holdout import regressed, cases

spec = importlib.util.spec_from_file_location('noise_watchdog', Path(__file__).parents[1]/'scripts'/'supervise_noise_pilot.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def write(root, name, data):
    (root/name).write_text(json.dumps(data))


def test_watchdog_stall_and_deadline(tmp_path):
    assert watchdog.health_reason(tmp_path, now=301, started=0, deadline=7200) == 'trainer_stalled'
    write(tmp_path, 'trainer_heartbeat.json', {'phase':'training','updated_at_unix':310})
    for i in range(4):
        write(tmp_path, f'worker_{i}.json', {'updated_at_unix':310})
    assert watchdog.health_reason(tmp_path, now=311, started=0, deadline=7200) is None
    write(tmp_path, 'worker_2.json', {'updated_at_unix':0})
    assert watchdog.health_reason(tmp_path, now=311, started=0, deadline=7200) == 'worker_2_stalled'
    assert watchdog.health_reason(tmp_path, now=7200, started=0, deadline=7200) == 'hard_deadline_stopped'


def test_validation_can_idle_training_workers(tmp_path):
    write(tmp_path, 'trainer_heartbeat.json', {'phase':'validating','updated_at_unix':400})
    assert watchdog.health_reason(tmp_path, now=401, started=0, deadline=7200) is None


def test_regression_is_not_negative_reward():
    initial = {'strict_passes':5,'invalid_rate':.2}
    assert not regressed({'strict_passes':4,'invalid_rate':.21,'reward':-5}, initial)
    assert regressed({'strict_passes':3,'invalid_rate':.2}, initial)
    assert regressed({'strict_passes':5,'invalid_rate':.4}, initial)
    manifest = cases()
    assert len(manifest) == 12
    assert all(sum(c['noise']['mode']==mode for c in manifest)==4 for mode in ('measured','estimated','unknown'))


def test_nonfinite_reward_stops_callback(tmp_path):
    import numpy as np
    import gymnasium as gym
    from stable_baselines3 import PPO
    from silq.agents.train_noise_pilot import _make_callback
    import time
    import pytest
    class Broken(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., (26,))
        action_space = gym.spaces.Box(-1., 1., (6,))
        def reset(self, *, seed=None, options=None):
            return np.zeros(26, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(26, dtype=np.float32), float('nan'), False, False, {}
    model = PPO('MlpPolicy', Broken(), n_steps=2, batch_size=2)
    cb = _make_callback(tmp_path, {'timesteps_requested':2, 'timesteps_effective':2,'wall_seconds':60}, time.monotonic()+60)
    with pytest.raises(ValueError, match='non-finite training rewards'):
        model.learn(2, callback=cb)


def test_callback_cooperative_deadline_keeps_finite_negative_rewards(tmp_path):
    import numpy as np
    import gymnasium as gym
    from stable_baselines3 import PPO
    from silq.agents.train_noise_pilot import _make_callback
    import time
    class Healthy(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., (26,))
        action_space = gym.spaces.Box(-1., 1., (6,))
        def reset(self, *, seed=None, options=None):
            return np.zeros(26, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(26, dtype=np.float32), -5., False, False, {}
    model = PPO('MlpPolicy', Healthy(), n_steps=2, batch_size=2)
    cb = _make_callback(tmp_path, {'timesteps_requested':2, 'timesteps_effective':2,'wall_seconds':60}, time.monotonic()-1)
    model.learn(2, callback=cb)
    assert cb.deadline_hit and cb.stop_reason == 'wall_budget'


def test_supervisor_kills_only_owned_tree_at_deadline(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    control, run = tmp_path/'control', tmp_path/'run'
    calls = []
    class Child:
        pid = 4242
        returncode = None
        def poll(self): return None
    monkeypatch.setattr(watchdog.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(watchdog.subprocess, 'run', lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0, stdout='', stderr=''))
    times = iter([0, 31, 31, 31])
    monkeypatch.setattr(watchdog.time, 'time', lambda: next(times))
    monkeypatch.setattr(sys, 'argv', ['supervisor', '--run-dir',str(run),'--control-dir',str(control),'--wall-seconds','30'])
    watchdog.main()
    assert calls == [['taskkill','/PID','4242','/T','/F']]
    assert json.loads((control/'supervisor.json').read_text())['status'] == 'hard_deadline_stopped'


def test_supervisor_reports_child_crash(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    control, run = tmp_path/'control', tmp_path/'run'
    monkeypatch.setattr(watchdog.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=4242, returncode=1, poll=lambda:1))
    monkeypatch.setattr(sys, 'argv', ['supervisor', '--run-dir',str(run),'--control-dir',str(control)])
    watchdog.main()
    result = json.loads((control/'supervisor.json').read_text())
    assert result['status'] == 'failed' and result['exit_code'] == 1
