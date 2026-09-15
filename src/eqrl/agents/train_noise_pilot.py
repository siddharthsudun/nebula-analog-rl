"""Run a bounded, noise-aware PPO pilot from a fresh policy.

This module is deliberately self-contained because the pilot is an experiment rather
than a replacement for the established sequential trainer.  A run owns one newly
created directory, records the exact rollout rounding used by SB3, and writes progress
often enough for an external watcher to distinguish a live run from a stalled one.

The noise environment is imported inside each worker.  That matters on Windows: the
worker has to configure ngspice before importing the module that eventually loads the
native simulator library.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


DEFAULT_TIMESTEPS = 5_120
MAX_TIMESTEPS = 5_120
FULL_TIMESTEPS = 40_960
FULL_WALL_SECONDS = 64_800.0
DEFAULT_N_ENVS = 4
DEFAULT_HORIZON = 20
DEFAULT_SEED = 2_026_0909
DEFAULT_WALL_SECONDS = 7_200.0
N_STEPS = 256
BATCH_SIZE = 128
CHECKPOINT_EVERY = 1_024
PROGRESS_EVERY = 128

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _set_single_threaded() -> None:
    """Keep each trainer and simulator process from oversubscribing the host."""

    # These variables must be set before numpy/torch is imported in a worker.  Setting
    # them explicitly also makes a parent process with a large inherited thread pool
    # reproducible.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Torch only allows the inter-op pool to be changed before work starts.
            # A worker can arrive here after its bootstrap has already touched torch.
            pass
    except Exception:
        # Training imports torch later and will report an actionable failure if it is
        # actually unavailable.  Thread limiting must not hide that original error.
        pass


def _configure_ngspice() -> None:
    """Apply the Windows ngspice layout used by the other simulator experiments.

    ``os.add_dll_directory`` returns a handle whose lifetime controls the search path.
    Keeping the handle in a module-level list prevents the directory from disappearing
    while a worker still has a resident ngspice DLL.
    """

    profile = Path(os.environ.get("USERPROFILE") or Path.home())
    install = profile / "eqrl-ngspice"
    library_bin = install / "Library" / "bin"
    shim = install / "shim"

    os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(library_bin / "ngspice{}.dll"))
    os.environ.setdefault("SPICE_LIB_DIR", str(install / "Library" / "share" / "ngspice"))
    os.environ.setdefault("PDK_ROOT", str(profile / "pdk"))

    path_entries = [str(shim), str(library_bin)]
    existing = os.environ.get("PATH", "")
    for entry in reversed(path_entries):
        if entry and entry not in existing.split(os.pathsep):
            existing = entry + os.pathsep + existing
    os.environ["PATH"] = existing

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None and library_bin.exists():
        resolved = str(library_bin.resolve())
        if not any(str(getattr(handle, "_name", "")) == resolved
                   for handle in _DLL_DIRECTORY_HANDLES):
            _DLL_DIRECTORY_HANDLES.append(add_dll_directory(resolved))


def _json_default(value: Any) -> Any:
    """Convert common numpy/torch scalar values without importing either package."""

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, collections.Counter):
        return dict(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    """Atomically replace a JSON artifact so a watcher never reads a torn file."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Windows readers can briefly hold a file without delete sharing.
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.02)


def _git_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes(repo_root: Path) -> dict[str, str]:
    """Hash every Python source module that can participate in the env/pilot path."""

    source_root = repo_root / "src" / "eqrl"
    hashes: dict[str, str] = {}
    if not source_root.is_dir():
        return hashes
    for source in sorted(source_root.rglob("*.py")):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        hashes[str(source.relative_to(repo_root)).replace("\\", "/")] = digest
    for relative in ('scripts/supervise_noise_pilot.py', 'server.py', 'dashboard/index.html',
                     'dashboard/static/app.js', 'dashboard/static/snr-input.js', 'dashboard/static/styles.css',
                     'docs/NOISE_PILOT_V2.md'):
        hashes[relative] = hashlib.sha256((repo_root/relative).read_bytes()).hexdigest()
    return hashes


def _provenance(repo_root: Path) -> dict[str, Any]:
    return {
        "git_head": _git_head(repo_root),
        "source_sha256": _source_hashes(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "module": "eqrl.agents.train_noise_pilot",
    }


def _safe_counter_value(value: Any) -> int | float | str | bool | None:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    value = _json_default(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _sum_attr(values: Any) -> Any:
    """Sum scalar VecEnv attributes while preserving dict counters."""

    if not isinstance(values, (list, tuple)):
        return _safe_counter_value(values)
    if not values:
        return 0
    if all(isinstance(v, dict) for v in values):
        merged: collections.Counter[str] = collections.Counter()
        for value in values:
            for key, item in value.items():
                try:
                    merged[str(key)] += int(item)
                except (TypeError, ValueError):
                    continue
        return dict(merged)
    try:
        return _safe_counter_value(sum(values))
    except (TypeError, ValueError):
        return [_safe_counter_value(value) for value in values]


def _env_counters(vec_env: Any) -> dict[str, Any]:
    """Read optional counters from NoiseEqualizerEnv without requiring one schema."""

    result: dict[str, Any] = {}
    get_attr = getattr(vec_env, "get_attr", None)
    if not callable(get_attr):
        return result
    for attr in (
        "n_sims",
        "n_invalid",
        "n_noise_failures",
        "n_noise_evaluations",
    ):
        try:
            result[attr] = _sum_attr(get_attr(attr))
        except (AttributeError, BrokenPipeError, ConnectionError, KeyError,
                OSError, RuntimeError, TypeError, ValueError):
            continue
    return result


def _noise_fields(info: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Extract mode, outcome, and pass/fail fields from compatible info schemas."""

    mode = info.get("noise_mode", info.get("noise_type"))
    outcome = info.get("noise_outcome")
    passed = info.get("noise_pass", info.get("noise_passed", info.get("passed")))
    if passed is None:
        passed = info.get("noise_ok")

    nested = info.get("noise")
    if isinstance(nested, dict):
        mode = nested.get("mode", mode)
        outcome = nested.get("outcome", outcome)
        if passed is None:
            passed = nested.get("pass", nested.get("passed", nested.get("ok")))
    evaluation = info.get("noise_evaluation")
    if isinstance(evaluation, dict):
        outcome = evaluation.get("status", evaluation.get("outcome", outcome))
        mode = evaluation.get("mode", mode)
        if passed is None:
            passed = evaluation.get("pass", evaluation.get("passed", evaluation.get("ok")))
    return mode, outcome, passed


def _make_callback(run_dir: Path, config: dict[str, Any], deadline: float) -> Any:
    """Build the SB3 callback lazily so importing this module stays lightweight."""

    from stable_baselines3.common.callbacks import BaseCallback

    class PilotCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.started = time.monotonic()
            self.callback_steps = 0
            self.next_progress = PROGRESS_EVERY
            self.deadline_hit = False
            self.stop_reason: str | None = None
            self.noise_modes: collections.Counter[str] = collections.Counter()
            self.noise_outcomes: collections.Counter[str] = collections.Counter()
            self.noise_pass = 0
            self.noise_fail = 0
            self.episodes = 0
            self.checkpoints: list[dict[str, Any]] = []

        def _record_infos(self) -> None:
            infos = self.locals.get("infos", [])
            if isinstance(infos, dict):
                infos = [infos]
            dones = self.locals.get("dones", [])
            if isinstance(dones, bool):
                dones = [dones]
            for index, info in enumerate(infos):
                if not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                done = bool(dones[index]) if index < len(dones) else False
                if episode is not None or done:
                    self.episodes += 1
                mode, outcome, passed = _noise_fields(info)
                if passed is None and isinstance(info.get("passed"), bool):
                    passed = info["passed"]
                if mode is not None:
                    self.noise_modes[str(mode)] += 1
                if outcome is not None:
                    self.noise_outcomes[str(outcome)] += 1
                if isinstance(passed, bool):
                    if passed:
                        self.noise_pass += 1
                    else:
                        self.noise_fail += 1

        def _payload(self, status: str = "running") -> dict[str, Any]:
            now = time.monotonic()
            actual = int(getattr(self, "num_timesteps", 0))
            return {
                "schema": "eqrl.noise_pilot.progress.v2",
                "status": status,
                "requested_timesteps": config["timesteps_requested"],
                "effective_timesteps": config["timesteps_effective"],
                "actual_timesteps": actual,
                "callback_steps": self.callback_steps,
                "optimizer_updates": int(getattr(self.model, "_n_updates", 0)),
                "elapsed_s": round(now - self.started, 3),
                "wall_seconds": config["wall_seconds"],
                "deadline_hit": self.deadline_hit,
                "stop_reason": self.stop_reason,
                "episodes": self.episodes,
                "noise_modes": dict(self.noise_modes),
                "noise_outcomes": dict(self.noise_outcomes),
                "noise_pass": self.noise_pass,
                "noise_fail": self.noise_fail,
                "checkpoints": list(self.checkpoints),
                "environment_counters": _env_counters(self.training_env),
            }

        def _dump(self, status: str = "running") -> None:
            _write_json(run_dir / "progress.json", self._payload(status))

        def _on_step(self) -> bool:
            import numpy as np
            for key in ("rewards", "actions", "new_obs"):
                if key in self.locals and not np.all(np.isfinite(self.locals[key])):
                    raise ValueError(f"non-finite training {key}")
            _write_json(run_dir / "trainer_heartbeat.json", {"phase": "training", "updated_at_unix": time.time(),
                "actual_timesteps": int(self.num_timesteps)})
            self.callback_steps += 1
            self._record_infos()
            actual = int(self.num_timesteps)
            if actual >= self.next_progress:
                while actual >= self.next_progress:
                    self.next_progress += PROGRESS_EVERY
                self._dump()

            # Check after recording the latest vector step.  This leaves the progress
            # artifact with the exact stopping point seen by the external watcher.
            if time.monotonic() >= deadline:
                self.deadline_hit = True
                self.stop_reason = "wall_budget"
                self._dump()
                return False
            return True

        def _on_training_end(self) -> None:
            self._dump()

    return PilotCallback()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        "--out",
        dest="run_dir",
        required=True,
        help="new output directory; it must not already exist",
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, choices=range(1, 25), default=DEFAULT_N_ENVS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--warm-start-frozen", action="store_true")
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        default=None,
        help="continue a full-profile run from a checkpoint .zip instead of building "
             "a fresh policy; requires --profile full, mutually exclusive with "
             "--warm-start-frozen",
    )
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=DEFAULT_WALL_SECONDS,
        help="cooperative callback deadline, capped at 7200 seconds; use parent supervisor for stalls",
    )
    parser.add_argument(
        "--smoke",
        "--smoke-steps",
        dest="smoke_steps",
        nargs="?",
        const=128,
        type=int,
        default=None,
        help="run a small pilot (default 128 steps) for wiring checks",
    )
    return parser


def _validated_config(args: argparse.Namespace) -> dict[str, Any]:
    cap = FULL_TIMESTEPS if args.profile == "full" else MAX_TIMESTEPS
    wall_cap = FULL_WALL_SECONDS if args.profile == "full" else DEFAULT_WALL_SECONDS
    if args.warm_start_frozen and args.profile != "full":
        raise ValueError("frozen-policy transfer requires the full profile")
    if args.resume_from and args.warm_start_frozen:
        raise ValueError("--resume-from and --warm-start-frozen are mutually exclusive")
    if args.resume_from and args.profile != "full":
        raise ValueError("--resume-from requires the full profile")
    if args.resume_from and not Path(args.resume_from).expanduser().exists():
        raise ValueError(f"--resume-from checkpoint not found: {args.resume_from}")
    if args.timesteps <= 0 or args.timesteps > cap:
        raise ValueError(f"--timesteps must be in [1, {cap}]")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if not (0.0 < args.wall_seconds <= wall_cap):
        raise ValueError(f"--wall-seconds must be in (0, {wall_cap}]")
    if args.smoke_steps is not None:
        if args.smoke_steps <= 0 or args.smoke_steps > MAX_TIMESTEPS:
            raise ValueError(f"--smoke steps must be in [1, {MAX_TIMESTEPS}]")
        requested = args.smoke_steps
    else:
        requested = args.timesteps

    rollout_steps = N_STEPS * args.n_envs
    if args.resume_from:
        # The already-completed prefix isn't known until the checkpoint loads in
        # main(), which recomputes rollouts_effective against the real resume offset.
        # Rounding the raw target up by rollout_steps here (as if starting fresh)
        # can false-positive over the cap for a target that is in fact reachable.
        effective = min(int(requested), cap)
    else:
        effective = int(math.ceil(requested / rollout_steps) * rollout_steps)
        if effective > cap:
            raise ValueError(f"rollout rounding would exceed the {cap}-step cap")
    return {
        "timesteps_requested": int(requested),
        "timesteps_cli": int(args.timesteps),
        "timesteps_effective": effective,
        "rollout_steps": rollout_steps,
        "rollouts_effective": effective // rollout_steps,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_envs": int(args.n_envs),
        "horizon": int(args.horizon),
        "seed": int(args.seed),
        "wall_seconds": float(args.wall_seconds),
        "smoke_steps": args.smoke_steps,
        "target_range": [5.0, 11.0],
        "channel_range": [8.0, 16.0],
        "fast": False,
        "guarded": True,
        "boost_tol": 1.5,
        "fresh_policy": not args.warm_start_frozen and not args.resume_from,
        "profile": args.profile,
        "warm_start_frozen": args.warm_start_frozen,
        "resume": str(Path(args.resume_from).expanduser()) if args.resume_from else None,
        "noise_model": {
            "location": "external CTLE input after channel",
            "spectrum": "flat PSD from 10 MHz to 5 GHz",
            "mode_sampling": "measured, estimated, unknown uniformly per episode",
            "unknown_range_vrms": [0.001, 0.05],
            "tx_swing_vpp": [.5, 1.0],
            "schema": "eqrl.snr.request.v2",
            "observation_features": 26,
            "band_choices_hz": [[1e7, 5e9], [1e7, 2.5e9], [1e8, 5e9], [1e8, 2.5e9]],
            "signal_references": ["tx_vpp", "ctle_input_vrms"],
            "eye_bits": 512,
            "noise_model": "iid equivalent white slicer noise calibrated from actual onoise_total",
            "scope": "five sampled range points; no BER certification",
        },
    }


def _worker_factory(config, rank, run_dir):
    # Import bootstrap helpers in the child. Serializing __main__ helpers after
    # DLL setup captures unpicklable native handles on Windows.
    def initialize():
        from eqrl.agents.train_noise_pilot import _configure_ngspice, _set_single_threaded
        _set_single_threaded()
        _configure_ngspice()
        from eqrl.envs.noise_env import NoiseEqualizerEnv
        env = NoiseEqualizerEnv(seed=config['seed'] + rank,
            horizon=config['horizon'], target_range=(5.0, 11.0),
            channel_range=(8.0, 16.0), fast=False, guarded=True, boost_tol=1.5,
            artifact_root=str(run_dir / 'raw' / f'env_{rank}'))
        env.heartbeat_path = run_dir / f'worker_{rank}.json'
        env._heartbeat('ready')
        return env
    return initialize


def transfer_nominal_policy(source, destination):
    """Copy learned nominal behavior; zero new observation columns, reset optimizer.

    Only first-layer input width may change. A different policy architecture fails
    explicitly rather than partially loading weights. The source is never mutated.
    """
    import torch
    old, new = source.policy.state_dict(), destination.policy.state_dict()
    if set(old) != set(new):
        raise ValueError("nominal and SNR policy parameter names differ")
    copied = {}
    for name, tensor in new.items():
        prior = old[name]
        if prior.shape == tensor.shape:
            copied[name] = prior.clone()
        elif tensor.ndim == prior.ndim == 2 and tensor.shape[0] == prior.shape[0] and prior.shape[1] == 18 and tensor.shape[1] == 26:
            copied[name] = torch.zeros_like(tensor)
            copied[name][:, :18] = prior
        else:
            raise ValueError(f"incompatible transfer shape for {name}: {prior.shape} -> {tensor.shape}")
    destination.policy.load_state_dict(copied, strict=True)
    destination.policy.optimizer.state.clear()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _validated_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    run_dir = Path(args.run_dir).expanduser()
    try:
        # mkdir(exist_ok=False) is the ownership boundary.  A collision never mutates
        # an older run and is reported before any simulator or PPO process is spawned.
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"run directory already exists: {run_dir}")

    started = time.monotonic()
    callback: Any = None
    vec_env: Any = None
    status = "failed"
    model = None
    error_trace: str | None = None
    config["run_dir"] = str(run_dir.resolve())
    config["started_at_unix"] = time.time()

    try:
        _set_single_threaded()
        _configure_ngspice()
        provenance = _provenance(_REPO_ROOT)
        _write_json(run_dir / "config.json", config)
        _write_json(run_dir / "provenance.json", provenance)

        _write_json(run_dir / "trainer_heartbeat.json", {"phase": "starting", "updated_at_unix": time.time()})
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        factories = [_worker_factory(config, rank, run_dir) for rank in range(config["n_envs"])]
        if config["n_envs"] == 1:
            vec_env = DummyVecEnv(factories)
        else:
            # Spawn is explicit so a simulator handle from the parent cannot leak into
            # a worker.  This is also the native multiprocessing mode on Windows.
            vec_env = SubprocVecEnv(factories, start_method="spawn")

        deadline = started + config["wall_seconds"]
        callback = _make_callback(run_dir, config, deadline)
        if config['resume']:
            resume_path = Path(config['resume'])
            model = PPO.load(str(resume_path), env=vec_env, device='cpu')
            resumed_timesteps = int(model.num_timesteps)
            remaining = config['timesteps_effective'] - resumed_timesteps
            if remaining <= 0:
                raise ValueError(
                    f"checkpoint {resume_path} is already at {resumed_timesteps} steps, "
                    f">= effective target {config['timesteps_effective']}")
            config['rollouts_effective'] = math.ceil(remaining / config['rollout_steps'])
            config['resumed_timesteps'] = resumed_timesteps
            _write_json(run_dir / 'config.json', config)
            _write_json(run_dir / 'initialization.json', {
                'source': str(resume_path), 'source_sha256': hashlib.sha256(resume_path.read_bytes()).hexdigest(),
                'resumed_timesteps': resumed_timesteps, 'new_snr_columns': 'n/a (resume)',
                'optimizer': 'restored from checkpoint', 'source_modified': False})
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                seed=config["seed"],
                verbose=0,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                gamma=0.95,
                gae_lambda=0.95,
                ent_coef=0.005,
                learning_rate=3e-4,
                tensorboard_log=None,
            )
            if config['warm_start_frozen']:
                nominal_path = _REPO_ROOT / 'results' / 'seq_clean40k.zip'
                transfer_nominal_policy(PPO.load(str(nominal_path), device='cpu'), model)
                _write_json(run_dir / 'initialization.json', {
                    'source': str(nominal_path), 'source_sha256': hashlib.sha256(nominal_path.read_bytes()).hexdigest(),
                    'new_snr_columns': 'zero', 'optimizer': 'fresh', 'source_modified': False})
        model.save(str(run_dir / "policy_initial.zip"))
        callback.init_callback(model)
        from eqrl.experiments.noise_holdout import cases, evaluate_policy, regressed
        import torch
        manifest = cases()
        _write_json(run_dir / 'holdout_manifest.json', manifest)
        comparisons = {}
        def validate(policy, label, **kw):
            _write_json(run_dir / 'trainer_heartbeat.json', {'phase': 'validating',
                'label': label, 'updated_at_unix': time.time()})
            evaluation = evaluate_policy(policy, manifest, run_dir, label=label, deadline=deadline, **kw)
            comparisons[label] = {k: v for k, v in evaluation.items() if k != 'rows'}
            _write_json(run_dir / 'comparison.json', {'scope': 'paired monitoring set, not production qualification',
                'strict_tolerance_db': 1.5, 'budget_per_case': 5, 'results': comparisons})
            return evaluation
        initial = validate(model, 'initial')
        best_key = (initial['strict_passes'], -initial['invalid_rate'])
        model.save(str(run_dir / 'policy_best.zip'))
        _write_json(run_dir / 'best_validation.json', {'label': 'initial', 'metrics': initial})
        frozen = PPO.load(str(_REPO_ROOT / 'results' / 'seq_clean40k.zip'), device='cpu')
        validate(frozen, 'frozen', legacy=True)
        validate(None, 'fixed', fixed=True)
        regression_streak = 0
        status = 'budget_stopped'
        for rollout in range(config['rollouts_effective']):
            if time.monotonic() >= deadline:
                callback.deadline_hit = True
                callback.stop_reason = 'wall_budget'
                break
            # Workers intentionally idle during held-out scoring. Refresh their
            # phase boundary before asking the watchdog to track them again.
            vec_env.env_method('_heartbeat', 'training_ready')
            _write_json(run_dir / 'trainer_heartbeat.json', {'phase': 'training', 'updated_at_unix': time.time()})
            model.learn(total_timesteps=config['rollout_steps'], progress_bar=False,
                        callback=callback, reset_num_timesteps=False)
            if any(not torch.isfinite(v).all() for v in model.policy.state_dict().values()):
                raise ValueError('non-finite policy parameters after optimizer update')
            actual = int(model.num_timesteps)
            checkpoint = run_dir / f'checkpoint_{actual:06d}_steps.zip'
            model.save(str(checkpoint))
            callback.checkpoints.append({'actual_timesteps': actual, 'path': checkpoint.name,
                'optimizer_updates': int(model._n_updates), 'after_optimizer_update': not callback.deadline_hit})
            callback._dump()
            if callback.deadline_hit:
                break
            evaluation = validate(model, f'{actual:06d}')
            regression_streak = regression_streak+1 if regressed(evaluation, initial) else 0
            _write_json(run_dir / 'regression.json', {'consecutive_regressions': regression_streak,
                'recorded_at_unix': time.time(),
                'strict_pass_drop_threshold': 2, 'invalid_rate_increase_threshold': .20,
                'initial': {k: initial[k] for k in ('strict_passes', 'invalid_rate')},
                'current': {k: evaluation[k] for k in ('strict_passes', 'invalid_rate')}})
            key = (evaluation['strict_passes'], -evaluation['invalid_rate'])
            if key > best_key:
                best_key = key
                model.save(str(run_dir / 'policy_best.zip'))
                _write_json(run_dir / 'best_validation.json', {'label': str(actual), 'metrics': evaluation})
            if regression_streak >= 2:
                status, callback.stop_reason = 'regression_stopped', 'two_consecutive_holdout_regressions'
                break
            if actual >= config['timesteps_effective']:
                status, callback.stop_reason = 'completed', 'target_reached'
        if status == 'budget_stopped':
            callback.deadline_hit = True
            callback.stop_reason = 'wall_budget'

        # The final model is saved only after a successful learn return.  Checkpoints
        # are already inside this same run directory.
        model.save(str(run_dir / "policy_final.zip"))
        if config['profile'] == 'full' and status == 'completed':
            from eqrl.experiments.noise_holdout import qualification_cases
            qualification = qualification_cases()
            _write_json(run_dir / 'qualification_manifest.json', qualification)
            qualification_results = {}
            for label, policy, options in (
                    ('qualified_best', PPO.load(str(run_dir / 'policy_best.zip'), device='cpu'), {}),
                    ('qualified_frozen', frozen, {'legacy': True}),
                    ('qualified_fixed', None, {'fixed': True})):
                evaluation = evaluate_policy(policy, qualification, run_dir, label=label,
                                             deadline=deadline, **options)
                qualification_results[label] = {k: v for k, v in evaluation.items() if k != 'rows'}
                _write_json(run_dir / 'qualification.json', {'results': qualification_results,
                    'production_promoted': False, 'complete': len(qualification_results) == 3})
    except TimeoutError:
        status = "budget_stopped"
        if callback is not None:
            callback.deadline_hit, callback.stop_reason = True, "wall_budget"
        if model is not None:
            model.save(str(run_dir / "policy_partial.zip"))
    except BaseException:
        status = "failed"
        if error_trace is None:
            error_trace = traceback.format_exc()
        else:
            error_trace = error_trace + "\n" + traceback.format_exc()
        # Keep the traceback visible to the parent supervisor and in the run directory.
        sys.stderr.write(error_trace)
        if not error_trace.endswith("\n"):
            sys.stderr.write("\n")
        try:
            (run_dir / "error.txt").write_text(error_trace, encoding="utf-8")
        except OSError:
            pass
    finally:
        # Capture the final counters while the VecEnv workers are still reachable.  The
        # close below intentionally follows this dump so SubprocVecEnv.get_attr cannot
        # turn an otherwise successful run into a torn final progress artifact.
        if callback is not None:
            try:
                callback._dump(status)
            except BaseException:
                dump_trace = traceback.format_exc()
                if status != "failed":
                    status = "failed"
                    error_trace = dump_trace
                sys.stderr.write(dump_trace)

        try:
            _write_json(run_dir / 'trainer_heartbeat.json', {'phase': 'closing', 'updated_at_unix': time.time()})
        except OSError:
            pass
        if vec_env is not None:
            try:
                vec_env.close()
            except BaseException:
                close_trace = traceback.format_exc()
                if status != "failed":
                    status = "failed"
                    error_trace = close_trace
                sys.stderr.write(close_trace)

        actual = int(getattr(callback, "num_timesteps", 0)) if callback is not None else 0
        callback_steps = int(getattr(callback, "callback_steps", 0)) if callback is not None else 0
        deadline_hit = bool(getattr(callback, "deadline_hit", False)) if callback is not None else False
        stop_reason = getattr(callback, "stop_reason", None) if callback is not None else None
        if status == "failed" and stop_reason is None:
            stop_reason = "exception"
        elapsed = round(time.monotonic() - started, 3)
        final = {
            "schema": "eqrl.noise_pilot.status.v2",
            "status": status,
            "completed": status == "completed",
            "budget_stopped": status == "budget_stopped",
            "failed": status == "failed",
            "requested_timesteps": config["timesteps_requested"],
            "effective_timesteps": config["timesteps_effective"],
            "actual_timesteps": actual,
            "callback_steps": callback_steps,
            "elapsed_s": elapsed,
            "wall_seconds": config["wall_seconds"],
            "deadline_hit": deadline_hit,
            "stop_reason": stop_reason,
            "error_traceback": error_trace,
        }
        try:
            _write_json(run_dir / "status.json", final)
        except OSError as exc:
            sys.stderr.write(f"could not write final run status: {exc}\n")

    return 0 if status == "completed" else 2 if status in {"budget_stopped", "regression_stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
