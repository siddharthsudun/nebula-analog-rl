"""Production PVT sizing repair with isolated search and independent acceptance."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
MAX_EVALUATIONS = 545
WALL_SECONDS = 900


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def search(output, evaluator=None):
    import cma
    import numpy as np
    from silq.circuits.ctle import DesignVars, encode_action, decode_action
    from silq.envs.pvt import corner_grid
    from silq.pvt_refinement import PVTEvaluator, summarize, full_grid_pass, rank_key, stress_corners
    from silq.specs import Spec
    config = json.loads((output/'config.json').read_text())
    spec = Spec(**config['spec'])
    grid = corner_grid(spec, 'full')
    ev = evaluator or PVTEvaluator(spec, output/'raw', limit=500)
    initial = config['initial']
    baseline = ev.evaluate_batch_parallel([initial[0]], grid)
    if not full_grid_pass(baseline['input'], grid):
        baseline.update(ev.evaluate_batch_parallel(initial[1:], grid))
    considered = [c for c in initial if c['id'] in baseline]
    write(output/'baselines.json', {c['id']:dict(candidate=c, rows=baseline[c['id']],
          summary=summarize(baseline[c['id']])) for c in considered})
    clean = [c for c in considered if full_grid_pass(baseline[c['id']], grid)]
    full = dict(baseline)
    if not clean:
        stresses = stress_corners([r for rs in baseline.values() for r in rs], grid, limit=8)
        write(output/'search_corners.json', stresses)
        best = min(considered, key=lambda c:rank_key(baseline[c['id']]))
        es = cma.CMAEvolutionStrategy(encode_action(DesignVars(**best['design'])).astype(float), .035,
            {'bounds':[0,1], 'popsize':8, 'seed':config['seed'], 'verbose':-9, 'maxiter':5})
        pool = []
        # Same continuous xs from CMA-ES often decode to the identical quantized
        # design, especially as sigma shrinks in later generations, and a finalist
        # can coincide with a design already scored at full-grid (baseline/an
        # earlier finalist). Caching by the decoded design is an exact-match reuse
        # of a real SPICE result already computed for that design and corner set --
        # not an approximation, so it changes zero verdicts, only skips redundant
        # simulation of a design we already have real rows for.
        def design_key(design):
            return json.dumps(design, sort_keys=True)
        stress_cache = {}
        for generation in range(5):
            xs = es.ask()
            batch = [dict(id=f'g{generation}_c{i}', origin='PVT CMA-ES sizing repair',
                          design=asdict(decode_action(x))) for i,x in enumerate(xs)]
            keys = [design_key(c['design']) for c in batch]
            to_eval = [c for c,k in zip(batch,keys) if k not in stress_cache]
            if to_eval:
                fresh = ev.evaluate_batch_parallel(to_eval, stresses)
                for c in to_eval:
                    stress_cache[design_key(c['design'])] = fresh[c['id']]
            rows = {c['id']: stress_cache[k] for c,k in zip(batch,keys)}
            ranked = sorted(range(8), key=lambda i:rank_key(rows[batch[i]['id']]))
            costs = np.empty(8)
            for rank,i in enumerate(ranked): costs[i] = rank
            es.tell(xs, costs.tolist())
            records = [dict(candidate=c, rows=rows[c['id']], summary=summarize(rows[c['id']])) for c in batch]
            pool.extend(records)
            write(output/f'generation_{generation}.json', records)
        full_cache = {design_key(c['design']): full[c['id']] for c in considered}
        finalists = [r['candidate'] for r in sorted(pool, key=lambda r:rank_key(r['rows']))[:2]]
        keys = [design_key(c['design']) for c in finalists]
        to_eval = [c for c,k in zip(finalists,keys) if k not in full_cache]
        if to_eval:
            fresh = ev.evaluate_batch_parallel(to_eval, grid)
            for c in to_eval:
                full_cache[design_key(c['design'])] = fresh[c['id']]
        full.update({c['id']: full_cache[k] for c,k in zip(finalists,keys)})
        considered.extend(finalists)
        clean = [c for c in considered if full_grid_pass(full[c['id']], grid)]
    winner = min(clean or considered, key=lambda c:rank_key(full[c['id']]))
    write(output/'selection.json', dict(winner=winner, provisionally_full_grid_passed=bool(clean),
        summary=summarize(full[winner['id']]), evaluations=ev.evaluations,
        corner_integrity_tt_ss_passed=ev.corner_integrity_passed,
        full_candidates={c['id']:dict(candidate=c, rows=full[c['id']], summary=summarize(full[c['id']])) for c in considered}))


def verification(output):
    # The same existing independent 45-corner acceptance and export contract.
    from silq.experiments.pvt_optimize import verify
    verify(output)


def run_repair(design, spec, *, output=None, seed=20260910, wall_seconds=WALL_SECONDS, notify=None):
    """Return verified result or explicit unverified state, never promote partial grids."""
    if not 1 <= wall_seconds <= WALL_SECONDS:
        raise ValueError('PVT wall budget must be between 1 and 900 seconds')
    if os.environ.get('SILQ_PVT_BACKEND', 'resident') == 'resident':
        return run_resident_repair(design, spec, output=output, seed=seed,
                                   wall_seconds=wall_seconds, notify=notify)
    output = Path(output) if output else ROOT/'results'/'pipeline_pvt'/uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    anchor = json.loads((ROOT/'results/delivered_circuit.json').read_text())['design']
    config = dict(spec=asdict(spec), seed=seed, wall_seconds=wall_seconds, evaluation_limit=MAX_EVALUATIONS,
        initial=[dict(id='input', origin='pipeline PPO/G3.2 output', design=design),
                 dict(id='anchor', origin='fixed delivered sizing, remeasured for this specification', design=anchor)])
    write(output/'config.json', config)
    source_paths = ['src/silq/pvt_repair.py','src/silq/pvt_refinement.py','src/silq/pipeline.py',
                    'src/silq/experiments/pvt_optimize.py','src/silq/specs.py','src/silq/guards.py',
                    'src/silq/circuits/ctle.py','src/silq/sim/eye.py','results/delivered_circuit.json',
                    'results/seq_clean40k.zip']
    write(output/'provenance.json', {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths})
    started = time.monotonic()
    result = dict(accepted=False, status='incomplete', artifact_dir=str(output.resolve()),
                  scope='schematic-level 45-corner PVT; no mismatch, extracted parasitics or SNR certification',
                  evaluations=0, cost_complete=False, workers=[])
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + env.get('PYTHONPATH','')
    for phase in ('search','verify'):
        remaining = wall_seconds - (time.monotonic()-started)
        if remaining <= 0:
            result['status'] = 'budget_exhausted'
            break
        if notify: notify('pvt', 'PVT: '+('checking corners and repairing sizing' if phase=='search' else 'independent full-grid verification'))
        with (output/f'{phase}.log').open('w',encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable,'-m','silq.pvt_repair','--worker',phase,'--output',str(output.resolve())],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'], capture_output=True, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) if os.name=='nt' else proc.kill()
                proc.wait()
                code = 'wall_budget'
        result['workers'].append(dict(phase=phase, returncode=code))
        if code != 0:
            result['status'] = 'budget_exhausted' if code=='wall_budget' else 'failed'
            break
        metrics = json.loads((output/f'{phase}_cost.json').read_text())
        result['evaluations'] += metrics['evaluations']
        for key in ('measure_all','analysis'):
            result[key] = result.get(key,0) + metrics[key]
    else:
        check = json.loads((output/'verification.json').read_text())
        selection = json.loads((output/'selection.json').read_text())
        # Revalidate the grid in the caller before accepting a child artifact.
        from silq.envs.pvt import corner_grid
        from silq.pvt_refinement import full_grid_pass
        result.update(accepted=check['accepted'] and full_grid_pass(check['rows'],corner_grid(spec,'full')),
                      verification=check, selection=selection, cost_complete=True)
        result['status'] = 'verified' if result['accepted'] else 'unresolved_within_budget'
    result['elapsed_seconds'] = time.monotonic()-started
    write(output/'result.json', result)
    return result


def run_resident_repair(design, spec, *, output=None, seed=20260910,
                        wall_seconds=WALL_SECONDS, notify=None):
    """Same search/acceptance algorithm on disjoint resident worker banks."""
    import queue
    from silq.pvt_workers import get_pool, ResidentEvaluator, WorkerFailure
    from silq.envs.pvt import corner_grid
    from silq.pvt_refinement import full_grid_pass
    from silq.experiments.pvt_optimize import verify
    output = Path(output) if output else ROOT/'results'/'pipeline_pvt'/uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    anchor = json.loads((ROOT/'results/delivered_circuit.json').read_text())['design']
    config = dict(spec=asdict(spec), seed=seed, wall_seconds=wall_seconds,
                  evaluation_limit=MAX_EVALUATIONS,
                  initial=[dict(id='input', origin='pipeline PPO/G3.2 output', design=design),
                           dict(id='anchor', origin='fixed delivered sizing, remeasured for this specification', design=anchor)])
    write(output/'config.json', config)
    source_paths = ['src/silq/pvt_repair.py', 'src/silq/pvt_refinement.py', 'src/silq/pvt_workers.py',
                    'src/silq/pipeline.py', 'src/silq/experiments/pvt_optimize.py', 'src/silq/specs.py',
                    'src/silq/guards.py', 'src/silq/evaluator.py', 'src/silq/sim/measures.py',
                    'src/silq/sim/server.py', 'src/silq/sim/probe.py', 'src/silq/sim/eye.py',
                    'src/silq/circuits/ctle.py', 'results/delivered_circuit.json']
    write(output/'provenance.json', {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths})
    started = time.monotonic()
    result = dict(accepted=False, status='incomplete', artifact_dir=str(output.resolve()),
                  scope='schematic-level 45-corner PVT; no mismatch, extracted parasitics or SNR certification',
                  evaluations=0, cost_complete=False, workers=[], backend='resident', measurement_cache=False)
    try:
        if notify:
            notify("pvt", "Preparing independent PVT workers.")
        pool = get_pool(deadline=started + wall_seconds)
        result['pool_wait_seconds'] = time.monotonic() - started
        result['pool_startup_seconds'] = pool.startup_seconds
        deadline = started + wall_seconds
        if time.monotonic() >= deadline:
            raise queue.Empty()
        for phase in ('search', 'verify'):
            if notify:
                notify('pvt', 'PVT: '+('full-grid search with resident workers' if phase=='search'
                                      else 'fresh acceptance in independent resident workers'))
            ev = ResidentEvaluator(pool, phase, spec, output/('raw' if phase=='search' else 'verification_raw'),
                                   500 if phase=='search' else 45, deadline)
            (search if phase=='search' else verify)(output, evaluator=ev)
            counts = dict(evaluations=ev.evaluations, measure_all=ev.measure_all, analysis=ev.analysis)
            write(output/f'{phase}_cost.json', counts)
            result['workers'].append(dict(phase=phase, returncode=0))
            for key, value in counts.items():
                result[key] = result.get(key, 0) + value
        check = json.loads((output/'verification.json').read_text())
        selection = json.loads((output/'selection.json').read_text())
        search_pids = {r['worker_pid'] for c in selection['full_candidates'].values() for r in c['rows']}
        verify_pids = {r['worker_pid'] for r in check['rows']}
        independent = not bool(search_pids & verify_pids)
        result.update(accepted=check['accepted'] and independent and full_grid_pass(check['rows'],corner_grid(spec,'full')),
                      verification=check, selection=selection, cost_complete=True,
                      independent_worker_processes=independent)
        result['status'] = 'verified' if result['accepted'] else 'unresolved_within_budget'
    except queue.Empty:
        result['status'] = 'budget_exhausted'
    except (WorkerFailure, BrokenPipeError, OSError) as exc:
        result.update(status='failed', error=str(exc))
    result['elapsed_seconds'] = time.monotonic() - started
    write(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['search','verify'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from silq.agents.train_noise_pilot import _configure_ngspice, _set_single_threaded
    _set_single_threaded(); _configure_ngspice()
    from silq.simcount import counting
    with counting() as counts:
        (search if args.worker=='search' else verification)(args.output)
    record = json.loads((args.output/('selection.json' if args.worker=='search' else 'verification.json')).read_text())
    write(args.output/f'{args.worker}_cost.json', dict(counts, evaluations=record['evaluations']))




def apply_pvt_stage(result, spec, *, output=None, seed=20260910, wall_seconds=WALL_SECONDS, notify=None, repair_result=None):
    """Third design stage: its verified sizing becomes the actual returned circuit."""
    from copy import deepcopy
    if not result.get('design'):
        result['architecture'] += ' -> PVT sizing repair + independent 45-corner verification'
        result['pvt'] = dict(accepted=False, status='no_candidate', evaluations=0, cost_complete=True)
        return result
    result['pre_pvt'] = {k:deepcopy(result.get(k)) for k in ('design','verification','status','solver','guidance')}
    repair = repair_result if repair_result is not None else run_repair(result['design'], spec, output=output, seed=seed, wall_seconds=wall_seconds, notify=notify)
    # Fastest certifies the three stress corners (tt/ss/ff) so it can still answer inside a
    # five-second budget; everything else certifies all 45. Every sentence this function
    # writes into the result is read by an engineer deciding whether the circuit is signed
    # off, so the count comes from the grid that actually ran -- never a constant. A
    # `run_repair` result predates the label and is always the full grid.
    label = repair.get('grid_label', 'full 45-corner')
    result['architecture'] += f' -> PVT sizing repair + independent {label} verification'
    result['pvt'] = repair
    cost = result.setdefault('cost', {})
    cost['pvt_corner_evaluations'] = repair['evaluations']
    cost['pvt_cost_complete'] = repair['cost_complete']
    cost['measure_all_pvt'] = repair.get('measure_all',0)
    cost['spice_analyses_pvt'] = repair.get('analysis',0)
    cost['measure_all_total'] = cost.get('measure_all_total',0) + cost['measure_all_pvt']
    cost['spice_analyses_total'] = cost.get('spice_analyses_total',0) + cost['spice_analyses_pvt']
    if not repair['accepted']:
        result['status'] = 'pvt_not_verified'
        result['overall_passed'] = False
        result['guidance'] = dict(headline='PVT acceptance not established within the repair budget.',
            detail='The returned nominal candidate is retained for diagnosis. Search exhaustion is not proof of physical infeasibility.')
        return result
    verified = repair['verification']
    winner = verified['candidate']
    result['design'] = winner['design']
    result['netlist'] = (Path(repair['artifact_dir'])/'verified_candidate.cir').read_text()
    tt = next(r for r in verified['rows'] if r['corner'][0]=='tt' and r['corner'][2]==27 and abs(r['corner'][1]-spec.vdd_nominal)<1e-6)
    result['verification'] = dict(guard_valid=True, guard_check=None, passed=True, checks=tt['checks'],
        failing=[], measures=tt['measures'], abs_err_db=tt['target_error_db'],
        scope=f'TT row from the independent {label} verification of the final circuit')
    from silq.pipeline import requirements_diff, competition_spec
    if requirements_diff(spec):
        from silq.sim.measures import Measures
        from silq.specs import hard_pass
        ok, checks = hard_pass(Measures(**tt['measures']),competition_spec(spec))
        result['verification'].update(competition_passed=ok, competition_checks=checks,
                                      competition_failing=[k for k,v in checks.items() if not v])
    result['status'] = 'solved'
    result['overall_passed'] = True
    result['provenance']['pvt_final_source'] = winner['origin']
    result['provenance']['pvt_sizing_changed'] = winner['design'] != result['pre_pvt']['design']
    result['provenance']['fixed_anchor_reused'] = winner['id']=='anchor'
    if winner['id']=='anchor': result['provenance']['is_ai_generated'] = False
    corners = repair.get('corners_checked', 45)
    result['guidance'] = dict(
        headline=f'Final circuit independently passes all {corners} checked PVT corners.'
                 if corners != 45 else 'Final circuit independently passes all 45 PVT corners.',
        detail='Schematic-level acceptance at this specification. Fixed-anchor reuse is identified in provenance.'
               if corners == 45 else
               f'Checked at {label}, not full sign-off: use Check PVT for the complete 45-corner grid. '
               'Fixed-anchor reuse is identified in provenance.')
    return result


if __name__ == '__main__':
    main()
