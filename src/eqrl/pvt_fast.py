"""Speculative parallel search and independent certification of initial candidates.

More measurements (both candidates in each bank), fewer sequential round trips.
Both full grids must pass. The two phase banks never share simulator instances.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
from eqrl.pvt_refinement import full_grid_pass, summarize, rank_key, recorded_netlist
from eqrl.pvt_repair import ROOT, write
from eqrl.envs.pvt import corner_grid
from eqrl.circuits.ctle import DesignVars


def certify(design, spec, pool, output, deadline, notify=None, include_anchor=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    anchor=json.loads((ROOT/'results/delivered_circuit.json').read_text())["design"]
    candidates=[dict(id="input",origin="pipeline PPO/G3.2 output",design=design)]
    if include_anchor and anchor != design:
        candidates.append(dict(id="anchor",origin="fixed delivered sizing, freshly remeasured",design=anchor))
    grid=corner_grid(spec,"full")
    write(output/'config.json',dict(spec=asdict(spec),initial=candidates))
    if notify:notify("pvt","Full 45-corner checks in two independent worker banks.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={phase:executor.submit(pool.evaluate,phase,candidates,grid,spec,output/phase,deadline) for phase in ("search","verify")}
        results={phase:f.result() for phase,f in futures.items()}
    rows,counts=results['search']; verified,vcounts=results['verify']
    clean=[c for c in candidates if full_grid_pass(rows[c['id']],grid) and full_grid_pass(verified[c['id']],grid)]
    winner=min(clean or candidates,key=lambda c:rank_key(rows[c['id']]))
    key=winner['id'];acceptance=verified[key]
    search_pids={r['worker_pid'] for rs in rows.values() for r in rs}
    verify_pids={r['worker_pid'] for rs in verified.values() for r in rs}
    independent=not bool(search_pids & verify_pids)
    accepted=bool(clean) and independent
    check=dict(candidate=winner,spec=asdict(spec),rows=acceptance,summary=summarize(acceptance),accepted=accepted,
        evaluations=vcounts['evaluations'],independent_process=independent,corner_integrity_tt_ss_passed=True)
    selection=dict(winner=winner,full_candidates={c['id']:dict(candidate=c,rows=rows[c['id']]) for c in candidates})
    write(output/'selection.json',selection);write(output/'verification.json',check)
    if accepted:
        write(output/'verified_candidate.json',check)
        (output/'verified_candidate.cir').write_text(recorded_netlist(DesignVars(**winner['design']),vdd=spec.vdd_nominal,temp_c=27,corner='tt'))
    result=dict(accepted=accepted,status='verified' if accepted else 'unresolved_within_budget',artifact_dir=str(output.resolve()),
        scope='Schematic-level 45-corner PVT; no mismatch, extracted parasitics or SNR certification',
        cost_complete=True,verification=check,selection=selection,independent_worker_processes=independent,
        backend='resident_speculative',measurement_cache=False,elapsed_seconds=time.monotonic()-started,
        **{k:counts[k]+vcounts[k] for k in counts})
    write(output/'result.json',result)
    return result
