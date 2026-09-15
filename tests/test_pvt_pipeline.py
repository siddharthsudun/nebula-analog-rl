import copy
from dataclasses import asdict
from pathlib import Path
import json
import pytest
from silq import pipeline
from silq.pvt_repair import apply_pvt_stage
from silq.pvt_refinement import full_grid_pass
from silq.envs.pvt import corner_grid


def sample():
    # Real accepted fixture from the independent previous case, no simulator needed.
    artifact=json.loads(Path('results/pvt_optimization_20260910_v3/verified_candidate.json').read_text())
    spec=pipeline.spec_for(artifact['spec']['target_boost_db'],artifact['spec']['channel_loss_db'],1.5)
    result=dict(design=dict(artifact['candidate']['design'],rs=3200.),verification={'passed':True},
        status='solved',architecture='PPO -> G3.2',provenance={'is_ai_generated':True},cost={})
    return result,spec,artifact


def test_failed_pvt_can_never_keep_nominal_solved(monkeypatch):
    import silq.pvt_repair as stage
    result,spec,_=sample();original=copy.deepcopy(result['design'])
    monkeypatch.setattr(stage,'run_repair',lambda *a,**k:dict(accepted=False,status='budget_exhausted',evaluations=45,cost_complete=False))
    apply_pvt_stage(result,spec)
    assert result['status']=='pvt_not_verified' and not result['overall_passed']
    assert result['design']==original and result['pre_pvt']['status']=='solved'
    assert 'infeasibility' in result['guidance']['detail']


def test_success_replaces_actual_design_netlist_and_measurement(monkeypatch,tmp_path):
    import silq.pvt_repair as stage
    result,spec,artifact=sample()
    artifact['candidate']['id']='anchor'
    (tmp_path/'verified_candidate.cir').write_text('* verified final sizing')
    monkeypatch.setattr(stage,'run_repair',lambda *a,**k:dict(accepted=True,status='verified',evaluations=135,
        cost_complete=True,verification=artifact,artifact_dir=str(tmp_path),measure_all=130,analysis=555))
    apply_pvt_stage(result,spec)
    assert result['design']==artifact['candidate']['design']
    assert result['design']!=result['pre_pvt']['design']
    assert result['netlist']=='* verified final sizing'
    assert result['verification']['passed'] and result['status']=='solved'
    assert result['provenance']['fixed_anchor_reused'] and not result['provenance']['is_ai_generated']
    assert result['cost']['spice_analyses_total']==555
    assert full_grid_pass(artifact['rows'],corner_grid(spec,'full'))


def test_pipeline_defaults_to_pvt_and_explicit_legacy_path_skips_it(monkeypatch):
    import silq.pvt_repair as stage
    result,spec,_=sample()
    result['spec']={'boost_tol_db':1.5}
    monkeypatch.setattr(pipeline,'_design_nominal',lambda *a,**k:copy.deepcopy(result))
    calls=[]
    monkeypatch.setattr(stage,'apply_pvt_stage',lambda *a,**k:calls.append((a,k)))
    pipeline.design(9.,12.)
    assert len(calls)==1
    pipeline.design(9.,12.,pvt=False)
    assert len(calls)==1


def test_historical_triage_is_complete_and_does_not_assert_infeasibility():
    from silq.experiments.pvt_corpus_repair import triage
    rows=triage()
    assert len(rows)==22 and len({r['spec_index'] for r in rows})==22
    assert rows[0]['spec_index']==2
    assert sum(36<=r['historical_passed']<=43 for r in rows)==12
    assert sum(r['historical_passed']==45 for r in rows)==1


def test_snr_pass_cannot_override_failed_pvt(monkeypatch):
    from silq.snr_pipeline import attach_noise_result
    from silq.snr_spec import SNRRequest
    from silq.sim.measures import Measures
    import silq.sim.snr_evaluation as scorer
    result,spec,artifact=sample()
    nominal=next(r for r in artifact['rows'] if r['corner'][0]=='tt' and r['corner'][2]==27)
    result.update(status='pvt_not_verified',pvt={'accepted':False},
        verification=dict(guard_valid=True,passed=True,measures=nominal['measures']))
    monkeypatch.setattr('silq.sim.server.get_server',lambda *a:object())
    monkeypatch.setattr(scorer,'evaluate_snr',lambda *a,**k:(Measures(),0.,True,{'passed':True,'points':[]}))
    attach_noise_result(result,SNRRequest.from_dict({'mode':'unknown'}),target=spec.target_boost_db,channel=spec.channel_loss_db)
    assert not result['overall_passed']
    assert result['status']=='pvt_not_verified'
