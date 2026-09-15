"""Repair the preregistered 22 historical nominal successes using the production stage."""
from dataclasses import asdict
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from silq.pvt_repair import ROOT, apply_pvt_stage, write
from silq.pipeline import spec_for


def triage():
    audit=json.loads((ROOT/'results/pvt_signoff_seed23.json').read_text())
    rows=[]
    for case in audit['candidates']:
        corners=list(case['corners'].values())
        rows.append(dict(spec_index=case['spec_index'], target_boost_db=case['target_boost_db'],
            channel_loss_db=case['channel_loss_db'], design=case['design'],
            historical_passed=sum(bool(r.get('guard_valid') and r.get('pass10')) for r in corners),
            guard_failures=dict(Counter(str(r.get('guard_check')) for r in corners if not r.get('guard_valid'))),
            metric_failures=dict(Counter(k for r in corners for k in r.get('failing',[])))))
    return sorted(rows,key=lambda r:(-r['historical_passed'],r['spec_index']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=triage()
    write(args.output/'triage.json',cases)
    config=dict(cases=22, per_case_corner_limit=545, total_corner_limit=11990,
        per_case_wall_seconds=900,total_wall_seconds=19800, order='historical pass count descending, then index',
        scope='paired repair of 22 historical nominal successes; not a fresh PPO solve-rate benchmark',
        method='same apply_pvt_stage called by production design(); exact full grid and independent repeat',
        infeasibility_rule='failure within a finite search budget is unresolved, never physical infeasibility')
    write(args.output/'config.json',config)
    results=[];started=time.monotonic()
    for case in cases:
        remaining=19800-(time.monotonic()-started)
        if remaining<1:break
        index=case['spec_index']
        write(args.output/'progress.json',dict(status='running',current_spec=index,completed=len(results),
            total=22,clean=sum(r['accepted'] for r in results),elapsed_seconds=time.monotonic()-started))
        spec=spec_for(case['target_boost_db'],case['channel_loss_db'],1.5)
        result=dict(design=case['design'],architecture='historical PPO + G3.2',
            spec=dict(target_boost_db=spec.target_boost_db,channel_loss_db=spec.channel_loss_db,boost_tol_db=1.5),
            status='historical_nominal_candidate',verification=None,provenance=dict(is_ai_generated=True),cost={})
        output=args.output/f'spec_{index:02d}'
        try:
            apply_pvt_stage(result,spec,output=output,seed=20260910+index,wall_seconds=min(900,remaining))
            write(output/'pipeline_result.json',result)
            pvt=result['pvt']
            baseline=json.loads((output/'baselines.json').read_text()) if (output/'baselines.json').exists() else {}
            check=pvt.get('verification',{})
            row=dict(spec_index=index,historical_passed=case['historical_passed'],
                fresh_input_passed=baseline.get('input',{}).get('summary',{}).get('passed'),
                accepted=pvt['accepted'], final_passed=check.get('summary',{}).get('passed'),
                final_source=check.get('candidate',{}).get('id'), status=pvt['status'],
                evaluations=pvt['evaluations'], cost_complete=pvt['cost_complete'],
                seconds=pvt['elapsed_seconds'], artifact_dir=str(output.resolve()),
                diagnosis='feasible witness independently verified' if pvt['accepted'] else 'unresolved within bounded sizing search; no infeasibility proof')
        except Exception as exc:
            row=dict(spec_index=index,historical_passed=case['historical_passed'],accepted=False,
                     status='failed',reason=f'{type(exc).__name__}: {exc}',cost_complete=False)
        results.append(row)
        summary=dict(complete=len(results)==22,total=22,evaluated=len(results),
            historical_clean=sum(c['historical_passed']==45 for c in cases),
            fresh_input_clean=sum(r.get('fresh_input_passed')==45 for r in results),
            independently_verified_clean=sum(r['accepted'] for r in results),
            evaluations_known=sum(r.get('evaluations',0) for r in results),
            cost_complete=all(r['cost_complete'] for r in results),
            elapsed_seconds=time.monotonic()-started,results=results,scope=config['scope'])
        write(args.output/'summary.json',summary)
        print(f"spec {index}: {row['status']}; {summary['independently_verified_clean']}/{len(results)} verified, of 22 total",flush=True)
    write(args.output/'progress.json',dict(status='completed' if len(results)==22 else 'budget_exhausted',
          completed=len(results),total=22,clean=sum(r['accepted'] for r in results),elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
