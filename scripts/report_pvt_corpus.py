"""Render live/final PVT corpus evidence; no simulator calls or design changes."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/pvt_corpus_20260910_v1'
OUT=ROOT/'docs/PVT_CORPUS_2026-09-10.md'


def render():
    triage=json.loads((RUN/'triage.json').read_text())
    summary=json.loads((RUN/'summary.json').read_text()) if (RUN/'summary.json').exists() else {'results':[]}
    progress=json.loads((RUN/'progress.json').read_text())
    rows={r['spec_index']:r for r in summary['results']}
    complete=summary.get('complete',False)
    lines=['# PVT pipeline integration and corpus evidence, 2026-09-10','',
        '**Status: '+('complete' if complete else 'running; incomplete corpus results')+'**. '+
        f"{summary.get('independently_verified_clean',0)} independently verified clean cases out of the fixed 22-case set; "+
        f"{summary.get('evaluated',0)} cases completed. Historical baseline: 1/22 clean.",'',
        'The historical audit contains 22 nominally successful candidates, each evaluated at 45 corners. '+
        'It is not 22 corners. This experiment measures the repair rate conditional on those 22 candidates; '+
        'it does not establish a fresh PPO solve rate on all requests. Pending cases are not counted as failures or successes.','',
        '## Verified corrections to the proposed plan','',
        '- Spec 0 was selected by lowest failing spec index, not greatest severity. Its 42/45 result was not the mildest failure: three cases passed 43/45.',
        '- There are 12 cases in the 36-43 passing-corner group, six in 17-23, three in 28-32, and one clean case.',
        '- The severe cases primarily fail DC-gain plausibility or saturation guards. Those failures do not establish a physical boost ceiling.',
        '- The previous case already became 45/45 with the fixed delivered-sizing anchor. CMA-ES improved margins; it was not solely responsible for closing that case.','',
        '## Production behavior','',
        '`silq.pipeline.design()` now runs nominal PPO/corpus search and G3.2, nominal verification, then `apply_pvt_stage`. '+
        'The stage performs fresh full-grid checks, searches if needed, and uses a second simulator process for final acceptance. '+
        'A successful repair replaces the actual returned sizing, SPICE netlist and verification measurements. '+
        'The previous candidate is retained under `pre_pvt`. A failed or interrupted PVT stage returns `pvt_not_verified`, never `solved`.','',
        'A fixed delivered-sizing anchor is remeasured at the requested target/channel and can be selected. '+
        'Such reuse is explicitly marked `fixed_anchor_reused` and `is_ai_generated=false`. PPO itself is unchanged. '+
        '`pvt=False` preserves the explicit historical nominal benchmark path; production uses PVT by default. '+
        'SNR remains optional and separate. If requested, its sampled assessment cannot override failed PVT acceptance. '+
        'This is not a joint SNR-by-PVT sign-off.','',
        'The dashboard shows a PVT stage and an independent 45-corner verdict. Final sizing, drawing, measurements and exports use the selected circuit. '+
        'PVT measurement costs are counted in the worker and included in total measurements and SPICE analyses; nominal optimizer costs remain labeled separately.','',
        '## Fixed protocol and limits','',
        'All 22 cases use the same production repair function, ordered by historical pass count descending. '+
        'First check the input on 45 corners. If clean, independently repeat it. Otherwise check the fixed anchor. '+
        'If neither is clean, use five CMA-ES generations of eight designs at eight selected stress corners, '+
        'fully check two finalists, then independently repeat the selected design on all 45 corners. '+
        'The six sizing variables, physical guards, ten hard checks and +/-1.5 dB tolerance are retained. '+
        'The process/supply/temperature grid is TT/SS/FF/SF/FS x 1.71/1.80/1.89 V x 0/27/125 C.','',
        'Per-case cap: 545 corner evaluations and 900 seconds. Full corpus cap: 11,990 corner evaluations and 19,800 seconds (5.5 hours). '+
        'Workers run sequentially for the corpus, alongside the separately supervised SNR job. '+
        'One additional live pipeline smoke uses target 8.92 dB and channel 12.34 dB, outside this historical set, '+
        'with its own 545-corner/900-second PVT cap. Search exhaustion is reported as unresolved, not infeasible.','',
        '## Case results','',
        '| Spec | Target dB | Channel dB | Historical /45 | Fresh input /45 | Independent final /45 | Source | Status |',
        '|---:|---:|---:|---:|---:|---:|---|---|']
    for c in triage:
        r=rows.get(c['spec_index'],{})
        source=r.get('final_source') or '-'
        lines.append(f"| {c['spec_index']} | {c['target_boost_db']:.3f} | {c['channel_loss_db']:.3f} | {c['historical_passed']} | {r.get('fresh_input_passed','pending')} | {r.get('final_passed','pending')} | {source} | {r.get('status','pending')} |")
    lines+=['','Recorded completed-case corner evaluations: '+str(summary.get('evaluations_known',0))+'. '+
            ('Counts complete for these completed cases.' if summary.get('cost_complete') else 'Some work may be uncounted after interruption; inspect raw artifacts.'),'',
        '## Severe-case diagnosis','',
        '| Spec | Historical guard failures | Current evidence |','|---:|---|---|']
    for c in triage:
        if c['historical_passed']>23: continue
        r=rows.get(c['spec_index'],{})
        causes=', '.join(f'{k}: {v}' for k,v in c['guard_failures'].items())
        lines.append(f"| {c['spec_index']} | {causes} | {r.get('diagnosis','Pending fresh measurements and bounded repair')} |")
    lines+=['','A passing design is a constructive feasibility witness at that specification. '+
        'An unsuccessful local search is not a certified upper bound for the topology. '+
        'PVT acceptance remains schematic-level: no Monte Carlo mismatch yield, extracted-layout parasitics, or low-BER certification is claimed.','',
        '## Evidence and validation','',
        '- Live machine-readable summary: `results/pvt_corpus_20260910_v1/summary.json`.',
        '- Each case records configuration, source hashes, fresh baselines, search candidates when needed, selection, independent verification, raw SPICE artifacts, costs and the production-format final result.',
        '- The delivered historical circuit and nominal PPO checkpoint are preserved. A repaired result is adopted for the individual live request, rather than silently replacing a global benchmark artifact.',
        '- Validation: 75 focused Python tests passed (eight separate historical end-to-end tests excluded); nine JavaScript tests passed. The live pipeline smoke is recorded separately below.']
    smoke=ROOT/'results/pvt_pipeline_smoke_20260910_v1/status.json'
    lines+=['','Live pipeline smoke: '+(json.dumps(json.loads(smoke.read_text())) if smoke.exists() else 'running; no completed acceptance record yet')+'.','']
    tmp=OUT.with_suffix('.tmp');tmp.write_text('\n'.join(lines),encoding='utf-8');tmp.replace(OUT)
    return complete or progress.get('status')=='budget_exhausted'


if __name__=='__main__':
    started=time.monotonic()
    while time.monotonic()-started<21000:
        try:
            if render():break
        except (OSError,ValueError):pass
        time.sleep(30)
