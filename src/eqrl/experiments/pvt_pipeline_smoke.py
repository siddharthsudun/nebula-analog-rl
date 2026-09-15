"""One bounded, unseen target/channel smoke of the actual production design() entry."""
import json
from pathlib import Path
import time
from eqrl.agents.train_noise_pilot import _configure_ngspice, _set_single_threaded


def main():
    _set_single_threaded(); _configure_ngspice()
    from eqrl.pipeline import design, describe
    output=Path('results/pvt_pipeline_smoke_20260910_v1')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=design(8.92,12.34,mode='fastest',pvt_output=output/'pvt',pvt_wall_seconds=900)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (output/'report.txt').write_text(describe(result),encoding='utf-8')
    assert result['pvt']['accepted'],result['pvt']['status']
    assert result['status']=='solved'
    assert result['verification']['passed']
    assert result['design']==result['pvt']['verification']['candidate']['design']
    assert result['netlist']==(output/'pvt'/'verified_candidate.cir').read_text()
    (output/'status.json').write_text(json.dumps(dict(complete=True,accepted=True,seconds=time.monotonic()-started,
         corner_evaluations=result['pvt']['evaluations'], target=8.92,channel=12.34)),encoding='utf-8')
    print('Live design() smoke passed',flush=True)


if __name__=='__main__':main()
