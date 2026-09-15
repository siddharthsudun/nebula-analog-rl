import json,subprocess,sys,time,threading
from silq import runtime


def fake_worker(monkeypatch, body):
    real=runtime.subprocess.Popen
    code="import json,sys,time; print(json.dumps({'kind':'ready'}),flush=True); command=json.loads(sys.stdin.readline()); rid=command['id']; "+body
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda args,**kwargs:real([sys.executable,'-u','-c',code],**kwargs))
    return runtime.Runtime()


def test_hard_watchdog_preserves_checkpoint_and_terminates_native_owner(monkeypatch):
    worker=fake_worker(monkeypatch,"print(json.dumps({'kind':'checkpoint','id':rid,'result':{'design':{'rs':2000},'status':'pvt_not_verified'}}),flush=True); time.sleep(20)")
    try:
        start=time.monotonic()
        result=worker.run(dict(target_boost_db=8,channel_loss_db=12,mode='auto'),.2,lambda event:None)
        assert time.monotonic()-start < .8
        assert result['design']=={'rs':2000}
        assert result['status']=='budget_exhausted' and result['overall_passed'] is False
        assert worker.closed and worker.proc.poll() is not None
    finally:worker.close()


def test_primary_returns_before_background_pareto_finishes(monkeypatch):
    worker=fake_worker(monkeypatch,"print(json.dumps({'kind':'result','id':rid,'result':{'design':{'rs':2000},'pareto_pending':True}}),flush=True); time.sleep(.3); print(json.dumps({'kind':'pareto','id':rid,'pareto':{'items':[{'id':'stable'}]}}),flush=True); print(json.dumps({'kind':'done','id':rid}),flush=True); time.sleep(1)")
    done=threading.Event();events=[]
    def emit(event):
        events.append(event)
        if event['kind']=='pareto_done':done.set()
    try:
        start=time.monotonic();result=worker.run(dict(target_boost_db=8,channel_loss_db=12,mode='fastest'),1,emit)
        assert time.monotonic()-start < .25
        assert result['design'] and not done.is_set()
        assert done.wait(2)
        assert any(e['kind']=='pareto' for e in events)
    finally:worker.close()


def test_manual_pvt_api_forwards_exact_design_and_frozen_requirements(monkeypatch):
    import server
    from tests.test_pareto_selection import BASE_DESIGN
    received=[]
    class Worker:
        def run(self,payload,limit,emit):
            received.append(payload);return dict(accepted=False,status='unresolved_within_budget')
    monkeypatch.setattr(runtime,'ready',lambda:True)
    monkeypatch.setattr(runtime,'get_runtime',lambda:Worker())
    result=server.pipeline_pvt(server.PvtCheckRequest(design=BASE_DESIGN,target_boost_db=8,requirements={'power_w_max':.004}))
    assert received[0]['design']==BASE_DESIGN
    assert received[0]['requirements']=={'power_w_max':.004}
    assert received[0]['kind']=='pvt' and not result['accepted']
