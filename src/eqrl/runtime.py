"""Prewarmed process-owned live pipeline with hard deadlines and streamed Pareto."""
from __future__ import annotations
import atexit, hashlib, json, os, queue, subprocess, sys, threading, time, uuid
from pathlib import Path
from eqrl.pvt_workers import WindowsJob, ROOT, configuration_stamp
from eqrl.realtime import LIMITS, empty_result

class NotReady(RuntimeError): pass


def stamp():
    digest=hashlib.sha256(configuration_stamp().encode())
    for name in ('pipeline.py','runtime.py','realtime.py','request_budget.py','pareto.py','pvt_fast.py','pvt_repair.py'):
        digest.update((ROOT/'src/eqrl'/name).read_bytes())
    return digest.hexdigest()


class Runtime:
    def __init__(self):
        self.stamp=stamp();self.closed=False;self.pending={};self.lock=threading.Lock()
        self.job=WindowsJob() if os.name=='nt' else None
        output=ROOT/'work/runtime'/uuid.uuid4().hex;output.mkdir(parents=True)
        self.log=(output/'worker.log').open('w',encoding='utf-8')
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
        env.setdefault('EQRL_PVT_LANES','3')
        env['PYTHONUTF8']='1'
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        self.ready=queue.Queue()
        self.proc=subprocess.Popen([sys.executable,'-u','-m','eqrl.runtime','--worker'],cwd=ROOT,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|(4 if self.job else 0))
        try:
            if self.job:self.job.attach_and_resume(self.proc)
            threading.Thread(target=self._read,daemon=True).start()
            message=self.ready.get(timeout=240)
            if message.get('kind')!='ready':raise NotReady(str(message))
        except BaseException:
            self.close();raise

    def _read(self):
        try:
            for line in self.proc.stdout:
                message=json.loads(line)
                destination=self.pending.get(message.get('id'), self.ready)
                destination.put(message)
        except Exception as exc:
            self.ready.put(dict(kind='error',error=str(exc)))
        finally:
            self.ready.put(dict(kind='error',error='Runtime worker exited before readiness'))
            for destination in list(self.pending.values()):destination.put(dict(kind='error',error='Runtime stopped'))

    def close(self):
        if self.closed:return
        self.closed=True
        if self.job:self.job.close()
        if self.proc.poll() is None:self.proc.kill()
        self.proc.wait()
        self.log.close()

    def run(self, payload, limit, emit):
        if self.closed or self.proc.poll() is not None:raise NotReady('Simulation runtime is not ready')
        started=time.monotonic();deadline=started+limit
        if not self.lock.acquire(blocking=False):raise NotReady('A primary design request is active')
        request_id=uuid.uuid4().hex;q=queue.Queue();self.pending[request_id]=q
        latest=None
        try:
            self.proc.stdin.write(json.dumps(dict(id=request_id,payload=payload,deadline=deadline-.10))+'\n');self.proc.stdin.flush()
            while True:
                message=q.get(timeout=max(.001,deadline-time.monotonic()-.04))
                kind=message['kind']
                if kind=='progress':emit(message['event'])
                elif kind=='checkpoint':
                    latest=message['result'];emit(dict(kind='preview',stage='verify',text='Nominal circuit ready for review; PVT pending.',result=latest))
                elif kind=='result':
                    result=message['result'];result['request_id']=request_id
                    if result.get('restart_required'):self.close()
                    result.setdefault('timing',{})['response_seconds']=time.monotonic()-started
                    if result.get('pareto_pending'):
                        emit(dict(kind='pareto_start',stage='pareto',text='Primary circuit ready; measuring alternatives in the background.'))
                        threading.Thread(target=self._background,args=(request_id,q,emit),daemon=True).start()
                    else:self.pending.pop(request_id,None)
                    return result
                elif kind=='error':
                    error=message.get('error','Runtime failed')
                    if error.startswith('ValueError:'):raise ValueError(error)
                    raise RuntimeError(error)
                if time.monotonic() >= deadline-.04:raise queue.Empty()
        except queue.Empty:
            self.close()
            if payload.get('kind')=='pvt':
                self.pending.pop(request_id,None)
                return dict(accepted=False,status='budget_exhausted',evaluations=0,cost_complete=False,
                    elapsed_seconds=time.monotonic()-started,scope='The exact requested sizing was not fully verified before the deadline.')
            result=latest or empty_result(payload['target_boost_db'],payload['channel_loss_db'],payload.get('mode','auto'))
            result.update(status='budget_exhausted',overall_passed=False,request_id=request_id,pareto_pending=False)
            result['pvt']=dict(accepted=False,status='budget_exhausted',evaluations=0,cost_complete=False,scope='Full PVT acceptance was not completed before the deadline.')
            result['timing']=dict(limit_seconds=limit,response_seconds=time.monotonic()-started,timed_out=True)
            self.pending.pop(request_id,None)
            return result
        finally:self.lock.release()

    def _background(self, request_id, q, emit):
        try:
            while True:
                message=q.get(timeout=30)
                if message['kind'] in ('done','error'):
                    emit(dict(kind='pareto_done',stage='pareto',text='Background comparison finished.'));break
                if message['kind']=='pareto':emit(dict(kind='pareto',stage='pareto',text='Measured circuit tradeoffs updated.',pareto=message['pareto']))
        except queue.Empty:
            emit(dict(kind='pareto_done',stage='pareto',text='Background comparison timed out.'))
        finally:self.pending.pop(request_id,None)


_runtime=None
_runtime_lock=threading.Lock()

def ready():
    return _runtime is not None and not _runtime.closed and _runtime.proc.poll() is None and _runtime.stamp==stamp()

def prepare():
    global _runtime
    with _runtime_lock:
        if not ready():
            if _runtime:_runtime.close()
            _runtime=Runtime()
    return _runtime

def get_runtime():
    if not ready():raise NotReady('Workers require initialization before a simulation can start')
    return _runtime

def close():
    global _runtime
    with _runtime_lock:
        if _runtime:_runtime.close();_runtime=None

atexit.register(close)


def worker_main():
    import faulthandler
    faulthandler.dump_traceback_later(12,repeat=True,file=sys.stderr)
    from contextlib import redirect_stdout
    from eqrl.agents.train_noise_pilot import _configure_ngspice
    print("runtime: configure",file=sys.stderr,flush=True)
    _configure_ngspice()
    wire=sys.stdout;write_lock=threading.Lock()
    def send(kind, request_id=None, **data):
        with write_lock:
            wire.write(json.dumps(dict(kind=kind,id=request_id,**data),default=str)+'\n');wire.flush()
    commands=queue.Queue();cancel=threading.Event()
    def read_commands():
        for line in sys.stdin:
            cancel.set();commands.put(json.loads(line))
    with redirect_stdout(sys.stderr):
        from eqrl import pipeline as pl
        from eqrl.pvt_workers import get_pool
        from eqrl.experiments.final_comparison import load_policy
        from eqrl.experiments.fastest_hedge import load_fastest_assets
        import server
        print('runtime: pool initialization',file=sys.stderr,flush=True)
        pool=get_pool();print('runtime: nominal initialization',file=sys.stderr,flush=True)
        server.get_evaluator('tt',fast=True,channel_loss_db=12.)
        load_policy(str(ROOT/pl.POLICY));load_fastest_assets()
        print('runtime: ready',file=sys.stderr,flush=True)
    faulthandler.cancel_dump_traceback_later()
    # Initialize native libraries before a background CRT stdin read starts.
    threading.Thread(target=read_commands,daemon=True).start()
    send('ready')
    while True:
        command=commands.get();cancel.clear()
        request_id=command['id'];payload=command['payload'];deadline=command['deadline']
        def emit(stage, text, **extra):send('progress',request_id,event=dict(stage=stage,text=text,**extra))
        try:
            with redirect_stdout(sys.stderr):
                server._emit=emit
                if payload.get('kind')=='pvt':
                    from eqrl.pvt_fast import certify
                    spec=pl.spec_for(payload['target_boost_db'],payload['channel_loss_db'],payload.get('tol',1.5),payload.get('requirements'))
                    result=certify(payload['design'],spec,pool,ROOT/'results/pipeline_pvt'/request_id,deadline,emit,include_anchor=False)
                    result['pareto_pending']=False
                else:
                    args={k:v for k,v in payload.items() if k not in ('noise_request','kind')}
                    with server._narrating():
                        result=pl.design(**args,wall_seconds=max(.001,deadline-time.monotonic()),
                            checkpoint=lambda r:send('checkpoint',request_id,result=r))
                    if payload.get('noise_request') is not None:
                        from eqrl.llm.snr_parser import resolve_snr_request
                        from eqrl.snr_pipeline import attach_noise_result
                        noise=resolve_snr_request(payload['noise_request'],payload['channel_loss_db'])
                        attach_noise_result(result,noise,target=payload['target_boost_db'],channel=payload['channel_loss_db'],requirements=payload.get('requirements'),seed=payload.get('spec_index',0))
                    try:result['describe_text']=pl.describe(result)
                    except (KeyError,TypeError):result['describe_text']=result.get('guidance',{}).get('headline','Circuit report')
            if pool.closed:
                result.update(pareto_pending=False,restart_required=True)
            send('result',request_id,result={k:v for k,v in result.items() if k!='_pareto_trace'})
            if pool.closed:return
            if not result.get('pareto_pending'):continue
            with redirect_stdout(sys.stderr):
                from eqrl import pareto
                spec=pl.spec_for(payload['target_boost_db'],payload['channel_loss_db'],result['spec']['boost_tol_db'],payload.get('requirements'))
                candidates=pareto.proposals(result,spec,30)
                items=[dict(design=result['design'],verification=result['verification'],origin='primary circuit')]
                result.setdefault('pareto',dict(objectives=list(pareto.OBJECTIVES),scope='Nominally verified tradeoffs; PVT is checked separately for each circuit.',evaluated=0))
                result['pareto']['measured_items']=items
                until=time.monotonic()+15
                for offset in range(0,len(candidates),5):
                    if cancel.is_set() or time.monotonic() >= until-1:break
                    batch=candidates[offset:offset+5]
                    rows,counts=pool.evaluate('search',batch,[('tt',spec.vdd_nominal,27.)],spec,ROOT/'results/pareto'/request_id,until)
                    for c in batch:
                        items.append(dict(design=c['design'],verification=pareto.verification_from_row(rows[c['id']][0]),origin=c['origin']))
                    result['pareto']['measured_items']=list(items)
                    result['pareto']['evaluated']+=len(batch)
                    pareto.finalize(result,spec)
                    if cancel.is_set():break
                    send('pareto',request_id,pareto=result['pareto'])
                    if result['pareto']['complete']:break
            send('done',request_id)
        except BaseException as exc:
            send('error',request_id,error=f'{type(exc).__name__}: {exc}')
            if pool.closed:raise

if __name__=='__main__':worker_main()
