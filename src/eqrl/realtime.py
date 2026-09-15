"""Live pipeline: primary circuit first, bounded search, then PVT certification.

Fastest certifies the three stress corners; every other mode certifies all 45. See
`GRID_MODE`.
"""
from copy import deepcopy
from dataclasses import asdict
import time, uuid
from pathlib import Path
from eqrl.request_budget import active, Budget, SearchBudgetExpired
from eqrl.pvt_repair import ROOT

LIMITS={"fastest":5.0,"auto":15.0,"thinking":40.0,"default":15.0,"retarget":15.0,"g32_acceptance":40.0}

#: Which corner grid each mode certifies against. Fastest checks the three stress
#: corners -- tt nominal, ss/low-V/hot, ff/high-V/cold -- because that is where this
#: topology actually fails, and because 45 corners cannot finish inside a fast answer.
#: Every other mode certifies the full 45-corner cross-product. Check PVT is always
#: full whatever mode produced the circuit, so the short grid costs no coverage the
#: engineer cannot get back on demand.
GRID_MODE={"fastest":"reduced"}

#: Seconds held back from the search so certification can finish inside the same budget.
FULL_RESERVE=4.0
#: Fastest certifies 3 corners instead of 45, so it needs a fraction of FULL_RESERVE --
#: but no longer zero, now that it certifies at all. Measured warm on this machine
#: (2026-09-11, delivered sizing + anchor, both banks): 3 corners 0.50 s, 45 corners
#: 3.12 s. One second buys the 0.50 s sweep plus room for the one in-flight evaluation
#: the search is allowed to overrun by, and still leaves the 4.0 s of search the
#: surrogate seed and FASTEST_BUDGET refinements need.
FASTEST_RESERVE=1.0


def search_reserve(mode, wall_seconds):
    """How much of the budget is withheld from the search for certification.

    Fastest reserves less because it certifies less: 3 corners x 2 candidates in each of
    2 banks, against 45 x 2 x 2 for every other mode. It is still a real reserve -- the
    budget is checked BEFORE each evaluation and never interrupts one, so the search can
    overrun its share by one evaluation and the certification has to survive that.
    """
    if mode == "fastest":
        return FASTEST_RESERVE
    return min(FULL_RESERVE, wall_seconds*.72)


def empty_result(target, channel, mode, spec_index=0, tol=1.5):
    return dict(mode=mode,architecture="PPO/G3.2 with bounded %s PVT"%(
                    "three-corner (tt/ss/ff)" if GRID_MODE.get(mode)=="reduced" else "full-grid"),
                status="budget_exhausted",overall_passed=False,
        design=None,netlist=None,verification=None,
        spec=dict(target_boost_db=target,channel_loss_db=channel,boost_tol_db=tol,spec_index=spec_index,requirements=[]),
        provenance=dict(is_ai_generated=True,policy="results/seq_clean40k.zip",g32_reached_target=False,fallback_invoked=False),
        solver=dict(best_design=None,best_boost_db=None),cost={},
        guidance=dict(headline="The time budget ended before a circuit was verified.",detail="No partial measurements are reported as passing."))


def checkpoint_result(result):
    r=deepcopy(result);r.pop('_pareto_trace',None)
    if not r.get('pvt',{}).get('accepted'):
        r.update(status='pvt_not_verified',overall_passed=False)
        grid='three-corner (tt/ss/ff)' if GRID_MODE.get(r.get('mode'))=='reduced' else 'full-grid'
        r['pvt']=dict(accepted=False,status='pending',evaluations=0,cost_complete=False,
            scope=f'Nominal circuit available for review; independent {grid} PVT is still pending.')
    return r


def design_realtime(target, channel, *, mode, requirements, model, spec_index, tol,
                    peak_probe, rescue_probe, allow_fallback, wall_seconds, checkpoint=None):
    from eqrl import pipeline as pl
    from eqrl.simcount import counting
    from eqrl.pvt_workers import get_pool
    from eqrl.pvt_fast import certify
    from eqrl.pvt_repair import apply_pvt_stage
    from eqrl.circuits.ctle import DesignVars,netlist
    started=time.monotonic();deadline=started+wall_seconds
    # The caller prewarms; never hide worker startup inside a nominal check.
    pl._note('pvt_prepare','Confirming pre-initialized simulation workers.')
    pool=get_pool(deadline=deadline)
    spec=pl.spec_for(target,channel,1.5 if tol is None else tol,requirements)
    budget=Budget(deadline, search_reserve(mode, wall_seconds))
    token=active.set(budget)
    result=None
    try:
        with counting() as cost:
            try:
                result=pl._design_nominal(target,channel,model=model,spec_index=spec_index,tol=tol,
                    peak_probe=peak_probe,rescue_probe=rescue_probe,allow_fallback=allow_fallback,mode=mode,requirements=requirements)
            except SearchBudgetExpired:
                result=empty_result(target,channel,mode,spec_index,spec.boost_target_tol_db)
                result['spec']['requirements']=pl.requirements_diff(spec)
                result['provenance']['search_budget_exhausted']=True
                if budget.records:
                    best=min(budget.records,key=lambda r:(not r.get('loose_pass',False),abs(r['boost_db']-target)))
                    dv=DesignVars(**best['design']);verification=pl.verify(dv,spec)
                    result.update(design=best['design'],verification=verification,
                        netlist=netlist(dv,vdd=spec.vdd_nominal,temp_c=27,corner='tt',analysis='ac',models='sky130'),
                        status=pl.SOLVED if verification['passed'] else pl.CLOSED_NOT_VERIFIED,
                        solver=dict(best_design=best['design'],best_boost_db=best['boost_db']))
                    result['_pareto_trace']=budget.records
                result['cost']=dict(optimizer_evals=budget.attempts,measure_all_total=cost['measure_all'],spice_analyses_total=cost['analysis'])
    finally:
        active.reset(token)
    result['request_id']=uuid.uuid4().hex
    result['timing']=dict(limit_seconds=wall_seconds,nominal_seconds=time.monotonic()-started)
    from eqrl.pareto import seed_primary
    seed_primary(result,spec)
    if result.get('design') and checkpoint:
        checkpoint(checkpoint_result(result))
    if result.get('design') and time.monotonic() < deadline:
        output=ROOT/'results/pipeline_pvt'/result['request_id']
        try:
            repair=certify(result['design'],spec,pool,output,deadline,pl._note,
                           grid_mode=GRID_MODE.get(mode,'full'))
            apply_pvt_stage(result,spec,repair_result=repair)
        except Exception as exc:
            # Running out of time during certification is an expected outcome of a bounded
            # budget, not a server fault. `pvt_workers.remaining()` RAISES queue.Empty past
            # the deadline rather than returning, and this call sits outside the try/finally
            # above, so before this guard an overrun escaped to the API as a bare 500 with
            # an empty message -- discarding a circuit the engineer already had on screen.
            # The contract is to hand back the nominal circuit and say plainly what was not
            # verified, so Check PVT can finish the job. Exception, not BaseException:
            # KeyboardInterrupt and SystemExit must still propagate.
            result=checkpoint_result(result)
            import queue
            result['pvt']['status']='budget_exhausted' if isinstance(exc,(queue.Empty,TimeoutError)) else 'failed'
            if str(exc):result['pvt']['error']=str(exc)
            # An overrun closes the worker pool (ResidentPool.evaluate re-raises after
            # self.close()), so the NEXT request would silently pay full pool respawn inside
            # its own budget -- the 40s-of-worker-startup symptom, re-armed. Say so rather
            # than letting that latency look mysterious.
            pl._note('pvt','PVT ran past this request\'s time budget. The nominal circuit is '
                           'returned unverified; simulation workers are being rebuilt.')
    elif result.get('design'):
        result=checkpoint_result(result)
        result['pvt']['status']='budget_exhausted'
    # The PVT stage may have replaced the primary sizing; seed the final reference afresh.
    result.pop('pareto',None)
    seed_primary(result,spec)
    result['timing']['elapsed_seconds']=time.monotonic()-started
    result['pareto_pending']=bool(result.get('design'))
    return pl._plain(result)
