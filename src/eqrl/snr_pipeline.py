"""Noise verification attached to the existing search, without checkpoint promotion."""
from dataclasses import fields
import time


def attach_noise_result(result, request, *, target, channel, requirements=None, seed=0):
    from eqrl.circuits.ctle import DesignVars
    from eqrl.pipeline import spec_for
    from eqrl.sim.measures import Measures
    from eqrl.sim.server import get_server
    from eqrl.sim.ngspice_runner import NgspiceError
    from eqrl.sim.snr_evaluation import evaluate_snr
    from eqrl.simcount import counting
    verification = result.get('verification') or {}
    result['nominal_status'] = result['status']
    detail = {'request': request.to_dict(), 'provenance': request.provenance,
              'conditional': request.mode == 'unknown', 'passed': False,
              'search_conditioned_on_noise': False,
              'search_note': 'Noise assesses this candidate; it did not steer the existing search.'}
    started = time.monotonic()
    counts = {'analysis': 0, 'measure_all': 0}
    if not result.get('design') or not verification.get('guard_valid', False):
        detail['status'] = 'nominal_invalid'
    else:
        try:
            names = {f.name for f in fields(DesignVars)}
            dv = DesignVars(**{k: v for k, v in result['design'].items() if k in names})
            names = {f.name for f in fields(Measures)}
            nominal = Measures(**{k: v for k, v in verification['measures'].items() if k in names})
            spec = spec_for(target, channel, 1.5, requirements)
            with counting() as counts:
                _, _, _, evaluation = evaluate_snr(get_server('tt'), dv, nominal, spec, request, seed=seed)
            detail.update(evaluation)
        except (NgspiceError, ValueError, OSError) as exc:
            detail.update(status='noise_measurement_failed', reason=str(exc))
    detail['elapsed_seconds'] = time.monotonic() - started
    detail['cost'] = {'spice_analyses': counts['analysis'], 'measure_all': counts['measure_all'],
                      'eye_evaluations': len(detail.get('points', []))}
    if 'cost' in result:
        result['cost']['noise_assessment'] = detail['cost']
        result['cost']['spice_analyses_total'] = result['cost'].get('spice_analyses_total', 0) + counts['analysis']
        result['cost']['measure_all_total'] = result['cost'].get('measure_all_total', 0) + counts['measure_all']
    result['noise_evaluation'] = detail
    # Preserve the nominal verification but prevent its status from masquerading
    # as an unconditional success for this expanded request.
    if result['status'] == 'solved':
        result['status'] = ('conditional_noise_pass' if detail['conditional'] else 'sampled_noise_pass') if detail['passed'] else 'noise_not_verified'
    result['overall_passed'] = bool(verification.get('passed') and detail['passed'] and result.get('pvt', {}).get('accepted', True))
    result['overall_conditional'] = detail['conditional']
    return result
