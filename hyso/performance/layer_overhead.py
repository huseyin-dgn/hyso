"""Opt-in, repeated fresh-worker profiler/control experiment, never a correction.

Both arms run owned eager eval/inference recipes. Parent model/RNG/allocator
handles never cross the boundary. Normal layer capture runs no control forwards.
"""
from dataclasses import dataclass,asdict
import statistics

from .cold import ColdStartRecipe
from .inference import _base
from .profiling import ProfilingProtocol,_execute_profile
from .result import LayerOverheadResult,OperatorProfileResult,_integer,_positive


@dataclass(frozen=True)
class LayerOverheadProtocol:
    trials:int=3
    max_coefficient_of_variation:float=.25

    def __post_init__(self):
        _integer(self.trials,2)
        if self.trials>5:raise ValueError('at most five paired worker trials')
        _positive(self.max_coefficient_of_variation)
        if self.max_coefficient_of_variation>1:raise ValueError('variance tolerance must be at most one')


_LIMITATIONS=[
    'Repeated fresh workers use identical recipe, seed, model/input hashes, precision, threads and device.',
    'Paired launch order alternates; warmup is identical. This does not prove a hardware steady state.',
    'Host perf_counter windows include synchronization before completion on CUDA, not GPU kernel time.',
    'Both arms include inference/autocast and loop timing; only the instrumented arm includes profiler, hooks, dispatch tracing and CUDA Events.',
    'Collector startup/export/shutdown and worker startup are outside the compared forward windows.',
    'Signed differences include instrumentation perturbation and cross-process scheduling/cache/clock variation.',
    'Variance policy is a diagnostic gate, not a statistical confidence interval or universal overhead constant.',
    'Do not subtract this experiment estimate from layer timings or CPU/CUDA elapsed intervals.',
    'Explicit experiment executes additional owned forwards; ordinary profile_layers analysis does not.',
    'No default RAM/VRAM quotas or resource-occupancy admission gates; errors and OOM propagate.']


def _metrics(pairs,recipe,capture,protocol):
    arms={'control':[],'instrumented':[]};differences=[];matched=True;reason=None
    first_context=pairs[0]['instrumented']['provenance'];first_output=pairs[0]['instrumented']['raw_evidence']['overhead_experiment']['output_sha256']
    for index,pair in enumerate(pairs):
        if pair['order']!=(['control','instrumented'] if index%2==0 else ['instrumented','control']):
            raise ValueError('invalid paired launch order')
        source=OperatorProfileResult(pair['instrumented']).to_dict();control=pair['control']
        exp=source['raw_evidence']['overhead_experiment'];env=source['provenance']['environment']
        _check_control(control,recipe,capture)
        expected_sync='synchronous_CPU' if recipe['device']=='cpu' else 'device_before_and_after_each_forward'
        equivalent=(source['provenance']==first_context and exp['output_sha256']==first_output and source['protocol']['recipe']==recipe and source['protocol']['configuration']==capture and
            control['model']['structural_descriptor']==source['provenance']['model']['structural_descriptor'] and
            control['model']['weights_sha256']==source['provenance']['model']['weights_sha256'] and
            control['input_sha256']==source['provenance']['input']['content_sha256'] and
            control['output_sha256']==exp['output_sha256'] and
            all(control[k]==env[k] for k in ('device','torch','python','gpu','cuda_build')) and
            control['threads']==exp['threads']==1 and control['synchronization']==exp['synchronization']==expected_sync and
            exp['mode']=='instrumented' and control['dtype']==('torch.float64' if recipe['precision']=='fp64' else 'torch.float32'))
        if recipe['device']!='cpu' and source['raw_evidence']['cuda_event_timing']['scope_reason'] is not None:equivalent=False
        if not equivalent:matched=False;reason='model_input_output_precision_device_or_scope_mismatch'
        values=exp['windows_ms'];_windows(values,capture['measure_iterations'])
        if values!=source['measurements']['overhead']['instrumented_host_forward_windows_ms']:
            raise ValueError('experiment/profile window mismatch')
        arms['control'].append(control['windows_ms']);arms['instrumented'].append(values)
        differences.append(statistics.mean(values)-statistics.mean(control['windows_ms']))
    statistics_by_arm={}
    stable=True
    for name,windows in arms.items():
        flat=[v for trial in windows for v in trial];means=[statistics.mean(v) for v in windows]
        avg=statistics.mean(flat);cv=statistics.stdev(flat)/avg;trial_cv=statistics.stdev(means)/statistics.mean(means)
        stable=stable and max(cv,trial_cv)<=protocol.max_coefficient_of_variation
        statistics_by_arm[name]=dict(status='measured',mean_ms=avg,stdev_ms=statistics.stdev(flat),
            coefficient_of_variation=cv,trial_mean_coefficient_of_variation=trial_cv,trial_means_ms=means,samples=len(flat))
    if matched and not stable:reason='excessive_within_or_between_worker_variation'
    ok=matched and stable;delta=statistics.mean(differences)
    return dict(status='estimated' if ok else 'unsupported',reason=reason,comparison_scope='synchronized_instrumented_forward_vs_control_in_fresh_workers',
        matched_context=matched,variance_accepted=stable,arms=statistics_by_arm,
        observed_paired_mean_differences_ms=differences,
        signed_overhead_ms=delta if ok else None,
        signed_overhead_percent=100*delta/statistics_by_arm['control']['mean_ms'] if ok else None,
        collector_lifecycle_overhead_percent=None,kernel_overhead_percent=None,
        interpretation='conditional_experiment_estimate; negative_difference_allowed; not_a_latency_correction')


def _windows(values,n):
    if type(values) is not list or len(values)!=n:raise ValueError('incomplete experiment windows')
    for v in values:_positive(v)


def _check_control(control,recipe,capture):
    if control['recipe']!=recipe or control['configuration']!=capture or control['mode']!='control':
        raise ValueError('control context mismatch')
    worker=control['worker']
    if worker['status']!='completed' or worker['exit_code']!=0 or not worker['process_terminated'] or not all(control['safety'].values()):
        raise ValueError('unsafe or failed control worker')
    _windows(control['windows_ms'],capture['measure_iterations'])
    for key in ('input_sha256','output_sha256'):
        if type(control[key]) is not str or len(control[key])!=64:raise ValueError('missing control tensor hash')


def measure_layer_overhead(recipe,*,protocol=LayerOverheadProtocol(),capture_protocol=None):
    """Explicit additional experiment; failures retain normal ProfilingError cleanup.

    At least two warmups and three forward windows per worker are required.
    Estimates remain unsupported when contexts differ or measured variance is high.
    Warmups are uninstrumented in both arms. Collector/observer first-use effects
    remain in the instrumented windows and can cause the variance gate to reject
    the estimate; no samples are silently discarded or replaced.
    """
    if type(recipe) is not ColdStartRecipe or type(protocol) is not LayerOverheadProtocol:raise TypeError('recipe and LayerOverheadProtocol required')
    capture=ProfilingProtocol(warmup_iterations=2,measure_iterations=5) if capture_protocol is None else capture_protocol
    if type(capture) is not ProfilingProtocol or capture.warmup_iterations<2 or capture.measure_iterations<3:
        raise ValueError('overhead comparison requires >=2 warmups and >=3 measured forwards per worker')
    pairs=[]
    for index in range(protocol.trials):
        order=['control','instrumented'] if index%2==0 else ['instrumented','control'];pair=dict(order=order)
        for mode in order:
            observed=_execute_profile(recipe,capture,experiment_mode=mode)
            pair[mode]=observed.to_dict() if mode=='instrumented' else observed
        pairs.append(pair)
    source=pairs[-1]['instrumented']
    data=_base('layer_overhead','P13',dict(id='paired_layer_profiler_experiment',version=1,
        scope='owned_fresh_worker_eval_inference',configuration=asdict(protocol),capture_configuration=asdict(capture),
        recipe=recipe.to_dict()),source['provenance'])
    data.update(requested=dict(pairs=protocol.trials),actual=dict(pairs=len(pairs)),
        raw_evidence=dict(pairs=pairs),measurements=_metrics(pairs,recipe.to_dict(),asdict(capture),protocol),
        diagnostics=dict(warnings=[],limitations=_LIMITATIONS))
    return LayerOverheadResult(data)


def _validate_layer_overhead(data):
    p=data['protocol'];protocol=LayerOverheadProtocol(**p['configuration']);capture=ProfilingProtocol(**p['capture_configuration'])
    recipe=ColdStartRecipe(**p['recipe']).to_dict();pairs=data['raw_evidence']['pairs']
    if p['id']!='paired_layer_profiler_experiment' or p['scope']!='owned_fresh_worker_eval_inference':raise ValueError('invalid overhead scope')
    if capture.warmup_iterations<2 or capture.measure_iterations<3:raise ValueError('insufficient overhead windows/warmup')
    if len(pairs)!=protocol.trials or data['actual']!=dict(pairs=protocol.trials) or data['requested']!=data['actual']:
        raise ValueError('overhead trial ledger mismatch')
    if data['measurements']!=_metrics(pairs,recipe,p['capture_configuration'],protocol):raise ValueError('unverified overhead estimate')
    if data['provenance']!=pairs[-1]['instrumented']['provenance'] or data['diagnostics']!=dict(warnings=[],limitations=_LIMITATIONS):
        raise ValueError('overhead provenance/limitations mismatch')
