"""P33 sampled physical-board energy; no TDP, utilization or CPU/GPU attribution guesses."""
import csv
import math
import os
import shutil
import subprocess
import time

from .runtime_result import RuntimeResult, _result, _int, _number, _name


def _parse_power(text, *, device_id, clock_id, start_ns, end_ns):
    _name(clock_id);_int(start_ns);_int(end_ns,start_ns)
    rows=list(csv.reader(text.strip().splitlines(),skipinitialspace=True))
    if len(rows)!=1 or len(rows[0])!=6:raise ValueError('one physical device / six sensor fields required')
    identity,pci,name,watts,temp,thermal=[v.strip() for v in rows[0]]
    if identity!=device_id:raise ValueError('power sensor physical device mismatch')
    _name(pci);_name(name)
    unsupported=('N/A','[N/A]','[Not Supported]','Not Supported','-','')
    power=None if watts in unsupported else float(watts)
    temperature=None if temp in unsupported else float(temp)
    if power is not None:_number(power)
    if temperature is not None:_number(temperature,-273.15)
    if thermal not in ('Active','Not Active')+unsupported:raise ValueError('unknown thermal health flag')
    return dict(status='measured' if power is not None else 'unsupported',reason=None if power is not None else 'power_sensor_unavailable',
        device_id=identity,pci_bus_id=pci,device_name=name,clock_id=clock_id,query_start_ns=start_ns,query_end_ns=end_ns,
        timestamp_ns=(start_ns+end_ns)//2,watts=power,unit='W',temperature_c=temperature,thermal_slowdown=thermal,
        backend='nvidia-smi.power.draw.instant',scope='physical_board',sensor_accuracy_watts=None,
        sensor_accuracy_reason='query_does_not_report_device_specific_accuracy_or_calibration',
        time_association='host_query_midpoint_bracket_sensor_refresh_latency_unknown')


def observe_nvidia_power(device_id, *, clock_id, timeout_seconds=1.):
    """Bounded read-only CLI query. No CUDA initialization or hidden averaged-power fallback."""
    _name(clock_id)
    if type(device_id) is not str or not device_id.startswith('GPU-'):raise ValueError('physical NVIDIA UUID required')
    _number(timeout_seconds,.001)
    if timeout_seconds>5:raise ValueError('bounded sensor timeout <=5s')
    tool=shutil.which('nvidia-smi');start=time.perf_counter_ns()
    try:
        if tool is None:raise FileNotFoundError('installed nvidia-smi unavailable')
        p=subprocess.run([tool,'--id='+device_id,
            '--query-gpu=uuid,pci.bus_id,name,power.draw.instant,temperature.gpu,clocks_event_reasons.hw_thermal_slowdown',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if p.returncode and (p.returncode<0 or p.returncode & 0xffffffff>=0x80000000):raise RuntimeError('native power sensor process crash')
        if p.returncode:raise OSError('NVIDIA power query exit '+str(p.returncode)+': '+p.stderr[-256:])
        return _parse_power(p.stdout,device_id=device_id,clock_id=clock_id,start_ns=start,end_ns=time.perf_counter_ns())
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        return dict(status='unsupported',reason=type(exc).__name__+': '+str(exc)[:512],device_id=device_id,
                    clock_id=clock_id,query_start_ns=start,query_end_ns=time.perf_counter_ns())


def _validate_power_sample(row, device_id, clock_id, *, require_power=True):
    if (row['device_id']!=device_id or row['clock_id']!=clock_id or row['unit']!='W'
            or row['backend']!='nvidia-smi.power.draw.instant' or row['scope']!='physical_board'):
        raise ValueError('power unit/device/backend/clock association mismatch')
    _int(row['query_start_ns']);_int(row['query_end_ns'],row['query_start_ns']);_int(row['timestamp_ns'])
    if not row['query_start_ns']<=row['timestamp_ns']<=row['query_end_ns']:raise ValueError('power timestamp outside query bracket')
    if row['watts'] is not None:_number(row['watts'])
    elif require_power:raise ValueError('missing measured power')
    if row['sensor_accuracy_watts'] is not None:_number(row['sensor_accuracy_watts'])
    else:_name(row.get('sensor_accuracy_reason'))
    if row['temperature_c'] is not None:_number(row['temperature_c'],-273.15)


def _integral(samples,start_ns,end_ns,max_gap_ns):
    """Piecewise-linear trapezoid of measured samples, clipped to exact host boundaries."""
    _int(start_ns);_int(end_ns,start_ns+1);_number(max_gap_ns,1)
    for sample in samples:_int(sample['timestamp_ns']);_number(sample['watts'])
    times=[s['timestamp_ns'] for s in samples]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('duplicate/nonmonotonic sensor timestamps')
    if len(samples)<2:return None,'insufficient_samples'
    if times[0]>start_ns or times[-1]<end_ns:return None,'unbracketed_window'
    pieces=[];covered=0
    for a,b in zip(samples,samples[1:]):
        left=max(start_ns,a['timestamp_ns']);right=min(end_ns,b['timestamp_ns'])
        if right<=left:continue
        span=b['timestamp_ns']-a['timestamp_ns']
        if span>max_gap_ns:return None,'sensor_sample_gap'
        slope=(b['watts']-a['watts'])/span
        va=a['watts']+slope*(left-a['timestamp_ns']);vb=a['watts']+slope*(right-a['timestamp_ns'])
        pieces.append((va+vb)/2*(right-left)/1e9);covered+=right-left
    if covered!=end_ns-start_ns:raise ValueError('energy integration coverage mismatch')
    return math.fsum(pieces),None


def analyze_energy(stability_result, *, max_gap_seconds=2., idle_max_age_seconds=60.):
    if type(stability_result) is not RuntimeResult or stability_result.to_dict()['feature_id']!='P31':raise TypeError('P31 RuntimeResult required')
    _number(max_gap_seconds,.001);_number(idle_max_age_seconds,.001)
    return _result('P33',dict(energy_version=2,stability=stability_result.to_dict(),max_gap_seconds=max_gap_seconds,idle_max_age_seconds=idle_max_age_seconds))


def _analyze_energy(e):
    if 'energy_version' in e:
        if type(e['energy_version']) is not int or e['energy_version']!=2:raise ValueError('unsupported energy evidence version')
        from ._energy_diagnostics import _analyze
        return _analyze(e)
    source=RuntimeResult(e['stability']).to_dict()
    if source['feature_id']!='P31':raise ValueError('P33 requires P31 completion/clock source')
    _number(e['max_gap_seconds'],.001);_number(e['idle_max_age_seconds'],.001)
    raw=source['evidence'];analysis=source['analysis'];header=raw['header']
    empty=dict(status='unsupported',reason=None,gross_joules=None,idle_mean_watts=None,net_above_idle_joules=None,
        joules_per_completed_image=None,joules_per_forward=None,net_joules_per_completed_image=None)
    if analysis['status']!='completed':return dict(empty,reason='incomplete_workload')
    identity=header['workload']
    if identity['kind']!='model_inference' or identity['unit']!='image' or not identity['device_id'].startswith('GPU-'):
        return dict(empty,reason='no_matching_physical_GPU_inference_source')
    if analysis['completed']==0 or analysis['steps']==0:return dict(empty,reason='zero_completed_work')
    points=[];idle=[];missing=0
    for row in raw['rows']:
        sample=row['gpu']
        if sample['status']!='measured':missing+=1;continue
        _validate_power_sample(sample,identity['device_id'],header['clock_id']);points.append(sample)
        if row['phase']=='idle':idle.append(sample)
    terminal=raw['terminal'];start=terminal['work_start_ns'];end=terminal['work_end_ns']
    if missing:return dict(empty,reason='missing_power_samples')
    if len(idle)<2:return dict(empty,reason='insufficient_idle_baseline')
    if idle[-1]['query_end_ns']>start or (start-idle[-1]['timestamp_ns'])/1e9>e['idle_max_age_seconds']:
        raise ValueError('idle baseline overlaps workload or is stale')
    gap=e['max_gap_seconds']*1e9
    idle_j,reason=_integral(idle,idle[0]['timestamp_ns'],idle[-1]['timestamp_ns'],gap)
    if reason:return dict(empty,reason='idle_'+reason)
    gross,reason=_integral(points,start,end,gap)
    if reason:return dict(empty,reason=reason)
    baseline=idle_j/((idle[-1]['timestamp_ns']-idle[0]['timestamp_ns'])/1e9)
    duration=(end-start)/1e9;net=gross-baseline*duration
    return dict(status='derived_from_measured_power_samples',reason=None,gross_joules=gross,idle_mean_watts=baseline,
        net_above_idle_joules=net,joules_per_completed_image=gross/analysis['completed'],
        joules_per_forward=gross/analysis['steps'],net_joules_per_completed_image=net/analysis['completed'],
        completed_images=analysis['completed'],completed_forwards=analysis['steps'],duration_seconds=duration,
        device_id=identity['device_id'],sample_count=len(points),idle_sample_count=len(idle),
        max_query_half_width_seconds=max((r['query_end_ns']-r['query_start_ns'])/2e9 for r in points),
        nominal_sensor_spec_energy_bound_joules=max(r['sensor_accuracy_watts'] for r in points)*duration,
        uncertainty='sensor_spec_only_not_total_bound; refresh_latency_and_between_sample_variation_unknown',
        integration='host_associated_instant_power_piecewise_linear_trapezoid_not_continuous_energy_counter',
        idle_scope='workload_inactive_physical_board_background_uncontrolled',
        attribution='whole_physical_board_not_model_exclusive; negative_idle_subtraction_preserved')
