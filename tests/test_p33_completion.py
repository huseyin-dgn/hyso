"""P33 offline power arithmetic/contracts. Constructed traces are not native proof."""
import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from hyso.performance import RuntimeResult,analyze_energy
from hyso.performance.runtime_result import _result,_hash
from hyso.performance.energy import _integral,_parse_power
from test_stability_performance import evidence

ROOT=Path(__file__).resolve().parents[1]


def seal(e):
    previous=_hash(e['header']);chunks=[];size=e['request']['protocol']['chunk_rows']
    for i,row in enumerate(e['rows']):row['index']=i
    for start in range(0,len(e['rows']),size):
        rows=e['rows'][start:start+size];index=len(chunks)
        previous=_hash(dict(schema_version=1,run_id=e['header']['run_id'],index=index,previous_sha256=previous,rows=rows))
        chunks.append(dict(name=f'chunk-{index:06d}.json',rows=len(rows),sha256=previous))
    e['chunks']=chunks
    if e['terminal'] is not None:e['terminal'].update(chunks=chunks,last_chunk_sha256=previous,rows=len(e['rows']))
    return e


def stamp(row,ns):
    row['timestamp_ns']=ns
    row['gpu'].update(timestamp_ns=ns-6,query_start_ns=ns-10,query_end_ns=ns-2)
    row['gpu_utilization'].update(query_start_ns=ns-20,query_end_ns=ns-11)


@pytest.fixture
def source():
    e=evidence();e['header']['workload'].update(kind='model_inference',unit='image',input_shape=[2,3,2,2])
    for p in (e['request']['protocol'],e['header']['protocol']):p['duration_seconds']=7.5
    for row in e['rows']:
        row['completed']=row['steps']*2
        if row['steps']:row['last_completion_ns']=row['timestamp_ns']-1000
    e['terminal'].update(completed=e['rows'][-1]['completed'],work_end_ns=e['rows'][-1]['last_completion_ns'])
    return seal(e)


def analyze(e,**kw):return analyze_energy(_result('P31',seal(e)),**kw).to_dict()['analysis']


def with_idle(e,watts=20):
    rows=[]
    for ns in (100_000_000,500_000_000):
        row=copy.deepcopy(e['rows'][0]);row['phase']='idle';stamp(row,ns);row['gpu']['watts']=watts;rows.append(row)
    e['rows']=rows+e['rows'];return seal(e)


def test_gross_energy_requires_no_idle_and_correct_call_image_denominators(source):
    a=analyze(source);duration=(source['terminal']['work_end_ns']-source['terminal']['work_start_ns'])/1e9
    assert a['status']=='derived_from_measured_power_samples' and a['gross_joules']==pytest.approx(10*duration)
    assert a['joules_per_forward']==pytest.approx(10*duration/60)
    assert a['joules_per_completed_image']==pytest.approx(10*duration/120)
    assert a['net_above_idle_joules'] is None and a['idle_baseline']['reason']=='insufficient_idle_baseline'
    assert a['measurements']['idle_power']['window'] is None
    assert a['measurements']['gross_energy']['kind']=='derived'
    assert a['measurements']['direct_counter_energy']['value'] is None


def test_idle_subtraction_preserves_negative_energy_and_is_not_model_exclusive(source):
    a=analyze(with_idle(source));assert a['idle_mean_watts']==20
    assert a['net_above_idle_joules']==pytest.approx(-a['gross_joules'])
    assert a['net_joules_per_completed_image']<0
    assert 'not_model_exclusive' in a['attribution']
    assert a['measurements']['sample_mean_power']['value']!=a['measurements']['active_time_weighted_mean_power']['value']


def test_stale_or_missing_idle_does_not_erase_gross_energy(source):
    e=with_idle(source);a=analyze(e,idle_max_age_seconds=.1)
    assert a['gross_joules'] is not None and a['net_above_idle_joules'] is None
    assert a['idle_baseline']['reason']=='idle_baseline_stale'
    e['rows'][0]['gpu'].update(status='unsupported',reason='sensor_missing',watts=None)
    a=analyze(e);assert a['gross_joules'] is not None and a['idle_mean_watts'] is None
    assert a['idle_baseline']['reason']=='idle_missing_power_samples'


def test_idle_overlapping_the_active_window_is_rejected(source):
    source=with_idle(source);source['terminal']['work_start_ns']=250_000_000
    with pytest.raises(ValueError,match='idle baseline overlaps'):analyze(source)


def test_missing_active_sample_only_reports_covered_energy_without_invalid_ratio(source):
    source['rows'][4]['gpu'].update(status='unsupported',reason='sensor_missing',watts=None)
    a=analyze(source)
    assert a['status']=='partial_power_coverage' and a['gross_joules'] is None
    assert a['measurements']['covered_energy']['value']>0
    assert a['coverage']['covered_seconds']<a['duration_seconds']
    assert a['joules_per_completed_image'] is a['joules_per_forward'] is None
    assert a['sampling']['missing_sample_indices']==[4]


def test_unbracketed_window_never_extrapolates_to_normalized_energy(source):
    # Move only the work-start earlier, preserving P31's duration/counter contract.
    source['terminal']['work_start_ns']-=100_000_000
    a=analyze(source)
    assert a['reason']=='unbracketed_window' and a['gross_joules'] is None
    assert a['measurements']['covered_energy']['value']>0
    assert a['joules_per_forward'] is None


def test_gap_coverage_does_not_divide_partial_joules_by_full_work(source):
    source['rows'].pop(4);a=analyze(source,max_gap_seconds=1.5)
    assert a['gross_joules'] is None and a['reason']=='sensor_sample_gap'
    assert a['coverage']['covered_seconds']+sum((h['end_ns']-h['start_ns'])/1e9 for h in a['coverage']['uncovered_intervals'])==pytest.approx(a['duration_seconds'])
    assert a['joules_per_forward'] is None


def test_aborted_run_uses_only_verified_zero_counter_completion_prefix(source):
    source['terminal']=None;source['controller']['reason']='controller_interrupted';source['incomplete_tail']=True
    a=analyze(source)
    assert a['status']=='derived_verified_prefix_from_measured_power_samples' and not a['source_complete']
    assert 'prefix' in a['measurement_context']['active_window']['scope']
    assert a['completed_forwards']==60 and a['completed_images']==120
    assert a['gross_joules']>0 and a['joules_per_forward']==pytest.approx(a['gross_joules']/60)


@pytest.mark.parametrize('damage',['pci','sensor_id','last_completion','no_new_work','unit','clock','negative_power'])
def test_rehashed_semantic_corruption_rejected(source,damage):
    if damage=='pci':
        for row in source['rows']:row['gpu']['pci_bus_id']='PCI0'
        source['rows'][4]['gpu']['pci_bus_id']='PCI1'
    elif damage=='sensor_id':
        for row in source['rows']:row['gpu']['sensor_id']='meter0'
        source['rows'][4]['gpu']['sensor_id']='meter1'
    elif damage=='last_completion':source['rows'][4]['last_completion_ns']=None
    elif damage=='no_new_work':source['rows'][0]['last_completion_ns']=source['terminal']['work_start_ns']
    elif damage=='unit':source['rows'][4]['gpu']['unit']='mW'
    elif damage=='clock':source['rows'][4]['gpu']['clock_id']='device_clock'
    else:source['rows'][4]['gpu']['watts']=-1
    with pytest.raises(ValueError):analyze(source)


@pytest.mark.parametrize('values',[[],[(0,1)],[(0,0),(1,0)]])
def test_empty_single_and_real_zero_power_are_distinct(values):
    points=[dict(timestamp_ns=t*10**9,watts=w) for t,w in values]
    result,reason=_integral(points,0,10**9,2*10**9)
    if len(points)<2:assert result is None and reason=='insufficient_samples'
    else:assert result==0 and reason is None


@pytest.mark.parametrize('points',[[dict(timestamp_ns=0,watts=-1)],
    [dict(timestamp_ns=0,watts=float('nan'))],[dict(timestamp_ns=-1,watts=1)],
    [dict(timestamp_ns=1,watts=1),dict(timestamp_ns=1,watts=2)],
    [dict(timestamp_ns=2,watts=1),dict(timestamp_ns=1,watts=2)]])
def test_invalid_short_or_duplicated_power_series_rejected(points):
    with pytest.raises(ValueError):_integral(points,0,10**9,2*10**9)


def test_irregular_integration_matches_independent_numpy_reference():
    x=np.array([0.,.4,1.5,3.]);y=np.array([8.,12.,10.,20.])
    points=[dict(timestamp_ns=int(t*10**9),watts=float(w)) for t,w in zip(x,y)]
    energy,reason=_integral(points,0,3*10**9,2*10**9)
    reference=np.sum(np.diff(x)*(y[1:]+y[:-1])/2)
    assert reason is None and energy==pytest.approx(reference)


@pytest.mark.parametrize('counter',[(1000,3000,False),(3000,1,True),(0,0,False)])
def test_energy_counter_wrap_reset_or_units_are_never_guessed(source,counter):
    for row in source['rows']:
        row['gpu'].update(backend='unsupported.cumulative_energy_counter',unit='mJ',counter_value=counter[0] if row['index']<4 else counter[1],counter_reset=counter[2])
    a=analyze(source)
    assert a['status']=='unsupported' and a['gross_joules'] is a['joules_per_forward'] is None
    assert 'counter_backend' in a['reason'] and a['measurements']['direct_counter_energy']['reason']
    assert a['measurement_context']['sensor'] is None


def test_negative_counter_value_rejected_even_though_counter_adapter_is_unsupported(source):
    source['rows'][0]['gpu'].update(backend='unsupported.energy_counter',counter_value=-1)
    with pytest.raises(ValueError):analyze(source)


def test_unverified_legacy_5w_is_not_published_as_an_error_bound(source):
    a=analyze(source)
    assert a['nominal_sensor_spec_energy_bound_joules'] is None
    assert a['uncertainty_details']['legacy_unverified_declared_accuracy_watts']==[5.]
    sample=_parse_power('GPU-test, PCI, GPU, 10, 50, Not Active',device_id='GPU-test',clock_id='clock',start_ns=1,end_ns=2)
    assert sample['sensor_accuracy_watts'] is None and sample['sensor_accuracy_reason']


def test_native_sensor_crash_stops_without_retry(monkeypatch):
    from hyso.performance import energy
    monkeypatch.setattr(energy.shutil,'which',lambda *a:'fake-sensor');calls=[]
    def crash(*a,**k):calls.append(1);return subprocess.CompletedProcess('fake-sensor',0xC0000409,'','controlled')
    monkeypatch.setattr(energy.subprocess,'run',crash)
    with pytest.raises(RuntimeError,match='crash'):energy.observe_nvidia_power('GPU-test',clock_id='clock')
    assert calls==[1]


def test_roundtrip_json_csv_identity_provenance_and_no_clobber(source,tmp_path):
    r=analyze_energy(_result('P31',source));value=r.to_dict()
    for suffix in ('json','csv'):
        path=tmp_path/('r.'+suffix);getattr(r,'to_'+suffix)(path)
        assert getattr(RuntimeResult,'from_'+suffix)(path).to_dict()==value
        with pytest.raises(FileExistsError):getattr(r,'to_'+suffix)(path)
    bad=copy.deepcopy(value);bad['analysis']['gross_joules']=0
    with pytest.raises(ValueError):RuntimeResult(bad)
    for m in value['analysis']['measurements'].values():
        assert {'name','unit','kind','device_id','domain','clock_id','window','method','reason','raw_evidence_reference','completed_work_reference','uncertainty_reference'}<=m.keys()


def test_recorded_trained_cpu_source_does_not_become_energy_from_utilization(monkeypatch):
    from hyso.performance import stability,energy,_telemetry
    def forbidden(*a,**k):pytest.fail('offline analysis attempted native workload or sensor query')
    for module,name in ((stability,'_run_owned'),(stability,'_factory'),(energy,'observe_nvidia_power'),(_telemetry,'cpu_observe'),(_telemetry,'gpu_observe')):
        monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    path=ROOT/'outputs/p31-trained-stability-20261030/trained-smoke/result.json'
    if not path.is_file():pytest.skip('recorded trained CPU evidence unavailable')
    before=torch.get_rng_state().clone();a=analyze_energy(RuntimeResult.from_json(path)).to_dict()['analysis']
    assert a['status']=='unsupported' and a['gross_joules'] is a['joules_per_forward'] is None
    assert a['completed_forwards']==235 and a['completed_images']==60160
    assert torch.equal(before,torch.get_rng_state())
