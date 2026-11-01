"""P32 offline trend evidence and small weakref regressions; no new inference."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from hyso.performance import RuntimeResult,RetentionTracker,analyze_memory_trend,reopen_stability
from hyso.performance.memory_trend import _fit
from hyso.performance.runtime_result import _hash,_result
from test_stability_performance import evidence,seal

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'outputs/p30-final-audit-20261029/regression-final'
TRAINED=ROOT/'outputs/p31-trained-stability-20261030/trained-smoke/result.json'


def rechain(e):
    # The analytical P31 helper uses two-row chunks.
    return seal(e)


@pytest.fixture
def source():
    e=evidence()
    for row in e['rows']:
        i=row['index'];stamp=row['timestamp_ns']
        # RSS bracket and commit bracket intentionally differ. Domain slopes
        # must use these clocks independently, never the journal/other sensor.
        row['rss']=dict(rss_bytes=100+i*2,lifetime_hwm_bytes=200,timestamp_ns=stamp-100,
            query_start_ns=stamp-110,query_end_ns=stamp-90,backend='analytical_reference_not_native')
        row['process_commit']=dict(value=200+i*5,unit='byte',status='measured',reason=None,
            query_start_ns=stamp-60,query_end_ns=stamp-40,
            scope='current_worker_process_private_commit_not_RSS_or_job_total')
        row['allocator']=dict(device_id=e['header']['workload']['device_id'],allocated_bytes=10,reserved_bytes=100+i*20,
            query_start_ns=stamp-30,query_end_ns=stamp-10)
    return rechain(e)


def result(source,**kw):return analyze_memory_trend(_result('P31',source),**kw).to_dict()['analysis']


def test_independent_domain_slopes_and_cache_not_leak(source):
    a=result(source,plateau_range_bytes=1,slope_tolerance_bytes_per_second=1)
    assert a['rss']['slope_bytes_per_second']==2 and a['private_commit']['slope_bytes_per_second']==5
    assert a['allocated']['slope_bytes_per_second']==0 and a['reserved']['slope_bytes_per_second']==20
    assert a['classification']=='allocator_cache_growth_observed' and not a['leak_proven']
    assert a['rss']['timeline'][0]['timestamp_ns']!=a['allocated']['timeline'][0]['timestamp_ns']
    assert a['rss']['n']==a['private_commit']['n']==a['allocated']['n']==8
    assert a['allocator_peak']['allocated_bytes'] is None and a['allocator_peak']['reason']
    assert a['rss']['sampled_peak_bytes']==116 and a['private_commit']['sampled_peak_bytes']==240


def test_no_rss_does_not_hide_commit_allocator_or_witnesses(source):
    for row in source['rows']:row['rss']=None
    a=result(rechain(source),plateau_range_bytes=1,slope_tolerance_bytes_per_second=1)
    assert a['rss']['n']==0 and a['rss']['baseline_bytes'] is None
    assert a['private_commit']['slope_bytes_per_second']==5 and a['reserved']['slope_bytes_per_second']==20
    assert a['classification']=='allocator_cache_growth_observed'


def test_private_commit_growth_cannot_be_masked_by_rss_plateau(source):
    for row in source['rows']:
        row['rss']['rss_bytes']=100;row['allocator']=None
    a=result(rechain(source),plateau_range_bytes=1,slope_tolerance_bytes_per_second=1)
    assert a['rss']['classification']=='sampled_plateau'
    assert a['classification']=='private_commit_change_unattributed' and not a['leak_proven']


def test_live_reference_growth_does_not_require_rss(source):
    for row in source['rows']:
        row['rss']=None;i=row['index'];objects=[dict(object_id=str(n),label='tensor'+str(n),alive=True,logical_bytes=4) for n in range(i)]
        row['retention']=dict(timestamp_ns=row['timestamp_ns']-1,objects=objects,alive_objects=i,logical_live_bytes=4*i)
    a=result(rechain(source))
    assert a['retention']['live_reference_growth'] and a['classification']=='tracked_live_reference_growth_observed'
    assert a['retention']['physical_storage_released'] is None and not a['leak_proven']


def test_observed_memory_decrease_is_not_clamped_or_hidden(source):
    for i,row in enumerate(source['rows']):row['rss']['rss_bytes']=100-2*i
    a=result(rechain(source))
    assert a['rss']['slope_bytes_per_second']==-2 and a['rss']['observed_delta_bytes']==-14
    assert a['rss']['adjacent_decrease_bytes']==14 and a['rss']['adjacent_increase_bytes']==0


def test_cpu_source_cannot_masquerade_as_cuda_allocator_scope(source):
    source['header']['workload']['device_id']='cpu';source['request']['recipe']['gpu_uuid']=None
    for row in source['rows']:
        row['gpu']={'status':'not_requested'};row['gpu_utilization']={'status':'not_requested'}
        row['allocator']['device_id']='cpu'
    with pytest.raises(ValueError,match='CPU-only source'):result(rechain(source))


def test_legacy_allocator_association_never_borrows_rss_clock(source):
    for row in source['rows']:
        row['allocator'].pop('query_start_ns');row['allocator'].pop('query_end_ns')
    a=result(rechain(source))
    point=a['allocated']['timeline'][0]
    assert point['timestamp_ns']==source['rows'][0]['timestamp_ns']
    assert point['timing_scope']=='legacy_journal_timestamp_association_not_query_time'
    assert point['timestamp_ns']!=a['rss']['timeline'][0]['timestamp_ns']


def test_warmup_stabilizes_and_boundaries_remain_separate(source):
    for row in source['rows']:
        i=row['index'];v=1000+i*100 if i<4 else 2000
        row['rss']['rss_bytes']=v;row['rss']['lifetime_hwm_bytes']=3000
        row['process_commit']['value']=v
        row['allocator']=None
    a=result(rechain(source),warmup_seconds=4,plateau_range_bytes=1,slope_tolerance_bytes_per_second=1)
    assert a['rss']['classification']=='sampled_plateau' and a['rss']['slope_bytes_per_second']==0
    assert a['rss']['full_work_window']['observed_delta_bytes']>0
    assert a['classification']=='warmup_growth_then_sampled_plateau' and not a['leak_proven']
    assert a['rss']['excluded_warmup_samples']>0 and a['rss']['boundary_observations']


def test_warmup_exceeds_window_without_fake_measurements(source):
    a=result(source,warmup_seconds=1e308)
    assert a['status']=='insufficient_samples' and a['rss']['n']==0
    assert a['rss']['slope_bytes_per_second'] is a['rss']['sampled_peak_bytes'] is None
    assert a['measurement_window']['warmup_cutoff_ns'] is None


def test_missing_points_and_real_time_gaps_are_not_filled(source):
    for i in (2,3,4):source['rows'][i]['rss']=None
    a=result(rechain(source))
    assert a['rss']['n']==5 and a['rss']['missing_sample_count']==3 and a['rss']['gap_count']==1
    assert a['private_commit']['n']==8 and a['private_commit']['gap_count']==0
    assert a['rss']['slope_bytes_per_second']==2


def test_query_straddling_work_boundary_is_visible_but_not_fitted(source):
    start=source['terminal']['work_start_ns'];row=source['rows'][0]
    row['rss'].update(query_start_ns=start-100,query_end_ns=start,timestamp_ns=start-50)
    a=result(rechain(source))
    assert a['rss']['excluded_outside_work_window']==1
    assert a['rss']['boundary_observations'][0]['query_end_ns']==start


def test_missing_private_commit_requires_null_and_reason(source):
    for row in source['rows']:row['process_commit'].update(value=None,status='unsupported',reason='recorded_backend_unavailable')
    a=result(rechain(source))
    assert a['private_commit']['n']==0 and a['private_commit']['baseline_bytes'] is None
    assert a['private_commit']['missing_sample_count']==9
    assert a['private_commit']['missing_samples'][0]['reason']=='recorded_backend_unavailable'


@pytest.mark.parametrize('n',[1,2,3])
def test_malformed_short_series_rejected_before_insufficient_classification(n):
    with pytest.raises(ValueError):_fit([(1,2)]*n if n>1 else [(1,-2)])


@pytest.mark.parametrize('damage',['rss_duplicate','allocator_duplicate','commit_duplicate','clock','unit','pid','scope','rss_scope',
    'fake_missing_commit','retention_resurrect','retention_missing_id','retention_relabel'])
def test_rehashed_source_cannot_hide_scope_time_or_lifetime_corruption(source,damage):
    # Legacy P31 validates outer/sample ordering; P32 validates these domain facts.
    a,b=source['rows'][1],source['rows'][2]
    if damage=='rss_duplicate':b['rss']=copy.deepcopy(a['rss'])
    elif damage=='allocator_duplicate':b['allocator']=copy.deepcopy(a['allocator'])
    elif damage=='commit_duplicate':b['process_commit']=copy.deepcopy(a['process_commit'])
    elif damage=='clock':a['rss']['clock_id']='media_clock'
    elif damage=='unit':a['process_commit']['unit']='KiB'
    elif damage=='pid':a['rss']['pid']=999
    elif damage=='scope':a['allocator']['scope']='physical_device_memory'
    elif damage=='rss_scope':a['rss']['scope']='model_footprint'
    elif damage=='fake_missing_commit':a['process_commit']['status']='unsupported'
    else:
        for i,row in enumerate(source['rows']):
            row['retention']=dict(timestamp_ns=row['timestamp_ns']-1,objects=[dict(object_id='id',label='t',alive=True,logical_bytes=4)],
                alive_objects=1,logical_live_bytes=4,scope='explicit_weakrefs_distinct_tensor_objects_not_storage_or_heap_census')
        if damage=='retention_resurrect':
            a['retention']['objects'][0].update(alive=False,logical_bytes=0);a['retention'].update(alive_objects=0,logical_live_bytes=0)
        elif damage=='retention_missing_id':b['retention'].update(objects=[],alive_objects=0,logical_live_bytes=0)
        else:b['retention']['objects'][0]['label']='other'
    with pytest.raises(ValueError):result(rechain(source))


def test_independent_numpy_least_squares_and_conditional_uncertainty():
    points=[(10**12+int(t*1e9),y) for t,y in zip((0,1,2,4,7),(20,22,21,29,31))]
    d=_fit(points);x=np.array([0.,1.,2.,4.,7.]);y=np.array([20.,22.,21.,29.,31.])
    slope,intercept=np.linalg.lstsq(np.column_stack([x,np.ones_like(x)]),y,rcond=None)[0]
    se=np.sqrt(np.sum((y-(intercept+slope*x))**2)/(len(x)-2)/np.sum((x-x.mean())**2))
    assert d['slope_bytes_per_second']==pytest.approx(slope) and d['standard_error']==pytest.approx(se)
    assert d['confidence_interval'] is None and 'conditional_iid' in d['uncertainty_scope']


def test_partial_source_only_claims_committed_observation_prefix(source):
    source['terminal']=None;source['controller']['reason']='heartbeat_timeout';source['incomplete_tail']=True
    a=result(rechain(source))
    assert a['status']=='derived' and not a['source_complete']
    assert a['measurement_window']['origin']=='committed_work_observation_prefix_not_exact_full_run'
    assert a['source_coverage']=='committed_prefix_only' and not a['leak_proven']


def test_roundtrip_json_csv_no_clobber_and_nested_source_tamper(source,tmp_path):
    r=analyze_memory_trend(_result('P31',source));value=r.to_dict()
    for suffix in ('json','csv'):
        path=tmp_path/('r.'+suffix);getattr(r,'to_'+suffix)(path)
        assert getattr(RuntimeResult,'from_'+suffix)(path).to_dict()==value
        with pytest.raises(FileExistsError):getattr(r,'to_'+suffix)(path)
    bad=copy.deepcopy(value);bad['evidence']['stability']['analysis']['completed']+=1
    bad['source_sha256']=_hash(dict(feature_id='P32',evidence=bad['evidence']))
    with pytest.raises(ValueError):RuntimeResult(bad)


def test_small_weakrefs_aliasing_release_and_global_rng_preserved():
    tracker=RetentionTracker();tensor=torch.ones(12);alias=tensor.view(3,4)
    tracker.watch(tensor,label='original');tracker.watch(alias,label='alias')
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    snap=tracker.snapshot();assert snap['logical_live_bytes']==96 and snap['alive_objects']==2
    del alias;gc.collect();assert tracker.snapshot()['alive_objects']==1
    del tensor;gc.collect();assert tracker.snapshot()['alive_objects']==0
    assert before[0]==random.getstate() and np.array_equal(before[1][1],np.random.get_state()[1])
    assert torch.equal(before[2],torch.get_rng_state())


@pytest.mark.skipif(not (PRIOR/'runtime-retained0/journal').is_dir(),reason='archived native CPU evidence unavailable')
def test_original_p32_retention_release_assertions_replayed_offline(monkeypatch):
    from test_runtime_analysis import test_real_weakref_retention_release_and_warmup
    monkeypatch.delenv('HYSO_RUNTIME_EVIDENCE',raising=False)
    retained=reopen_stability(PRIOR/'runtime-retained0/journal');released=reopen_stability(PRIOR/'runtime-release0/journal')
    test_real_weakref_retention_release_and_warmup(retained,released)
    assert retained.to_dict()['evidence']['controller']['cleanup']['active_processes']==0
    assert released.to_dict()['evidence']['controller']['cleanup']['active_processes']==0


@pytest.mark.skipif(not TRAINED.is_file(),reason='archived trained Model B evidence unavailable')
def test_real_trained_model_offline_analysis_starts_no_worker_or_sensor(monkeypatch):
    from hyso.performance import stability,memory,_telemetry,_stability_windows
    def forbidden(*a,**kw):pytest.fail('offline analysis attempted native execution or live sensor')
    for module,name in ((stability,'_run_owned'),(stability,'_factory'),(memory,'_observe'),(_telemetry,'cpu_observe'),
                        (_telemetry,'gpu_observe'),(_stability_windows,'_observe_commit')):monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    r=RuntimeResult.from_json(TRAINED);model=r.to_dict()['evidence']['header']['workload']
    assert model['checkpoint']['sha256']=='8279ea41b6b26d9d5e0454327d4548e78fa89ae70a7cd6b7948e76ed6446aaef'
    assert hashlib.sha256(Path(model['checkpoint']['path']).read_bytes()).hexdigest()==model['checkpoint']['sha256']
    before=torch.get_rng_state().clone();a=analyze_memory_trend(r,warmup_seconds=.5).to_dict()['analysis']
    assert a['rss']['n']>=3 and a['private_commit']['n']>=3 and a['classification']=='sampled_plateau'
    assert a['allocated']['status']=='unsupported' and a['allocated']['baseline_bytes'] is None
    assert a['retention']['live_reference_growth'] is None and not a['leak_proven']
    assert torch.equal(before,torch.get_rng_state())
