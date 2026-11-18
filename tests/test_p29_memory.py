"""P29 three-dimensional references, source binding, native CPU and legacy reads."""
import copy
import csv
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from hyso import EvaluationCohort, QualityExperimentResult, compare_performance_quality
from hyso.quality import _derive, _hash, _result
from hyso._quality_memory import pareto
from hyso.checkpoint import _rng_state
from trainer_replay_support import equal


def cohort():
    x=torch.tensor([1.,-1.]).reshape(1,1,1,2).repeat(12,1,1,1)
    return EvaluationCohort(x,[1]*12,sample_ids=[str(i) for i in range(12)],
        class_names=['no','yes'],split='test',preprocessing_id='p29-memory-fixed')


def model(correct=True, extra=0):
    m=nn.Sequential(nn.Flatten(),nn.Linear(2,2))
    with torch.no_grad():
        m[1].weight.copy_(torch.tensor([[-1.,0.],[1.,0.]])*(1 if correct else -1));m[1].bias.zero_()
    if extra:m.register_buffer('registered_extra',torch.zeros(extra))
    return m


def save(result,name):
    destination=os.environ.get('HYSO_P29_EVIDENCE')
    if destination:
        p=Path(destination);p.mkdir(parents=True,exist_ok=True)
        result.to_json(p/(name+'.json'));result.to_csv(p/(name+'.csv'))
    return result


@pytest.fixture(scope='module')
def logical():
    return save(compare_performance_quality(model(False,8),model(),cohort(),iterations=2), 'logical-native')


@pytest.fixture(scope='module')
def rss():
    return save(compare_performance_quality(model(),model(),cohort(),iterations=2,
        memory_metric='cpu_rss_sampled_peak_bytes'), 'rss-native')


@pytest.mark.parametrize('left,right,expected',[
    ((.7,10,100),(.9,8,80),'candidate'),
    ((.9,8,80),(.7,10,100),'baseline'),
    ((.8,10,100),(.9,8,120),'neither'),
    ((.9,10,100),(.8,8,80),'neither'),
    ((.8,10,100),(.8,10,100),'neither'),
    ((.8,10,100),(.8,10,80),'candidate'),
    ((.8,10,100),(.8,12,80),'neither')])
def test_independent_three_axis_reference_table(left,right,expected):
    # Explicit examples, not native timing/quality evidence.
    assert pareto(*left,*right)==expected


def test_actual_logical_three_axes_and_regression_reference(logical):
    d=logical.to_dict();a=d['analysis'];s=d['sources']
    assert a['memory']['baseline']['value']==56 and a['memory']['candidate']['value']==24
    assert a['memory']['baseline']['status']=='derived'
    assert a['memory']['delta_bytes']==-32
    assert a['memory']['relative_change']==pytest.approx(-32/56)
    assert a['memory_regression'] is False and a['quality']['percentage_points']==100
    la=s['baseline']['performance']['stats']['pooled']['mean_ms']
    lb=s['candidate']['performance']['stats']['pooled']['mean_ms']
    assert a['pareto_dominates']==('candidate' if lb<=la else 'neither')
    assert a['winner'] is None and a['pareto_dimensions']==['quality','latency','memory']
    s['baseline'],s['candidate']=s['candidate'],s['baseline']
    a=_derive('P29',s)
    assert a['memory']['relative_change']==pytest.approx(32/24)
    assert a['memory_regression'] is True and a['quality_regression'] is True


def test_memory_threshold_is_independent_of_other_dimensions(logical):
    s=logical.to_dict()['sources'];s['baseline'],s['candidate']=s['candidate'],s['baseline']
    # Baseline memory 24, candidate 56: change 4/3 exceeds every valid <=1 threshold.
    s['policy']['max_memory_regression']=1
    a=_derive('P29',s)
    assert a['memory_regression'] is True
    s['policy']['max_memory_regression']=-.1
    with pytest.raises(ValueError):_derive('P29',s)


def test_missing_one_memory_source_cannot_claim_three_dimensional_dominance(logical):
    s=logical.to_dict()['sources']
    s['candidate']['memory'].update(source=None,missing_reason='not_collected',completed_forwards=0)
    a=_derive('P29',s)
    assert a['pareto_dominates'] is None and a['memory_regression'] is None
    assert a['memory']['candidate']['value'] is None and a['memory']['delta_bytes'] is None
    assert a['quality']['candidate']==1 and a['winner'] is None


@pytest.mark.parametrize('metric',[None,'cuda_allocated_peak_bytes','cuda_reserved_peak_bytes','physical_vram_bytes'])
def test_missing_and_unsupported_domains_make_no_gpu_calls(metric,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('P29 must not initialize CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    r=compare_performance_quality(model(),model(),cohort(),iterations=2,memory_metric=metric).to_dict()
    assert r['analysis']['memory']['status']=='unsupported'
    assert r['analysis']['pareto_dominates'] is None
    assert all(r['sources'][key]['memory']['source'] is None for key in ('baseline','candidate'))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['device','runtime','precision','model','structure','run','cohort','input','protocol','host'])
def test_memory_binding_rejects_foreign_identity_even_with_new_outer_hash(logical,damage):
    d=logical.to_dict();r=d['sources']['candidate']['memory']['binding']
    key={'device':'device','runtime':'runtime','precision':'precision','model':'model_sha256',
         'structure':'structure_sha256','run':'benchmark_run_id','cohort':'cohort_sha256',
         'input':'input_sha256','protocol':'protocol_sha256','host':'host_id'}[damage]
    r[key]='foreign';d['source_sha256']=_hash(d['sources'])
    with pytest.raises(ValueError):QualityExperimentResult(d)


@pytest.mark.parametrize('metric',['cpu_rss_sampled_peak_bytes','cuda_allocated_peak_bytes','cuda_reserved_peak_bytes','physical_vram_bytes'])
def test_memory_domain_relabel_rejected(logical,metric):
    s=logical.to_dict()['sources'];s['candidate']['memory']['metric']=metric
    with pytest.raises(ValueError):_derive('P29',s)


def test_foreign_model_memory_and_forged_aggregate_rejected(logical):
    s=logical.to_dict()['sources']
    s['candidate']['memory']['source']=s['baseline']['memory']['source']
    with pytest.raises(ValueError):_derive('P29',s)
    s=logical.to_dict()['sources']
    s['candidate']['memory']['source']['measurements']['parameter_buffer_bytes']['value']=1
    with pytest.raises(ValueError):_derive('P29',s)


@pytest.mark.parametrize('field',['python','torch','platform_detail','device'])
def test_native_memory_source_runtime_scope_mismatch(logical,field):
    s=logical.to_dict()['sources']
    s['candidate']['memory']['source']['provenance']['environment'][field]='foreign'
    with pytest.raises(ValueError):_derive('P29',s)


@pytest.mark.parametrize('version',[None,1,True,3])
def test_memory_cannot_be_hidden_by_version_downgrade(logical,version):
    s=logical.to_dict()['sources']
    if version is None:s.pop('comparison_version')
    else:s['comparison_version']=version
    with pytest.raises(ValueError):_derive('P29',s)


def test_zero_logical_baseline_has_no_invented_relative_regression():
    no_parameters=nn.Sequential(nn.Flatten(),nn.Softmax(dim=1))
    d=compare_performance_quality(no_parameters,model(),cohort(),iterations=2).to_dict()
    a=d['analysis']
    assert a['memory']['baseline']['value']==0 and a['memory']['candidate']['value']==24
    assert a['memory']['relative_change'] is None and a['memory_regression'] is None
    assert a['memory']['relative_status']=='undefined_zero_baseline'
    assert a['pareto_dominates']=='neither'  # Better quality, greater logical memory.


def test_memory_regression_threshold_boundary_from_native_bytes():
    d=compare_performance_quality(model(extra=2),model(extra=4),cohort(),iterations=2,max_memory_regression=.25).to_dict()
    assert d['analysis']['memory']['baseline']['value']==32
    assert d['analysis']['memory']['candidate']['value']==40
    assert d['analysis']['memory']['relative_change']==.25
    assert d['analysis']['memory_regression'] is False
    s=d['sources'];s['policy']['max_memory_regression']=.249
    assert _derive('P29',s)['memory_regression'] is True


def test_logical_shared_objects_and_views_match_p11_convention():
    m=model()
    m.register_buffer('first_view',m[1].weight.detach())
    m.register_buffer('same_view_object',m.first_view)
    d=compare_performance_quality(m,m,cohort(),iterations=2).to_dict()
    # Parameter weight + bias =24; distinct buffer Tensor view =16;
    # registering that same buffer object twice does not add another16.
    assert d['analysis']['memory']['baseline']['value']==40
    assert d['analysis']['memory']['candidate']['value']==40


def test_rss_native_scope_cleanup_and_separate_window(rss):
    d=rss.to_dict();a=d['analysis'];assert a['memory']['status']=='comparable'
    for key in ('baseline','candidate'):
        record=d['sources'][key]['memory'];source=record['source']
        assert record['completed_forwards']==1 and source['actual']['observations']>=2
        assert a['memory'][key]['value']==max(x['rss_bytes'] for x in source['raw_evidence']['observations'])
        assert source['provenance']['environment']['device']=='cpu'
    assert 'process' in a['memory']['scope'] and a['winner'] is None
    import threading
    assert not any(t.name=='hyso-rss-sampler' for t in threading.enumerate())


def test_rss_sampling_protocol_mismatch_rejected(rss):
    s=rss.to_dict()['sources'];s['candidate']['memory']['source']['protocol']['configuration']['interval_seconds']=.02
    with pytest.raises(ValueError,match='protocol'):_derive('P29',s)


def test_rss_source_cannot_be_logical_or_cuda(rss,logical):
    s=rss.to_dict()['sources'];s['candidate']['memory']['source']=logical.to_dict()['sources']['candidate']['memory']['source']
    with pytest.raises(ValueError):_derive('P29',s)
    s=rss.to_dict()['sources'];s['candidate']['memory']['source']['provenance']['environment']['device']='cuda:0'
    with pytest.raises(ValueError):_derive('P29',s)


def test_missing_rss_backend_does_not_create_a_value(monkeypatch):
    from hyso.performance import memory
    def unavailable():raise memory._BackendUnavailable('injected missing RSS backend')
    monkeypatch.setattr(memory,'_observe',unavailable)
    d=compare_performance_quality(model(),model(),cohort(),iterations=2,memory_metric='cpu_rss_sampled_peak_bytes').to_dict()
    assert d['analysis']['pareto_dominates'] is None
    assert d['analysis']['memory']['baseline']['value'] is None
    assert d['sources']['baseline']['memory']['completed_forwards']==0


def test_legacy_json_and_csv_semantics_preserved(logical,tmp_path):
    s=logical.to_dict()['sources'];s.pop('comparison_version')
    s['policy'].pop('memory_metric');s['policy'].pop('memory_protocol');s['policy'].pop('max_memory_regression')
    for key in ('baseline','candidate'):s[key].pop('memory')
    legacy=_result('P29',s);old=legacy.to_dict()
    assert 'memory' not in old['analysis'] and 'comparison_version' not in old['analysis']
    legacy.to_json(tmp_path/'old.json');legacy.to_csv(tmp_path/'old.csv')
    assert QualityExperimentResult(json.loads((tmp_path/'old.json').read_text())).to_dict()==old
    rows=list(csv.DictReader((tmp_path/'old.csv').open()))
    assert list(rows[0])==['feature_id','experiment_id','source_sha256','field','value']
    assert all(not r['field'].startswith('analysis.memory') for r in rows)
    d=logical.to_dict();QualityExperimentResult(d).to_csv(tmp_path/'new.csv')
    rows=list(csv.DictReader((tmp_path/'new.csv').open()))
    assert any(r['field']=='analysis.memory.delta_bytes' for r in rows)
    assert any(r['field']=='provenance.candidate.memory.binding.model_sha256' for r in rows)


def test_state_rng_gradients_and_fp64_counts_preserved():
    m=model(extra=3);m.train();m[1].eval()
    for p in m.parameters():p.grad=torch.ones_like(p)
    before=copy.deepcopy(m.state_dict());grads=[p.grad.clone() for p in m.parameters()]
    modes=[x.training for x in m.modules()];rng=_rng_state()
    d=compare_performance_quality(m,m,cohort(),precision='fp64',iterations=2,memory_metric='cpu_rss_sampled_peak_bytes').to_dict()
    equal(m.state_dict(),before);equal(_rng_state(),rng)
    assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),grads))
    assert modes==[x.training for x in m.modules()]
    assert d['sources']['baseline']['memory']['binding']['precision']=='fp64'


def test_invalid_options_rejected_before_benchmark(monkeypatch):
    import hyso.quality as module
    monkeypatch.setattr(module,'_bundle',lambda *a,**kw:pytest.fail('unexpected benchmark'))
    for options in (dict(memory_metric='mixed_bytes'),dict(memory_protocol={}),dict(max_memory_regression=True)):
        with pytest.raises((ValueError,TypeError)):
            compare_performance_quality(model(),model(),cohort(),**options)
