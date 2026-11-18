"""Integrated P24/P27/P28/P29 actual CPU and independent references."""
import copy
import csv
import json
import math
import os
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from hyso import (EvaluationCohort,QualityExperimentResult,measure_time_to_quality,
                  evaluate_robustness,evaluate_precision_quality,compare_performance_quality)
from hyso.quality import _derive,_delta,_hash,_precision
from hyso.checkpoint import _rng_state
from trainer_replay_support import equal


def cohort(n=12,split='test',labels=None):
    x=torch.arange(1,n+1).float().reshape(n,1,1,1)
    x=torch.cat([x,-x],dim=3)
    return EvaluationCohort(x,[1]*n if labels is None else labels,sample_ids=[str(i) for i in range(n)],
        class_names=['no','yes'],split=split,preprocessing_id='fixed-v1')


def classifier(correct=True):
    m=nn.Sequential(nn.Flatten(),nn.Linear(2,2))
    with torch.no_grad():
        m[1].weight.copy_(torch.tensor([[-1.,0.],[1.,0.]])*(1 if correct else -1));m[1].bias.zero_()
    return m


def flip(x,g,severity):return -x if severity else x


def evidence(result,name):
    directory=os.environ.get('HYSO_QUALITY_EVIDENCE')
    if directory:
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        result.to_json(path/(name+'.json'));result.to_csv(path/(name+'.csv'))
    return result


@pytest.fixture(scope='module')
def robustness():return evidence(evaluate_robustness(classifier(),cohort(),flip,corruption_id='sign-flip',severity=1.,seed=21),'p27-paired')


def test_robustness_actual_pairing_delta_rng_state(robustness):
    d=robustness.to_dict();assert d['analysis']['accuracy']['percentage_points']==-100
    assert d['analysis']['macro_f1']['percentage_points']==-50
    assert d['analysis']['paired_samples']==12
    m=classifier();before=copy.deepcopy(m.state_dict());rng=_rng_state()
    evaluate_robustness(m,cohort(),flip,corruption_id='identity',severity=0.,seed=99)
    equal(m.state_dict(),before);equal(_rng_state(),rng)


@pytest.mark.parametrize('field',['sample_ids','targets','preprocessing_id','configuration','inputs'])
def test_robustness_mismatch_rejected(robustness,field):
    s=robustness.to_dict()['sources'];b=s['perturbed_cohort']
    if field=='sample_ids':b[field].reverse()
    elif field=='targets':b[field][0]=0
    elif field=='preprocessing_id':b[field]='other'
    elif field=='configuration':b[field]['split']='other'
    else:b[field].pop()
    with pytest.raises((ValueError,KeyError)):_derive('P27',s)


def test_perturbation_failure_no_result():
    m=classifier();rng=_rng_state()
    def bad(x,g,s):raise RuntimeError('perturbation failed')
    with pytest.raises(RuntimeError):evaluate_robustness(m,cohort(),bad,corruption_id='bad',severity=.5,seed=2)
    equal(_rng_state(),rng)
    with pytest.raises(ValueError):evaluate_robustness(classifier(),cohort(),lambda x,g,s:x[:,:,:,:1],corruption_id='bad',severity=.5,seed=2)


@pytest.mark.parametrize('precision',['fp32','fp64','bf16','int8_dynamic'])
def test_actual_precision_recipes_quality(precision):
    m=classifier();before=copy.deepcopy(m.state_dict());rng=_rng_state()
    r=evidence(evaluate_precision_quality(m,cohort(),candidate=precision),'p28-'+precision).to_dict()
    assert r['analysis']['accuracy']['baseline']==1
    assert r['analysis']['accuracy']['candidate']==1
    assert r['analysis']['accuracy']['percentage_points']==0
    assert r['sources']['candidate_precision']['policy']==precision
    equal(m.state_dict(),before);equal(_rng_state(),rng)


def test_delta_units_zero_baseline_and_unsupported():
    assert _delta(.5,.6)['percentage_points']==pytest.approx(10)
    assert _delta(.5,.6)['relative_percent']==pytest.approx(20)
    assert _delta(0,.6)['relative_percent'] is None
    r=evaluate_precision_quality(classifier(False),cohort(),candidate='fp64').to_dict()
    assert r['analysis']['accuracy']['relative_status']=='undefined_zero_baseline'
    with pytest.raises(NotImplementedError):evaluate_precision_quality(classifier(),cohort(),candidate='amp_fp16')


def train_factory(seed):
    from hyso.training import Trainer
    m=nn.Sequential(nn.Flatten(),nn.Linear(2,2))
    with torch.no_grad():m[1].weight.zero_();m[1].bias.zero_()
    trainer=Trainer(m,optimizer=torch.optim.SGD(m.parameters(),lr=1),loss_fn=nn.CrossEntropyLoss(),device='cpu')
    return trainer,DataLoader(TensorDataset(torch.zeros(4,1,1,2),torch.ones(4,dtype=torch.long)),batch_size=4)


@pytest.fixture(scope='module')
def target_result():return evidence(measure_time_to_quality(train_factory,cohort(4,split='holdout'),seeds=[1,2,3],target=1.,epochs=2),'p24-reached')


def test_real_repeated_seeds_first_crossing_and_eval_exclusion(target_result):
    d=target_result.to_dict();a=d['analysis'];assert a['reached']==3 and a['censored']==0
    times=[]
    for run,summary in zip(d['sources']['runs'],a['runs']):
        crossing=summary['first_observed_crossing'];assert crossing['epoch']==1
        assert crossing['training_seconds']==run['timing']['epochs'][0]['training_s']
        assert crossing['interval_low_seconds']==0
        assert run['timing']['actual_epochs']==2
        assert run['timing']['epochs'][0]['total_s']>crossing['training_seconds']
        times.append(crossing['training_seconds'])
    mean=sum(times)/3
    assert a['reached_sample_variance_seconds2']==pytest.approx(sum((t-mean)**2 for t in times)/2)


def test_real_censoring_and_initial_crossing():
    def frozen(seed):
        t,l=train_factory(seed)
        for group in t.optimizer.param_groups:group['lr']=0.
        return t,l
    c=cohort(4,split='holdout')
    r=evidence(measure_time_to_quality(frozen,c,seeds=[11,12],target=1.,epochs=2,cadence=2),'p24-censored').to_dict()
    assert r['analysis']['censored']==2
    assert r['analysis']['reached_mean_seconds'] is None
    assert r['analysis']['reached_sample_variance_seconds2'] is None
    assert all(x['censor_training_seconds']>0 for x in r['analysis']['runs'])
    r=measure_time_to_quality(frozen,c,seeds=[11,12],target=0.,epochs=1).to_dict()
    assert all(x['first_observed_crossing']['training_seconds']==0 for x in r['analysis']['runs'])


@pytest.mark.parametrize('damage',['cadence','seed','training','evaluation','overlap','data'])
def test_target_invalid_evidence_rejected(target_result,damage):
    s=target_result.to_dict()['sources']
    if damage=='cadence':s['cadence']=2
    elif damage=='seed':s['runs'][1]['seed']=1
    elif damage=='training':s['runs'][1]['timing']=copy.deepcopy(s['runs'][0]['timing'])
    elif damage=='evaluation':s['runs'][0]['checkpoints'].pop()
    elif damage=='overlap':s['runs'][0]['checkpoints'][1]['start_ns']=s['runs'][0]['timing']['epochs'][0]['train_start_ns']
    else:s['runs'][1]['training_data_sha256']='foreign'
    with pytest.raises(ValueError):_derive('P24',s)


def test_target_failure_and_budgets():
    rng=_rng_state()
    def bad(seed):raise RuntimeError('factory failed')
    with pytest.raises(RuntimeError):measure_time_to_quality(bad,cohort(4),seeds=[1,2],target=.9)
    equal(_rng_state(),rng)
    for seeds in ([1],[1,1],list(range(9))):
        with pytest.raises(ValueError):measure_time_to_quality(train_factory,cohort(4),seeds=seeds,target=.9)


@pytest.fixture(scope='module')
def comparison():return evidence(compare_performance_quality(classifier(False),classifier(),cohort(),iterations=3,independent_samples=True,min_quality=.8),'p29-comparison')


def test_actual_pareto_constraints_regression_and_exact_test(comparison):
    d=comparison.to_dict();a=d['analysis']
    assert a['quality_constraint_met'] and not a['quality_regression']
    assert a['quality']['percentage_points']==100
    assert a['quality_significance']['p_value']==2/2**12
    assert a['quality_significance']['significant']
    la=d['sources']['baseline']['performance']['stats']['pooled']['mean_ms'];lb=d['sources']['candidate']['performance']['stats']['pooled']['mean_ms']
    assert a['pareto_dominates']==('candidate' if lb<=la else 'neither')
    assert a['timing_significance']['p_value'] is None


def test_significance_insufficient_and_not_attested(comparison):
    s=comparison.to_dict()['sources'];s['policy']['independent_samples']=False
    assert _derive('P29',s)['quality_significance']['p_value'] is None
    s['candidate']=copy.deepcopy(s['baseline']);s['policy']['independent_samples']=True
    a=_derive('P29',s);assert a['quality_significance']['status']=='insufficient_discordant_pairs'
    assert a['pareto_dominates']=='neither'
    s['policy']['min_quality']=1.
    assert not _derive('P29',s)['quality_constraint_met']


@pytest.mark.parametrize('damage',['host','precision','data','protocol','device','runtime'])
def test_comparison_mismatch_rejected(comparison,damage):
    s=comparison.to_dict()['sources']
    if damage=='host':s['candidate']['host_id']='other'
    elif damage=='precision':s['candidate']['precision']['policy']='fp64'
    elif damage=='data':s['candidate']['evaluation']['source']['source_id']='other'
    elif damage=='protocol':s['candidate']['performance']['configuration']['measure_iterations']+=1
    elif damage=='device':s['candidate']['performance']['environment']['device']='cuda:0'
    else:s['candidate']['performance']['environment']['python_version']='other'
    with pytest.raises(ValueError):_derive('P29',s)


@pytest.mark.parametrize('source',['robustness','target_result','comparison'])
def test_artifact_integrity_roundtrip_no_clobber(request,source,tmp_path):
    r=request.getfixturevalue(source);d=r.to_dict();p=tmp_path/'result.json';c=tmp_path/'result.csv'
    r.to_json(p);r.to_csv(c)
    assert QualityExperimentResult(json.loads(p.read_text())).to_dict()==d
    with pytest.raises(FileExistsError):r.to_json(p)
    with pytest.raises(FileExistsError):r.to_csv(c)
    with c.open(newline='') as f:assert list(csv.DictReader(f))
    detached=r.to_dict();detached['sources'].clear();assert r.to_dict()==d
    with pytest.raises((TypeError,AttributeError)):r._payload='bad'
    for field in ('analysis','sources','source_sha256'):
        bad=copy.deepcopy(d);bad[field]={} if field!='source_sha256' else 'bad'
        with pytest.raises(ValueError):QualityExperimentResult(bad)


def test_precision_artifact_roundtrip(tmp_path):
    r=evaluate_precision_quality(classifier(),cohort(),candidate='fp64')
    assert QualityExperimentResult(r.to_dict()).to_dict()==r.to_dict()
    r.to_json(tmp_path/'precision.json');r.to_csv(tmp_path/'precision.csv')
    for export,path in ((r.to_json,tmp_path/'precision.json'),(r.to_csv,tmp_path/'precision.csv')):
        with pytest.raises(FileExistsError):export(path)
    d=r.to_dict();d['sources']['candidate_precision']['verified_execution']['batches']=0
    d['source_sha256']=_hash(d['sources'])
    with pytest.raises(ValueError):QualityExperimentResult(d)


@pytest.mark.parametrize('damage',['benchmark_input','swapped_timing','weights','witness','quality_runtime','batch_protocol','architecture'])
def test_independent_cross_source_binding(comparison,damage):
    s=comparison.to_dict()['sources'];b=s['candidate']
    if damage=='benchmark_input':b['performance']['input']['content_sha256']='0'*64
    elif damage=='architecture':b['performance']['model']['structural_descriptor']['modules'][0]['qualified_class']='other.Model'
    elif damage=='swapped_timing':
        for key in ('performance','benchmark_model_sha256','state_witness'):
            b[key]=copy.deepcopy(s['baseline'][key])
    elif damage=='weights':b['state_witness'][0]['bytes_hex']='00'*len(bytes.fromhex(b['state_witness'][0]['bytes_hex']))
    elif damage=='witness':b['state_witness'].pop()
    else:
        ev=b['evaluation']
        if damage=='quality_runtime':ev['provenance']['torch']='other'
        else:ev['accounting']['batch_sizes']=[6,6];b['precision']['verified_execution']['batches']=2
        ev['source_sha256']=_hash({k:v for k,v in ev.items() if k not in ('source_sha256','metrics','status')})
    with pytest.raises(ValueError):_derive('P29',s)


def test_macro_f1_no_invalid_significance(comparison):
    s=comparison.to_dict()['sources'];s['policy']['metric']='macro_f1'
    a=_derive('P29',s)
    assert a['quality_significance']['status']=='unsupported_mcnemar_is_accuracy_only'
    assert a['quality_significance']['p_value'] is None
    s['baseline'],s['candidate']=s['candidate'],s['baseline']
    assert _derive('P29',s)['quality_regression']


def test_model_buffers_modes_rng_and_fp64_binding():
    m=classifier();m.register_buffer('offset',torch.tensor(1.))
    m.register_buffer('counter',torch.tensor(0));m.train();m[1].eval()
    before=copy.deepcopy(m.state_dict());modes=[c.training for c in m.modules()];rng=_rng_state()
    r=compare_performance_quality(m,m,cohort(4),precision='fp64',iterations=2)
    assert r.to_dict()['analysis']['quality']['percentage_points']==0
    equal(m.state_dict(),before);equal(_rng_state(),rng)
    assert [c.training for c in m.modules()]==modes


def test_precision_dtype_fallback_and_missing_quantized_operator_rejected():
    class CastBack(nn.Module):
        def forward(self,x):return x.float().reshape(len(x),2)
    with pytest.raises(ValueError,match='fallback'):
        evaluate_precision_quality(CastBack(),cohort(),candidate='bf16')
    from hyso.quality import _PrecisionAdapter
    q,_=_precision(classifier(),'int8_dynamic')
    # A quantized module exists but its Linear is skipped by the executed path.
    class Skip(nn.Sequential):
        def forward(self,x):return x.reshape(len(x),2)
    q.model.__class__=Skip
    with pytest.raises(ValueError,match='execution not observed'):q(cohort().tensors()[0])


@pytest.mark.parametrize('damage',['empty','nan','duplicate_ids','invalid_label','unsupported_task','infinite_severity'])
def test_cohort_and_corruption_boundaries(damage):
    x,y=cohort(2).tensors();kw=dict(sample_ids=['a','b'],class_names=['no','yes'],split='test',preprocessing_id='v1')
    if damage=='infinite_severity':
        with pytest.raises(ValueError):evaluate_robustness(classifier(),cohort(),flip,corruption_id='bad',severity=float('inf'),seed=1)
        return
    if damage=='empty':x=x[:0];y=y[:0];kw['sample_ids']=[]
    elif damage=='nan':x[0,0,0,0]=float('nan')
    elif damage=='duplicate_ids':kw['sample_ids']=['a','a']
    elif damage=='invalid_label':y[0]=2
    else:kw['task']='detection'
    with pytest.raises((ValueError,NotImplementedError)):EvaluationCohort(x,y,**kw)


def test_binary_threshold_and_seeded_random_corruption():
    m=nn.Sequential(nn.Flatten(),nn.Linear(2,1))
    with torch.no_grad():m[1].weight.zero_();m[1].bias.zero_()
    x,_=cohort(4).tensors()
    c=EvaluationCohort(x,[1]*4,sample_ids=list('abcd'),class_names=['no','yes'],split='test',preprocessing_id='v1',task='binary')
    def noise(x,g,s):return x+s*torch.randn(x.shape,generator=g)
    a=evaluate_robustness(m,c,noise,corruption_id='gaussian',severity=.1,seed=7).to_dict()
    b=evaluate_robustness(m,c,noise,corruption_id='gaussian',severity=.1,seed=7).to_dict()
    assert a['sources']['perturbed_cohort']==b['sources']['perturbed_cohort']
    assert a['analysis']['accuracy']['baseline']==1


def test_failures_preserve_model_rng_no_success():
    class Fail(nn.Module):
        def forward(self,x):raise RuntimeError('inference failed')
    m=Fail();baseline=classifier();rng=_rng_state()
    with pytest.raises(RuntimeError,match='inference failed'):
        evaluate_robustness(m,cohort(),flip,corruption_id='id',severity=0,seed=1)
    equal(_rng_state(),rng)
    with pytest.raises(RuntimeError,match='inference failed'):
        compare_performance_quality(baseline,m,cohort(),iterations=2)
    equal(_rng_state(),rng)


def test_seed_reuse_and_incomplete_training_fail():
    trainer,loader=train_factory(1)
    with pytest.raises(ValueError,match='fresh'):
        measure_time_to_quality(lambda seed:(trainer,loader),cohort(4),seeds=[1,2],target=1,epochs=1)
    def fail(seed):
        t,l=train_factory(seed)
        class TrainFailure(nn.Module):
            def __init__(self,model):super().__init__();self.model=model
            def forward(self,x):
                if self.training:raise RuntimeError('training failed')
                return self.model(x)
        t.model=TrainFailure(t.model)
        return t,l
    rng=_rng_state()
    with pytest.raises(RuntimeError,match='training failed'):
        measure_time_to_quality(fail,cohort(4),seeds=[1,2],target=1,epochs=1)
    equal(_rng_state(),rng)


def test_rehashed_analysis_tamper_rejected(robustness):
    d=robustness.to_dict();d['analysis']['accuracy']['percentage_points']=0
    d['source_sha256']=_hash(d['sources'])
    with pytest.raises(ValueError):QualityExperimentResult(d)


def test_seed_training_protocol_mismatch_rejected(target_result):
    s=target_result.to_dict()['sources'];s['runs'][1]['training_protocol']['grad_clip']=1.
    with pytest.raises(ValueError,match='protocol'):_derive('P24',s)
    t,l=train_factory(1)
    custom=DataLoader(l.dataset,batch_size=4,collate_fn=lambda samples:samples)
    with pytest.raises(ValueError,match='sequential'):
        measure_time_to_quality(lambda seed:(t,custom),cohort(4),seeds=[1,2],target=1,epochs=1)


def test_actual_training_parity_and_rng():
    owners=[]
    def owned(seed):
        t,l=train_factory(seed);owners.append(t);return t,l
    rng=_rng_state()
    measure_time_to_quality(owned,cohort(4),seeds=[5,6],target=1,epochs=1)
    equal(_rng_state(),rng)
    control,loader=train_factory(5);control.fit(loader,epochs=1)
    for t in owners:
        equal(t.model.state_dict(),control.model.state_dict())
        equal(t.optimizer.state_dict(),control.optimizer.state_dict())
        assert t.history==control.history and t.update_count==control.update_count
