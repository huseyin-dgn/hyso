"""P29 full-cohort coverage with explicitly bounded inference batches."""
import copy,json
from pathlib import Path
import pytest
import torch
from torch import nn
from hyso import EvaluationCohort,QualityExperimentResult,compare_performance_quality
from hyso.quality import _hash,_derive
from hyso.checkpoint import _rng_state
from trainer_replay_support import equal


class Observed(nn.Sequential):
    sizes=[]
    def forward(self,x):
        type(self).sizes.append(len(x))
        return super().forward(x)


def fixture(n=13):
    x=torch.arange(n*2).float().reshape(n,1,1,2)/100
    return EvaluationCohort(x,[0]*n,sample_ids=list(map(str,range(n))),class_names=['0','1'],
        split='test',preprocessing_id='regression-fixture')


@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_batched_tail_timing_memory_and_state_bindings(precision,tmp_path):
    m=Observed(nn.Flatten(),nn.Linear(2,2));m.train();state=copy.deepcopy(m.state_dict());rng=_rng_state()
    Observed.sizes=[]
    result=compare_performance_quality(m,m,fixture(),precision=precision,iterations=2,batch_size=5)
    d=result.to_dict();s=d['sources'];a=d['analysis']
    assert d['schema_version']==1 and s['comparison_version']==3
    assert a['accuracy']['percentage_points']==a['macro_f1']['percentage_points']==0
    assert a['winner'] is None
    assert a['inference_protocol']['quality_samples']==13
    assert max(Observed.sizes)==5 and 3 in Observed.sizes
    for key in ('baseline','candidate'):
        bundle=s[key]
        assert bundle['evaluation']['accounting']['batch_sizes']==[5,5,3]
        assert bundle['performance']['input']['shape']==[5,1,1,2]
        assert bundle['inference_batch']['sample_ids']==list(map(str,range(5)))
        assert bundle['memory']['binding']['inference_batch_sha256']==_hash(bundle['inference_batch'])
    equal(m.state_dict(),state);equal(_rng_state(),rng);assert m.training
    result.to_json(tmp_path/'result.json');result.to_csv(tmp_path/'result.csv')
    assert QualityExperimentResult(json.loads((tmp_path/'result.json').read_text())).to_dict()==d
    text=(tmp_path/'result.csv').read_text()
    assert 'provenance.candidate.inference_batch' in text and 'provenance.candidate.memory' in text


def test_full_10000_stream_no_full_tensor_materialization(monkeypatch):
    c=fixture(10000);model=Observed(nn.Flatten(),nn.Linear(2,2));Observed.sizes=[]
    def forbidden(*a,**kw):raise AssertionError('must not reconstruct entire cohort')
    monkeypatch.setattr(EvaluationCohort,'tensors',forbidden)
    d=compare_performance_quality(model,model,c,batch_size=256,iterations=2).to_dict()
    assert max(Observed.sizes)==256 and 16 in Observed.sizes
    for key in ('baseline','candidate'):
        assert d['sources'][key]['evaluation']['accounting']['batch_sizes']==[256]*39+[16]
        assert d['sources'][key]['evaluation']['metrics']['sample_count']==10000
    with pytest.raises(ValueError,match='explicit batch_size'):
        compare_performance_quality(model,model,c)


@pytest.fixture(scope='module')
def observed():
    m=nn.Sequential(nn.Flatten(),nn.Linear(2,2))
    return compare_performance_quality(m,m,fixture(),batch_size=5,iterations=2).to_dict()


@pytest.mark.parametrize('damage',['version','policy','sample_id','start','hash','shape','missing','counts','source_id','memory','runtime'])
def test_batched_rehashed_foreign_or_misaligned_sources_rejected(observed,damage):
    d=copy.deepcopy(observed);s=d['sources'];b=s['candidate']
    if damage=='version':s['comparison_version']=2
    elif damage=='policy':s['policy']['batch_size']=6
    elif damage=='sample_id':b['inference_batch']['sample_ids'][0]='foreign'
    elif damage=='start':b['inference_batch']['sample_start']=1
    elif damage=='hash':b['inference_batch']['input_sha256']='0'*64
    elif damage=='shape':b['performance']['input']['shape'][0]=13
    elif damage=='missing':b.pop('inference_batch')
    elif damage=='counts':b['evaluation']['accounting']['batch_sizes']=[13]
    elif damage=='source_id':b['evaluation']['source']['source_id']='foreign'
    elif damage=='memory':b['memory']['binding']['inference_batch_sha256']='0'*64
    else:b['performance']['environment']['torch_version']='foreign'
    d['source_sha256']=_hash(s)
    with pytest.raises(ValueError):QualityExperimentResult(d)


def test_batched_missing_memory_never_claims_winner(observed):
    s=copy.deepcopy(observed['sources'])
    s['candidate']['memory'].update(source=None,missing_reason='not_measured',completed_forwards=0)
    a=_derive('P29',s)
    assert a['pareto_dominates'] is None and a['winner'] is None
    assert a['memory_regression'] is None


@pytest.mark.parametrize('size',[0,-1,True,1.2])
def test_invalid_explicit_batch_does_not_start_inference(monkeypatch,size):
    def forbidden(*a,**kw):raise AssertionError('invalid batch must not forward')
    monkeypatch.setattr(nn.Module,'_call_impl',forbidden)
    m=nn.Linear(2,2)
    with pytest.raises(ValueError):compare_performance_quality(m,m,fixture(),batch_size=size)


def test_archived_p29_v1_v2_json_csv_bytes_unchanged(tmp_path):
    root=Path(__file__).parents[1]
    pairs=[(root/'docs/audit/p24-p27-p28-p29-evidence/p29-comparison.json',root/'docs/audit/p24-p27-p28-p29-evidence/p29-comparison.csv')]
    folder=Path('C:/Users/hdgn5/hyso-p29-memory-evidence')
    pairs.extend((folder/f'{name}.json',folder/f'{name}.csv') for name in ('logical-native','rss-native'))
    for i,(source,csv_source) in enumerate(pairs):
        data=json.loads(source.read_text(encoding='utf-8'));result=QualityExperimentResult(data)
        assert result.to_dict()==data
        target=tmp_path/f'{i}.csv';result.to_csv(target)
        assert target.read_bytes()==csv_source.read_bytes()
