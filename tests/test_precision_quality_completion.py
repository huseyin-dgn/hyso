"""P28 CPU-only regression fixtures; no performance sweeps or model training."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from hyso.quality import (EvaluationCohort,QualityExperimentResult,evaluate_precision_quality,
    _cohort,_delta,_hash,_input_chunks,_paired)
from hyso.checkpoint import _rng_state
from test_quality_experiments import classifier,cohort,equal


@pytest.mark.parametrize('n,width',[(10000,2),(1300,784)])
def test_large_lossless_cohort_streams_without_materializing_all_tensors(monkeypatch,n,width):
    x=torch.arange(n*width,dtype=torch.float32).remainder(11).reshape(n,1,1,width)/11
    c=EvaluationCohort(x,[0]*n,sample_ids=[str(i) for i in range(n)],class_names=['0','1'],
        split='test',preprocessing_id='fixture-unfitted')
    data=c.to_dict()
    assert type(data['inputs']) is dict
    assert torch.equal(torch.cat(list(_input_chunks(data))),x)
    assert _cohort(data).sha256==c.sha256
    def forbidden(self):raise AssertionError('large evaluation must stream')
    monkeypatch.setattr(EvaluationCohort,'tensors',forbidden)
    model=nn.Sequential(nn.Flatten(),nn.Linear(width,2))
    result=evaluate_precision_quality(model,c,candidate='fp64',batch_size=256).to_dict()
    assert result['analysis']['paired_samples']==n
    for key in ('baseline','candidate'):
        assert result['sources'][key]['metrics']['sample_count']==n
        assert sum(result['sources'][key]['accounting']['batch_sizes'])==n
        assert result['sources'][key+'_precision']['observed_input_stream_sha256']==hashlib.sha256(x.numpy().tobytes()).hexdigest()


@pytest.mark.parametrize('damage',['missing_chunk','duplicate_chunk','bytes','hash','shape','label','ids','dtype'])
def test_encoded_cohort_rejects_corrupt_or_misaligned_data(damage):
    c=EvaluationCohort(torch.zeros(5000,1,1,2),[0]*5000,sample_ids=list(map(str,range(5000))),
        class_names=['a','b'],split='test',preprocessing_id='fixture')
    d=c.to_dict()
    if damage=='missing_chunk':d['inputs']['chunks']=[]
    elif damage=='duplicate_chunk':d['inputs']['chunks']*=2
    elif damage=='bytes':d['inputs']['chunks'][0]['data']='YWJj'
    elif damage=='hash':d['inputs']['chunks'][0]['sha256']='0'*64
    elif damage=='shape':d['inputs']['shape'][0]+=1
    elif damage=='label':d['targets'].pop()
    elif damage=='ids':d['sample_ids'][1]=d['sample_ids'][0]
    else:d['dtype']='torch.float64'
    with pytest.raises(ValueError):_cohort(d)


@pytest.mark.parametrize('field',['sample_ids','targets','preprocessing_id','configuration','inputs'])
def test_exact_pairing_rejected(field):
    a=cohort().to_dict();b=copy.deepcopy(a)
    if field=='sample_ids':b[field].reverse()
    elif field=='targets':b[field][0]=0
    elif field=='preprocessing_id':b[field]='different'
    elif field=='configuration':b[field]['class_names'].reverse()
    else:b[field][0][0][0][0]+=1
    with pytest.raises(ValueError):_paired(a,b,same_inputs=True)


@pytest.mark.parametrize('policy',['fp32','fp64','bf16'])
def test_real_batch_dtype_operator_input_and_state_proof(policy):
    model=classifier();state=copy.deepcopy(model.state_dict());rng=_rng_state()
    d=evaluate_precision_quality(model,cohort(),candidate=policy,batch_size=5).to_dict()
    for key in ('baseline','candidate'):
        ev=d['sources'][key];proof=d['sources'][key+'_precision']
        assert ev['accounting']['batch_sizes']==[5,5,2]
        assert proof['execution_version']==2 and proof['device']=='cpu'
        assert len(proof['batch_observations'])==3
        assert all(row['finite'] and row['operators'] for row in proof['batch_observations'])
    equal(model.state_dict(),state);equal(_rng_state(),rng)
    assert d['analysis']['changed_predictions']==len(d['analysis']['prediction_changes'])
    # Defaults may exceed cohort length: the single partial batch remains valid.
    assert evaluate_precision_quality(model,cohort(),candidate='fp32',batch_size=100).to_dict()['sources']['baseline']['accounting']['batch_sizes']==[12]


@pytest.mark.parametrize('damage',['operator','autocast','stream','batch_hash','shape','batch','device','targets','order'])
def test_rehashed_execution_or_pairing_tamper_rejected(damage):
    d=evaluate_precision_quality(classifier(),cohort(),candidate='bf16',batch_size=5).to_dict();s=d['sources']
    proof=s['candidate_precision'];row=proof['batch_observations'][0]
    if damage=='operator':row['operators'][0]['output_dtype']='torch.float32'
    elif damage=='autocast':row['autocast_enabled']=False
    elif damage=='stream':proof['observed_input_stream_sha256']='0'*64
    elif damage=='batch_hash':row['input_sha256']='0'*64
    elif damage=='shape':row['output_shape']=[5,3]
    elif damage=='batch':proof['batch_observations'].pop()
    elif damage=='device':proof['device']='cuda:0'
    elif damage=='targets':s['cohort']['targets'][0]=0
    else:s['cohort']['sample_ids'].reverse()
    d['source_sha256']=_hash(s)
    with pytest.raises(ValueError):QualityExperimentResult(d)


def test_independent_metrics_and_percentage_units():
    from hyso.evaluation import evaluate_predictions
    d=evaluate_predictions([0,0,1,1],[0,1,1,1],class_names=['a','b'],split='test').to_dict()['metrics']
    assert d['accuracy']==.75 and d['confusion_matrix']==[[1,1],[0,2]]
    assert d['averages']['macro']['precision']==pytest.approx((1+2/3)/2)
    assert d['averages']['macro']['recall']==.75
    assert d['averages']['macro']['f1']==pytest.approx((2/3+4/5)/2)
    assert _delta(.8,.75)['percentage_points']==pytest.approx(-5)
    assert _delta(.8,.75)['relative_percent']==pytest.approx(-6.25)
    assert _delta(0,1)['relative_percent'] is None
    assert _delta(0,1)['relative_status']=='undefined_zero_baseline'


def test_unsupported_cnn_int8_and_cuda_unavailable_no_fallback(monkeypatch):
    cnn=nn.Sequential(nn.Conv2d(1,2,1),nn.Flatten(),nn.Linear(4,2))
    with pytest.raises(ValueError,match='INT8 recipe'):evaluate_precision_quality(cnn,cohort(),candidate='int8_dynamic')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(NotImplementedError,match='no CPU fallback'):
        evaluate_precision_quality(classifier(),cohort(),candidate='amp_fp16',device='cuda:0')


def test_native_nonfinite_output_rejected_and_rng_preserved():
    class Invalid(nn.Module):
        def forward(self,x):return torch.full((len(x),2),float('nan'))
    rng=_rng_state()
    with pytest.raises(ValueError):evaluate_precision_quality(Invalid(),cohort(),candidate='fp32')
    equal(_rng_state(),rng)


def test_export_reopen_and_legacy_p28_results(tmp_path):
    result=evaluate_precision_quality(classifier(),cohort(),candidate='fp64')
    result.to_json(tmp_path/'quality.json');result.to_csv(tmp_path/'quality.csv')
    assert QualityExperimentResult(json.loads((tmp_path/'quality.json').read_text())).to_dict()==result.to_dict()
    assert 'analysis.changed_predictions' in (tmp_path/'quality.csv').read_text()
    folder=Path(__file__).parents[1]/'docs/audit/p24-p27-p28-p29-evidence'
    for policy in ('fp32','fp64','bf16','int8_dynamic'):
        data=json.loads((folder/f'p28-{policy}.json').read_text(encoding='utf-8'))
        assert QualityExperimentResult(data).to_dict()==data
