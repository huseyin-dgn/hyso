"""Evidence admission/provenance boundaries, not a performance benchmark suite."""
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from torch import nn

from hyso.analysis import EvidenceError, EvidenceScope, EvidenceRecord, EvidenceIndex, compare_evidence
from hyso.performance import measure_throughput, ThroughputProtocol


@pytest.fixture(scope='module')
def native_payload():
    # Real tiny forward only to produce portable native schema evidence.
    return measure_throughput(nn.Identity(),torch.ones(2,1,3,5),
        protocol=ThroughputProtocol(batch_count=2,warmup_iterations=0,repeats=1)).to_dict()


@pytest.fixture
def artifact(tmp_path,native_payload):
    path=tmp_path/'throughput.json';path.write_text(json.dumps(native_payload),encoding='utf-8')
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def record(artifact,**kwargs):
    path,sha=artifact
    return EvidenceRecord.from_json(path,expected_sha256=sha,feature_id='P02',**kwargs)


def test_native_result_integrity_versions_defensive_copy(artifact,native_payload):
    r=record(artifact);d=r.to_dict()
    assert d['source']['schema_version']==1 and d['record_version']==d['adapter_version']==1
    assert d['native_payload']==native_payload
    assert d['metrics'][0]['value']>0 and d['scope']['device']=='cpu'
    assert d['source']['native_status']=='measured'
    assert d['bindings']['device']=='native' and d['bindings']['checkpoint_sha256']=='missing'
    assert d['observed_at_utc'] is None and d['imported_at_utc'] is not None
    d['native_payload'].clear();assert r.to_dict()['native_payload']==native_payload
    with pytest.raises(FrozenInstanceError):r._payload='changed'


@pytest.mark.parametrize('field,value', [('device','cuda:0'),('batch',99),('precision','fp64'),('input_shape',(2,1,9,9)),('protocol_id','not_this_protocol'),('phase','training')])
def test_declared_scope_cannot_override_native_facts(artifact,field,value):
    with pytest.raises(EvidenceError,match='native_scope_mismatch'):
        record(artifact,scope=EvidenceScope(**{field:value}))


@pytest.mark.parametrize('field,left,right',[
 ('model_id','model-A','model-B'),('checkpoint_sha256','a'*64,'b'*64),
 ('dataset_id','dataset-A','dataset-B'),('split','train','test'),
 ('preprocessing_id','divide255','standardize'),('backend','eager','ort'),
 ('workload_id','run1','run2'),
 ('input_kind','synthetic_recipe','real_data'),('clock_id','worker-A','worker-B')])
def test_comparison_rejects_cross_identity_domain(artifact,field,left,right):
    a=record(artifact,scope=EvidenceScope(**{field:left}));b=record(artifact,scope=EvidenceScope(**{field:right}))
    c=compare_evidence(a,b,domain='throughput')
    assert c['status']=='incompatible' and field in c['mismatched_fields']


def test_missing_and_matching_declarations_are_not_verified(artifact):
    a=record(artifact,scope=EvidenceScope(checkpoint_sha256='a'*64))
    r=compare_evidence(a,a,domain='throughput')
    assert r['status']=='insufficient_evidence'
    assert 'dataset_id' in r['missing_fields'] and 'checkpoint_sha256' in r['declared_only_fields']
    wrong=compare_evidence(a,a,domain='process_rss')
    assert wrong['status']=='incompatible' and 'measurement_domain' in wrong['mismatched_fields']


def test_corrupt_source_even_with_recomputed_file_hash_rejected(artifact):
    path,_=artifact;d=json.loads(path.read_text());d['measurements']['value']+=1
    path.write_text(json.dumps(d),encoding='utf-8');sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(EvidenceError,match='invalid_native_evidence'):
        record((path,sha))


@pytest.mark.parametrize('raw,code',[(b'{','malformed_json'),(b'{"x":1,"x":2}','duplicate_json_key'),(b'{"x":NaN}','nonfinite_json')])
def test_invalid_json(tmp_path,raw,code):
    path=tmp_path/'bad.json';path.write_bytes(raw)
    with pytest.raises(EvidenceError,match=code):record((path,hashlib.sha256(raw).hexdigest()))


def test_missing_size_limit_hash_feature_and_version_errors(artifact,tmp_path):
    with pytest.raises(EvidenceError,match='unreadable_artifact'):record((tmp_path/'missing.json','a'*64))
    with pytest.raises(EvidenceError,match='artifact_size_limit'):record(artifact,max_bytes=2)
    with pytest.raises(EvidenceError,match='artifact_sha256_mismatch'):record((artifact[0],'a'*64))
    with pytest.raises(EvidenceError,match='source_feature_mismatch'):
        EvidenceRecord.from_json(artifact[0],expected_sha256=artifact[1],feature_id='P01')
    path=artifact[0];d=json.loads(path.read_text());d['schema_version']=99;path.write_text(json.dumps(d))
    with pytest.raises(EvidenceError,match='unsupported_source_version'):record((path,hashlib.sha256(path.read_bytes()).hexdigest()))


def test_unavailable_null_not_zero_and_index_roundtrip(tmp_path):
    r=EvidenceRecord.unavailable('P08',status='unsupported',reason='INT8 recipe does not support this CNN')
    assert r.to_dict()['metrics']==[] and r.to_dict()['native_payload'] is None
    assert r.to_dict()['validation']['status']=='unsupported'
    idx=EvidenceIndex([r]);idx.to_json(tmp_path/'index.json')
    assert EvidenceIndex.from_json(tmp_path/'index.json').to_dict()==idx.to_dict()
    with pytest.raises(FileExistsError):idx.to_json(tmp_path/'index.json')
    with pytest.raises(EvidenceError,match='duplicate_record'):idx.add(r)
    with pytest.raises(EvidenceError,match='unavailable_requires_reason'):EvidenceRecord.unavailable('P01',reason='')


def test_old_native_golden_p19_remains_schema1(tmp_path):
    from fixture_support import load_fixture
    d=load_fixture('training-p19-schema1');path=tmp_path/'old.json';path.write_text(json.dumps(d),encoding='utf-8')
    r=EvidenceRecord.from_json(path,feature_id='P19',expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),origin='reused')
    assert r.to_dict()['native_payload']==d and r.to_dict()['origin']=='reused'


def test_backend_settings_dictionary_is_not_a_backend_name(tmp_path):
    from test_performance_inference import source
    from hyso.performance import analyze_latency
    result=analyze_latency(source([[1.,2.]]));path=tmp_path/'latency.json';result.to_json(path)
    r=EvidenceRecord.from_json(path,feature_id='P01',expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert r.to_dict()['scope']['backend'] is None
    assert r.to_dict()['bindings']['backend']=='missing'


def test_admission_never_executes_or_imports_factories(artifact,monkeypatch):
    from hyso.training import Trainer
    from hyso.benchmark import BenchmarkRunner
    import importlib
    def forbidden(*args,**kwargs):raise AssertionError('Evidence must not execute a workload')
    monkeypatch.setattr(Trainer,'fit',forbidden);monkeypatch.setattr(BenchmarkRunner,'run',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden);monkeypatch.setattr(torch.cuda,'current_device',forbidden)
    monkeypatch.setattr(importlib,'import_module',forbidden)
    r=record(artifact);assert r.to_dict()['validation']['status']=='native_validated'
