"""Small synthetic capture fixtures; no benchmark/model/device execution."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope, compare_evidence
from hyso.analysis._codec import digest
from hyso.advisor import (AdvisorError, ComparisonContract, ComparisonResult,
                          NativeMetricResolver, compare_records)


def payload(run, value):
    rows=[[value]*20]
    return dict(format='hyso.performance.latency_analysis',schema_version=1,
        result_id=run+'00000000-0000-0000-0000-000000000000'[1:],feature_ids=['P01'],
        identity=dict(experiment_run_id=None,source_result_id=run),
        protocol=dict(id='resident_tail',version=1,scope='synchronous_host_forward',method='linear_(n-1)*q',
                      quantiles=[.95],tail_policy=dict(min_samples=20,min_tail_samples=1)),
        provenance=dict(model=dict(weights_sha256='a'*64,structural_descriptor={'modules':['fixture']}),
            input=dict(shape=[1,1,2,2],batch_size=1,content_sha256='b'*64,dtype='torch.float32',layout='torch.strided'),
            environment=dict(device='cpu',backend='fixture-native-backend',torch_version='fixture',python_version='fixture',operating_system='fixture'),
            configuration=dict(precision='fp32',warmup_iterations=0,measure_iterations=20,repeats=1,inference_mode=True,
                               forced_model_mode='eval',input_policy='fixed_resident_input',autocast_enabled=False),
            comparison_capture=dict(format='hyso.native_comparison_capture',schema_version=1,authority='native_capture',
                run_id=run,sample_sha256=digest(rows),hardware_id='c'*64,checkpoint_sha256='d'*64,
                cohort_sha256='e'*64,preprocessing_sha256='f'*64,input_family_sha256='1'*64,
                dataset_id='fixture-dataset',split='test',workload_id='fixture-workload',clock_id='fixture-clock')),
        requested=dict(samples=20,repeats=1),actual=dict(samples=20,repeats=1,samples_per_repeat=[20]),
        measurements=[dict(quantile=.95,value=value,unit='ms',status='derived',reason=None,effective_tail_count=1.0)],
        raw_evidence=dict(source_result_id=run,source_sha256=digest(dict(run=run,rows=rows)),latency_samples_ms=rows),
        diagnostics=dict(warnings=[],limitations=['Synthetic fixture, not a performance claim.']))


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from hyso.benchmark import BenchmarkRunner
        from hyso.training import Trainer
        def forbidden(*a,**k):raise AssertionError('Workload/device call forbidden')
        stack=ExitStack();cls.addClassCleanup(stack.close)
        for obj,name in [(torch.cuda,'init'),(torch.cuda,'is_available'),(torch.cuda,'device_count'),
                         (torch.nn.Module,'_call_impl'),(BenchmarkRunner,'run'),(Trainer,'fit')]:
            stack.enter_context(patch.object(obj,name,forbidden))
        cls.contract=ComparisonContract(min_samples=20,min_tail_samples=1)

    def make(self, left=None, right=None, declared_dataset=None):
        records=[]
        with tempfile.TemporaryDirectory() as directory:
            for i,n in enumerate((left or payload('1',10.),right or payload('2',7.))):
                raw=json.dumps(n,sort_keys=True).encode();path=Path(directory)/(str(i)+'.json');path.write_bytes(raw)
                records.append(EvidenceRecord.from_json(path,feature_id='P01',expected_sha256=hashlib.sha256(raw).hexdigest(),
                    scope=EvidenceScope(model_id='caller-label',dataset_id=declared_dataset or n['provenance'].get('comparison_capture',{}).get('dataset_id','fixture-dataset'),
                        checkpoint_sha256=n['provenance'].get('comparison_capture',{}).get('checkpoint_sha256','d'*64))))
        source=EvidenceIndex(records).analyze(context=AnalysisContext(workload_id='fixture',task='classification',model={},dataset={}))
        return NativeMetricResolver(source),records

    def result(self,left=None,right=None,contract=None):
        resolver,records=self.make(left,right)
        return compare_records(resolver,records[0].record_id,resolver,records[1].record_id,contract=contract or self.contract)

    def test_independent_native_captures_comparable_no_delta(self):
        r=self.result();self.assertEqual(r.status,'COMPARABLE')
        self.assertEqual((r.performance_effect,r.statistical_independence),('not_computed','not_certified'))
        self.assertEqual(ComparisonResult.from_dict(r.to_dict()),r)

    def test_model_and_checkpoint_change_requires_permission(self):
        right=payload('2',7.)
        right['provenance']['model']['weights_sha256']='3'*64
        right['provenance']['model']['structural_descriptor']={'modules':['new-fixture']}
        right['provenance']['comparison_capture']['checkpoint_sha256']='4'*64
        self.assertEqual(self.result(right=right).status,'INCOMPATIBLE')
        self.assertEqual(self.result(right=right,contract=replace(self.contract,allowed_changes=('model',))).status,'COMPARABLE')

    def test_batch_change_explicit_and_nonbatch_shape_fixed(self):
        n=payload('2',7.);n['provenance']['input'].update(shape=[2,1,2,2],batch_size=2,content_sha256='5'*64)
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')
        allowed=replace(self.contract,allowed_changes=('batch',))
        self.assertEqual(self.result(right=n,contract=allowed).status,'COMPARABLE')
        n['provenance']['input']['shape']=[2,1,3,2]
        self.assertEqual(self.result(right=n,contract=allowed).status,'INCOMPATIBLE')

    def test_missing_or_declared_native_metadata_never_promoted(self):
        for mutate in (lambda p:p['provenance'].pop('comparison_capture'),
                       lambda p:p['provenance']['comparison_capture'].update(authority='declared'),
                       lambda p:p['provenance']['comparison_capture'].pop('hardware_id')):
            n=payload('2',7.);mutate(n)
            self.assertEqual(self.result(right=n).status,'INSUFFICIENT_EVIDENCE')

    def test_hardware_backend_protocol_data_mismatches(self):
        variants=[('hardware_id','6'*64),('dataset_id','other'),('split','validation'),('cohort_sha256','7'*64),
                  ('preprocessing_sha256','8'*64),('input_family_sha256','9'*64),('clock_id','other')]
        for key,value in variants:
            n=payload('2',7.);n['provenance']['comparison_capture'][key]=value
            with self.subTest(field=key):self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')
        n=payload('2',7.);n['provenance']['environment']['backend']='other'
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')
        n=payload('2',7.);n['provenance']['configuration']['warmup_iterations']=1
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')

    def test_same_underlying_capture_even_with_new_analysis_id(self):
        n=payload('1',7.)
        n['result_id']='30000000-0000-0000-0000-000000000000'
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')
        resolver,records=self.make()
        self.assertEqual(compare_records(resolver,records[0].record_id,resolver,records[0].record_id,contract=self.contract).status,'INCOMPATIBLE')

    def test_sample_binding_and_count(self):
        n=payload('2',7.);n['provenance']['comparison_capture']['sample_sha256']='0'*64
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')
        self.assertEqual(self.result(contract=replace(self.contract,min_samples=21)).status,'INSUFFICIENT_EVIDENCE')
        self.assertEqual(self.result(contract=replace(self.contract,min_tail_samples=2)).status,'INSUFFICIENT_EVIDENCE')

    def test_declarations_cannot_override_native_capture(self):
        resolver,records=self.make(declared_dataset='conflicting-caller-label')
        result=compare_records(resolver,records[0].record_id,resolver,records[1].record_id,contract=self.contract)
        self.assertEqual(result.status,'INCOMPATIBLE')
        self.assertTrue(any(r.code=='declaration_conflicts_with_capture' for r in result.reasons))

    def test_precision_requires_permission_and_does_not_ignore_other_settings(self):
        n=payload('2',7.);n['provenance']['configuration']['precision']='fp64';n['provenance']['input']['dtype']='torch.float64'
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')
        self.assertEqual(self.result(right=n,contract=replace(self.contract,allowed_changes=('precision',))).status,'COMPARABLE')
        n=payload('2',7.);n['provenance']['configuration']['autocast_enabled']=True
        self.assertEqual(self.result(right=n).status,'INCOMPATIBLE')

    def test_invalid_policy_and_unknown_version(self):
        for kwargs in (dict(allowed_changes=('hardware',)),dict(min_samples=0),dict(min_tail_samples=True)):
            with self.assertRaises(AdvisorError):ComparisonContract(**kwargs)
        d=self.contract.to_dict();d['schema_version']=2
        with self.assertRaises(AdvisorError):ComparisonContract.from_dict(d)

    def test_existing_record_stays_insufficient_and_3a_is_unchanged(self):
        n=json.loads((Path(__file__).parent/'fixtures/advisor-native-v1.json').read_text())['payloads']['P01']
        other=copy.deepcopy(n);other['result_id']='30000000-0000-0000-0000-000000000000'
        other['identity']['source_result_id']='new-fixture-run'
        other['raw_evidence'].update(source_result_id='new-fixture-run',source_sha256='9'*64)
        resolver,records=self.make(left=n,right=other)
        before=records[0].to_dict()
        r=compare_records(resolver,records[0].record_id,resolver,records[1].record_id,contract=self.contract)
        self.assertEqual(r.status,'INSUFFICIENT_EVIDENCE')
        self.assertEqual(records[0].to_dict(),before)
        self.assertNotEqual(compare_evidence(records[0],records[1],domain='latency')['status'],'compatible')
        self.assertEqual({b.field:b.authority for b in r.left.bindings}['checkpoint_sha256'],'declared')

    def test_reused_source_hash_not_independent(self):
        left,right=payload('1',10.),payload('2',7.)
        right['raw_evidence']['source_sha256']=left['raw_evidence']['source_sha256']
        self.assertEqual(self.result(left,right).status,'INCOMPATIBLE')

    def test_known_protocol_mismatch_is_not_hidden_by_missing_metadata(self):
        left=payload('1',10.);left['provenance'].pop('comparison_capture')
        right=payload('2',7.);right['provenance']['configuration']['warmup_iterations']=1
        self.assertEqual(self.result(left,right).status,'INCOMPATIBLE')
        right=payload('2',7.);right['provenance']['input']['content_sha256']='8'*64
        self.assertEqual(self.result(left,right).status,'INCOMPATIBLE')


if __name__=='__main__':unittest.main(verbosity=2)
