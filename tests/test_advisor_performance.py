"""Tiny read-only comparisons; arithmetic fixtures never certify provenance."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord
from hyso.analysis._codec import digest
from hyso.advisor import (AdvisorError, ComparisonContract, NativeMetric as M,
    NativeMetricResolver, PerformanceComparison, compare_performance)
from hyso.advisor._json import canonical
from hyso.advisor.performance import _difference
from test_advisor_comparison import payload
import test_advisor_comparison as fixtures


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from hyso.benchmark import BenchmarkRunner
        from hyso.training import Trainer
        stack = ExitStack(); cls.addClassCleanup(stack.close)
        def forbidden(*a, **k): raise AssertionError('Workload forbidden')
        for obj, name in ((torch.cuda,'init'),(torch.cuda,'is_available'),(torch.cuda,'device_count'),
                          (torch.nn.Module,'_call_impl'),(BenchmarkRunner,'run'),(Trainer,'fit')):
            stack.enter_context(patch.object(obj,name,forbidden))
        cls.policy = ComparisonContract(min_samples=20,min_tail_samples=1)

    def pair(self, left=None, right=None):
        return fixtures.ComparisonTests().make(left,right)

    def run_pair(self, left=None, right=None, policy=None):
        resolver, records = self.pair(left,right)
        return compare_performance(resolver,records[0].record_id,resolver,records[1].record_id,
                                   contract=policy or self.policy)

    def scalar_pair(self, metric, a, b, unit, domain):
        """Isolated arithmetic test inputs, not an eligibility bypass API."""
        r, records = self.pair()
        x,y = [r.resolve(v.record_id,M.FORWARD_P95).resolved for v in records]
        return tuple(replace(m,metric=metric,value=v,reference=replace(m.reference,unit=unit,domain=domain))
                     for m,v in ((x,a),(y,b)))

    def test_latency_and_deterministic_roundtrip(self):
        result=self.run_pair(); d=result.difference
        self.assertEqual((d.baseline.value,d.candidate.value,d.difference,d.percent_change), (10.,7.,-3.,-30.))
        self.assertEqual((d.observed_direction,d.metric_effect),('decrease','favorable'))
        self.assertEqual(PerformanceComparison.from_dict(result.to_dict()),result)
        resolver,records=self.pair()
        args=(resolver,records[0].record_id,resolver,records[1].record_id)
        self.assertEqual(compare_performance(*args,contract=self.policy).to_json(),
                         compare_performance(*args,contract=self.policy).to_json())

    def test_throughput_direction(self):
        d=_difference(*self.scalar_pair(M.COMPLETED_IMAGES_PER_SECOND,100.,150.,'images/s','throughput'))
        self.assertEqual((d.difference,d.percent_change,d.metric_effect),(50.,50.,'favorable'))

    def test_memory_increase_and_decrease(self):
        for metric,domain in ((M.RSS_CURRENT,'process_rss'),(M.MODEL_LOGICAL_BYTES,'model_logical_bytes'),
                              (M.ARTIFACT_BYTES,'serialized_artifact_bytes')):
            for value,effect in ((120.,'unfavorable'),(80.,'favorable')):
                d=_difference(*self.scalar_pair(metric,100.,value,'byte',domain))
                self.assertEqual(d.metric_effect,effect)

    def test_accuracy_percentage_points_are_separate(self):
        d=_difference(*self.scalar_pair(M.TEST_ACCURACY,.8,.9,'fraction','quality'))
        self.assertEqual((d.difference,d.percent_change,d.percentage_point_change),(.1,12.5,10.))

    def test_speed_quality_tradeoff_no_model_wide_claim(self):
        speed=self.run_pair().difference
        quality=_difference(*self.scalar_pair(M.TEST_MACRO_F1,.9,.8,'fraction','quality'))
        self.assertEqual((speed.metric_effect,quality.metric_effect),('favorable','unfavorable'))
        self.assertIn('single_metric_only',self.run_pair().limitations)
        self.assertIn('P29_independent_validation_incomplete',self.run_pair().limitations)

    def test_ineligible_pairs_have_no_numbers(self):
        right=payload('2',7.);right['provenance'].pop('comparison_capture')
        result=self.run_pair(right=right)
        self.assertEqual(result.eligibility.status,'INSUFFICIENT_EVIDENCE');self.assertIsNone(result.difference)
        right=payload('2',7.);right['provenance']['comparison_capture']['hardware_id']='5'*64
        result=self.run_pair(right=right)
        self.assertEqual(result.eligibility.status,'INCOMPATIBLE');self.assertIsNone(result.difference)

    def test_zero_baseline_and_unchanged(self):
        d=_difference(*self.scalar_pair(M.RSS_CURRENT,0.,10.,'byte','process_rss'))
        self.assertIsNone(d.percent_change);self.assertEqual(d.difference,10.)
        d=_difference(*self.scalar_pair(M.TEST_ACCURACY,0.,0.,'fraction','quality'))
        self.assertEqual((d.observed_direction,d.percentage_point_change),('unchanged',0.))
        # Native latency forbids zero even before comparison; retain that gate.
        with self.assertRaises(ValueError):self.run_pair(left=payload('1',0.))

    def test_invalid_numbers_and_overflow(self):
        for value in (-1.,float('nan'),float('inf'),True):
            with self.assertRaises(AdvisorError):
                _difference(*self.scalar_pair(M.RSS_CURRENT,value,10.,'byte','process_rss'))
        with self.assertRaises(AdvisorError):
            _difference(*self.scalar_pair(M.TEST_ACCURACY,.5,1.1,'fraction','quality'))
        with self.assertRaises(AdvisorError):
            _difference(*self.scalar_pair(M.RSS_CURRENT,5e-324,1e308,'byte','process_rss'))

    def test_wrong_scope_unit_direction_rejected(self):
        a,b=self.scalar_pair(M.FORWARD_P95,10.,7.,'ms','latency')
        with self.assertRaises(AdvisorError):_difference(a,replace(b,native_context_json=canonical({'measurement_scope':'other'})))
        with self.assertRaises(AdvisorError):_difference(a,replace(b,reference=replace(b.reference,unit='seconds')))
        with self.assertRaises(AdvisorError):replace(_difference(a,b),preferred_direction='higher')

    def test_reused_record_and_allowed_model_boundaries(self):
        r,records=self.pair(); result=compare_performance(r,records[0].record_id,r,records[0].record_id,contract=self.policy)
        self.assertIsNone(result.difference);self.assertEqual(result.eligibility.status,'INCOMPATIBLE')
        n=payload('2',7.);n['provenance']['model']['weights_sha256']='6'*64
        result=self.run_pair(right=n,policy=replace(self.policy,allowed_changes=('model',)))
        self.assertIsNotNone(result.difference);self.assertIn('allowed_change:model',result.warnings)

    def test_allowed_batch_and_precision_keep_scope_warnings(self):
        for field in ('batch','precision'):
            n=payload('2',7.)
            if field=='batch':n['provenance']['input'].update(shape=[2,1,2,2],batch_size=2,content_sha256='5'*64)
            else:
                n['provenance']['configuration']['precision']='fp64'
                n['provenance']['input']['dtype']='torch.float64'
            result=self.run_pair(right=n,policy=replace(self.policy,allowed_changes=(field,)))
            self.assertIsNotNone(result.difference)
            self.assertIn('allowed_change:'+field,result.warnings)
            self.assertIn('allowed_changes_limit_interpretation',result.limitations)

    def test_existing_native_records_not_repaired(self):
        data=json.loads((Path(__file__).parent/'fixtures/advisor-native-v1.json').read_text())['payloads']
        for feature,metric in (('P02',M.COMPLETED_IMAGES_PER_SECOND),('P10',M.RSS_CURRENT),('P11',M.MODEL_LOGICAL_BYTES)):
            a=copy.deepcopy(data[feature]);b=copy.deepcopy(a);b['result_id']='30000000-0000-0000-0000-000000000000'
            r,ids=self.native_pair(a,b,feature)
            result=compare_performance(r,ids[0],r,ids[1],contract=replace(self.policy,metric=metric))
            self.assertIsNone(result.difference)

    def native_pair(self, a, b, feature):
        records=[]
        with tempfile.TemporaryDirectory() as directory:
            for i,n in enumerate((a,b)):
                raw=canonical(n).encode();p=Path(directory)/f'{i}.json';p.write_bytes(raw)
                records.append(EvidenceRecord.from_json(p,feature_id=feature,expected_sha256=hashlib.sha256(raw).hexdigest()))
        report=EvidenceIndex(records).analyze(context=AnalysisContext(workload_id='fixture',task='classification',model={},dataset={}))
        return NativeMetricResolver(report),[v.record_id for v in records]

    def test_native_throughput_capture_integration(self):
        original=json.loads((Path(__file__).parent/'fixtures/advisor-native-v1.json').read_text())['payloads']['P02']
        pair=[]
        for i in range(2):
            n=copy.deepcopy(original);n['result_id']=str(i+1)+'0000000-0000-0000-0000-000000000000'
            for w in n['raw_evidence']['windows']:
                for k in ('start_ns','end_ns','admission_end_ns'):w[k]+=i*1000000000
            p=n['provenance']; common=payload(str(i+1),10.)['provenance']
            p['configuration']=common['configuration'];p['configuration'].update(n['protocol']['configuration'])
            p['input']['layout']='torch.strided'
            p['environment'].update(backend='fixture',torch_version='fixture',python_version='fixture',operating_system='fixture')
            p['comparison_capture']=common['comparison_capture']
            p['comparison_capture'].update(run_id=n['result_id'],sample_sha256=digest(n['raw_evidence']))
            pair.append(n)
        r,ids=self.native_pair(*pair,'P02')
        result=compare_performance(r,ids[0],r,ids[1],contract=replace(self.policy,metric=M.COMPLETED_IMAGES_PER_SECOND))
        self.assertEqual(result.eligibility.status,'COMPARABLE')
        self.assertEqual(result.difference.observed_direction,'unchanged')

    def capture_context(self, n, index):
        """Synthetic producer context, never appended to real artifacts."""
        p=n['provenance']; common=payload(str(index),10.)['provenance']
        for key in ('model','input','configuration'):
            p[key]=copy.deepcopy(common[key])
        p['configuration'].update(n.get('protocol',{}).get('configuration',{}))
        p.setdefault('environment',{}).update(common['environment'])
        p['comparison_capture']=common['comparison_capture']
        p['comparison_capture'].update(run_id=n.get('result_id',n.get('evaluation_id')),
            sample_sha256=digest(n.get('raw_evidence',n.get('raw',{}))))
        return n

    def test_native_memory_capture_integration(self):
        original=json.loads((Path(__file__).parent/'fixtures/advisor-native-v1.json').read_text())['payloads']['P10']
        pair=[]
        for i in range(2):
            n=copy.deepcopy(original);n['result_id']=str(i+1)+'0000000-0000-0000-0000-000000000000'
            offset=i*1000000000
            for k in ('start_ns','end_ns'):n['actual'][k]+=offset
            for row in n['raw_evidence']['observations']:
                for k in ('query_start_ns','query_end_ns','timestamp_ns'):row[k]+=offset
                for k in ('rss_bytes','lifetime_hwm_bytes'):row[k]+=i*1000
            for row in n['measurements'].values():row['value']+=i*1000
            pair.append(self.capture_context(n,i+1))
        r,ids=self.native_pair(*pair,'P10')
        result=compare_performance(r,ids[0],r,ids[1],contract=replace(self.policy,metric=M.RSS_CURRENT,min_samples=1))
        self.assertEqual(result.eligibility.status,'COMPARABLE')
        self.assertEqual((result.difference.difference,result.difference.metric_effect),(1000.,'unfavorable'))

    def test_native_accuracy_missing_conditions_not_repaired(self):
        from hyso import evaluate_predictions
        pair=[]
        for i,pred in enumerate(([0,1,1,1],[0,0,1,1])):
            # Only fixed-label confusion arithmetic; no inference/measurement.
            n=evaluate_predictions([0,0,1,1],pred,class_names=['a','b'],split='test').to_dict()
            pair.append(n)
        r,ids=self.native_pair(*pair,'P26')
        result=compare_performance(r,ids[0],r,ids[1],contract=replace(self.policy,metric=M.TEST_ACCURACY,min_samples=1))
        self.assertEqual(result.eligibility.status,'INSUFFICIENT_EVIDENCE')
        self.assertIsNone(result.difference)


if __name__=='__main__':unittest.main()
