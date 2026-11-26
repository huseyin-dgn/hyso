"""Stdlib-only replay/negative guards. No Torch, model execution or GPU launch."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO),str(REPO/'tools/final_acceptance')]
from common import read_json,file_hash
from tools.cuda_acceptance import native_identity as identity
from tools.cuda_acceptance.feature_evidence import verify_feature,SafetyViolation
from tools.cuda_acceptance.p25_evidence import verify_amp,native_hash
from tools.cuda_acceptance.p25_p35_review import equivalent

BASE=REPO/'outputs/cuda-functional-acceptance-20261126'
FIX=BASE/'p25-p35-remediation'


def reseal(d):
    d['source_sha256']=native_hash(d['source_p19'])
    d['source_p20']['source_p19']=copy.deepcopy(d['source_p19'])
    d['source_p20']['source_sha256']=d['source_sha256']
    d['source_p20_sha256']=native_hash(d['source_p20'])
    d['sha256']=native_hash({k:v for k,v in d.items() if k!='sha256'})
    return d


def one_attempt(d,index):
    d=copy.deepcopy(d);row=d['observations'][index];attempt_id=row['attempt_id'];window=row['window_id']
    d['observations']=[row];s=d['source_p19']
    s['observations']=[r for r in s['observations'] if r['window_id']==window]
    s['counters']['before']['attempts']=attempt_id-1;s['counters']['before']['microbatches']=row['microbatch_ids'][0]-1
    s['counters']['after'].update(attempts=attempt_id,microbatches=row['microbatch_ids'][-1],applied=int(row['applied']))
    s['epochs'][0].update(attempts=1,microbatches=1,windows=1,samples=row['samples'],applied=int(row['applied']),skipped=int(row['skipped']))
    d['source_p20']['observations']=[r for r in d['source_p20']['observations'] if r['attempt_id'] in (None,attempt_id)]
    d['actual'].update(windows=1,attempts=1)
    d['summary']=dict(attempts=1,applied=int(row['applied']),skipped=int(row['skipped']),recovered_after_skip_attempt_ids=[],
        recovery_scope='this_segment_only',scale_changes={k:int(row['scale_change']==k) for k in ('increased','unchanged','decreased')})
    return reseal(d)


class FinalTwoTests(unittest.TestCase):
    def setUp(self):
        self.p25=read_json(BASE/'attempts/6e4ce6cb9cbc4678855632a71f4fa3b8/P25.json')['data']
        self.loaded=read_json(BASE/'attempts/8fda2515f46f4ca88eed4310f5238701/ort-loaded-runtime.json')
        self.p35=read_json(BASE/'attempts/8fda2515f46f4ca88eed4310f5238701/P35-native-raw.json')
        self.original=read_json(FIX/'before-plan.json')['feature_tools']['ort']
        self.dependency=read_json(FIX/'driver-package-verification.json')
        self.expected=copy.deepcopy(self.original)
        backend=self.dependency['files']['backend']
        self.expected['allowed_binaries'][backend['path']]=backend['sha256']

    def test_real_amp_skip_then_finite_update_preserves_nan_evidence(self):
        before=copy.deepcopy(self.p25)
        verify_feature('P25',self.p25,{})
        self.assertEqual(self.p25,before)
        first,last=self.p25['observations']
        self.assertEqual(first['preclip']['summary']['nan_tensor_count'],4)
        self.assertIsNone(first['preclip']['summary']['global_l2']['value'])
        self.assertFalse(first['optimizer_executed']);self.assertTrue(last['optimizer_executed'])
        self.assertEqual(self.p25['summary']['recovered_after_skip_attempt_ids'],[2])
        self.assertTrue(all(r['overflow_cause']=='unknown' for r in self.p25['observations']))

    def test_finite_update_only_does_not_invent_recovery(self):
        d=one_attempt(self.p25,1)
        verify_feature('P25',d,{})
        self.assertEqual(d['summary']['recovered_after_skip_attempt_ids'],[])

    def test_zero_successful_updates_is_rejected(self):
        d=one_attempt(self.p25,0)
        with self.assertRaisesRegex(ValueError,'successful finite optimizer update'):verify_amp(d)
        with self.assertRaisesRegex(ValueError,'real optimizer updates'):verify_feature('P25',d,{})

    def test_resealed_fake_amp_claims_are_rejected(self):
        def stage(d,attempt):return next(r for r in d['source_p20']['observations'] if r['stage']=='optimizer_execution' and r['attempt_id']==attempt)
        mutations={
            'skip_claims_execution':lambda d:d['observations'][0].update(optimizer_executed=True),
            'fake_native_stage':lambda d:stage(d,1).update(status='measured'),
            'no_scaler_step':lambda d:next(r for r in d['source_p20']['observations'] if r['stage']=='scaler_step_attempt' and r['attempt_id']==1).update(status='not_executed'),
            'no_scaler_update':lambda d:next(r for r in d['source_p20']['observations'] if r['stage']=='scaler_update' and r['attempt_id']==1).update(status='not_executed'),
            'successful_update_without_hook':lambda d:stage(d,2).update(status='not_executed'),
            'wrong_counter':lambda d:d['source_p19']['counters']['after'].update(applied=2),
            'wrong_P19_skip':lambda d:next(r for r in d['source_p19']['observations'] if r['kind']=='attempt' and r['attempt_id']==1).update(applied=True),
            'wrong_recovery_id':lambda d:d['summary'].update(recovered_after_skip_attempt_ids=[1]),
            'invented_recovery_id':lambda d:d['summary'].update(recovered_after_skip_attempt_ids=[99]),
            'nonfinite_zero_fill':lambda d:d['observations'][0]['preclip']['tensors'][0]['gradient'].update(global_l2=0.),
            'hidden_nan_flag':lambda d:d['observations'][0]['preclip']['tensors'][0]['gradient'].update(has_nan=False),
            'invented_finite_norm':lambda d:d['observations'][1]['preclip']['summary']['global_l2'].update(value=0.),
            'known_overflow_cause':lambda d:d['observations'][0].update(overflow_cause='loss_scale_too_high'),
            'no_scale_backoff':lambda d:d['observations'][0].update(scale_after=65536.,scale_change='unchanged'),
            'scale_discontinuity':lambda d:d['observations'][1].update(scale_before=16384.),
            'foreign_stage_owner':lambda d:stage(d,2).update(window_id=1),
            'foreign_microbatch':lambda d:d['observations'][1].update(microbatch_ids=[1]),
            'not_native_AMP':lambda d:d['source_p19']['provenance']['training'].update(amp=False),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                d=copy.deepcopy(self.p25);mutate(d);reseal(d)
                with self.assertRaises((ValueError,KeyError)):verify_feature('P25',d,{})

    def test_unbound_gradient_or_source_hash_is_rejected(self):
        for field in ('sha256','source_sha256','source_p20_sha256'):
            with self.subTest(field=field):
                d=copy.deepcopy(self.p25);d[field]='0'*64
                with self.assertRaises(ValueError):verify_feature('P25',d,{})

    def test_missing_dependency_is_not_automatically_allowed(self):
        before=copy.deepcopy(self.original)
        with self.assertRaisesRegex(identity.DLLIntegrityError,'not admitted'):
            identity.validate_ort_identity(self.loaded,self.p35,self.original)
        self.assertEqual(self.original,before)
        identity.validate_ort_identity(self.loaded,self.p35,self.expected)

    def test_pre_admission_dependency_comes_from_registered_signed_package(self):
        value=identity.driver_dependency();files=value['files']
        self.assertEqual(files['backend']['path'],self.dependency['files']['backend']['path'])
        self.assertEqual(files['backend']['sha256'],self.loaded['loaded_binaries'][files['backend']['path']])
        self.assertEqual(files['wrapper']['identity'],files['loader']['identity'])
        self.assertNotEqual(files['wrapper']['identity'],files['backend']['identity'])
        self.assertTrue(all(row['signature']=='Valid' for row in files.values()))
        with patch.object(identity,'loaded_binaries',side_effect=AssertionError('prepare must not inspect loaded modules')),patch.object(identity,'driver_dependency',return_value=value):
            inventory=identity.ort_inventory(self.original['site'],REPO.parent/'unused')
        self.assertIn(files['backend']['path'],inventory['allowed_binaries'])

    def test_unsigned_or_mixed_version_driver_is_rejected(self):
        metadata=[self.dependency['files'][role] for role in ('wrapper','loader','backend','service_driver')]
        for field,value in (('signature','NotSigned'),('version','0.0.0.0'),('company','Untrusted Publisher')):
            with self.subTest(field=field):
                rows=copy.deepcopy(metadata);rows[2][field]=value
                with patch.object(identity.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(rows))):
                    with self.assertRaises(identity.DLLIntegrityError):identity.driver_dependency()

    def test_windows_case_variants_keep_hash_requirement(self):
        loaded=copy.deepcopy(self.loaded)
        loaded['loaded_binaries']={p.upper().replace('\\','/'):sha for p,sha in loaded['loaded_binaries'].items()}
        identity.validate_ort_identity(loaded,self.p35,self.expected)

    def test_true_hash_or_unadmitted_path_is_a_safety_failure(self):
        key=next(iter(self.loaded['loaded_binaries']))
        for action in ('hash','path','duplicate'):
            with self.subTest(action=action):
                loaded=copy.deepcopy(self.loaded)
                if action=='hash':loaded['loaded_binaries'][key]='0'*64
                elif action=='path':loaded['loaded_binaries']['C:/unadmitted/nvcuda64.dll']=loaded['loaded_binaries'].pop(key)
                else:loaded['loaded_binaries'][key.upper()]=loaded['loaded_binaries'][key]
                with self.assertRaises(identity.DLLIntegrityError) as error:identity.validate_ort_identity(loaded,self.p35,self.expected)
                self.assertIsInstance(error.exception,SafetyViolation)

    def test_cuda_nodes_runtime_and_cleanup_remain_required(self):
        value=dict(result=self.p35,loaded_runtime=self.loaded,planned_runtime=self.expected)
        verify_feature('P35',value,{})
        mutations={
            'listed_only':lambda d:d['result']['analysis'].update(events=[]),
            'wrong_status':lambda d:d['result']['analysis'].update(status='unknown'),
            'CPU_fallback':lambda d:d['result']['analysis']['events'][0].update(provider='CPUExecutionProvider'),
            'cleanup':lambda d:d['result']['evidence']['controller']['cleanup'].update(active_processes=1),
            'runtime':lambda d:d['loaded_runtime'].update(version='unknown'),
            'worker':lambda d:d['loaded_runtime'].update(pid=1),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                d=copy.deepcopy(value);mutate(d)
                with self.assertRaises((ValueError,SafetyViolation)):verify_feature('P35',d,{})

    def test_accepted_feature_paths_remain_unchanged(self):
        for name in ('feature_evidence.py','six_group_review.py'):
            before=(FIX/'before-source/tools/cuda_acceptance'/name).read_text(encoding='utf-8')
            current=(REPO/'tools/cuda_acceptance'/name).read_text(encoding='utf-8')
            self.assertTrue(equivalent(before,current))
        text=(REPO/'tools/cuda_acceptance/feature_evidence.py').read_text(encoding='utf-8')
        self.assertFalse(equivalent(text,text.replace("pending_batches']==0","pending_batches']==1")))


if __name__=='__main__':
    result=unittest.main(exit=False)
    if 'torch' in sys.modules or 'onnxruntime' in sys.modules:raise RuntimeError('offline guards imported a GPU runtime')
    raise SystemExit(not result.result.wasSuccessful())
