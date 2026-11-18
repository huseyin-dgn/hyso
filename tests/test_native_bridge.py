"""G2 bounded bridge tests. Fake native receipts are protocol doubles, not native proof."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyso.analysis.feature_plan import ModelContext, FeaturePlanRequest, plan_features
from hyso.analysis.native_bridge import (LiveCPUModel, TrustedCPUExecutionPolicy,
    NativeBridgeError, execute_scoped_cpu, _checked_capture, _observed_pair)


class NativeDouble:
    def __init__(self, payload): self.payload = payload
    def to_dict(self): return copy.deepcopy(self.payload)


def fake_receipt(feature, binding, policy, *, input_hash='a'*64, batch_count=None):
    if feature == 'P02':
        return NativeDouble(dict(feature_ids=['P02'], schema_version=1,
            provenance=dict(input=dict(shape=[2,1,2,2], dtype='torch.float32',content_sha256=input_hash),
                environment=dict(device='cpu')),
            protocol=dict(configuration=dict(batch_count=policy.batch_count if batch_count is None else batch_count,
                                              precision='fp32',device='cpu')),
            measurements=dict(value=100. if binding.model_id=='A' else 110.,unit='images/s',status='measured'),
            actual=dict(pending_batches=0,completed_batches=policy.batch_count)))
    return NativeDouble(dict(feature_ids=['P11'], schema_version=1,
        protocol=dict(byte_accounting='unique_tensor_object_logical'),
        provenance=dict(environment=dict(device='cpu')),
        measurements=dict(parameter_buffer_bytes=dict(value=100 if binding.model_id=='A' else 180,
                                                      unit='byte',status='derived'))))


def new_model(name):
    return LiveCPUModel(name, nn.Sequential(nn.Flatten(),nn.Linear(4,2)),torch.zeros(2,1,2,2))


def new_plan(mode='selected',features=('P02','P11'),models=('A',)):
    contexts=tuple(ModelContext(x, input_contract=True) for x in models)
    if mode=='single':return plan_features(FeaturePlanRequest(mode, contexts, features=(features[0],)))
    if mode=='full':return plan_features(FeaturePlanRequest(mode, contexts))
    return plan_features(FeaturePlanRequest(mode, contexts, features=features))


class G2BridgeContract(unittest.TestCase):
    policy=TrustedCPUExecutionPolicy(explicit_trusted_inprocess_consent=True)

    def run_mocked(self, plan, bindings, features=('P02','P11'), **kwargs):
        with patch('hyso.analysis.native_bridge._invoke_native', side_effect=fake_receipt) as native:
            result=execute_scoped_cpu(plan,bindings,authorized_features=features,
                        policy=kwargs.get('policy', self.policy))
        self.assertEqual(native.call_count, len(features)*len(bindings))
        return result

    def test_default_execution_refuses_consent(self):
        with self.assertRaisesRegex(NativeBridgeError,'requires_explicit_consent'):
            execute_scoped_cpu(new_plan(),(new_model('A'),),authorized_features=('P02',))

    def test_selected_p02_p11_calls_real_producer_contract_positions(self):
        result=self.run_mocked(new_plan(),(new_model('A'),))
        self.assertEqual(result.executed_feature_ids,('P02','P11'))
        self.assertEqual([c.feature_id for c in result.captures],['P02','P11'])
        self.assertEqual(result.acceptance,'NOT_EVALUATED')
        self.assertEqual(result.scope,'trusted_inprocess_cpu;not_isolated_no_hard_runtime_memory_limit')
        self.assertEqual(result.comparisons,())
        self.assertEqual(result.omitted_features,())

    def test_single_feature_footprint_does_not_require_input(self):
        plan=new_plan(mode='single',features=('P11',))
        a=LiveCPUModel('A',nn.Linear(2,3))
        result=self.run_mocked(plan,(a,),('P11',))
        self.assertEqual(len(result.captures),1)

    def test_two_models_have_scoped_native_observed_diffs(self):
        result=self.run_mocked(new_plan(models=('A','B')),(new_model('A'),new_model('B')))
        self.assertEqual(len(result.captures),4)
        self.assertEqual([x.status for x in result.comparisons],['OBSERVED_DESCRIPTIVE_ONLY']*2)
        self.assertEqual({x.feature_id:x.observed_difference for x in result.comparisons},
                         {'P02':10.,'P11':80})
        self.assertTrue(all('winner' not in x.reason for x in result.comparisons))

    def test_mismatched_native_input_blocks_comparison(self):
        a=_checked_capture('A','P02',fake_receipt('P02',new_model('A'),self.policy))
        b=_checked_capture('B','P02',fake_receipt('P02',new_model('B'),self.policy,input_hash='b'*64))
        c=_observed_pair('P02',a,b)
        self.assertEqual(c.status,'BLOCKED')
        self.assertIsNone(c.observed_difference)
        self.assertIsNone(c.baseline)

    def test_mismatched_native_protocol_blocks_comparison(self):
        a=_checked_capture('A','P02',fake_receipt('P02',new_model('A'),self.policy))
        b=_checked_capture('B','P02',fake_receipt('P02',new_model('B'),self.policy,batch_count=3))
        self.assertEqual(_observed_pair('P02',a,b).status,'BLOCKED')

    def test_full_plan_does_not_execute_other_33(self):
        result=self.run_mocked(new_plan(mode='full'),(new_model('A'),))
        self.assertEqual(len(result.requested_feature_ids),35)
        self.assertEqual(len(result.omitted_features),33)
        self.assertTrue(all(f in result.requested_feature_ids for f,r in result.omitted_features))
        self.assertEqual(result.acceptance,'NOT_EVALUATED')

    def test_selected_unapproved_feature_explicitly_not_run(self):
        result=self.run_mocked(new_plan(),(new_model('A'),),('P11',))
        self.assertEqual(result.omitted_features,(('P02','not_authorized'),))

    def test_unsupported_and_not_selected_rejected_pre_execution(self):
        plan=new_plan(mode='full')
        for ids in (('P31',),('P02','P02'),('P36',)):
            with self.subTest(ids=ids), self.assertRaises(NativeBridgeError):
                execute_scoped_cpu(plan,(new_model('A'),),authorized_features=ids,policy=self.policy)

    def test_missing_input_blocks_before_any_native_run(self):
        a=LiveCPUModel('A',nn.Linear(4,2))
        with patch('hyso.analysis.native_bridge._invoke_native') as native:
            with self.assertRaises(NativeBridgeError):
                execute_scoped_cpu(new_plan(),(a,),authorized_features=('P02','P11'),policy=self.policy)
            native.assert_not_called()

    def test_preflight_rejects_gpu_and_double_size(self):
        plan=new_plan()
        b=new_model('A')
        b=LiveCPUModel('A',b.model,torch.zeros(2,1,2,2,dtype=torch.float64))
        with self.assertRaisesRegex(NativeBridgeError,'bounded_cpu_fp32'):
            execute_scoped_cpu(plan,(b,),authorized_features=('P02',),policy=self.policy)
        p=TrustedCPUExecutionPolicy(explicit_trusted_inprocess_consent=True,max_input_bytes=4)
        with self.assertRaisesRegex(NativeBridgeError,'bounded_cpu_fp32'):
            execute_scoped_cpu(plan,(new_model('A'),),authorized_features=('P02',),policy=p)

    def test_preflight_rejects_untrusted_g1_status(self):
        plan=plan_features(FeaturePlanRequest('single',(ModelContext('A',input_contract=False),),features=('P02',)))
        with patch('hyso.analysis.native_bridge._invoke_native') as native:
            with self.assertRaisesRegex(NativeBridgeError,'G1_prerequisites_not_proposed'):
                execute_scoped_cpu(plan,(new_model('A'),),authorized_features=('P02',),policy=self.policy)
            native.assert_not_called()

    def test_same_model_object_rejected_for_two_identities(self):
        m=new_model('A')
        with self.assertRaisesRegex(NativeBridgeError,'shared_model_object'):
            execute_scoped_cpu(new_plan(models=('A','B')),
                (m,LiveCPUModel('B',m.model,m.input_tensor)),
                authorized_features=('P11',),policy=self.policy)

    def test_invalid_native_throughput_not_a_success(self):
        source=fake_receipt('P02',new_model('A'),self.policy)
        source.payload['actual']['pending_batches']=1
        with self.assertRaisesRegex(NativeBridgeError,'invalid_native_throughput'):
            _checked_capture('A','P02',source)

    def test_invalid_native_footprint_not_a_success(self):
        source=fake_receipt('P11',new_model('A'),self.policy)
        source.payload['measurements']['parameter_buffer_bytes']['value']=-1
        with self.assertRaisesRegex(NativeBridgeError,'invalid_native_footprint'):
            _checked_capture('A','P11',source)

    def test_sha_and_json_copy_are_stable(self):
        row=_checked_capture('A','P11',fake_receipt('P11',new_model('A'),self.policy))
        payload=row.to_dict();payload['native_result']['feature_ids'].clear()
        self.assertEqual(row.to_dict()['native_result']['feature_ids'],['P11'])
        self.assertEqual(len(row.source_sha256),64)
        self.assertEqual(row.to_dict()['status'],'MEASURED_NATIVE_SCOPE')

    def test_bounded_policy_rejects_fakes_and_unbounded_counts(self):
        for kwargs in ({'batch_count':1000000},{'repeats':0},{'warmup_iterations':-1},
                       {'max_input_bytes':64*1024*1024},{'explicit_trusted_inprocess_consent':1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(NativeBridgeError):
                TrustedCPUExecutionPolicy(**kwargs)

    def test_does_not_modify_original_weights_in_preflight_and_mock_bridge(self):
        b=new_model('A');w=b.model[1].weight.detach().clone();mode=b.model.training
        self.run_mocked(new_plan(),(b,))
        self.assertTrue(torch.equal(b.model[1].weight,w))
        self.assertEqual(b.model.training,mode)


if __name__=='__main__': unittest.main()
