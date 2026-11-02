"""G2 actual native CPU PyTorch integration, requires installed HYSo source package.

These are real P02/P11 producer invocations (not mocked) on small trusted CPU
models. They do not establish CUDA, arbitrary-model safety or 35-feature READY.
"""
import unittest

import torch
from torch import nn

from hyso.analysis.feature_plan import ModelContext, FeaturePlanRequest, plan_features
from hyso.analysis.native_bridge import (LiveCPUModel, TrustedCPUExecutionPolicy,
                                         execute_scoped_cpu)


class G2RealNativeCPU(unittest.TestCase):
    policy = TrustedCPUExecutionPolicy(explicit_trusted_inprocess_consent=True,
                                       warmup_iterations=1,batch_count=2,repeats=1)

    @staticmethod
    def model():
        return nn.Sequential(nn.Flatten(), nn.Linear(4,8), nn.ReLU(), nn.Linear(8,2))

    def test_native_p11_measure_footprint_preserves_model(self):
        model=self.model(); mode=model.training
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        plan=plan_features(FeaturePlanRequest('single', (ModelContext('A'),), features=('P11',)))
        report=execute_scoped_cpu(plan, (LiveCPUModel('A',model),),
                                  authorized_features=('P11',),policy=self.policy)
        payload=report.captures[0].to_dict()['native_result']
        self.assertEqual(payload['feature_ids'],['P11'])
        self.assertEqual(payload['measurements']['parameter_buffer_bytes']['status'],'derived')
        self.assertGreater(payload['measurements']['parameter_buffer_bytes']['value'],0)
        self.assertEqual(model.training,mode)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_native_p02_two_distinct_models_comparison_and_integrity(self):
        models=[self.model(),self.model()]
        input_a=torch.ones(2,1,2,2)
        input_b=input_a.clone()
        snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
        modes=[m.training for m in models]
        plan=plan_features(FeaturePlanRequest('single',
            (ModelContext('A',input_contract=True),ModelContext('B',input_contract=True)),
            features=('P02',)))
        report=execute_scoped_cpu(plan,(LiveCPUModel('A',models[0],input_a),
            LiveCPUModel('B',models[1],input_b)),authorized_features=('P02',),policy=self.policy)
        self.assertEqual(len(report.captures),2)
        self.assertEqual(len(report.comparisons),1)
        self.assertEqual(report.comparisons[0].status,'OBSERVED_DESCRIPTIVE_ONLY')
        for m,mode,saved,cap in zip(models,modes,snapshots,report.captures):
            self.assertEqual(m.training,mode)
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items()))
            raw=cap.to_dict()['native_result']
            self.assertEqual(raw['measurements']['unit'],'images/s')
            self.assertGreater(raw['measurements']['value'],0)
            self.assertEqual(raw['actual']['pending_batches'],0)


if __name__=='__main__': unittest.main()
