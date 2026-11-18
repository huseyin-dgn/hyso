"""G3 real opt-in Windows native worker tests on small bundled trusted CPU fixture.

Runs real Torch P02/P11 in separately limited processes. No real user checkpoint
or CUDA is accessed. Historical native-crash risk is not declared resolved.
"""
import hashlib
import os
from pathlib import Path
import unittest

from hyso.analysis.feature_plan import FeaturePlanRequest, ModelContext
from hyso.analysis.isolated_bridge import (TrustedFactoryBinding,
    IsolatedCPUExecutionPolicy, execute_isolated_cpu)


@unittest.skipUnless(os.name == 'nt', 'Windows Job Object native verification only')
class G3WindowsRealCPU(unittest.TestCase):
    policy = IsolatedCPUExecutionPolicy(explicit_authorization=True,
        timeout_seconds=60, job_memory_limit_bytes=2*1024**3)

    def binding(self, model_id, width):
        p=Path(__file__).with_name('g3_trusted_factory.py').resolve()
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        return TrustedFactoryBinding(model_id,str(p),sha,kwargs=dict(width=width))

    def test_single_worker_native_P11(self):
        req=FeaturePlanRequest('single',(ModelContext('A',trusted_factory=True),),features=('P11',))
        got=execute_isolated_cpu(req,(self.binding('A',2),),authorized_features=('P11',),policy=self.policy)
        self.assertEqual(got.status,'COMPLETED',got.reason)
        self.assertTrue(got.worker_receipt['assigned_before_resume'])
        self.assertEqual(got.worker_receipt['active_owned_processes'],0)
        self.assertEqual(got.native_bridge_report['captures'][0]['native_result']['feature_ids'],['P11'])
        self.assertEqual(got.acceptance,'NOT_EVALUATED')

    def test_two_models_P02_P11_worker_comparison(self):
        req=FeaturePlanRequest('selected',(
            ModelContext('A',trusted_factory=True,input_contract=True),
            ModelContext('B',trusted_factory=True,input_contract=True)),
            features=('P02','P11'))
        got=execute_isolated_cpu(req,(self.binding('A',2),self.binding('B',4)),
                                 authorized_features=('P02','P11'),policy=self.policy)
        self.assertEqual(got.status,'COMPLETED',got.reason)
        self.assertEqual(len(got.native_bridge_report['captures']),4)
        self.assertEqual(len(got.native_bridge_report['comparisons']),2)
        self.assertEqual(got.worker_receipt['active_owned_processes'],0)

if __name__=='__main__':unittest.main()
