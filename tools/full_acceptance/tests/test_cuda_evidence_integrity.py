import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from full_acceptance.cuda_evidence_integrity import verify_native_pass

SHA = 'a' * 64
TASK = '04.cuda.A.latency'

class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        artifact=self.root/'latency.json';artifact.write_text('measured',encoding='utf-8')
        evidence=[dict(path=str(artifact),bytes=artifact.stat().st_size,
                       sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())]
        self.write('result.json',dict(id=TASK,status='PASS',manifest_sha256=SHA,evidence=evidence))
        self.write('supervisor-receipt.json',dict(experiment=TASK,manifest_sha256=SHA,
                  operation='inference_cuda_subtask',exit_code=0,reason=None,
                  native_crash=False,owned_job_only=True,cleanup=dict(active_processes=0)))
        self.write('ready.json',dict(manifest_sha256=SHA,gate=dict(exact_job_membership=True,kill_on_close=True)))
        self.observed=dict(id=TASK,status='PASS',evidence_directory=str(self.root),result_file=str(self.root/'result.json'))
    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value),encoding='utf-8')
    def test_valid_native_pass_files(self):
        self.assertTrue(verify_native_pass(TASK,SHA,self.observed,self.root)['evidence_integrity'])
    def test_corrupt_artifact_rejected(self):
        (self.root/'latency.json').write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError): verify_native_pass(TASK,SHA,self.observed,self.root)
    def test_fake_job_gate_rejected(self):
        self.write('ready.json',dict(manifest_sha256=SHA,gate=dict(exact_job_membership=False,kill_on_close=True)))
        with self.assertRaises(ValueError): verify_native_pass(TASK,SHA,self.observed,self.root)
    def test_unsafe_result_path_rejected(self):
        self.observed['result_file']=str(self.root/'not-result.json')
        with self.assertRaises(ValueError): verify_native_pass(TASK,SHA,self.observed,self.root)
    def test_missing_receipt_rejected(self):
        (self.root/'supervisor-receipt.json').unlink()
        with self.assertRaises(ValueError): verify_native_pass(TASK,SHA,self.observed,self.root)
    def test_wrong_manifest_rejected(self):
        with self.assertRaises(ValueError): verify_native_pass(TASK,'b'*64,self.observed,self.root)
    def test_empty_measurement_rejected(self):
        self.write('result.json',dict(id=TASK,status='PASS',manifest_sha256=SHA,evidence=[]))
        with self.assertRaises(ValueError): verify_native_pass(TASK,SHA,self.observed,self.root)
