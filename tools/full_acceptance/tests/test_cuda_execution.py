"""No subprocess, CUDA, or Windows Job creation."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from full_acceptance.cuda_execution import run_diagnostic


class DiagnosticIntegrationTests(unittest.TestCase):
    def manifest(self, root):
        return dict(output=str(root),hardware={'gpu_uuid':'GPU-abc'},
                    experiments=[dict(id='04.cuda',handler='inference_cuda',
                                      authorization={'approved':True},outer_seconds=60)],
                    source_snapshot={'root':str(root/'snapshot'), 'tree_sha256':'tree'})

    def test_rejects_wrong_manifest_path_without_supervisor(self):
        with TemporaryDirectory() as d:
            r=Path(d);m=self.manifest(r)
            with patch('full_acceptance.safety.supervise') as supervisor:
                with self.assertRaises(ValueError):
                    run_diagnostic(m,manifest_path=r/'wrong.json',stop_file=r/'stop-request.json')
                supervisor.assert_not_called()

    def test_stop_prevents_supervisor(self):
        with TemporaryDirectory() as d:
            r=Path(d);(r/'manifest.json').write_text('{}');(r/'stop-request.json').write_text('{}')
            with patch('full_acceptance.safety.supervise') as supervisor:
                with self.assertRaises(RuntimeError):
                    run_diagnostic(self.manifest(r),manifest_path=r/'manifest.json',stop_file=r/'stop-request.json')
                supervisor.assert_not_called()

    def test_never_upgrades_legacy_to_pass(self):
        with TemporaryDirectory() as d:
            r=Path(d);(r/'manifest.json').write_text('{}')
            def fake_supervise(m,e,*,operation,output,stop_file):
                return dict(id=e['id'],status='PASS',attempted=True,
                            result_file='result.json', evidence_directory='evidence',
                            supervisor=dict(experiment=e['id'],
                                            manifest_sha256=__import__('hashlib').sha256(b'{}').hexdigest(),
                                            exit_code=0,native_crash=False,reason=None,
                                            cleanup={'active_processes':0}))
            with patch('full_acceptance.cuda_execution.publish') as pub:
                result=run_diagnostic(self.manifest(r),manifest_path=r/'manifest.json',
                                      stop_file=r/'stop-request.json',supervise_fn=fake_supervise)
            self.assertEqual(result['status'],'UNSUPPORTED')
            self.assertFalse(result['attempted'])
            self.assertEqual(result['diagnostic_subtasks_attempted'],23)
            self.assertEqual(result['diagnostic_status'],'PASS')
            pub.assert_called_once()

    def test_isolated_failure_continues_but_no_legacy_pass(self):
        with TemporaryDirectory() as d:
            r=Path(d);(r/'manifest.json').write_text('{}');calls=[]
            h=__import__('hashlib').sha256(b'{}').hexdigest()
            def supervisor(m,e,*,operation,output,stop_file):
                calls.append(e['id'])
                failed=len(calls)==2
                return dict(id=e['id'],status='FAIL' if failed else 'PASS',
                     reason='ValueError' if failed else None,attempted=True,
                     result_file='result.json',evidence_directory='evidence',
                     supervisor=dict(experiment=e['id'],manifest_sha256=h,
                                     exit_code=1 if failed else 0,native_crash=False,
                                     reason=None,cleanup={'active_processes':0}))
            with patch('full_acceptance.cuda_execution.publish'):
                result=run_diagnostic(self.manifest(r),manifest_path=r/'manifest.json',
                                      stop_file=r/'stop-request.json',supervise_fn=supervisor)
            self.assertEqual(len(calls),23)
            self.assertEqual(result['status'],'FAIL')
            self.assertFalse(result['attempted'])

if __name__=='__main__':unittest.main()
