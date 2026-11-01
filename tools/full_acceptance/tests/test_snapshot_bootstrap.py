"""Regression for ordinary terminal invocation mutating a sealed snapshot."""
from pathlib import Path
import importlib.util,json,marshal,os,struct,subprocess,sys,tempfile,unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from full_acceptance import core,safety
from test_infrastructure import Fixture

PYTHON=core.DEFAULT_PYTHON

class SnapshotBootstrapTests(unittest.TestCase):
    def invoke_without_cache_flags(self,script):
        env=dict(os.environ)
        env.pop('PYTHONDONTWRITEBYTECODE',None)
        # -I/-S isolates installed startup customization. Deliberately no -B.
        return subprocess.run([str(PYTHON),'-I','-S','-c',script],env=env,capture_output=True,text=True,timeout=30)

    def test_cache_enabled_coordinator_does_not_mutate_loaded_snapshot(self):
        with tempfile.TemporaryDirectory(prefix='hyso-snapshot-load-') as tmp:
            root=Path(tmp).resolve();folder=root/'source/hyso/performance';folder.mkdir(parents=True)
            coordinator=root/'coordinator/tools/full_acceptance';coordinator.mkdir(parents=True)
            for name in ('__init__.py','core.py','registry.py','safety.py'):
                (coordinator/name).write_bytes((core.ROOT/'tools/full_acceptance'/name).read_bytes())
            for name in ('_training_process.py','_stream_resources.py'):
                (folder/name).write_bytes((core.ROOT/'hyso/performance'/name).read_bytes())
            frozen=root/'source'
            before={p.relative_to(frozen).as_posix():core.sha(p) for p in frozen.rglob('*') if p.is_file()}
            script=("import sys,json;sys.path.insert(0,"+repr(str(coordinator.parent))+");"
                "from full_acceptance.safety import load_file;"
                "assert sys.dont_write_bytecode is False;"
                "a=load_file("+repr(str(folder/'_training_process.py'))+",'job_probe');"
                "b=load_file("+repr(str(folder/'_stream_resources.py'))+",'memory_probe');"
                "assert callable(a._Job) and callable(b._StreamResources);"
                "print(json.dumps(dict(bytecode_enabled=not sys.dont_write_bytecode,helpers_loaded=2)))")
            p=self.invoke_without_cache_flags(script)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertTrue(json.loads(p.stdout)['bytecode_enabled'])
            after={p.relative_to(frozen).as_posix():core.sha(p) for p in frozen.rglob('*') if p.is_file()}
            self.assertEqual(before,after,'cache-enabled coordinator wrote files into source snapshot')

    def test_explicit_source_loader_never_uses_preexisting_bytecode(self):
        with tempfile.TemporaryDirectory(prefix='hyso-snapshot-cache-') as tmp:
            path=Path(tmp).resolve()/'trusted.py';path.write_text("VALUE='SOURCE'\n",encoding='utf-8')
            cached=Path(importlib.util.cache_from_source(str(path)));cached.parent.mkdir()
            info=path.stat()
            harmless_cache=compile("VALUE='CACHE'\n",str(path),'exec')
            cached.write_bytes(importlib.util.MAGIC_NUMBER+struct.pack('<III',0,int(info.st_mtime),info.st_size)+marshal.dumps(harmless_cache))
            before=core.sha(cached)
            loaded=safety.load_file(path,'explicit_source_probe')
            self.assertEqual(loaded.VALUE,'SOURCE','loader used cache rather than checked source bytes')
            self.assertEqual(core.sha(cached),before,'loader must not rewrite existing cache evidence')

    def test_entrypoint_disables_bytecode_before_first_package_import(self):
        with tempfile.TemporaryDirectory(prefix='hyso-snapshot-entry-') as tmp:
            tools=Path(tmp).resolve()/'tools';tools.mkdir();entry=tools/'run_full_acceptance.py'
            entry.write_bytes((core.ROOT/'tools/run_full_acceptance.py').read_bytes())
            package=tools/'full_acceptance';package.mkdir();(package/'__init__.py').write_text('')
            (package/'cli.py').write_text("import sys,json\ndef main():\n print(json.dumps(dict(bytecode_disabled=sys.dont_write_bytecode)))\n return 0\n")
            env=dict(os.environ);env.pop('PYTHONDONTWRITEBYTECODE',None)
            p=subprocess.run([str(PYTHON),'-I','-S',str(entry)],env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertTrue(json.loads(p.stdout)['bytecode_disabled'])
            self.assertEqual(list(tools.rglob('*.pyc')),[])

class SnapshotCacheRejectionTests(Fixture):
    def test_unpinned_pyc_still_rejected_by_exact_file_set(self):
        p=self.snapshot/'tools/__pycache__/run_full_acceptance.cpython-310.pyc';p.parent.mkdir();p.write_bytes(b'unpinned cache fixture')
        before=core.sha(p)
        with self.assertRaisesRegex(ValueError,'cached/unpinned'):
            core.verify(self.path,expected=self.hash,assets=False)
        self.assertEqual(core.sha(p),before)

class SnapshotExecutionContractTests(Fixture):
    def test_execute_contract_keeps_unapproved_native_stages_closed(self):
        from full_acceptance import cli
        from unittest.mock import patch
        import types
        self.old_matrix();calls=[]
        def safe_double(manifest,experiment,**kwargs):
            calls.append(experiment['id'])
            return dict(status='PASS',attempted=True,metrics=[],criteria={},supervisor=dict(cleanup=dict(active_processes=0)))
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash)
        with patch.object(cli,'verify',return_value=self.m),patch.object(cli,'preserve',return_value={}),patch('full_acceptance.safety.supervise',side_effect=safe_double):
            self.assertEqual(cli.execute(args),2)  # Explicitly incomplete, not native acceptance.
            self.assertEqual(calls,['01.preflight','02.regression','03.cpu','12.replay'])
            rows={r['id']:r for r in core.read(self.out/'stage-results.json')['experiments']}
            for ident in ('04.cuda','05.training','08.camera','10.stability'):
                self.assertEqual(rows[ident]['status'],'NOT_AUTHORIZED')
                self.assertFalse(rows[ident]['attempted'])
            with self.assertRaises(FileExistsError):cli.execute(args)
            self.assertEqual(len(calls),4,'second execution must not retry any worker')

    def test_snapshot_verification_failure_prevents_START_and_worker(self):
        from full_acceptance import cli
        from unittest.mock import patch
        import types
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash)
        with patch.object(cli,'verify',side_effect=ValueError('unexpected snapshot file')),patch('full_acceptance.safety.supervise') as run:
            with self.assertRaisesRegex(ValueError,'snapshot'):cli.execute(args)
            run.assert_not_called()
        self.assertFalse((self.out/'STARTED.json').exists())

if __name__=='__main__':unittest.main()
