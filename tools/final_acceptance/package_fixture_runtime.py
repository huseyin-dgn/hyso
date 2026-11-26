"""Copied beside the external installed-package harness, never imported from checkout."""
import hashlib
import json
import os
import sys
import time
import traceback
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

_FSYNC=os.fsync
_SHA256=hashlib.sha256
_ENCODE=json.JSONEncoder(sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode
_CLOCK=time.perf_counter
_UTC_NOW=datetime.now


def attach(namespace):
    cls, out, args = namespace['InstalledCPUWorkflow'], namespace['OUT'], namespace['args']
    torch, api, advisor = namespace['torch'], namespace['api'], namespace['advisor']
    expected = json.loads(Path(args.resume_state).read_text()) if args.resume_state else {}
    terminals = expected.get('terminal_cases', {})
    bindings = expected.get('artifact_bindings', {})
    for name, sha in bindings.items():
        path = out / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError('EVIDENCE_INVALID: lost/changed package fixture ' + name)
    # Interrupted files are moved, not overwritten; only fully checkpointed artifacts stay active.
    if args.resume_state:
        partial = out / ('partial-' + uuid.uuid4().hex)
        for path in list(out.iterdir()):
            if path.is_file() and path.name not in bindings and not path.name.startswith('package-events-') and path.name != '.owned-package.json':
                partial.mkdir(exist_ok=True)
                path.rename(partial / path.name)
    marker = out / '.owned-package.json'
    campaign = os.environ['HYSO_FINAL_CAMPAIGN_ID']
    if marker.exists() and json.loads(marker.read_text())['campaign_id'] != campaign:
        raise RuntimeError('package fixture ownership mismatch')
    if not marker.exists():
        marker.write_text(json.dumps(dict(campaign_id=campaign)), encoding='utf-8')
    passed = {name for name, status in terminals.items() if status == 'PASS'}
    if 'test_10_public_Builder_Trainer_CPU_episode' in passed:
        from hyso import TrainingResult
        cls.trained = TrainingResult((out / 'api-training.json').read_text(encoding='utf-8'))
        model = namespace['builder']().build()
        model.load_state_dict(torch.load(out / 'owned-api-state.pt', map_location='cpu', weights_only=True))
        cls.session = namespace['SimpleNamespace'](evaluation_model=model)
    if 'test_20_trained_model_to_Analyzer_Advisor' in passed:
        cls.unified = api.UnifiedAnalysisReport.from_json(out / 'api-analysis.json')
        cls.recommendations = advisor.RecommendationReport.from_json(out / 'api-advisor.json')
    if 'test_35_actual_CLI_Analyzer_Advisor' in passed:
        cls.cli_unified = api.UnifiedAnalysisReport.from_json(out / 'cli-analysis.json')
    native_fit_test = cls.test_10_public_Builder_Trainer_CPU_episode
    def persist_owned_model(self):
        native_fit_test(self)
        torch.save(self.session.evaluation_model.state_dict(), out / 'owned-api-state.pt')
    cls.test_10_public_Builder_Trainer_CPU_episode = persist_owned_model
    attempt = os.environ['HYSO_FINAL_ATTEMPT_ID']
    journal = out / ('package-events-' + attempt + '.jsonl')
    previous, sequence, failed = '0' * 64, 0, False
    source_commit=os.environ['HYSO_FINAL_COMMIT']
    source_sha=_SHA256(Path(namespace['__file__']).read_bytes()).hexdigest()
    observer_sha=_SHA256(Path(__file__).read_bytes()).hexdigest()
    environment = hashlib.sha256(json.dumps(dict(python=sys.version, torch=str(torch.__version__),
        kind=args.kind, artifact_sha256=os.environ['HYSO_PACKAGE_SHA256']), sort_keys=True).encode()).hexdigest()
    def checksum(value):
        return _SHA256(_ENCODE(value).encode('utf-8')).hexdigest()
    def emit(kind, test, **fields):
        nonlocal previous, sequence, failed
        if failed:raise RuntimeError('JOURNAL_POISONED: package journal persistence failed')
        row = dict(schema_version=2, sequence=sequence+1, previous_sha256=previous,
            time_utc=_UTC_NOW(timezone.utc).isoformat(), kind=kind,
            nodeid='package:' + args.kind + '::' + test._testMethodName, group='package-' + args.kind,
            attempt_id=attempt, source_commit=source_commit,
            environment_id=environment, test_source_sha256=source_sha,observer_sha256=observer_sha, **fields)
        row['sha256'] = checksum(row)
        try:
            with journal.open('ab') as stream:
                stream.write(_ENCODE(row).encode('utf-8') + b'\n')
                stream.flush()
                _FSYNC(stream.fileno())
        except BaseException:
            failed=True
            raise
        sequence=row['sequence']
        previous = row['sha256']
    def artifacts():
        result = {}
        for path in out.iterdir():
            if path.is_file() and not path.name.startswith('package-events-') and path.name not in ('test-summary.json', '.owned-package.json'):
                with path.open('rb+') as stream:
                    _FSYNC(stream.fileno())
                result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result
    for stage, method in [('setup', '_callSetUp'), ('call', '_callTestMethod'), ('teardown', '_callTearDown')]:
        original = getattr(cls, method)
        def wrapped(self, *positional, _stage=stage, _original=original):
            if _stage == 'setup':
                emit('test_start', self)
            begin = _CLOCK()
            try:
                value = _original(self, *positional)
            except BaseException as exc:
                if _stage=='teardown':
                    self._hyso_pending_teardown=dict(outcome='failed',duration=_CLOCK()-begin,traceback=traceback.format_exc())
                    raise
                emit('phase', self, when=_stage, outcome='skipped' if isinstance(exc, unittest.SkipTest) else 'failed',
                    is_subtest=False, duration=_CLOCK()-begin, traceback=traceback.format_exc())
                raise
            if _stage=='teardown':
                self._hyso_pending_teardown=dict(outcome='passed',duration=_CLOCK()-begin,traceback=None)
                return value
            emit('phase', self, when=_stage, outcome='passed', is_subtest=False,
                 duration=_CLOCK()-begin, traceback=None,
                 artifact_bindings=artifacts() if _stage == 'teardown' else {})
            return value
        setattr(cls, method, wrapped)
    result_cls=namespace['RecordedResult']
    native_subtest=result_cls.addSubTest
    def record_subtest(self,test,subtest,error):
        emit('phase',test,when='call',outcome='failed' if error else 'passed',is_subtest=True,
            duration=0,subtest_context=str(subtest),
            traceback=self._exc_info_to_string(error,subtest) if error else None)
        return native_subtest(self,test,subtest,error)
    result_cls.addSubTest=record_subtest
    native_skip=result_cls.addSkip
    def record_skip(self,test,reason):
        parent=getattr(test,'test_case',None)
        if parent is not None:
            emit('phase',parent,when='call',outcome='skipped',is_subtest=True,duration=0,
                subtest_context=str(test),traceback=reason)
        return native_skip(self,test,reason)
    result_cls.addSkip=record_skip
    native_cleanup=cls._callCleanup
    def record_cleanup(self,function,*args,**kwargs):
        begin=_CLOCK()
        try:return native_cleanup(self,function,*args,**kwargs)
        except BaseException:
            emit('phase',self,when='teardown',outcome='failed',is_subtest=True,is_cleanup=True,
                duration=_CLOCK()-begin,traceback=traceback.format_exc())
            raise
    cls._callCleanup=record_cleanup
    native_cleanups=cls.doCleanups
    def finish_cleanups(self):
        begin=_CLOCK()
        try:return native_cleanups(self)
        finally:
            pending=getattr(self,'_hyso_pending_teardown',dict(outcome='passed',duration=0,traceback=None))
            pending['duration']+=_CLOCK()-begin
            emit('phase',self,when='teardown',is_subtest=False,artifact_bindings=artifacts(),**pending)
    cls.doCleanups=finish_cleanups
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
    return unittest.TestSuite(test for test in suite if test._testMethodName not in terminals)
