"""Admitted child only. This file is never imported by static preparation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Journal, REPO, canonical, digest, file_hash, new_json, read_json
from campaign_collection import canonical_nodeid, profile_for
from collections import Counter


def main():
    request_path = Path(sys.argv[1])
    request = read_json(request_path)
    attempt = request_path.parent
    identity = {k: request[k] for k in ('attempt_id', 'group', 'source_commit', 'environment_id')}
    journal = Journal(attempt / 'events.jsonl', identity)
    try:
        return run_payload(request,attempt,journal)
    except BaseException:
        if not journal.failed:
            journal.emit('framework_error',traceback=traceback.format_exc())
        raise
    finally:
        journal.close()
        new_json(attempt/'journal-terminal.json',dict(schema_version=2,**identity,
            committed_sequence=journal.sequence,committed_head_sha256=journal.previous,
            writer_failed=journal.failed,closed=True,protocol='hyso.event-journal.v2'))


def run_payload(request,attempt,journal):
    journal.emit('worker_start', pid=os.getpid(), profile=request['profile'])
    if request['mode']=='startup-probe':
        if not os.environ.get('SYSTEMROOT'):
            raise RuntimeError('SYSTEMROOT missing in actual worker environment')
        if len(os.urandom(32))!=32 or 'torch' in sys.modules or 'pytest' in sys.modules:
            raise RuntimeError('stdlib startup probe failed')
        journal.emit('startup_probe_pass',python=sys.executable,pid=os.getpid(),
            systemroot_present=True,hash_randomization=bool(sys.flags.hash_randomization),
            random_bytes_available=True,torch_imported=False,pytest_imported=False)
        return 0
    if request['mode'] == 'command':
        try:
            result = subprocess.run(request['command'], cwd=request['cwd'],
                timeout=request['command_timeout_seconds'], check=False)
            journal.emit('operation_finish', returncode=result.returncode)
            return result.returncode
        except BaseException as exc:
            journal.emit('operation_error', exception=type(exc).__name__, traceback=traceback.format_exc())
            return 1
    inventory=request['inventory']
    paths={r['path']:r['test_source_sha256'] for r in inventory.values()}
    for name in request['files']:
        path=canonical_nodeid(name)
        if path not in paths or file_hash(REPO/path)!=paths[path]:
            journal.emit('integrity_error',reason='Unreviewed/changed file refused before pytest import: '+path)
            return 2
        if request['profile']!='infrastructure':
            for parent in (REPO/path).parents:
                if not parent.is_relative_to(REPO):break
                if request['profile']=='stdlib' and not parent.is_relative_to(REPO/'tests/forensics'):break
                fixture=parent/'conftest.py'
                if fixture.is_file():
                    relative=fixture.relative_to(REPO).as_posix()
                    if file_hash(fixture)!=request.get('conftest_sha256',{}).get(relative):
                        journal.emit('integrity_error',reason='Unreviewed/changed pytest fixture refused: '+relative)
                        return 2
    sys.path[:0] = [str(REPO), str(REPO / 'tests'), str(REPO / 'tools'),
                    str(REPO / 'tools/full_acceptance')]
    if request['profile'] in ('stdlib', 'infrastructure'):
        sys.path[:0] = [str(REPO / 'tests/forensics'), str(REPO / 'tools/forensics')]
        if 'torch' in sys.modules:
            raise RuntimeError('forensic parent must remain Torch-free')
    else:
        import torch
        if torch.version.cuda is not None or torch.cuda.is_initialized():
            raise RuntimeError('CPU_ONLY_RUNTIME_REQUIRED')
        def denied(*args, **kwargs):
            raise AssertionError('CPU campaign attempted real CUDA initialization')
        torch.cuda._lazy_init = denied
        torch._C._cuda_init = denied
        journal.emit('runtime', torch_version=str(torch.__version__), torch_source=torch.__file__,
                     cuda_initialized=False, python=sys.executable)
    import pytest
    expected = set(request.get('selected_nodes', inventory))
    forbidden = set(request['blocked_nodes'])
    known=set(request.get('known_nodes',inventory))

    class Recorder:
        def pytest_collection_modifyitems(self, session, config, items):
            hashes={i.path:file_hash(i.path) for i in {item.path:item for item in items}.values()}
            actual = [dict(nodeid=canonical_nodeid(i.nodeid),raw_nodeid=i.nodeid,
                path=i.path.relative_to(REPO).as_posix(),line=i.location[1]+1,
                test_source_sha256=hashes[i.path],markers=sorted(m.name for m in i.iter_markers())) for i in items]
            new_json(attempt / 'collected.json', actual)
            counts=Counter(r['nodeid'] for r in actual)
            keep, excluded = [], []
            for item in items:
                node=canonical_nodeid(item.nodeid)
                allowed = (node in inventory and node in expected and node not in forbidden
                    and counts[node]==1 and profile_for(node)==request['profile'])
                (keep if allowed else excluded).append(item)
            items[:] = keep
            config.hook.pytest_deselected(items=excluded)
            new_json(attempt / 'selection.json', dict(selected=[canonical_nodeid(i.nodeid) for i in keep],
                blocked=[canonical_nodeid(i.nodeid) for i in excluded if canonical_nodeid(i.nodeid) in forbidden],
                unknown=[canonical_nodeid(i.nodeid) for i in excluded if canonical_nodeid(i.nodeid) not in known],
                duplicates={n:c for n,c in counts.items() if c>1}))
            journal.emit('collection_finished', collected=len(actual), selected=len(keep))

        def pytest_collectreport(self, report):
            if report.failed:
                journal.emit('collection_error', collection_nodeid=report.nodeid,
                             traceback=str(report.longrepr))

        def pytest_runtest_logstart(self, nodeid, location):
            raw_nodeid=nodeid
            nodeid=canonical_nodeid(nodeid)
            entry = inventory[nodeid]
            actual = file_hash(REPO / entry['path'])
            if actual != entry['test_source_sha256']:
                raise RuntimeError('EVIDENCE_INVALID: test source changed after admission')
            journal.emit('test_start', nodeid=nodeid, test_source_sha256=actual,
                         source_file=entry['path'], line=location[1] + 1,raw_nodeid=raw_nodeid)

        def pytest_runtest_logreport(self, report):
            context = getattr(report, 'context', None)
            subtest = context is not None or 'subtest' in type(report).__name__.lower()
            nodeid=canonical_nodeid(report.nodeid)
            journal.emit('phase', nodeid=nodeid,raw_nodeid=report.nodeid, when=report.when,
                outcome=report.outcome, duration=report.duration, is_subtest=subtest,
                subtest_context=repr(context) if subtest else None,
                traceback=str(report.longrepr) if report.longrepr else None,
                wasxfail=str(getattr(report, 'wasxfail', '')),
                test_source_sha256=inventory[nodeid]['test_source_sha256'])

        def pytest_internalerror(self, excrepr, excinfo):
            journal.emit('framework_error', traceback=str(excrepr))

        def pytest_sessionfinish(self, session, exitstatus):
            torch_loaded = sys.modules.get('torch')
            if request['profile'] in ('stdlib', 'infrastructure') and torch_loaded is not None:
                journal.emit('integrity_error', reason='Forensic parent imported Torch')
            journal.emit('session_finish', exit_code=int(exitstatus),
                torch_imported=torch_loaded is not None,
                cuda_initialized=bool(torch_loaded and torch_loaded.cuda.is_initialized()))

    args = ['-c', str(REPO / 'tests/pytest.ini'), '--rootdir', str(REPO), '-q',
        '--tb=long', '--full-trace', '--continue-on-collection-errors',
        '--basetemp', str(Path(request['work']) / 'pytest'),
        '--junitxml', str(attempt / 'junit.xml')]
    if request['profile'] == 'stdlib':
        args += ['--noconftest'] if all(f.startswith('tools/full_acceptance/') for f in request['files']) else ['--confcutdir', str(REPO / 'tests/forensics')]
    elif request['profile'] == 'infrastructure':
        args += ['--noconftest']
    else:
        args += ['--ignore', str(REPO / 'tests/forensics')]
    if request['mode'] == 'collect':
        args.append('--collect-only')
    args += request['files']
    try:
        return int(pytest.main(args, plugins=[Recorder()]))
    except BaseException:
        journal.emit('framework_error', traceback=traceback.format_exc())
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
