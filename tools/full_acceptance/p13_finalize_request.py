"""Separate one-shot offline finalizer authorization. No CUDA/Nsight invocation.

A request against a prepared capture is WAITING and cannot be authorized. After
the actual capture has a clean terminal, prepare a NEW request against its exact
authorized child manifest. No future artifact hashes are invented.
"""
from pathlib import Path
import argparse
import json
import sys

# core.verify uses the established full_acceptance.registry import contract.
_toolroot = str(Path(__file__).resolve().parents[1])
if _toolroot not in sys.path:
    sys.path.insert(0, _toolroot)
from . import core


def capture_state(path, expected):
    m = core.verify(path, expected=expected)
    out = Path(m['output'])
    if not (out/'STARTED.json').exists():
        return m, False, []
    status = core.read(out/'run-status.json')
    if (status.get('controller_active') is not False
            or status.get('preservation_status') != 'PASS'
            or status.get('finalization_errors')
            or status.get('manifest_sha256') != expected
            or status.get('run_id') != m['run_id']
            or status.get('state') not in ('INCOMPLETE','EXECUTION_COMPLETE_WITH_RESULTS')):
        raise ValueError('capture controller has no clean terminal; cannot prepare finalization')
    approved = [e['id'] for e in m['experiments'] if e['authorization']['approved']]
    if approved != ['02.regression','06.p13_capture']:
        raise ValueError('capture plus its strict CPU regression prerequisite authorization required')
    rows = core.read(out/'stage-results.json')['experiments']
    regression, = [r for r in rows if r['id']=='02.regression']
    if (regression.get('status') != 'PASS' or regression.get('attempted') is not True
            or regression.get('criteria',{}).get('safe_regressions') is not True
            or regression.get('pytest_exit') != 0):
        raise ValueError('strict CPU regression prerequisite is not PASS')
    row, = [r for r in rows if r['id']=='06.p13_capture']
    if row.get('status') != 'PASS' or row.get('attempted') is not True:
        raise ValueError('capture stage is not PASS')
    refs = [core.record(out/name) for name in
            ('STARTED.json','run-status.json','stage-results.json','preservation-verification.json')]
    refs += [core.record(p) for p in sorted((out/'stages/06.p13_capture').rglob('*')) if p.is_file()]
    if not refs[4:]:raise ValueError('capture artifacts missing')
    refs += [core.record(p) for p in sorted((out/'stages/02.regression').rglob('*')) if p.is_file()]
    return m, True, refs


def prepare(capture, expected, output):
    capture=Path(capture).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('new finalizer request directory required')
    m, ready, refs=capture_state(capture,expected)
    prior=Path(m['output']).resolve()
    if output==prior or prior in output.parents:raise ValueError('cannot write into capture evidence')
    value=dict(format='hyso.p13.finalizer.request',version=1,
        state='READY_FOR_SEPARATE_APPROVAL' if ready else 'WAITING_FOR_COMPLETED_CAPTURE',
        capture_manifest=core.record(capture),capture_artifacts=refs,
        capture_source_tree_sha256=m['source_snapshot']['tree_sha256'],
        implementation=core.record(__file__),
        finalizer=core.record(Path(__file__).with_name('p13_capture_finalize.py')),
        output=str(output),authorization=dict(approved=False,basis=None),
        native_execution=False,layer_acceptance=False,automatic_retry=False)
    output.mkdir(parents=True)
    core.publish(output/'manifest.json',value,new=True)
    return value


def verify(path, expected):
    path=Path(path).resolve()
    if core.sha(path)!=expected:raise ValueError('finalizer request SHA256 mismatch')
    r=core.read(path)
    if (r.get('format')!='hyso.p13.finalizer.request' or r.get('version')!=1
            or Path(r['output']).resolve()!=path.parent or r.get('automatic_retry') is not False
            or r.get('native_execution') is not False or r.get('layer_acceptance') is not False):
        raise ValueError('invalid finalizer request')
    for ref in [r['capture_manifest'],r['implementation'],r['finalizer'],*r['capture_artifacts']]:
        if core.sha(ref['path'])!=ref['sha256']:raise ValueError('finalizer bound artifact changed')
    if r['implementation']!=core.record(__file__):raise ValueError('finalizer driver changed')
    if (Path(r['output'])/'STARTED.json').exists():raise FileExistsError('finalizer request already consumed')
    return r


def require_ready(r):
    if r['state']!='READY_FOR_SEPARATE_APPROVAL':
        raise ValueError('waiting request cannot execute; prepare NEW request after actual clean capture')
    m, ready, refs=capture_state(r['capture_manifest']['path'],r['capture_manifest']['sha256'])
    if not ready or refs!=r['capture_artifacts'] or m['source_snapshot']['tree_sha256']!=r['capture_source_tree_sha256']:
        raise ValueError('capture binding changed')


def authorize(path, expected, output, basis):
    r=verify(path,expected);require_ready(r)
    if r['authorization']['approved'] is not False or not basis.strip():
        raise ValueError('new separate explicit finalizer approval required')
    output=Path(output).resolve()
    prior=Path(r['capture_manifest']['path']).parent
    if output==prior or prior in output.parents:raise ValueError('cannot write into capture evidence')
    output.mkdir(parents=True,exist_ok=False)
    r.update(output=str(output),parent_request=core.record(path),
             authorization=dict(approved=True,basis=basis,approved_at=core.now()))
    core.publish(output/'manifest.json',r,new=True)
    return r


def execute(path, expected):
    r=verify(path,expected);require_ready(r)
    if r['authorization']['approved'] is not True or not r['authorization'].get('basis'):
        raise PermissionError('finalizer NOT_AUTHORIZED')
    root=Path(r['output'])
    core.publish(root/'STARTED.json',dict(manifest_sha256=expected,one_shot=True),new=True)
    from .p13_capture_finalize import build
    build(r['capture_manifest']['path'],r['capture_manifest']['sha256'],root/'finalized')
    core.publish(root/'result.json',dict(status='PROVENANCE_VALIDATED_NOT_06_LAYER_ACCEPTANCE',
        manifest_sha256=expected,configuration=core.record(root/'finalized/p13-evidence-configuration.json')),new=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','authorize','execute'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--approval-text')
    a=p.parse_args(argv)
    if a.mode=='prepare':
        if a.output is None:p.error('--output required')
        r=prepare(a.manifest,a.sha256,a.output)
    elif a.mode=='authorize':
        if a.output is None or not a.approval_text:p.error('--output and --approval-text required')
        r=authorize(a.manifest,a.sha256,a.output,a.approval_text)
    else:
        execute(a.manifest,a.sha256);return 0
    target=Path(r['output'])/'manifest.json'
    print(json.dumps(dict(manifest=str(target),sha256=core.sha(target),state=r['state'],
                         authorization=r['authorization'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
