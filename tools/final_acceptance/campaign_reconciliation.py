"""Read-only legacy bindings and preservation checks; no model deserialization."""
from pathlib import Path
import shutil

from common import LEGACY, ROOT, REPO, atomic_json, digest, file_hash, read_json, utc


def verify_files(bindings):
    rows = {}
    for name, binding in bindings.items():
        path=Path(name)
        expected=binding.get('sha256') if isinstance(binding,dict) else binding
        actual=file_hash(path) if path.is_file() else None
        rows[name]=dict(expected_sha256=expected,actual_sha256=actual,
            status='PASS' if actual==expected else 'EVIDENCE_INVALID')
    return rows


def protected_inputs():
    return verify_files(read_json(LEGACY/'trained-input-preservation-before.json',{}))


def preserve_legacy(review):
    """Copy only named evidence, never arbitrary user files or environments."""
    declared=read_json(LEGACY/'historical-evidence-bindings.json',{})
    checked=verify_files(declared)
    p31=read_json(REPO/'outputs/full-library-acceptance-20261125/b4-p23-p31-20261125/P31-scope-review.json',{})
    p31_bindings={r['source']['path']:r['source'] for r in p31.get('historical_sources',[])}
    atomic_json(ROOT/'preparation/P31-historical-integrity.json',dict(checks=verify_files(p31_bindings),
        historical_only=True,no_new_run=True,expected_models=2))
    files={Path(name) for name in review['evidence_bindings']}
    for path in list(files):
        stem=path.name[:-len('-reports.jsonl')]
        files.update(LEGACY/(stem+suffix) for suffix in ('-controller.json','-runtime.json','-exit.json','.xml'))
    files.update(Path(n) for n in declared)
    files.update(LEGACY/n for n in ('P01-P35-draft.json','historical-evidence-bindings.json',
        'risk-preflight.json','failure-review.json','blocked-nodeids.json','trained-input-preservation-before.json',
        'installed-packages.json','archive-audit.json','wheel-E2E-summary.json','sdist-E2E-summary.json'))
    mapping={}
    for source in sorted(files):
        if not source.is_file() or source.is_symlink() or source.stat().st_size>64*1024**2:
            continue
        sha=file_hash(source)
        destination=ROOT/'legacy-evidence'/(sha[:16]+'-'+source.name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(source,destination)
        if file_hash(destination)!=sha:
            raise RuntimeError('EVIDENCE_INVALID: retained legacy copy changed')
        mapping[str(source)]=dict(sha256=sha,bundle_path=destination.relative_to(ROOT).as_posix())
    result=dict(checked_utc=utc(),historical_bindings=checked,
        historical_bindings_valid=all(r['status']=='PASS' for r in checked.values()),copies=mapping,
        scope='Original CPU/CUDA scopes remain historical; no new measurement provenance is manufactured.')
    atomic_json(ROOT/'preparation/legacy-preservation.json',result)
    return result


def validate_checkpoint_evidence(cp):
    checked={}
    for value in cp.data['results'].values():
        path=value.get('evidence_file')
        expected=value.get('evidence_sha256')
        if path and expected and path not in checked:
            checked[path]=dict(sha256=expected)
    for attempt in cp.data.get('attempts',{}).values():
        folder=Path(attempt['directory'])
        for filename,key in [('controller.json','controller_sha256'),('events.jsonl','journal_sha256'),('collected.json','collected_sha256')]:
            if attempt.get(key):checked[str(folder/filename)]=dict(sha256=attempt[key])
        receipt=attempt.get('controller',{})
        for filename,key in [('request.json','request_sha256'),('bootstrap.py','bootstrap_sha256'),
                ('terminal-evidence.json','terminal_evidence_sha256'),('journal-terminal.json','worker_terminal_sha256')]:
            if receipt.get(key):checked[str(folder/filename)]=dict(sha256=receipt[key])
    invalid={n:r for n,r in verify_files(checked).items() if r['status']!='PASS'}
    if invalid:
        atomic_json(ROOT/'preparation/invalid-evidence.json',invalid)
        raise RuntimeError('EVIDENCE_INVALID: retained raw result hash changed')
