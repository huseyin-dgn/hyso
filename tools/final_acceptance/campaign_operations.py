"""Deferred package work and bounded evidence harvesting; never runs at import."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

from common import LEGACY, REPO, ROOT, atomic_json, file_hash, new_json, read_json

PREVIOUS = REPO / 'outputs/package-clean-cpu-54e64e9-20261125'
SAFE_RAW_NAMES = {'controller-terminal.json', 'parent-receipt.json', 'worker-receipt.json',
    'worker-identity.json', 'worker-phases.jsonl', 'loaded-python-identities.json',
    'loaded-native-identities.json', 'timeout-result.json', 'timeout-phases.json',
    'test-summary.json', 'result-summary.json', 'result.json', 'controller-identity.json', 'manifest.json'}


def mirror_package_journals(scratch, target):
    """A readable prefix is durable while the original single writer keeps running."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in Path(scratch).glob('packages-*/*-external/E2E/package-events-*.jsonl'):
        destination = target / source.name
        temporary = destination.with_suffix('.tmp')
        if destination.exists():
            old=destination.read_bytes()
            complete=old[:old.rfind(b'\n')+1]
            with source.open('rb') as incoming:
                if incoming.read(len(complete))!=complete:
                    raise RuntimeError('EVIDENCE_INVALID: package journal prefix changed')
        with source.open('rb') as incoming, temporary.open('wb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
        paths.append(destination)
    return paths


def harvest(work, attempt):
    result, total = [], 0
    for path in Path(work).rglob('*'):
        if path.name not in SAFE_RAW_NAMES or not path.is_file() or path.is_symlink():
            continue
        if not path.resolve().is_relative_to(Path(work).resolve()):
            continue
        size = path.stat().st_size
        if size > 32 * 1024**2 or total + size > 128 * 1024**2 or len(result) >= 512:
            continue
        target = Path(attempt) / 'harvested' / path.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        result.append(dict(path=str(target), sha256=file_hash(target), bytes=size))
        total += size
    atomic_json(Path(attempt) / 'harvest.json', result)
    return result


def cleanup_owned(work, scratch, campaign_id):
    """Remove only this campaign's marker-bound temporary directory, never a link target."""
    work, scratch = Path(work), Path(scratch).resolve()
    resolved = work.resolve()
    if not resolved.is_relative_to(scratch) or resolved == scratch or work.is_symlink():
        raise ValueError('refusing cleanup outside the named scratch root')
    marker = read_json(work / '.hyso-owned-work.json', {})
    if marker.get('campaign_id') != campaign_id:
        raise ValueError('refusing cleanup without the owned marker')
    # Python 3.10 on Windows does not recurse into directory junctions in rmtree.
    # Remove reparse entries explicitly before the ordinary tree, for auditable bounds.
    def prune(directory):
        for entry in os.scandir(directory):
            attributes = getattr(entry.stat(follow_symlinks=False), 'st_file_attributes', 0)
            if attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
                if entry.is_dir(follow_symlinks=False):
                    os.rmdir(entry.path)
                else:
                    os.unlink(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                prune(entry.path)
    prune(resolved)
    def writable_remove(function, name, exc):
        path = Path(name)
        if not path.resolve().is_relative_to(resolved):
            raise ValueError('cleanup error path escapes owned scratch')
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        function(name)
    shutil.rmtree(resolved, onerror=writable_remove)


def prepare_package_workspace(scratch, campaign_id, commit):
    work = Path(scratch) / ('packages-' + commit[:12])
    if work.exists():
        if read_json(work / '.hyso-owned-work.json', {}).get('campaign_id') != campaign_id:
            raise ValueError('existing package workspace lacks matching ownership marker')
        for name in ('package_fixture_runtime.py','verify_installed_payload.py'):
            current=Path(__file__).with_name(name);copied=work/'fixtures'/name
            if copied.is_file() and file_hash(copied)!=file_hash(current):
                history=work/'fixtures/version-history';history.mkdir(exist_ok=True)
                saved=history/(file_hash(copied)+'-'+name)
                if not saved.exists():shutil.copyfile(copied,saved)
                shutil.copyfile(current,copied)
    else:
        work.mkdir()
        new_json(work / '.hyso-owned-work.json', dict(campaign_id=campaign_id, attempt_id='packages'))
        for name in ('source', 'fixtures', 'dist'):
            (work / name).mkdir()
        shutil.copyfile(PREVIOUS / 'clean_fixture.py', work / 'fixtures/clean_fixture.py')
    # Reuse the already validated extra native capture test verbatim from its retained adapter.
    tree = ast.parse((LEGACY / 'package.py').read_text(encoding='utf-8'))
    extra = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'extra' for t in n.targets))
    guide_and_csv = '''
    def test_71_current_contextual_help_and_four_language_offline_guide(self):
        import inspect,hyso_guide
        contract=advisor.CapturedCPUComparisonContract
        self.assertIn('HYSo Guide: comparison.models',inspect.getdoc(contract))
        self.assertEqual(hyso_guide.topic_for_api('hyso.advisor.CapturedCPUComparisonContract'),'comparison.models')
        for language in ('tr','en','de','es'):
            topic=hyso_guide.show_topic('comparison.models',language)
            self.assertEqual(topic['id'],'comparison.models');self.assertTrue(topic['text'])
            self.assertIn('?lang='+language+'#comparison.models',topic['uri'])

    def test_72_native_CSV_preserves_JSON_capture(self):
        import csv
        for feature,typ in [('P01',perf.LatencyAnalysis),('P02',perf.ThroughputResult),('P11',perf.FootprintResult)]:
            raw=json.loads((OUT/('A-'+feature+'-capture.json')).read_text())
            value=typ(raw);path=OUT/(feature+'-capture.csv');value.to_csv(path)
            with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            # Native CSV is a typed projection; JSON retains the complete record.
            self.assertEqual(value.to_dict(),raw)
            canonical=OUT/(feature+'-capture-roundtrip.json');value.to_json(canonical)
            self.assertEqual(json.loads(canonical.read_text(encoding='utf-8')),raw)
            common=dict(format=raw['format'],schema_version=raw['schema_version'],
                result_id=raw['result_id'],device=raw['provenance']['environment']['device'],
                scope=raw['protocol']['scope'])
            if feature=='P01':
                expected=[dict(common,source_result_id=raw['identity']['source_result_id'],
                    sample_count=raw['actual']['samples'],**metric) for metric in raw['measurements']]
            elif feature=='P02':
                expected=[dict(common,**window) for window in raw['raw_evidence']['windows']]
                for row in expected:
                    row['completed_id_range']=json.dumps(row['completed_id_range'])
                    row.update(unit='images/s',status='measured',reason=None)
            else:
                expected=[dict(common,metric=name,**metric) for name,metric in raw['measurements'].items()]
            def cell(v):
                if v is None:return ''
                if type(v) is str and v.lstrip(' '+chr(9)+chr(13)+chr(10)).startswith(('=','+','-','@')):return "'"+v
                return str(v)
            self.assertTrue(expected);self.assertEqual(len(rows),len(expected))
            for actual,wanted in zip(rows,expected):
                self.assertEqual(set(actual),set(wanted))
                self.assertEqual(actual,{k:cell(v) for k,v in wanted.items()})
'''
    harness = work / 'fixtures/verify_installed.py'
    source = (PREVIOUS / 'verify_installed.py').read_text(encoding='utf-8')
    needle = "for lang in ('tr','en','de','es'):"
    if source.count(needle) != 1:
        raise ValueError('retained package harness projection changed')
    source = source.replace(needle, extra + '\n' + guide_and_csv + '\n' + needle)
    source = source.replace("args=parser.parse_args()", "parser.add_argument('--resume-state');args=parser.parse_args()")
    source = source.replace('OUT.mkdir()', 'OUT.mkdir(exist_ok=True)')
    load_suite = 'unittest.defaultTestLoader.loadTestsFromTestCase(InstalledCPUWorkflow)'
    source = source.replace(load_suite, '__import__("package_fixture_runtime").attach(globals())')
    if harness.is_file() and harness.read_text(encoding='utf-8')!=source:
        # Retain the 34 passing cases only if this refresh changes exactly the
        # reviewed CSV assertion, including unchanged fixture/setup/global code.
        trees=[ast.parse(text) for text in (harness.read_text(encoding='utf-8'),source)]
        for tree in trees:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InstalledCPUWorkflow')
            cls.body=[n for n in cls.body if not (isinstance(n,ast.FunctionDef)
                and n.name=='test_72_native_CSV_preserves_JSON_capture')]
        if ast.dump(trees[0])!=ast.dump(trees[1]):
            raise RuntimeError('EVIDENCE_INVALID: unreviewed package harness changes; retained cases cannot be reused')
        history=work/'fixtures/version-history';history.mkdir(exist_ok=True)
        saved=history/(file_hash(harness)+'-verify_installed.py')
        if not saved.exists():shutil.copyfile(harness,saved)
    harness.write_text(source, encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('package_fixture_runtime.py'), work / 'fixtures/package_fixture_runtime.py')
    shutil.copyfile(Path(__file__).with_name('verify_installed_payload.py'), work / 'fixtures/verify_installed_payload.py')
    return work


def package_commands(work, runtime, commit):
    python = runtime['cpu_python']
    source, dist = work / 'source', work / 'dist'
    commands = [dict(name='package-archive', command=['git', 'archive', '--format=zip',
        '--output', str(work / 'source.zip'), commit], cwd=str(REPO)),
        dict(name='package-extract', internal='extract')]
    for kind, option in (('wheel', '--wheel'), ('sdist', '--sdist')):
        commands.append(dict(name='package-build-' + kind, command=[python, '-I', '-B', '-m', 'build',
            '--no-isolation', option, '--outdir', str(dist), str(source)], cwd=str(work)))
    return commands


def install_commands(work, runtime, kind):
    python = work / (kind + '-venv/Scripts/python.exe')
    cwd = work / (kind + '-external')
    cwd.mkdir(exist_ok=True)
    suffix = '*.whl' if kind == 'wheel' else '*.tar.gz'
    artifacts = list((work / 'dist').glob(suffix))
    if len(artifacts) != 1:
        return [], dict(status='ERROR', reason_code='PACKAGE_BUILD_FAILED')
    artifact = artifacts[0]
    base = runtime['cpu_python']
    steps = [dict(name=kind + '-venv', command=[base, '-I', '-B', '-m', 'venv', '--without-pip', str(python.parents[1])], cwd=str(work)),
        dict(name=kind + '-ensurepip', command=[str(python), '-I', '-B', '-m', 'ensurepip'], cwd=str(cwd)),
        dict(name=kind + '-empty', command=[str(python), '-I', '-B', '-c',
            "import importlib.util,sys,site;assert sys.prefix!=sys.base_prefix;assert not site.ENABLE_USER_SITE;assert importlib.util.find_spec('hyso') is None;assert importlib.util.find_spec('torch') is None"], cwd=str(cwd)),
        dict(name=kind + '-dependencies', command=[str(python), '-I', '-B', '-m', 'pip', '--isolated',
            'install', '--no-index', '--find-links', runtime['wheelhouse'], '--no-compile',
            '-r', str(PREVIOUS / 'requirements.txt'), 'torch==2.5.1+cpu', 'torchvision==0.20.1+cpu'], cwd=str(cwd)),
        dict(name=kind + '-install', command=[str(python), '-I', '-B', '-m', 'pip', '--isolated',
            'install', '--no-index', '--no-build-isolation', '--find-links', runtime['wheelhouse'],
            '--no-compile', '--report', str(work / (kind + '-install.json')), str(artifact)], cwd=str(cwd)),
        dict(name=kind + '-pip-check', command=[str(python), '-I', '-B', '-m', 'pip', 'check'], cwd=str(cwd)),
        dict(name=kind + '-source-hashes', command=[str(python), '-I', '-B',
            str(work / 'fixtures/verify_installed_payload.py'), str(work / 'source'), str(REPO),
            str(cwd / 'installed-payload.json')], cwd=str(cwd)),
        dict(name=kind + '-E2E', command=[str(python), '-I', '-B', str(work / 'fixtures/verify_installed.py'),
            '--env', str(python.parents[1]), '--output', str(cwd / 'E2E'), '--repo', str(REPO), '--kind', kind],
            cwd=str(cwd), timeout=900)]
    return steps, dict(artifact=str(artifact), sha256=file_hash(artifact), source_commit=None)


def expected_package_cases():
    """Static expansion of the retained harness; never imports the installed product."""
    tree = ast.parse((PREVIOUS / 'verify_installed.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InstalledCPUWorkflow')
    names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
    # The retained harness adds 8 language cases dynamically below the class.
    template = (PREVIOUS / 'verify_installed.py').read_text(encoding='utf-8')
    assignment = next(line for line in template.splitlines() if 'setattr(InstalledCPUWorkflow' in line)
    if "'test_40_API_reports_'" not in assignment:
        raise ValueError('review the retained language case expansion')
    names += ['test_40_API_reports_'+lang for lang in ('tr','en','de','es')]
    names += ['test_50_CLI_reports_'+lang for lang in ('tr','en','de','es')]
    names += ['test_70_native_captured_CPU_comparison_contract',
              'test_71_current_contextual_help_and_four_language_offline_guide',
              'test_72_native_CSV_preserves_JSON_capture']
    # The extra method name is obtained literally rather than guessed from the prior run.
    extra_tree = ast.parse((LEGACY / 'package.py').read_text(encoding='utf-8'))
    extra = next(ast.literal_eval(n.value) for n in extra_tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id=='extra' for t in n.targets))
    extra_method = next(n.name for n in ast.parse('class X:\n'+extra).body[0].body if isinstance(n,ast.FunctionDef))
    names[-3] = extra_method
    return ['package:'+kind+'::'+name for kind in ('wheel','sdist') for name in sorted(set(names))]


def audit_archives(work):
    source = Path(work) / 'source'
    profile = read_json(source / 'commercial-profile.json')
    expected = set(profile['python_files'])
    function = next(n for n in ast.parse((source / 'setup.py').read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'commercial_payload')
    namespace = {'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'reviewed_commercial_projection', 'exec'), namespace)
    rows = []
    for path in (Path(work) / 'dist').iterdir():
        try:
            if path.suffix == '.whl':
                with zipfile.ZipFile(path) as archive:
                    files = {n: archive.read(n) for n in archive.namelist() if not n.endswith('/')}
                extra = set()
            else:
                with tarfile.open(path) as archive:
                    files = {m.name.split('/', 1)[1]: archive.extractfile(m).read()
                             for m in archive.getmembers() if m.isfile() and '/' in m.name}
                extra = {'setup.py'}
            actual = {n for n in files if n.endswith('.py')}
            if actual != expected | extra:
                raise ValueError('commercial Python allowlist mismatch')
            for name in expected:
                if files[name] != namespace['commercial_payload'](name, (source / name).read_bytes()):
                    raise ValueError('source payload mismatch: ' + name)
            if any('/outputs/' in '/' + n or '.vscode/' in n or n.endswith(('.bak', '.pth', '.pt', '.patch')) for n in files):
                raise ValueError('forbidden private/development artifact in package')
            rows.append(dict(path=str(path), sha256=file_hash(path), status='PASS', modules=len(expected)))
        except BaseException as exc:
            rows.append(dict(path=str(path), status='FAIL', error=type(exc).__name__ + ': ' + str(exc)))
    return rows


def copy_package_reports(work, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    summaries = []
    for kind in ('wheel', 'sdist'):
        folder = Path(work) / (kind + '-external/E2E')
        summary = read_json(folder / 'test-summary.json')
        if summary:
            summaries.append(summary)
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in ('.json', '.md', '.html', '.csv', '.stdout', '.stderr'):
                out = target / kind / path.name
                out.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, out)
    return summaries
