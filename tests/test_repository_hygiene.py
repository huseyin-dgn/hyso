"""Portable fixtures, resolvable user documentation and scoped artifact ignores."""
import json
from pathlib import Path
import re
import subprocess


ROOT=Path(__file__).resolve().parents[1]


def test_local_artifacts_ignored_without_masking_source_or_fixtures():
    ignored=['.cursorindexingignore','.env.local','private.key','private.pfx',
        'evidence/run.json','evidence/run.csv','checkpoints/run.pth',
        'weights/model.pt','profiling/run.json','cuda-logs/run.txt','scratch_p21_local.py']
    legitimate=['.env.example','.env.sample','tests/fixtures/new-fixture.json','tests/new_test.py',
        'hyso/new_module.py','docs/guides/new-guide.md']
    tracked=subprocess.check_output(['git','--no-optional-locks','ls-files'],cwd=ROOT,text=True).splitlines()
    paths=ignored+legitimate+tracked
    result=subprocess.run(['git','--no-optional-locks','check-ignore','--no-index','-z','--stdin'],
        cwd=ROOT,input=('\0'.join(paths)+'\0').encode(),capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert set(result.stdout.decode().strip('\0').split('\0'))==set(ignored)


def test_user_documentation_relative_links_resolve():
    paths=[ROOT/'README.md',ROOT/'hyso/README.md',*(ROOT/'docs/guides').glob('*.md')]
    assert len(paths)==15
    for path in paths:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','mailto:','#')):continue
            assert (path.parent/target.split('#')[0]).is_file(),(path.relative_to(ROOT),target)
    secondary=(ROOT/'hyso/README.md').read_text(encoding='utf-8')
    assert '[MIT License](../LICENSE)' in secondary and '[project README](../README.md)' in secondary


def test_fixture_environment_is_portable_and_keeps_structural_reference():
    manifest=json.loads((ROOT/'tests/fixtures/model_state_manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema_version']==1 and manifest['models'] and manifest['source_sha256']
    assert manifest['environment']['executable'] is None
    assert manifest['environment']['platform']=='Windows'
    assert re.fullmatch(r'\d+\.\d+\.\d+',manifest['environment']['python'])
    for path,digest in manifest['source_sha256'].items():
        assert (ROOT/path).is_file() and re.fullmatch('[0-9a-f]{64}',digest)
    assert all(case['state_dict'] and case['constructor'] for case in manifest['models'])
