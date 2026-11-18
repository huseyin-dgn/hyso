"""Portable fixtures, resolvable user documentation and scoped artifact ignores."""
from fixture_support import load_fixture, example_fixture
import hashlib
import runpy
import ast
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
        'hyso/new_module.py','docs/guides/new-guide.md','docs/roadmap/new-roadmap.md']
    tracked=subprocess.check_output(['git','--no-optional-locks','ls-files'],cwd=ROOT,text=True).splitlines()
    paths=ignored+legitimate+tracked
    result=subprocess.run(['git','--no-optional-locks','check-ignore','--no-index','-z','--stdin'],
        cwd=ROOT,input=('\0'.join(paths)+'\0').encode(),capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert set(result.stdout.decode().strip('\0').split('\0'))==set(ignored)


def test_executable_fixtures_preserve_every_original_python_block():
    expected = {'README.md': {'path': 'tests/fixtures/examples/readme.py', 'blocks': 3, 'sha256': 'd8fa68cddd3f7642921e602048ef99b36ae7e6655dac3cfeaa7ce5694ab7f434'}, 'docs/guides/experiment-manager.md': {'path': 'tests/fixtures/examples/experiment_manager.py', 'blocks': 5, 'sha256': 'de086512312d598657943cf20418857fea020de7f084e40c71a1215d940adeff'}, 'docs/guides/performance-dataloader.md': {'path': 'tests/fixtures/examples/performance_dataloader.py', 'blocks': 3, 'sha256': '50777020ea514f9b74eddc5bbe36ec2617ef8da919617b30cff4488c23b98791'}, 'docs/guides/performance-gradient-amp.md': {'path': 'tests/fixtures/examples/performance_gradient_amp.py', 'blocks': 1, 'sha256': 'b677c240a8b1fbdf704e2514d19b2be58a2da1e89cb8fbb7d7e9613ac60ad954'}, 'docs/guides/performance-inference.md': {'path': 'tests/fixtures/examples/performance_inference.py', 'blocks': 3, 'sha256': '268ca4219c48b4f832b92dc831aea36f36eaae4ea90a2b07b79554d5cd857827'}, 'docs/guides/performance-max-batch.md': {'path': 'tests/fixtures/examples/performance_max_batch.py', 'blocks': 2, 'sha256': 'edce75ed1cdd100de4a6e1a26a344625cbbb0af35aa00dc5631a20cc0510f9a6'}, 'docs/guides/performance-memory.md': {'path': 'tests/fixtures/examples/performance_memory.py', 'blocks': 4, 'sha256': '4bad9d7d9a3b6b7981641020a719530a804e4e13aacd75a913445cc72e669e03'}, 'docs/guides/performance-pipeline.md': {'path': 'tests/fixtures/examples/performance_pipeline.py', 'blocks': 4, 'sha256': 'af05d102999af7cf217aa0363182c8dc5d899d16ebb4b19258138187be6e3e60'}, 'docs/guides/performance-profiling.md': {'path': 'tests/fixtures/examples/performance_profiling.py', 'blocks': 3, 'sha256': 'af63768afe2059b2c800d87f09f6e5503aaf957bacc3c4bfbac7bd8fc7815ac7'}, 'docs/guides/performance-sweeps.md': {'path': 'tests/fixtures/examples/performance_sweeps.py', 'blocks': 3, 'sha256': '3973e2b4d7736d40a1f5a54e1be163ebac1711822349fe8128e98a8033a4179e'}, 'docs/guides/performance-training-breakdown-throughput.md': {'path': 'tests/fixtures/examples/performance_training_breakdown_throughput.py', 'blocks': 2, 'sha256': '41eabd46476ca9951006b19f7b5a71da52a432ad0959bfa6da9782f908aef4a9'}, 'docs/guides/performance-training-memory.md': {'path': 'tests/fixtures/examples/performance_training_memory.py', 'blocks': 1, 'sha256': 'eea981e1332db4b9189d8f24d060a09955b65b1ea24a68c8d12a4dc2932e9976'}, 'docs/guides/performance-training-timing.md': {'path': 'tests/fixtures/examples/performance_training_timing.py', 'blocks': 2, 'sha256': '3617b4b1d54f214d6dc9666a7937b28b222d36f6f38c8cfefe82dc1cb3d61da0'}, 'docs/guides/performance-transfer-utilization.md': {'path': 'tests/fixtures/examples/performance_transfer_utilization.py', 'blocks': 2, 'sha256': 'f7d4a098a09261e0eb6dca4b09a17a0e01321e15a2beab83dea987512b9e6560'}}
    assert len(expected) == 14
    for document, proof in expected.items():
        fixture = example_fixture(document)
        blocks = runpy.run_path(str(fixture))['BLOCKS']
        assert len(blocks) == proof['blocks']
        assert hashlib.sha256(json.dumps(blocks, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest() == proof['sha256']
        for block in blocks:
            ast.parse(block)
    # Consolidation preserves the original checkout's public README and guides.
    # Inspect tracked paths so ignored private evidence remains outside this contract.
    tracked = subprocess.check_output(
        ['git', '--no-optional-locks', 'ls-files'], cwd=ROOT, text=True).splitlines()
    assert not any(path.startswith(('docs/audit/', 'docs/design/')) for path in tracked)
    assert (ROOT / 'README.md').is_file()
    assert (ROOT / 'docs/guides/performance-profiling.md').is_file()


def test_fixture_environment_is_portable_and_keeps_structural_reference():
    manifest=load_fixture('model_state_manifest')
    assert manifest['schema_version']==1 and manifest['models'] and manifest['source_sha256']
    assert manifest['environment']['executable'] is None
    assert manifest['environment']['platform']=='Windows'
    assert re.fullmatch(r'\d+\.\d+\.\d+',manifest['environment']['python'])
    for path,digest in manifest['source_sha256'].items():
        assert (ROOT/path).is_file() and re.fullmatch('[0-9a-f]{64}',digest)
    assert all(case['state_dict'] and case['constructor'] for case in manifest['models'])
