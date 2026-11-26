"""Release policy metadata and denied paths only; no CUDA or long-run workload."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import hyso_guide
from hyso import analysis
from hyso.performance import stability
from hyso.analysis.native_adapters import _runtime_admission

ROOT=Path(__file__).resolve().parents[1]


def test_matrix_api_and_four_language_help_share_release_scope():
    source=json.loads((ROOT/'docs/user-guide/content.json').read_text(encoding='utf-8'))['release_support']
    installed=json.loads((ROOT/'hyso/data/user-guide-topics.json').read_text(encoding='utf-8'))['release_support']
    assert source==installed and source['full_cuda_acceptance'] is False
    rows={r['feature_id']:r for r in source['features']}
    assert list(rows)==['P%02d'%i for i in range(1,36)]
    for row in analysis.feature_catalog():assert row.to_dict()['release_support']==rows[row.feature_id]
    assert rows['P23']['subscopes']['cuda_saved_activation']=='UNSUPPORTED'
    assert rows['P31']['cuda']=='DEFERRED'
    for lang in ('tr','en','de','es'):
        for topic in ('analysis.p23','analysis.p31','support.cpu-gpu'):
            text=hyso_guide.show_topic(topic,lang)['text']
            assert 'BLOCKED_B4_UNRESOLVED' in text and '7200' in text and 'DEFERRED' in text


def test_deferred_cuda_long_run_is_rejected_before_public_owned_and_worker_side_effects(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('deferred path started work')
    monkeypatch.setattr(stability,'_factory',forbidden)
    monkeypatch.setattr(stability.subprocess,'Popen',forbidden)
    recipe=stability.WorkloadRecipe('release_scope:never',gpu_uuid='GPU-'+'a'*16)
    protocol=stability.StabilityProtocol(duration_seconds=7200,interval_seconds=1,max_samples=10000)
    request=dict(recipe=asdict(recipe),protocol=asdict(protocol))
    for entry in ('public','owned','worker'):
        directory=tmp_path/entry
        with pytest.raises(NotImplementedError,match='P31_CUDA_LONG_RUN_DEFERRED'):
            if entry=='public':stability.measure_stability(recipe,directory,protocol=protocol)
            elif entry=='owned':stability._run_owned(directory,'stability',request,startup_timeout=1,heartbeat_timeout=1,max_runtime=1,max_bytes=1024)
            else:stability._worker_stability(directory,request)
        assert not directory.exists()


def test_cpu_and_short_gpu_scope_remain_independent():
    for recipe,seconds in [({'gpu_uuid':None},7200),({'gpu_uuid':'GPU-'+'a'*16},2),({'gpu_uuid':'GPU-'+'a'*16},7199.9)]:
        assert stability._check_release_stability_scope(recipe,{'duration_seconds':seconds}) is None
    for seconds in (7200,7201):
        with pytest.raises(NotImplementedError,match='P31_CUDA_LONG_RUN_DEFERRED'):
            stability._check_release_stability_scope({'gpu_uuid':'GPU-'+'a'*16},{'duration_seconds':seconds})


def test_native_preflight_returns_controlled_unsupported_and_cli_exit_code(tmp_path):
    from hyso.cli.analysis import _code
    recipe=stability.WorkloadRecipe('release_scope:never',gpu_uuid='GPU-'+'a'*16)
    protocol=stability.StabilityProtocol(duration_seconds=7200,interval_seconds=1,max_samples=10000)
    source=SimpleNamespace(recipe=recipe,directory=str(tmp_path/'not-created'),options={'protocol':protocol})
    binding=SimpleNamespace(features=('P31',),source=source,device='cuda:0',policy=SimpleNamespace(allow_capture=True,allow_file_writes=True,allow_long_run=True,allow_cuda=True,allow_power_sampling=True))
    status,reasons=_runtime_admission(binding,'P31')
    assert status=='UNSUPPORTED' and reasons==['P31_CUDA_LONG_RUN_DEFERRED']
    assert _code([dict(selection='selected',status=status)],execution=True)==(4,'unsupported')
    assert not Path(source.directory).exists()
