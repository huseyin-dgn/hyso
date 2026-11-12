"""Optional offline validation of explicitly supplied saved native stage artifacts.

HYSO_STAGE10_SAVED_ROOT points at an existing Model-B/stages directory. These
tests read snapshots and never rerun their capture or hardware acceptance.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from hyso import analysis as api
from hyso.analysis._codec import read_json, EvidenceError
from test_native_execution import no_workloads, make_binding, execute, selected


FILES = {
    'P01': '03.cpu/B/latency.json', 'P02': '03.cpu/B/throughput.json',
    'P03': '03.cpu/B/pipeline.json', 'P04': '03.cpu/B/stages.json',
    'P05': '03.cpu/B/cold-start.json', 'P06': '03.cpu/B/batch-sweep.json',
    'P07': '03.cpu/B/resolution-sweep.json', 'P08': '03.cpu/B/precision-sweep.json',
    'P09': '04.cuda/B/benchmark.json', 'P10': '03.cpu/B/process-memory.json',
    'P11': '03.cpu/B/footprint.json', 'P12': '06.layer/operations.json',
    'P13': '06.layer/verified-cuda-modules.json', 'P14': '06.layer/operator-profile.json',
    'P15': '04.cuda/transfer-True-True.json', 'P16': '04.cuda/B/utilization.json',
    'P17': '03.cpu/B/capacity-fp32.json', 'P18': '06.roofline/roofline.json',
    'P19': '05.training/cpu/timing.json', 'P20': '05.training/cpu/stages.json',
    'P21': '05.training/cpu/loader.json', 'P22': '05.training/cpu/throughput.json',
    'P23': '05.training/cpu/isolated-memory.json', 'P24': '05.training/cpu/time-to-target.json',
    'P25': '05.training/cpu/gradient.json', 'P26': '03.cpu/B/quality.json',
    'P27': '03.cpu/B/robustness.json', 'P28': '04.cuda/B/quality-amp_fp16.json',
    'P29': '03.cpu/B/tradeoff-trial-0/fp32.json', 'P30': '08.video/stream.json',
    'P31': '07.energy/stability.json', 'P32': '07.energy/stability.json',
    'P33': '07.energy/energy.json', 'P34': '07.devices/B-comparison.json',
    'P35': '09.provider_cuda/B-provider.json',
}


@pytest.mark.parametrize('feature', FILES)
def test_saved_native_feature_offline(feature):
    location = os.environ.get('HYSO_STAGE10_SAVED_ROOT')
    if location is None:
        pytest.skip('explicit saved native stage directory required')
    path = Path(location) / FILES[feature]
    if feature == 'P14':
        # This ownership capture exceeds the established 64 MiB EvidenceRecord
        # boundary. Verify explicit rejection without parsing a 191 MB payload.
        assert path.stat().st_size > 64*1024**2
        with pytest.raises(EvidenceError, match='artifact_size_limit'):
            read_json(path, max_bytes=64*1024**2)
        return
    reference = api.ArtifactSource.from_path(path)
    payload, _, _ = read_json(path, reference.sha256)
    binding = make_binding(api.RetainedSource(payload), (feature,), artifacts=(reference,))
    row = selected(execute(binding))[feature]
    assert row['status'] == 'COMPLETED', (feature, row['reasons'])
    if feature != 'P32':
        assert row['evidence_record']['native_payload'] == payload
    else:
        assert row['evidence_record']['native_payload']['evidence']['stability'] == payload
    assert row['acceptance'] == 'NOT_EVALUATED'
    assert reference.verify()['status'] == 'verified'


@pytest.mark.parametrize('relative,feature', [
    ('04.cuda/B/pipeline.json', 'P04'),
    ('05.training/cuda-0/timing.json', 'P22'),
    ('05.training/cuda-0/isolated-memory.json', 'P23'),
    ('09.provider_cpu/B-provider.json', 'P35'),
])
def test_additional_saved_CUDA_and_ORT_families(relative, feature):
    location = os.environ.get('HYSO_STAGE10_SAVED_ROOT')
    if location is None:
        pytest.skip('explicit saved native stage directory required')
    payload, _, _ = read_json(Path(location) / relative)
    row = selected(execute(make_binding(api.RetainedSource(payload), (feature,))))[feature]
    assert row['status'] == 'COMPLETED', row['reasons']
