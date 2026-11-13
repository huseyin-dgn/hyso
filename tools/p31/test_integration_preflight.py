"""Source-overlay pinning checks only; never starts the long-run entry point."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import launch_p31 as launcher


@pytest.fixture
def pinned():
    manifest=dict(schema_version=2,commit=launcher.COMMIT,protocol=launcher.fixed_protocol(),
        repository=str(launcher.REPO),source_status=' M hyso/performance/devices.py',
        source_sha256={'hyso/performance/devices.py':'a'*64})
    def git(*args):
        return launcher.COMMIT if args[0]=='rev-parse' else manifest['source_status']
    with patch.object(launcher,'git',side_effect=git),patch.object(launcher,'source_inventory',
            return_value=copy.deepcopy(manifest['source_sha256'])):
        yield manifest


def test_no_commit_overlay_pins_exact_bytes(pinned):
    launcher.verify_source_snapshot(pinned)


@pytest.mark.parametrize('field',['schema_version','commit','protocol','repository','source_sha256'])
def test_changed_manifest_rejected(pinned,field):
    bad=copy.deepcopy(pinned)
    if field=='protocol':bad[field]['duration_seconds']=1
    elif field=='source_sha256':bad[field]['extra.py']='b'*64
    elif field=='schema_version':bad[field]=1
    else:bad[field]='changed'
    with pytest.raises(RuntimeError):launcher.verify_source_snapshot(bad)


def test_changed_git_status_rejected(pinned):
    with patch.object(launcher,'git',side_effect=lambda *a:launcher.COMMIT if a[0]=='rev-parse' else 'new file'):
        with pytest.raises(RuntimeError):launcher.verify_source_snapshot(pinned)


def test_changed_source_bytes_rejected(pinned):
    with patch.object(launcher,'source_inventory',return_value={'hyso/performance/devices.py':'b'*64}):
        with pytest.raises(RuntimeError):launcher.verify_source_snapshot(pinned)


def test_fixed_original_long_protocol_and_fresh_external_output(tmp_path):
    p=launcher.fixed_protocol()
    assert (p['duration_seconds'],p['interval_seconds'],p['chunk_rows'],p['max_samples'])==(7200,5,12,2000)
    assert all(p[k] is None for k in ('process_memory_bytes','job_memory_bytes','max_processes'))
    with pytest.raises(RuntimeError):launcher.configure_output(launcher.REPO/'evidence')
    with pytest.raises(RuntimeError):launcher.configure_output(tmp_path)
