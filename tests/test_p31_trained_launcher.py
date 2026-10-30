"""Fault injection for the reviewed P31 launcher; no long/native model workload."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOL=Path(__file__).resolve().parents[1]/'tools/p31/trained_acceptance.py'
sys.path.insert(0,str(TOOL.parent))
spec=importlib.util.spec_from_file_location('trained_launcher_under_test',TOOL)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_no_explicit_manifest_hash_cannot_execute(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['trained_acceptance.py','start','--audit',str(tmp_path)])
    monkeypatch.setattr(launcher,'execute',lambda *a:pytest.fail('unexpected execution'))
    monkeypatch.setattr(launcher,'verify',lambda *a:pytest.fail('unexpected verification before required argument'))
    with pytest.raises(SystemExit):launcher.main()
    assert not (tmp_path/'STARTED.json').exists()


def test_mismatched_manifest_hash_is_rejected_without_factory(tmp_path):
    (tmp_path/'long-run-manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='SHA256 mismatch'):launcher.verify(tmp_path,'0'*64)


def test_one_shot_marker_prevents_observer_or_retry(monkeypatch,tmp_path):
    (tmp_path/'STARTED.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['trained_acceptance.py','start','--audit',str(tmp_path),'--approved-sha256','a'*64])
    monkeypatch.setattr(launcher,'verify',lambda *a:{'outer_watchdog_seconds':7440})
    monkeypatch.setattr(launcher,'execute',lambda *a:pytest.fail('unexpected retry'))
    with pytest.raises(FileExistsError,match='one-shot'):launcher.main()


def test_preflight_has_no_execution_side_effects(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(sys,'argv',['trained_acceptance.py','preflight','--audit',str(tmp_path),'--approved-sha256','a'*64])
    monkeypatch.setattr(launcher,'verify',lambda *a:{})
    monkeypatch.setattr(launcher,'execute',lambda *a:pytest.fail('preflight executed a workload'))
    launcher.main();receipt=json.loads(capsys.readouterr().out)
    assert receipt['owned_workers_started']==receipt['factory_calls']==receipt['CUDA_queries']==0
    assert not receipt['long_run_started'] and list(tmp_path.iterdir())==[]


def test_export_uses_one_detached_snapshot_and_preserves_nulls(tmp_path):
    class Result:
        calls=0
        def to_dict(self):
            self.calls+=1
            return dict(evidence=dict(header=dict(clock_id='clock'),rows=[dict(timestamp_ns=10,rss=None,allocator=None,
                process_commit=dict(value=None,reason='missing',query_start_ns=8,query_end_ns=9))]))
        def to_json(self,path):path.write_text('{}')
        def to_csv(self,path):path.write_text('fixture')
        def stability_diagnostics(self):return {}
    r=Result();launcher.export(r,tmp_path)
    assert r.calls==1 and 'missing' in (tmp_path/'memory.csv').read_text()


def test_export_failure_retains_final_receipt_and_never_retries(tmp_path,monkeypatch):
    from hyso import performance
    (tmp_path/'trained-workload.json').write_text('{}');calls=[]
    def measured(*args,**kwargs):calls.append('measure');return object()
    def fail_export(*args):raise OSError('controlled export failure')
    monkeypatch.setattr(performance,'measure_stability',measured)
    monkeypatch.setattr(launcher,'export',fail_export)
    with pytest.raises(OSError,match='controlled export failure'):launcher.execute(tmp_path,{},'short')
    d=json.loads((tmp_path/'short/launcher-final.json').read_text())
    assert calls==['measure'] and d['status']=='INCOMPLETE' and not d['automatic_retry']
    assert d['phase']=='export' and d['error']['type']=='OSError'
