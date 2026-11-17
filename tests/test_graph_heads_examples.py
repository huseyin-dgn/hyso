"""Public multi-head examples and the distribution import closure, CPU only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
import torch
from hyso.training import resume
from hyso.reporting import PresentationReport
from examples.models.graph_evidence_workflow import train_heads, compare_saved_heads
from test_graph_heads_persistence import prohibit_workloads
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def test_two_head_example_resume_store_and_saved_pair(tmp_path,monkeypatch):
    save=resume.save_last
    def interrupt(*args):
        save(*args)
        raise InterruptedError()
    with monkeypatch.context() as scoped:
        scoped.setattr(resume,'save_last',interrupt)
        with pytest.raises(InterruptedError):train_heads(tmp_path/'interrupted')
    a=train_heads(tmp_path/'A',resume_from=tmp_path/'interrupted/last.pth')
    complete=train_heads(tmp_path/'complete')
    assert a['history']==complete['history'] and a['applied_updates']==4
    assert a['history']['train_acc']==[None,None]
    assert a['imported_graph_trust']=='unverified'
    assert set(a['per_head_metrics'])=={'disease','severity'}
    assert all(m['sample_count']==4 for m in a['per_head_metrics'].values())
    state=torch.load(a['checkpoint'],weights_only=True)
    assert state['format_version']==2
    assert_state(state['model'],torch.load(complete['checkpoint'],weights_only=True)['model'])
    b=train_heads(tmp_path/'B',variant='B')
    prohibit_workloads(monkeypatch)
    summary=compare_saved_heads([tmp_path/'A/heads.json',tmp_path/'B/heads.json'],tmp_path/'reports',
        head_pairs={'disease':'disease','severity':'severity'})
    for head,row in summary.items():
        assert row['structure']=='COMPARABLE'
        assert all(r['status']=='COMPARABLE' for r in row['metrics'] if r['feature_id']=='P26')
        assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None
                   for r in row['metrics'] if r['feature_id']=='P12')
    for language in ('tr','en'):
        path=tmp_path/'reports'/language
        report=PresentationReport.from_json(path.with_suffix('.json')).to_dict()
        assert report['presentation_only'] and not report['authoritative_evidence']
        assert len([s for s in report['sources'] if s['kind']=='graph_multi_head_evaluation'])==2
        for suffix in ('.md','.html'):
            text=path.with_suffix(suffix).read_text(encoding='utf8')
            assert all(marker in text for marker in ('disease','severity','INSUFFICIENT_EVIDENCE','COMPARABLE'))


def test_two_input_example_states_actual_unsupported_boundaries(tmp_path):
    code="""
import runpy, sys, torch
def forbidden(*args,**kwargs):raise AssertionError('Unexpected CUDA work')
torch.cuda.is_available=lambda:False
torch.cuda.init=forbidden;torch.cuda._lazy_init=forbidden;torch.cuda.device_count=forbidden
output=sys.argv[1]
sys.argv=['graph_evidence_workflow','--multi-head','--two-inputs','--output',output]
runpy.run_module('examples.models.graph_evidence_workflow',run_name='__main__')
assert not torch.cuda.is_initialized()
"""
    completed=subprocess.run([sys.executable,'-c',code,str(tmp_path/'named')],
        capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stdout+completed.stderr
    result=json.loads(completed.stdout)
    assert result['applied_updates']==4 and result['checkpoint'] is None
    assert result['per_head_metrics'] is None and 'evaluation' in result['unsupported']
    assert not (tmp_path/'named/last.pth').exists()
    with pytest.raises(ValueError,match='two-input multi-loss resume is unsupported'):
        train_heads(tmp_path/'unsupported',two_inputs=True,resume_from=tmp_path/'named/last.pth')
    assert not (tmp_path/'unsupported').exists()


def test_allowlisted_public_imports_do_not_depend_on_checkout_modules(tmp_path):
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/'commercial-profile.json').read_text(encoding='utf8'))
    distribution=tmp_path/'distribution'
    for name in profile['python_files']:
        target=distribution/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    code="""
import inspect, pathlib, sys, torch
root=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
def forbidden(*args,**kwargs):raise AssertionError('Unexpected CUDA initialization')
torch.cuda.is_available=lambda:False
torch.cuda.init=forbidden;torch.cuda._lazy_init=forbidden;torch.cuda.device_count=forbidden
from hyso import TrainingConfig, Trainer, analysis
from hyso.training import NamedTensorDataset, NamedTargetDataset
from hyso.analysis import MultiHeadGraphEvaluationResult, evaluate_graph_heads, compare_graph_heads
assert 'loss_fn' in inspect.signature(TrainingConfig.prepare).parameters
assert pathlib.Path(inspect.getfile(NamedTargetDataset)).resolve().is_relative_to(root)
assert pathlib.Path(inspect.getfile(evaluate_graph_heads)).resolve().is_relative_to(root)
assert not torch.cuda.is_initialized()
"""
    completed=subprocess.run([sys.executable,'-I','-c',code,str(distribution)],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stdout+completed.stderr
