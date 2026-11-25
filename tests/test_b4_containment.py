"""B4 containment: no native GPU work, no guard-removing diagnostic escape."""
import copy
from dataclasses import asdict
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hyso import analysis as api,performance as perf
from hyso.performance import isolated_training_memory as isolated
from hyso.performance._saved_activation import _SavedActivationRecorder

ROOT=Path(__file__).resolve().parents[1]
BLOCKED='BLOCKED_B4_UNRESOLVED'


def forbid(*a,**kw):raise AssertionError('B4 containment crossed a forbidden execution boundary')


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    for name in ('_lazy_init','init','is_available','device_count','set_device','synchronize','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,forbid)


def request(device='cuda:0',saved=True):
    recipe=perf.TrainingMemoryRecipe('absent:factory',device=device)
    protocol=perf.IsolatedTrainingMemoryProtocol(saved_activation_lifetime=saved,
        memory_protocol=perf.TrainingMemoryProtocol(device=device))
    return recipe,protocol


@pytest.mark.parametrize('device',['cuda:0','cuda:1','cuda:17'])
@pytest.mark.parametrize('override',[None,'HYSO_ALLOW_CUDA_SAVED_ACTIVATION','HYSO_B4_RESOLVED','HYSO_DEVELOPMENT'])
def test_public_capture_denied_before_worker_queries_or_tempfiles(device,override,monkeypatch):
    recipe,protocol=request(device)
    if override:monkeypatch.setenv(override,'1')
    monkeypatch.setattr(isolated,'_launch',forbid)
    monkeypatch.setattr(isolated.tempfile,'TemporaryDirectory',forbid)
    monkeypatch.setattr(torch.cuda,'is_initialized',forbid)
    monkeypatch.setattr(isolated,'_activation_adapter',forbid)
    data=perf.measure_training_memory_isolated(recipe,protocol=protocol).to_dict()
    assert data['outcome']=='unsupported' and BLOCKED in data['reason']
    assert data['measurement'] is data['receipt'] is data['worker_receipt'] is None
    assert perf.IsolatedTrainingMemoryResult(data).to_dict()==data


def test_worker_gate_precedes_CUDA_state_query_and_runtime_setup(monkeypatch):
    recipe,protocol=request()
    monkeypatch.setattr(torch.cuda,'is_initialized',forbid)
    monkeypatch.setattr(isolated,'_SavedActivationRecorder',forbid)
    with pytest.raises(NotImplementedError,match=BLOCKED):
        isolated._run_recipe(dict(recipe=recipe.to_dict(),protocol=asdict(protocol),torch_version=str(torch.__version__)))


@pytest.mark.parametrize('entry',['constructor','pack'])
def test_direct_recorder_alternative_entry_cannot_accept_CUDA(entry):
    tensor=SimpleNamespace(device=torch.device('cuda:0'))
    if entry=='constructor':
        model=SimpleNamespace(parameters=lambda:[tensor],buffers=lambda:[])
        with pytest.raises(NotImplementedError,match=BLOCKED):_SavedActivationRecorder(model,8,32)
    else:
        recorder=_SavedActivationRecorder(torch.nn.Linear(2,2),8,32)
        with pytest.raises(NotImplementedError,match=BLOCKED):recorder.pack(tensor)
        assert not recorder.records and not recorder.events and not recorder.slots


def binding(device='cuda:0',saved=True):
    recipe,protocol=request(device,saved)
    return api.NativeBinding('b4-test',('A',),('P23',),api.RecipeSource(recipe,options=dict(protocol=protocol)),
        device=device,policy=api.OperationPolicy(allow_capture=True,allow_training=True,allow_cuda=True))


@pytest.mark.parametrize('allow_cuda',[False,True])
def test_analyzer_preflight_and_authorized_execution_keep_B4_reason(allow_cuda,monkeypatch):
    from dataclasses import replace
    b=binding();b=replace(b,policy=api.OperationPolicy(allow_capture=True,allow_training=True,allow_cuda=allow_cuda))
    monkeypatch.setattr(perf,'measure_training_memory_isolated',forbid)
    plan=api.plan_analysis(api.ModelContext('A'),features='P23',device='cuda:0',execution_mode='native')
    review=api.preflight_analysis(plan,(b,))
    row=next(r for r in review.to_dict()['features'] if r['feature_id']=='P23')
    assert row['status']=='UNSUPPORTED' and BLOCKED in row['reasons']
    result=api.execute_analysis(review,authorized_features=('P23',),explicit_authorization=True)
    row=next(r for r in result.to_dict()['features'] if r['feature_id']=='P23')
    assert row['status']=='UNSUPPORTED' and BLOCKED in row['reasons'] and row['evidence_record'] is None


def test_cli_cannot_authorize_the_disabled_measurement(monkeypatch,capsys):
    from hyso.cli import main
    import hyso.cli.analysis as cli
    monkeypatch.setattr(cli,'load_factory',lambda name:lambda **kw:[binding()])
    monkeypatch.setattr(perf,'measure_training_memory_isolated',forbid)
    code=main(['analysis','run','--features','P23','--model-id','A','--device','cuda:0',
        '--factory','unused:factory','--trust-factory','--execute','--authorize','P23','--allow-capture',
        '--allow-training','--allow-cuda','--json'])
    payload=json.loads(capsys.readouterr().out)
    assert code==4 and BLOCKED in json.dumps(payload)
    assert all(r['evidence_record'] is None for r in payload['result']['features'] if r['feature_id']=='P23')


@pytest.mark.parametrize('device,saved',[('cpu',False),('cpu',True),('cuda:0',False)])
def test_supported_scope_is_not_blanket_disabled(device,saved,monkeypatch):
    class ReachedOwnedLauncher(Exception):pass
    def launcher(*a,**kw):raise ReachedOwnedLauncher
    recipe,protocol=request(device,saved)
    monkeypatch.setattr(isolated,'_launch',launcher)
    with pytest.raises(ReachedOwnedLauncher):perf.measure_training_memory_isolated(recipe,protocol=protocol)


def test_failed_capture_is_not_zero_or_success_and_old_envelopes_remain_readable(tmp_path,monkeypatch):
    recipe,protocol=request();monkeypatch.setattr(isolated,'_launch',forbid)
    value=perf.measure_training_memory_isolated(recipe,protocol=protocol)
    assert value.diagnostics()['core_status']=='not_executed'
    assert value.basic_metrics()['status']=='not_executed'
    assert all(m['value'] is None and m['status']=='unavailable' and BLOCKED in m['reason']
        for m in value.basic_metrics()['metrics'].values())
    value.to_json(tmp_path/'blocked.json')
    assert perf.IsolatedTrainingMemoryResult(json.loads((tmp_path/'blocked.json').read_text())).to_dict()==value.to_dict()
    data=value.to_dict();data['reason']='archived native failure 0xC0000409'
    data['sha256']=isolated._hash({k:v for k,v in data.items() if k!='sha256'})
    assert perf.IsolatedTrainingMemoryResult(data).to_dict()==data
    data['outcome']='success';data['reason']=None
    data['sha256']=isolated._hash({k:v for k,v in data.items() if k!='sha256'})
    with pytest.raises(ValueError):perf.IsolatedTrainingMemoryResult(data)


@pytest.fixture
def forensic(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools/forensics'))
    import p23_forensic_runner as runner
    return runner


@pytest.mark.parametrize('target,filename',[('recipe','isolated_training_memory.py'),('saved','_saved_activation.py')])
def test_generated_diagnostics_retain_all_production_guards(forensic,target,filename):
    source=ROOT/'hyso/performance'/filename;before=source.read_text(encoding='utf-8')
    candidate=forensic.transform(source,target)
    assert candidate.count(BLOCKED)==before.count(BLOCKED)>0
    assert 'PRIVATE P23 diagnostic copy only' not in candidate
    if target=='saved':assert candidate==before
    else:assert candidate==before.replace('from ._saved_activation import ','from ._p23_diagnostic_saved import ')


@pytest.mark.parametrize('entry',['spawn','supervise','controller','internal','worker','cli'])
def test_forensic_native_entry_refused_before_file_reads_spawn_or_admission(forensic,entry,monkeypatch):
    monkeypatch.setattr(forensic.subprocess,'Popen',forbid)
    monkeypatch.setattr(forensic,'verify',forbid)
    monkeypatch.setattr(forensic,'approval',forbid)
    monkeypatch.setattr(forensic.admission,'collect_p23',forbid)
    if entry=='cli':
        monkeypatch.setattr(sys,'argv',['p23_forensic_runner','--mode','native'])
        with pytest.raises(SystemExit) as e:forensic.main()
        assert e.value.code==2
        return
    with pytest.raises(NotImplementedError,match=BLOCKED):
        if entry=='spawn':forensic.spawn(None,None,None,'native',None,None,None)
        elif entry=='supervise':forensic.supervise(None,None,'native',None,None)
        elif entry=='controller':forensic.controller(None,None,'native',None)
        elif entry=='internal':forensic.internal(SimpleNamespace(mode='native'))
        else:
            import p23_forensic_worker as worker
            worker.run(None,None,'native',None,None,{})


def cpu_workload(*,device,state_path=None):
    """Owned tiny SiLU+Dropout fixture, no source/weight changes to user models."""
    if device!='cpu':raise ValueError('CPU-only containment regression fixture')
    # PyTorch AdamW checks availability even on CPU; use a mock instead of a
    # driver query, and forbid initialization inside the owned interpreter too.
    torch.cuda.is_available=lambda:False
    torch.cuda._lazy_init=forbid;torch._C._cuda_init=forbid
    from hyso.training import Trainer
    from torch.utils.data import TensorDataset,DataLoader
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.SiLU(),torch.nn.Dropout(.2),torch.nn.Linear(4,2)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    trainer=Trainer(model,optimizer=optimizer,loss_fn=torch.nn.CrossEntropyLoss(),device=device,amp=False,accum_steps=2)
    data=TensorDataset(torch.linspace(-1,1,15).reshape(5,3),torch.tensor([0,1,0,1,0]))
    if state_path:
        from training_memory_factories import state_digest
        original=trainer.fit
        def fit(*a,**kw):
            result=original(*a,**kw)
            Path(state_path).write_text(json.dumps(dict(state_sha256=state_digest(trainer))),encoding='utf-8')
            return result
        trainer.fit=fit
    return dict(trainer=trainer,train_loader=DataLoader(data,batch_size=2,shuffle=True),val_loader=DataLoader(data,batch_size=2))


def test_CPU_SiLU_Dropout_owned_hooks_preserve_training_weights_and_parent_RNG(tmp_path):
    from training_memory_factories import state_digest
    random.seed(71);np.random.seed(71);torch.manual_seed(71)
    baseline=cpu_workload(device='cpu')
    baseline['trainer'].fit(baseline['train_loader'],baseline['val_loader'],epochs=2)
    expected=state_digest(baseline['trainer'])
    weights={k:v.clone() for k,v in baseline['trainer'].model.state_dict().items()}
    parent_rng=copy.deepcopy((random.getstate(),np.random.get_state(),torch.get_rng_state()))
    path=tmp_path/'worker-state.json'
    recipe=perf.TrainingMemoryRecipe('test_b4_containment:cpu_workload',factory_search_path=ROOT/'tests',seed=71,epochs=2,
        kwargs=dict(state_path=str(path)))
    protocol=perf.IsolatedTrainingMemoryProtocol(saved_activation_lifetime=True)
    value=perf.measure_training_memory_isolated(recipe,protocol=protocol);data=value.to_dict()
    assert data['outcome']=='success',data['reason']
    assert json.loads(path.read_text())['state_sha256']==expected
    assert all(torch.equal(v,baseline['trainer'].model.state_dict()[k]) for k,v in weights.items())
    assert random.getstate()==parent_rng[0] and np.array_equal(np.random.get_state()[1],parent_rng[1][1]) and torch.equal(torch.get_rng_state(),parent_rng[2])
    summary=data['measurement']['activation']['summary']
    assert summary['pack_events']==summary['release_events']>0 and summary['live_payloads_at_end']==0
    assert data['receipt']['admitted'] is True and data['receipt']['job']['active_processes']==0
    assert data['measurement']['activation']['physical_activation_peak']['value'] is None


@pytest.mark.parametrize('bootstrap',['training','stability'])
def test_owned_bootstrap_refuses_foreign_admission_before_import(bootstrap,tmp_path,monkeypatch):
    from hyso.performance import _training_memory_worker,_runtime_worker
    module=_training_memory_worker if bootstrap=='training' else _runtime_worker
    (tmp_path/'request.json').write_text(json.dumps(dict(token='b4-test',mode='stability')),encoding='utf-8')
    (tmp_path/'admit.json').write_text(json.dumps(dict(token='foreign',pid=0)),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['bootstrap',str(tmp_path)])
    monkeypatch.setattr(module.importlib if bootstrap=='training' else isolated.importlib,'import_module',forbid)
    with pytest.raises((ValueError,RuntimeError),match='admission'):module.main()
    assert not (tmp_path/'ready.json').exists() and not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('language',['tr','en','de','es'])
def test_blocked_execution_is_visible_in_Markdown_and_HTML_without_fake_metrics(language,monkeypatch):
    from hyso.reporting import create_report
    monkeypatch.setattr(perf,'measure_training_memory_isolated',forbid)
    plan=api.plan_analysis(api.ModelContext('A'),features='P23',device='cuda:0',execution_mode='native')
    result=api.execute_analysis(api.preflight_analysis(plan,(binding(),)),authorized_features=('P23',),explicit_authorization=True)
    context=api.AnalysisContext(workload_id='b4-disabled',task='classification',model={'id':'A'},dataset={})
    for level in ('beginner','technical','expert'):
        report=create_report(result,context=context,language=language,level=level)
        md=report.to_markdown();html=report.to_html()
        assert '| A | P23 |' in md and BLOCKED in md.split('```')[0]
        assert BLOCKED in html and '<td>P23</td>' in html
    records=result.to_evidence_index().to_dict()['records']
    assert all(r['native_payload'] is None and not r['metrics'] for r in records)
    p23=next(r for r in records if r['feature_id']=='P23')
    assert p23['validation']['status']=='unsupported' and BLOCKED in p23['validation']['reason']
