"""CPU API ergonomics and CLI previews; no implicit workloads or CUDA."""
import copy
import json

import pytest
import torch
from torch.utils.data import TensorDataset

from hyso import ModelBuilder, TrainingConfig, TrainingConfigError, Trainer
from hyso.cli import main


def dataset(shape=(1,4,4)):
    return TensorDataset(torch.ones(4,*shape),torch.tensor([0,1,0,1]))


def model():
    return ModelBuilder((1,4,4),2).add('conv2d',in_channels=1,out_channels=2).build()


def invoke(capsys,*args):
    code=main([*map(str,args),'--json'])
    captured=capsys.readouterr()
    return code,json.loads(captured.out),captured.err


def forbid_work(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('implicit workload or CUDA')
    monkeypatch.setattr(Trainer,'fit',forbidden)
    for name in ('init','_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    return forbidden


@pytest.mark.parametrize('validation',[False,True])
def test_dataset_shortcut_matches_prepared_session_without_fit(validation,monkeypatch):
    forbid_work(monkeypatch)
    config=TrainingConfig(epochs=1,batch_size=2,validation=validation)
    ds=dataset();val=dataset() if validation else None
    original=model();same=copy.deepcopy(original);before=torch.get_rng_state().clone()
    train_loader,val_loader=config.make_loaders(ds,val)
    old=config.prepare(original,train_loader,val_loader)
    new=config.prepare_datasets(same,ds,val)
    assert torch.equal(before,torch.get_rng_state())
    assert old.settings()==new.settings()
    assert new.train_loader.dataset is ds and new.trainer.model is same
    assert (new.val_loader is None) == (val is None)
    assert new.settings()['state']=='prepared'
    assert config.to_dict()['schema_version']==1
    for key,value in same.state_dict().items():assert torch.equal(value,original.state_dict()[key])


def test_dataset_shortcut_preserves_explicit_validation_and_model_placement(monkeypatch):
    forbid_work(monkeypatch)
    with pytest.raises(TrainingConfigError,match='validation'):
        TrainingConfig(batch_size=2).prepare_datasets(model(),dataset(),dataset())
    wrong=model().double()
    with pytest.raises(TrainingConfigError,match='device/dtype'):
        TrainingConfig(batch_size=2).prepare_datasets(wrong,dataset())
    assert next(wrong.parameters()).dtype==torch.float64


def test_dataset_shortcut_forwards_existing_session_options(monkeypatch):
    forbid_work(monkeypatch)
    cfg=TrainingConfig(batch_size=2);m=model();ds=dataset();sentinel=object();observed={}
    callback=lambda row:None
    def prepare(self,model,train,val,**kwargs):
        observed.update(model=model,train=train,val=val,options=kwargs)
        return sentinel
    monkeypatch.setattr(TrainingConfig,'prepare',prepare)
    kwargs=dict(model_config={'example':'metadata'},loss_fn={'a':1},loss_weights={'a':.5},on_epoch=callback)
    assert cfg.prepare_datasets(m,ds,**kwargs) is sentinel
    assert observed['model'] is m and observed['train'].dataset is ds
    assert observed['options']==kwargs


def test_known_shape_error_reports_context_and_separate_remedy(monkeypatch):
    forbid_work(monkeypatch)
    with pytest.raises(TrainingConfigError) as error:
        TrainingConfig(batch_size=2).prepare_datasets(model(),dataset((1,5,5)))
    assert 'train_loader supplies (1, 5, 5)' in str(error.value)
    assert 'expected (1, 4, 4)' in str(error.value)
    assert 'batch dimension excluded' in str(error.value)
    assert 'dataset sample shape' in error.value.suggestion


def test_modular_shape_error_only_claims_known_channels(monkeypatch):
    forbid_work(monkeypatch)
    m=ModelBuilder.modular(input_channels=3,num_classes=2,widths=(3,),repeats=(1,),strides=(1,),norm=None).build()
    with pytest.raises(TrainingConfigError) as error:
        TrainingConfig(batch_size=2).prepare_datasets(m,dataset())
    assert 'expected CHW with C=3' in str(error.value)
    assert 'supplies (1, 4, 4)' in str(error.value)


def test_validation_shape_error_labels_the_correct_loader(monkeypatch):
    forbid_work(monkeypatch)
    with pytest.raises(TrainingConfigError) as error:
        TrainingConfig(batch_size=2,validation=True).prepare_datasets(model(),dataset(),dataset((1,5,5)))
    assert 'val_loader supplies (1, 5, 5)' in str(error.value)


@pytest.mark.parametrize('group',['models','train','analysis','compare','advisor','reports','experiments'])
def test_group_help_gives_real_command_examples(group,capsys,monkeypatch):
    forbid_work(monkeypatch)
    assert main([group,'--help'])==0
    text=capsys.readouterr().out
    assert 'Examples:' in text and 'hyso '+group in text


@pytest.mark.parametrize('command',[['train','run','training.json'],['analysis','run','--features','P01'],['models','inspect','model.json']])
@pytest.mark.parametrize('flags',[['--execute','--dry-run'],['--dry-run','--execute']])
def test_conflicting_execution_flags_fail_before_any_file_or_factory_work(command,flags,capsys,monkeypatch):
    forbid_work(monkeypatch)
    code,envelope,error=invoke(capsys,*command,*flags)
    assert code==2 and envelope['status']=='error'
    assert 'not allowed with argument' in error
    assert 'Suggestion:' in error and '--help' in error
    assert 'Traceback' not in error


def test_explicit_training_preview_needs_no_execution_consent_and_never_imports(tmp_path,capsys,monkeypatch):
    forbid_work(monkeypatch)
    import hyso.cli.train as cli
    monkeypatch.setattr(cli,'load_factory',lambda *args:pytest.fail('preview imported a factory'))
    path=tmp_path/'training.json';TrainingConfig().to_json(path)
    code,value,_=invoke(capsys,'train','run',path,'--dry-run','--factory','absent:factory')
    assert code==0 and value['status']=='dry_run'
    assert value['result']['blockers']==['--trust-factory','--allow-training']
    assert not value['result']['execution_authorized'] and not value['result']['workload_started']
    # Existing default behavior and execution authorization checks stay intact.
    assert invoke(capsys,'train','run',path,'--factory','absent:factory')[0]==3
    assert invoke(capsys,'train','run',path,'--execute','--factory','absent:factory')[0]==3


def test_explicit_inspection_preview_never_builds(tmp_path,capsys,monkeypatch):
    forbid_work(monkeypatch)
    path=tmp_path/'model.json';ModelBuilder((1,4,4),2).to_json(path)
    monkeypatch.setattr(ModelBuilder,'build',lambda *args:pytest.fail('preview constructed model'))
    code,value,_=invoke(capsys,'models','inspect',path,'--dry-run')
    assert code==0 and value['status']=='dry_run'
    assert not value['result']['model_constructed']
    assert invoke(capsys,'models','inspect',path)[0]==3


def test_analysis_preview_preserves_missing_evidence_and_consent_gates(capsys,monkeypatch):
    forbid_work(monkeypatch)
    import hyso.cli.analysis as cli
    monkeypatch.setattr(cli,'load_factory',lambda *args:pytest.fail('preview imported a factory'))
    code,value,_=invoke(capsys,'analysis','run','--features','P01','--factory','absent:factory','--dry-run')
    assert code==3 and value['status']=='blocked'
    assert not value['result']['execution_authorized']
    assert invoke(capsys,'analysis','run','--features','P01','--dry-run','--authorize','P01')[0]==2


def test_errors_separate_remedy_preserve_debug_and_machine_schema(capsys,monkeypatch):
    forbid_work(monkeypatch)
    import hyso.cli.models as cli
    def fail(*args):raise RuntimeError('fixture failure without a known diagnosis')
    monkeypatch.setattr(cli,'run',fail)
    code,value,error=invoke(capsys,'models','list')
    assert code==1 and value['schema_version']==1 and value['error']=='fixture failure without a known diagnosis'
    assert 'Suggestion:' in error and 'Traceback' not in error
    assert 'shape' not in error and 'input format' not in error
    assert 'Traceback' in invoke(capsys,'--debug','models','list')[2]


def test_python_known_error_remedy_reaches_cli_without_traceback(capsys,monkeypatch):
    forbid_work(monkeypatch)
    import hyso.cli.models as cli
    def fail(*args):
        TrainingConfig(batch_size=2).prepare_datasets(model(),dataset((1,5,5)))
    monkeypatch.setattr(cli,'run',fail)
    code,value,error=invoke(capsys,'models','list')
    assert code==2 and 'expected (1, 4, 4)' in value['error']
    assert 'Suggestion: Match the dataset sample shape' in error and 'Traceback' not in error


@pytest.mark.parametrize('exception,expected_code',[(SystemExit(7),5),(KeyboardInterrupt(),130)])
@pytest.mark.parametrize('debug',[False,True])
def test_interruption_and_early_exit_keep_a_separate_remedy(exception,expected_code,debug,capsys,monkeypatch):
    import hyso.cli.models as cli
    def fail(*args):raise exception
    monkeypatch.setattr(cli,'run',fail)
    code,value,error=invoke(capsys,*(['--debug'] if debug else []),'models','list')
    assert code==expected_code and value['exit_code']==expected_code
    assert 'Suggestion:' in error and 'before retrying' in error
    assert ('Traceback' in error) is debug
