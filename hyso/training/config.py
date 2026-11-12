"""Versioned training settings only; no dataset, model, callable or code loading."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from .session import TrainingSession


class TrainingConfigError(ValueError):
    """Invalid/unsupported settings or a preparation contract mismatch."""


def _plain(v, depth=0):
    if depth > 8: raise TrainingConfigError('settings nesting exceeds 8')
    if v is None or type(v) in (str, bool, int):
        if type(v) is str and len(v) > 4096: raise TrainingConfigError('settings string too long')
        return v
    if type(v) is float and math.isfinite(v): return v
    if type(v) in (tuple,list) and len(v) <= 32: return [_plain(x,depth+1) for x in v]
    if type(v) is dict and len(v) <= 64 and all(type(k) is str for k in v):
        return {k:_plain(x,depth+1) for k,x in v.items()}
    raise TrainingConfigError('settings require bounded plain finite JSON values')


def _int(v, name, minimum=1):
    if type(v) is not int or not minimum <= v <= 2**31-1:
        raise TrainingConfigError(f'{name} must be an integer >= {minimum}')


def _number(v, name, minimum=0, *, positive=False):
    try: valid=type(v) in (int,float) and math.isfinite(v) and v>=minimum and not (positive and v==0)
    except OverflowError: valid=False
    if not valid:
        raise TrainingConfigError(f'{name} must be finite and {">" if positive else ">="} {minimum}')


def _options(given, defaults, name):
    if given is None: given = {}
    if type(given) is not dict or set(given)-set(defaults):
        raise TrainingConfigError(f'unknown/invalid {name} options')
    return dict(defaults, **given)


def _resolved(c):
    """Validate and resolve profile defaults without hardware queries or allocation."""
    for key in ('epochs','batch_size','accum_steps'): _int(c[key],key)
    if c['num_workers'] != 0 or type(c['num_workers']) is not int:
        raise TrainingConfigError('this configuration profile supports num_workers=0 only')
    for key in ('validation','shuffle','drop_last'):
        if type(c[key]) is not bool: raise TrainingConfigError(f'{key} must be bool')
    if c['task'] not in ('multiclass','binary'): raise TrainingConfigError('task must be multiclass or binary')
    if type(c['device']) is not str or not (c['device']=='cpu' or re.fullmatch(r'cuda:(0|[1-9][0-9]*)',c['device'])):
        raise TrainingConfigError('device must be cpu or an explicit cuda:index')
    if c['precision'] not in ('fp32','fp64','amp_fp16'): raise TrainingConfigError('unsupported precision')
    if c['precision']=='amp_fp16' and c['device']=='cpu': raise TrainingConfigError('amp_fp16 requires CUDA; CPU AMP is unsupported')
    _number(c['learning_rate'],'learning_rate',positive=True)
    if c['grad_clip'] is not None: _number(c['grad_clip'],'grad_clip',positive=True)
    if c['ema_decay'] is not None:
        _number(c['ema_decay'],'ema_decay')
        if c['ema_decay'] > 1: raise TrainingConfigError('ema_decay must be <=1')
    if c['optimizer']=='sgd':
        opt=_options(c['optimizer_params'],dict(momentum=0,dampening=0,weight_decay=0,nesterov=False),'SGD')
        for k in ('momentum','dampening','weight_decay'): _number(opt[k],k)
        if type(opt['nesterov']) is not bool: raise TrainingConfigError('nesterov must be bool')
        if opt['nesterov'] and (opt['momentum']<=0 or opt['dampening']!=0):
            raise TrainingConfigError('Nesterov requires momentum>0 and dampening=0')
    elif c['optimizer'] in ('adam','adamw'):
        opt=_options(c['optimizer_params'],dict(betas=[.9,.999],eps=1e-8,weight_decay=0),'Adam/AdamW')
        if type(opt['betas']) is not list or len(opt['betas'])!=2: raise TrainingConfigError('betas must contain two numbers')
        for b in opt['betas']:
            _number(b,'beta')
            if b>=1: raise TrainingConfigError('betas must be <1')
        _number(opt['eps'],'eps',positive=True);_number(opt['weight_decay'],'weight_decay')
    else: raise TrainingConfigError('optimizer must be sgd, adam or adamw')
    scheduler=c['scheduler']
    if scheduler is None:
        if c['scheduler_params'] not in (None,{}): raise TrainingConfigError('scheduler_params requires a scheduler')
        sched={}
    elif scheduler=='step':
        sched=_options(c['scheduler_params'],dict(step_size=1,gamma=.1),'StepLR')
        _int(sched['step_size'],'step_size');_number(sched['gamma'],'gamma')
    elif scheduler=='cosine':
        sched=_options(c['scheduler_params'],dict(t_max=c['epochs'],eta_min=0.),'CosineAnnealingLR')
        _int(sched['t_max'],'t_max');_number(sched['eta_min'],'eta_min')
    elif scheduler=='onecycle':
        sched=_options(c['scheduler_params'],dict(max_lr=None,pct_start=.3,div_factor=25.,final_div_factor=1e4,cycle_momentum=False),'OneCycleLR')
        for k in ('max_lr','pct_start','div_factor','final_div_factor'): _number(sched[k],k,positive=True)
        if sched['pct_start']>=1: raise TrainingConfigError('pct_start must be <1')
        if type(sched['cycle_momentum']) is not bool: raise TrainingConfigError('cycle_momentum must be bool')
    elif scheduler=='plateau':
        sched=_options(c['scheduler_params'],dict(factor=.1,patience=10,threshold=1e-4,cooldown=0,min_lr=0,eps=1e-8),'ReduceLROnPlateau')
        _int(sched['patience'],'patience',0);_int(sched['cooldown'],'cooldown',0)
        for k in ('factor','threshold','min_lr','eps'): _number(sched[k],k)
        if sched['factor']>=1: raise TrainingConfigError('plateau factor must be <1')
        if not c['validation']: raise TrainingConfigError('plateau requires validation=True')
    else: raise TrainingConfigError('scheduler must be None, step, cosine, onecycle or plateau')
    early=None
    if c['early_stopping'] is not None:
        early=_options(c['early_stopping'],dict(patience=7,delta=0.,restore_best_weights=True),'early_stopping')
        _int(early['patience'],'early-stopping patience');_number(early['delta'],'delta')
        if type(early['restore_best_weights']) is not bool: raise TrainingConfigError('restore_best_weights must be bool')
        if not c['validation']: raise TrainingConfigError('early stopping requires validation=True')
    for k in ('save_last_to','resume_from'):
        if c[k] is not None and (type(c[k]) is not str or not c[k].strip()): raise TrainingConfigError(f'{k} must be a nonempty path string or None')
    return opt,sched,early


@dataclass(frozen=True,init=False)
class TrainingConfig:
    """Immutable schema1 request, separate from model architecture and data.

    Args:
        epochs: Positive total budget, including restored epochs.
        batch_size: Positive batch size; supplied loaders must match exactly.
        optimizer: sgd, adam or adamw; standard native PyTorch implementations.
        learning_rate: Positive optimizer-construction LR, not necessarily a scheduled LR.
        optimizer_params: Whitelisted native options; default weight_decay is explicitly 0.
        scheduler: None, step, cosine, onecycle or plateau (validation loss, min).
        scheduler_params: Whitelisted settings; OneCycle requires max_lr. Effective
            initial LR and resolved defaults are exposed by session.settings().
        device: cpu or explicit cuda:index; caller moves the model before prepare.
        precision: fp32, fp64 or CUDA amp_fp16 (FP32 master weights).
        accum_steps: Positive Trainer gradient-accumulation count.
        grad_clip: Optional positive gradient-norm bound.
        ema_decay: Optional [0,1] EMA decay.
        early_stopping: Optional patience/delta/restore_best_weights dictionary.
        validation: Require a matching validation loader when True.
        shuffle: Standard RandomSampler when True, SequentialSampler otherwise.
        drop_last: Explicit training-loader policy; validation never drops samples.
        num_workers: Zero in this v1 user-experience profile.
        task: multiclass CE or binary BCEWithLogits classification.
        save_last_to: Explicit checkpoint destination, activated only by fit().
        resume_from: Explicit checkpoint source; original Trainer validates restore.

    JSON contains no model, dataset, loader, callable, optimizer object or code.
    Loading settings never runs training or authorizes accelerator use. No seed,
    dtype cast or model movement is silently applied. Use torch.manual_seed yourself.
    """
    _payload: str

    def __init__(self, *, epochs: int=5, batch_size: int=32, optimizer: str='adamw',
                 learning_rate: float=1e-3, optimizer_params: dict[str,Any] | None=None,
                 scheduler: str | None=None, scheduler_params: dict[str,Any] | None=None,
                 device: str='cpu', precision: str='fp32', accum_steps: int=1,
                 grad_clip: float | None=None, ema_decay: float | None=None,
                 early_stopping: dict[str,Any] | None=None, validation: bool=False,
                 shuffle: bool=False, drop_last: bool=False, num_workers: int=0,
                 task: str='multiclass', save_last_to: str | None=None, resume_from: str | None=None) -> None:
        c=_plain(dict(epochs=epochs,batch_size=batch_size,optimizer=optimizer,learning_rate=learning_rate,
            optimizer_params=optimizer_params,scheduler=scheduler,scheduler_params=scheduler_params,
            device=device,precision=precision,accum_steps=accum_steps,grad_clip=grad_clip,ema_decay=ema_decay,
            early_stopping=early_stopping,validation=validation,shuffle=shuffle,drop_last=drop_last,
            num_workers=num_workers,task=task,save_last_to=save_last_to,resume_from=resume_from))
        _resolved(c)
        object.__setattr__(self,'_payload',json.dumps(dict(format='hyso.training_config',schema_version=1,**c),sort_keys=True,allow_nan=False))

    def to_dict(self) -> dict[str,Any]:
        """Return a detached request, including user-supplied option dictionaries."""
        return json.loads(self._payload)

    @classmethod
    def from_dict(cls, data: dict[str,Any]) -> TrainingConfig:
        """Load exact schema1 fields; unknown fields/objects are rejected."""
        d=_plain(data)
        if type(d) is not dict or d.get('format')!='hyso.training_config' or type(d.get('schema_version')) is not int or d['schema_version']!=1:
            raise TrainingConfigError('unsupported training configuration format/version')
        if set(d)!=set(cls().to_dict()): raise TrainingConfigError('unknown/missing training configuration fields')
        del d['format'];del d['schema_version']
        return cls(**d)

    def to_json(self, path: str | Path, *, overwrite: bool=False) -> None:
        """Write settings only; refuse an existing file by default."""
        if type(overwrite) is not bool: raise TrainingConfigError('overwrite must be bool')
        with Path(path).open('w' if overwrite else 'x',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(self.to_dict(),indent=2,allow_nan=False)+'\n')

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        """Read <=64 KiB UTF-8 JSON; reject duplicates/nonfinite values, never code."""
        with Path(path).open('rb') as f: raw=f.read(65537)
        if len(raw)>65536: raise TrainingConfigError('training settings exceed 64 KiB')
        def pairs(items):
            d={}
            for k,v in items:
                if k in d: raise TrainingConfigError(f'duplicate key: {k}')
                d[k]=v
            return d
        def invalid(v): raise TrainingConfigError(f'nonfinite number: {v}')
        try: return cls.from_dict(json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=invalid))
        except (ValueError,TypeError,RecursionError) as error: raise TrainingConfigError(str(error)) from error

    def make_loaders(self, train_dataset: Dataset, val_dataset: Dataset | None=None) -> tuple[DataLoader,DataLoader | None]:
        """Construct ordinary finite CPU-side loaders; do not iterate or seed them."""
        from torch.utils.data import Dataset,IterableDataset,DataLoader
        c=self.to_dict()
        if c['validation'] != (val_dataset is not None): raise TrainingConfigError('validation flag and validation dataset must agree')
        for ds in (train_dataset,val_dataset):
            if ds is not None and (not isinstance(ds,Dataset) or isinstance(ds,IterableDataset) or len(ds)<1):
                raise TrainingConfigError('finite nonempty map-style Dataset required')
        return (DataLoader(train_dataset,batch_size=c['batch_size'],shuffle=c['shuffle'],drop_last=c['drop_last'],num_workers=0),
                None if val_dataset is None else DataLoader(val_dataset,batch_size=c['batch_size'],shuffle=False,drop_last=False,num_workers=0))

    def prepare(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None=None, *,
                model_config: dict[str,Any] | None=None,
                on_epoch: Callable[[dict[str,Any]],None] | None=None) -> TrainingSession:
        """Validate before updates, create native optimizer/scheduler and one Trainer.

        Model config is separate; required for the existing ModularCNN-only
        automatic checkpoint profile. on_epoch is a Python observer, never JSON;
        it is incompatible with that callback-free checkpoint profile.
        """
        from .session import TrainingSession
        return TrainingSession(self,model,train_loader,val_loader,model_config=model_config,on_epoch=on_epoch)
