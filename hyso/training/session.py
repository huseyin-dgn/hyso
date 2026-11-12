"""User-experience adapter around ONE existing Trainer.fit, not a training engine."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import inspect
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler, TensorDataset

from .config import TrainingConfig, TrainingConfigError, _plain, _resolved
from .trainer import Trainer
from ..models.modular import ModularCNN
from ..models.builder import BuiltClassifier
from ..callbacks.callbacks import EarlyStopping


def _loader(loader, c, *, validation=False):
    if type(loader) is not DataLoader or type(loader.batch_sampler) is not BatchSampler:
        raise TrainingConfigError('this profile requires a standard DataLoader/BatchSampler')
    shuffle = False if validation else c['shuffle']
    expected_sampler = RandomSampler if shuffle else SequentialSampler
    if (loader.batch_size != c['batch_size'] or loader.num_workers != c['num_workers']
            or loader.drop_last != (False if validation else c['drop_last'])
            or loader.batch_sampler.batch_size != loader.batch_size
            or loader.batch_sampler.drop_last != loader.drop_last
            or loader.batch_sampler.sampler is not loader.sampler
            or type(loader.sampler) is not expected_sampler
            or (shuffle and loader.sampler.replacement)):
        raise TrainingConfigError('loader batch_size/workers/drop_last/sampling must match TrainingConfig exactly')
    if len(loader)<1: raise TrainingConfigError('loader must contain at least one batch')
    # Do not consume random transforms or iterate arbitrary caller datasets.
    if type(loader.dataset) is TensorDataset:
        tensors=loader.dataset.tensors
        dtype=torch.float64 if c['precision']=='fp64' else torch.float32
        if len(tensors)!=2 or tensors[0].dtype!=dtype:
            raise TrainingConfigError('TensorDataset must supply inputs with the requested precision and targets')
        target=tensors[1]
        if c['task']=='multiclass' and target.dtype!=torch.long:
            raise TrainingConfigError('multiclass targets must be torch.long')
        if c['task']=='binary' and not target.is_floating_point():
            raise TrainingConfigError('binary targets must be floating hard labels')


@dataclass(frozen=True)
class TrainingResult:
    """Detached request/effective/history/status report from a single successful fit.

    This is a UX report, not P19 timing evidence or a new checkpoint format.
    A resumed history may contain an unmeasured checkpoint prefix. Epoch interval
    timings exist only for observer-enabled, non-checkpoint runs.
    """
    _payload: str

    def to_dict(self) -> dict[str,Any]:
        return json.loads(self._payload)

    @property
    def history(self) -> dict[str,list[float | None]]:
        return self.to_dict()['history']

    def log_to(self, run) -> list[dict[str,Any]]:
        """Explicitly log history/observed intervals to a running training run.

        Does not start/finish the ExperimentManager run or re-execute training.
        Tracking failures propagate; retry/lifecycle policy belongs to the caller.
        History metrics can include a restored prefix; interval metrics cannot.
        """
        state=run.to_dict()
        if state['kind']!='training' or state['status']!='running':
            raise TrainingConfigError('log_to requires a running ExperimentManager training run')
        data=self.to_dict();records=[]
        for key,values in data['history'].items():
            for epoch,value in enumerate(values,1):
                if value is not None:
                    records.append(dict(name=key,value=value,unit='percent' if key.endswith('acc') else 'mean_loss',
                                        epoch=epoch,source='Trainer.history_including_any_restored_prefix'))
        for row in data['epoch_events']:
            records.append(dict(name='epoch_observer_interval',value=row['interval_seconds'],unit='s',
                                epoch=row['epoch'],source='Trainer_callback_interval_including_validation_and_observer_overhead'))
        saved = []
        # Respect the Manager's existing 256-event transaction bound. Replaying
        # an identical history reuses the original per-axis metric events.
        for start in range(0, len(records), 256):
            saved.extend(run.log_metrics(records[start:start + 256]))
        return saved


class TrainingSession:
    """Prepared native optimizer/scheduler/Trainer with explicit applied settings.

    Prefer TrainingConfig.prepare(). Model device/dtype is verified, never silently
    changed. Access trainer/evaluation_model for the existing APIs. fit() delegates
    once and never retries; an exception can leave training state advanced.
    """
    def __init__(self, config: TrainingConfig, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader | None=None, *, model_config: dict[str,Any] | None=None,
                 on_epoch: Callable[[dict[str,Any]],None] | None=None) -> None:
        if type(config) is not TrainingConfig: raise TrainingConfigError('TrainingConfig required')
        self.config=config;c=config.to_dict();self._c=c
        opt,sched,early=_resolved(c)
        if not isinstance(model,nn.Module): raise TrainingConfigError('model must be nn.Module')
        if c['validation'] != (val_loader is not None): raise TrainingConfigError('validation flag and loader must agree')
        _loader(train_loader,c)
        if val_loader is not None: _loader(val_loader,c,validation=True)
        if type(model) in (ModularCNN,BuiltClassifier):
            outputs=model.classifier.out_features
            if (c['task']=='multiclass' and outputs<2) or (c['task']=='binary' and outputs!=1):
                raise TrainingConfigError('known classifier head does not match requested task')
            for loader in (train_loader,val_loader):
                if loader is not None and type(loader.dataset) is TensorDataset:
                    shape=tuple(loader.dataset.tensors[0].shape[1:])
                    if len(shape)!=3 or (type(model) is ModularCNN and shape[0]!=model.input_channels) or (type(model) is BuiltClassifier and shape!=model.input_shape):
                        raise TrainingConfigError('TensorDataset input shape does not match the known model contract')
        checkpoint_mode=c['save_last_to'] is not None or c['resume_from'] is not None
        if checkpoint_mode and (type(model) is not ModularCNN or model_config is None):
            raise TrainingConfigError('automatic checkpoint/resume requires standard ModularCNN plus separate model_config; BuiltClassifier is unsupported')
        if on_epoch is not None:
            if checkpoint_mode: raise TrainingConfigError('on_epoch is incompatible with the existing callback-free checkpoint profile')
            if not callable(on_epoch) or inspect.iscoroutinefunction(on_epoch) or inspect.iscoroutinefunction(getattr(on_epoch,'__call__',None)):
                raise TrainingConfigError('on_epoch must be a synchronous callable')
            try: inspect.signature(on_epoch).bind({})
            except (ValueError,TypeError) as error: raise TrainingConfigError('on_epoch must accept one independent dictionary') from error
        if c['resume_from'] is not None and not Path(c['resume_from']).is_file(): raise TrainingConfigError('resume source must exist')
        try: device=torch.device(c['device'])
        except (RuntimeError,ValueError) as error: raise TrainingConfigError('invalid explicit device') from error
        dtype=torch.float64 if c['precision']=='fp64' else torch.float32
        for tensor in [*model.parameters(),*model.buffers()]:
            if tensor.device!=device or tensor.is_complex() or (tensor.is_floating_point() and tensor.dtype!=dtype):
                raise TrainingConfigError('model device/dtype does not match request; explicitly move/cast before prepare')
        steps=c['epochs']*((len(train_loader)+c['accum_steps']-1)//c['accum_steps'])
        if c['scheduler']=='onecycle' and (steps<2 or abs(sched['pct_start']*steps-1)<1e-12):
            raise TrainingConfigError('OneCycle requires >=2 planned updates and nondegenerate pct_start*total_steps != 1')
        optimizer={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[c['optimizer']](model.parameters(),lr=c['learning_rate'],**opt)
        lr_at_optimizer_creation=[float(g['lr']) for g in optimizer.param_groups]
        native=None
        if c['scheduler']=='step': native=torch.optim.lr_scheduler.StepLR(optimizer,**sched)
        elif c['scheduler']=='cosine': native=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=sched['t_max'],eta_min=sched['eta_min'])
        elif c['scheduler']=='plateau': native=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',**sched)
        elif c['scheduler']=='onecycle': native=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=steps,**sched)
        self._epochs=[];self._mark=None;self._on_epoch=on_epoch;self._checkpoint_mode=checkpoint_mode
        self.train_loader,self.val_loader=train_loader,val_loader
        self.trainer=Trainer(model,optimizer=optimizer,loss_fn=nn.CrossEntropyLoss() if c['task']=='multiclass' else nn.BCEWithLogitsLoss(),
            device=c['device'],task=c['task'],scheduler=native,amp=c['precision']=='amp_fp16',accum_steps=c['accum_steps'],
            grad_clip=c['grad_clip'],ema_decay=c['ema_decay'],early_stopping=None if early is None else EarlyStopping(**early,verbose=False),
            callbacks=() if checkpoint_mode else (self._observe,),model_config=model_config)
        # Use the existing metadata preflight, not a second checkpoint serializer.
        if checkpoint_mode:
            from .resume import prepare_save
            prepare_save(self.trainer,train_loader,val_loader,c['epochs'],c['save_last_to'] or c['resume_from'])
        self._state='prepared'
        self._resolved=dict(optimizer_params=opt,scheduler_params=sched,early_stopping=early,
                            learning_rate_at_optimizer_creation=lr_at_optimizer_creation,planned_optimizer_updates=steps)
        self._prepared_effective=self._effective()

    @property
    def evaluation_model(self) -> nn.Module:
        """Existing Trainer inference target: EMA when enabled, otherwise live model."""
        return self.trainer.evaluation_model

    def _effective(self):
        t=self.trainer
        actual_scheduler={}
        if t.scheduler is not None:
            for key in ('step_size','gamma','T_max','eta_min','factor','patience','threshold','cooldown','min_lrs','eps','mode',
                        'total_steps','cycle_momentum','_schedule_phases','last_epoch'):
                if hasattr(t.scheduler,key): actual_scheduler[key]=_plain(getattr(t.scheduler,key))
        return dict(device=str(t.device),precision=self._c['precision'],amp=t.amp,
            task=t.task,loss_type=type(t.loss_fn).__name__,
            model_floating_dtypes=sorted({str(x.dtype) for x in [*t.model.parameters(),*t.model.buffers()] if x.is_floating_point()}),
            model_devices=sorted({str(x.device) for x in [*t.model.parameters(),*t.model.buffers()]}),
            optimizer_type=type(t.optimizer).__module__+'.'+type(t.optimizer).__qualname__,
            optimizer_groups=[_plain({k:v for k,v in g.items() if k!='params'}) for g in t.optimizer.param_groups],
            scheduler_type=None if t.scheduler is None else type(t.scheduler).__name__,
            scheduler_settings=actual_scheduler,
            batch_size=self.train_loader.batch_size,num_workers=self.train_loader.num_workers,
            shuffle=type(self.train_loader.sampler) is RandomSampler,drop_last=self.train_loader.drop_last,
            batches_per_epoch=len(self.train_loader),validation=self.val_loader is not None,
            accum_steps=t.accum_steps,grad_clip=t.grad_clip,
            ema_decay=None if t.ema is None else t.ema.decay,
            early_stopping=None if t.early_stopping is None else
                {key:getattr(t.early_stopping,key) for key in ('patience','delta','restore_best_weights')})

    def settings(self) -> dict[str,Any]:
        """Detached requested, resolved and actual native settings; no workload execution.

        OneCycle may change optimizer LR at scheduler construction. Resumed values
        are restored by Trainer inside fit; this snapshot makes no pre-restore claim.
        """
        return dict(requested=copy.deepcopy(self._c),model_config=copy.deepcopy(self.trainer.model_config),resolved=copy.deepcopy(self._resolved),
                    effective=self._effective(),state=self._state)

    def _observe(self,event,logs):
        now=time.perf_counter()
        if event=='train_begin': self._mark=now
        elif event=='epoch_end':
            row=dict(logs,interval_seconds=now-self._mark)
            self._mark=now;self._epochs.append(copy.deepcopy(row))
            if self._on_epoch is not None: self._on_epoch(copy.deepcopy(row))

    def fit(self) -> TrainingResult:
        """Execute exactly one existing Trainer.fit; return history and UX observations.

        With automatic checkpoint/resume no callbacks are installed: per-epoch
        timing is unavailable, not synthesized from total wall time. Total fit
        wall seconds include validation, serialization, restore and observers.
        """
        if self._state!='prepared': raise TrainingConfigError('session is one-shot; create a fresh session, never implicitly retry')
        if type(self.config) is not TrainingConfig or self.config.to_dict()!=self._c:
            raise TrainingConfigError('prepared configuration changed; create a fresh session')
        if self.trainer.callbacks != (() if self._checkpoint_mode else (self._observe,)):
            raise TrainingConfigError('prepared callbacks changed; use the explicit on_epoch argument')
        _loader(self.train_loader,self._c)
        if self.val_loader is not None: _loader(self.val_loader,self._c,validation=True)
        if self._effective()!=self._prepared_effective: raise TrainingConfigError('prepared native settings changed before fit')
        self._state='running';start=time.perf_counter()
        try:
            history=self.trainer.fit(self.train_loader,self.val_loader,epochs=self._c['epochs'],
                                     save_last_to=self._c['save_last_to'],resume_from=self._c['resume_from'])
        except BaseException:
            self._state='failed';raise
        wall=time.perf_counter()-start;self._state='completed'
        path=self._c['save_last_to']
        checkpoint=dict(save_last_to=path,resume_from=self._c['resume_from'],
                        file_present_after_fit=False if path is None else Path(path).is_file(),
                        scope='file presence is not proof of a new write; completed resume may be a no-op',
                        resume_status='not_requested' if self._c['resume_from'] is None else 'original_Trainer_restore_completed')
        payload=dict(format='hyso.training_session_result',schema_version=1,settings=self.settings(),
            prepared_effective=self._prepared_effective,history=history,epoch_events=self._epochs,
            fit_wall_seconds=wall,checkpoint=checkpoint,completed_history_epochs=len(history['train_loss']),
            applied_updates=self.trainer.update_count,
            applied_updates_scope='cumulative_Trainer_count_including_any_restored_prefix',
            epoch_timing_scope='unavailable_callback_free_checkpoint_profile' if self._checkpoint_mode else
                'observer_intervals_including_validation_and_observer_overhead_not_P19_or_GPU_time',
            history_scope='Trainer.history_may_include_unmeasured_restored_prefix')
        return TrainingResult(json.dumps(payload,sort_keys=True,allow_nan=False))
