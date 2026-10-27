"""P24/P27/P28/P29: bounded CPU experiments with embedded source evidence."""
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import platform
import random
import statistics
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, BatchSampler
from torch.utils.data._utils.collate import default_collate

from .evaluation import (EvaluationResult, evaluate_classification, _metadata, _model_hash,
                         _hash, _predicted)
from .benchmark import BenchmarkResult, BenchmarkRunner, compare_results
from .benchmark.result import _plain, _csv_text, _write_export
from .checkpoint import _rng_state, _restore_rng

__all__ = ["EvaluationCohort", "QualityExperimentResult", "measure_time_to_quality",
           "evaluate_robustness", "evaluate_precision_quality", "compare_performance_quality"]


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high: raise ValueError("invalid bounded integer")


def _name(value):
    if type(value) is not str or not value.strip(): raise ValueError("explicit provenance name required")


def _fraction(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite fraction in [0,1] required")


def _seed(seed):
    _integer(seed,0,2**32-1)
    random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)


@dataclass(frozen=True, init=False)
class EvaluationCohort:
    """Explicit paired IDs, labels, order, split, preprocessing and input content."""
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, inputs, targets, *, sample_ids, class_names, split, preprocessing_id, task="multiclass"):
        _name(preprocessing_id)
        config=_metadata(task,class_names,split,0)
        if (type(inputs) is not torch.Tensor or inputs.device.type!='cpu' or inputs.dtype not in (torch.float32,torch.float64)
                or inputs.ndim<2 or not 1 <= len(inputs) <= 4096 or inputs.numel()>1000000
                or not torch.isfinite(inputs).all()): raise ValueError("bounded nonempty finite CPU cohort required")
        y=targets.detach().tolist() if isinstance(targets,torch.Tensor) else _plain(targets)
        if task=='binary' and type(y) is list:
            y=[int(v) if type(v) in (int,float) and v in (0,1) else v for v in y]
        from .evaluation import _labels
        _labels(y,len(class_names))
        ids=_plain(sample_ids)
        if (type(ids) is not list or len(ids)!=len(inputs) or len(y)!=len(inputs)
                or any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids)):
            raise ValueError("unique ordered sample IDs and exact labels required")
        data=dict(inputs=inputs.tolist(),dtype=str(inputs.dtype),targets=y,sample_ids=ids,
                  configuration=config,preprocessing_id=preprocessing_id)
        object.__setattr__(self,'_payload',json.dumps(data,allow_nan=False))

    def to_dict(self): return json.loads(self._payload)

    @property
    def sha256(self): return _hash(self.to_dict())

    def tensors(self):
        d=self.to_dict()
        return torch.tensor(d['inputs'],dtype={'torch.float32':torch.float32,'torch.float64':torch.float64}[d['dtype']]), torch.tensor(d['targets'])


def _cohort(data):
    if type(data) is not dict or set(data)!={'inputs','dtype','targets','sample_ids','configuration','preprocessing_id'}:
        raise ValueError('invalid cohort fields')
    c=data['configuration']
    x=torch.tensor(data['inputs'],dtype={'torch.float32':torch.float32,'torch.float64':torch.float64}[data['dtype']])
    result=EvaluationCohort(x,data['targets'],sample_ids=data['sample_ids'],class_names=c['class_names'],
        split=c['split'],preprocessing_id=data['preprocessing_id'],task=c['task'])
    if _hash(result.to_dict())!=_hash(data):raise ValueError('invalid cohort encoding')
    return result


def _paired(a,b, *, same_inputs):
    for key in ('sample_ids','targets','configuration','preprocessing_id','dtype'):
        if a[key]!=b[key]:raise ValueError('paired cohort mismatch: '+key)
    if torch.tensor(a['inputs']).shape!=torch.tensor(b['inputs']).shape:raise ValueError('paired shape mismatch')
    if same_inputs and a['inputs']!=b['inputs']:raise ValueError('input content mismatch')


def _evaluate(model,cohort,batch_size=16):
    x,y=cohort.tensors();c=cohort.to_dict()['configuration']
    if c['task']=='binary':y=y.float()
    return evaluate_classification(model,DataLoader(TensorDataset(x,y),batch_size=batch_size),
        class_names=c['class_names'],split=c['split'],task=c['task'],source_id=cohort.sha256)


def _evaluation(data,cohort):
    d=EvaluationResult(data).to_dict()
    if (d['status']!='complete' or d['raw']['targets']!=cohort['targets']
            or d['configuration']!=cohort['configuration'] or d['source']['kind']!='model_inference'
            or d['source']['source_id']!=_hash(cohort)):raise ValueError('evaluation/cohort binding mismatch')
    return d


def _score(evaluation,metric):
    if metric=='accuracy':return evaluation['metrics']['accuracy']
    if metric=='macro_f1':return evaluation['metrics']['averages']['macro']['f1']
    raise ValueError('metric must be accuracy or macro_f1')


def _delta(a,b):
    return dict(baseline=a,candidate=b,percentage_points=100*(b-a),
                relative_percent=100*(b-a)/a if a else None,
                relative_status='defined' if a else 'undefined_zero_baseline')


def _quant_digest(model):
    entries=[]
    for name,m in model.named_modules():
        if isinstance(m,torch.ao.nn.quantized.dynamic.Linear):
            w=m.weight()
            entries.append([name,w.int_repr().tolist(),w.q_scale(),w.q_zero_point(),
                            None if m.bias() is None else m.bias().detach().tolist()])
    if not entries:raise ValueError('no dynamic quantized Linear weights')
    return _hash(entries)


class _PrecisionAdapter(nn.Module):
    """P26-compatible float output; P08 recipes, actual dtype/operator guards."""
    def __init__(self,model,policy):
        super().__init__();self.model=model;self.policy=policy
        self.register_buffer('precision_policy_witness',torch.tensor(list(hashlib.sha256(policy.encode()).digest()),dtype=torch.uint8))
        if policy=='int8_dynamic':
            self.register_buffer('quantized_weight_witness',torch.tensor(list(bytes.fromhex(_quant_digest(model))),dtype=torch.uint8))

    def forward(self,x):
        if self.policy=='int8_dynamic':
            expected=bytes(self.quantized_weight_witness.tolist()).hex()
            if _quant_digest(self.model)!=expected:raise ValueError('quantized weights changed')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                out=self.model(x.float())
            if 'quantized::linear_dynamic' not in {e.key for e in trace.key_averages()}:
                raise ValueError('quantized execution not observed')
            if out.dtype!=torch.float32:raise ValueError('unexpected quantized output dtype')
            if _quant_digest(self.model)!=expected:raise ValueError('quantized weights changed')
            return out
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=self.policy=='bf16'):
            out=self.model(x.double() if self.policy=='fp64' else x.float())
        expected=torch.bfloat16 if self.policy=='bf16' else torch.float64 if self.policy=='fp64' else torch.float32
        if out.dtype!=expected:raise ValueError('precision fallback forbidden')
        return out.float() if self.policy=='bf16' else out


def _precision(model,policy):
    if policy not in ('fp32','fp64','bf16','int8_dynamic'):raise NotImplementedError('CPU precision recipe unsupported')
    from .performance.sweeps import _prepare
    owned,proof=_prepare(model,policy,torch.device('cpu'))
    adapter=_PrecisionAdapter(owned,policy)
    return adapter,dict(policy=policy,p08_recipe=proof,source_master_sha256=_model_hash(model),adapter_state_sha256=_model_hash(adapter),
                        execution_guard='per_batch_dtype' if policy!='int8_dynamic' else 'per_batch_quantized::linear_dynamic')


def _precision_evaluation(adapter,proof,cohort,*,batch_size=16):
    result=_evaluate(adapter,cohort,batch_size=batch_size).to_dict()
    # Success means every isolated inference batch passed the forward guards.
    proof['verified_execution']=dict(batches=len(result['accounting']['batch_sizes']),
        native_output_dtype={'fp32':'torch.float32','fp64':'torch.float64','bf16':'torch.bfloat16','int8_dynamic':'torch.float32'}[proof['policy']],
        operators=['quantized::linear_dynamic'] if proof['policy']=='int8_dynamic' else [])
    return result


def evaluate_robustness(model,cohort,perturbation, *, corruption_id, severity, seed):
    if type(cohort) is not EvaluationCohort:raise TypeError('explicit cohort required')
    _name(corruption_id);_fraction(severity);_integer(seed,0,2**32-1)
    if not callable(perturbation):raise TypeError('perturbation callable required')
    rng=_rng_state()
    try:
        _seed(seed);x,y=cohort.tensors();g=torch.Generator().manual_seed(seed);changed=[]
        for sample in x:
            value=perturbation(sample.unsqueeze(0).clone(),g,severity)
            if type(value) is not torch.Tensor or value.shape!=sample.unsqueeze(0).shape or value.dtype!=x.dtype or value.device.type!='cpu':
                raise ValueError('perturbation must preserve each sample shape/dtype/device')
            changed.append(value.detach().clone())
        d=cohort.to_dict();c=d['configuration']
        other=EvaluationCohort(torch.cat(changed),y,sample_ids=d['sample_ids'],class_names=c['class_names'],
            split=c['split'],preprocessing_id=d['preprocessing_id'],task=c['task'])
        _seed(seed);clean=_evaluate(model,cohort).to_dict()
        _seed(seed);perturbed=_evaluate(model,other).to_dict()
        try:code_hash=hashlib.sha256(inspect.getsource(perturbation).encode()).hexdigest()
        except (OSError,TypeError):code_hash=None
        return _result('P27',dict(clean_cohort=d,perturbed_cohort=other.to_dict(),clean=clean,perturbed=perturbed,
            corruption=dict(id=corruption_id,severity=severity,seed=seed,calls=len(x),
                operation=getattr(perturbation,'__qualname__',type(perturbation).__qualname__),source_sha256=code_hash)))
    finally:_restore_rng(rng)


def evaluate_precision_quality(model,cohort, *, candidate='int8_dynamic'):
    if type(cohort) is not EvaluationCohort:raise TypeError('explicit cohort required')
    rng=_rng_state()
    try:
        master=_model_hash(model)
        baseline,bproof=_precision(model,'fp32');owned,proof=_precision(model,candidate)
        return _result('P28',dict(cohort=cohort.to_dict(),master_state_sha256=master,
            baseline=_precision_evaluation(baseline,bproof,cohort),candidate=_precision_evaluation(owned,proof,cohort),
            baseline_precision=bproof,candidate_precision=proof))
    finally:_restore_rng(rng)


def measure_time_to_quality(factory,cohort, *, seeds, target, epochs=3,cadence=1,metric='accuracy'):
    """2..8 fresh CPU fits, <=10 epochs/seed, <=256 train/eval samples."""
    from .training import Trainer
    from .performance.training import measure_training
    if type(cohort) is not EvaluationCohort:raise TypeError('explicit cohort required')
    _fraction(target);_integer(epochs,1,10);_integer(cadence,1,epochs)
    if type(seeds) not in (list,tuple) or not 2<=len(seeds)<=8 or len(set(seeds))!=len(seeds):raise ValueError('2..8 distinct seeds required')
    for seed in seeds:_integer(seed,0,2**32-1)
    if len(cohort.to_dict()['targets'])>256:raise ValueError('evaluation experiment budget exceeded')
    if cohort.to_dict()['configuration']['split'].strip().lower() in ('train','training'):raise ValueError('held-out split required')
    if metric not in ('accuracy','macro_f1'):raise ValueError('unsupported target metric')
    runs=[];owners=[];training_protocol=None;rng=_rng_state()
    try:
        for seed in seeds:
            _seed(seed);trainer,loader=factory(seed)
            if (type(trainer) is not Trainer or trainer.device.type!='cpu' or trainer.callbacks or trainer.ema is not None
                    or trainer.early_stopping is not None or trainer._epochs is not None):raise ValueError('fresh ordinary CPU Trainer without callbacks/EMA/earlystop required')
            if (type(loader) is not DataLoader or type(loader.dataset) is not TensorDataset or loader.num_workers!=0
                    or type(loader.sampler) is not SequentialSampler or type(loader.batch_sampler) is not BatchSampler
                    or loader.collate_fn is not default_collate or loader.pin_memory or loader.worker_init_fn is not None
                    or loader.batch_size is None or loader.batch_sampler.drop_last
                    or loader.drop_last or not 1<=len(loader.dataset)<=256 or len(loader)>256):raise ValueError('bounded sequential TensorDataset training required')
            if trainer.task!=cohort.to_dict()['configuration']['task']:raise ValueError('training/evaluation task mismatch')
            current={t.untyped_storage().data_ptr() for t in trainer.model.parameters() if t.numel()}
            if any(current & {t.untyped_storage().data_ptr() for t in previous.model.parameters() if t.numel()} for previous in owners):
                raise ValueError('seed runs must own independent models')
            if trainer.optimizer.state:raise ValueError('seed run optimizer must be fresh')
            from .benchmark.runner import _profile,_descriptor
            from .training.resume import _optimizer_plan,_scheduler_plan,_plain as _configuration
            dtype=next(trainer.model.parameters()).dtype
            modules,entries=_profile(trainer.model,dtype,torch.device('cpu'))
            structure,_=_descriptor(modules,entries)
            conditions=_configuration(dict(model=structure,optimizer=_optimizer_plan(trainer),scheduler=_scheduler_plan(trainer),
                grad_clip=trainer.grad_clip,accum_steps=trainer.accum_steps,amp=trainer.amp,batch_size=loader.batch_size,
                loss=dict(type=type(trainer.loss_fn).__module__+'.'+type(trainer.loss_fn).__qualname__,
                          reduction=trainer.loss_fn.reduction,ignore_index=getattr(trainer.loss_fn,'ignore_index',None),
                          label_smoothing=getattr(trainer.loss_fn,'label_smoothing',None),
                          tensors={k:dict(dtype=str(t.dtype),shape=list(t.shape),values=t.detach().tolist())
                                   for k,t in trainer.loss_fn.state_dict().items()})))
            if training_protocol is not None and _hash(conditions)!=training_protocol:
                raise ValueError('seed training protocol mismatch')
            training_protocol=_hash(conditions)
            train_hash=_hash([t.detach().tolist() for t in loader.dataset.tensors])
            owners.append(trainer);checkpoints=[]
            def evaluate(epoch):
                model_state=_model_hash(trainer.model)
                start=time.perf_counter_ns();result=_evaluate(trainer.model,cohort).to_dict();end=time.perf_counter_ns()
                checkpoints.append(dict(epoch=epoch,start_ns=start,end_ns=end,model_state_sha256=model_state,evaluation=result))
            evaluate(0)
            def callback(event,logs):
                if event=='epoch_end' and (logs['epoch']%cadence==0 or logs['epoch']==epochs):evaluate(logs['epoch'])
            trainer.callbacks=(callback,)
            try:timing=measure_training(trainer,loader,epochs=epochs).to_dict()
            finally:trainer.callbacks=()
            runs.append(dict(seed=seed,training_protocol=conditions,training_data_sha256=train_hash,timing=timing,checkpoints=checkpoints))
        return _result('P24',dict(cohort=cohort.to_dict(),seeds=list(seeds),target=target,epochs=epochs,cadence=cadence,metric=metric,runs=runs))
    finally:_restore_rng(rng)


def _bundle(model,cohort,precision,iterations,memory_metric='logical_model_bytes',memory_protocol=None):
    from .benchmark.runner import _profile,_descriptor,_bits
    adapter,proof=_precision(model,precision)
    if precision not in ('fp32','fp64'):raise NotImplementedError('P29 resident timing supports fp32/fp64')
    x,_=cohort.tensors();x=x.double() if precision=='fp64' else x.float()
    if x.ndim!=4:raise ValueError('P29 BenchmarkRunner requires NCHW cohort')
    modules,entries=_profile(adapter.model,x.dtype,torch.device('cpu'))
    descriptor,digest=_descriptor(modules,entries)
    if sum(t.numel()*t.element_size() for _,_,t in entries)>16*1024*1024:
        raise ValueError('P29 model witness exceeds 16 MiB budget')
    # The legacy P26 and benchmark hashes use different names/order/encodings.
    # Byte witnesses let an importer independently reproduce both hashes.
    witness=[dict(kind=kind,name=name,shape=list(t.shape),dtype=str(t.dtype),object_alias=meta['object_alias'],
                  bytes_hex=_bits(t).numpy().tobytes().hex())
             for (kind,name,t),meta in zip(entries,descriptor['tensors'])]
    adapter.register_buffer('benchmark_structure_witness',torch.tensor(list(bytes.fromhex(_hash(descriptor))),dtype=torch.uint8))
    proof['adapter_state_sha256']=_model_hash(adapter)
    evaluation=_precision_evaluation(adapter,proof,cohort,batch_size=len(x))
    performance=BenchmarkRunner(adapter.model,device='cpu',precision=precision,example_input=x,
        warmup_iterations=1,measure_iterations=iterations,repeats=2,seed=0).run().to_dict()
    bundle=dict(evaluation=evaluation,performance=performance,precision=proof,
        benchmark_model_sha256=digest,state_witness=witness,host_id=platform.node())
    from ._quality_memory import capture
    bundle['memory']=capture(bundle,cohort.to_dict(),adapter.model,x,memory_metric,memory_protocol)
    return bundle


def compare_performance_quality(baseline,candidate,cohort, *, precision='fp32',iterations=5,
        metric='accuracy',min_quality=0.,max_quality_drop=.01,max_latency_regression=.05,independent_samples=False,
        memory_metric='logical_model_bytes',max_memory_regression=.05,memory_protocol=None):
    """Observed quality/latency/memory, same CPU FP32/64 protocol.

    Memory defaults to P11 registered logical bytes. Opt-in process RSS executes
    one separate guarded forward per model; it is neither model-only nor exact peak.
    CUDA/physical domains stay unsupported, and absent memory cannot select a 3D winner.
    """
    from dataclasses import asdict
    from ._quality_memory import MEMORY_METRICS
    from .performance.memory import MemoryProtocol
    if type(cohort) is not EvaluationCohort:raise TypeError('explicit cohort required')
    if memory_metric not in MEMORY_METRICS:raise ValueError('unsupported P29 memory metric')
    if memory_protocol is not None and (memory_metric!='cpu_rss_sampled_peak_bytes' or type(memory_protocol) is not MemoryProtocol):
        raise TypeError('MemoryProtocol applies only to explicit CPU RSS measurement')
    _integer(iterations,2,50)
    for v in (min_quality,max_quality_drop,max_latency_regression,max_memory_regression):_fraction(v)
    if metric not in ('accuracy','macro_f1') or type(independent_samples) is not bool:raise ValueError('invalid comparison policy')
    rng=_rng_state()
    try:
        return _result('P29',dict(comparison_version=2,cohort=cohort.to_dict(),baseline=_bundle(baseline,cohort,precision,iterations,memory_metric,memory_protocol),
            candidate=_bundle(candidate,cohort,precision,iterations,memory_metric,memory_protocol),policy=dict(metric=metric,min_quality=min_quality,
            max_quality_drop=max_quality_drop,max_latency_regression=max_latency_regression,
            independent_samples=independent_samples,memory_metric=memory_metric,max_memory_regression=max_memory_regression,
            memory_protocol=asdict(memory_protocol or MemoryProtocol()) if memory_metric=='cpu_rss_sampled_peak_bytes' else None)))
    finally:_restore_rng(rng)


def _precision_binding(evaluation,proof):
    if (proof['policy'] not in ('fp32','fp64','bf16','int8_dynamic')
            or evaluation['source']['model_state_sha256']!=proof['adapter_state_sha256']):raise ValueError('precision/source mismatch')
    recipe=proof['p08_recipe']
    expected=dict(batches=len(evaluation['accounting']['batch_sizes']),
        native_output_dtype={'fp32':'torch.float32','fp64':'torch.float64','bf16':'torch.bfloat16','int8_dynamic':'torch.float32'}[proof['policy']],
        operators=['quantized::linear_dynamic'] if proof['policy']=='int8_dynamic' else [])
    if _hash(proof['verified_execution'])!=_hash(expected):raise ValueError('missing/mismatched precision execution evidence')
    if proof['policy']=='int8_dynamic':
        if (recipe['quantization_recipe']!='dynamic_qint8_Linear' or recipe['execution_proof']['weight_dtype']!='torch.qint8'
                or proof['execution_guard']!='per_batch_quantized::linear_dynamic'):raise ValueError('missing quantized execution guard')
    elif proof['execution_guard']!='per_batch_dtype':raise ValueError('missing dtype execution guard')


def _performance_binding(bundle,evaluation,cohort):
    p=BenchmarkResult(bundle['performance']).to_dict()
    policy=bundle['precision']['policy']
    if policy not in ('fp32','fp64') or p['configuration']['precision']!=policy:
        raise ValueError('unsupported/mismatched timing precision')
    x,_=_cohort(cohort).tensors();x=x.double() if policy=='fp64' else x.float()
    expected=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    if (p['input']['content_sha256']!=expected or p['input']['shape']!=list(x.shape)
            or p['input']['dtype']!=str(x.dtype) or evaluation['accounting']['batch_sizes']!=[len(x)]):
        raise ValueError('performance input differs from labeled cohort')
    if any(evaluation['provenance'][e]!=p['environment'][b] for e,b in
           (('python','python_version'),('torch','torch_version'),('platform','platform'))):
        raise ValueError('performance/evaluation runtime mismatch')
    entries=bundle['state_witness'];metadata=p['model']['structural_descriptor']['tensors']
    if len(entries)!=len(metadata):raise ValueError('incomplete model witness')
    digest=hashlib.sha256();payloads=[];total=0
    for row,meta in zip(entries,metadata):
        if set(row)!={'kind','name','shape','dtype','object_alias','bytes_hex'}:
            raise ValueError('invalid model witness fields')
        if any(row[k]!=meta[k] for k in ('kind','name','shape','dtype','object_alias')):
            raise ValueError('model witness descriptor mismatch')
        raw=bytes.fromhex(row['bytes_hex']);total+=len(raw)
        dtype=getattr(torch,row['dtype'].removeprefix('torch.'),None)
        if not isinstance(dtype,torch.dtype) or len(raw)!=math.prod(row['shape'])*torch.tensor([],dtype=dtype).element_size() or total>16*1024*1024:
            raise ValueError('invalid/budget-exceeded model witness')
        digest.update(str((row['kind'],row['name'],tuple(row['shape']),row['dtype'])).encode());digest.update(raw)
        payloads.append(raw)
    if digest.hexdigest()!=p['model']['weights_sha256'] or digest.hexdigest()!=bundle['benchmark_model_sha256']:
        raise ValueError('benchmark model witness mismatch')
    digest=hashlib.sha256()
    for kind in ('parameter','buffer'):
        if kind=='buffer':
            digest.update(json.dumps(['precision_policy_witness','torch.uint8',[32]]).encode())
            digest.update(hashlib.sha256(policy.encode()).digest())
            digest.update(json.dumps(['benchmark_structure_witness','torch.uint8',[32]]).encode())
            digest.update(bytes.fromhex(_hash(p['model']['structural_descriptor'])))
        seen=set()
        for row,raw in zip(entries,payloads):
            if row['kind']!=kind or row['object_alias'] in seen:continue
            seen.add(row['object_alias'])
            digest.update(json.dumps(['model.'+row['name'],row['dtype'],row['shape']]).encode());digest.update(raw)
    if digest.hexdigest()!=evaluation['source']['model_state_sha256']:
        raise ValueError('quality/timing weights differ')


def _derive(feature,s):
    if feature=='P27':
        a,b=s['clean_cohort'],s['perturbed_cohort'];_cohort(a);_cohort(b);_paired(a,b,same_inputs=False)
        left,right=_evaluation(s['clean'],a),_evaluation(s['perturbed'],b)
        if left['source']['model_state_sha256']!=right['source']['model_state_sha256']:raise ValueError('robustness model mismatch')
        p=s['corruption'];_name(p['id']);_fraction(p['severity']);_integer(p['seed'],0,2**32-1)
        _name(p['operation']);_integer(p['calls'],1,4096)
        if p['calls']!=len(a['sample_ids']):raise ValueError('missing perturbation calls')
        return dict(status='measured',accuracy=_delta(_score(left,'accuracy'),_score(right,'accuracy')),
                    macro_f1=_delta(_score(left,'macro_f1'),_score(right,'macro_f1')),paired_samples=len(a['sample_ids']))
    if feature=='P28':
        c=s['cohort'];_cohort(c);a,b=_evaluation(s['baseline'],c),_evaluation(s['candidate'],c)
        _precision_binding(a,s['baseline_precision']);_precision_binding(b,s['candidate_precision'])
        if s['baseline_precision']['policy']!='fp32':raise ValueError('FP32 quality baseline required')
        if any(s[k]['source_master_sha256']!=s['master_state_sha256'] for k in ('baseline_precision','candidate_precision')):
            raise ValueError('precision master source mismatch')
        return dict(status='measured',accuracy=_delta(_score(a,'accuracy'),_score(b,'accuracy')),
            macro_f1=_delta(_score(a,'macro_f1'),_score(b,'macro_f1')),candidate_precision=s['candidate_precision']['policy'])
    if feature=='P24':
        from .performance.training import TrainingPerformanceResult
        c=s['cohort'];_cohort(c);_fraction(s['target']);_integer(s['epochs'],1,10);_integer(s['cadence'],1,s['epochs'])
        if not 2<=len(s['seeds'])<=8 or len(set(s['seeds']))!=len(s['seeds']):raise ValueError('invalid seed protocol')
        if [r['seed'] for r in s['runs']]!=s['seeds']:raise ValueError('missing seed runs')
        summaries=[];segments=set();training_hashes=set();protocols=set()
        schedule=[0]+[i for i in range(1,s['epochs']+1) if i%s['cadence']==0 or i==s['epochs']]
        for run in s['runs']:
            _integer(run['seed'],0,2**32-1)
            p=TrainingPerformanceResult(run['timing']).to_dict()
            if p['actual_epochs']!=s['epochs'] or p['segment_id'] in segments:raise ValueError('incomplete/reused training')
            segments.add(p['segment_id']);training_hashes.add(run['training_data_sha256'])
            protocols.add(_hash(dict(p19=p['provenance'],configuration=run['training_protocol'])))
            if [r['epoch'] for r in run['checkpoints']]!=schedule:raise ValueError('evaluation cadence mismatch')
            crossed=None;previous=0.;checkpoints=[]
            for checkpoint in run['checkpoints']:
                epoch=checkpoint['epoch'];ev=_evaluation(checkpoint['evaluation'],c)
                _integer(epoch,0,s['epochs'])
                _integer(checkpoint['start_ns'],0,2**63-1);_integer(checkpoint['end_ns'],0,2**63-1)
                if ev['source']['model_state_sha256']!=checkpoint['model_state_sha256']:
                    raise ValueError('checkpoint model/evaluation mismatch')
                if not checkpoint['start_ns']<=checkpoint['end_ns']:raise ValueError('invalid evaluation clock')
                if epoch==0:
                    if checkpoint['end_ns']>p['epochs'][0]['train_start_ns']:raise ValueError('initial evaluation overlaps training')
                else:
                    row=p['epochs'][epoch-1]
                    if not row['train_end_ns']<=checkpoint['start_ns']<=checkpoint['end_ns']<=row['end_ns']:
                        raise ValueError('evaluation overlaps training clock')
                elapsed=sum(e['training_s'] for e in p['epochs'][:epoch])
                score=_score(ev,s['metric'])
                checkpoints.append(dict(epoch=epoch,training_seconds=elapsed,score=score,
                    evaluation_seconds=(checkpoint['end_ns']-checkpoint['start_ns'])/1e9))
                if crossed is None and score>=s['target']:
                    crossed=dict(epoch=epoch,training_seconds=elapsed,interval_low_seconds=previous,interval_high_seconds=elapsed)
                previous=elapsed
            horizon=sum(e['training_s'] for e in p['epochs'])
            summaries.append(dict(seed=run['seed'],status='reached' if crossed else 'right_censored',
                first_observed_crossing=crossed,censor_training_seconds=None if crossed else horizon,checkpoints=checkpoints))
        if len(training_hashes)!=1 or len(protocols)!=1:raise ValueError('seed training data/protocol mismatch')
        times=[r['first_observed_crossing']['training_seconds'] for r in summaries if r['status']=='reached']
        return dict(status='measured',runs=summaries,reached=len(times),censored=len(summaries)-len(times),
            reached_mean_seconds=statistics.mean(times) if times else None,
            reached_sample_variance_seconds2=statistics.variance(times) if len(times)>=2 else None,
            variance_scope='reached_runs_only_not_censor_adjusted',timing_scope='sum_P19_training_s_excludes_evaluation')
    if feature=='P29':
        version=s.get('comparison_version',1)
        if type(version) is not int or version not in (1,2):raise ValueError('invalid P29 comparison version')
        if version==1 and (any('memory' in s[k] for k in ('baseline','candidate'))
                or any(k in s['policy'] for k in ('memory_metric','memory_protocol','max_memory_regression'))):
            raise ValueError('memory-aware evidence requires P29 comparison_version=2')
        c=s['cohort'];_cohort(c);policy=s['policy']
        for k in ('min_quality','max_quality_drop','max_latency_regression'):_fraction(policy[k])
        if type(policy['independent_samples']) is not bool:raise ValueError('explicit independence policy required')
        a,b=s['baseline'],s['candidate'];ea,eb=_evaluation(a['evaluation'],c),_evaluation(b['evaluation'],c)
        for bundle,evaluation in ((a,ea),(b,eb)):
            _precision_binding(evaluation,bundle['precision'])
            _performance_binding(bundle,evaluation,c)
        comparison=compare_results(BenchmarkResult(a['performance']),BenchmarkResult(b['performance']))
        if comparison['status']!='compatible' or a['host_id']!=b['host_id'] or not a['host_id']:raise ValueError('incompatible performance conditions')
        qa,qb=_score(ea,policy['metric']),_score(eb,policy['metric'])
        la=a['performance']['stats']['pooled']['mean_ms'];lb=b['performance']['stats']['pooled']['mean_ms']
        dominance=('candidate' if lb<=la and qb>=qa and (lb<la or qb>qa) else
                   'baseline' if la<=lb and qa>=qb and (la<lb or qa>qb) else 'neither')
        pa=_predicted(ea['raw'],ea['configuration']);pb=_predicted(eb['raw'],eb['configuration'])
        wins=sum(x!=y and z==y for x,z,y in zip(pa,pb,c['targets']))
        losses=sum(x==y and z!=y for x,z,y in zip(pa,pb,c['targets']))
        n=wins+losses
        significant=None;pvalue=None
        status='insufficient_discordant_pairs' if policy['independent_samples'] else 'unsupported_independence_not_attested'
        if policy['metric']!='accuracy':status='unsupported_mcnemar_is_accuracy_only'
        if policy['metric']=='accuracy' and policy['independent_samples'] and n>=10:
            pvalue=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n)
            significant=pvalue<.05;status='exact_paired_mcnemar_caller_attested_independence'
        result=dict(status='measured',quality=_delta(qa,qb),latency_relative_change=(lb-la)/la,
            pareto_dominates=dominance,quality_constraint_met=qb>=policy['min_quality'],
            quality_regression=qa-qb>policy['max_quality_drop'],latency_regression=(lb-la)/la>policy['max_latency_regression'],
            quality_significance=dict(status=status,p_value=pvalue,significant=significant,discordant_pairs=n,alpha=.05),
            timing_significance=dict(status='unsupported_no_independent_timing_experiments',p_value=None),compatibility=comparison)
        if version==2:
            from ._quality_memory import compare
            return compare(a,b,c,policy,qa,la,qb,lb,result)
        return result
    raise ValueError('unsupported quality feature')


@dataclass(frozen=True,init=False)
class QualityExperimentResult:
    __slots__=('_payload',)
    _payload:str

    def __init__(self,data):
        # Embedded P19/P26/benchmark nesting exceeds the legacy plain depth cap.
        data=json.loads(json.dumps(data,allow_nan=False));self._check(data)
        object.__setattr__(self,'_payload',json.dumps(data,allow_nan=False))

    @staticmethod
    def _check(data):
        try:
            if (set(data)!={'schema_version','format','feature_id','experiment_id','sources','source_sha256','analysis'}
                    or type(data['schema_version']) is not int or data['schema_version']!=1
                    or data['format']!='hyso.quality_experiment' or str(uuid.UUID(data['experiment_id']))!=data['experiment_id']):
                raise ValueError('invalid quality result schema')
            if data['source_sha256']!=_hash(data['sources']):raise ValueError('quality evidence hash mismatch')
            if _hash(data['analysis'])!=_hash(_derive(data['feature_id'],data['sources'])):raise ValueError('derived quality evidence mismatch')
        except (KeyError,TypeError,IndexError,OverflowError,RuntimeError) as exc:raise ValueError('malformed quality evidence') from exc

    def to_dict(self):return json.loads(self._payload)

    def to_json(self,path,*,overwrite=False):
        d=self.to_dict();self._check(d);_write_export(path,json.dumps(d,indent=2,allow_nan=False)+'\n',overwrite)

    def to_csv(self,path,*,overwrite=False):
        d=self.to_dict();self._check(d);rows=[]
        def walk(value,path):
            if type(value) is dict:
                for key,item in value.items():walk(item,path+'.'+key)
            elif type(value) is list:
                for i,item in enumerate(value):walk(item,path+'.'+str(i))
            else:rows.append(dict(feature_id=d['feature_id'],experiment_id=d['experiment_id'],source_sha256=d['source_sha256'],field=path,value=value))
        walk(d['analysis'],'analysis')
        # CSV is a readable summary; canonical JSON retains raw source evidence.
        s=d['sources'];feature=d['feature_id']
        for key in (('clean_cohort','perturbed_cohort') if feature=='P27' else ('cohort',)):
            c=s[key]
            walk(dict(sha256=_hash(c),configuration=c['configuration'],preprocessing_id=c['preprocessing_id'],
                      sample_ids=c['sample_ids']), 'provenance.'+key)
        if feature=='P24':
            walk({k:s[k] for k in ('seeds','target','epochs','cadence','metric')},'provenance.protocol')
            walk([dict(seed=r['seed'],segment_id=r['timing']['segment_id'],training_data_sha256=r['training_data_sha256'],
                       evaluation_ids=[p['evaluation']['evaluation_id'] for p in r['checkpoints']])
                  for r in s['runs']], 'provenance.runs')
        elif feature=='P27':walk(s['corruption'],'provenance.corruption')
        if feature in ('P27','P28','P29'):
            for key in (('clean','perturbed') if feature=='P27' else ('baseline','candidate')):
                ev=s[key]['evaluation'] if feature=='P29' else s[key]
                walk(dict(evaluation_id=ev['evaluation_id'],source=ev['source'],source_sha256=ev['source_sha256'],
                          runtime=ev['provenance']), 'provenance.'+key)
                if feature=='P28':walk(s[key+'_precision'],'provenance.'+key+'.precision')
                if feature=='P29':
                    walk(dict(run_id=s[key]['performance']['identity']['run_id'],precision=s[key]['precision'],
                              model_sha256=s[key]['benchmark_model_sha256'],host_id=s[key]['host_id']),
                         'provenance.'+key+'.benchmark')
                    if s.get('comparison_version')==2:
                        walk(dict(binding=s[key]['memory']['binding'],metric=s[key]['memory']['metric'],
                                  source_result_id=s[key]['memory']['source']['result_id'] if s[key]['memory']['source'] else None),
                             'provenance.'+key+'.memory')
        if feature=='P29':walk(s['policy'],'provenance.policy')
        _write_export(path,_csv_text(['feature_id','experiment_id','source_sha256','field','value'],rows),overwrite)


def _result(feature,sources):
    return QualityExperimentResult(dict(schema_version=1,format='hyso.quality_experiment',feature_id=feature,
        experiment_id=str(uuid.uuid4()),sources=sources,source_sha256=_hash(sources),analysis=_derive(feature,sources)))
