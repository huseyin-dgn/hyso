"""P24/P27/P28/P29: bounded explicit CPU/CUDA experiments with embedded source evidence."""
import typing as _typing
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
import base64
import zlib
from contextvars import ContextVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, SequentialSampler, BatchSampler
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


def _input_chunks(data):
    """Decode at most one encoded slab; legacy nested-list cohorts still work."""
    dtype={'torch.float32':torch.float32,'torch.float64':torch.float64}[data['dtype']]
    value=data['inputs']
    if type(value) is list:
        yield torch.tensor(value,dtype=dtype)
        return
    if (type(value) is not dict or set(value)!={'encoding','shape','chunks'}
            or value['encoding']!='zlib_base64_tensor_chunks_v1'
            or type(value['shape']) is not list or len(value['shape'])<2
            or any(type(n) is not int or n<1 for n in value['shape'])
            or type(value['chunks']) is not list or not value['chunks']):
        raise ValueError('invalid encoded cohort')
    count=0;sample_bytes=math.prod(value['shape'][1:])*torch.empty((),dtype=dtype).element_size()
    for row in value['chunks']:
        if (type(row) is not dict or set(row)!={'samples','data','sha256'}
                or type(row['samples']) is not int or not 1<=row['samples']<=min(value['shape'][0]-count,max(1,1024**2//sample_bytes))):
            raise ValueError('invalid cohort chunk coverage')
        expected=row['samples']*sample_bytes
        decoder=zlib.decompressobj()
        try:raw=decoder.decompress(base64.b64decode(row['data'],validate=True),expected+1)
        except zlib.error as error:raise ValueError('invalid cohort compression') from error
        if (len(raw)!=expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or hashlib.sha256(raw).hexdigest()!=row['sha256']):
            raise ValueError('invalid cohort chunk content')
        chunk=torch.frombuffer(bytearray(raw),dtype=dtype).reshape(row['samples'],*value['shape'][1:])
        if not torch.isfinite(chunk).all():raise ValueError('nonfinite cohort chunk')
        count+=len(chunk)
        yield chunk
    if count!=value['shape'][0]:raise ValueError('incomplete cohort chunks')


def _input_shape(data):
    return data['inputs']['shape'] if type(data['inputs']) is dict else list(torch.tensor(data['inputs']).shape)


def _batch_input_hashes(data,sizes):
    chunks=iter(_input_chunks(data));chunk=next(chunks);offset=0
    for size in sizes:
        digest=hashlib.sha256();remaining=size
        while remaining:
            if offset==len(chunk):chunk=next(chunks);offset=0
            take=min(remaining,len(chunk)-offset)
            digest.update(chunk[offset:offset+take].contiguous().numpy().tobytes())
            offset+=take;remaining-=take
        yield digest.hexdigest()


def _autocast_state(device):
    try:return torch.is_autocast_enabled(device),str(torch.get_autocast_dtype(device))
    except (TypeError,AttributeError):
        return ((torch.is_autocast_enabled(),str(torch.get_autocast_gpu_dtype())) if device=='cuda'
                else (torch.is_autocast_cpu_enabled(),str(torch.get_autocast_cpu_dtype())))


class _CohortDataset(Dataset):
    """Sequential evaluation keeps one decompressed slab, not Python image lists."""
    def __init__(self,data):
        self.data=data;self.iterator=iter(_input_chunks(data));self.chunk=None;self.start=0
    def __len__(self):return len(self.data['targets'])
    def __getitem__(self,index):
        if self.chunk is not None and index<self.start:raise ValueError('cohort requires sequential order')
        while self.chunk is None or index>=self.start+len(self.chunk):
            if self.chunk is not None:self.start+=len(self.chunk)
            self.chunk=next(self.iterator)
        return self.chunk[index-self.start],self.data['targets'][index]


@dataclass(frozen=True, init=False)
class EvaluationCohort:
    """Explicit paired IDs, labels, order, split, preprocessing and input content.

    Args:
        inputs (_typing.Any): Named input specifications for the builder, ordered references for add(), or supported runtime inputs for forward(). Required.
        targets (_typing.Any): Ground-truth targets with the task’s required shape, dtype and label space. Required.
        sample_ids (_typing.Any): Unique ordered sample identities for cohort alignment. Required.
        class_names (_typing.Any): Ordered class labels matching the prediction class dimension. Required.
        split (_typing.Any): Explicit dataset split label used in evaluation identity. Required.
        preprocessing_id (_typing.Any): Identity of the preprocessing used to produce this fixed cohort. Required.
        task (str): Explicit supported task; classification metrics do not imply detection or segmentation support. Default: 'multiclass'.

    Returns:
        An instance of EvaluationCohort with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        EvaluationCohort(inputs, targets, sample_ids=sample_ids, class_names=class_names, split=split, preprocessing_id=preprocessing_id)

    HYSo Guide: analysis.p26
    Offline: hyso docs show analysis.p26 --lang en
    Open: hyso docs open analysis.p26 --lang en
    """
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, inputs: '_typing.Any', targets: '_typing.Any', *, sample_ids: '_typing.Any', class_names: '_typing.Any', split: '_typing.Any', preprocessing_id: '_typing.Any', task: _typing.Literal['multiclass', 'binary']="multiclass") -> 'None':
        """Initialize EvaluationCohort with the supplied settings.

        Args:
            inputs (_typing.Any): Named input specifications for the builder, ordered references for add(), or supported runtime inputs for forward(). Required.
            targets (_typing.Any): Ground-truth targets with the task’s required shape, dtype and label space. Required.
            sample_ids (_typing.Any): Unique ordered sample identities for cohort alignment. Required.
            class_names (_typing.Any): Ordered class labels matching the prediction class dimension. Required.
            split (_typing.Any): Explicit dataset split label used in evaluation identity. Required.
            preprocessing_id (_typing.Any): Identity of the preprocessing used to produce this fixed cohort. Required.
            task (str): Explicit supported task; classification metrics do not imply detection or segmentation support. Default: 'multiclass'.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            EvaluationCohort(inputs, targets, sample_ids=sample_ids, class_names=class_names, split=split, preprocessing_id=preprocessing_id)

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        _name(preprocessing_id)
        config=_metadata(task,class_names,split,0)
        if (type(inputs) is not torch.Tensor or inputs.device.type!='cpu' or inputs.dtype not in (torch.float32,torch.float64)
                or inputs.ndim<2 or len(inputs)<1): raise ValueError("nonempty finite CPU cohort required")
        y=targets.detach().tolist() if isinstance(targets,torch.Tensor) else _plain(targets)
        if task=='binary' and type(y) is list:
            y=[int(v) if type(v) in (int,float) and v in (0,1) else v for v in y]
        from .evaluation import _labels
        _labels(y,len(class_names))
        ids=_plain(sample_ids)
        if (type(ids) is not list or len(ids)!=len(inputs) or len(y)!=len(inputs)
                or any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids)):
            raise ValueError("unique ordered sample IDs and exact labels required")
        # Preserve the legacy representation for old small cohorts. Large inputs
        # use lossless slabs, avoiding millions of Python float objects in JSON.
        encoded=[];slab=max(1,1024**2//(inputs[0].numel()*inputs.element_size()))
        for part in inputs.split(slab):
            if not torch.isfinite(part).all():raise ValueError('nonfinite cohort input')
            if len(inputs)>4096 or inputs.numel()>1000000:
                raw=part.detach().contiguous().numpy().tobytes()
                encoded.append(dict(samples=len(part),data=base64.b64encode(zlib.compress(raw)).decode('ascii'),
                    sha256=hashlib.sha256(raw).hexdigest()))
        payload=(dict(encoding='zlib_base64_tensor_chunks_v1',shape=list(inputs.shape),chunks=encoded)
                 if encoded else inputs.tolist())
        data=dict(inputs=payload,dtype=str(inputs.dtype),targets=y,sample_ids=ids,
                  configuration=config,preprocessing_id=preprocessing_id)
        object.__setattr__(self,'_payload',json.dumps(data,allow_nan=False))

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        return json.loads(self._payload)

    @property
    def sha256(self) -> '_typing.Any':
        """Return the content identity of this fixed evaluation cohort.

        Returns:
            The canonical SHA-256 digest string of this retained content.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.sha256

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        return _hash(self.to_dict())

    def tensors(self) -> '_typing.Any':
        """Return this object's recorded tensor data according to its native copy policy.

        Returns:
            Defensive clones of the fixed input and target tensors.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.tensors()

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        d=self.to_dict()
        return torch.cat(list(_input_chunks(d))),torch.tensor(d['targets'])


def _cohort(data):
    if type(data) is not dict or set(data)!={'inputs','dtype','targets','sample_ids','configuration','preprocessing_id'}:
        raise ValueError('invalid cohort fields')
    c=data['configuration']
    if type(data['inputs']) is dict:
        shape=_input_shape(data);count=sum(len(chunk) for chunk in _input_chunks(data))
        c=data['configuration'];ids=data['sample_ids'];targets=data['targets']
        if (count!=shape[0] or len(targets)!=count or len(ids)!=count
                or any(type(i) is not str or not i for i in ids) or len(set(ids))!=count
                or c!=_metadata(c['task'],c['class_names'],c['split'],c['zero_division'])):
            raise ValueError('invalid encoded cohort identity')
        from .evaluation import _labels
        _labels(targets,len(c['class_names']));_name(data['preprocessing_id'])
        result=object.__new__(EvaluationCohort)
        object.__setattr__(result,'_payload',json.dumps(data,allow_nan=False))
        return result
    x=torch.tensor(data['inputs'],dtype={'torch.float32':torch.float32,'torch.float64':torch.float64}[data['dtype']])
    result=EvaluationCohort(x,data['targets'],sample_ids=data['sample_ids'],class_names=c['class_names'],
        split=c['split'],preprocessing_id=data['preprocessing_id'],task=c['task'])
    if _hash(result.to_dict())!=_hash(data):raise ValueError('invalid cohort encoding')
    return result


def _paired(a,b, *, same_inputs):
    for key in ('sample_ids','targets','configuration','preprocessing_id','dtype'):
        if a[key]!=b[key]:raise ValueError('paired cohort mismatch: '+key)
    if _input_shape(a)!=_input_shape(b):raise ValueError('paired shape mismatch')
    if same_inputs and a['inputs']!=b['inputs']:raise ValueError('input content mismatch')


def _evaluate(model,cohort,batch_size=16,device='cpu'):
    data=cohort.to_dict();c=data['configuration']
    dataset=_CohortDataset(data)
    if type(data['inputs']) is list:
        x,y=cohort.tensors()
        dataset=TensorDataset(x,y.float() if c['task']=='binary' else y)
    def collate(rows):
        x,y=default_collate(rows)
        return x,y.float() if c['task']=='binary' else y.long()
    return evaluate_classification(model,DataLoader(dataset,batch_size=batch_size,collate_fn=collate),
        class_names=c['class_names'],split=c['split'],task=c['task'],source_id=cohort.sha256,device=device)


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
        device=x.device;enabled=self.policy in ('bf16','amp_fp16')
        low=torch.float16 if self.policy=='amp_fp16' else torch.bfloat16
        expected=low if enabled else torch.float64 if self.policy=='fp64' else torch.float32
        records=[];handles=[];collector=_PRECISION_EXECUTION.get()
        def hook(name):
            def observe(module,args,out):
                active,dtype=_autocast_state(device.type)
                records.append(dict(name=name,operator=type(module).__name__,input_dtype=str(args[0].dtype),
                    weight_dtype=str(module.weight.dtype),output_dtype=str(out.dtype),device=str(out.device),
                    autocast_enabled=active,autocast_dtype=dtype if active else None))
            return observe
        if collector is not None:
            for name,module in self.model.named_modules():
                if isinstance(module,(nn.Conv2d,nn.Linear)):handles.append(module.register_forward_hook(hook(name)))
        try:
            with torch.autocast(device.type,dtype=low,enabled=enabled):
                out=self.model(x.double() if self.policy=='fp64' else x.float())
        finally:
            for handle in handles:handle.remove()
        if out.dtype!=expected:raise ValueError('precision fallback forbidden')
        if collector is not None:
            if enabled and not records:raise NotImplementedError('no supported precision operator execution observed')
            master=torch.float64 if self.policy=='fp64' else torch.float32
            if any(r['output_dtype']!=str(expected) or r['weight_dtype']!=str(master)
                   or r['device']!=str(device) or r['autocast_enabled']!=enabled
                   or r['autocast_dtype']!=(str(low) if enabled else None) for r in records):
                raise ValueError('precision operator fallback forbidden')
            if out.device!=device or not torch.isfinite(out).all():raise ValueError('invalid native precision output')
            raw=x.detach().cpu().contiguous().numpy().tobytes()
            collector['input_hash'].update(raw)
            collector['batches'].append(dict(input_shape=list(x.shape),input_dtype=str(x.dtype),device=str(device),
                input_sha256=hashlib.sha256(raw).hexdigest(),
                output_shape=list(out.shape),native_output_dtype=str(out.dtype),finite=True,
                autocast_enabled=enabled,autocast_dtype=str(low) if enabled else None,
                operators=records,completion='output materialized by evaluation'))
        return out.float() if enabled else out


_PRECISION_EXECUTION=ContextVar('hyso_precision_quality_execution',default=None)


def _precision(model,policy,device='cpu'):
    device=torch.device(device)
    allowed=('fp32','fp64','bf16','int8_dynamic') if device.type=='cpu' else ('fp32','amp_fp16','bf16')
    if device.type not in ('cpu','cuda') or policy not in allowed:raise NotImplementedError('precision recipe unsupported on selected device')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise NotImplementedError('CUDA unavailable; no CPU fallback')
        device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    from .performance.sweeps import _prepare
    owned,proof=_prepare(model,policy,device)
    adapter=_PrecisionAdapter(owned,policy).to(device)
    return adapter,dict(policy=policy,p08_recipe=proof,source_master_sha256=_model_hash(model,allow_device=device.type=='cuda'),adapter_state_sha256=_model_hash(adapter,allow_device=device.type=='cuda'),
                        execution_guard='per_batch_dtype' if policy!='int8_dynamic' else 'per_batch_quantized::linear_dynamic')


def _precision_evaluation(adapter,proof,cohort,*,batch_size=16,device='cpu',strict=False):
    records=dict(batches=[],input_hash=hashlib.sha256())
    token=_PRECISION_EXECUTION.set(records if strict and proof['policy']!='int8_dynamic' else None)
    try:result=_evaluate(adapter,cohort,batch_size=batch_size,device=device).to_dict()
    finally:_PRECISION_EXECUTION.reset(token)
    # Success means every isolated inference batch passed the forward guards.
    proof['verified_execution']=dict(batches=len(result['accounting']['batch_sizes']),
        native_output_dtype={'fp32':'torch.float32','fp64':'torch.float64','bf16':'torch.bfloat16','amp_fp16':'torch.float16','int8_dynamic':'torch.float32'}[proof['policy']],
        operators=['quantized::linear_dynamic'] if proof['policy']=='int8_dynamic' else [])
    if strict and proof['policy']!='int8_dynamic':
        if len(records['batches'])!=len(result['accounting']['batch_sizes']):raise ValueError('missing precision batch proof')
        proof.update(execution_version=2,device=result['source']['device'],batch_observations=records['batches'],
            observed_input_stream_sha256=records['input_hash'].hexdigest())
    return result


def _quality_device(device):
    selected=torch.device(device)
    if selected.type not in ('cpu','cuda'):raise NotImplementedError('quality supports CPU or CUDA only')
    if selected.type=='cuda':
        if not torch.cuda.is_available():raise NotImplementedError('CUDA unavailable; no CPU fallback')
        selected=torch.device('cuda',torch.cuda.current_device() if selected.index is None else selected.index)
        torch.cuda.current_stream(selected)  # Initialize before the caller's RNG snapshot.
    return str(selected) if selected.type=='cuda' else 'cpu'


def evaluate_robustness(model: '_typing.Any',cohort: '_typing.Any',perturbation: '_typing.Any', *, corruption_id: '_typing.Any', severity: '_typing.Any', seed: '_typing.Any', device: 'str'='cpu') -> '_typing.Any':
    """

    CPU perturbations remain on the immutable cohort; inference uses the explicit CPU/CUDA device.
    Evaluate clean and perturbed predictions under the paired robustness contract.

    Args:
        model (_typing.Any): Caller-owned model; device, dtype and task must match the native operation. Required.
        cohort (_typing.Any): Fixed evaluation inputs, targets, labels and sample identity. Required.
        perturbation (_typing.Any): Explicit perturbed prediction/cohort evidence; no perturbation is synthesized implicitly. Required.
        corruption_id (_typing.Any): Stable identifier of the applied corruption. Required.
        severity (_typing.Any): Explicit corruption severity recorded in the comparison. Required.
        seed (_typing.Any): Random seed recorded/applied by this operation; reproducibility still depends on the device/backend. Required.

        device (str): Explicit CPU or indexed CUDA device, without CPU fallback. Default: 'cpu'.
    Returns:
        A QualityExperimentResult with the explicit perturbation and baseline evidence.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        evaluate_robustness(model, cohort, perturbation, corruption_id=corruption_id, severity=severity, seed=seed)

    HYSo Guide: analysis.p27
    Offline: hyso docs show analysis.p27 --lang en
    Open: hyso docs open analysis.p27 --lang en
    """
    device=_quality_device(device)
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
        _seed(seed);clean=_evaluate(model,cohort,device=device).to_dict()
        _seed(seed);perturbed=_evaluate(model,other,device=device).to_dict()
        try:code_hash=hashlib.sha256(inspect.getsource(perturbation).encode()).hexdigest()
        except (OSError,TypeError):code_hash=None
        sources=dict(clean_cohort=d,perturbed_cohort=other.to_dict(),clean=clean,perturbed=perturbed,
            corruption=dict(id=corruption_id,severity=severity,seed=seed,calls=len(x),
                operation=getattr(perturbation,'__qualname__',type(perturbation).__qualname__),source_sha256=code_hash))
        if device!='cpu':sources['execution_device']=device
        return _result('P27',sources)
    finally:_restore_rng(rng)


def evaluate_precision_quality(model: '_typing.Any',cohort: '_typing.Any', *, candidate: _typing.Literal['fp32', 'fp64', 'amp_fp16', 'bf16', 'int8_dynamic']='int8_dynamic',device: 'str'='cpu',batch_size: 'int'=16) -> '_typing.Any':
    """Paired labeled quality against a fresh same-device FP32 master copy.

    CPU: FP32/FP64/BF16/dynamic Linear INT8. CUDA opt-in: FP32/AMP FP16/BF16.
    Large cohorts stream lossless encoded slabs; metrics retain all labels/logits.
    No performance benchmark, training, fallback or statistical claim is made.

    Args:
        model (_typing.Any): Caller-owned model; device, dtype and task must match the native operation. Required.
        cohort (_typing.Any): Fixed evaluation inputs, targets, labels and sample identity. Required.
        candidate (str): Validated retained candidate source or predictions required by this comparison. Default: 'int8_dynamic'.
        device (str): Explicit cpu or cuda:index device; this setting grants no GPU admission. Default: 'cpu'.
        batch_size (int): Positive batch-size setting or metadata assumption; no implicit batch search. Default: 16.

    Returns:
        The retained evaluate precision quality value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        evaluate_precision_quality(model, cohort)

    HYSo Guide: analysis.p28
    Offline: hyso docs show analysis.p28 --lang en
    Open: hyso docs open analysis.p28 --lang en
    """
    if type(cohort) is not EvaluationCohort:raise TypeError('explicit cohort required')
    rng=_rng_state()
    try:
        master=_model_hash(model)
        if type(batch_size) is not int or batch_size<1:raise ValueError('positive integer batch_size required')
        baseline,bproof=_precision(model,'fp32',device);owned,proof=_precision(model,candidate,device)
        result=_result('P28',dict(cohort=cohort.to_dict(),master_state_sha256=master,
            baseline=_precision_evaluation(baseline,bproof,cohort,batch_size=batch_size,device=device,strict=True),
            candidate=_precision_evaluation(owned,proof,cohort,batch_size=batch_size,device=device,strict=True),
            baseline_precision=bproof,candidate_precision=proof))
        if _model_hash(model)!=master:raise ValueError('original precision master changed')
        return result
    finally:_restore_rng(rng)


def measure_time_to_quality(factory: '_typing.Any',cohort: '_typing.Any', *, seeds: '_typing.Any', target: '_typing.Any', epochs: 'int'=3,cadence: 'int'=1,metric: _typing.Literal['accuracy', 'macro_f1']='accuracy',device: 'str'='cpu') -> '_typing.Any':
    """

    The factory must return a fresh Trainer on the selected device; CUDA evaluation is synchronized by its native reader.2..8 fresh CPU fits, <=10 epochs/seed, <=256 train/eval samples.

    Args:
        factory (_typing.Any): Trusted importable factory identity required by the worker recipe. Required.
        cohort (_typing.Any): Fixed evaluation inputs, targets, labels and sample identity. Required.
        seeds (_typing.Any): Explicit trial seeds. Required.
        target (_typing.Any): Explicit target value/model/context required by this operation. Required.
        epochs (int): Positive total epoch budget, including a restored prefix; not extra epochs after resume. Default: 3.
        cadence (int): Observation cadence under this protocol. Default: 1.
        metric (str): Exact supported metric selector; units and missing-value status are preserved. Default: 'accuracy'.

        device (str): Explicit CPU or indexed CUDA device, without CPU fallback. Default: 'cpu'.
    Returns:
        The retained measure time to quality value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        measure_time_to_quality(factory, cohort, seeds=seeds, target=target)

    HYSo Guide: analysis.p24
    Offline: hyso docs show analysis.p24 --lang en
    Open: hyso docs open analysis.p24 --lang en
    """
    from .training import Trainer
    from .performance.training import measure_training
    device=_quality_device(device)
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
            if (type(trainer) is not Trainer or str(trainer.device)!=device or trainer.callbacks or trainer.ema is not None
                    or trainer.early_stopping is not None or trainer._epochs is not None):raise ValueError('fresh ordinary same-device Trainer without callbacks/EMA/earlystop required')
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
            modules,entries=_profile(trainer.model,dtype,torch.device(device))
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
                model_state=_model_hash(trainer.model,allow_device=device!='cpu')
                start=time.perf_counter_ns();result=_evaluate(trainer.model,cohort,device=device).to_dict();end=time.perf_counter_ns()
                checkpoints.append(dict(epoch=epoch,start_ns=start,end_ns=end,model_state_sha256=model_state,evaluation=result))
            evaluate(0)
            def callback(event,logs):
                if event=='epoch_end' and (logs['epoch']%cadence==0 or logs['epoch']==epochs):evaluate(logs['epoch'])
            trainer.callbacks=(callback,)
            try:timing=measure_training(trainer,loader,epochs=epochs).to_dict()
            finally:trainer.callbacks=()
            runs.append(dict(seed=seed,training_protocol=conditions,training_data_sha256=train_hash,timing=timing,checkpoints=checkpoints))
        sources=dict(cohort=cohort.to_dict(),seeds=list(seeds),target=target,epochs=epochs,cadence=cadence,metric=metric,runs=runs)
        if device!='cpu':sources['execution_device']=device
        return _result('P24',sources)
    finally:_restore_rng(rng)


def _timing_input(data,precision,batch_size):
    if batch_size is None:
        x,_=_cohort(data).tensors()
    else:
        remaining=min(batch_size,len(data['targets']));parts=[]
        for chunk in _input_chunks(data):
            parts.append(chunk[:remaining]);remaining-=len(parts[-1])
            if not remaining:break
        x=torch.cat(parts)
    return x.double() if precision=='fp64' else x.float()


def _inference_batch(data,x,batch_size):
    return dict(requested_batch_size=batch_size,selection='first_ordered_evaluation_batch',
        sample_start=0,sample_count=len(x),sample_ids=data['sample_ids'][:len(x)],
        input_sha256=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())


def _bundle(model,cohort,precision,iterations,memory_metric='logical_model_bytes',memory_protocol=None,batch_size=None,device='cpu',allow_cuda_peak_reset=False):
    from .benchmark.runner import _profile,_descriptor,_bits
    adapter,proof=_precision(model,precision,device)
    if precision not in (('fp32','fp64') if device=='cpu' else ('fp32','amp_fp16')):raise NotImplementedError('unsupported resident timing precision')
    data=cohort.to_dict();x=_timing_input(data,precision,batch_size)
    if x.ndim!=4:raise ValueError('P29 BenchmarkRunner requires NCHW cohort')
    if device!='cpu':x=x.to(device)
    modules,entries=_profile(adapter.model,x.dtype,torch.device(device))
    descriptor,digest=_descriptor(modules,entries)
    if sum(t.numel()*t.element_size() for _,_,t in entries)>16*1024*1024:
        raise ValueError('P29 model witness exceeds 16 MiB budget')
    # The legacy P26 and benchmark hashes use different names/order/encodings.
    # Byte witnesses let an importer independently reproduce both hashes.
    witness=[dict(kind=kind,name=name,shape=list(t.shape),dtype=str(t.dtype),object_alias=meta['object_alias'],
                  bytes_hex=_bits(t).numpy().tobytes().hex())
             for (kind,name,t),meta in zip(entries,descriptor['tensors'])]
    adapter.register_buffer('benchmark_structure_witness',torch.tensor(list(bytes.fromhex(_hash(descriptor))),dtype=torch.uint8,device=device))
    proof['adapter_state_sha256']=_model_hash(adapter,allow_device=device!='cpu')
    evaluation=_precision_evaluation(adapter,proof,cohort,batch_size=batch_size or len(x),device=device,strict=device!='cpu')
    performance=BenchmarkRunner(adapter.model,device=device,precision=precision,example_input=x,
        warmup_iterations=1,measure_iterations=iterations,repeats=2,seed=0,
        measure_cuda_memory=device!='cpu' and memory_metric in ('cuda_allocated_peak_bytes','cuda_reserved_peak_bytes') and allow_cuda_peak_reset).run().to_dict()
    bundle=dict(evaluation=evaluation,performance=performance,precision=proof,
        benchmark_model_sha256=digest,state_witness=witness,host_id=platform.node())
    if batch_size is not None:bundle['inference_batch']=_inference_batch(data,x.detach().cpu(),batch_size)
    if device!='cpu':
        props=torch.cuda.get_device_properties(device);physical=getattr(props,'uuid',None)
        if physical is None:raise NotImplementedError('native GPU UUID unavailable; cannot bind comparison')
        bundle['gpu_provenance']=dict(device=device,physical_gpu_uuid=str(physical),cuda_build=torch.version.cuda,
            torch_version=str(torch.__version__),stream_id=int(torch.cuda.current_stream(device).cuda_stream))
    from ._quality_memory import capture
    bundle['memory']=capture(bundle,cohort.to_dict(),adapter.model,x,memory_metric,memory_protocol)
    return bundle


def compare_performance_quality(baseline: '_typing.Any',candidate: '_typing.Any',cohort: '_typing.Any', *, precision: _typing.Literal['fp32', 'fp64']='fp32',iterations: 'int'=5,
        metric: _typing.Literal['accuracy', 'macro_f1']='accuracy',min_quality: 'float'=0.,max_quality_drop: 'float'=.01,max_latency_regression: 'float'=.05,independent_samples: 'bool'=False,
        memory_metric: _typing.Literal[None, 'logical_model_bytes', 'cpu_rss_sampled_peak_bytes', 'cuda_allocated_peak_bytes', 'cuda_reserved_peak_bytes', 'physical_vram_bytes']='logical_model_bytes',max_memory_regression: 'float'=.05,memory_protocol: '_typing.Any'=None,batch_size: '_typing.Any'=None,device: 'str'='cpu',allow_cuda_peak_reset: 'bool'=False) -> '_typing.Any':
    """

    Observed quality/latency/memory with explicit same-device protocols.
    CUDA uses comparison_version=4 and device-bound native evidence;
    allocator peak reset is opt-in and irreversible.

    Memory defaults to P11 registered logical bytes. Opt-in process RSS executes
    one separate guarded forward per model; it is neither model-only nor exact peak.
    CUDA allocator peaks require explicit reset consent. Physical VRAM remains
    unsupported, and absent memory cannot select a 3D winner.
    Explicit batch_size streams the full quality cohort, and times/probes its
    first ordered batch separately. Large encoded cohorts require this choice;
    they are never automatically submitted as one inference batch.

    Args:
        baseline (_typing.Any): Validated retained baseline source or predictions required by this comparison. Required.
        candidate (_typing.Any): Validated retained candidate source or predictions required by this comparison. Required.
        cohort (_typing.Any): Fixed evaluation inputs, targets, labels and sample identity. Required.
        precision (str): Supported numerical format; runtime/task/device restrictions remain authoritative. Default: 'fp32'.
        iterations (int): Explicit iteration count within the measurement protocol. Default: 5.
        metric (str): Exact supported metric selector; units and missing-value status are preserved. Default: 'accuracy'.
        min_quality (float): Minimum required quality under the selected metric and its native unit. Default: 0.0.
        max_quality_drop (float): Maximum allowed quality decrease under the selected quality metric. Default: 0.01.
        max_latency_regression (float): Maximum allowed latency regression under the native comparison convention. Default: 0.05.
        independent_samples (bool): Declare whether the supplied evidence represents independent samples; not inferred from counts. Default: False.
        memory_metric (str): Exact retained memory metric selected for the comparison. Default: 'logical_model_bytes'.
        max_memory_regression (float): Maximum allowed regression under the selected memory metric. Default: 0.05.
        memory_protocol (_typing.Any): Typed memory measurement settings; boundaries and device guards remain explicit. Default: None.
        batch_size (_typing.Any): Positive batch-size setting or metadata assumption; no implicit batch search. Default: None.

        device (str): Explicit CPU or indexed CUDA device, without CPU fallback. Default: 'cpu'.
        allow_cuda_peak_reset (bool): Explicit consent for native allocator peak reset in an exclusive owned process. Default: False.
    Returns:
        The retained compare performance quality value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        compare_performance_quality(baseline, candidate, cohort)

    HYSo Guide: analysis.p29
    Offline: hyso docs show analysis.p29 --lang en
    Open: hyso docs open analysis.p29 --lang en
    """
    from dataclasses import asdict
    from ._quality_memory import MEMORY_METRICS
    from .performance.memory import MemoryProtocol
    selected=torch.device(device)
    if type(allow_cuda_peak_reset) is not bool:raise TypeError('explicit peak-reset boolean required')
    if selected.type=='cuda' and memory_metric in ('cuda_allocated_peak_bytes','cuda_reserved_peak_bytes') and not allow_cuda_peak_reset:
        raise ValueError('CUDA allocator peak reset requires allow_cuda_peak_reset=True in an exclusive owned process')
    if selected.type=='cuda' and batch_size is None:batch_size=16
    if type(cohort) is not EvaluationCohort:raise TypeError('explicit cohort required')
    if batch_size is not None and (type(batch_size) is not int or batch_size<1):raise ValueError('positive integer batch_size required')
    if batch_size is None and type(cohort.to_dict()['inputs']) is dict:
        raise ValueError('large cohort requires explicit batch_size; no automatic full-cohort inference')
    if memory_metric not in MEMORY_METRICS:raise ValueError('unsupported P29 memory metric')
    if memory_protocol is not None and (memory_metric!='cpu_rss_sampled_peak_bytes' or type(memory_protocol) is not MemoryProtocol):
        raise TypeError('MemoryProtocol applies only to explicit CPU RSS measurement')
    _integer(iterations,2,50)
    for v in (min_quality,max_quality_drop,max_latency_regression,max_memory_regression):_fraction(v)
    if metric not in ('accuracy','macro_f1') or type(independent_samples) is not bool:raise ValueError('invalid comparison policy')
    device=_quality_device(device)
    rng=_rng_state()
    try:
        policy=dict(metric=metric,min_quality=min_quality,
            max_quality_drop=max_quality_drop,max_latency_regression=max_latency_regression,
            independent_samples=independent_samples,memory_metric=memory_metric,max_memory_regression=max_memory_regression,
            memory_protocol=asdict(memory_protocol or MemoryProtocol()) if memory_metric=='cpu_rss_sampled_peak_bytes' else None)
        if batch_size is not None:policy['batch_size']=batch_size
        if device!='cpu':policy.update(device=device,allow_cuda_peak_reset=allow_cuda_peak_reset)
        return _result('P29',dict(comparison_version=4 if device!='cpu' else 3 if batch_size is not None else 2,cohort=cohort.to_dict(),
            baseline=_bundle(baseline,cohort,precision,iterations,memory_metric,memory_protocol,batch_size,device,allow_cuda_peak_reset),
            candidate=_bundle(candidate,cohort,precision,iterations,memory_metric,memory_protocol,batch_size,device,allow_cuda_peak_reset),policy=policy))
    finally:_restore_rng(rng)


def _precision_binding(evaluation,proof):
    if (proof['policy'] not in ('fp32','fp64','bf16','amp_fp16','int8_dynamic')
            or evaluation['source']['model_state_sha256']!=proof['adapter_state_sha256']):raise ValueError('precision/source mismatch')
    recipe=proof['p08_recipe']
    expected=dict(batches=len(evaluation['accounting']['batch_sizes']),
        native_output_dtype={'fp32':'torch.float32','fp64':'torch.float64','bf16':'torch.bfloat16','amp_fp16':'torch.float16','int8_dynamic':'torch.float32'}[proof['policy']],
        operators=['quantized::linear_dynamic'] if proof['policy']=='int8_dynamic' else [])
    if _hash(proof['verified_execution'])!=_hash(expected):raise ValueError('missing/mismatched precision execution evidence')
    if proof['policy']=='int8_dynamic':
        if (recipe['quantization_recipe']!='dynamic_qint8_Linear' or recipe['execution_proof']['weight_dtype']!='torch.qint8'
                or proof['execution_guard']!='per_batch_quantized::linear_dynamic'):raise ValueError('missing quantized execution guard')
    elif proof['execution_guard']!='per_batch_dtype':raise ValueError('missing dtype execution guard')
    if proof.get('execution_version')==2:
        rows=proof['batch_observations'];device=proof['device'];low=proof['policy'] in ('bf16','amp_fp16')
        if device!=evaluation['source']['device'] or len(rows)!=expected['batches']:raise ValueError('precision device/batch mismatch')
        for row,n in zip(rows,evaluation['accounting']['batch_sizes']):
            shape=row['output_shape'];classes=len(evaluation['configuration']['class_names'])
            valid_shape=(shape in ([n],[n,1]) if evaluation['configuration']['task']=='binary' else shape==[n,classes])
            if (row['input_shape'][0]!=n or not valid_shape or row['native_output_dtype']!=expected['native_output_dtype']
                    or row['device']!=device or row['finite'] is not True or row['autocast_enabled']!=low
                    or row['autocast_dtype']!=(expected['native_output_dtype'] if low else None)
                    or (low and not row['operators'])):raise ValueError('invalid precision batch proof')
            master='torch.float64' if proof['policy']=='fp64' else 'torch.float32'
            if any(op['operator'] not in ('Conv2d','Linear') or op['device']!=device
                   or op['output_dtype']!=expected['native_output_dtype'] or op['weight_dtype']!=master
                   or op['autocast_enabled']!=low or op['autocast_dtype']!=(expected['native_output_dtype'] if low else None)
                   for op in row['operators']):raise ValueError('invalid precision operator proof')


def _performance_binding(bundle,evaluation,cohort):
    p=BenchmarkResult(bundle['performance']).to_dict()
    policy=bundle['precision']['policy']
    if policy not in (('fp32','amp_fp16') if p['environment']['device'].startswith('cuda:') else ('fp32','fp64')) or p['configuration']['precision']!=policy:
        raise ValueError('unsupported/mismatched timing precision')
    selection=bundle.get('inference_batch')
    batch_size=selection['requested_batch_size'] if selection is not None else None
    if batch_size is not None and (type(batch_size) is not int or batch_size<1):raise ValueError('invalid inference batch size')
    x=_timing_input(cohort,policy,batch_size)
    if selection is not None and selection!=_inference_batch(cohort,x,batch_size):raise ValueError('invalid benchmark subset identity')
    count=len(cohort['targets']);step=batch_size or len(x)
    sizes=[min(step,count-start) for start in range(0,count,step)]
    expected=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    if (p['input']['content_sha256']!=expected or p['input']['shape']!=list(x.shape)
            or p['input']['dtype']!=str(x.dtype) or evaluation['accounting']['batch_sizes']!=sizes):
        raise ValueError('performance input differs from labeled cohort')
    if any(evaluation['provenance'][e]!=p['environment'][b] for e,b in
           (('python','python_version'),('torch','torch_version'),('platform','platform'))):
        raise ValueError('performance/evaluation runtime mismatch')
    if evaluation['source']['device']!=p['environment']['device']:raise ValueError('quality/performance device mismatch')
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
        if left['source']['device']!=right['source']['device'] or s.get('execution_device','cpu')!=left['source']['device']:raise ValueError('robustness device mismatch')
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
        if a['source']['device']!=b['source']['device']:raise ValueError('precision baseline device mismatch')
        input_hash=hashlib.sha256()
        for chunk in _input_chunks(c):input_hash.update(chunk.numpy().tobytes())
        for key in ('baseline_precision','candidate_precision'):
            proof=s[key]
            if proof.get('execution_version')==2:
                if proof['observed_input_stream_sha256']!=input_hash.hexdigest():raise ValueError('precision input stream mismatch')
                ev=s['baseline'] if key=='baseline_precision' else s['candidate']
                expected_hashes=list(_batch_input_hashes(c,ev['accounting']['batch_sizes']))
                if [row['input_sha256'] for row in proof['batch_observations']]!=expected_hashes:raise ValueError('precision batch input mismatch')
                if any(row['input_dtype']!=c['dtype'] or row['input_shape'][1:]!=_input_shape(c)[1:]
                       for row in proof['batch_observations']):raise ValueError('precision input contract mismatch')
        result=dict(status='measured',accuracy=_delta(_score(a,'accuracy'),_score(b,'accuracy')),
            macro_f1=_delta(_score(a,'macro_f1'),_score(b,'macro_f1')),candidate_precision=s['candidate_precision']['policy'])
        if s['baseline_precision'].get('execution_version')==2:
            pa=_predicted(a['raw'],a['configuration']);pb=_predicted(b['raw'],b['configuration'])
            changes=[dict(sample_id=i,target=y,baseline=x,candidate=z) for i,y,x,z in zip(c['sample_ids'],c['targets'],pa,pb) if x!=z]
            result.update(paired_samples=len(pa),changed_predictions=len(changes),prediction_changes=changes,device=a['source']['device'])
        return result
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
            if p['provenance']['runtime']['device']!=s.get('execution_device','cpu'):raise ValueError('time-to-target device mismatch')
            if p['actual_epochs']!=s['epochs'] or p['segment_id'] in segments:raise ValueError('incomplete/reused training')
            segments.add(p['segment_id']);training_hashes.add(run['training_data_sha256'])
            protocols.add(_hash(dict(p19=p['provenance'],configuration=run['training_protocol'])))
            if [r['epoch'] for r in run['checkpoints']]!=schedule:raise ValueError('evaluation cadence mismatch')
            crossed=None;previous=0.;checkpoints=[]
            for checkpoint in run['checkpoints']:
                epoch=checkpoint['epoch'];ev=_evaluation(checkpoint['evaluation'],c)
                if ev['source']['device']!=p['provenance']['runtime']['device']:raise ValueError('checkpoint evaluation device mismatch')
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
        if type(version) is not int or version not in (1,2,3,4):raise ValueError('invalid P29 comparison version')
        if version<3 and ('batch_size' in s['policy'] or any('inference_batch' in s[k] for k in ('baseline','candidate'))):
            raise ValueError('batched evidence requires P29 comparison_version=3')
        if version==1 and (any('memory' in s[k] for k in ('baseline','candidate'))
                or any(k in s['policy'] for k in ('memory_metric','memory_protocol','max_memory_regression'))):
            raise ValueError('memory-aware evidence requires P29 comparison_version=2')
        c=s['cohort'];_cohort(c);policy=s['policy']
        for k in ('min_quality','max_quality_drop','max_latency_regression'):_fraction(policy[k])
        if type(policy['independent_samples']) is not bool:raise ValueError('explicit independence policy required')
        a,b=s['baseline'],s['candidate'];ea,eb=_evaluation(a['evaluation'],c),_evaluation(b['evaluation'],c)
        if version in (3,4):
            size=policy.get('batch_size')
            if type(size) is not int or size<1:raise ValueError('invalid comparison batch size')
            if any(type(bundle.get('inference_batch')) is not dict for bundle in (a,b)):
                raise ValueError('missing comparison inference batch')
            if any(bundle['inference_batch']['requested_batch_size']!=size for bundle in (a,b)):
                raise ValueError('comparison batch policy mismatch')
            if a['inference_batch']!=b['inference_batch']:raise ValueError('model timing subsets differ')
        for bundle,evaluation in ((a,ea),(b,eb)):
            _precision_binding(evaluation,bundle['precision'])
            _performance_binding(bundle,evaluation,c)
        if version==4:
            device=policy.get('device')
            if type(device) is not str or not device.startswith('cuda:') or type(policy.get('allow_cuda_peak_reset')) is not bool:raise ValueError('invalid GPU comparison policy')
            for bundle in (a,b):
                p=bundle['performance'];proof=bundle.get('gpu_provenance',{})
                if set(proof)!={'device','physical_gpu_uuid','cuda_build','torch_version','stream_id'}:raise ValueError('missing GPU provenance')
                if proof['device']!=device or p['environment']['device']!=device or proof['torch_version']!=p['environment']['torch_version'] or not proof['cuda_build'] or type(proof['physical_gpu_uuid']) is not str or not proof['physical_gpu_uuid'] or type(proof['stream_id']) is not int:raise ValueError('GPU provenance mismatch')
                if policy['memory_metric'] in ('cuda_allocated_peak_bytes','cuda_reserved_peak_bytes') and not policy['allow_cuda_peak_reset']:raise ValueError('missing allocator reset consent')
            if a['gpu_provenance']!=b['gpu_provenance']:raise ValueError('different GPU runtime/device/stream')
        elif any(bundle['performance']['environment']['device']!='cpu' for bundle in (a,b)):
            raise ValueError('CUDA evidence requires comparison_version=4')
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
        if version in (2,3,4):
            from ._quality_memory import compare
            result=compare(a,b,c,policy,qa,la,qb,lb,result)
        if version in (3,4):
            result.update(comparison_version=version,accuracy=_delta(_score(ea,'accuracy'),_score(eb,'accuracy')),
                macro_f1=_delta(_score(ea,'macro_f1'),_score(eb,'macro_f1')),
                latency=dict(baseline_mean_ms=la,candidate_mean_ms=lb,delta_ms=lb-la,relative_change=(lb-la)/la),
                inference_protocol=dict(requested_batch_size=policy['batch_size'],quality_samples=len(c['targets']),
                    timed_batch_samples=a['inference_batch']['sample_count'],timing_scope='resident_first_ordered_evaluation_batch'))
        return result
    raise ValueError('unsupported quality feature')


@dataclass(frozen=True,init=False)
class QualityExperimentResult:
    """Read and export validated P29 — Quality/speed/memory tradeoffs results; unavailable measurements retain their native status.

    Args:
        data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

    Returns:
        An instance of QualityExperimentResult with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        QualityExperimentResult(data)

    HYSo Guide: analysis.p29
    Offline: hyso docs show analysis.p29 --lang en
    Open: hyso docs open analysis.p29 --lang en
    """
    __slots__=('_payload',)
    _payload:str

    def __init__(self,data: 'dict[str, _typing.Any]') -> 'None':
        # Embedded P19/P26/benchmark nesting exceeds the legacy plain depth cap.
        """Initialize QualityExperimentResult with the supplied settings.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            QualityExperimentResult(data)

        HYSo Guide: analysis.p29
        Offline: hyso docs show analysis.p29 --lang en
        Open: hyso docs open analysis.p29 --lang en
        """
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

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analysis.p29
        Offline: hyso docs show analysis.p29 --lang en
        Open: hyso docs open analysis.p29 --lang en
        """
        return json.loads(self._payload)

    def to_json(self,path: '_typing.Any',*,overwrite: 'bool'=False) -> 'None':
        """Serialize this object using its native JSON format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json(path)

        HYSo Guide: analysis.p29
        Offline: hyso docs show analysis.p29 --lang en
        Open: hyso docs open analysis.p29 --lang en
        """
        d=self.to_dict();self._check(d);_write_export(path,json.dumps(d,indent=2,allow_nan=False)+'\n',overwrite)

    def to_csv(self,path: '_typing.Any',*,overwrite: 'bool'=False) -> 'None':
        """Export this object using its native CSV format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_csv(path)

        HYSo Guide: analysis.p29
        Offline: hyso docs show analysis.p29 --lang en
        Open: hyso docs open analysis.p29 --lang en
        """
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
                    if s.get('comparison_version') in (2,3,4):
                        walk(dict(binding=s[key]['memory']['binding'],metric=s[key]['memory']['metric'],
                                  source_result_id=s[key]['memory']['source']['result_id'] if s[key]['memory']['source'] else None),
                             'provenance.'+key+'.memory')
                    if s.get('comparison_version') in (3,4):
                        walk(s[key]['inference_batch'],'provenance.'+key+'.inference_batch')
        if feature=='P29':walk(s['policy'],'provenance.policy')
        _write_export(path,_csv_text(['feature_id','experiment_id','source_sha256','field','value'],rows),overwrite)


def _result(feature,sources):
    return QualityExperimentResult(dict(schema_version=1,format='hyso.quality_experiment',feature_id=feature,
        experiment_id=str(uuid.uuid4()),sources=sources,source_sha256=_hash(sources),analysis=_derive(feature,sources)))
