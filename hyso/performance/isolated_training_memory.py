"""P23 trusted construction recipes in bounded fresh Windows interpreters.

Independent wire type; legacy in-process P23 schema1 and never-reset stay intact.
No parent Tensor/Trainer serialization, factory import, training or CUDA query.
"""
from dataclasses import asdict,dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
import uuid

import numpy as np
import torch

from hyso.benchmark.result import _plain,_csv_text,_write_export
from .training import _int,TrainingTimingProtocol,TrainingPerformanceResult,_measure_training
from .training_details import _source
from .training_memory import (TrainingMemoryProtocol,TrainingMemoryError,_MemoryRecorder,_logical_summary,
                              _rss_summary,_memory_diagnostics,_validate_stage_pairs)
from .training_memory import _cuda
from ._training_process import _launch,_bounded_read
from ._saved_activation import _SavedActivationRecorder, _validate_activation


@dataclass(frozen=True,init=False)
class TrainingMemoryRecipe:
    """Trusted importable factory(device=..., **kwargs), construction only.

    Return a dict with trainer/train_loader/val_loader and optional stream. It
    executes arbitrary trusted Python in the child; this is not a security sandbox.
    """
    __slots__=("_payload",)
    _payload:str

    def __init__(self,factory,*,device="cpu",kwargs=None,factory_search_path=None,seed=0,epochs=1,
                 save_last_to=None,resume_from=None):
        self._initialize(True,factory,device=device,kwargs=kwargs,factory_search_path=factory_search_path,
                         seed=seed,epochs=epochs,save_last_to=save_last_to,resume_from=resume_from)

    @classmethod
    def _from_saved(cls,data):
        """Validate stored metadata without admitting factory execution or resolving paths."""
        recipe=object.__new__(cls)
        recipe._initialize(False,**data)
        return recipe

    def _initialize(self,execution,/,factory,*,device="cpu",kwargs=None,factory_search_path=None,
                    seed=0,epochs=1,save_last_to=None,resume_from=None):
        if type(factory) is not str or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",factory) or factory.startswith("__main__:"):
            raise ValueError("trusted importable module:qualified_factory required")
        if type(device) is not str or not re.fullmatch(r"cpu|cuda:\d+",device):
            raise ValueError("explicit cpu or indexed cuda device required")
        _int(seed);_int(epochs,1)
        if seed>2**32-1 or epochs>100000:raise ValueError("seed/epoch ceiling exceeded")
        kwargs={} if kwargs is None else kwargs
        def check(value):
            if value is None or type(value) in (str,bool,int):return
            if type(value) is float and math.isfinite(value):return
            if type(value) is list:
                for item in value:check(item)
                return
            if type(value) is dict and all(type(k) is str for k in value):
                for item in value.values():check(item)
                return
            raise TypeError("recipe contains non-plain/live Python state")
        if type(kwargs) is not dict or "device" in kwargs:raise ValueError("plain kwargs without reserved device required")
        check(kwargs)
        if factory_search_path is not None:
            if execution:
                factory_search_path=str(Path(factory_search_path).resolve())
                if not Path(factory_search_path).is_dir():raise ValueError("factory search directory missing")
            elif (type(factory_search_path) is not str or "\x00" in factory_search_path
                  or not Path(factory_search_path).is_absolute()):
                raise ValueError("stored factory search path must be an absolute path string")
        for path in (save_last_to,resume_from):
            if path is not None and type(path) is not str:raise TypeError("checkpoint paths must be explicit strings")
        data=dict(factory=factory,device=device,kwargs=kwargs,factory_search_path=factory_search_path,
                  seed=seed,epochs=epochs,save_last_to=save_last_to,resume_from=resume_from)
        payload=json.dumps(data,allow_nan=False,separators=(",",":"))
        if len(payload.encode("utf-8"))>65536:raise ValueError("recipe exceeds 64KiB")
        object.__setattr__(self,"_payload",payload)

    def to_dict(self):return json.loads(self._payload)


@dataclass(frozen=True)
class IsolatedTrainingMemoryProtocol:
    """Owned-worker deadlines and evidence bounds; RAM/allocator limits opt-in.

    None leaves the corresponding OS memory/allocator quota unset. Process-tree
    ownership, timeout and bounded publication remain independent of those limits.
    """
    timeout_seconds:float=60.
    process_memory_bytes:int|None=None
    job_memory_bytes:int|None=None
    max_processes:int=4
    cuda_memory_fraction:float|None=None
    max_result_bytes:int=16*1024**2
    max_log_bytes:int=65536
    deterministic:bool=True
    allow_peak_reset:bool=False
    max_windows:int=10000
    saved_activation_lifetime:bool=False
    max_saved_tensors:int=10000
    max_saved_tensor_events:int=40000
    memory_protocol:TrainingMemoryProtocol=TrainingMemoryProtocol()
    timing_protocol:TrainingTimingProtocol=TrainingTimingProtocol()

    def __post_init__(self):
        for name,minimum,maximum in (("process_memory_bytes",512*1024**2,32*1024**3),
                ("job_memory_bytes",512*1024**2,64*1024**3),("max_processes",1,16),
                ("max_result_bytes",16384,64*1024**2),("max_log_bytes",1024,1048576),("max_windows",1,100000),
                ("max_saved_tensors",1,100000),("max_saved_tensor_events",1,400000)):
            value=getattr(self,name)
            if name in ("process_memory_bytes","job_memory_bytes") and value is None:continue
            _int(value,minimum)
            if value>maximum:raise ValueError(name+" exceeds ceiling")
        if (self.job_memory_bytes is not None and self.process_memory_bytes is not None
                and self.job_memory_bytes<self.process_memory_bytes):raise ValueError("job memory must cover one process")
        for name,maximum in (("timeout_seconds",3600),("cuda_memory_fraction",1)):
            value=getattr(self,name)
            if name=="cuda_memory_fraction" and value is None:continue
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=maximum:raise ValueError("invalid "+name)
        if any(type(v) is not bool for v in (self.deterministic,self.allow_peak_reset,self.saved_activation_lifetime)):raise TypeError("boolean policy required")
        if type(self.memory_protocol) is not TrainingMemoryProtocol or type(self.timing_protocol) is not TrainingTimingProtocol:
            raise TypeError("explicit native memory/timing protocols required")


def _protocol(data):
    copied=dict(data);copied["memory_protocol"]=TrainingMemoryProtocol(**copied["memory_protocol"])
    copied["timing_protocol"]=TrainingTimingProtocol(**copied["timing_protocol"])
    return IsolatedTrainingMemoryProtocol(**copied)


def _hash(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()


def _activation():return dict(value=None,status="unsupported",reason="no_validated_saved_tensor_lifetime_adapter")


class _IsolatedRecorder(_MemoryRecorder):
    """Single reset owner, constructed only after OS admission in this worker."""
    def __init__(self,trainer,timing_protocol,protocol,token,activation=None):
        self.isolated_protocol=protocol;self.token=token;self.windows=[];self.window=None
        self.activation=activation
        super().__init__(trainer,timing_protocol,protocol.memory_protocol)

    def sample(self):
        # Keep the raw native tuple. An inconsistent Windows tuple invalidates
        # the RSS domain, without discarding independent training/CUDA evidence.
        # No clamp, retry, synthetic HWM or successful RSS summary is permitted.
        from .memory import _windows_rss, _BackendUnavailable
        if os.name != "nt":return super().sample()
        start=time.perf_counter_ns()
        try:rss,hwm=_windows_rss()
        except (OSError,_BackendUnavailable) as error:
            self.rss_reason=str(error);return
        end=time.perf_counter_ns()
        if end<start:raise ValueError("non-monotonic RSS query clock")
        if hwm<rss or (self.rss and hwm<self.rss[-1]["lifetime_hwm_bytes"]):
            self.rss_reason="inconsistent_native_windows_rss_hwm; raw_tuple_retained"
        self.rss.append(dict(timestamp_ns=(start+end)//2,query_start_ns=start,query_end_ns=end,
            rss_bytes=rss,lifetime_hwm_bytes=hwm,backend="windows.GetProcessMemoryInfo.working_set"))

    def microbatch_begin(self,microbatch_id,new_window):
        if new_window:
            if len(self.windows)>=self.isolated_protocol.max_windows:raise RuntimeError("isolated window record budget exhausted")
            reset=self.stream is not None and self.isolated_protocol.allow_peak_reset
            row=dict(epoch_id=self.epoch["epoch_id"],window_id=self.window_id+1,start_ns=time.perf_counter_ns(),
                device=str(self.trainer.device),stream_id=int(self.stream.cuda_stream) if self.stream is not None else None,
                reset_performed=reset,reset_owner_pid=os.getpid() if reset else None,
                reset_owner_probe_id=self.token if reset else None,
                status="measured" if reset else "unsupported",
                reason=None if reset else "CPU_training" if self.stream is None else "explicit_reset_permission_not_granted",
                previous=None,after_reset=None,end=None,reset_end_ns=None,
                previous_peak_domain="process_lifetime" if not reset or not self.windows else "since_previous_reset",
                peak_domain="owned_training_window" if reset else None,
                start_completion_wait_ns=0,end_completion_wait_ns=0,
                peak_allocated_bytes=None,peak_reserved_bytes=None,incremental_peak_allocated_bytes=None,
                incremental_peak_reserved_bytes=None,allocation_retries_delta=None,oom_count_delta=None)
            if reset:
                self._check_stream();start=time.perf_counter_ns();self.stream.synchronize()
                row["start_completion_wait_ns"]=time.perf_counter_ns()-start
                row["previous"]=_cuda(self.trainer.device)
                torch.cuda.reset_peak_memory_stats(self.trainer.device)
                row["after_reset"]=_cuda(self.trainer.device);row["reset_end_ns"]=time.perf_counter_ns()
            self.window=row
        super().microbatch_begin(microbatch_id,new_window)
        if self.activation is not None:
            self.activation.context=dict(epoch_id=self.epoch["epoch_id"],microbatch_id=microbatch_id,window_id=self.window_id)

    def stage_begin(self,name):
        super().stage_begin(name)
        if self.activation is not None:self.activation.stage=name

    def stage_end(self,name):
        try:super().stage_end(name)
        finally:
            if self.activation is not None:self.activation.stage=None

    def attempt_end(self,updates,samples):
        row=self.window
        if row is None:raise RuntimeError("memory attempt without owned window")
        before=self.active["attempt"][0]["update_before"]
        row.update(attempt_id=self.active["attempt"][0]["attempt_id"],samples=samples,
                   applied=updates==before+1,skipped=updates==before)
        if row["reset_performed"]:
            self._check_stream();start=time.perf_counter_ns();self.stream.synchronize()
            row["end_completion_wait_ns"]=time.perf_counter_ns()-start
            row["end"]=_cuda(self.trainer.device)
            for kind in ("allocated","reserved"):
                row["peak_"+kind+"_bytes"]=row["end"]["lifetime_peak_"+kind+"_bytes"]
                row["incremental_peak_"+kind+"_bytes"]=row["peak_"+kind+"_bytes"]-row["previous"][kind+"_bytes"]
            for key in ("allocation_retries","oom_count"):
                row[key+"_delta"]=row["end"][key]-row["previous"][key]
        row["end_ns"]=time.perf_counter_ns();self.windows.append(row);self.window=None
        super().attempt_end(updates,samples)

    def close(self):
        try:super().close()
        finally:self.window=None


def _run_recipe(request):
    """Called only by admitted stdlib bootstrap. Imports/allocations stay child-owned."""
    recipe=TrainingMemoryRecipe(**request["recipe"]).to_dict();protocol=_protocol(request["protocol"])
    if str(torch.__version__)!=request["torch_version"]:raise RuntimeError("worker runtime Torch mismatch")
    if torch.cuda.is_initialized():raise RuntimeError("CUDA initialized before owned allocator setup")
    device=torch.device(recipe["device"])
    if protocol.saved_activation_lifetime and device.type=="cuda":
        raise NotImplementedError("CUDA saved-activation adapter disabled: P23 R1 native-crash gate unresolved (0xC0000409)")
    if device.type=="cuda":
        if not torch.cuda.is_available() or device.index>=torch.cuda.device_count():raise NotImplementedError("requested CUDA device unavailable")
        torch.cuda.set_device(device)
        if protocol.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(protocol.cuda_memory_fraction,device)
        if torch.cuda.get_allocator_backend()!="native":raise NotImplementedError("only native CUDA allocator is validated")
    torch.set_num_threads(1)
    if protocol.deterministic:
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(recipe["seed"]);np.random.seed(recipe["seed"]);torch.manual_seed(recipe["seed"])
    if recipe["factory_search_path"]:sys_path_append(recipe["factory_search_path"])
    module,name=recipe["factory"].split(":");factory=importlib.import_module(module)
    for part in name.split("."):factory=getattr(factory,part)
    workload=factory(device=recipe["device"],**recipe["kwargs"])
    if type(workload) is not dict or not {"trainer","train_loader","val_loader"}<=set(workload) or set(workload)-{"trainer","train_loader","val_loader","stream"}:
        raise TypeError("factory must return trainer/train_loader/val_loader and optional stream")
    trainer=workload["trainer"]
    from hyso.training import Trainer
    from torch.utils.data import DataLoader
    if type(trainer) is not Trainer or trainer.device!=device:raise ValueError("factory Trainer/device mismatch")
    for key in ("train_loader","val_loader"):
        loader=workload[key]
        if loader is None and key=="val_loader":continue
        if type(loader) is not DataLoader or loader.num_workers!=0:
            raise NotImplementedError("initial isolated profile requires DataLoader num_workers=0")
    stream=workload.get("stream")
    if stream is not None and (device.type!="cuda" or not isinstance(stream,torch.cuda.Stream) or stream.device!=device):
        raise ValueError("factory selected stream/device mismatch")
    activation=_SavedActivationRecorder(trainer.model,protocol.max_saved_tensors,protocol.max_saved_tensor_events) if protocol.saved_activation_lifetime else None
    captured=[]
    def factory_recorder(t,p):
        recorder=_IsolatedRecorder(t,p,protocol,request["token"],activation);captured.append(recorder);return recorder
    try:
        from contextlib import nullcontext
        with (torch.cuda.stream(stream) if stream is not None else nullcontext()), (activation.hooks() if activation is not None else nullcontext()):
            timing=_measure_training(trainer,workload["train_loader"],workload["val_loader"],epochs=recipe["epochs"],
                protocol=protocol.timing_protocol,save_last_to=recipe["save_last_to"],resume_from=recipe["resume_from"],
                recorder_factory=factory_recorder)
        recorder=captured[0]
        if recorder.errors:raise TrainingMemoryError("RSS sampler failure: "+recorder.errors[0])
        projection=timing._export_data()
        projection["limitations"][1]="P19 projection of isolated P23 memory capture; no P20 stage timing"
        projection["instrumentation"]["event_policy"]="P19 bounded event pairs; memory queries do not record events"
        timing=TrainingPerformanceResult(projection)
        source,digest=_source(timing)
        return dict(source_p19=source,source_sha256=digest,source_segment_id=source["segment_id"],
            boundaries=recorder.boundaries,rss=dict(observations=recorder.rss,reason=recorder.rss_reason,
                summary=_rss_summary(recorder.rss,recorder.rss_reason)),windows=recorder.windows,activation=activation.result() if activation is not None else _activation(),
            provenance=dict(pid=os.getpid(),torch=str(torch.__version__),device=str(device),
                stream_id=source["instrumentation"]["stream_id"],package_source=__file__,
                factory=recipe["factory"],cuda_allocator_fraction=protocol.cuda_memory_fraction if device.type=="cuda" else None),
            cuda_domains=dict(pre_probe_lifetime=recorder.boundaries[0]["cuda"],
                boundary_peaks="since_most_recent_reset" if protocol.allow_peak_reset else "process_lifetime",
                reset_is_irreversible=protocol.allow_peak_reset,
                physical_vram=dict(value=None,status="unsupported",reason="allocator_counters_are_not_physical_VRAM")),
            coverage="partial")
    finally:
        try:
            for recorder in captured:recorder.close()
        finally:
            if activation is not None:activation.close()
            captured.clear();workload.clear()


def sys_path_append(path):
    import sys
    sys.path.append(path)


def _validate_windows(measured,source,protocol,probe_id,pid):
    windows=measured["windows"]
    source_windows=[r for r in source["observations"] if r["kind"]=="window"]
    attempts={r["window_id"]:r for r in source["observations"] if r["kind"]=="attempt"}
    if len(windows)!=len(source_windows) or len(windows)>protocol.max_windows:raise ValueError("incomplete/budgeted memory windows")
    expected_reset=protocol.allow_peak_reset and protocol.memory_protocol.device.startswith("cuda:")
    for row,window in zip(windows,source_windows):
        attempt=attempts[window["window_id"]]
        for key in ("epoch_id","window_id","samples"):
            _int(row[key],1)
            if row[key]!=window[key]:raise ValueError("memory window sample/ID mismatch")
        if any(row[k]!=attempt[k] for k in ("attempt_id","applied","skipped")):raise ValueError("memory window attempt ownership mismatch")
        if row["device"]!=protocol.memory_protocol.device or row["stream_id"]!=source["instrumentation"]["stream_id"]:raise ValueError("window device/stream mismatch")
        for key in ("start_ns","end_ns","start_completion_wait_ns","end_completion_wait_ns"):_int(row[key])
        if not row["start_ns"]<=window["start_ns"]<=attempt["start_ns"]<=row["end_ns"]<=window["end_ns"]:
            raise ValueError("invalid memory window completion boundary")
        if type(row["reset_performed"]) is not bool or row["reset_performed"]!=expected_reset:raise ValueError("unauthorized counter reset")
        if row["previous_peak_domain"]!=("process_lifetime" if not expected_reset or row is windows[0] else "since_previous_reset") or row["peak_domain"]!=("owned_training_window" if expected_reset else None):raise ValueError("invalid peak domain")
        metrics=("peak_allocated_bytes","peak_reserved_bytes","incremental_peak_allocated_bytes", "incremental_peak_reserved_bytes","allocation_retries_delta","oom_count_delta")
        if not expected_reset:
            reason="CPU_training" if row["device"]=="cpu" else "explicit_reset_permission_not_granted"
            if row["status"]!="unsupported" or row["reason"]!=reason or any(row[k] is not None for k in
                    (*metrics,"previous","after_reset","end","reset_end_ns","reset_owner_pid","reset_owner_probe_id")):
                raise ValueError("unreset window cannot claim true peak")
            continue
        if row["status"]!="measured" or row["reason"] is not None or row["reset_owner_pid"]!=pid or row["reset_owner_probe_id"]!=probe_id:
            raise ValueError("invalid reset ownership evidence")
        if not row["start_ns"]<=row["reset_end_ns"]<=window["start_ns"]:raise ValueError("peak reset outside owned window start")
        previous,after,end=row["previous"],row["after_reset"],row["end"]
        for observation in (previous,after,end):
            if observation["allocator_backend"]!="native":raise ValueError("non-native allocator peak unvalidated")
            for key in ("allocated_bytes","reserved_bytes","lifetime_peak_allocated_bytes","lifetime_peak_reserved_bytes","allocation_retries","oom_count"):_int(observation[key])
            if observation["allocated_bytes"]>observation["reserved_bytes"] or observation["allocated_bytes"]>observation["lifetime_peak_allocated_bytes"] or observation["reserved_bytes"]>observation["lifetime_peak_reserved_bytes"]:raise ValueError("inconsistent allocator snapshot")
        for kind in ("allocated","reserved"):
            current=kind+"_bytes";peak="peak_"+current;increment="incremental_"+peak
            _int(row[peak]);_int(row[increment])
            if previous[current]!=after[current] or after["lifetime_"+peak]!=after[current]:raise ValueError("reset baseline changed or reset not observed")
            if row[peak]!=end["lifetime_"+peak] or row[peak]<max(previous[current],end[current]) or row[increment]!=row[peak]-previous[current]:raise ValueError("invalid actual/incremental window peak")
        for key in ("allocation_retries","oom_count"):
            _int(row[key+"_delta"])
            if row[key+"_delta"]!=end[key]-previous[key] or after[key]!=previous[key]:raise ValueError("counter delta/reset inconsistency")


def _validate(data):
    required={"schema_version","format","feature_ids","probe_id","recipe","protocol","outcome","reason",
              "receipt","worker_receipt","measurement","limitations","sha256"}
    if set(data)!=required or type(data["schema_version"]) is not int or data["schema_version"]!=1 or data["feature_ids"]!=["P23"] or data["format"]!="hyso.performance.isolated_training_memory":raise ValueError("invalid isolated P23 schema")
    if data["sha256"]!=_hash({k:v for k,v in data.items() if k!="sha256"}):raise ValueError("isolated P23 checksum mismatch")
    uuid.UUID(data["probe_id"]);recipe=TrainingMemoryRecipe._from_saved(data["recipe"]).to_dict();protocol=_protocol(data["protocol"])
    if recipe["device"]!=protocol.memory_protocol.device:raise ValueError("recipe/protocol device mismatch")
    outcome=data["outcome"]
    if outcome not in ("success","unsupported","workload_exception","measurement_error","resource_limit","timeout","oom","native_crash","invalid_receipt"):
        raise ValueError("invalid isolated outcome")
    receipt=data["receipt"]
    if receipt is not None:
        if receipt["job"]["active_processes"]!=0 or receipt["kill_on_close"] is not True or receipt["admitted"] is not True:raise ValueError("owned process cleanup unverified")
        if receipt["limits"]!=dict(process_memory_bytes=protocol.process_memory_bytes,job_memory_bytes=protocol.job_memory_bytes,max_processes=protocol.max_processes):raise ValueError("resource limit mismatch")
        if any(len(receipt[k].encode("utf-8"))>protocol.max_log_bytes*3 for k in ("stdout","stderr")):raise ValueError("unbounded worker logs")
        classified,_=_classify(receipt,data["worker_receipt"],data["probe_id"])
        if outcome!=classified and not (outcome=="invalid_receipt" and classified=="success"):
            raise ValueError("outcome disagrees with OS/worker receipts")
    if outcome!="success":
        if data["measurement"] is not None or not data["reason"]:raise ValueError("failed worker cannot publish completed measurement")
        return
    worker=data["worker_receipt"]
    if receipt is None or receipt["exit_code"]!=0 or receipt["monitor_outcome"] is not None or worker["status"]!="success" or worker["pid"]!=receipt["launcher_pid"] or worker["token"]!=data["probe_id"]:
        raise ValueError("successful exit/admission receipt required")
    measured=data["measurement"]
    if set(measured)!={"source_p19","source_sha256","source_segment_id","boundaries","rss","windows","activation","provenance","coverage","cuda_domains"}:raise ValueError("invalid isolated measurement fields")
    source,digest=_source(TrainingPerformanceResult(measured["source_p19"]))
    if measured["source_segment_id"]!=source["segment_id"] or measured["source_sha256"]!=digest or measured["coverage"]!="partial":raise ValueError("invalid source/coverage claim")
    if protocol.saved_activation_lifetime:_validate_activation(measured["activation"],source,protocol)
    elif measured["activation"]!=_activation():raise ValueError("activation observation requires explicit opt-in")
    if measured["provenance"]["pid"]!=worker["pid"] or measured["provenance"]["device"]!=recipe["device"] or source["provenance"]["runtime"]["device"]!=recipe["device"]:raise ValueError("worker provenance mismatch")
    boundaries=measured["boundaries"]
    if measured["cuda_domains"]!=dict(pre_probe_lifetime=boundaries[0]["cuda"],
            boundary_peaks="since_most_recent_reset" if protocol.allow_peak_reset else "process_lifetime",
            reset_is_irreversible=protocol.allow_peak_reset,
            physical_vram=dict(value=None,status="unsupported",reason="allocator_counters_are_not_physical_VRAM")):raise ValueError("invalid CUDA domains")
    if not 2<=len(boundaries)<=protocol.memory_protocol.max_boundaries or boundaries[0]["boundary"]!="ready" or boundaries[-1]["boundary"]!="segment_end":raise ValueError("invalid bounded memory ledger")
    total=0;end=0
    for row in boundaries:
        if not end<=row["query_start_ns"]<=row["query_end_ns"]:raise ValueError("unordered memory queries")
        end=row["query_end_ns"];logical=row["logical"];total+=len(logical["tensors"])+len(logical["optimizer_metadata"])
        if not logical["optimizer_supported"]:raise ValueError("isolated native optimizer evidence required")
        if logical["summary"]!=_logical_summary(logical["tensors"]):raise ValueError("logical summary mismatch")
        for tensor in logical["tensors"]:
            if tensor["status"]=="measured" and tensor["logical_bytes"]!=tensor["elements"]*tensor["element_size_bytes"]:raise ValueError("logical byte mismatch")
    for boundary,kind in (("pre_microbatch","microbatch"),("post_backward","microbatch"),("pre_attempt","attempt"),("post_attempt","attempt")):
        if [r[kind+"_id"] for r in boundaries if r["boundary"]==boundary]!=[r[kind+"_id"] for r in source["observations"] if r["kind"]==kind]:raise ValueError("boundary/source ledger mismatch")
    if total>protocol.memory_protocol.max_tensor_records:raise ValueError("unbounded logical records")
    _validate_stage_pairs(boundaries)
    rss=measured["rss"]
    if len(rss["observations"])>protocol.memory_protocol.max_samples or rss["summary"]!=_rss_summary(rss["observations"],rss["reason"]):raise ValueError("invalid RSS evidence")
    last=hwm=0;backend=None
    for sample in rss["observations"]:
        for key in ("timestamp_ns","query_start_ns","query_end_ns","rss_bytes","lifetime_hwm_bytes"):_int(sample[key])
        if not last<=sample["query_start_ns"]<=sample["timestamp_ns"]<=sample["query_end_ns"] or sample["timestamp_ns"]!=(sample["query_start_ns"]+sample["query_end_ns"])//2:raise ValueError("invalid RSS query ordering")
        if sample["lifetime_hwm_bytes"]<max(hwm,sample["rss_bytes"]) and rss["reason"]!="inconsistent_native_windows_rss_hwm; raw_tuple_retained":raise ValueError("unmarked inconsistent RSS/HWM tuple")
        if sample["backend"] not in ("windows.GetProcessMemoryInfo.working_set","linux.proc_self_status") or (backend is not None and backend!=sample["backend"]):raise ValueError("RSS backend changed")
        last=sample["query_end_ns"];hwm=sample["lifetime_hwm_bytes"];backend=sample["backend"]
    _validate_windows(measured,source,protocol,data["probe_id"],worker["pid"])


@dataclass(frozen=True,init=False)
class IsolatedTrainingMemoryResult:
    __slots__=("_payload",)
    _payload:str

    def __init__(self,data):
        copied=_plain(data);self._check(copied)
        object.__setattr__(self,"_payload",json.dumps(copied,allow_nan=False,separators=(",",":")))

    @staticmethod
    def _check(data):
        try:_validate(data)
        except (KeyError,TypeError,AttributeError) as error:raise ValueError("malformed isolated P23 result") from error

    def to_dict(self):return json.loads(self._payload)
    def _export_data(self):
        data=_plain(self.to_dict());self._check(data);return data
    def diagnostics(self):
        """Independent core/peak/activation coverage without changing v1 exports."""
        data=self._export_data();measurement=data["measurement"]
        if measurement is None:
            return dict(core_status="not_executed",outcome=data["outcome"],reason=data["reason"],
                        scope="isolated_recipe_fit")
        return _memory_diagnostics(measurement["boundaries"],measurement["rss"],data["recipe"]["device"],
            isolated=True,windows=measurement["windows"],activation=measurement["activation"])
    def to_json(self,path,*,overwrite=False):
        _write_export(path,json.dumps(self._export_data(),sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n",overwrite)
    def to_csv(self,path,*,overwrite=False):
        data=self._export_data();columns=["probe_id","source_segment_id","kind","boundary","epoch_id","microbatch_id","window_id","attempt_id","query_start_ns","query_end_ns",
            "parameter_bytes","gradient_bytes","optimizer_bytes","allocated_bytes","reserved_bytes","counter_peak_allocated_bytes","counter_peak_reserved_bytes",
            "reset_performed","peak_allocated_bytes","peak_reserved_bytes","samples","applied","skipped",
            "saved_id","action","timestamp_ns","logical_bytes","storage_id","storage_bytes","registered_storage",
            "live_logical_bytes","live_storage_bytes","live_nonparameter_storage_bytes"]
        measurement=data["measurement"]
        rows=[]
        if measurement is not None:
            for r in measurement["boundaries"]:
                row={k:r.get(k) for k in columns};row["kind"]="boundary"
                for component in ("parameter","gradient","optimizer"):row[component+"_bytes"]=r["logical"]["summary"][component]["logical_bytes"]
                cuda=r["cuda"] or {}
                for key in ("allocated_bytes","reserved_bytes"):row[key]=cuda.get(key)
                for kind in ("allocated","reserved"):row["counter_peak_"+kind+"_bytes"]=cuda.get("lifetime_peak_"+kind+"_bytes")
                rows.append(row)
            for r in measurement["windows"]:rows.append(dict(r,kind="window",query_start_ns=r["start_ns"],query_end_ns=r["end_ns"]))
            activation=measurement["activation"]
            if "records" in activation:
                records={r["saved_id"]:r for r in activation["records"]}
                for event in activation["events"]:rows.append(dict(records[event["saved_id"]],**event,kind="saved_tensor"))
            for row in rows:
                row.update(probe_id=data["probe_id"],source_segment_id=measurement["source_segment_id"])
                for key in list(row):
                    if key not in columns:del row[key]
        _write_export(path,_csv_text(columns,rows),overwrite)


def _classify(receipt,worker,token):
    if receipt["monitor_outcome"] is not None:return receipt["monitor_outcome"],"parent resource/timeout monitor terminated owned job"
    if receipt["exit_code"] not in (0,1,2):return "native_crash","abnormal worker exit; OOM/root cause not inferred"
    if receipt["exit_code"]!=0:return "invalid_receipt","bootstrap/publication failed with Python exit status"
    if worker is None or worker.get("token")!=token or worker.get("pid")!=receipt["launcher_pid"]:
        return "invalid_receipt","worker token/PID/result mismatch"
    if set(worker) not in ({"token","pid","status","reason","measurement"},{"token","pid","status","reason"}) or worker["status"] not in ("success","unsupported","workload_exception","measurement_error","resource_limit","oom"):
        return "invalid_receipt","unrecognized worker outcome"
    return worker["status"],worker["reason"]


def measure_training_memory_isolated(recipe,*,protocol=None):
    """Spawn one trusted construction recipe; parent never imports the factory.

    Outcome and coverage are separate: exit success does not imply activation
    lifetime or physical VRAM coverage. No retry or parent allocator/RNG calls.
    """
    if type(recipe) is not TrainingMemoryRecipe:raise TypeError("TrainingMemoryRecipe required; no live Trainer/model transfer")
    recipe_data=TrainingMemoryRecipe(**recipe.to_dict()).to_dict()
    protocol=IsolatedTrainingMemoryProtocol() if protocol is None else protocol
    if type(protocol) is not IsolatedTrainingMemoryProtocol:raise TypeError("IsolatedTrainingMemoryProtocol required")
    if recipe_data["device"]!=protocol.memory_protocol.device:raise ValueError("recipe/protocol device mismatch")
    if protocol.allow_peak_reset and recipe_data["device"]=="cpu":raise ValueError("peak reset permission requires explicit CUDA device")
    token=str(uuid.uuid4());receipt=worker=measurement=None
    if os.name!="nt":outcome,reason="unsupported","native isolated profile verified on Windows only"
    else:
        with tempfile.TemporaryDirectory(prefix="hyso-training-memory-") as directory:
            request=dict(token=token,recipe=recipe_data,protocol=asdict(protocol),
                         package_root=str(Path(__file__).resolve().parents[2]),torch_version=str(torch.__version__))
            receipt,paths=_launch(directory,request,protocol)
            try:
                ready=_bounded_read(paths["ready"],4096)
                worker=_bounded_read(paths["result"],protocol.max_result_bytes)
                if ready!={"token":token,"pid":receipt["launcher_pid"],"phase":"job_admitted_before_import"}:worker=None
            except (OSError,ValueError,UnicodeError):worker=None
            outcome,reason=_classify(receipt,worker,token)
            if outcome=="success":measurement=worker["measurement"]
    data=dict(schema_version=1,format="hyso.performance.isolated_training_memory",feature_ids=["P23"],
        probe_id=token,recipe=recipe_data,protocol=asdict(protocol),outcome=outcome,reason=reason,
        receipt=receipt,worker_receipt=None if worker is None else {k:v for k,v in worker.items() if k!="measurement"},
        measurement=measurement,limitations=["Trusted arbitrary Python construction recipe; not a security sandbox.",
            "Windows private commit limits are not RSS limits; CUDA allocator fraction is not a physical VRAM limit.",
            "Current-process RSS excludes workers; initial loader profile requires num_workers=0.",
            "Saved-payload host lifetime is opt-in native eager coverage; original tensor/storage/device-free lifetime and physical VRAM unsupported.",
            "Checkpointing, custom autograd, nested hooks, retained/reused graphs and hidden streams are outside validated activation coverage."])
    data["sha256"]=_hash(data)
    try:return IsolatedTrainingMemoryResult(data)
    except ValueError as error:
        if outcome!="success":raise
        data.update(outcome="invalid_receipt",reason="measurement validation rejected: "+str(error)[:500],measurement=None)
        data["sha256"]=_hash({k:v for k,v in data.items() if k!="sha256"})
        return IsolatedTrainingMemoryResult(data)
