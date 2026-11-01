"""Bounded saved-payload observations for the isolated native eager profile.

Hooks never detach, clone, transfer or modify a tensor. Only the native packed
slot owns each payload; the observer keeps weak references to those slots.
Release is host saved-payload disposal, never proof of physical storage release.
"""
import math
import sys
import time
import weakref

import torch


class _Payload:
    __slots__ = ("tensor", "owner", "saved_id", "version", "__weakref__")

    def __init__(self, tensor, owner, saved_id, version):
        self.tensor, self.owner, self.saved_id, self.version = tensor, owner, saved_id, version

    def clear(self, forced=False):
        owner = self.owner
        if owner is not None:
            self.owner = None
            self.tensor = None
            owner.release(self.saved_id, forced)

    def __del__(self):
        self.clear()


class _SavedActivationRecorder:
    def __init__(self, model, max_records, max_events):
        if any(getattr(m,"inplace",False) for m in model.modules()):
            raise NotImplementedError("in-place modules are outside validated saved-payload coverage")
        self.max_records, self.max_events = max_records, max_events
        self.records, self.events, self.slots, self.storages = [], [], {}, {}
        self.registered = {self.key(t) for t in (*model.parameters(), *model.buffers())}
        self.context = None
        self.stage = None
        self.storage_serial = 0
        self.live_logical = self.live_storage = self.live_nonparameter = 0
        self.event_overflow = False
        self.query_ns = 0

    @staticmethod
    def key(tensor):
        return (str(tensor.device), tensor.untyped_storage()._cdata)

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def event(self, action, saved_id):
        if len(self.events) >= self.max_events:
            self.event_overflow = True
            return
        self.events.append(dict(action=action, saved_id=saved_id, timestamp_ns=time.perf_counter_ns(),
            live_logical_bytes=self.live_logical, live_storage_bytes=self.live_storage,
            live_nonparameter_storage_bytes=self.live_nonparameter))

    def pack(self, tensor):
        start = time.perf_counter_ns()
        frame=sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_globals.get("__name__")=="torch.utils.checkpoint":
                    raise NotImplementedError("activation checkpointing replaces saved hooks; lifetime coverage unsupported")
                frame=frame.f_back
        finally:del frame
        stage=self.stage
        if stage=="backward" and torch._C._current_autograd_node() is None:stage="backward_setup"
        if self.context is None or stage not in ("forward", "loss_metrics", "backward_setup"):
            raise NotImplementedError("saved tensor outside native training forward/loss; graph reuse/checkpointing unsupported")
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda") or tensor.is_complex():
            raise NotImplementedError("saved tensor requires native dense strided CPU/CUDA tensor")
        if len(self.records) >= self.max_records or len(self.events) >= self.max_events:
            raise OverflowError("saved tensor record/event budget exhausted")
        # Native SavedVariable skips its version check when unpack hooks exist.
        # Keep that error boundary without changing tensor/graph ownership.
        version = tensor._version
        key = self.key(tensor)
        if key not in self.storages:
            self.storage_serial += 1
            size = tensor.untyped_storage().nbytes()
            registered = key in self.registered
            self.storages[key] = [self.storage_serial, size, registered, 0]
            self.live_storage += size
            if not registered:self.live_nonparameter += size
        storage = self.storages[key]
        storage[3] += 1
        saved_id = len(self.records) + 1
        row = dict(saved_id=saved_id, **self.context, stage=stage, dtype=str(tensor.dtype),
            device=str(tensor.device), shape=list(tensor.shape), stride=list(tensor.stride()),
            elements=tensor.numel(), element_size_bytes=tensor.element_size(),
            logical_bytes=tensor.numel()*tensor.element_size(), storage_id=storage[0],
            storage_bytes=storage[1], storage_offset=tensor.storage_offset(), registered_storage=storage[2],
            pack_ns=start, unpack_count=0, release_ns=None, release_kind=None)
        self.records.append(row)
        self.live_logical += row["logical_bytes"]
        payload = _Payload(tensor, self, saved_id, version)
        self.slots[saved_id] = (weakref.ref(payload), key)
        self.event("pack", saved_id)
        self.query_ns += time.perf_counter_ns()-start
        return payload

    def unpack(self, payload):
        start = time.perf_counter_ns()
        node = torch._C._current_autograd_node()
        if node is not None and type(node).__module__ != "builtins":
            raise NotImplementedError("custom autograd/checkpoint backward is outside validated native eager coverage")
        if len(self.events) >= self.max_events:raise OverflowError("saved tensor event budget exhausted")
        if type(payload) is not _Payload or payload.owner is not self or payload.tensor is None:
            raise RuntimeError("invalid/reused saved payload")
        if payload.tensor._version != payload.version:
            raise RuntimeError("a tensor saved for backward was modified by an inplace operation")
        self.records[payload.saved_id-1]["unpack_count"] += 1
        self.event("unpack", payload.saved_id)
        self.query_ns += time.perf_counter_ns()-start
        return payload.tensor

    def release(self, saved_id, forced):
        start = time.perf_counter_ns()
        _, key = self.slots.pop(saved_id)
        row = self.records[saved_id-1]
        row.update(release_ns=start, release_kind="cleanup" if forced else "native_payload_disposal")
        self.live_logical -= row["logical_bytes"]
        storage = self.storages[key]
        storage[3] -= 1
        if storage[3] == 0:
            self.live_storage -= storage[1]
            if not storage[2]:self.live_nonparameter -= storage[1]
            del self.storages[key]
        self.event("cleanup" if forced else "release", saved_id)
        self.query_ns += time.perf_counter_ns()-start

    def close(self):
        # Abandoned graphs can contain a slot -> original tensor -> grad_fn cycle.
        # Break only owned packed slots after fit unwinds; never invoke global GC.
        for reference, _ in list(self.slots.values()):
            payload = reference()
            if payload is not None:payload.clear(forced=True)
        self.context = self.stage = None
        self.registered.clear()

    def result(self):
        if self.slots:raise NotImplementedError("saved payload survives fit; retained graph lifetime unsupported")
        if self.event_overflow:raise OverflowError("saved tensor event budget exhausted during release")
        return dict(status="partial", reason="native_eager_saved_payloads_only; not_physical_activation_memory",
            records=self.records, events=self.events,
            summary=_summary(self.records, self.events), hook_query_ns=self.query_ns,
            lifetime=dict(status="measured", scope="host_autograd_saved_payload_pack_to_native_disposal"),
            physical_activation_peak=dict(value=None, status="unsupported", reason="saved_payload_bytes_are_not_VRAM"),
            tensor_storage_lifetime=dict(value=None, status="unsupported", reason="other_alias_owners_and_device_free_not_observed"))


def _summary(records, events):
    return dict(saved_tensors=len(records), pack_events=sum(e["action"]=="pack" for e in events),
        unpack_events=sum(e["action"]=="unpack" for e in events),
        release_events=sum(e["action"]=="release" for e in events),
        total_saved_logical_bytes=sum(r["logical_bytes"] for r in records),
        peak_live_saved_logical_bytes=max((e["live_logical_bytes"] for e in events), default=0),
        peak_live_saved_storage_bytes=max((e["live_storage_bytes"] for e in events), default=0),
        peak_live_nonparameter_storage_bytes=max((e["live_nonparameter_storage_bytes"] for e in events), default=0),
        live_payloads_at_end=sum(r["release_ns"] is None for r in records))


def _validate_activation(data, source, protocol):
    if set(data)!={"status","reason","records","events","summary","hook_query_ns","lifetime","physical_activation_peak","tensor_storage_lifetime"}:
        raise ValueError("invalid saved activation schema")
    if data["status"]!="partial" or data["reason"]!="native_eager_saved_payloads_only; not_physical_activation_memory":raise ValueError("invalid activation coverage")
    if data["lifetime"]!=dict(status="measured",scope="host_autograd_saved_payload_pack_to_native_disposal") or data["physical_activation_peak"]!=dict(value=None,status="unsupported",reason="saved_payload_bytes_are_not_VRAM") or data["tensor_storage_lifetime"]!=dict(value=None,status="unsupported",reason="other_alias_owners_and_device_free_not_observed"):
        raise ValueError("invalid activation lifetime/physical domain")
    records, events = data["records"], data["events"]
    if len(records)>protocol.max_saved_tensors or len(events)>protocol.max_saved_tensor_events:raise ValueError("unbounded saved tensor ledger")
    micro={r["microbatch_id"]:r for r in source["observations"] if r["kind"]=="microbatch"}
    by_id={}
    item_sizes={"torch.bool":1,"torch.uint8":1,"torch.int8":1,"torch.int16":2,"torch.int32":4,"torch.int64":8,
        "torch.float16":2,"torch.bfloat16":2,"torch.float32":4,"torch.float64":8}
    for i,row in enumerate(records,1):
        if set(row)!={"saved_id","epoch_id","microbatch_id","window_id","stage","dtype","device","shape","stride","elements","element_size_bytes","logical_bytes","storage_id","storage_bytes","storage_offset","registered_storage","pack_ns","unpack_count","release_ns","release_kind"}:raise ValueError("invalid saved tensor record")
        linked=micro.get(row["microbatch_id"])
        if row["saved_id"]!=i or linked is None or any(row[k]!=linked[k] for k in ("epoch_id","window_id")) or row["stage"] not in ("forward","loss_metrics","backward_setup"):raise ValueError("invalid saved tensor source association")
        if row["device"]!=protocol.memory_protocol.device or not linked["start_ns"]<=row["pack_ns"]<=row["release_ns"]<=linked["end_ns"]:raise ValueError("saved payload outside source microbatch")
        for k in ("elements","element_size_bytes","logical_bytes","storage_bytes","storage_offset","pack_ns","unpack_count","release_ns","storage_id"):
            if type(row[k]) is not int or row[k]<0:raise ValueError("invalid saved tensor integer")
        if any(type(v) is not int or v<0 for v in (*row["shape"],*row["stride"])) or item_sizes.get(row["dtype"])!=row["element_size_bytes"]:raise ValueError("invalid saved tensor dtype/shape/stride")
        if row["storage_id"]<1 or type(row["registered_storage"]) is not bool or row["elements"]!=math.prod(row["shape"]) or row["logical_bytes"]!=row["elements"]*row["element_size_bytes"] or len(row["shape"])!=len(row["stride"]) or row["release_kind"]!="native_payload_disposal":raise ValueError("invalid saved tensor byte/release evidence")
        if row["elements"] and (row["storage_offset"]+sum((d-1)*s for d,s in zip(row["shape"],row["stride"]))+1)*row["element_size_bytes"]>row["storage_bytes"]:raise ValueError("saved tensor exceeds storage extent")
        by_id[i]=row
    live, storages, unpacks = {}, {}, {}
    last=0;seen_storages=set();next_pack=1
    for event in events:
        if set(event)!={"action","saved_id","timestamp_ns","live_logical_bytes","live_storage_bytes","live_nonparameter_storage_bytes"}:raise ValueError("invalid saved tensor event")
        row=by_id.get(event["saved_id"])
        if row is None or type(event["timestamp_ns"]) is not int or event["timestamp_ns"]<last:raise ValueError("unordered saved tensor event")
        if any(type(event[k]) is not int or event[k]<0 for k in ("saved_id","live_logical_bytes","live_storage_bytes","live_nonparameter_storage_bytes")):raise ValueError("invalid saved tensor event integer")
        last=event["timestamp_ns"];i=row["saved_id"];s=row["storage_id"]
        if not row["pack_ns"]<=last or (event["action"]!="release" and last>row["release_ns"]):raise ValueError("event outside saved payload lifetime")
        if event["action"]=="pack":
            if i in live or i in unpacks or i!=next_pack:raise ValueError("duplicate/unordered pack")
            next_pack+=1
            live[i]=row;unpacks[i]=0
            if s not in storages:
                if s in seen_storages or s!=len(seen_storages)+1:raise ValueError("reused/unordered storage identity")
                seen_storages.add(s)
            if s in storages and storages[s][:2]!=(row["storage_bytes"],row["registered_storage"]):raise ValueError("inconsistent saved storage aliases")
            info=storages.setdefault(s,(row["storage_bytes"],row["registered_storage"],set()));info[2].add(i)
        elif event["action"]=="unpack":
            if i not in live:raise ValueError("unpack after release/before pack")
            unpacks[i]+=1
        elif event["action"]=="release":
            if i not in live or not row["release_ns"]<=last<=micro[row["microbatch_id"]]["end_ns"]:raise ValueError("invalid native saved payload release")
            del live[i];storages[s][2].remove(i)
            if not storages[s][2]:del storages[s]
        else:raise ValueError("successful activation result cannot include cleanup/incomplete events")
        expected=(sum(r["logical_bytes"] for r in live.values()),sum(s[0] for s in storages.values()),sum(s[0] for s in storages.values() if not s[1]))
        if tuple(event[k] for k in ("live_logical_bytes","live_storage_bytes","live_nonparameter_storage_bytes"))!=expected:raise ValueError("saved tensor live/alias ledger mismatch")
    if live or len(unpacks)!=len(records) or any(unpacks[i]!=r["unpack_count"] for i,r in by_id.items()) or len(events)!=2*len(records)+sum(unpacks.values()):raise ValueError("incomplete saved tensor lifecycle")
    if records and {r["microbatch_id"] for r in records}!=set(micro):raise ValueError("incomplete activation microbatch coverage")
    if micro and not records:raise ValueError("no observed native saved tensors")
    if data["summary"]!=_summary(records,events) or type(data["hook_query_ns"]) is not int or data["hook_query_ns"]<0:raise ValueError("invalid activation summary/overhead")
