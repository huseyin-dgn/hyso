"""Sealed P23 fixture and durable phase ledger; never used by production APIs."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class Journal:
    def __init__(self, path, nonce, sha):
        self.stream = Path(path).open('xb', buffering=0)
        self.nonce, self.sha, self.rows, self.bytes = nonce, sha, 0, 0

    def emit(self, phase, **fields):
        row = dict(sequence=self.rows+1, pid=os.getpid(), nonce=self.nonce,
                   manifest_sha256=self.sha, time_ns=time.time_ns(),
                   monotonic_ns=time.perf_counter_ns(), phase=phase, **fields)
        raw = (json.dumps(row, allow_nan=False)+'\n').encode()
        if self.rows >= 2000 or self.bytes+len(raw) > 1024**2:
            raise OverflowError('bounded P23 phase journal exhausted')
        if self.stream.write(raw) != len(raw): raise OSError('short phase journal write')
        os.fsync(self.stream.fileno())
        self.rows += 1
        self.bytes += len(raw)


def materialize():
    import torch
    x, y = torch.linspace(-1, 1, 15).reshape(5, 3), torch.tensor([0, 1, 0, 1, 0])
    def row(tensor):
        raw = tensor.contiguous().numpy().tobytes()
        return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return dict(x=row(x), y=row(y), torch=str(torch.__version__), cuda_initialized=torch.cuda.is_initialized())


_journal, _contract = None, None


def tiny(*, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from hyso.training import Trainer
    actual = materialize()
    # Native worker can have a context; CPU data identity is independently checked.
    if any(actual[k] != _contract[k] for k in ('x', 'y', 'torch')):
        raise ValueError('exact CPU input identity mismatch')
    _journal.emit('fixture.entry', input_contract=actual)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, .02, total_steps=4, cycle_momentum=False)
    trainer = Trainer(model, optimizer=opt, loss_fn=torch.nn.CrossEntropyLoss(), device=device,
        amp=False, accum_steps=2, ema_decay=.8, scheduler=scheduler, grad_clip=.1)
    data = TensorDataset(torch.linspace(-1, 1, 15).reshape(5, 3), torch.tensor([0, 1, 0, 1, 0]))
    for name, module in model.named_modules():
        module.register_forward_pre_hook(lambda *a, name=name: _journal.emit('module.forward.entry', module=name))
        module.register_forward_hook(lambda *a, name=name: _journal.emit('module.forward.exit', module=name))
    _journal.emit('fixture.exit', model='Linear-Dropout-Linear', device=device)
    return dict(trainer=trainer, train_loader=DataLoader(data, batch_size=2, shuffle=True),
                val_loader=DataLoader(data, batch_size=2), stream=None)


def instrument(cls, names, journal):
    for name in names:
        original = getattr(cls, name)
        def wrapped(self, *args, _name=name, _original=original, **kwargs):
            fields = {}
            if _name == 'pack':
                fields = dict(device=str(args[0].device), shape=list(args[0].shape),
                              candidate_saved_id=len(self.records)+1, context=self.context, stage=self.stage)
            elif _name in ('unpack', 'release'):
                fields = dict(saved_id=args[0].saved_id if _name == 'unpack' else args[0])
                if _name == 'release': fields['forced_cleanup'] = args[1] if len(args) > 1 else kwargs['forced']
            elif _name in ('stage_begin', 'stage_end'): fields = dict(stage=args[0])
            journal.emit(_name+'.entry', **fields)
            try:
                result = _original(self, *args, **kwargs)
            except BaseException as exc:
                journal.emit(_name+'.exception', exception=type(exc).__name__, **fields)
                raise
            if _name == 'pack': fields['saved_id'] = result.saved_id
            journal.emit(_name+'.exit', **fields)
            return result
        setattr(cls, name, wrapped)


def loaded_identity(root, manifest):
    from p23_forensic_runner import digest
    identities = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not path or not (name == 'hyso' or name.startswith(('hyso.', 'torch.', 'numpy.')) or name in ('torch', 'numpy')):
            continue
        # These two ModuleType subclasses deliberately report relative, virtual
        # __file__ values. Bind their real implementation; never treat a cwd
        # "_ops.py" as imported code or silently skip an unknown real module.
        provider = {'torch.ops': ('torch._ops', '_Ops'), 'torch.classes': ('torch._classes', '_Classes')}.get(name)
        if provider is not None:
            if type(module).__module__ != provider[0] or type(module).__name__ != provider[1]:
                raise ValueError('unexpected virtual Torch module implementation')
            path = sys.modules[provider[0]].__file__
        path = Path(path).resolve()
        expected = (manifest['source_files'].get(path.relative_to(root/'source').as_posix())
                    if path.is_relative_to(root/'source') else manifest['runtime_files'].get(str(path)))
        if expected is None or digest(path) != expected:
            raise ValueError('unbound loaded source/runtime: '+str(path))
        identities[str(path)] = expected
    return identities


def final_admission(work, initial, reviewed, token, sha, manifest, *, cuda_initialized):
    """One stdlib/NVML capture after imports/hashing, before any CUDA context.

    No polling, new workload classification, model, tensor or CUDA operation.
    All helper processes remain in the existing worker's bounded Windows Jobs.
    """
    from p23_forensic_runner import publish, admission
    if cuda_initialized():
        raise ValueError('CUDA initialized before final capture')
    _journal.emit('final_capture.entry', cuda_initialized=False)
    raw = admission.collect_p23()
    publish(work/'final-capture-raw.json', raw)
    fresh = admission.bind_capture(raw, run_id=manifest['run_id'], manifest_sha256=sha,
        stage='worker-final', nonce=token, worker_pid=os.getpid())
    publish(work/'final-capture.json', fresh)
    review = admission.refreshed_review(initial, reviewed, fresh,
        run_id=manifest['run_id'], manifest_sha256=sha, nonce=token, worker_pid=os.getpid())
    publish(work/'final-review.json', review)
    if cuda_initialized():
        raise ValueError('CUDA initialized during stdlib final capture')
    context = dict(run_id=manifest['run_id'], manifest_sha256=sha,
                   stage='worker-final', nonce=token, worker_pid=os.getpid())
    decision = admission.evaluate_bound(fresh, review, **context)
    publish(work/'admission-worker-final.json', decision)
    _journal.emit('final_capture.exit', eligible=decision['eligible'],
                  capture_age_ns=time.time_ns()-fresh['collection_start_time_ns'],
                  cuda_initialized=False)
    if not decision['eligible']:
        raise ValueError('final GPU admission rejected: '+','.join(decision['blockers']))
    # Include publication/journaling time in freshness, without collecting again.
    if not admission.evaluate_bound(fresh, review, **context)['eligible']:
        raise ValueError('final GPU admission expired before recipe entry')


def run(root, work, mode, token, sha, manifest):
    from p23_forensic_runner import publish, digest, read, admission
    global _journal, _contract
    _journal = Journal(work/'worker-phases.jsonl', token, sha)
    _contract = manifest['input_contract']
    _journal.emit('admitted', source_tree_sha256=admission.policy.canonical_hash(manifest['source_files']),
        runtime_tree_sha256=admission.policy.canonical_hash(manifest['runtime_files']), input_contract=_contract,
        cuda_imported='torch' in sys.modules)
    if mode not in ('native', 'cpu-recipe'):
        if 'torch' in sys.modules: raise RuntimeError('offline case imported Torch')
        if mode in ('tree', 'timeout', 'outer-timeout'):
            child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c',
                "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(120)']); print(p.pid,flush=True); time.sleep(120)"],
                creationflags=subprocess.CREATE_NO_WINDOW)
            _journal.emit('owned.descendant', pid_child=child.pid)
            # Wait for grandchild creation before testing tree cleanup.
            time.sleep(.3)
        if mode in ('timeout', 'outer-timeout'): time.sleep(120)
        if mode == 'exception': raise RuntimeError('P23 CPU injected workload exception')
        if mode in ('failfast', 'collector-timeout', 'collector-limit'):
            _journal.emit('fault_injection.entry', kind='RaiseFailFastException')
            api = c.WinDLL('kernel32', use_last_error=True)
            from p23_forensic_windows import ExceptionRecord
            record = ExceptionRecord(code=0xc0000409, flags=1, count=1)
            record.information[0] = 2
            api.RaiseFailFastException.argtypes = [c.POINTER(ExceptionRecord), c.c_void_p, c.c_ulong]
            api.RaiseFailFastException(c.byref(record), None, 0)
            raise AssertionError('fail-fast returned')
        if mode == 'memory':
            api = c.WinDLL('kernel32', use_last_error=True)
            api.VirtualAlloc.argtypes = [c.c_void_p, c.c_size_t, c.c_ulong, c.c_ulong]
            api.VirtualAlloc.restype = c.c_void_p
            api.VirtualFree.argtypes = [c.c_void_p, c.c_size_t, c.c_ulong]
            api.VirtualFree.restype = c.c_int
            pointer = api.VirtualAlloc(None, 8*1024**3, 0x3000, 4)
            error = c.get_last_error()
            if pointer:
                api.VirtualFree(pointer, 0, 0x8000)
                raise RuntimeError('over-budget private commit unexpectedly allowed')
            _journal.emit('memory_limit.denied', requested=8*1024**3, error=error)
        if mode == 'logs': os.write(1, b'x'*70000)
        _journal.emit('worker.complete', cuda_imported=False)
        publish(work/'result.json', dict(cpu_only=True, cuda_imported=False, pid=os.getpid(), nonce=token, manifest_sha256=sha))
        return
    device = 'cuda:0' if mode == 'native' else 'cpu'
    capture, review = (read(work/'capture.json'), read(work/'review.json')) if mode == 'native' else (None, None)
    def admit():
        if mode == 'cpu-recipe': return
        value = admission.evaluate_bound(capture, review, run_id=manifest['run_id'],
            manifest_sha256=sha, stage='controller-preflight',
            nonce=capture['binding']['nonce'], worker_pid=0)
        if not value['eligible']: raise ValueError('fresh GPU admission expired/rejected')
    admit()
    sys.path[:0] = [str(root/'source'), manifest['site_packages']]
    _journal.emit('torch.import.entry')
    import torch
    from hyso.performance import _p23_diagnostic_recipe as recipe
    from hyso.performance import _p23_diagnostic_saved as saved
    _journal.emit('torch.import.exit', torch=str(torch.__version__), cuda_initialized=torch.cuda.is_initialized())
    if torch.cuda.is_initialized(): raise ValueError('CUDA initialized before final admission')
    identities = loaded_identity(root, manifest)
    publish(work/'loaded-python-identities.json', identities)
    # DLL paths are checked by the actual loaded process image, not guessed exports.
    from ctypes import wintypes as w
    api = c.WinDLL('kernel32', use_last_error=True)
    api.GetModuleHandleW.argtypes, api.GetModuleHandleW.restype = [w.LPCWSTR], w.HMODULE
    api.GetModuleFileNameW.argtypes, api.GetModuleFileNameW.restype = [w.HMODULE, w.LPWSTR, w.DWORD], w.DWORD
    dlls = {}
    bound_paths = {Path(p).resolve(): expected for p, expected in manifest['runtime_files'].items()}
    for path, expected in manifest['runtime_files'].items():
        if Path(path).suffix.lower() not in ('.dll', '.pyd'): continue
        handle = api.GetModuleHandleW(Path(path).name)
        if not handle: continue  # nvcuda may be loaded only on first initialization
        buf = c.create_unicode_buffer(32768)
        if not api.GetModuleFileNameW(handle, buf, len(buf)):
            raise ValueError('loaded DLL identity mismatch: '+path)
        actual = Path(buf.value).resolve()
        expected = bound_paths.get(actual)
        if expected is None or (str(actual) not in dlls and digest(actual) != expected):
            raise ValueError('unbound/changed loaded DLL: '+str(actual))
        dlls[str(actual)] = expected
    publish(work/'loaded-native-identities.json', dlls)
    instrument(saved._SavedActivationRecorder, ('pack', 'unpack', 'release', 'close'), _journal)
    instrument(recipe._IsolatedRecorder, ('microbatch_begin', 'stage_begin', 'stage_end', 'attempt_end', 'close'), _journal)
    original_reset = torch.cuda.reset_peak_memory_stats
    def reset(*args, **kwargs):
        _journal.emit('peak_reset.entry')
        result = original_reset(*args, **kwargs)
        _journal.emit('peak_reset.exit')
        return result
    torch.cuda.reset_peak_memory_stats = reset
    protocol = recipe.IsolatedTrainingMemoryProtocol(
        memory_protocol=recipe.TrainingMemoryProtocol(device=device),
        timeout_seconds=60, process_memory_bytes=4*1024**3, job_memory_bytes=8*1024**3,
        max_processes=4, cuda_memory_fraction=.15, max_result_bytes=2*1024**2,
        max_log_bytes=65536, deterministic=True, allow_peak_reset=True,
        max_windows=32, saved_activation_lifetime=True, max_saved_tensors=256, max_saved_tensor_events=1024)
    request = dict(recipe=recipe.TrainingMemoryRecipe('p23_forensic_worker:tiny', device=device,
        factory_search_path=str(root/'source'), seed=71, epochs=2).to_dict(),
        protocol=recipe.asdict(protocol), torch_version='2.5.1+cu121', token=token)
    if mode == 'native':
        final_admission(work, capture, review, token, sha, manifest,
                        cuda_initialized=torch.cuda.is_initialized)
    _journal.emit('recipe.entry', device=device)
    measurement = recipe._run_recipe(request)
    _journal.emit('recipe.exit', device=device)
    recipe._validate_activation(measurement['activation'], measurement['source_p19'], protocol)
    # Diagnostic completion requires real pack/unpack/native payload disposal.
    summary = measurement['activation']['summary']
    if not summary['pack_events'] or not summary['unpack_events'] or summary['release_events'] != summary['pack_events'] or summary['live_payloads_at_end']:
        raise ValueError('incomplete saved-payload diagnostic coverage')
    if len(measurement['windows']) != 4 or any(r['status'] != ('measured' if mode == 'native' else 'unsupported') for r in measurement['windows']):
        raise ValueError('incomplete memory windows')
    loaded_identity(root, manifest)
    publish(work/'result.json', dict(pid=os.getpid(), nonce=token, manifest_sha256=sha,
        diagnostic_only=True, p23_acceptance='PARTIAL', measurement=measurement,
        cuda_initialized=torch.cuda.is_initialized(), cpu_only=mode == 'cpu-recipe'))
    _journal.emit('worker.complete', diagnostic_only=True)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--materialize': raise ValueError('internal materialization only')
    sys.path.insert(0, sys.argv[2])
    print(json.dumps(materialize()))
