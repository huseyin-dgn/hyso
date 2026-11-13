"""Real Windows RSS gate in fresh, externally owned subprocesses.

No Linux-native claim. This harness owns worker launch/termination, not the API.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import weakref

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--installed-target',type=Path)
parser.add_argument('--worker-case',choices=('touch','failure','crash'))
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(args.installed_target.resolve() if args.installed_target else ROOT))
import torch
from hyso.performance import measure_memory, MemoryProtocol, measure_footprint
from hyso.performance import memory as engine

if platform.system()!='Windows': raise RuntimeError('Windows native gate requires actual Windows')
torch.set_num_threads(1)


def worker():
    if args.worker_case=='crash':
        measure_memory(lambda:os._exit(7),protocol=MemoryProtocol(interval_seconds=.002))
        raise AssertionError('crashed process must not produce result')
    if args.worker_case=='failure':
        def fail():
            time.sleep(.01)
            raise LookupError('controlled workload error')
        before=torch.get_rng_state()
        try: measure_memory(fail,protocol=MemoryProtocol(interval_seconds=.002))
        except LookupError: pass
        else: raise AssertionError('expected failure')
        assert not [t for t in threading.enumerate() if t.name=='hyso-rss-sampler']
        assert not engine._RUN_LOCK.locked() and torch.equal(before,torch.get_rng_state())
        return dict(error_cleanup=True,pid=os.getpid(),sampler_threads=0)
    baseline=measure_memory().to_dict()
    import ctypes
    cache_before=len(ctypes._pointer_type_cache)
    for _ in range(20): engine._windows_rss()
    cache_after=len(ctypes._pointer_type_cache)
    assert cache_before==cache_after, 'native RSS queries must not grow ctypes pointer-type cache'
    original=engine._observe
    held=[]; observed=threading.Event(); phase=[]; refs=[]
    def observe():
        row=original()
        if held: phase.append(row); observed.set()
        return row
    engine._observe=observe
    def workload():
        tensor=torch.empty(16*1024*1024,dtype=torch.float32)
        tensor.fill_(2.5)
        refs.append(weakref.ref(tensor)); held.append(tensor)
        assert observed.wait(10)
        held.clear(); del tensor
    try: measured=measure_memory(workload,protocol=MemoryProtocol(interval_seconds=.002)).to_dict()
    finally: engine._observe=original
    assert all(ref() is None for ref in refs)
    before=baseline['measurements']['current_rss_bytes']['value']
    assert phase and max(row['rss_bytes'] for row in phase)>before
    rows=measured['raw_evidence']['observations']
    assert measured['measurements']['sampled_peak_rss_bytes']['value']==max(row['rss_bytes'] for row in rows)
    assert rows[-1]['lifetime_hwm_bytes']>=max(row['rss_bytes'] for row in rows)
    assert measured['protocol']['lifetime_peak_reset'] is False
    assert not torch.cuda.is_initialized()
    assert not [t for t in threading.enumerate() if t.name=='hyso-rss-sampler']
    model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2))
    artifact=args.output.with_suffix('.state.pt')
    footprint=measure_footprint(model,serialize_to=artifact,verify_roundtrip=True).to_dict()
    assert footprint['raw_evidence']['artifact']['size_bytes']==artifact.stat().st_size
    return dict(pid=os.getpid(),baseline=baseline,measured=measured,held_phase=phase,footprint=footprint,
                allocation_bytes=16*1024*1024*4,live_tensor_released=True,cuda_initialized=False,
                pointer_type_cache_before=cache_before,pointer_type_cache_after=cache_after)


if args.worker_case:
    payload=worker()
    payload['platform']=platform.platform()
    payload['source']=str(Path(engine.__file__).resolve())
    if args.installed_target: assert Path(engine.__file__).resolve().is_relative_to(args.installed_target.resolve())
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
else:
    records=[]
    for case in ('touch','failure','crash'):
        output=args.output.with_name(args.output.stem+'-'+case+'.json')
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker-case',case,'--output',str(output)]
        if args.installed_target: command+=['--installed-target',str(args.installed_target.resolve())]
        start=time.time()
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1')) as process:
            try: stdout,stderr=process.communicate(timeout=60)
            except BaseException:
                process.kill(); process.communicate()
                raise
            expected=7 if case=='crash' else 0
            assert process.returncode==expected,(case,stdout,stderr)
            assert process.poll() is not None
        if case=='crash': assert not output.exists()
        record=dict(id=case,status='PASS',argv=command,exit_code=process.returncode,wall_seconds=time.time()-start,
                    output=str(output) if case!='crash' else None,process_terminated=True)
        records.append(record)
        print(case,'PASS',flush=True)
    args.output.write_text(json.dumps(dict(platform=platform.platform(),python=sys.version,torch=str(torch.__version__),
        verified_platform='Windows',linux_native='not_run',cases=records),indent=2),encoding='utf-8')
