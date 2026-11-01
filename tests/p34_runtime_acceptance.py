"""One short CPU + physical GPU comparison, with complete export/cleanup checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hyso.performance import WorkloadRecipe,DeviceProtocol,compare_devices,RuntimeResult


def assert_pid_absent(pid):
    import ctypes as c
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
    api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL
    handle=api.OpenProcess(0x1000,False,pid)
    if handle:
        api.CloseHandle(handle);raise AssertionError('owned worker PID still exists')
    assert c.get_last_error()==87  # ERROR_INVALID_PARAMETER: no such PID.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=False)
    recipe=WorkloadRecipe('device_acceptance_factories:cnn',search_path=str(ROOT/'tests'))
    protocol=DeviceProtocol(warmup=3,samples=20,seed=34,precision='float32',
        startup_timeout_seconds=60,step_timeout_seconds=20,max_runtime_seconds=45,
        process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4)
    result=compare_devices(recipe,['cpu',args.gpu_uuid],args.evidence/'workers',protocol=protocol)
    j=args.evidence/'comparison.json';c=args.evidence/'comparison.csv'
    result.to_json(j);result.to_csv(c)
    assert RuntimeResult.from_json(j).to_dict()==result.to_dict()
    assert RuntimeResult.from_csv(c).to_dict()==result.to_dict()
    data=result.to_dict();a=data['analysis'];assert a['comparable'],a
    for run,row in zip(data['evidence']['runs'],a['devices']):
        assert row['status']=='measured' and run['report']['status']=='completed'
        assert run['controller']['exit_code']==0 and run['controller']['reason'] is None
        assert run['controller']['cleanup']['active_processes']==0
        assert_pid_absent(run['controller']['launcher_pid'])
        assert row['identity']['execution']['output_dtype']=='torch.float32'
        assert row['energy']['value'] is None and row['quality']['value'] is None
    assert a['axes']['latency']['comparable'] and a['axes']['throughput']['comparable']
    assert a['axes']['logical_model_bytes']['comparable']
    assert not a['axes']['rss_after_bytes']['comparable']
    assert not a['axes']['cuda_peak_allocated_bytes']['comparable']
    assert a['ranking'] is None
    (args.evidence/'summary.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
    print(json.dumps(a,indent=2),flush=True)


if __name__=='__main__':main()
