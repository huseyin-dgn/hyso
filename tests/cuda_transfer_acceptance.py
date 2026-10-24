"""Small RTX transfer matrix; abort on any failure, never retry a native crash.

Run directly: python tests/cuda_transfer_acceptance.py --output <new directory>
Uses only PyTorch and HYSo, with no occupancy gate or default resource quotas.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import torch
    from hyso.performance import TransferProtocol,TransferResult,TransferError,measure_transfers
    assert torch.cuda.is_available(),'native CUDA acceptance was not run'
    torch.set_num_threads(1);torch.cuda.set_device(0)
    model=torch.nn.Linear(2,2).cuda().train()
    model.weight.grad=torch.ones_like(model.weight);grad=model.weight.grad
    hook=model.register_forward_hook(lambda *args:None)
    hooks=dict(model._forward_hooks);weights={k:v.clone() for k,v in model.state_dict().items()}
    grad_value=grad.clone();stream=torch.cuda.Stream();torch.cuda.synchronize()
    cases=[]
    report=dict(environment=dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),source=str(root)),cases=cases)
    def save():
        (args.output/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for pinned in (False,True):
            for nonblocking in (False,True):
                name=f'pinned_{pinned}_nonblocking_{nonblocking}'
                state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
                allocator=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),
                    torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
                previous=torch.cuda.current_stream()
                try:
                    with torch.cuda.stream(stream):
                        result=measure_transfers(protocol=TransferProtocol(pinned=pinned,non_blocking=nonblocking,
                            warmup_iterations=1,measure_iterations=2,repeats=1,probe_overlap=pinned and nonblocking))
                        assert torch.cuda.current_stream()==stream
                except TransferError as error:
                    cases.append(dict(id=name,status='FAILED',diagnostics=error.diagnostics));save()
                    raise  # Includes native crashes; no retry or next case.
                finally:
                    assert torch.cuda.current_stream()==previous
                    assert random.getstate()==state[0] and np.array_equal(np.random.get_state()[1],state[1][1])
                    assert torch.equal(torch.get_rng_state(),state[2]) and torch.equal(torch.cuda.get_rng_state(),state[3])
                    assert allocator==(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),
                        torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
                    assert model.training and model.weight.grad is grad and torch.equal(grad,grad_value)
                    assert dict(model._forward_hooks)==hooks
                    assert all(torch.equal(model.state_dict()[k],v) for k,v in weights.items())
                d=result.to_dict();raw=d['raw_evidence'];m=d['measurements']
                assert Path(d['provenance']['environment']['package_source']).resolve().is_relative_to(root)
                assert d['actual']==dict(completed_copies=4,warmup_copies=2,validation_copies=4,pending_copies=0)
                assert raw['worker']['process_terminated'] and raw['worker']['exit_code']==0
                assert all(d['protocol']['configuration'][key] is None for key in
                    ('cpu_memory_limit_bytes','input_limit_bytes','cuda_memory_fraction'))
                for row in raw['copies']:
                    assert row['transfer_bytes']==1048576 and row['equal'] and row['terminal_complete']
                    assert row['host_pinned'] is pinned and row['non_blocking_requested'] is nonblocking
                    assert row['start_stream_id']==row['end_stream_id']==raw['stream_id']
                assert m['transfer_status']=='measured'
                for summary in m['directions']:
                    assert summary['completed_copies']==2 and summary['bytes']==2*1048576
                    assert summary['effective_bandwidth_gb_s']>0
                    assert summary['completed_bytes_per_second']/1e9==summary['effective_bandwidth_gb_s']
                assert m['overlap']['status']=='unsupported' and m['overlap']['value'] is None
                if m['asynchronous_execution']['value'] is True:
                    assert m['asynchronous_execution']['observed_copy_indices']
                    assert all(not raw['copies'][i]['start_complete_after_enqueue']
                        for i in m['asynchronous_execution']['observed_copy_indices'])
                else: assert m['asynchronous_execution']['status']=='unsupported'
                path=args.output/(name+'.json');result.to_json(path);result.to_csv(path.with_suffix('.csv'))
                assert TransferResult(json.loads(path.read_text())).to_dict()==d
                with path.with_suffix('.csv').open(newline='',encoding='utf-8') as f:
                    assert [json.loads(r['payload']) for r in csv.DictReader(f)]==raw['copies']
                cases.append(dict(id=name,status='PASS',directions=m['directions'],
                    asynchronous_execution=m['asynchronous_execution'],overlap=m['overlap'],
                    probe=raw['overlap_probe'],resources=raw['resources']));save()
        report.update(transfer_bandwidth='PASS',host_asynchrony=('OBSERVED' if any(
            c['asynchronous_execution']['value'] is True for c in cases) else 'UNSUPPORTED'),hardware_overlap='UNSUPPORTED')
        save();print(json.dumps(report,allow_nan=False))
    finally: hook.remove()


if __name__=='__main__':main()
