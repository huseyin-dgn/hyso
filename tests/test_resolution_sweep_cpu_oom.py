"""P07 must stop admission after a CPU allocation failure, without real OOM."""
import pytest
import torch
from hyso.performance import SweepProtocol,run_sweep


@pytest.mark.parametrize('error',[
    RuntimeError('DefaultCPUAllocator: not enough memory: you tried to allocate 4096 bytes.'),
    RuntimeError("DefaultCPUAllocator: can't allocate memory: 4096 bytes"),
    MemoryError('injected input allocation failure; no real resource exhaustion'),
])
def test_p07_cpu_allocation_failure_stops_resolution_admission(error):
    model=torch.nn.Sequential(torch.nn.Conv2d(1,2,1),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
    called=[]
    def factory(shape,dtype,device):
        called.append(shape)
        if shape[2:]==(7,9):raise error
        return torch.ones(shape,dtype=dtype,device=device)
    result=run_sweep(model,factory,axis='resolution',points=[(5,7),(7,9),(9,13)],
        protocol=SweepProtocol(warmup_iterations=0,measure_iterations=1,continue_on_error=True))
    rows=result.to_dict()['raw_evidence']['points']
    assert [r['status'] for r in rows]==['measured','failed','not_run']
    assert rows[1]['reason'].startswith('out_of_memory:')
    assert rows[1]['result'] is None and rows[2]['result'] is None
    assert called==[(1,1,5,7),(1,1,7,9)]
