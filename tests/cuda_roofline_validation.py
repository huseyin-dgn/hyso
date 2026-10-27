"""Single small, owned P18 CUDA check; no hardware calibration or stress test."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,profile_operators,analyze_roofline,RooflineResult)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--admit',required=True);parser.add_argument('--case')
    args=parser.parse_args();deadline=time.monotonic()+15
    while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.01)
    assert Path(args.admit).exists() and not torch.cuda.is_initialized()
    before=torch.get_rng_state().clone()
    recipe=ColdStartRecipe('profiling_factories:tiny_cuda',input_shape=(2,2,3,3),device='cuda:0',
        factory_search_path=Path(__file__).parent)
    profile=profile_operators(recipe,protocol=ProfilingProtocol(warmup_iterations=0,measure_iterations=1,
        cpu_memory_limit_bytes=4*1024**3,input_limit_bytes=32*1024**2,cuda_memory_fraction=.1,timeout_seconds=60))
    result=analyze_roofline(profile);data=result.to_dict();m=data['measurements']
    assert m['logical_flops']['value']==402
    assert m['timing']['status']=='measured' and m['timing']['clock']=='torch.cuda.Event.elapsed_time_ms'
    assert m['achieved_compute']['status']=='derived' and m['achieved_compute']['value']>0
    assert abs(m['achieved_compute']['value']-402/m['timing']['total_seconds']/1e9)<1e-9
    assert m['hardware_roof_calibration']['status']==m['bottleneck']['status']=='unsupported'
    assert m['hardware_headroom']['value'] is None
    assert not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state())
    path=Path(args.output);result.to_json(path);result.to_csv(path.with_suffix('.csv'))
    assert RooflineResult.from_csv(path.with_suffix('.csv')).to_dict()==data
    print(json.dumps(dict(status='passed',logical_flops_per_batch=402,gflops_per_second=m['achieved_compute']['value'],
        elapsed_seconds=m['timing']['total_seconds'],traffic_status=m['traffic']['status'],
        actual_DRAM_traffic='unsupported',hardware_calibration='unsupported',parent_cuda_uninitialized=True)))


if __name__=='__main__':main()
