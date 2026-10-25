"""One small owned CUDA experiment; built-in allocation trace, no saved hooks."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,LayerOverheadProtocol,
    measure_layer_overhead,profile_layers,OperatorProfileResult,LayerOverheadResult)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--admit',required=True);p.add_argument('--case')
    args=p.parse_args();deadline=time.monotonic()+15
    while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.01)
    assert Path(args.admit).exists()
    # Test-only explicit quota; never part of the normal user defaults.
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1,0)
    model=torch.nn.Linear(2,2).cuda().train();model.weight.grad=torch.ones_like(model.weight)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9);optimizer.step()
    gradient=model.weight.grad;gradient_copy=gradient.clone();weights={k:v.clone() for k,v in model.state_dict().items()}
    momentum=optimizer.state[model.weight]['momentum_buffer'];momentum_copy=momentum.clone()
    handle=model.register_forward_hook(lambda m,a,o:None);hooks=dict(model._forward_hooks)
    stream=torch.cuda.Stream();previous=torch.cuda.current_stream()
    py=random.getstate();ns=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();peak=torch.cuda.max_memory_allocated()
    recipe=ColdStartRecipe('layer_diagnostic_factories:Memory',factory_search_path=Path(__file__).parent,
        input_shape=(2,2,3,3),device='cuda:0')
    try:
        with torch.cuda.stream(stream):
            result=measure_layer_overhead(recipe,protocol=LayerOverheadProtocol(trials=2),capture_protocol=
                ProfilingProtocol(warmup_iterations=2,measure_iterations=3,profile_memory=True,timeout_seconds=45,
                    cpu_memory_limit_bytes=4*1024**3,input_limit_bytes=32*1024**2,cuda_memory_fraction=.1))
            data=result.to_dict();assert data['measurements']['matched_context']
            source=OperatorProfileResult(data['raw_evidence']['pairs'][-1]['instrumented']);layers=profile_layers(source)
            m=layers.to_dict()['measurements'];leaf=next(r for r in m['modules'] if r['path']=='leaf')
            assert m['cuda_event_timing']['status']=='measured'
            assert leaf['memory']['cuda']['status']=='measured'
            assert leaf['memory']['cuda']['inclusive']['allocation_bytes']>=3*2*36*4
            assert leaf['memory']['cuda']['inclusive']['net_change_bytes']==(
                leaf['memory']['cuda']['inclusive']['allocation_bytes']-leaf['memory']['cuda']['inclusive']['deallocation_bytes'])
            assert leaf['memory']['cuda']['peak_bytes'] is leaf['memory']['cuda']['retained_bytes'] is None
            high=leaf['memory']['cuda']['inclusive']['max_invocation_observed_net_high_water']
            assert high['status'] in ('derived','partial') and high['upper_bound_bytes']>0
            assert high['lower_bound_bytes']<=high['upper_bound_bytes']
            experiment=data['measurements']
            assert experiment['selection']['start_index']==2 and experiment['selection']['forward_windows_per_worker']==3
            assert experiment['uncertainty']['pairs']==2
            if experiment['status']=='unsupported':assert experiment['signed_overhead_percent'] is None
            assert torch.cuda.current_stream()==stream
        assert torch.cuda.current_stream()==previous
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert random.getstate()==py and np.array_equal(ns[1],np.random.get_state()[1]) and ns[2:]==np.random.get_state()[2:]
        assert torch.cuda.memory_allocated()==allocated and torch.cuda.memory_reserved()==reserved and torch.cuda.max_memory_allocated()==peak
        assert model.training and model.weight.grad is gradient and torch.equal(gradient,gradient_copy) and dict(model._forward_hooks)==hooks
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
        assert optimizer.state[model.weight]['momentum_buffer'] is momentum and torch.equal(momentum,momentum_copy)
        output=Path(args.output);result.to_json(output);result.to_csv(output.with_suffix('.csv'))
        assert LayerOverheadResult.from_csv(output.with_suffix('.csv')).to_dict()==data
        layers.to_json(output.with_name(output.stem+'-layers.json'));layers.to_csv(output.with_name(output.stem+'-layers.csv'))
        print(json.dumps(dict(status='passed',memory='observed_allocation_attribution',peak='unsupported',
            overhead=data['measurements']['status'],overhead_reason=data['measurements']['reason'],parent_state_preserved=True,
            observed_net_high_water=high,uncertainty=data['measurements']['uncertainty'],
            kernel_attribution=m['cuda_layer_status'])))
    finally:handle.remove()


if __name__=='__main__':main()
