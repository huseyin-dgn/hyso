"""Small allocator history acceptance; independent counter reset is TEST ONLY."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,profile_operators,profile_layers,LayerProfileResult,
    measure_layer_overhead,LayerOverheadProtocol,LayerOverheadResult)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--admit',required=True);p.add_argument('--case')
    args=p.parse_args();deadline=time.monotonic()+15
    while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.01)
    assert Path(args.admit).exists()
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1,0)
    model=torch.nn.Linear(2,2).cuda().train();model.weight.grad=torch.ones_like(model.weight)
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9);opt.step()
    gradient=model.weight.grad;grad=gradient.clone();momentum=opt.state[model.weight]['momentum_buffer'];mom=momentum.clone()
    weights={k:v.clone() for k,v in model.state_dict().items()};handle=model.register_forward_hook(lambda *a:None);hooks=dict(model._forward_hooks)
    stream=torch.cuda.Stream();previous=torch.cuda.current_stream();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();peak=torch.cuda.max_memory_allocated()
    output=Path(args.output);reference=output.with_name(output.stem+'-reference.json');cases=[]
    try:
        with torch.cuda.stream(stream):
            if args.case=='overhead':
                recipe=ColdStartRecipe('layer_diagnostic_factories:NestedRetaining',input_shape=(1,1,2,8),
                    factory_search_path=Path(__file__).parent,device='cuda:0')
                overhead=measure_layer_overhead(recipe,protocol=LayerOverheadProtocol(trials=2),capture_protocol=
                    ProfilingProtocol(warmup_iterations=2,measure_iterations=3,profile_memory=True,allocation_lifetimes=True,
                        cpu_memory_limit_bytes=4*1024**3,input_limit_bytes=32*1024**2,cuda_memory_fraction=.1,timeout_seconds=45))
                observed=overhead.to_dict();assert observed['measurements']['matched_context']
                if observed['measurements']['status']=='unsupported':assert observed['measurements']['signed_overhead_percent'] is None
                for pair in observed['raw_evidence']['pairs']:
                    assert pair['instrumented']['measurements']['allocator_diagnostics']['cuda']['status']=='measured'
                path=output.with_name(output.stem+'-result.json');overhead.to_json(path);overhead.to_csv(path.with_suffix('.csv'))
                assert LayerOverheadResult.from_csv(path.with_suffix('.csv')).to_dict()==observed
                cases.append(dict(kind='overhead',measurements=observed['measurements']))
            for factory,shape,warmup,measured,kwargs in (() if args.case=='overhead' else (
                ('ReferencePeak',(1,1,1,1024),1,1,dict(reference_path=str(reference))),
                ('NestedRetaining',(1,1,2,8),1,2,{}))):
                recipe=ColdStartRecipe('layer_diagnostic_factories:'+factory,input_shape=shape,model_kwargs=kwargs,
                    factory_search_path=Path(__file__).parent,device='cuda:0')
                source=profile_operators(recipe,protocol=ProfilingProtocol(warmup_iterations=warmup,measure_iterations=measured,
                    profile_memory=True,allocation_lifetimes=True,cpu_memory_limit_bytes=4*1024**3,
                    input_limit_bytes=32*1024**2,cuda_memory_fraction=.1,timeout_seconds=45))
                result=profile_layers(source);data=result.to_dict();d=data['measurements']['allocator_diagnostics'];cuda=d['cuda']
                assert cuda['status']=='measured' and cuda['physical_peak_bytes'] is None
                if factory=='ReferencePeak':
                    independent=json.loads(reference.read_text());row=cuda['invocations'][0]
                    assert row['baseline_allocated_bytes']==independent['baseline']
                    assert row['peak_allocated_bytes']==independent['peak']
                    assert row['end_allocated_bytes']==independent['after']
                    assert independent['same_storage'] and row['retained_new_allocation_bytes']==4096
                else:
                    assert len(cuda['invocations'])==6
                    assert all(r['retained_new_allocation_bytes']==512 for r in cuda['invocations'])
                    assert len(next(m for m in d['modules'] if m['module_id']=='m1')['invocation_ids'])==4
                assert source.to_dict()['raw_evidence']['safety']['allocation_history_closed']
                assert data['raw_evidence']['allocation_evidence']['collector_peak_reset_called'] is False
                path=output.with_name(output.stem+'-'+factory+'.json');result.to_json(path);result.to_csv(path.with_suffix('.csv'))
                assert LayerProfileResult(json.loads(path.read_text())).to_dict()==data
                cases.append(dict(factory=factory,cuda=cuda))
            assert torch.cuda.current_stream()==stream
        assert torch.cuda.current_stream()==previous
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert torch.cuda.memory_allocated()==allocated and torch.cuda.memory_reserved()==reserved and torch.cuda.max_memory_allocated()==peak
        assert model.training and model.weight.grad is gradient and torch.equal(gradient,grad) and dict(model._forward_hooks)==hooks
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
        assert opt.state[model.weight]['momentum_buffer'] is momentum and torch.equal(momentum,mom)
        output.write_text(json.dumps(dict(status='passed',cases=cases,parent_state_preserved=True),indent=2))
        print(json.dumps(dict(status='passed',cases=len(cases),parent_state_preserved=True,
            absolute_allocator_peak='not_repeated_in_overhead_experiment' if args.case=='overhead' else 'independent_counter_verified',
            physical_peak='unsupported',kernel='unsupported_backend_capability')))
    finally:handle.remove()


if __name__=='__main__':main()
