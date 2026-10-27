"""One small P14 capture under nsys --pytorch=autograd-nvtx.

No torch.profiler nesting: Nsight enables emit_nvtx at the Torch import.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sqlite',type=Path,help='validate saved evidence only; no GPU execution')
    args=parser.parse_args()
    if args.sqlite:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from hyso.performance import analyze_nsys_operators
        result=analyze_nsys_operators(args.sqlite,capture_path=args.output/'capture.json')
        assert result['status']=='complete' and result['mapped_kernel_count']==result['kernel_count']==6
        expected={'aten::linear':4,'aten::addmm':4,'aten::t':4,'aten::transpose':4,
            'aten::as_strided':4,'aten::relu':2,'aten::clamp_min':2}
        assert {o['name']:o['count'] for o in result['operator_summary']}==expected
        for index in range(2):
            calls=[o for o in result['operators'] if o['forward_index']==index]
            assert {name:sum(o['name']==name for o in calls) for name in expected}=={k:v//2 for k,v in expected.items()}
        ops={o['nvtx_id']:o for o in result['operators']}
        for k in result['kernels']:
            chain=[ops[i]['name'] for i in k['operator_chain_nvtx_ids']]
            assert chain in (['aten::linear','aten::addmm'],['aten::relu','aten::clamp_min'])
            assert k['duration_us']>0 and k['operator_nvtx_id']==k['operator_chain_nvtx_ids'][-1]
        assert any(k['start_ns']>ops[k['operator_nvtx_id']]['end'] for k in result['kernels'])
        assert abs(sum(o['exclusive_kernel_us'] for o in result['operators'])-result['total_kernel_us'])<1e-9
        (args.output/'operators-nsys.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(status='PASS',kernel_count=result['kernel_count'],
            total_kernel_us=result['total_kernel_us'],mapped_kernel_count=result['mapped_kernel_count'])))
        return
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(0)
    token=str(uuid.uuid4())
    model=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,8)).cuda().eval()
    with torch.no_grad():
        for p in model.parameters():p.fill_(.125 if p.ndim==2 else .25)
        x=torch.ones((4,32),device='cuda:0')
        for _ in range(2):model(x)
        torch.cuda.synchronize()
        assert int(torch.cuda.cudart().cudaProfilerStart())==0
        lifecycle_start=time.perf_counter_ns();outputs=[];windows=[]
        for i in range(2):
            torch.cuda.nvtx.range_push(f'hyso.p14/{token}/forward/{i}')
            start=time.perf_counter_ns();out=model(x);torch.cuda.synchronize()
            end=time.perf_counter_ns();torch.cuda.nvtx.range_pop()
            outputs.append(out);windows.append((end-start)/1e6)
        lifecycle_end=time.perf_counter_ns()
        assert int(torch.cuda.cudart().cudaProfilerStop())==0
        valid=[bool(torch.equal(out.cpu(),torch.full((4,8),17.25))) for out in outputs]
        assert all(valid)
    receipt=dict(format='hyso.performance.p14.nsys_capture',version=1,trace_token=token,
        worker_pid=os.getpid(),status='completed',warmup_forwards=2,measured_forwards=2,
        output_equal=valid,terminal_completed=True,model='Linear(32,32)/ReLU/Linear(32,8)',
        input_shape=[4,32],output_shape=[4,8],expected_value=17.25,
        environment=dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0)),
        overhead=dict(instrumented_forward_wall_ms=windows,
            capture_scope_wall_ms=(lifecycle_end-lifecycle_start)/1e6,
            perturbation_ms=None,reason='no_uninstrumented_baseline; not_GPU_kernel_duration'))
    (args.output/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
