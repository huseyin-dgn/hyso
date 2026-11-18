"""One bounded public-API CUDA capture, independent of optional tracing tools.

Run directly (pytest's conftest deliberately hides CUDA):
    python tests/cuda_operator_acceptance.py --output <new directory>

Missing kernel tracing passes core acceptance, never advanced acceptance.
No occupancy/PID admission gate or RAM/VRAM quota is applied.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    import torch
    from hyso.performance import (ColdStartRecipe,ProfilingProtocol,OperatorProfileResult,
        profile_operators,profile_layers,analyze_flops)
    # Discovery does not initialize the parent's CUDA context or allocator.
    before=torch.cuda.is_initialized();rng=torch.get_rng_state().clone()
    assert torch.cuda.is_available(),'CUDA unavailable: this native acceptance case was not run'
    protocol=ProfilingProtocol(warmup_iterations=1,measure_iterations=1,timeout_seconds=45)
    source=profile_operators(ColdStartRecipe('profiling_factories:tiny_cuda',
        factory_search_path=root/'tests',input_shape=(1,2,3,3),device='cuda:0'),
        protocol=protocol,trace_path=args.output/'trace.json')
    d=source.to_dict();raw=d['raw_evidence'];m=d['measurements']
    assert torch.cuda.is_initialized()==before and torch.equal(torch.get_rng_state(),rng)
    assert d['actual']==dict(warmup_forwards=1,measured_forwards=1)
    assert all(d['protocol']['configuration'][key] is None for key in
               ('cpu_memory_limit_bytes','input_limit_bytes','cuda_memory_fraction'))
    assert raw['worker']['status']=='completed' and raw['worker']['process_terminated']
    assert raw['worker']['exit_code']==0 and all(raw['safety'].values())
    assert m['operators'] and any(r['self_cpu_us']>0 for r in m['operators'])
    assert any(r['name']=='aten::linear' for r in m['operators'])
    for row in m['operators']:
        events=[e for e in raw['events'] if e['device_type']=='CPU' and e['scope']==0 and e['name']==row['name']]
        assert row['status']=='measured' and row['count']==len(events)
        assert row['cpu_total_us']==math.fsum(e['cpu_total_us'] for e in events)
        assert row['self_cpu_us']==math.fsum(e['self_cpu_us'] for e in events)
    trace=(args.output/'trace.json').read_bytes()
    assert hashlib.sha256(trace).hexdigest()==raw['trace']['sha256']
    assert any(e.get('cat')=='cpu_op' for e in json.loads(trace)['traceEvents'])
    source.to_json(args.output/'operators.json');source.to_csv(args.output/'operators.csv')
    assert OperatorProfileResult(json.loads((args.output/'operators.json').read_text())).to_dict()==d
    with (args.output/'operators.csv').open(newline='',encoding='utf-8') as stream:
        assert [json.loads(r['payload']) for r in csv.DictReader(stream)]==m['operators']
    layer=profile_layers(source).to_dict()
    counts=analyze_flops(source).to_dict()
    assert counts['measurements']['status']=='complete'
    assert all(r['cuda_activity_us'] is None for r in raw['invocations'])
    assert layer['measurements']['cuda_event_timing']['aggregation'].endswith('not_kernel_time_or_inference_latency')
    activity=m['cuda_activity'];attribution=m['cuda_attribution']
    if not m['cuda_kernels']:
        assert activity['status']=='unsupported' and activity['total_kernel_us'] is None
        assert activity['kernel_count'] is None and attribution['status']=='unsupported'
        assert attribution['mapped_kernel_count']==0 and attribution['operators']==[]
        assert m['status']=='partial' and m['reason']=='cuda_kernel_activity_unavailable_or_invalid'
    else:
        assert m['cuda_kernels']==raw['kernel_events']
        for k in m['cuda_kernels']:
            e=json.loads(trace)['traceEvents'][k['trace_index']]
            assert e['cat']=='kernel' and e['name']==k['name'] and e.get('dur')==k['duration_us']
    advanced=('PASS' if activity['status']=='measured' and attribution['status']=='complete'
              else 'UNSUPPORTED' if activity['status']=='unsupported' else 'PARTIAL')
    report=dict(core_operator_profiling='PASS',advanced_kernel_profiling=advanced,
        cuda_activity=activity,cuda_attribution_status=attribution['status'],
        cuda_attribution_reason=attribution['reason'],
        mapped_kernel_count=attribution['mapped_kernel_count'],
        cuda_event_layer_status=layer['measurements']['cuda_event_timing']['status'],
        worker=raw['worker'],supported_activities=raw['supported_activities'],
        torch=str(torch.__version__),cuda_build=torch.version.cuda,
        package_source=d['provenance']['environment']['package_source'])
    (args.output/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,allow_nan=False))


if __name__=='__main__':main()
