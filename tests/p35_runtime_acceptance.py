"""Explicit bounded P35 native captures. CPU and accelerated phases are separate."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hyso.performance import WorkloadRecipe,ProviderProtocol,profile_execution_providers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['cpu','accelerated'],required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True)
    plan=ProviderProtocol(warmup=0,runs=1,startup_timeout_seconds=60,
                          step_timeout_seconds=30,max_runtime_seconds=45)
    summaries=[]
    def capture(name,factory,providers,expected,optimization):
        gpu=args.gpu_uuid if 'CUDAExecutionProvider' in providers else None
        recipe=WorkloadRecipe(factory,search_path=str(ROOT/'tests'),gpu_uuid=gpu)
        result=profile_execution_providers(recipe,args.evidence/name,providers=providers,
            expected=expected,optimization=optimization,runtime_search_path=str(args.runtime.resolve()),protocol=plan,
            provider_options={'CUDAExecutionProvider':{'device_id':0,'gpu_mem_limit':64*1024**2}} if gpu else None)
        result.to_json(args.evidence/(name+'.json'));result.to_csv(args.evidence/(name+'.csv'))
        d=result.provider_diagnostics();raw=result.to_dict()
        (args.evidence/(name+'-diagnostics.json')).write_text(json.dumps(d,indent=2))
        row=dict(name=name,status=raw['analysis']['status'],runtime=d['runtime'],
            available=d['available_providers'],enabled=d['enabled_session_providers'],
            observed=d['observed_execution_providers'],events=d['events'],
            mapping=d['source_node_mapping'],expectation_mismatch=d['expectation_mismatch'],
            cleanup=raw['evidence']['controller']['cleanup'],worker_exit=raw['evidence']['controller']['exit_code'],
            worker_pid=raw['evidence']['controller']['launcher_pid'],completed_runs=d['completed_runs'],
            reason=d['reason'])
        summaries.append(row)
        (args.evidence/(args.phase+'-summary.json')).write_text(json.dumps(summaries,indent=2))
        print(json.dumps(row),flush=True)
        assert row['cleanup']['active_processes']==0 and row['worker_exit']==0
        assert d['completed_runs']==1 and raw['evidence']['report']['status']=='completed',row
        return result
    if args.phase=='cpu':
        result=capture('cpu-relu','deployment_support:onnx_model',['CPUExecutionProvider'],{'relu':'CPUExecutionProvider'},'disabled')
        d=result.provider_diagnostics()
        assert d['observed_execution_providers']==['CPUExecutionProvider']
        assert d['source_node_mapping']['status']=='observed'
        assert result.to_dict()['analysis']['fallback_detected'] is False
        fused=capture('cpu-fusion','provider_acceptance_factories:fusion_model',['CPUExecutionProvider'],{'*':'CPUExecutionProvider'},'all')
        import base64,onnx
        report=fused.to_dict()['evidence']['report']
        original=onnx.load_model_from_string(base64.b64decode(report['model_base64']))
        optimized=onnx.load_model_from_string(base64.b64decode(report['optimized_base64']))
        assert len(optimized.graph.node)<len(original.graph.node),'fusion did not occur; do not fabricate it'
        assert fused.provider_diagnostics()['source_node_mapping']['status']=='unknown'
        assert fused.to_dict()['analysis']['fallback_detected'] is None
    else:
        if not args.gpu_uuid:raise ValueError('explicit GPU UUID required')
        result=capture('cuda-relu','deployment_support:onnx_model',['CUDAExecutionProvider','CPUExecutionProvider'],{'relu':'CUDAExecutionProvider'},'disabled')
        assert 'CUDAExecutionProvider' in result.provider_diagnostics()['observed_execution_providers']
        partition=capture('cuda-cpu-partition','provider_acceptance_factories:partition_model',
            ['CUDAExecutionProvider','CPUExecutionProvider'],{'*':'CUDAExecutionProvider'},'disabled')
        d=partition.provider_diagnostics()
        assert set(d['observed_execution_providers'])=={'CUDAExecutionProvider','CPUExecutionProvider'}
        assert any(e['source_node']=='scaler_cpu' and e['provider']=='CPUExecutionProvider' for e in d['events'])
        assert any(e['source_node']=='relu_gpu' and e['provider']=='CUDAExecutionProvider' for e in d['events'])
        assert d['expectation_mismatch']['status']=='observed'
        # Placement fallback is not an execution-error/retry transition.
        assert d['execution_fallback']['detected'] is None
    print(json.dumps(dict(phase=args.phase,status='PASS',native_sessions=len(summaries))),flush=True)


if __name__=='__main__':main()
