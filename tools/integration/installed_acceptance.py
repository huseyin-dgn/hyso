"""CPU execution of integrated APIs from an installed artifact; no GPU work."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--ort',type=Path,required=True)
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.target.resolve()))
    sys.path.append('C:/Users/hdgn5/.conda/envs/torch_gpu/Lib/site-packages')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',MPLBACKEND='Agg')
    import torch,hyso
    from hyso import EvaluationCohort,QualityExperimentResult,compare_performance_quality
    from hyso.performance import (ColdStartRecipe,ProfilingProtocol,analyze_flops,
        WorkloadRecipe,DeviceProtocol,compare_devices,ProviderProtocol,profile_execution_providers,RuntimeResult)
    assert Path(hyso.__file__).resolve().is_relative_to(args.target.resolve())
    torch.set_num_threads(1);before=torch.get_rng_state().clone()
    p12=analyze_flops(ColdStartRecipe('torch.nn:Conv2d',model_kwargs=dict(in_channels=2,out_channels=4,
        kernel_size=3,padding=1,groups=2,bias=True),input_shape=(2,2,5,7),precision='fp64'),
        protocol=ProfilingProtocol(warmup_iterations=0,measure_iterations=1,timeout_seconds=45))
    p12.to_json(args.evidence/'p12.json');p12.to_csv(args.evidence/'p12.csv')
    assert p12.to_dict()['protocol']['operation_model_version']==3
    assert p12.to_dict()['measurements']['total_flops_per_batch']==5320
    assert torch.equal(before,torch.get_rng_state())
    def model(correct,extra):
        value=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(2,2))
        with torch.no_grad():
            value[1].weight.copy_(torch.tensor([[-1.,0.],[1.,0.]])*(1 if correct else -1));value[1].bias.zero_()
        if extra:value.register_buffer('extra',torch.zeros(extra))
        return value
    baseline=model(False,8);candidate=model(True,0)
    cohort=EvaluationCohort(torch.tensor([1.,-1.]).reshape(1,1,1,2).repeat(12,1,1,1),[1]*12,
        sample_ids=[str(i) for i in range(12)],class_names=['no','yes'],split='test',preprocessing_id='installed-fixed')
    before=torch.get_rng_state().clone()
    p29=compare_performance_quality(baseline,candidate,cohort,iterations=3)
    p29.to_json(args.evidence/'p29.json');p29.to_csv(args.evidence/'p29.csv')
    d=p29.to_dict();assert d['sources']['comparison_version']==2
    assert d['analysis']['memory']['delta_bytes']==-32 and d['analysis']['winner'] is None
    assert QualityExperimentResult(json.loads((args.evidence/'p29.json').read_text())).to_dict()==d
    tests=str(args.repository.resolve()/'tests')
    p34=compare_devices(WorkloadRecipe('deployment_support:torch_model',search_path=tests),['cpu'],
        args.evidence/'p34-worker',protocol=DeviceProtocol(warmup=0,samples=3))
    p34.to_json(args.evidence/'p34.json');p34.to_csv(args.evidence/'p34.csv')
    assert p34.to_dict()['evidence']['comparison_version']==2
    assert p34.to_dict()['analysis']['devices'][0]['status']=='measured'
    p35=profile_execution_providers(WorkloadRecipe('deployment_support:onnx_model',search_path=tests),
        args.evidence/'p35-worker',providers=['CPUExecutionProvider'],expected={'relu':'CPUExecutionProvider'},
        optimization='disabled',runtime_search_path=str(args.ort.resolve()),protocol=ProviderProtocol(warmup=0,runs=1))
    p35.to_json(args.evidence/'p35.json');p35.to_csv(args.evidence/'p35.csv')
    assert p35.to_dict()['analysis']['status']=='expected_execution_observed'
    for stem in ('p34','p35'):
        assert RuntimeResult.from_json(args.evidence/(stem+'.json')).to_dict()==RuntimeResult.from_csv(args.evidence/(stem+'.csv')).to_dict()
        request=json.loads((args.evidence/(stem+'-worker')/('0/request.json' if stem=='p34' else 'request.json')).read_text())
        assert Path(request['package_root']).resolve()==args.target.resolve()
    for relative in ('native-p34/comparison.json','native-p35/cuda-cpu-partition.json','native-p31-short/result.json'):
        RuntimeResult.from_json(args.native/relative)
    RuntimeResult.from_json(args.repository/'tests/fixtures/device-comparison-v1.json')
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
    summary=dict(status='PASS',source=hyso.__file__,torch=str(torch.__version__),python=sys.version,
        P12_operation_model_version=3,P29_comparison_version=2,P34_comparison_version=2,
        P35_graph_analysis_version=2,installed_workers=True,parent_rng_preserved=True,cuda_initialized=False)
    (args.evidence/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
