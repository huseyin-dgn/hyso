"""E13 preparation only: separate evidence, no import/build/inference/install."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def prepare(python,output):
    spec=importlib.util.spec_from_file_location('export_inventory',ROOT/'hyso/export/dependencies.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:inventory=module.runtime_inventory(python)
    except module.ExportError as exc:
        from importlib import metadata
        python=Path(python).resolve(strict=True);site=python.parent/'Lib/site-packages'
        versions={d.metadata.get('Name','').lower().replace('_','-'):d.version for d in metadata.distributions(path=[str(site)])}
        inventory=dict(python=str(python),python_sha256=sha(python),site=str(site),versions=versions,
            issues=[str(exc)],tensorrt=dict(status='BLOCKED',distributions={n:v for n,v in versions.items() if n.startswith('tensorrt')},
            allowed_binaries={},binary_identities={},provenance={},runtime_verified=False))
        for name in ('tensorrt-cu12','onnx','torch'):
            if name not in versions:inventory['issues'].append('missing distribution: '+name)
        inventory['issues']+=['invalid NumPy remnant: '+p.name for p in site.iterdir() if p.name.startswith(('~umpy','-umpy','~numpy'))]
    old_root=ROOT/'outputs/export-acceptance-20261127'
    preserved={str(p):sha(p) for p in old_root.iterdir() if p.is_file() and p.suffix=='.json'}
    checkpoint=json.loads((old_root/'checkpoint.json').read_text(encoding='utf-8'))
    previous=json.loads((old_root/'plan.json').read_text(encoding='utf-8'))
    paths=list((ROOT/'hyso/export').glob('*.py'))+list((ROOT/'tools/export_acceptance').glob('*.py'))
    paths+=list((ROOT/'tests').glob('test_export_*.py'))
    paths += [ROOT/'setup.cfg',ROOT/'commercial-profile.json',ROOT/'hyso/cli/export.py',ROOT/'hyso/cli/main.py']
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    changed=[p for p,s in previous['source_sha256'].items() if (ROOT/p).is_file() and sha(ROOT/p)!=s]
    nodes=[]
    for p in sorted((ROOT/'tests').glob('test_export_development*.py')):
        for item in ast.parse(p.read_text(encoding='utf-8')).body:
            if isinstance(item,ast.FunctionDef) and item.name.startswith('test_'):
                nodes.append(p.relative_to(ROOT).as_posix()+'::'+item.name)
    gpu=[dict(id=name,status='BLOCKED' if inventory['tensorrt']['status']=='BLOCKED' else 'NOT_TESTED',
        dependency='tensorrt-cu12 10.x + selected CUDA Torch + checked ONNX/CPU reference',
        admission='owned Windows Job; physical UUID; RECORD DLL hashes/file identities; existing budgets; zero retry')
        for name in ('E03_TRT','E04_TRT','E07_TRT_FP16','E08_TRT_QDQ','E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON')]
    plan=dict(format='hyso.export.e13.prepare',schema_version=1,prepare_only=True,gpu_execution_started=False,
        package_changes=False,runtime=inventory,cpu_scenarios=nodes,gpu_scenarios=gpu,
        precision_scope=dict(fp32='CPU supported; native requires admission',fp16='mixed ONNX; TensorRT flag; CPU cast coverage can remain BLOCKED',
            int8='dynamic MatMul/Gemm or static symmetric QDQ Conv/MatMul/Gemm; native NOT_TESTED'),
        preserved_evidence=preserved,previous_counts=dict(cpu=4,gpu=3),
        previous_checkpoint_summary=dict(results={k:v.get('status') for k,v in checkpoint.get('results',{}).items()},safety_hold=checkpoint.get('safety_hold')),
        source_sha256=sources,changed_prior_sources=changed,
        evidence_impact=dict(historical_records_modified=False,current_source_revalidation_required=bool(changed),
            reason='Export/API/native allocator sources changed; historical PASS remains historical and is not promoted to current-source acceptance',
            affected=['E01-E06 export proofs','E03/E04 native admission and execution'],
            unrelated_campaign_revalidation_required=False),
        limits=dict(workspace_bytes=64*1024**2,max_gpu_bytes=384*1024**2,allocator_fraction=.15,
            process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4,timeout_seconds=90),
        native_execution_policy='Preparation does not authorize execution. Use the owned campaign with a reviewed current-source/runtime transition; never run pytest GPU cases directly.')
    output=Path(output).resolve()
    if output==old_root.resolve() or output.is_relative_to(old_root.resolve()):raise ValueError('historical evidence output is forbidden')
    output.mkdir(parents=True,exist_ok=True);target=output/'prepare.json'
    if target.exists():raise FileExistsError('choose a new E13 preparation directory')
    temporary=output/'prepare.json.tmp'
    with temporary.open('x',encoding='utf-8') as stream:json.dump(plan,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    temporary.replace(target)
    if any(sha(p)!=s for p,s in preserved.items()):raise RuntimeError('historical evidence changed during preparation')
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',required=True,action='store_true')
    select=parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--runtime-python',type=Path);select.add_argument('--conda-env')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();python=args.runtime_python
    if args.conda_env:
        result=subprocess.run(['conda','env','list','--json'],capture_output=True,text=True,check=True,timeout=30,shell=os.name=='nt')
        candidates=[Path(p)/'python.exe' for p in json.loads(result.stdout)['envs'] if Path(p).name==args.conda_env]
        if len(candidates)!=1:raise ValueError('Conda environment must resolve uniquely')
        python=candidates[0]
    print(prepare(python,args.output))

if __name__=='__main__':main()
