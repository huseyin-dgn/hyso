"""Tiny real engineering scenarios; native calls execute only in user admission."""
from pathlib import Path
import json
import os
import shutil
import subprocess

def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def model():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,2)).eval()

def execute(name):
    import pytest
    import torch
    from hyso import export as ex
    attempt=Path(os.environ['HYSO_EXPORT_ATTEMPT']);folder=attempt/name;folder.mkdir(exist_ok=False)
    status='FAIL';raw=None
    try:
        x=torch.ones(2,3);m=model()
        if name=='E01':
            raw=ex.availability();assert all(r['runtime_verified'] is False for r in raw['dependencies'].values())
            p=folder/'model.pt';r=ex.libtorch(m,x,p);r.verify()
            before=p.read_bytes()
            try:ex.libtorch(m,x,p)
            except ex.ExportError as e:assert e.code=='EXISTS'
            else:raise AssertionError('default overwrite permitted')
            assert p.read_bytes()==before
            raw['conversion']=r.to_dict()
        elif name=='E02_CPU':
            r=ex.onnx(m,x,folder/'model.onnx',input_names=['image'],output_names=['logits'],dynamic_shapes={'image':{0:dict(name='batch',min=1,max=8)}})
            raw=r.to_dict();assert raw['deployment_verified'] and raw['runtime']['analysis']['status']=='expected_execution_observed'
        elif name=='E05_CPU':
            raw={}
            for method in ('trace','script'):
                # Sequential uses typed native modules and source-readable script.
                r=ex.libtorch(m,x,folder/(method+'.pt'),method=method);raw[method]=r.to_dict()
                assert raw[method]['deployment_verified'] and raw[method]['cpp_runtime']['status']=='NOT_RUN'
        elif name=='E06_EP':
            r=ex.torch_export(m,x,folder/'model.pt2',input_names=['image'],dynamic_shapes={'image':{0:dict(name='batch',min=1,max=8)}})
            raw=r.to_dict();assert raw['deployment_verified'] and raw['export_constraints']
        elif name=='E06_AOT':
            r=ex.torch_export(m,x,folder/'model.pt2')
            ex.aotinductor(r,folder/'compiled')
            raise AssertionError('Windows AOT compilation was incorrectly admitted')
        elif name=='E05_CPP':
            capability=ex.availability();raw=capability
            if not capability['compilers']['cmake'] or not any(capability['compilers'][n] for n in ('cl','g++')):
                status='BLOCKED';pytest.skip('CMake/compiler not configured; Python TorchScript is not C++ runtime acceptance')
            sdk=Path(torch.__file__).parent
            if not (sdk/'share/cmake/Torch/TorchConfig.cmake').is_file() or not (sdk/'include/torch/script.h').is_file():
                status='BLOCKED';pytest.skip('Compatible LibTorch headers/CMake SDK not configured')
            r=ex.libtorch(m,x,folder/'cpp.pt')
            with torch.no_grad():expected=m(x)
            x.numpy().tofile(folder/'input.f32');expected.numpy().tofile(folder/'expected.f32')
            repo=Path(__file__).resolve().parents[2]
            from torch.utils import cmake_prefix_path
            commands=[['cmake','-S',str(repo/'examples/export/libtorch'),'-B',str(folder/'build'),'-DCMAKE_PREFIX_PATH='+cmake_prefix_path],
                ['cmake','--build',str(folder/'build'),'--config','Release','--parallel','1']]
            raw=dict(export=r.to_dict(),commands=[])
            for command in commands:
                p=subprocess.run(command,capture_output=True,text=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
                raw['commands'].append(dict(argv=command,exit_code=p.returncode,stdout=p.stdout[-16384:],stderr=p.stderr[-16384:]))
                dump(folder/'raw.json',raw)
                if p.returncode!=0:
                    dump(attempt/'user-safety-stop.json',dict(scenario=name,code='CPP_NATIVE_BUILD_FAILED',raw=raw))
                    status='BLOCKED';pytest.exit('C++ build failed under owned Job; no retry',returncode=3)
            executable=next((folder/'build').rglob('hyso_libtorch_check.exe'))
            command=[str(executable),str(folder/'cpp.pt'),str(folder/'input.f32'),str(folder/'expected.f32'),'2,3','2,2']
            p=subprocess.run(command,capture_output=True,text=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
            raw['runtime']=dict(argv=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
            dump(folder/'raw.json',raw)
            if p.returncode not in (0,2,3):
                dump(attempt/'user-safety-stop.json',dict(scenario=name,code='CPP_NATIVE_CRASH',raw=raw))
                status='BLOCKED';pytest.exit('C++ native crash; explicit revalidation required',returncode=3)
            observed=json.loads(p.stdout);assert p.returncode==0 and observed['matched'] and observed['finite'] and observed['device']=='cpu' and observed['elements']==4
        else:raise ValueError('unknown export scenario')
        status='PASS'
    except ex.ExportError as exc:
        raw=dict(scenario=name,code=exc.code,reason=str(exc),result=exc.result)
        if exc.code in ('OWNERSHIP','DLL_INTEGRITY','RUNTIME_IDENTITY','ARTIFACT_INTEGRITY','RESULT_INTEGRITY','NATIVE_SAFETY_STOP','MODEL_PRESERVATION'):
            dump(attempt/'user-safety-stop.json',raw);status='BLOCKED';pytest.exit('Export ownership/native safety stop: '+exc.code,returncode=3)
        if exc.result and any(r.get('status')=='BLOCKED' for r in exc.result.get('stages',{}).values()):
            status='BLOCKED';pytest.skip(str(exc))
        raise
    except subprocess.TimeoutExpired as exc:
        raw=dict(scenario=name,code='CPP_NATIVE_TIMEOUT',reason=str(exc))
        dump(attempt/'user-safety-stop.json',raw);status='BLOCKED';pytest.exit('Native timeout; no automatic retry',returncode=3)
    finally:
        if raw is not None:dump(folder/'raw.json',raw)
        dump(folder/'result.json',dict(scenario=name,status=status,public_api='hyso.export',device='cpu',actual=raw))

def gpu_group(ctx):
    from hyso import export as ex
    import torch
    from tools.cuda_acceptance.cases import MissingEvidence
    from tools.cuda_acceptance.feature_evidence import SafetyViolation
    manifest=ctx.plan['feature_tools']['ort'];allowed=dict(manifest['allowed_binaries'])
    allowed.update({r['path']:r['sha256'] for r in ctx.plan['runtime']['binary_expectations'].values()})
    allowed.update(ctx.plan['feature_tools']['tensorrt']['allowed_binaries'])
    dependencies=ctx.plan['feature_tools']['export_cuda']
    allowed.update(dependencies['allowed_binaries'])
    native=ex.NativeConfig(ctx.uuid,allowed,runtime_search_path=manifest['site'],dependency_identities=dependencies['binary_identities'])
    def run(name):
        folder=ctx.folder/name;folder.mkdir(exist_ok=False)
        try:
            x=torch.ones(2,3);m=model()
            if name=='E02_CUDA':r=ex.onnx(m,x,folder/'model.onnx',runtime_device='cuda',native=native)
            elif name=='E05_CUDA':r=ex.libtorch(m,x,folder/'model.pt',runtime_device='cuda',native=native)
            elif name=='E03_TRT':
                if not ex.availability()['dependencies']['tensorrt']['discoverable']:raise MissingEvidence('TensorRT dependency unavailable; no build attempted')
                a=ex.onnx(m,x,folder/'source.onnx',config=ex.ExportConfig(validate_runtime=False))
                r=ex.tensorrt(a,folder/'model.engine',example_input={'input0':x},native=native)
            elif name=='E04_TRT':r=ex.pytorch_tensorrt(m,x,folder/'model.engine',native=native)
            elif name in ('E07_TRT_FP16','E08_TRT_QDQ','E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON'):
                a=ex.onnx(m,x,folder/'source.onnx',config=ex.ExportConfig(validate_runtime=False))
                precision='fp32';config=ex.ExportConfig()
                if name=='E07_TRT_FP16':precision='fp16';config=ex.ExportConfig(atol=1e-3,rtol=1e-2)
                if name=='E08_TRT_QDQ':
                    a=ex.quantize(a,folder/'qdq.onnx',mode='static',calibration=[{'input0':x},{'input0':x*.5}],
                        dataset_id='E13_synthetic_engineering_fixture_not_representative',representative=False,
                        config=ex.ExportConfig(validate_runtime=False));precision='int8';config=ex.ExportConfig(atol=.1,rtol=.1)
                r=ex.tensorrt(a,folder/'model.engine',example_input={'input0':x},native=native,precision=precision,config=config)
                extension={}
                if name in ('E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON'):
                    runtime=ex.Runtime(r,device='cuda',native=native,evidence_directory=folder)
                    if name=='E09_TRT_COHORT':
                        extension['validation']=ex.validate(ex.Runtime(m),runtime,[x,x*.5])
                        assert extension['validation']['numerical']['status']=='PASS'
                    else:
                        measured=ex.benchmark(runtime,x,warmup=1,iterations=3);extension['benchmark']=measured
                        if name=='E11_TRT_COMPARISON':
                            baseline=ex.benchmark(ex.Runtime(m),x,warmup=1,iterations=3)
                            compared=ex.compare(baseline,measured);extension['comparison']=compared
                            assert not compared['comparable'] and compared['speedup'] is None
                data=r.to_dict();data['development_extension']=extension
                from hyso.export.contracts import seal
                r=seal(data)
            else:raise ValueError(name)
            data=r.to_dict();dump(folder/'raw.json',data)
            return data
        except ex.ExportError as exc:
            dump(folder/'raw.json',dict(code=exc.code,reason=str(exc),result=exc.result))
            if exc.code in ('NATIVE_SAFETY_STOP','DLL_INTEGRITY','RUNTIME_IDENTITY','ARTIFACT_INTEGRITY','RESULT_INTEGRITY','GPU_IDENTITY','OWNERSHIP','MODEL_PRESERVATION'):raise SafetyViolation(str(exc)) from exc
            if exc.result and any(v.get('status')=='BLOCKED' for v in exc.result.get('stages',{}).values()):raise MissingEvidence(str(exc)) from exc
            raise
    for name in ('E02_CUDA','E05_CUDA','E03_TRT','E04_TRT','E07_TRT_FP16','E08_TRT_QDQ','E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON'):
        ctx.run(name,lambda name=name:run(name),scope='E01_E06_bounded_actual_export_runtime_not_general_hardware_acceptance')
