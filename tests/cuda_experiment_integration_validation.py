"""Focused real CUDA integration. No CUDA fallback or synthetic result fixtures."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from hyso.benchmark import BenchmarkRunner, compare_results
from hyso.experiments import ExperimentManager


def snapshot(model):
    return dict(state={n:v.clone() for n,v in model.state_dict().items()},
        refs=[id(p) for p in model.parameters()],
        grad_refs=[id(p.grad) for p in model.parameters()],
        grads=[p.grad.clone() for p in model.parameters()],
        modes=[m.training for m in model.modules()], python=random.getstate(),
        numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),gpu=torch.cuda.get_rng_state(0).clone())


def unchanged(model, before):
    assert [id(p) for p in model.parameters()] == before["refs"]
    assert [id(p.grad) for p in model.parameters()] == before["grad_refs"]
    assert [m.training for m in model.modules()] == before["modes"]
    assert all(torch.equal(v,before["state"][n]) for n,v in model.state_dict().items())
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),before["grads"]))
    assert random.getstate() == before["python"]
    current = np.random.get_state()
    assert current[0]==before["numpy"][0] and np.array_equal(current[1],before["numpy"][1]) and current[2:]==before["numpy"][2:]
    assert torch.equal(torch.get_rng_state(),before["cpu"])
    assert torch.equal(torch.cuda.get_rng_state(0),before["gpu"])


def forbidden(*args, **kwargs):
    raise AssertionError("tracking/export/comparison called model/CUDA runtime")


if not torch.cuda.is_available():
    raise RuntimeError("native CUDA is required; no CPU fallback")
torch.set_num_threads(1)
torch.cuda.set_device(0)
args.output.mkdir(parents=True,exist_ok=False)
root=args.output/"store"
m=ExperimentManager(root)
m.close()
ExperimentManager.migrate_store(root)
m=ExperimentManager(root)
e=m.create_experiment("native integration")
records=[]
results=[]
for precision in ("fp32","amp_fp16"):
    model=nn.Sequential(nn.Conv2d(1,2,3,padding=1),nn.BatchNorm2d(2),nn.AdaptiveAvgPool2d(1)).cuda()
    model.train()
    model[1].eval()
    for p in model.parameters(): p.grad=torch.ones_like(p)
    before=snapshot(model)
    result=BenchmarkRunner(model,device="cuda:0",input_shape=(2,1,5,5),precision=precision,
        warmup_iterations=1,measure_iterations=2,repeats=2,measure_cuda_memory=True).run()
    unchanged(model,before)
    data=result.to_dict()
    assert data["environment"]["device"]=="cuda:0" and data["configuration"]["precision"]==precision
    assert data["cuda_memory"]["status"]=="measured"
    assert len(data["timing"]["latency_samples_ms"])==2
    before=snapshot(model)
    with ExitStack() as stack:
        for obj,name in ((model,"forward"),(torch.cuda,"synchronize"),(torch.cuda,"Event"),
            (torch.cuda,"reset_peak_memory_stats"),(torch.cuda,"memory_allocated"),
            (torch.cuda,"memory_reserved"),(torch.cuda,"max_memory_allocated"),
            (torch.cuda,"max_memory_reserved"),(torch,"load"),(torch.optim.SGD,"step")):
            stack.enter_context(patch.object(obj,name,forbidden))
        run=m.create_run(e["id"],"benchmark",{"precision":precision})
        run.start()
        artifacts=[]
        for filename,export in (("result.json",result.to_json),("summary.csv",result.to_csv),("samples.csv",result.to_samples_csv)):
            path=args.output/(precision+"-"+filename)
            export(path)
            a=run.add_artifact(path,kind="benchmark_json" if filename.endswith("json") else "benchmark_csv",
                ownership="copy",metadata={"benchmark_run_id":data["identity"]["run_id"],"experiment_run_id":run.id})
            assert m.verify_artifact(a["id"])["status"]=="verified"
            artifacts.append(a)
        assert compare_results(result,result)["status"]=="compatible"
        memory=compare_results(result,result)["memory_comparability"]
        assert memory["status"]=="insufficient_metadata" and not memory["absolute_peak_model_comparable"]
        run.finish()
        m.close()
        m=ExperimentManager(root)
        assert m.get_run(run.id).read_metrics()==[]
        for a in artifacts:
            assert m.get_artifact(a["id"])["producer_run_id"]==run.id
            assert m.verify_artifact(a["id"])["status"]=="verified"
    unchanged(model,before)
    results.append(result)
    records.append(dict(precision=precision,run_id=run.id,benchmark_id=data["identity"]["run_id"],
        artifacts=artifacts,model_rng_preserved=True,no_tracking_runtime_calls=True,
        timing=data["timing"],cuda_memory=data["cuda_memory"]))
assert compare_results(*results)["status"]=="incompatible"
# A real disabled probe keeps timing eligibility but removes memory eligibility.
before=snapshot(model)
disabled=BenchmarkRunner(model,device="cuda:0",input_shape=(2,1,5,5),precision="amp_fp16",
    warmup_iterations=1,measure_iterations=2,repeats=2,measure_cuda_memory=False).run()
unchanged(model,before)
separate=compare_results(results[-1],disabled)
assert separate["status"]=="compatible" and separate["memory_comparability"]["status"]=="not_comparable"
m.close()
report=dict(status="PASS",native_scenarios=len(records),torch=str(torch.__version__),cuda_build=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),source=sys.modules[ExperimentManager.__module__].__file__,records=records)
(args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="records"}))
