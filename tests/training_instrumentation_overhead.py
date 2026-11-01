"""Observational full-fit wall distributions, never a fixed overhead guarantee."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,os.environ.get('HYSO_TEST_PACKAGE_ROOT',str(Path(__file__).resolve().parents[1])))
import torch
from hyso.training import Trainer
from hyso.performance import measure_training,measure_training_detailed


def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if args.device=='cuda':
        assert torch.cuda.is_available(); torch.cuda.set_per_process_memory_fraction(.15)
    values={m:[] for m in ('normal','P19','detailed')}
    for repetition in range(3):
        # Rotate order to expose, rather than hide, first-use/order sensitivity.
        modes=list(values); modes=modes[repetition:]+modes[:repetition]
        for mode in modes:
            torch.manual_seed(832)
            model=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,2)).to(args.device)
            t=Trainer(model,optimizer=torch.optim.Adam(model.parameters(),lr=.01),loss_fn=torch.nn.CrossEntropyLoss(),
                      device=args.device,accum_steps=2)
            data=list(zip(torch.ones(17,16).split(4),torch.arange(17).remainder(2).split(4)))
            if args.device=='cuda': torch.cuda.current_stream().synchronize()
            start=time.perf_counter_ns()
            if mode=='normal': t.fit(data,epochs=3)
            elif mode=='P19': measure_training(t,data,epochs=3)
            else: measure_training_detailed(t,data,epochs=3)
            if args.device=='cuda': torch.cuda.current_stream().synchronize()
            values[mode].append((time.perf_counter_ns()-start)/1e6)
    summary={k:dict(samples_ms=v,median_ms=statistics.median(v),min_ms=min(v),max_ms=max(v)) for k,v in values.items()}
    Path(args.output).write_text(json.dumps(dict(device=args.device,scope='full fit host wall including instrumentation/result construction; selected-stream completion',
        repeats=3,observational=True,unmeasured_instrumentation_overhead=None,distribution=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__': main()
