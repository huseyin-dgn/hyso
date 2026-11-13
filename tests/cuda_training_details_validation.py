"""P20/P22 native cases reuse the established deterministic P19 replay fixtures."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, os.environ.get('HYSO_TEST_PACKAGE_ROOT', str(Path(__file__).resolve().parents[1])))
import torch
from hyso.performance import measure_training_detailed, TrainingTimingProtocol
import hyso.performance.training_details as details
import cuda_training_timing_validation as baseline


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.15)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    captures=[]; rows=[]
    def measure(*args,**kwargs):
        kwargs['protocol']=TrainingTimingProtocol(event_buffer=2)
        result=measure_training_detailed(*args,**kwargs)
        stages=result.stages.to_dict(); rates=result.throughput.to_dict()
        parent=result.timing.to_dict()
        attempts=[r for r in parent['observations'] if r['kind']=='attempt']
        for stage, count in [('optimizer_execution',sum(r['applied'] for r in attempts)),
                             ('scaler_step_attempt',len(attempts)),('scaler_update',len(attempts)),
                             ('ema_update',sum(r['applied'] for r in attempts)),
                             ('onecycle_scheduler',sum(r['applied'] for r in attempts))]:
            assert sum(r['stage']==stage and r['status']=='measured' for r in stages['observations'])==count
        skipped=[r for r in stages['observations'] if r['stage']=='optimizer_execution' and r['status']=='not_executed']
        assert len(skipped)==sum(r['skipped'] for r in attempts)
        assert all(r['host_ms'] is None and r['cuda_event_ms'] is None for r in skipped)
        assert all(r['cuda_event_ms'] is not None and r['cuda_event_ms']>=0 for r in stages['observations'] if r['status']=='measured')
        assert rates['segment']['completed']['applied']==sum(r['applied'] for r in attempts)
        captures.append(dict(timing=parent,stages=stages,throughput=rates))
        return result.timing
    with patch.object(baseline,'measure_training',measure):
        for opt in (torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW):
            for amp in (False,True):
                for stream in (False,True):
                    for overflow in ((False,True) if amp else (False,)):
                        rows.append(baseline.run(opt,amp,stream,overflow)['id'])
        rows.extend(baseline.checkpoint_case(amp)['id'] for amp in (False,True))
    # Inject a Python-side CUDA event recording failure inside optimizer hooks.
    m=torch.nn.Linear(3,2).cuda(); t=baseline.Trainer(m,optimizer=torch.optim.SGD(m.parameters(),lr=.01),
        loss_fn=torch.nn.CrossEntropyLoss(),device='cuda')
    original=details._StageRecorder._event
    def fail_in_step(recorder):
        if 'stage:scaler_step_attempt' in recorder.active: raise RuntimeError('injected CUDA event failure')
        return original(recorder)
    try:
        with patch.object(details._StageRecorder,'_event',fail_in_step):
            measure_training_detailed(t,[(torch.ones(2,3),torch.tensor([0,1]))],epochs=1)
    except RuntimeError as exc: assert 'injected CUDA event failure' in str(exc)
    else: raise AssertionError('event failure was not raised')
    assert not t.optimizer._optimizer_step_pre_hooks and not t.optimizer._optimizer_step_post_hooks
    assert not hasattr(t,'_stage_recorder') and not hasattr(t,'_timing_recorder')
    assert all(p.grad is None for p in t.model.parameters())
    rows.append('injected_cuda_event_failure_cleanup')
    if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):
        assert Path(details.__file__).resolve().is_relative_to(Path(os.environ['HYSO_TEST_PACKAGE_ROOT']).resolve())
    Path(args.output).write_text(json.dumps(dict(status='passed',scenarios=rows,results=captures,
        package_source=details.__file__),indent=2),encoding='utf-8')
    print(f'{len(rows)} P20/P22 CUDA scenarios passed')


if __name__=='__main__': main()
