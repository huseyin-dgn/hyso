"""Trusted P31 CPU-only trained checkpoint adapter, separate from old launch plans."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def make_model_b(plan_path,plan_sha256):
    # All native construction is invoked only by the admitted owned worker.
    import os
    import numpy as np
    import torch
    from torch import nn
    from hyso.models.modular import ModularCNN
    from hyso.training import Trainer
    from hyso.checkpoint import load_training_checkpoint
    from hyso._evaluation_evidence import model_hash
    from hyso.performance import InferenceWorkload
    if os.environ.get('HYSO_RUNTIME_ADMITTED_PID')!=str(os.getpid()):raise ValueError('owned worker admission required')
    if sha(plan_path)!=plan_sha256:raise ValueError('trained workload plan changed')
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['device']!='cpu' or plan['precision']!='fp32':raise ValueError('this reviewed adapter is CPU FP32 only')
    if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA context in CPU adapter')
    torch.set_num_threads(plan['cpu_threads']);torch.set_num_interop_threads(plan['interop_threads'])
    for key in ('checkpoint','dataset'):
        if sha(plan[key]['path'])!=plan[key]['sha256']:raise ValueError(key+' fingerprint changed')
    core=ModularCNN(**plan['model_config']);optimizer=torch.optim.AdamW(core.parameters(),lr=.001,weight_decay=.0001)
    trainer=Trainer(core,optimizer=optimizer,loss_fn=nn.CrossEntropyLoss(),device='cpu',amp=False,model_config=plan['model_config'])
    receipt=load_training_checkpoint(plan['checkpoint']['path'],core,optimizer=optimizer,scaler=trainer.scaler,
        map_location='cpu',restore_rng=False,capture_evidence=True)
    core.eval()
    if receipt['completed_epochs']!=6 or model_hash(core)!=plan['checkpoint']['model_state_sha256']:
        raise ValueError('trained state/epoch mismatch')
    with np.load(plan['dataset']['path'],allow_pickle=False) as data:
        inputs=torch.from_numpy(data['x_test'][:plan['batch_size']].copy()).unsqueeze(1).float().div_(255)
    if list(inputs.shape)!=plan['input_shape']:raise ValueError('input contract mismatch')
    input_sha=hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    base=InferenceWorkload(core,inputs,device='cpu')
    if model_hash(base.model)!=plan['checkpoint']['model_state_sha256']:raise ValueError('owned model copy changed trained state')
    # Exactly one explicit setup warmup, excluded from the observed-run counters.
    if base.step()!=plan['batch_size']:raise ValueError('warmup completion mismatch')
    identity=dict(base.identity,precision='fp32',input_dtype=str(inputs.dtype),input_sha256=input_sha,
        checkpoint=plan['checkpoint'],training_completed_epochs=6,training_status='trained_MNIST_checkpoint',
        preprocessing=plan['preprocessing'],cpu_threads=plan['cpu_threads'],interop_threads=plan['interop_threads'],
        setup_warmup_calls=1,measurement_scope='resident_fixed_first_256_official_MNIST_test_images_repeated_no_training')
    class Work:
        def __init__(self):self.identity=identity
        def step(self):return base.step()
        def close(self):
            base.close()
            if model_hash(base.model)!=plan['checkpoint']['model_state_sha256']:raise ValueError('trained model state changed during run')
            if sha(plan['checkpoint']['path'])!=plan['checkpoint']['sha256']:raise ValueError('checkpoint changed during run')
            if torch.cuda.is_initialized():raise RuntimeError('CPU run initialized CUDA')
    return Work()
