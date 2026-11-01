"""Trusted native factories; called only by admitted acceptance workers."""
from pathlib import Path
import hashlib,os
from .core import read,sha,digest

def context(manifest_path):
    m=read(manifest_path)
    if os.environ.get('HYSO_FULL_ACCEPTANCE_ADMITTED')!=sha(manifest_path):
        raise PermissionError('acceptance-owned native admission required')
    return m

def configure(m):
    import torch
    p=m['protocols']['inference']
    torch.set_num_threads(p['cpu_threads'])
    if torch.get_num_interop_threads()!=p['interop_threads']:torch.set_num_interop_threads(p['interop_threads'])

def load_model(m,arm='B'):
    import torch
    from hyso.models.modular import ModularCNN
    from hyso.training import Trainer
    from hyso.checkpoint import load_training_checkpoint
    from hyso._evaluation_evidence import model_hash
    item=m['assets']['models'][arm]
    if sha(item['path'])!=item['sha256']:raise ValueError('checkpoint changed')
    model=ModularCNN(**item['config']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    trainer=Trainer(model,optimizer=optimizer,loss_fn=torch.nn.CrossEntropyLoss(),device='cpu',amp=False,model_config=item['config'])
    receipt=load_training_checkpoint(item['path'],model,optimizer=optimizer,scaler=trainer.scaler,
        map_location='cpu',restore_rng=False,capture_evidence=True)
    model.eval()
    if receipt['completed_epochs']!=item['completed_epochs'] or model_hash(model)!=item['recorded_state_sha256']:
        raise ValueError('trained state/epoch identity mismatch')
    if sum(p.numel() for p in model.parameters())!=item['parameters']:raise ValueError('architecture/parameter mismatch')
    return model

def data(m,*,split='test',count=None):
    import numpy as np,torch
    ds=m['assets']['dataset']
    if sha(ds['path'])!=ds['sha256']:raise ValueError('dataset changed')
    if split=='test':indices=slice(0,count)
    else:
        splitfile=m['assets']['split_indices']
        if sha(splitfile['path'])!=splitfile['sha256']:raise ValueError('split changed')
        with np.load(splitfile['path'],allow_pickle=False) as d:
            key='validation' if 'validation' in d.files and split=='val' else split
            indices=d[key][:count].copy()
    with np.load(ds['path'],allow_pickle=False) as d:
        suffix='test' if split=='test' else 'train'
        raw=d['x_'+suffix][indices].copy();labels=d['y_'+suffix][indices].astype(np.int64)
    return torch.from_numpy(raw).unsqueeze(1).float().div_(255),torch.from_numpy(labels)

def model_only(manifest_path,arm='B'):
    m=context(manifest_path);configure(m);return load_model(m,arm)

def model_pair(manifest_path,arm='B',batch=None):
    m=context(manifest_path);configure(m);model=load_model(m,arm)
    x,_=data(m,count=batch or m['protocols']['inference']['batch']);item=m['assets']['models'][arm]
    from hyso.performance._device_diagnostics import _state_hash
    return model,x,dict(kind='trained_checkpoint',model_id=item['model_id'],checkpoint_path=item['path'],
        checkpoint_sha256=item['sha256'],expected_loaded_state_sha256=_state_hash(model),
        preprocessing_id=m['assets']['dataset']['preprocessing'],
        preprocessing_sha256=digest(dict(algorithm='uint8_div255_FP32_NCHW',dataset_sha256=m['assets']['dataset']['sha256'],
            indices='official_test_first_'+str(len(x)))))

def training_recipe(*,device,manifest_path):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from hyso.training import Trainer
    m=context(manifest_path);configure(m);p=m['protocols']['training']
    model=load_model(m,'B');model.train()
    trainer=Trainer(model,optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001),
        loss_fn=torch.nn.CrossEntropyLoss(),device=device,amp=False,model_config=m['assets']['models']['B']['config'])
    x,y=data(m,split='train',count=p['train_samples']);vx,vy=data(m,split='val',count=p['validation_samples'])
    train=DataLoader(TensorDataset(x,y),batch_size=p['batch'],shuffle=False,num_workers=0)
    val=DataLoader(TensorDataset(vx,vy),batch_size=p['batch'],shuffle=False,num_workers=0)
    return dict(trainer=trainer,train_loader=train,val_loader=val)

def inference_work(manifest_path,device='cpu'):
    import torch
    from hyso.performance import InferenceWorkload
    from hyso._evaluation_evidence import model_hash
    m=context(manifest_path);model,x,origin=model_pair(manifest_path,'B',batch=256)
    base=InferenceWorkload(model,x,device='cpu' if device=='cpu' else 'cuda:0',gpu_uuid=None if device=='cpu' else device)
    if base.step()!=256:raise ValueError('warmup completion mismatch')
    identity=dict(base.identity,precision='fp32',checkpoint=m['assets']['models']['B'],input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        preprocessing=m['assets']['dataset']['preprocessing'],setup_warmup_calls=1,trained=True,origin=origin)
    class Work:
        def __init__(self):self.identity=identity
        def step(self):return base.step()
        def close(self):
            base.close()
            if model_hash(model)!=m['assets']['models']['B']['recorded_state_sha256']:raise ValueError('master state mutated')
            if sha(m['assets']['models']['B']['path'])!=m['assets']['models']['B']['sha256']:raise ValueError('checkpoint mutated')
    return Work()

def onnx_trained(manifest_path,arm='B',export_path=None):
    import numpy as np,torch
    m=context(manifest_path);model,x,origin=model_pair(manifest_path,arm,batch=1)
    if export_path is None:raise ValueError('explicit owned-stage export output required')
    path=Path(export_path)
    if path.exists():raise FileExistsError(path)
    torch.onnx.export(model,x,str(path),input_names=['input'],output_names=['logits'],opset_version=13,
        do_constant_folding=True,dynamic_axes=None)
    # ORT's existing adapter contract is exactly (ONNX bytes, NumPy feeds).
    # Export provenance remains in the outer stage's sealed checkpoint/config.
    return path.read_bytes(),{'input':x.numpy()}

def frame_processor(manifest_path):
    """Only a separately supplied, hashed trained camera adapter is eligible."""
    import importlib
    m=context(manifest_path);config=m['optional']['camera_adapter']['configuration']
    checkpoint=config.get('checkpoint')
    if not checkpoint or sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('camera trained checkpoint required')
    source=config['factory_source']
    if sha(source['path'])!=source['sha256']:raise ValueError('camera adapter source changed')
    import sys
    sys.path.insert(0,str(Path(source['path']).parent))
    module,name=config['factory'].split(':');factory=getattr(importlib.import_module(module),name)
    if Path(importlib.import_module(module).__file__).resolve()!=Path(source['path']).resolve():raise ValueError('camera factory import path mismatch')
    processor=factory(**config.get('kwargs',{}))
    if not hasattr(processor,'identity') or processor.identity.get('trained') is not True:
        raise ValueError('camera adapter must expose trained=True identity and frame input/preprocessing/output contract')
    return processor
