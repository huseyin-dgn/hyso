"""Small real-file CPU/spawn acceptance for P21; Job supervision is test-only."""
import argparse
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from hyso.training import Trainer
from hyso.performance import (WorkerTaskDataLoader, measure_training_with_loader,
    analyze_worker_tasks, trace_data_stage)
from hyso.performance._loader_worker_lifecycle import shutdown_traced_workers
from trainer_replay_support import seed, snapshot, equal


def decode(value):
    with Image.open(io.BytesIO(value)) as image:
        image.load()
        return np.array(image.convert('RGB'), copy=True)


def transform(value):
    # A declared functional-test delay, not a claim about real-world disk speed.
    time.sleep(.01)
    return torch.from_numpy(value).float().flatten()/255 + torch.rand(())*.01 + random.random()*.01 + np.random.random()*.01


class DiskSamples(Dataset):
    def __init__(self, root): self.root = Path(root)
    def __len__(self): return 8
    def __getitem__(self, index):
        raw = trace_data_stage('load', Path.read_bytes, self.root/f'{index}.png', sample_id=str(index))
        image = trace_data_stage('decode', decode, raw, sample_id=str(index))
        return trace_data_stage('transform', transform, image, sample_id=str(index)), index % 2


def initialize_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    random.seed(info.seed + worker_id)
    np.random.seed((info.seed + worker_id) % 2**32)
    torch.manual_seed(info.seed + worker_id)
    torch.set_num_threads(1)


def cleanup(loader, traced):
    if traced:
        return shutdown_traced_workers(loader.trace_iterator, timeout=8)
    return None


def run(case):
    torch.set_num_threads(1)
    records, pids = [], []
    with tempfile.TemporaryDirectory(prefix='p21-files-') as directory:
        root = Path(directory)
        for i in range(8):
            Image.fromarray((np.arange(48).reshape(4,4,3)+i).astype(np.uint8)).save(root/f'{i}.png')
        for observed in (False, True):
            seed(137)
            batches = [[3,0], [7,1,5], [2,6,4]]
            options = dict(batch_sampler=batches, num_workers=2, multiprocessing_context='spawn',
                persistent_workers=True, prefetch_factor=2, timeout=45,
                worker_init_fn=initialize_worker, generator=torch.Generator().manual_seed(721))
            traced = observed and case == 'task_images'
            loader = (WorkerTaskDataLoader(DiskSamples(root), **options) if traced else DataLoader(DiskSamples(root), **options))
            component_ids = {k:id(getattr(loader,k)) for k in ['dataset','sampler','batch_sampler','generator','worker_init_fn','collate_fn']}
            assert loader.batch_sampler is batches
            epochs = []
            def callback(event, logs):
                if event == 'epoch_end': epochs.append([w.pid for w in loader._iterator._workers])
            model = torch.nn.Sequential(torch.nn.Linear(48,4), torch.nn.Dropout(.2), torch.nn.Linear(4,2))
            trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(),lr=.01),
                loss_fn=torch.nn.CrossEntropyLoss(), device='cpu', callbacks=(callback,))
            seen = []
            hook = model.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
            try:
                if observed: measured = measure_training_with_loader(trainer,loader,epochs=2)
                else: trainer.fit(loader,epochs=2)
            finally: hook.remove()
            records.append((snapshot(trainer),loader.generator.get_state().clone(),seen))
            assert {k:id(getattr(loader,k)) for k in component_ids} == component_ids
            assert len(epochs) == 2 and epochs[0] == epochs[1] and len(set(epochs[0])) == 2
            assert all(w.is_alive() for w in loader._iterator._workers)
            pids.append(epochs)
            states = cleanup(loader, traced)
            if observed:
                task_result = analyze_worker_tasks(measured.loader,loader,worker_states=states) if traced else None
                diag = measured.loader.diagnostics(worker_tasks=task_result)
                source = measured.loader.to_dict()
                assert measured.loader.to_dict() == source
                assert len(diag['batch_timeline']) == 6
                assert [r['samples'] for r in diag['batch_timeline']] == [2,3,3]*2
                assert all(r['cpu_forward_backward_wall_ms'] is not None for r in diag['batch_timeline'])
                assert all(s['samples_per_second'] > 0 for s in diag['findings'])
                assert diag['queue_task_scopes']['physical_queue_residence']['value'] is None
                assert diag['queue_task_scopes']['queue_occupancy']['value'] is None
                assert diag['queue_task_scopes']['prefetch_efficiency']['value'] is None
                assert diag['recommendations'] == []
                if traced:
                    assert diag['worker']['association'] == dict(execution='verified',epoch='verified',batch='verified')
                    for stage in ['load','decode','transform']:
                        assert diag['worker']['stages'][stage]['count'] == 16
                        assert diag['worker']['stages'][stage]['status'] == 'measured'
                    assert diag['worker']['stages']['transform']['mean_ms'] >= 10
                    assert all(r['worker_task']['worker_service']['milliseconds'] > 0 for r in diag['batch_timeline'])
                    assert all(r['worker_task']['parent_buffer']['residence_ms'] >= 0 for r in diag['batch_timeline'])
                    loader.trace_channel.close()
                else:
                    assert diag['worker']['status'] == 'unsupported'
                    assert all(r['worker_task'] is None for r in diag['batch_timeline'])
            del loader
            gc.collect()
            deadline = time.monotonic()+8
            while mp.active_children() and time.monotonic()<deadline:
                time.sleep(.05); gc.collect()
            assert not mp.active_children()
        equal(records[0],records[1])
        assert not torch.cuda.is_initialized()
        return dict(case=case,controlled_transform_delay_ms=10,real_files=True,
            exact_batch_model_optimizer_rng_generator_history=True,persistent_worker_pids=pids,
            all_workers_exited=True,parent=source,diagnostics=diag,
            task_evidence=task_result.to_dict() if task_result is not None else None,
            content_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in records[0][2])).hexdigest())


if __name__ == '__main__':
    mp.freeze_support()
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['standard_images','task_images'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--admit',type=Path,required=True)
    args=parser.parse_args();deadline=time.monotonic()+15
    while not args.admit.exists() and time.monotonic()<deadline: time.sleep(.01)
    assert args.admit.exists(), 'test Job admission required'
    args.output.write_text(json.dumps(run(args.case),indent=2)+'\n',encoding='utf-8')
    print('PASS',args.case,flush=True)
