"""Production-only CPU probe for old/new Torch; no pytest or test-only AMP API.

Run directly. --output retains plain checkpoints for a --peer check in another
runtime. --legacy accepts genuine pre-fix Torch 2.5.1 checkpoint evidence.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hyso.models import EBO, bcam
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint, _scheduler_state


def exact(a, b):
    if isinstance(b, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(b, dict):
        assert list(a) == list(b)
        for key in b: exact(a[key], b[key])
    elif isinstance(b, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b): exact(left, right)
    else: assert a == b


def seed(value):
    random.seed(value); np.random.seed(value)
    torch.random.default_generator.manual_seed(value)


def components(optimizer, strategy, three_phase):
    net = torch.nn.Linear(3, 2)
    opt = (torch.optim.SGD(net.parameters(), lr=.01, momentum=.9) if optimizer == 'sgd'
           else torch.optim.AdamW(net.parameters(), lr=.01))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.03, total_steps=10,
                anneal_strategy=strategy, three_phase=three_phase)
    return net, opt, sched


def step(net, opt, sched):
    opt.zero_grad()
    net(torch.arange(6, dtype=torch.float32).reshape(2, 3)).square().mean().backward()
    opt.step(); sched.step()


def component_cases(output, peer):
    count = 0
    for optimizer in ('sgd', 'adamw'):
        for strategy in ('cos', 'linear'):
            for three in (False, True):
                name = optimizer + '-' + strategy + '-' + str(three)
                seed(7)
                net, opt, sched = components(optimizer, strategy, three)
                for _ in range(3): step(net, opt, sched)
                path = output / (name + '.pth')
                save_training_checkpoint(path, net, optimizer=opt, scheduler=sched)
                raw = torch.load(path, weights_only=True, map_location='cpu')
                assert raw['format_version'] == 1
                assert raw['scheduler']['state_dict']['_anneal_func_type'] == strategy
                assert 'anneal_func' not in raw['scheduler']['state_dict']
                target, target_opt, target_sched = components(optimizer, 'linear' if strategy == 'cos' else 'cos', three)
                load_training_checkpoint(path, target, optimizer=target_opt, scheduler=target_sched)
                exact(target.state_dict(), net.state_dict())
                exact(target_opt.state_dict(), opt.state_dict())
                exact(_scheduler_state(target_sched), _scheduler_state(sched))
                step(net, opt, sched); step(target, target_opt, target_sched)
                exact(target.state_dict(), net.state_dict())
                exact(target_opt.state_dict(), opt.state_dict())
                exact(_scheduler_state(target_sched), _scheduler_state(sched))
                future = {'lr': opt.param_groups[0]['lr'],
                          'momentum': opt.param_groups[0].get('momentum'),
                          'betas': list(opt.param_groups[0]['betas']) if 'betas' in opt.param_groups[0] else None}
                (output / (name + '.json')).write_text(json.dumps(future), encoding='utf-8')
                if peer:
                    load_training_checkpoint(peer / (name + '.pth'), target,
                                             optimizer=target_opt, scheduler=target_sched)
                    step(target, target_opt, target_sched)
                    expected = json.loads((peer / (name + '.json')).read_text(encoding='utf-8'))
                    assert target_opt.param_groups[0]['lr'] == expected['lr']
                    assert target_opt.param_groups[0].get('momentum') == expected['momentum']
                    if expected['betas'] is not None:
                        assert list(target_opt.param_groups[0]['betas']) == expected['betas']
                count += 1
    return count


def tiny():
    return EBO(input_channels=3, num_classes=2, conv_channels=[4, 8], residual_per_block=1,
               p_drop=0., att_reduction=2, norm_type='auto', device='cpu', use_depthwise=False)


def loader():
    data = torch.randn(8, 3, 16, 16, generator=torch.Generator().manual_seed(17))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, torch.arange(8)%2),
                                       batch_size=3, shuffle=True)


def resume_case(output, scheduler, strategy='cos'):
    opts = dict(epochs=5, lr=.005, weight_decay=.001, optimizer_name='adamw', use_ema=True,
                use_amp=False, accum_steps=2, grad_clip=.5, scheduler_name=scheduler,
                max_lr=.01, scheduler_params=({'step_size':2, 'gamma':.5} if scheduler=='steplr'
                else {'anneal_strategy': strategy}))
    train = loader()
    seed(19); full = tiny()
    full.fit(train, save_last_to=output/'full.pth', **opts)
    expected = torch.load(output/'full.pth', weights_only=True, map_location='cpu')
    seed(19); first = tiny()
    first.optimizer = torch.optim.AdamW(first.model.parameters(), lr=.005, weight_decay=.001)
    attempts = []
    class Stop(Exception): pass
    def interrupt(*args):
        attempts.append(True)
        if len(attempts) == 7: raise Stop
    handle = first.optimizer.register_step_pre_hook(interrupt)
    try: first.fit(train, save_last_to=output/'cut.pth', **opts)
    except Stop: pass
    else: raise AssertionError('controlled interruption missing')
    finally: handle.remove()
    cut = torch.load(output/'cut.pth', weights_only=True)
    assert cut['completed_epochs'] == 3 and cut['metrics']['ebo_fit']['epochs'] == 5
    seed(999); resumed = tiny()
    resumed.fit(loader(), resume_from=output/'cut.pth', save_last_to=output/'last.pth', **opts)
    actual = torch.load(output/'last.pth', weights_only=True, map_location='cpu')
    exact(actual, expected)
    assert actual['scheduler']['state_dict']['last_epoch'] == (5 if scheduler=='steplr' else 10)
    return {'scheduler':scheduler, 'strategy':strategy if scheduler=='onecycle' else None,
            'exact_resume':True, 'optimizer_steps':10}


def legacy_case(directory):
    # These files were created by the actual pre-fix 2.5.1 production API.
    net = torch.nn.Linear(3,2)
    opt = torch.optim.SGD(net.parameters(), lr=.01, momentum=.9)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.03, total_steps=4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    from hyso.models import EMA
    ema = EMA(net)
    load_training_checkpoint(directory/'linear.pth', net, optimizer=opt, scheduler=sched, scaler=scaler, ema=ema)
    raw = torch.load(directory/'linear.pth', weights_only=True)
    exact(net.state_dict(), raw['model']['state_dict'])
    exact(_scheduler_state(sched), raw['scheduler']['state_dict'])
    opts = dict(epochs=2, lr=.005, weight_decay=.001, optimizer_name='adamw', use_ema=True,
                use_amp=False, accum_steps=2, scheduler_name='onecycle', max_lr=.01,
                scheduler_params={'cycle_momentum':False})
    train = loader()
    val = torch.utils.data.DataLoader(train.dataset, batch_size=3)
    seed(999)
    resumed = tiny()
    history = resumed.fit(train, val_loader=val, resume_from=directory/'cut.pth', **opts)
    assert len(history['train_loss']) == 2 and resumed.lr_scheduler.scheduler.last_epoch == 4
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--peer', type=Path)
    parser.add_argument('--legacy', type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with (args.output/'training.log').open('w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        count = component_cases(args.output, args.peer)
        resumes = [resume_case(args.output, 'steplr'), resume_case(args.output, 'onecycle'),
                   resume_case(args.output, 'onecycle', 'linear')]
        legacy = legacy_case(args.legacy) if args.legacy else None
        import matplotlib.pyplot as plt
        plt.close('all')
    assert not torch.cuda.is_initialized()
    print(json.dumps({'torch':torch.__version__, 'source':bcam.__file__, 'component_cases':count,
                      'peer_cases':count if args.peer else 0, 'resume':resumes,
                      'legacy_251':legacy, 'cuda_initialized':False}))


if __name__ == '__main__': main()
