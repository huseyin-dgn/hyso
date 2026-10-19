"""Safe OneCycle representation bridging and strict validation/rollback."""
import copy
import pytest
import torch
from hyso.checkpoint import (save_training_checkpoint, load_training_checkpoint,
                             TrainingCheckpointError, _scheduler_state)
from test_training_checkpoint import assert_state
from test_ebo_training_resume import presentation


def components(optimizer='adamw', strategy='cos', legacy=False):
    model = torch.nn.Linear(3, 2)
    opt = (torch.optim.SGD(model.parameters(), lr=.01, momentum=.9) if optimizer == 'sgd'
           else torch.optim.AdamW(model.parameters(), lr=.01))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.03, total_steps=10,
                                                anneal_strategy=strategy, three_phase=True)
    if legacy:
        # Native 2.5.1 explicitly supports this old representation in _anneal_func.
        # Keep the actual 2.2.2 field order, rather than appending a synthetic key.
        function = {'cos': type(sched)._annealing_cos, 'linear': type(sched)._annealing_linear}[strategy]
        native = {('anneal_func' if key == '_anneal_func_type' else key):
                  (function if key == '_anneal_func_type' else value)
                  for key, value in sched.__dict__.items()}
        sched.__dict__.clear()
        sched.__dict__.update(native)
    return model, opt, sched


def step(model, opt, sched):
    opt.zero_grad()
    model(torch.arange(6, dtype=torch.float32).reshape(2, 3)).square().mean().backward()
    opt.step(); sched.step()


@pytest.mark.parametrize('optimizer', ['sgd', 'adamw'])
@pytest.mark.parametrize('strategy', ['cos', 'linear'])
@pytest.mark.parametrize('legacy', [False, True])
def test_onecycle_round_trip_and_next_step(tmp_path, optimizer, strategy, legacy):
    model, opt, sched = components(optimizer, strategy, legacy)
    for _ in range(3): step(model, opt, sched)
    path = tmp_path/'cycle.pth'
    save_training_checkpoint(path, model, optimizer=opt, scheduler=sched)
    raw = torch.load(path, weights_only=True)
    state = raw['scheduler']['state_dict']
    assert raw['format_version'] == 1
    assert state['_anneal_func_type'] == strategy and 'anneal_func' not in state
    if not legacy:
        assert_state(state, sched.state_dict())  # Existing 2.5.1 file representation unchanged.
    target, target_opt, target_sched = components(optimizer, 'linear' if strategy == 'cos' else 'cos', not legacy)
    load_training_checkpoint(path, target, optimizer=target_opt, scheduler=target_sched)
    assert_state(target_opt.state_dict(), opt.state_dict())
    assert_state(_scheduler_state(target_sched), _scheduler_state(sched))
    step(model, opt, sched); step(target, target_opt, target_sched)
    assert_state(target.state_dict(), model.state_dict())
    assert_state(target_opt.state_dict(), opt.state_dict())  # LR and SGD momentum/Adam beta1 included.
    assert_state(_scheduler_state(target_sched), _scheduler_state(sched))
    assert target_sched.last_epoch == 4 and target_sched.total_steps == 10


@pytest.mark.parametrize('bad', ['unknown', 'integer', 'missing', 'conflicting', 'extra'])
def test_bad_annealing_state_rejected_before_any_mutation(tmp_path, bad):
    model, opt, sched = components()
    step(model, opt, sched)
    path = tmp_path/'invalid.pth'
    save_training_checkpoint(path, model, optimizer=opt, scheduler=sched, include_rng=True)
    payload = torch.load(path, weights_only=True)
    state = payload['scheduler']['state_dict']
    if bad == 'unknown': state['_anneal_func_type'] = 'custom'
    elif bad == 'integer': state['_anneal_func_type'] = 1
    elif bad == 'missing': del state['_anneal_func_type']
    elif bad == 'conflicting': state['anneal_func'] = 'cos'
    else: state['unknown_state'] = 3
    torch.save(payload, path)
    target, target_opt, target_sched = components(legacy=True)
    before = [copy.deepcopy(obj.state_dict()) for obj in (target, target_opt, target_sched)]
    rng = torch.get_rng_state().clone()
    with pytest.raises(TrainingCheckpointError, match='OneCycleLR'):
        load_training_checkpoint(path, target, optimizer=target_opt, scheduler=target_sched, restore_rng=True)
    for obj, expected in zip((target, target_opt, target_sched), before):
        assert_state(obj.state_dict(), expected)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('change', ['same_name', 'both', 'unknown_fields'])
def test_untrusted_live_onecycle_representation_rejected(tmp_path, change):
    model, opt, sched = components(legacy=True)
    if change == 'same_name':
        def fake(*args): raise AssertionError('must never execute arbitrary callable')
        fake.__name__ = type(sched)._annealing_cos.__name__
        sched.anneal_func = fake
    elif change == 'both': sched._anneal_func_type = 'cos'
    else: sched.custom_state = 1
    with pytest.raises(TrainingCheckpointError, match='OneCycleLR'):
        save_training_checkpoint(tmp_path/'invalid.pth', model, optimizer=opt, scheduler=sched)
    assert not (tmp_path/'invalid.pth').exists()


@pytest.mark.parametrize('legacy', [False, True])
def test_onecycle_load_failure_rolls_back_native_state(tmp_path, monkeypatch, legacy):
    model, opt, sched = components(strategy='linear')
    step(model, opt, sched)
    path = tmp_path/'rollback.pth'
    save_training_checkpoint(path, model, optimizer=opt, scheduler=sched)
    target, target_opt, target_sched = components(legacy=legacy)
    before = [copy.deepcopy(obj.state_dict()) for obj in (target, target_opt, target_sched)]
    original = type(sched).load_state_dict
    calls = []
    def fail_once(self, state):
        calls.append(True)
        original(self, state)
        if len(calls) == 1: raise RuntimeError('failure after native scheduler restore')
    monkeypatch.setattr(type(sched), 'load_state_dict', fail_once)
    with pytest.raises(TrainingCheckpointError, match='previous state restored'):
        load_training_checkpoint(path, target, optimizer=target_opt, scheduler=target_sched)
    assert len(calls) == 2
    for obj, expected in zip((target, target_opt, target_sched), before):
        assert_state(obj.state_dict(), expected)


def test_external_callable_checkpoint_never_uses_unsafe_load(tmp_path, monkeypatch):
    model, opt, sched = components()
    path = tmp_path/'external.pth'
    save_training_checkpoint(path, model, optimizer=opt, scheduler=sched)
    payload = torch.load(path, weights_only=True)
    payload['scheduler']['state_dict']['anneal_func'] = type(sched)._annealing_cos
    torch.save(payload, path)  # External pickle; intentionally not our safe save API.
    original, requests = torch.load, []
    def observed(*args, **kwargs):
        requests.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', observed)
    with pytest.raises(TrainingCheckpointError, match='weights-only'):
        load_training_checkpoint(path, model, optimizer=opt, scheduler=sched)
    assert len(requests) == 1 and requests[0]['weights_only'] is True


def test_annealing_plan_disagreement_rejected(tmp_path, monkeypatch, presentation):
    from test_ebo_training_resume import interrupted, loader, options
    from training_helpers import tiny_ebo
    opts = options(scheduler='onecycle')
    path = tmp_path/'plan.pth'
    interrupted(monkeypatch, tiny_ebo(), loader(), path, opts, stop=3)
    payload = torch.load(path, weights_only=True)
    payload['scheduler']['state_dict']['_anneal_func_type'] = 'linear'
    torch.save(payload, path)
    target = tiny_ebo()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(TrainingCheckpointError, match='configuration/budget'):
        target.fit(loader(), resume_from=path, **opts)
    assert_state(target.state_dict(), before)
