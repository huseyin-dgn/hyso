"""P23-only validation recipes for the existing trained A/B checkpoints.

Bounded real train/validation subsets; originals are read-only. No forensic code.
"""
import hashlib
import json
from pathlib import Path
import sys
import weakref

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso._evaluation_evidence import model_hash
from hyso.checkpoint import load_training_checkpoint
from hyso.models.modular import ModularCNN
from hyso.training import Trainer


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def trained_pair(*, device, arm, assets_path, state_path=None, failure=False,
                 require_no_activation_adapter=False):
    if require_no_activation_adapter:
        assert "hyso.performance._saved_activation" not in sys.modules
        from hyso.performance import isolated_training_memory
        def forbidden(*args, **kwargs):
            raise AssertionError("basic worker entered saved-activation adapter/hooks")
        isolated_training_memory._activation_adapter = forbidden
        torch.autograd.graph.saved_tensors_hooks = forbidden
    assets = json.loads(Path(assets_path).read_bytes())
    item = assets["models"][arm]
    assert file_sha(item["path"]) == item["sha256"]
    model = ModularCNN(**item["config"])
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loader = Trainer(model, optimizer=opt, loss_fn=torch.nn.CrossEntropyLoss(), device="cpu",
                     amp=False, model_config=item["config"])
    receipt = load_training_checkpoint(item["path"], model, optimizer=opt, scaler=loader.scaler,
                                      map_location="cpu", restore_rng=False, capture_evidence=True)
    assert receipt["completed_epochs"] == item["completed_epochs"]
    assert model_hash(model) == item["recorded_state_sha256"]
    assert sum(p.numel() for p in model.parameters()) == item["parameters"]
    # Match the existing acceptance training recipe: loaded weights, fresh AdamW.
    model.train()
    model.to(device)  # Trainer requires placement before optimizer construction.
    trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001),
                      loss_fn=torch.nn.CrossEntropyLoss(), device=device, amp=False, model_config=item["config"])
    for key in ("dataset", "split_indices"):
        assert file_sha(assets[key]["path"]) == assets[key]["sha256"]
    with np.load(assets["split_indices"]["path"], allow_pickle=False) as split:
        train_ids = split["train"][:9].copy()
        val_ids = split["validation" if "validation" in split.files else "val"][:5].copy()
    with np.load(assets["dataset"]["path"], allow_pickle=False) as data:
        def subset(ids):
            return TensorDataset(torch.from_numpy(data["x_train"][ids].copy()).unsqueeze(1).float().div_(255),
                                 torch.from_numpy(data["y_train"][ids].astype(np.int64)))
        train, val = subset(train_ids), subset(val_ids)
    if failure:
        def fail(*args):
            raise RuntimeError("P23 basic controlled forward failure")
        model.register_forward_pre_hook(fail)
    if state_path:
        owner = weakref.ref(trainer)
        def fit(*args, **kwargs):
            from training_memory_factories import state_digest
            current = owner()
            result = Trainer.fit(current, *args, **kwargs)
            if require_no_activation_adapter:
                assert "hyso.performance._saved_activation" not in sys.modules
            Path(state_path).write_text(json.dumps(dict(state_sha256=state_digest(current), fits=1)), encoding="utf-8")
            return result
        trainer.fit = fit
    return dict(trainer=trainer, train_loader=DataLoader(train, batch_size=4, shuffle=False),
                val_loader=DataLoader(val, batch_size=4, shuffle=False))
