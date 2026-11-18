"""Real PNG/CUDA resume integration; run directly, outside CPU pytest fixtures."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings
from unittest.mock import patch

# Import the existing GPU harness before torch initializes CUDA.
from cuda_runtime_validation import model, seed, no_plot
import numpy as np
import torch
import torchvision
from PIL import Image

from hyso.checkpoint import TrainingCheckpointError, _rng_state
from hyso.models import bcam
from hyso.models.WithSE import EMA
from test_cv_dataset_resume import image_loader
from test_ebo_training_resume import options
from test_training_checkpoint import assert_state


def make_images(root):
    y, x = np.mgrid[:32, :32]
    for label, name in enumerate(("alpha", "beta")):
        (root / name).mkdir(parents=True)
        for index in range(4):
            pixels = np.stack([(x * 11 + index * 31) % 256,
                               (y * 13 + label * 91) % 256,
                               ((x + y) * 7 + index * 19) % 256], axis=-1).astype(np.uint8)
            Image.fromarray(pixels).save(root / name / (str(index) + ".png"))


def fit_trace(net, train, validation, opts, trace=None, **paths):
    """Observe real batches, backwards, optimizer steps and scaler without changing math."""
    if trace is None:
        trace = {}
    trace.update(batches=[], updates=0, backward=0, attempts=0, active_scaler=False,
                 ema_updates=0, ema_formula=False)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opts["optimizer_name"]]
    net.optimizer = optimizer_class(net.model.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"])
    handles = []

    def forward(module, inputs, output):
        assert inputs[0].device.type == output.device.type == "cuda"
        if module.training:
            assert output.dtype == torch.float16
        data = inputs[0].detach().cpu().contiguous().numpy()
        trace["batches"].append([module.training, len(data), hashlib.sha256(memoryview(data)).hexdigest()])

    def stepped(*args):
        trace["updates"] += 1

    def backward(gradient):
        assert gradient.device.type == "cuda"
        trace["backward"] += 1
        return gradient

    handles.append(net.model.register_forward_hook(forward))
    handles.append(net.optimizer.register_step_post_hook(stepped))
    handles.append(next(net.model.parameters()).register_hook(backward))
    real_scaler, real_clip = torch.cuda.amp.GradScaler, torch.nn.utils.clip_grad_norm_

    def scaler(**kwargs):
        value = real_scaler(**kwargs)  # Default scale; no synthetic overflow outcomes.
        assert value.is_enabled()
        trace["active_scaler"] = True
        return value

    def clip(*args, **kwargs):
        trace["attempts"] += 1
        return real_clip(*args, **kwargs)

    real_update = EMA.update

    def ema_update(shadow, live):
        check = trace["ema_updates"] == 0
        if check:
            previous = {name: p.detach().clone() for name, p in shadow.ema_model.named_parameters()}
        real_update(shadow, live)
        trace["ema_updates"] += 1
        if check:
            for name, current in live.named_parameters():
                expected = previous[name].mul(shadow.decay).add(current.detach(), alpha=1 - shadow.decay)
                assert_state(dict(shadow.ema_model.named_parameters())[name], expected)
            for name, buffer in live.named_buffers():
                assert_state(dict(shadow.ema_model.named_buffers())[name], buffer)
            trace["ema_formula"] = True

    try:
        with patch.object(torch.cuda.amp, "GradScaler", scaler), \
                patch.object(torch.nn.utils, "clip_grad_norm_", clip), patch.object(EMA, "update", ema_update):
            history = net.fit(train, val_loader=validation, **paths, **opts)
        return trace, history
    finally:
        for handle in handles:
            handle.remove()


def resume_case(root, directory, subset, random, scheduler, optimizer):
    opts = options(optimizer=optimizer, scheduler=scheduler, ema=True)
    reads = []
    real_open = Image.open

    def open_image(source, *args, **kwargs):
        reads.append(Path(getattr(source, "name", source)).relative_to(root).as_posix())
        return real_open(source, *args, **kwargs)

    def train():
        return image_loader(root, subset=subset, random=random, shuffle=not subset)

    def validation():
        return image_loader(root, subset=subset, shuffle=False)

    full_path, cut_path, last_path = [directory / name for name in ("full.pth", "cut.pth", "last.pth")]
    with patch.object(Image, "open", open_image):
        seed(19)
        full = model()
        before = [p.detach().clone() for p in full.parameters()]
        full_trace, history = fit_trace(full, train(), validation(), opts, save_last_to=full_path)
        assert any(not torch.equal(p, old) for p, old in zip(full.parameters(), before))
        expected = torch.load(full_path, map_location="cpu", weights_only=True)
        full_reads = list(reads)
        reads.clear()
        seed(19)
        first = model()
        real_save = bcam.save_training_checkpoint
        partial = {}

        class Stop(Exception):
            pass

        def save(*args, **kwargs):
            real_save(*args, **kwargs)
            if kwargs["completed_epochs"] == 3:
                raise Stop

        with patch.object(bcam, "save_training_checkpoint", save):
            try:
                fit_trace(first, train(), validation(), opts, trace=partial, save_last_to=cut_path)
            except Stop:
                pass
            else:
                raise AssertionError("third-epoch interruption did not occur")
        cut = torch.load(cut_path, map_location="cpu", weights_only=True)
        assert cut["completed_epochs"] == 3
        seed(999)
        resumed = model()
        assert not torch.equal(torch.get_rng_state(), cut["rng_state"]["torch_cpu"])
        assert not torch.equal(torch.cuda.get_rng_state(), cut["rng_state"]["cuda"][0])
        resumed_trace, actual_history = fit_trace(resumed, train(), validation(), opts,
                                                 resume_from=cut_path, save_last_to=last_path)
        actual = torch.load(last_path, map_location="cpu", weights_only=True)
        assert_state(actual, expected)
        assert_state(resumed.state_dict(), full.state_dict())
        assert reads == full_reads and len(reads) == 83
        assert partial["batches"] + resumed_trace["batches"] == full_trace["batches"]
        assert len(full_trace["batches"]) == 30
        assert full_trace["backward"] == 15
        assert full_trace["attempts"] == 10
        assert partial["backward"] == 9 and resumed_trace["backward"] == 6
        assert partial["attempts"] == 6 and resumed_trace["attempts"] == 4
        assert full_trace["active_scaler"] and partial["active_scaler"] and resumed_trace["active_scaler"]
        assert all(trace["ema_formula"] for trace in (full_trace, partial, resumed_trace))
        assert full_trace["ema_updates"] == 15
        assert partial["ema_updates"] == 9 and resumed_trace["ema_updates"] == 6
        updates = full_trace["updates"]
        assert partial["updates"] + resumed_trace["updates"] == updates
        assert 0 < updates <= 10
        assert all(int(entry["step"].item()) == updates for entry in resumed.optimizer.state.values())
        assert actual_history == history and all(len(values) == 5 for values in history.values())
        assert all(np.isfinite(values).all() for values in history.values())
        assert actual["completed_epochs"] == 5 and actual["format_version"] == 1
        counter = actual["scheduler"]["state_dict"]["last_epoch"]
        assert counter == (updates if scheduler == "onecycle" else 5)
        if scheduler == "onecycle":
            assert cut["scheduler"]["state_dict"]["total_steps"] == 10
            assert actual["scheduler"]["state_dict"]["total_steps"] == 10
        for name, buffer in resumed.model.named_buffers():
            assert_state(dict(resumed.ema.ema_model.named_buffers())[name], buffer)
    return dict(dataset="Subset(ImageFolder)" if subset else "ImageFolder", random_transforms=random,
                optimizer=optimizer, scheduler=scheduler, exact_state=True, exact_rng=True,
                image_reads=len(reads), exact_batch_order=True, updates=updates, attempts=10,
                scheduler_counter=counter, ema_formula_exact=True,
                final_scale=actual["scaler"]["state_dict"]["scale"])


def negative_cases(root, cut_path):
    results = []
    for change in ("image", "class_mapping", "transform", "subset_order", "batch_size", "accumulation"):
        train = image_loader(root, subset=True, shuffle=False)
        validation = image_loader(root, subset=True, shuffle=False)
        base = train.dataset.dataset
        opts = options(optimizer="adamw", scheduler="onecycle", ema=True)
        sample = root / "alpha/0.png"
        raw, stat = sample.read_bytes(), sample.stat()
        if change == "image":
            # Metadata fingerprint must catch size changes, not same-metadata edits.
            sample.write_bytes(raw + b"changed")
        elif change == "class_mapping":
            base.class_to_idx = {"alpha": 1, "beta": 0}
        elif change == "transform":
            base.transform.transforms[-1].mean = [0.4] * 3
        elif change == "subset_order":
            train.dataset.indices.reverse()
        elif change == "batch_size":
            train = torch.utils.data.DataLoader(train.dataset, batch_size=4)
        elif change == "accumulation":
            opts["accum_steps"] = 1
        seed(999)
        net = model()
        net.ema = EMA(net.model)
        state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        ema_state = {name: value.detach().clone() for name, value in net.ema.ema_model.state_dict().items()}
        rng = _rng_state()
        try:
            with patch.object(torch.nn.Module, "load_state_dict", side_effect=AssertionError("state load before rejection")) as load:
                try:
                    net.fit(train, val_loader=validation, resume_from=cut_path, **opts)
                except TrainingCheckpointError as error:
                    message = str(error)
                else:
                    raise AssertionError("incompatible CV resume accepted: " + change)
                assert load.call_count == 0
            assert_state(net.state_dict(), state)
            assert_state(net.ema.ema_model.state_dict(), ema_state)
            assert_state(_rng_state(), rng)
            assert net.optimizer.state_dict()["state"] == {}
            results.append(dict(change=change, rejected_before_load=True, model_rng_unchanged=True, error=message))
        finally:
            if change == "image":
                sample.write_bytes(raw)
                os.utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return results


def main():
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability()),
                  cuda_available=True, visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
                  memory_free_total=list(torch.cuda.mem_get_info()))
    matrix = [(False, False, "steplr", "adam"), (False, True, "onecycle", "adamw"),
              (True, True, "steplr", "adam"), (True, False, "onecycle", "adamw")]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            patch.object(bcam, "summary", lambda *args: None), patch.object(bcam.plt, "figure", no_plot), \
            patch.object(bcam, "classification_report", lambda *args, **kwargs: ""), \
            warnings.catch_warnings(record=True) as observed, \
            tempfile.TemporaryDirectory(prefix="hyso-cuda-images-") as temporary:
        directory = Path(temporary)
        root = directory / "images"
        make_images(root)
        result["positive"] = []
        for index, config in enumerate(matrix):
            case_dir = directory / str(index)
            case_dir.mkdir()
            result["positive"].append(resume_case(root, case_dir, *config))
        result["negative"] = negative_cases(root, case_dir / "cut.pth")
    torch.cuda.synchronize()
    result["warnings"] = sorted(set(str(item.message) for item in observed))
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    print("CUDA_IMAGEFOLDER=" + json.dumps(result))


if __name__ == "__main__":
    main()
