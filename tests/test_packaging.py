"""Build/install the distribution outside the checkout using the existing runtime."""

import ast
import configparser
import email
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "hyso/benchmark/__init__.py", "hyso/benchmark/runner.py", "hyso/benchmark/result.py",
    "hyso/training/resume.py",
    "hyso/training/__init__.py", "hyso/training/trainer.py",
    "hyso/models/modular.py",
    "hyso/models/attention.py",
    "hyso/models/blocks.py",
    "hyso/checkpoint.py",
    "hyso/__init__.py", "hyso/models/__init__.py", "hyso/models/bcam.py",
    "hyso/models/WithSE.py", "hyso/models/fit.py",
    "hyso/augmentation/__init__.py", "hyso/augmentation/augmentation.py",
    "hyso/callbacks/__init__.py", "hyso/callbacks/callbacks.py",
}
DEPENDENCIES = {"torch", "torchvision", "numpy", "matplotlib", "seaborn",
                "scikit-learn", "tqdm", "torchsummary"}
CONFIG_FILES = ("pyproject.toml", "setup.cfg", "MANIFEST.in", "README.md", "LICENSE")


def command(args, cwd):
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-I", "-B", *args], cwd=cwd,
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    work = tmp_path_factory.mktemp("packaging")
    stage = work / "source"
    stage.mkdir()
    for name in (*CONFIG_FILES, *MODULES):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    # Realistic unwanted content must stay out of both archives.
    for name in ("docs/audit/sentinel.md", "tests/test_sentinel.py",
                 "hyso/dist/old.whl", "hyso/models/weights.pth",
                 "hyso/models/__pycache__/sentinel.pyc"):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("excluded", encoding="utf-8")
    dist = work / "dist"
    log = command(["-m", "build", "--no-isolation", "--sdist", "--wheel",
                   "--outdir", str(dist), str(stage)], work)
    (work / "build.log").write_text(log, encoding="utf-8")
    return work, next(dist.glob("*.whl")), next(dist.glob("*.tar.gz"))


def test_canonical_packaging_configuration():
    assert (ROOT / "pyproject.toml").is_file()
    assert not (ROOT / "hyso/pyproject.toml").exists()
    assert not (ROOT / "hyso/setup.cfg").exists()
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "setup.cfg", encoding="utf-8")
    assert cfg["metadata"]["name"] == "hyso"
    assert cfg["metadata"]["version"] == "0.2.0"
    assert cfg["options"]["python_requires"] == ">=3.10"
    assert cfg["options.packages.find"]["where"] == "."
    for name in MODULES:
        ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), feature_version=(3, 10))


def test_wheel_and_sdist_content_and_metadata(distributions):
    _, wheel, sdist = distributions
    assert wheel.name == "hyso-0.2.0-py3-none-any.whl"
    assert sdist.name == "hyso-0.2.0.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        wheel_meta = email.message_from_bytes(archive.read("hyso-0.2.0.dist-info/METADATA"))
        assert {name for name in wheel_names if name.endswith(".py")} == MODULES
        assert archive.read("hyso-0.2.0.dist-info/top_level.txt").decode().split() == ["hyso"]
        license_names = [name for name in wheel_names if name.endswith("/LICENSE")]
        assert len(license_names) == 1
        assert archive.read(license_names[0]) == (ROOT / "LICENSE").read_bytes()
        for name in MODULES:
            assert archive.read(name) == (ROOT / name).read_bytes()
    with tarfile.open(sdist) as archive:
        names = {name.removeprefix("hyso-0.2.0/") for name in archive.getnames()}
        sdist_meta = email.message_from_bytes(archive.extractfile("hyso-0.2.0/PKG-INFO").read())
        sdist_requires = archive.extractfile("hyso-0.2.0/hyso.egg-info/requires.txt").read().decode().splitlines()
        assert {name for name in names if name.endswith(".py")} == MODULES
        assert set(CONFIG_FILES) <= names
        # setuptools adds egg_info settings to the sdist's setup.cfg.
        for name in ("README.md", "LICENSE", "pyproject.toml", "MANIFEST.in", *MODULES):
            assert archive.extractfile("hyso-0.2.0/" + name).read() == (ROOT / name).read_bytes()
    for name in wheel_names | names:
        assert "sentinel" not in name and not name.endswith((".pth", ".pt", ".ckpt", ".pyc"))
        assert not any(part in {"docs", "tests", "__pycache__", "dist", "build"} for part in Path(name).parts)
    for metadata in (wheel_meta, sdist_meta):
        assert metadata["Name"] == "hyso" and metadata["Version"] == "0.2.0"
        assert metadata["Requires-Python"] == ">=3.10"
        description = metadata.get_payload(decode=True).decode("utf-8")
        # Email metadata may use CRLF; compare every content line unchanged.
        assert description.strip().splitlines() == (ROOT / "README.md").read_text(encoding="utf-8").strip().splitlines()
    requirements = [Requirement(value) for value in wheel_meta.get_all("Requires-Dist")]
    assert {str(Requirement(value)) for value in sdist_requires if value.strip()} == {str(item) for item in requirements}
    assert {item.name for item in requirements} == DEPENDENCIES
    assert str(next(item.specifier for item in requirements if item.name == "torch")) == ">=2.2.2"
    assert str(next(item.specifier for item in requirements if item.name == "torchvision")) == ">=0.17.2"
    assert not next(item.specifier for item in requirements if item.name == "numpy")


SMOKE = """
import importlib, importlib.metadata, json, pathlib, sys
target = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import torch, hyso
torch.set_num_threads(1)
from hyso.models import EBO, ELO, CEMO, EMA, CNNSENormalWithFit
from hyso.models.modular import ModularCNN
from hyso.training import Trainer
from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.augmentation import build_transforms, get_loaders
from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
for module, names in {
    'hyso.checkpoint': ('save_training_checkpoint', 'load_training_checkpoint', 'TrainingCheckpointError'),
    'hyso.models.bcam': ('EBO', 'ELO'),
    'hyso.models.blocks': ('ConvNormAct', 'ResidualBlock'),
    'hyso.models.attention': ('ECAAttention', 'CoordinateAttention'),
    'hyso.models.modular': ('ModularCNN',),
    'hyso.training': ('Trainer',),
    'hyso.training.trainer': ('Trainer',),
    'hyso.training.resume': ('prepare_save', 'save_last', 'restore_checkpoint'),
    'hyso.benchmark': ('BenchmarkRunner', 'BenchmarkResult'),
    'hyso.benchmark.runner': ('BenchmarkRunner',),
    'hyso.benchmark.result': ('BenchmarkResult',),
    'hyso.models.WithSE': ('CEMO', 'EMA'),
    'hyso.models.fit': ('CNNSENormalWithFit',),
    'hyso.augmentation.augmentation': ('build_transforms', 'get_loaders'),
    'hyso.callbacks.callbacks': ('EarlyStopping', 'ModelCheckpoint', 'LRSchedulerCallback'),
}.items():
    loaded = importlib.import_module(module)
    assert pathlib.Path(loaded.__file__).resolve().is_relative_to(target)
    assert all(hasattr(loaded, name) for name in names)
assert pathlib.Path(hyso.__file__).resolve().is_relative_to(target)
assert importlib.metadata.version('hyso') == '0.2.0'
x = torch.randn(2, 3, 16, 16)
models = [
    EBO(conv_channels=[4, 8], residual_per_block=1, device='cpu'),
    ELO(conv_channels=[4, 8], residual_per_block=1),
    CEMO(conv_channels=[4, 8], residuals_per_stage=1, use_ema=False, device='cpu'),
]
with torch.no_grad():
    for model in models:
        output = model.eval()(x)
        assert output.shape == (2, 10) and output.device.type == 'cpu'
        assert torch.isfinite(output).all()
    modular = ModularCNN(input_channels=1, num_classes=2, widths=(4, 6),
                         repeats=(1, 1), strides=(1, 2), attention='coordinate').eval()
    modular_input = torch.randn(2, 1, 15, 19)
    assert modular.forward_features(modular_input).shape == (2, 6, 8, 10)
    modular_output = modular(modular_input)
    assert modular_output.shape == (2, 2) and torch.isfinite(modular_output).all()
optimizer = torch.optim.SGD(modular.parameters(), lr=0.01)
before = modular.classifier.weight.detach().clone()
history = Trainer(modular, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss()).fit(
    [(modular_input, torch.tensor([0, 1]))], epochs=1)
assert len(history['train_loss']) == 1 and history['val_loss'] == [None]
assert torch.isfinite(torch.tensor(history['train_loss'])).all()
assert not torch.equal(before, modular.classifier.weight)
# Exercise B2's public API using installed files, retaining the B1 smoke above.
linear = torch.nn.Linear(2, 2).double()
update_optimizer = torch.optim.AdamW(linear.parameters(), lr=0.01)
update_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    update_optimizer, max_lr=0.03, total_steps=4, cycle_momentum=False)
update_trainer = Trainer(linear, optimizer=update_optimizer,
                         loss_fn=torch.nn.CrossEntropyLoss(), accum_steps=2,
                         grad_clip=0.1, scheduler=update_scheduler, amp=False)
update_x = torch.tensor([[1., 0.], [0., 1.], [-1., 1.], [1., 1.], [-1., 0.]], dtype=torch.float64)
update_y = torch.tensor([0, 1, 1, 0, 1])
update_before = linear.weight.detach().clone()
update_history = update_trainer.fit(list(zip(update_x.split([2, 2, 1]), update_y.split([2, 2, 1]))), epochs=2)
assert update_trainer.attempt_count == update_trainer.update_count == update_scheduler.last_epoch == 4
assert not torch.equal(update_before, linear.weight)
assert all(p.grad is None for p in linear.parameters())
assert update_history['val_loss'] == [None, None]
# B3 uses the legacy algorithms through the installed generic lifecycle.
early = EarlyStopping(patience=1, verbose=False, restore_best_weights=True)
lifecycle_model = torch.nn.Linear(2, 2)
with torch.no_grad():
    lifecycle_model.weight.zero_()
    lifecycle_model.bias.zero_()
lifecycle_optimizer = torch.optim.SGD(lifecycle_model.parameters(), lr=0.3)
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(lifecycle_optimizer, patience=0, factor=0.5)
lifecycle_events = []
lifecycle = Trainer(lifecycle_model, optimizer=lifecycle_optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
                    ema_decay=0.5, early_stopping=early, scheduler=plateau,
                    callbacks=(lambda event, logs: lifecycle_events.append(event),))
lifecycle_history = lifecycle.fit([(torch.zeros(2, 2), torch.ones(2, dtype=torch.long))],
                                  [(torch.zeros(3, 2), torch.zeros(3, dtype=torch.long))], epochs=5)
assert len(lifecycle_history['val_loss']) == 2 and early.early_stop
assert lifecycle.update_count == lifecycle.ema_update_count == plateau.last_epoch == 2
assert lifecycle_events == ['train_begin', 'epoch_end', 'epoch_end', 'train_end']
assert all(torch.equal(value, early.best_state[name]) for name, value in lifecycle.evaluation_model.state_dict().items())
assert not torch.equal(lifecycle.model.bias, lifecycle.evaluation_model.bias)
save_config = dict(input_channels=1, num_classes=2, widths=[3], repeats=[1], strides=[1], attention='eca', norm='batch')
save_model = ModularCNN(**save_config)
save_optimizer = torch.optim.AdamW(save_model.parameters(), lr=0.01)
save_scheduler = torch.optim.lr_scheduler.StepLR(save_optimizer, step_size=1)
save_trainer = Trainer(save_model, optimizer=save_optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
                       model_config=save_config, scheduler=save_scheduler, ema_decay=0.8, accum_steps=2)
save_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
    torch.randn(5, 1, 5, 5), torch.tensor([0, 1, 1, 0, 1])), batch_size=2)
save_path = pathlib.Path.cwd() / ('generic-last-' + target.name + '.pth')
save_trainer.fit(save_loader, epochs=2, save_last_to=save_path)
saved = torch.load(save_path, map_location='cpu', weights_only=True)
assert saved['format_version'] == 1 and saved['completed_epochs'] == 2
assert saved['metrics']['generic_trainer']['progress']['updates'] == 4
assert saved['optimizer']['state_dict']['state'] and saved['ema'] is not None
assert saved['rng_state'] is not None
restore_model = ModularCNN(**save_config)
restore_optimizer = torch.optim.AdamW(restore_model.parameters(), lr=0.01)
restore_scheduler = torch.optim.lr_scheduler.StepLR(restore_optimizer, step_size=1)
restored = Trainer(restore_model, optimizer=restore_optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
                   model_config=save_config, scheduler=restore_scheduler, ema_decay=0.8, accum_steps=2)
restored_history = restored.fit(save_loader, epochs=2, resume_from=save_path)
assert restored_history == saved['history']
assert restored.update_count == restored.attempt_count == 4 and restore_scheduler.last_epoch == 2
assert all(torch.equal(tensor, saved['model']['state_dict'][name]) for name, tensor in restore_model.state_dict().items())
assert all(torch.equal(tensor, saved['ema']['state_dict'][name]) for name, tensor in restored.evaluation_model.state_dict().items())
import runpy
replay_support = runpy.run_path(sys.argv[2])
replay_result = replay_support['run_replay'](
    replay_support['CRITICAL'], pathlib.Path.cwd() / ('installed-replay-' + target.name))
assert replay_result['epochs'] == 5 and replay_result['resumed_attempts'] == 4
assert replay_result['exact_bytes'] and replay_result['sample_order_equal'] and replay_result['rng_equal']
benchmark_rows = []
for precision, dtype in [('fp32', torch.float32), ('fp64', torch.float64)]:
    bench_model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.BatchNorm2d(2),
                                      torch.nn.Dropout(0.5), torch.nn.AdaptiveAvgPool2d(1)).to(dtype=dtype)
    bench_model[1].eval()
    bench_state = {name: value.clone() for name, value in bench_model.state_dict().items()}
    bench_modes = [module.training for module in bench_model.modules()]
    bench_rng = torch.get_rng_state().clone()
    bench_result = BenchmarkRunner(bench_model, input_shape=(2, 1, 5, 7), precision=precision,
                                   warmup_iterations=1, measure_iterations=2, repeats=2).run()
    assert isinstance(bench_result, BenchmarkResult)
    bench_data = bench_result.to_dict()
    assert bench_data['schema_version'] == 1 and bench_data['benchmark_type'] == 'inference'
    assert bench_data['stats']['pooled']['count'] == 4 and bench_data['stats']['pooled']['mean_ms'] > 0
    assert all(len(row) == 2 and all(t > 0 for t in row) for row in bench_data['timing']['latency_samples_ms'])
    assert bench_data['throughput']['method'] == 'derived_serial_service_rate'
    assert bench_data['parameters']['total_parameters'] == sum(p.numel() for p in bench_model.parameters())
    assert [module.training for module in bench_model.modules()] == bench_modes
    assert all(torch.equal(value, bench_state[name]) for name, value in bench_model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), bench_rng)
    assert BenchmarkResult(bench_data).to_dict() == bench_data
    assert json.loads(json.dumps(bench_data, allow_nan=False)) == bench_data
    benchmark_rows.append(bench_data)
for name, module in list(sys.modules.items()):
    if name == 'hyso' or name.startswith('hyso.'):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(target)
assert not torch.cuda.is_initialized()
print(json.dumps({'source': hyso.__file__, 'version': importlib.metadata.version('hyso'),
                  'modular_source': importlib.import_module('hyso.models.modular').__file__,
                  'modular_output_shape': list(modular_output.shape),
                  'trainer_source': importlib.import_module('hyso.training.trainer').__file__,
                  'trainer_history': history,
                  'trainer_updates': update_trainer.update_count,
                  'lifecycle_epochs': len(lifecycle_history['val_loss']),
                  'saved_epochs': saved['completed_epochs'],
                  'restored_updates': restored.update_count,
                  'replay': replay_result,
                  'cpu_benchmarks': len(benchmark_rows),
                  'benchmarks': benchmark_rows,
                  'benchmark_source': importlib.import_module('hyso.benchmark.runner').__file__,
                  'cpu_models': 3, 'cuda_initialized': torch.cuda.is_initialized()}))
"""


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_installed_distribution_import_and_cpu_forward(distributions, kind):
    work, wheel, sdist = distributions
    target = work / ("installed-" + kind)
    artifact = wheel if kind == "wheel" else sdist
    command(["-m", "pip", "--isolated", "--no-cache-dir", "install", "--no-index",
             "--no-deps", "--no-build-isolation", "--no-compile",
             "--target", str(target), str(artifact)], work)
    observed = json.loads(command(["-c", SMOKE, str(target), str(ROOT / "tests/trainer_replay_support.py")], work).strip().splitlines()[-1])
    assert Path(observed["source"]).resolve().is_relative_to(target.resolve())
    assert observed["cpu_models"] == 3 and observed["cuda_initialized"] is False
    assert Path(observed["trainer_source"]).resolve().is_relative_to(target.resolve())
    assert len(observed["trainer_history"]["train_loss"]) == 1
    assert observed["trainer_updates"] == 4
    assert observed["lifecycle_epochs"] == 2
    assert observed["saved_epochs"] == 2
    assert observed["restored_updates"] == 4
    assert observed["replay"]["epochs"] == 5 and observed["replay"]["resumed_attempts"] == 4
    assert observed["replay"]["exact_bytes"] and observed["replay"]["rng_equal"]
    assert observed["cpu_benchmarks"] == 2
    assert Path(observed["benchmark_source"]).resolve().is_relative_to(target.resolve())
    assert {row["configuration"]["precision"] for row in observed["benchmarks"]} == {"fp32", "fp64"}
    (work / (kind + "-smoke.json")).write_text(json.dumps(observed), encoding="utf-8")
