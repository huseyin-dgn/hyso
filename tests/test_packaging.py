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
    "hyso/evaluation.py", "hyso/quality.py", "hyso/_quality_memory.py",
    "hyso/performance/_cpu_layer_memory.py", "hyso/performance/_layer_allocator.py",
    "hyso/performance/_layer_memory.py", "hyso/performance/_loader_diagnostics.py",
    "hyso/performance/_loader_task_iterator.py", "hyso/performance/_operator_activity.py",
    "hyso/performance/_profile_evidence.py", "hyso/performance/_stability_diagnostics.py",
    "hyso/performance/_transfer_activity.py", "hyso/performance/layer_overhead.py",
    "hyso/performance/roofline.py", "hyso/performance/roofline_counters.py",
    "hyso/performance/_loader_dataset_trace.py", "hyso/performance/_loader_event_channel.py",
    "hyso/performance/_loader_stage_trace.py", "hyso/performance/_loader_task_trace.py",
    "hyso/performance/_loader_trace_groups.py", "hyso/performance/_loader_trace_validation.py",
    "hyso/performance/_loader_worker_lifecycle.py", "hyso/performance/_loader_worker_trace.py",
    "hyso/performance/_runtime_worker.py", "hyso/performance/devices.py",
    "hyso/performance/energy.py", "hyso/performance/memory_trend.py",
    "hyso/performance/providers.py", "hyso/performance/runtime_result.py",
    "hyso/performance/stability.py", "hyso/performance/streaming.py",
    "hyso/performance/worker_task_result.py", "hyso/performance/worker_trace_result.py",
    "hyso/performance/_saved_activation.py",
    "hyso/performance/isolated_training_memory.py",
    "hyso/performance/_training_process.py",
    "hyso/performance/_training_memory_worker.py",
    "hyso/performance/gradients.py",
    "hyso/performance/training_memory.py",
    "hyso/performance/dataloader.py",
    "hyso/performance/training_details.py",
    "hyso/performance/training.py",
    "hyso/performance/transfers.py", "hyso/performance/_transfer_worker.py",
    "hyso/performance/utilization.py", "hyso/performance/_telemetry.py",
    "hyso/performance/profiling.py", "hyso/performance/_profile_worker.py", "hyso/performance/_operation_models.py",
    "hyso/performance/capacity.py", "hyso/performance/_batch_worker.py",
    "hyso/performance/sweeps.py",
    "hyso/performance/pipeline.py", "hyso/performance/cold.py", "hyso/performance/_cold_worker.py",
    "hyso/performance/memory.py",
    "hyso/performance/__init__.py", "hyso/performance/inference.py", "hyso/performance/result.py",
    "hyso/experiments/__init__.py", "hyso/experiments/manager.py",
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
CONFIG_FILES = ("pyproject.toml", "setup.cfg", "MANIFEST.in", "LICENSE")


def command(args, cwd):
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-I", "-B", *args], cwd=cwd,
                            env=env, capture_output=True, text=True, timeout=240)
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
                 "hyso/experiments/experiments.sqlite3", "hyso/experiments/artifacts/sentinel.json",
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
    assert {path.relative_to(ROOT).as_posix() for path in (ROOT / "hyso").rglob("*.py")} == MODULES
    assert len(MODULES) == 80
    assert (ROOT / "pyproject.toml").is_file()
    assert not (ROOT / "hyso/pyproject.toml").exists()
    assert not (ROOT / "hyso/setup.cfg").exists()
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "setup.cfg", encoding="utf-8")
    assert not cfg.has_option('metadata','author_email')
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
        for name in ("LICENSE", "pyproject.toml", "MANIFEST.in", *MODULES):
            assert archive.extractfile("hyso-0.2.0/" + name).read() == (ROOT / name).read_bytes()
    for name in wheel_names | names:
        assert "sentinel" not in name and not name.endswith((".pth", ".pt", ".ckpt", ".pyc", ".sqlite3", ".db"))
        assert not any(part in {"docs", "tests", "__pycache__", "dist", "build"} for part in Path(name).parts)
    for metadata in (wheel_meta, sdist_meta):
        assert metadata["Name"] == "hyso" and metadata["Version"] == "0.2.0"
        assert metadata["Requires-Python"] == ">=3.10"
        description = metadata.get_payload(decode=True).decode("utf-8")
        # Email metadata may use CRLF; compare every content line unchanged.
        assert description.strip() == "HYSo Python library."
    requirements = [Requirement(value) for value in wheel_meta.get_all("Requires-Dist")]
    section=None;sdist_requirements=[]
    for value in sdist_requires:
        if not value.strip():continue
        if value.startswith('['):
            assert value=='[deployment]'
            section='deployment';continue
        sdist_requirements.append(Requirement(value+('; extra == "deployment"' if section else '')))
    assert {str(item) for item in sdist_requirements} == {str(item) for item in requirements}
    assert {item.name for item in requirements if item.marker is None} == DEPENDENCIES
    optional={item.name:item for item in requirements if item.marker is not None}
    assert set(optional)=={'onnx','onnxruntime'}
    assert str(optional['onnx'].specifier)=='==1.20.0'
    assert str(optional['onnxruntime'].specifier)=='==1.23.2'
    assert all(item.marker.evaluate({'extra':'deployment'}) and not item.marker.evaluate({'extra':''})
               for item in optional.values())
    assert wheel_meta['Author-email'] is None and sdist_meta['Author-email'] is None
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
from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results
from hyso.experiments import ExperimentManager, ExperimentError
from hyso.performance import (TailPolicy, ThroughputProtocol, LatencyAnalysis, ThroughputResult,
                              analyze_latency, measure_throughput)
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
    'hyso.benchmark': ('BenchmarkRunner', 'BenchmarkResult', 'compare_results'),
    'hyso.benchmark.runner': ('BenchmarkRunner',),
    'hyso.benchmark.result': ('BenchmarkResult', 'compare_results'),
    'hyso.experiments': ('ExperimentManager', 'ExperimentError'),
    'hyso.experiments.manager': ('ExperimentManager', 'ExperimentError'),
    'hyso.performance': ('TailPolicy', 'ThroughputProtocol', 'LatencyAnalysis', 'ThroughputResult',
                         'analyze_latency', 'measure_throughput', 'BatchSearchProtocol', 'CapacityResult',
                         'find_max_batch_size', 'ProfilingProtocol', 'ProfilingError', 'profile_operators',
                         'profile_layers', 'analyze_flops', 'OperatorProfileResult', 'LayerProfileResult',
                         'OperationCountResult'),
    'hyso.performance.inference': ('analyze_latency', 'measure_throughput'),
    'hyso.performance.result': ('LatencyAnalysis', 'ThroughputResult', 'MemoryResult', 'FootprintResult'),
    'hyso.performance.memory': ('MemoryProtocol', 'measure_memory', 'measure_footprint'),
    'hyso.performance.pipeline': ('PipelineSample', 'PipelineStage', 'PipelineProtocol', 'measure_pipeline', 'analyze_stages'),
    'hyso.performance.sweeps': ('SweepProtocol', 'run_sweep'),
    'hyso.performance.capacity': ('BatchSearchProtocol', 'find_max_batch_size'),
    'hyso.performance.profiling': ('ProfilingProtocol', 'profile_operators', 'profile_layers', 'analyze_flops'),
    'hyso.performance._profile_worker': ('main',),
    'hyso.performance._operation_models': ('cost',),
    'hyso.performance.transfers': ('TransferProtocol', 'measure_transfers', 'TransferError'),
    'hyso.performance.utilization': ('UtilizationProtocol', 'measure_utilization'),
    'hyso.performance._telemetry': ('cpu_observe', 'gpu_observe'),
    'hyso.performance._transfer_worker': ('main',),
    'hyso.performance._batch_worker': ('main',),
    'hyso.performance.cold': ('ColdStartRecipe', 'ColdStartProtocol', 'measure_cold_start'),
    'hyso.performance._cold_worker': ('main',),
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
    import csv
    export_prefix = pathlib.Path.cwd() / ('benchmark-' + target.name + '-' + precision)
    json_path, summary_path, samples_path = [export_prefix.with_suffix(suffix) for suffix in ('.json', '.csv', '.samples.csv')]
    bench_result.to_json(json_path)
    bench_result.to_csv(summary_path)
    bench_result.to_samples_csv(samples_path)
    assert json.loads(json_path.read_text(encoding='utf-8')) == bench_data
    with summary_path.open(encoding='utf-8', newline='') as handle:
        summary_rows = list(csv.DictReader(handle))
    with samples_path.open(encoding='utf-8', newline='') as handle:
        sample_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 1 and float(summary_rows[0]['mean_ms']) == bench_data['stats']['pooled']['mean_ms']
    assert summary_rows[0]['cuda_memory_status'] == 'not_applicable' and summary_rows[0]['cuda_peak_allocated_bytes'] == ''
    assert len(sample_rows) == 4 and [float(row['latency_ms']) for row in sample_rows] == sum(bench_data['timing']['latency_samples_ms'], [])
    assert compare_results(bench_result, BenchmarkResult(bench_data))['status'] == 'compatible'
    benchmark_rows.append(bench_data)
assert compare_results(BenchmarkResult(benchmark_rows[0]), BenchmarkResult(benchmark_rows[1]))['status'] == 'incompatible'
import sqlite3, tempfile
tracking_root = pathlib.Path(tempfile.mkdtemp(prefix='hyso-installed-experiment-')) / 'runs'
tracking_manager = ExperimentManager(tracking_root)
tracking_experiment = tracking_manager.create_experiment('installed smoke')
tracking_run = tracking_manager.create_run(tracking_experiment['id'], 'training', {'seed': 42})
tracking_run.start()
tracking_run.log_metric(name='train/loss', value=.42, unit='mean_loss', source='caller', epoch=1)
assert tracking_run.read_metrics()[0]['value'] == .42
tracking_run.finish()
tracking_snapshot = tracking_run.to_dict()
tracking_manager.close()
tracking_reopened = ExperimentManager(tracking_root)
assert tracking_reopened.get_run(tracking_run.id).to_dict() == tracking_snapshot
assert tracking_snapshot['status'] == 'completed' and tracking_snapshot['metric_count'] == 1
assert tracking_reopened._db.execute('PRAGMA user_version').fetchone()[0] == 1
tracking_reopened.close()
migration = ExperimentManager.migrate_store(tracking_root)
assert migration['status'] == 'migrated'
with sqlite3.connect(migration['backup']) as backup:
    assert backup.execute('PRAGMA user_version').fetchone()[0] == 1
tracking_b2 = ExperimentManager(tracking_root)
assert tracking_b2.get_run(tracking_run.id).read_metrics()[0]['value'] == .42
preserved = tracking_b2.get_run(tracking_run.id).to_dict()
assert all(preserved[key] == value for key, value in tracking_snapshot.items())
artifact_run = tracking_b2.create_run(tracking_experiment['id'], 'training', {'attempt': 1})
artifact_run.start()
artifact_source = tracking_root.parent / 'untrusted-checkpoint'
artifact_source.write_bytes(b'arbitrary bytes: never deserialize')
reference = artifact_run.add_artifact(artifact_source, retain_locator=True)
owned = artifact_run.add_artifact(artifact_source, ownership='copy')
assert tracking_b2.verify_artifact(reference['id'])['status'] == 'verified'
assert tracking_b2.verify_artifact(owned['id'])['status'] == 'verified'
assert (tracking_root / owned['managed_relative_path']).read_bytes() == artifact_source.read_bytes()
recovered = tracking_b2.recover_interrupted(artifact_run.id, artifact_run.to_dict()['revision'], 'coordinated stopped writer')
assert recovered['status'] == 'interrupted' and recovered['writer_session_id'] is None
retry_run = tracking_b2.create_run(tracking_experiment['id'], 'training', {'attempt': 2},
    retry_of_run_id=artifact_run.id, resume_artifact_id=owned['id'])
assert retry_run.to_dict()['attempt_no'] == 2
tracking_b2.link_artifact(retry_run.id, reference['id'], role='input')
retry_run.start()
retry_run.log_metric(name='loss',value=.2,unit='loss',source='caller',epoch=1)
retry_run.finish()
assert tracking_b2.reconcile_storage()['status'] == 'reconciled'
retry_id = retry_run.id
tracking_b2.close()
tracking_b2 = ExperimentManager(tracking_root)
assert tracking_b2.get_run(retry_id).to_dict()['status'] == 'completed'
assert tracking_b2.get_artifact(owned['id'])['producer_run_id'] == artifact_run.id
tracking_b2.close()
# Execute the documented manual integration with installed production modules.
import re
fixture_spec = importlib.util.spec_from_file_location('fixture_support', sys.argv[3])
fixture_support = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_support)
manual = {}
guide_path = fixture_support.example_fixture("docs/guides/experiment-manager.md")
for block in re.findall(r'```python\\n(.*?)```', guide_path.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(guide_path), 'exec'), manual)
manual_manager = ExperimentManager(manual['root'])
assert manual_manager.get_run(manual['training_id']).to_dict()['status'] == 'completed'
assert len(manual_manager.get_run(manual['training_id']).read_metrics()) == 8
assert manual_manager.verify_artifact(manual['artifact_id'])['status'] == 'verified'
manual_manager.close()
performance_manual = {}
performance_guide = fixture_support.example_fixture("docs/guides/performance-inference.md")
for block in re.findall(r'```python\\n(.*?)```', performance_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(performance_guide), 'exec'), performance_manual)
assert performance_manual['latency'].to_dict()['measurements'][-1]['status'] == 'derived'
assert performance_manual['throughput'].to_dict()['actual']['completed_images'] == 80
assert isinstance(performance_manual['latency'], LatencyAnalysis)
assert isinstance(performance_manual['throughput'], ThroughputResult)
memory_manual = {}
memory_guide = fixture_support.example_fixture("docs/guides/performance-memory.md")
for block in re.findall(r'```python\\n(.*?)```', memory_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(memory_guide), 'exec'), memory_manual)
assert memory_manual['current'].to_dict()['measurements']['current_rss_bytes']['value'] > 0
assert memory_manual['footprint'].to_dict()['raw_evidence']['artifact']['format_validation'] == 'roundtrip_verified'
pipeline_manual = {}
pipeline_guide = fixture_support.example_fixture("docs/guides/performance-pipeline.md")
for block in re.findall(r'```python\\n(.*?)```', pipeline_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(pipeline_guide), 'exec'), pipeline_manual)
assert pipeline_manual['pipeline'].to_dict()['actual']['completed'] == 4
assert pipeline_manual['cold'].to_dict()['actual']['measured'] == 1
assert all(row['provenance']['torch'] == str(torch.__version__) for row in pipeline_manual['cold'].to_dict()['raw_evidence']['trials'])
sweep_manual = {}
sweep_guide = fixture_support.example_fixture("docs/guides/performance-sweeps.md")
for block in re.findall(r'```python\\n(.*?)```', sweep_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(sweep_guide), 'exec'), sweep_manual)
assert sweep_manual['batches'].to_dict()['actual']['measured'] == 4
assert sweep_manual['resolutions'].to_dict()['actual']['measured'] == 3
assert sweep_manual['precisions'].to_dict()['actual']['measured'] == 4
capacity_manual = {}
capacity_guide = fixture_support.example_fixture("docs/guides/performance-max-batch.md")
for block in re.findall(r'```python\\n(.*?)```', capacity_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(capacity_guide), 'exec'), capacity_manual)
assert capacity_manual['capacity'].to_dict()['measurements']['maximum_proven_within_range'] == 3
assert all(pathlib.Path(row['provenance']['package_source']).resolve().is_relative_to(target)
           for row in capacity_manual['data']['raw_evidence']['candidates'])
profiling_manual = {}
profiling_guide = fixture_support.example_fixture("docs/guides/performance-profiling-v3.md")
for block in re.findall(r'```python\\n(.*?)```', profiling_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(profiling_guide), 'exec'), profiling_manual)
assert profiling_manual['tiny']['total_flops_per_batch'] == 5320
assert profiling_manual['count_data']['protocol']['operation_model_version'] == 3
assert profiling_manual['count_data']['measurements']['status'] == 'complete'
assert pathlib.Path(profiling_manual['operator_data']['provenance']['environment']['package_source']).resolve().is_relative_to(target)
telemetry_manual = {}
telemetry_guide = fixture_support.example_fixture("docs/guides/performance-transfer-utilization.md")
for block in re.findall(r'```python\\n(.*?)```', telemetry_guide.read_text(encoding='utf-8'), re.S):
    exec(compile(block, str(telemetry_guide), 'exec'), telemetry_manual)
assert telemetry_manual['utilization_data']['actual']['gpu_queries'] == 0
assert pathlib.Path(telemetry_manual['utilization_data']['provenance']['environment']['package_source']).resolve().is_relative_to(target)
from hyso.performance import TransferProtocol, TransferResult, TransferError, measure_transfers, UtilizationResult
try:
    measure_transfers(protocol=TransferProtocol(shape=(16,), warmup_iterations=0, measure_iterations=1, repeats=1))
except TransferError as error:
    assert error.diagnostics['status'] == 'unsupported' and error.diagnostics['process_terminated']
else:
    raise AssertionError('CPU-only admission unexpectedly completed GPU transfer')
from hyso.performance import TrainingTimingProtocol, TrainingPerformanceResult, measure_training
timing_model = torch.nn.Linear(3, 2)
timing_trainer = Trainer(timing_model, optimizer=torch.optim.SGD(timing_model.parameters(), lr=.01),
                        loss_fn=torch.nn.CrossEntropyLoss(), accum_steps=2)
timing_batches = list(zip(torch.ones(5, 3).split(2), torch.tensor([0, 1, 0, 1, 0]).split(2)))
timing = measure_training(timing_trainer, timing_batches, epochs=2).to_dict()
assert timing['actual_epochs'] == 2 and timing['counters']['after']['applied'] == 4
assert [r['samples'] for r in timing['observations'] if r['kind'] == 'window'] == [4, 1, 4, 1]
from hyso.performance import measure_training_detailed, analyze_training_throughput, TrainingStageResult, TrainingThroughputResult
detailed_model = torch.nn.Linear(3, 2)
detailed_trainer = Trainer(detailed_model, optimizer=torch.optim.AdamW(detailed_model.parameters(), lr=.01),
                          loss_fn=torch.nn.CrossEntropyLoss(), accum_steps=2)
detailed = measure_training_detailed(detailed_trainer, timing_batches, epochs=2)
assert detailed.throughput.to_dict()['segment']['completed']['samples'] == 10
assert sum(r['stage'] == 'optimizer_execution' and r['status'] == 'measured' for r in detailed.stages.to_dict()['observations']) == 4
assert TrainingStageResult(detailed.stages.to_dict()).to_dict() == detailed.stages.to_dict()
assert TrainingThroughputResult(detailed.throughput.to_dict()).to_dict() == analyze_training_throughput(detailed.timing).to_dict()
from hyso.performance import measure_training_with_loader, DataLoaderAnalysisResult
loader_model = torch.nn.Linear(3, 2)
loader_trainer = Trainer(loader_model, optimizer=torch.optim.SGD(loader_model.parameters(), lr=.01),
                         loss_fn=torch.nn.CrossEntropyLoss(), accum_steps=2)
actual_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])), batch_size=2)
loader_result = measure_training_with_loader(loader_trainer, actual_loader, epochs=1)
assert loader_result.loader.to_dict()['summaries'][0]['samples'] == 5
assert DataLoaderAnalysisResult(loader_result.loader.to_dict()).to_dict() == loader_result.loader.to_dict()
assert loader_result.loader.to_dict()['worker_service']['value'] is None
from hyso.performance import measure_training_memory, TrainingMemoryResult
memory_model = torch.nn.Linear(3, 2)
memory_trainer = Trainer(memory_model, optimizer=torch.optim.AdamW(memory_model.parameters(), lr=.01),
                         loss_fn=torch.nn.CrossEntropyLoss(), accum_steps=2)
training_memory = measure_training_memory(memory_trainer, actual_loader, epochs=1)
assert TrainingMemoryResult(training_memory.to_dict()).to_dict() == training_memory.to_dict()
assert training_memory.to_dict()['observations'][0]['logical']['summary']['optimizer']['logical_bytes'] == 0
assert training_memory.to_dict()['observations'][-1]['logical']['summary']['optimizer']['logical_bytes'] > 0
assert training_memory.to_dict()['activation']['value'] is None
from hyso.performance import TrainingMemoryRecipe, IsolatedTrainingMemoryProtocol, IsolatedTrainingMemoryResult, measure_training_memory_isolated
isolated_recipe = TrainingMemoryRecipe('training_memory_factories:tiny', factory_search_path=pathlib.Path(sys.argv[2]).parent,
                                     seed=71, epochs=2, kwargs={'optimizer': 'adamw'})
isolated = measure_training_memory_isolated(isolated_recipe, protocol=IsolatedTrainingMemoryProtocol(saved_activation_lifetime=True))
isolated_data = isolated.to_dict()
assert isolated_data['outcome'] == 'success', isolated_data
assert pathlib.Path(isolated_data['measurement']['provenance']['package_source']).resolve().is_relative_to(target)
assert isolated_data['measurement']['activation']['summary']['saved_tensors'] > 0
assert isolated_data['measurement']['activation']['summary']['live_payloads_at_end'] == 0
assert isolated_data['receipt']['job']['active_processes'] == 0
assert IsolatedTrainingMemoryResult(isolated_data).to_dict() == isolated_data
from hyso.performance import measure_gradient_diagnostics, GradientDiagnosticsResult
gradient_model = torch.nn.Linear(3, 2)
gradient_trainer = Trainer(gradient_model, optimizer=torch.optim.AdamW(gradient_model.parameters(), lr=.01),
                          loss_fn=torch.nn.CrossEntropyLoss(), accum_steps=2, grad_clip=.1)
gradient_result = measure_gradient_diagnostics(gradient_trainer, actual_loader, epochs=1)
assert GradientDiagnosticsResult(gradient_result.to_dict()).to_dict() == gradient_result.to_dict()
assert [r['samples'] for r in gradient_result.to_dict()['observations']] == [4, 1]
assert gradient_result.to_dict()['summary']['applied'] == 2
assert all(r['postclip']['summary']['global_l2']['value'] < .100001 for r in gradient_result.to_dict()['observations'])
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
                  'benchmark_exports': 6, 'benchmark_comparisons': ['compatible', 'compatible', 'incompatible'],
                  'benchmarks': benchmark_rows,
                  'benchmark_source': importlib.import_module('hyso.benchmark.runner').__file__,
                  'experiment_source': importlib.import_module('hyso.experiments.manager').__file__,
                  'sqlite_version': sqlite3.sqlite_version,
                  'experiment_snapshot': tracking_snapshot,
                  'telemetry_guide': True, 'telemetry_cpu_samples': telemetry_manual['utilization_data']['actual']['cpu_samples'],
                  'profiling_guide': True, 'profiling_tiny_flops': 5320,
                  'capacity_guide': True, 'capacity_proven_max': 3,
                  'performance_guide': True, 'performance_samples': 1000, 'performance_images': 80,
                  'memory_guide': True, 'rss_backend': memory_manual['current'].to_dict()['raw_evidence']['observations'][0]['backend'],
                  'sweep_guide': True, 'sweep_precision_points': 4, 'pipeline_guide': True, 'pipeline_samples': 4, 'cold_trials': pipeline_manual['cold'].to_dict()['actual'],
                  'isolated_activation': True, 'training_timing': True, 'manual_integration': True, 'cpu_models': 3, 'cuda_initialized': torch.cuda.is_initialized()}))
"""


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_installed_distribution_import_and_cpu_forward(distributions, kind):
    work, wheel, sdist = distributions
    target = work / ("installed-" + kind)
    artifact = wheel if kind == "wheel" else sdist
    command(["-m", "pip", "--isolated", "--no-cache-dir", "install", "--no-index",
             "--no-deps", "--no-build-isolation", "--no-compile",
             "--target", str(target), str(artifact)], work)
    observed = json.loads(command(["-c", SMOKE, str(target), str(ROOT / "tests/trainer_replay_support.py"), str(ROOT / "tests/fixture_support.py")], work).strip().splitlines()[-1])
    assert Path(observed["source"]).resolve().is_relative_to(target.resolve())
    assert observed["manual_integration"] is True
    assert observed["training_timing"] is True
    assert observed['isolated_activation'] is True
    assert observed['performance_guide'] and observed['performance_samples'] == 1000 and observed['performance_images'] == 80
    assert observed['memory_guide'] and observed['rss_backend'].startswith('windows.' if os.name == 'nt' else 'linux.')
    assert observed['sweep_guide'] and observed['sweep_precision_points'] == 4
    assert observed['capacity_guide'] and observed['capacity_proven_max'] == 3
    assert observed['profiling_guide'] and observed['profiling_tiny_flops'] == 5320
    assert observed['telemetry_guide'] and observed['telemetry_cpu_samples'] >= 2
    assert observed['pipeline_guide'] and observed['pipeline_samples'] == 4 and observed['cold_trials']['measured'] == 1
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
    assert observed["benchmark_exports"] == 6 and observed["benchmark_comparisons"] == ['compatible', 'compatible', 'incompatible']
    assert Path(observed["benchmark_source"]).resolve().is_relative_to(target.resolve())
    assert Path(observed["experiment_source"]).resolve().is_relative_to(target.resolve())
    assert observed["experiment_snapshot"]["status"] == "completed"
    assert observed["experiment_snapshot"]["metric_count"] == 1
    assert {row["configuration"]["precision"] for row in observed["benchmarks"]} == {"fp32", "fp64"}
    (work / (kind + "-smoke.json")).write_text(json.dumps(observed), encoding="utf-8")
