# Experiment Manager: manuel Trainer ve Benchmark kullanımı

Canlı SQLite store için caller'a ait, yerel, senkronize olmayan dedicated bir
dizin seçin. OneDrive veya network filesystem kullanmayın. Aşağıdaki örnekler
yalnız public API kullanır; Trainer/Runner içine callback veya tracking eklemez.
Kod blokları aynı Python oturumunda kullanılabilir. Dosyalar caller'ın seçtiği
yerel çalışma dizininde kalır; manager reference dosyalarını silmez.

## 1. Küçük, tekrar kullanılabilir kayıt fonksiyonları

Bu fonksiyonlar uygulama örneğidir; HYSo public API'sine eklenmez.
History kayıtları fit sonrasında en çok256 event içeren atomic batch'lerle yazılır.
Accuracy yüzde kalır; None validation event oluşturmaz. LR/duration üretilmez.
`first_new_epoch` restored prefix'i dışarıda bırakır; bilinmiyorsa resume history'sini
loglamayın. Her batch atomic'tir; bütün history tek transaction değildir.

```python
import json
from pathlib import Path

METRICS = {
    "train_loss": ("train/loss", "mean_loss"),
    "train_acc": ("train/accuracy", "percent"),
    "val_loss": ("validation/loss", "mean_loss"),
    "val_acc": ("validation/accuracy", "percent"),
}


def log_history(run, history, *, first_new_epoch=1):
    if type(first_new_epoch) is not int or first_new_epoch < 1:
        raise ValueError("first_new_epoch must be a positive integer")
    batch = []
    for index in range(first_new_epoch - 1, len(history["train_loss"])):
        for key, (name, unit) in METRICS.items():
            value = history[key][index]
            if value is not None:
                batch.append(dict(name=name, value=value, unit=unit,
                                  source="trainer_history", epoch=index + 1))
            if len(batch) == 256:
                run.log_metrics(batch)
                batch = []
    if batch:
        run.log_metrics(batch)


def train_and_record(run, trainer, train, val=None, *, epochs,
                     checkpoint_path, resume_from=None, first_new_epoch=1,
                     ownership="copy"):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise ValueError("choose a fresh checkpoint output path")
    # All plans/components/loaders belong to the caller. Checkpoint profile
    # requires callbacks=() and the original total epoch budget on resume.
    run.start()
    try:
        history = trainer.fit(train, val, epochs=epochs,
                              save_last_to=checkpoint_path, resume_from=resume_from)
    except BaseException as workload_error:
        # Secondary tracking errors cannot replace the original workload error.
        # A completed-epoch observation is not proof that its checkpoint saved.
        try:
            log_history(run, trainer.history, first_new_epoch=first_new_epoch)
        except Exception:
            pass
        if checkpoint_path.is_file():
            try:
                run.add_artifact(checkpoint_path, kind="checkpoint",
                                 ownership=ownership, role="output")
            except Exception:
                pass
        try:
            if isinstance(workload_error, KeyboardInterrupt):
                run.interrupt("caller_keyboard_interrupt")
            else:
                run.fail("training_exception", failure_domain="workload",
                         workload_outcome="failed")
        except Exception:
            pass
        raise
    try:
        log_history(run, history, first_new_epoch=first_new_epoch)
    except Exception:
        try:
            run.fail("history_logging", failure_domain="tracking",
                     workload_outcome="succeeded")
        except Exception:
            pass
        raise
    try:
        # Terminal/no-op resume may produce no new checkpoint file.
        artifact = (run.add_artifact(checkpoint_path, kind="checkpoint",
                    ownership=ownership, role="output")
                    if checkpoint_path.is_file() else None)
    except Exception:
        try:
            run.fail("checkpoint_ingest", failure_domain="artifact",
                     workload_outcome="succeeded")
        except Exception:
            pass
        raise
    try:
        no_new_epochs = len(history["train_loss"]) < first_new_epoch
        run.finish(reason="resume_no_new_epochs" if no_new_epochs else "completed")
    except Exception:
        try:
            run.fail("tracking_finalize", failure_domain="tracking",
                     workload_outcome="succeeded")
        except Exception:
            pass
        raise
    return history, artifact


def record_benchmark(run, runner, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    run.start()
    # Measurement returns before any export/artifact I/O starts.
    result = runner.run()
    data = result.to_dict()
    association = dict(benchmark_schema_version=data["schema_version"],
                       benchmark_run_id=data["identity"]["run_id"],
                       experiment_run_id=run.id, assertion_source="caller")
    artifacts = []
    for filename, kind, export in (
            ("result.json", "benchmark_json", result.to_json),
            ("summary.csv", "benchmark_summary", result.to_csv),
            ("samples.csv", "benchmark_samples", result.to_samples_csv)):
        path = directory / filename
        export(path)
        artifacts.append(run.add_artifact(path, kind=kind, ownership="copy",
                                          role="output", metadata=association))
    run.finish()
    return result, artifacts


def record_comparison(run, left, right, path, artifact_ids):
    from hyso.benchmark import compare_results
    report = compare_results(left, right)
    # compare_results returns validated finite plain data. No default=str or pickle.
    encoded = json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    run.start()
    artifact = run.add_artifact(path, kind="comparison_json", ownership="copy",
                               role="report", metadata={"input_artifact_ids": artifact_ids,
                               "assertion_source": "caller"})
    run.finish()
    return report, artifact
```

Benchmark/export/comparison hatasında örnek completed üretmez;
caller `run.fail(..., failure_domain="tracking" veya "artifact",
workload_outcome="succeeded")` ile ölçüm başarısı ile kayıt hatasını ayırmalıdır.
Storage hatası bu kaydı da engellerse running kayıt kalabilir. Workload exception
aktifken terminal kayıt hatasını yakalayıp asıl exception'ı yeniden yükseltin.

## 2. Create/open, explicit migration ve gerçek eğitim

Yeni store schema1 başlar. Migration offline ve explicit'tir; bütün writer'ları
koordine edin. Aşağıdaki örnek yerel temporary dizin kullanır. Gerçek uygulamada
kalıcı, senkronize olmayan local path verin. ModularCNN'in yedi constructor alanı,
bounded TensorDataset/num_workers=0 ve callbacks=() checkpoint sözleşmesidir.

```python
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from hyso.experiments import ExperimentManager
from hyso.models.modular import ModularCNN
from hyso.training import Trainer
from hyso.benchmark import BenchmarkRunner

work = Path(tempfile.mkdtemp(prefix="hyso-manual-"))
root = work / "runs"
manager = ExperimentManager(root)
manager.close()
ExperimentManager.migrate_store(root)
manager = ExperimentManager(root)
experiment = manager.create_experiment("manual integration")
config = dict(input_channels=1, num_classes=2, widths=[3], repeats=[1],
              strides=[1], attention="eca", norm="batch")
inputs = torch.linspace(-1, 1, 4 * 25).reshape(4, 1, 5, 5)
targets = torch.tensor([0, 1, 0, 1])
train = DataLoader(TensorDataset(inputs, targets), batch_size=2, num_workers=0)
val = DataLoader(TensorDataset(inputs, targets), batch_size=2, num_workers=0)
model = ModularCNN(**config)
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
trainer = Trainer(model, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(),
                  model_config=config, callbacks=())
training = manager.create_run(experiment["id"], "training", config)
history, checkpoint_artifact = train_and_record(
    training, trainer, train, val, epochs=2, checkpoint_path=work / "last.pth")

# Reference alternative: add it before terminal finish in your own workflow.
# reference = running_run.add_artifact(path, ownership="reference", role="output")
# manager.verify_artifact(reference["id"], path=path)
# Locator is redacted by default; retain_locator=True opts into path persistence.
assert manager.verify_artifact(checkpoint_artifact["id"])["status"] == "verified"
```

## 3. Failed/interrupted parent, retry ve gerçek resume

Parent failed/interrupted olmalıdır; completed parent retry edilemez. Eski workload'u
önce durdurun; stale running kayıt için `recover_interrupted(id, expected_revision,
reason)` explicit çağrısı kullanın. Recovery process/GPU işini durdurmaz.

Aşağıdaki fonksiyon caller'ın kendi ürettiği, SHA doğrulanmış checkpoint'i alır.
Public checkpoint loader disposable inspector bileşenlerine metadata/state yükler;
Manager deserialize etmez. Gerçek strict resume ayrı fresh Trainer.fit çağrısıdır.
`make_trainer` aynı model/optimizer/scheduler/scaler/EMA/ES planını üretmelidir.
Checkpoint metadata'sı güvenle okunamıyorsa epoch sınırını tahmin etmeyin.

```python
from hyso.checkpoint import load_training_checkpoint


def retry_from_checkpoint(manager, parent_id, artifact_id, checkpoint_path,
                          make_trainer, train, val, *, epochs, output_path):
    if manager.verify_artifact(artifact_id, path=checkpoint_path)["status"] != "verified":
        raise ValueError("checkpoint identity is unavailable")
    inspector = make_trainer()
    metadata = load_training_checkpoint(checkpoint_path, inspector.model,
        optimizer=inspector.optimizer, scheduler=inspector.scheduler,
        scaler=inspector.scaler, ema=inspector.ema, restore_rng=False)
    completed = metadata["completed_epochs"]
    parent = manager.get_run(parent_id).to_dict()
    retry = manager.create_run(parent["experiment_id"], parent["kind"], parent["config"],
        provenance={"resume_completed_epochs": completed},
        retry_of_run_id=parent_id, resume_artifact_id=artifact_id)
    fresh = make_trainer()
    history, artifact = train_and_record(retry, fresh, train, val, epochs=epochs,
        checkpoint_path=output_path, resume_from=checkpoint_path,
        first_new_epoch=completed + 1)
    assert all(history[key][:completed] == values
               for key, values in metadata["history"].items())
    return retry, fresh, history, artifact
```

`epochs` original total budget'tır; “kalan epoch” değildir. Completed veya terminal
checkpoint resume'u yeni update/metric üretmeyebilir; `resume_no_new_epochs` reason
bunu açıkça belirtir. Input artifact producer'ı eski Run olarak kalır. Callback
ekleyerek checkpoint profile'ını gevşetmeyin. Hash eşliği strict model/optimizer/
epoch/CUDA RNG uyumu yerine geçmez; bunları mevcut Trainer doğrular.

## 4. Benchmark export, artifact ve comparison

```python
benchmark = manager.create_run(experiment["id"], "benchmark", {"precision": "fp32"})
runner = BenchmarkRunner(model, input_shape=(2, 1, 5, 5), device="cpu",
                         warmup_iterations=1, measure_iterations=2, repeats=2)
left, left_artifacts = record_benchmark(benchmark, runner, work / "benchmark-a")
other = manager.create_run(experiment["id"], "benchmark", {"precision": "fp32"})
right, right_artifacts = record_benchmark(other, runner, work / "benchmark-b")
comparison = manager.create_run(experiment["id"], "evaluation", {"purpose": "comparison"})
input_ids = [left_artifacts[0]["id"], right_artifacts[0]["id"]]
for artifact_id in input_ids:
    manager.link_artifact(comparison.id, artifact_id, role="input")
report, report_artifact = record_comparison(comparison, left, right,
                                           work / "comparison.json", input_ids)
```

Benchmark identity UUID ile Experiment Run UUID farklıdır; metadata ikisini
ilişkilendirir. Raw latency samples metric_events'e çoğaltılmaz. Schema1 export
Result döndükten sonra yapılır; GPU timing/memory probe içine I/O eklenmez.
Public `from_json/load_json` API'si yoktur; burada live Result'lar karşılaştırılır.
CPU `fp64` için model/input dtype'ını float64 yapıp precision="fp64" seçin.

`compatible`, `incompatible`, `insufficient_metadata` timing koşullarını anlatır.
Architecture farkı tek başına incompatibility değildir. Timing ile memory
comparability ayrıdır; CPU memory `not_applicable`, GPU disabled probe
`not_comparable`, ölçülmüş fakat fresh-process isolation doğrulanmamış memory
`insufficient_metadata` olabilir. Baseline/cache/process scope model-only fiziksel
VRAM veya winner/speedup/significance/Pareto garantisi vermez.

## 5. Close/reopen ve inceleme

```python
training_id, artifact_id = training.id, checkpoint_artifact["id"]
manager.close()
manager = ExperimentManager(root)
print(manager.list_runs(experiment_id=experiment["id"]))
print(manager.get_run(training_id).read_metrics())
print(manager.get_artifact(artifact_id))
assert manager.verify_artifact(artifact_id)["status"] == "verified"
manager.close()
```

Run/query ve metric pagination için public cursor/limit kullanın. Artifact expected
SHA/size immutable'dır; verification observation'ları eklenir. SQLite ve filesystem
ortak atomic transaction değildir. Commit belirsizliğini başarı saymayın; explicit
request ID/read-back kullanın. Disk kapalıysa başarılı training bile running
kalabilir; reopen otomatik completion yapmaz. Reconciliation offline, explicit ve
bounded'dır; external reference/owned file otomatik silinmez. Tam power-loss veya
hostile filesystem sandbox garantisi yoktur.
