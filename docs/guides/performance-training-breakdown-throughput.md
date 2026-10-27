# Training stage breakdown ve throughput — P20 / P22

`measure_training_detailed` bir gerçek `Trainer.fit` çağrısından üç ayrı immutable
result üretir. `timing` mevcut P19 schema1, `stages` P20, `throughput` P22'dir.
Temel `measure_training` kullanımında stage recorder veya optimizer hook eklenmez.
Trainer constructor/fit, checkpoint v1 ve Experiment Manager schema'ları değişmez.

```python
import torch
from torch import nn
from hyso.training import Trainer
from hyso.performance import (TrainingStageProtocol, TrainingTimingProtocol,
    measure_training_detailed, analyze_training_throughput)

model = nn.Linear(3, 2)
trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),
    loss_fn=nn.CrossEntropyLoss(), accum_steps=2, grad_clip=.5, ema_decay=.8)
train = list(zip(torch.ones(5, 3).split(2), torch.tensor([0, 1, 0, 1, 0]).split(2)))
result = measure_training_detailed(trainer, train, epochs=2,
    protocol=TrainingTimingProtocol(event_buffer=32, max_records=1000),
    stage_protocol=TrainingStageProtocol(max_records=1000))
timing = result.timing.to_dict()
stages = result.stages.to_dict()
throughput = result.throughput.to_dict()
assert [r['samples'] for r in timing['observations'] if r['kind'] == 'window'] == [4, 1, 4, 1]
assert throughput['segment']['completed']['samples'] == 10
assert throughput['segment']['completed']['microbatches'] == 6
assert throughput['segment']['completed']['attempts'] == 4
assert throughput['segment']['completed']['applied'] == 4
# Saf analiz; ikinci fit, timer veya model çağrısı yoktur.
assert analyze_training_throughput(result.timing).to_dict() == throughput
```

Tam imza:
`measure_training_detailed(trainer, train_loader, val_loader=None, *, epochs=5,
protocol=None, stage_protocol=None, save_last_to=None, resume_from=None)`.
Fresh standart Trainer ve mevcut Trainer device/precision/checkpoint kısıtları
geçerlidir. `TrainingStageProtocol(max_records=100000)` combined P19+P20 kayıt
bütçesidir; tavan1000000. P19'un kendi kayıt limiti de geçerlidir. Limit aşılırsa
explicit `TrainingTimingError` gelir; eksik başarılı result oluşturulmaz ve
ilerlemiş training state geri alınmış sayılmaz.

## P20 stage tablosu

| Stage adı | Gerçek işlem / parent |
|---|---|
| input_preparation | `_batch`: input/target validation/preparation ve transfer birlikte; microbatch. Saf transfer bandwidth ölçümü değildir. |
| forward | `model(inputs)`; microbatch. |
| loss_metrics | `_loss_and_correct`, loss/correctness checks; microbatch. |
| backward | `scaler.scale(loss * size).backward()`; microbatch. |
| unscale | Native `scaler.unscale_`; attempt. AMP disabled ise gerçek çağrı no-op olabilir. |
| gradient_check_normalization | Finite kontrolü ve gerçek window sample sayısına gradient bölümü; attempt. |
| gradient_clipping | Gerçek `clip_grad_norm_`; kapalıysa veya overflow gradient nedeniyle çalışmadıysa null/not_executed. |
| scaler_step_attempt | Gerçek `scaler.step(optimizer)` çağrısı; attempt. |
| optimizer_execution | Sadece native optimizer gerçekten çalışırsa public pre/post hook sınırı; scaler_step_attempt içine nested. |
| scaler_update | Gerçek `scaler.update`; attempt, skipped attempt dahil. |
| ema_update | Mevcut validation+EMA update; yalnız applied update ve EMA enabled. |
| onecycle_scheduler | Gerçek OneCycle step; yalnız applied update ve ilgili scheduler. |
| zero_grad | Epoch başındaki zero_grad veya attempt finally cleanup; scope/parent ID bunları ayırır. |
| epoch_scheduler | StepLR/Cosine/Plateau gerçek step; epoch_total parent, validation sonrasında. |

GradScaler attempt gerçek optimizer execution değildir. Overflow'da attempt ve
scaler_update ölçülür; optimizer_execution, EMA ve OneCycle için duration alanları
`null`, status=`not_executed`, reason açıktır. Sahte sıfır optimizer süresi yoktur.
Public optimizer pre/post hook'ları yalnız detailed capture'da scaler step boyunca
bağlanır; `finally` ile kaldırılır. Optimizer.step monkeypatch edilmez; mevcut
Trainer post-hook ownership kararı değişmez. Kayıtlar model/tensor taşımaz.

## Clock, completion ve overhead

`host_ms` monotonic `perf_counter_ns` başlangıç/bitiş farkı /1e6'dır. CUDA'da
`cuda_event_ms` ayrı selected-current-stream interval'idir. CPU'da device alanı
null'dır. CUDA event interval kernel active time değildir; launch gaps, stream
idle ve instrumentation etkisi içerebilir. CUPTI kernel attribution unsupported.
Default/non-default selected stream doğrulanır; gizli stream ve concurrent process
attribution yoktur. Exact CUDA replay mevcut deterministik/CUBLAS/TF32 fixture
koşullarıyla sınırlıdır; recorder global determinism ayarı değiştirmez.

P19 ve P20 aynı bounded pending event kuyruğunu paylaşır: P19 `event_buffer`
toplam kapalı pair sınırıdır, ayrıca en fazla dört açık scope bulunur. Her stage
device-wide sync yoktur. Buffer dolunca son kapalı event beklenir; yalnız ready
event'lerin elapsed değeri okunur. Training terminal ve epoch scheduler event'leri
için ayrı drain uygulanır. Exception'da recorder, event referansları ve geçici
optimizer hook'ları temizlenir; `_update` zero_grad cleanup korunur.

Süreler inclusive'dir. Özellikle optimizer_execution, scaler_step_attempt içinde;
ikisi microbatch/window/attempt/epoch süreleriyle toplanamaz. Host ve CUDA clock
değerleri de birbirine eklenmez. Summary yalnız aynı stage'in host sürelerini
biriktirir; stage yüzdesi veya cross-stage toplamı üretmez. Stage toplamının epoch
wall'a eşit olması beklenmez: iterator wait, mode setup, kontroller, validation,
callback/ES ve observer/drain overhead kapsamları farklıdır. Ayrı ölçülmeyen
instrumentation overhead bilinmiyor kabul edilir; sıfır diye doldurulmaz.

## P22 denominator ve ledger

Her tamamlanan epoch ve yeni segment için samples/microbatches/windows/attempts/
applied/skipped sayıları vardır. Rates: samples/s, microbatches/s, attempts/s,
applied updates/s. Payda tamamlanmış epoch'ların **instrumented training wall**
süreleridir; validation hariçtir. Segment hızı epoch rate ortalaması değil,
`sum(completed_count) / sum(training_wall_seconds)` değeridir. Son kısa batch ve
accumulation tail gerçek sample sayısıyla hesaplanır. Skipped attempt applied
update hızına eklenmez. Warmup gerçek epoch'ları etiketler; ek eğitim üretmez ve
default segment hesabından silmez. End-to-end epoch throughput burada üretilmez.

`analyze_training_throughput(TrainingPerformanceResult)` P19'un raw sample/window/
attempt/counter ledger'ını tekrar doğrular. Model/fit/timer çalıştırmaz. Resume
prefix süreleri uydurulmaz; completed checkpoint boş segment ve null/unavailable
rate üretir. EarlyStopping yalnız tamamlanan epoch'ları sayar. Sıfır duration veya
nonfinite rate için infinity/NaN yerine null/status/reason vardır.

## JSON/CSV ve Experiment Manager

P20/P22 schema1 bağımsızdır; eski P19 JSON'u `TrainingPerformanceResult(data)` ile
okunmaya devam eder. Yeni result'lar source segment ID, canonical source SHA256 ve
embedded tam P19 evidence taşır; association/provenance/raw/derived consistency
deserialize ve export sırasında doğrulanır. `to_dict` defensive copy döndürür.
JSON tam canonical kayıttır. P20 CSV raw stage observation, P22 CSV epoch summary
ve son satırda segment aggregate içerir; boş epoch_id segment satırıdır. Export
atomic/no-clobber; `overwrite=True` açık seçenektir.

```python
import hashlib
from pathlib import Path
import tempfile
from hyso.experiments import ExperimentManager

# Fit bitti. Tracking hatası bu ölçümü yeniden çalıştırmaz.
with tempfile.TemporaryDirectory(prefix='hyso-b9-') as directory:
    root = Path(directory)
    manager = ExperimentManager(root / 'store')
    manager.close()
    ExperimentManager.migrate_store(root / 'store')  # mevcut artifact store akışı
    manager = ExperimentManager(root / 'store')
    run = manager.create_run(manager.create_experiment('training-B9')['id'], 'training', {})
    run.start()
    for name, component in [('P20', result.stages), ('P22', result.throughput)]:
        path = root / (name + '.json')
        component.to_json(path)
        component.to_csv(root / (name + '.csv'))
        artifact = run.add_artifact(path, kind='report', ownership='copy',
            metadata={'source_segment_id': timing['segment_id']})
        assert artifact['expected_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
        csv_path = root / (name + '.csv')
        csv_artifact = run.add_artifact(csv_path, kind='report', ownership='copy',
            metadata={'source_segment_id': timing['segment_id']})
        assert csv_artifact['expected_sha256'] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert manager.verify_artifact(csv_artifact['id'])['status'] == 'verified'
    for epoch in throughput['epochs']:
        metric = epoch['rates']['samples_per_second']
        if metric['value'] is not None:
            run.log_metric(name='training.samples', value=metric['value'], unit=metric['unit'],
                           source='P22', epoch=epoch['epoch_id'])
    for row in stages['summary']:
        if row['host_ms'] is not None:
            run.log_metric(name='training.stage.' + row['stage'], value=row['host_ms'],
                           unit='ms', source='P20', step=0, step_kind='observation')
    run.finish()
    identity = run.id
    manager.close()
    manager = ExperimentManager(root / 'store')
    assert len(manager.get_run(identity).read_metrics()) >= len(throughput['epochs'])
    manager.close()
```

Yeni tablo/migration/run kind yoktur. Unsupported/null duration metriklerini
Manager'a sıfır olarak yazmayın. Ölçüm recorder'ı DB/dosya yazmaz; caller callbacks
ve opt-in checkpoint I/O mevcut Trainer davranışıdır. Ölçüm tamamlandıktan sonra
tracking başarısız olursa eldeki result tekrar export/track edilir; fit tekrarlanmaz.

Üç tekrar normal/P19/detailed full-fit dağılımları, native ve packaging kanıtları:
[B9 completion audit](../audit/m2-p-b9-completion-p20-p22.md). B4 reliability riski
açık; bu özellikler B6 kernel profiling, B7 overlap veya P18 counter gate'ini kapatmaz.
