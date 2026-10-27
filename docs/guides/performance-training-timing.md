# Training timing — P19

P20 stage breakdown ve P22 saf throughput analizi ayrı additive API'lerle
sunulur: [breakdown/throughput rehberi](performance-training-breakdown-throughput.md).
`measure_training` ve geçmiş P19 schema1 sözleşmesi korunur; tek fit'te üç result
için `measure_training_detailed`, mevcut P19 evidence için `analyze_training_throughput`.

`hyso.performance.measure_training` caller-owned, fresh bir `Trainer` üzerinde
gerçek `fit` çağrısını bir kez çalıştırır ve `TrainingPerformanceResult` döndürür.
History aynı Trainer'da kalır. Model, optimizer ve RNG gerçek eğitimin yaptığı
şekilde ilerler; eski state'e restore veya otomatik replay yapılmaz.
Trainer constructor/fit imzaları ve callback listesi değişmez.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import TrainingTimingProtocol, measure_training

model = nn.Linear(3, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
trainer = Trainer(model, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), accum_steps=2)
train = DataLoader(TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])), batch_size=2)
result = measure_training(trainer, train, epochs=2,
    protocol=TrainingTimingProtocol(max_records=1000, event_buffer=32, warmup_epochs=0))
data = result.to_dict()
assert [e['samples'] for e in data['epochs']] == [5, 5]
assert [e['microbatches'] for e in data['epochs']] == [3, 3]
assert [r['samples'] for r in data['observations'] if r['kind'] == 'window'] == [4, 1, 4, 1]
```

API:
`measure_training(trainer, train_loader, val_loader=None, *, epochs=5,
protocol=None, save_last_to=None, resume_from=None)`.
Standart Trainer gerektirir; önceden başlamış veya recorder bağlı Trainer reddedilir.
Mevcut Trainer device/precision/loader/checkpoint kısıtları aynen geçerlidir.

## Sınırlar ve birimler

| Kayıt | Scope |
|---|---|
| Epoch training_s | Model mode setup → loader iteration/transfer → forward/loss/check/backward → bütün update window'ları → training metrics → completion drain; saniye. |
| Epoch validation_s | Varsa `_validate` ve completion; yoksa null; saniye. EMA aktifse mevcut evaluation_model kullanılır. |
| Epoch total_s | Training başlangıcı → validation → epoch scheduler/ES/history/epoch_end callback bitişi; saniye. |
| Microbatch host_ms | Batch yield sonrası `_batch` başlangıcı → sample-weighted backward bitişi; optimizer attempt hariç; milisaniye. |
| Window host_ms | İlk microbatch başlangıcı → `_update` bitişi; aradaki loader beklemeleri ve tüm microbatch'ler dahil. |
| Attempt host_ms | `_update` giriş/çıkış sınırı; normalization/clip/scaler/EMA/OneCycle/zero_grad dahil. Applied step stage decomposition değildir. |

CPU clock `time.perf_counter_ns`; ham timestamp'lar ns, ms için /1e6 ve s için
/1e9. Epoch duration microbatch latency yerine kullanılmaz. Scopes iç içedir;
window/attempt/microbatch süreleri toplanarak epoch süresi üretilmez.
Checkpoint disk I/O, checkpoint restore, train_begin/train_end callback'leri,
best-weight finalization, JSON/CSV export ve Experiment Manager tracking epoch
ölçümünün dışındadır. Checkpoint I/O sonraki epoch başlangıcından da önce kalır.
Caller callback/loader/model'in kendi I/O maliyeti ilgili scope içinde kalabilir.
Forward/backward/optimizer stage decomposition **P20'ye aittir ve burada yoktur**.

CUDA'da `host_ms` asenkron dispatch/check/sync içeren host wall aralığıdır;
`cuda_event_ms` girişte seçilen current stream'in ayrı event aralığıdır. Aynı
sütunda birleştirilmez. GPU event interval host launch gaps/stream idle zamanı
içerebilir; saf kernel active time veya CUPTI attribution değildir.
Default ve seçilmiş non-default stream desteklenir; sınırda stream değişimi
reddedilir. Gizli stream ve concurrent process attribution desteklenmez.
Bit düzeyinde CUDA replay kanıtı mevcut native replay gate'in deterministik
algoritma/CUBLAS/TF32 koşullarıyla sınırlıdır. Recorder bu ayarları değiştirmez;
deterministik olmayan CUDA algoritmalarında normal eğitimin tekrarı da farklı
byte'lar üretebilir. Ayrıntılı kontrol ve başarısız ilk fixture kaydı audit'tedir.

`event_buffer` tamamlanmış event pair kuyruğunu sınırlar (1–4096). Ayrıca en fazla
üç açık scope pair vardır. Buffer dolunca son kapalı event beklenir; tüm interval
sonuçları yalnız completion/query sonrası okunur. Epoch training bitişinde terminal
drain ve validation bitişinde selected-stream completion uygulanır. Her microbatch
için device-wide synchronize yoktur; küçük buffer chunk drain overhead artırır.
Zamanlar instrumentation/event allocation/drain maliyetini içerir; bare baseline
latency veya perturbation-free ölçüm diye sunulmaz.

`max_records` segmentte epoch+microbatch+window+attempt kayıtlarının sert sınırıdır
(default100000, en fazla1000000). RAM'de bütün geçmişi sınırsız tutma/spool yoktur.
Limit aşımında `TrainingTimingError` gelir; eksik başarılı result üretilmez.
Eğitim ilerlemiş olabilir; hata rollback veya yeniden eğitim izni değildir.
Recorder bağlantısı ve event referansları hata halinde de temizlenir.
Uzun koşular için yeterli bütçe önceden seçilmelidir; disk chunk streaming bu
sürümde yoktur. 3 microbatch/2 window örneği epoch başına 8 kayıt tutar.

Warmup ek eğitim üretmez: `warmup_epochs` gerçek segmentin ilk epoch'larını
etiketler, ledger'dan silmez. Validation ayrı ve total epoch içinde tutulur.
Attempt, applied ve skipped sayıları mevcut Trainer counter kararıyla uyumludur;
AMP overflow attempt sayılır, applied update sayılmaz. Optimizer ownership,
scaler, EMA ve scheduler recorder tarafından sürülmez.

## Export ve Experiment Manager

Result JSON-backed immutable'dır; `to_dict()` bağımsız kopya döndürür. JSON bütün
epoch/provenance/ledger bilgisini, CSV ham observation satırlarını taşır. İkisi de
atomic/no-clobber export kullanır; `overwrite=True` açıkça seçilebilir.
Export'tan önce evidence aritmetiği/counter/clock consistency yeniden doğrulanır.

```python
import hashlib
import tempfile
from pathlib import Path
from hyso.experiments import ExperimentManager

# measure_training tamamlandı; bundan sonra tracking hatası fit'i tekrarlamaz.
with tempfile.TemporaryDirectory(prefix='hyso-p19-') as directory:
    root = Path(directory)
    result.to_json(root / 'timing.json')
    result.to_csv(root / 'timing.csv')
    manager = ExperimentManager(root / 'store')
    manager.close()
    ExperimentManager.migrate_store(root / 'store')  # mevcut artifact store akışı
    manager = ExperimentManager(root / 'store')
    run = manager.create_run(manager.create_experiment('P19')['id'], 'training', {})
    run.start()
    artifact = run.add_artifact(root / 'timing.json', kind='report', ownership='copy')
    assert artifact['expected_sha256'] == hashlib.sha256((root / 'timing.json').read_bytes()).hexdigest()
    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
    for epoch in data['epochs']:
        run.log_metric(name='training.epoch.wall', value=epoch['training_s'],
                       unit='s', source='P19', epoch=epoch['epoch_id'])
    run.finish()
    identity = run.id
    manager.close()
    manager = ExperimentManager(root / 'store')
    assert len(manager.get_run(identity).read_metrics()) == data['actual_epochs']
    manager.close()
```

Yeni run kind/tablo/schema yoktur. Tracking başarısız olursa eldeki result yeniden
export/track edilir; training yeniden çalıştırılmaz. `resume_from` mevcut v1
checkpoint sözleşmesini kullanır. Her ölçüm yeni `segment_id` üretir; yalnız yeni
epoch'ların timings'i vardır. Checkpoint prefix/history için sahte süre üretilmez;
completed checkpoint `no_new_epochs` ve boş observations döndürür. Önceki segment
artifact'ları doğrulanmadan cumulative active time iddia edilmez.

B4 native reliability riski açık kalır. Destekli native senaryo kanıtları ve
test kapsamı için [B9 audit](../audit/m2-p-b9-training-timing.md); counter/kernel
sınırı için [B8 gate](../audit/m2-p-b8-roofline-capability-gate.md).
