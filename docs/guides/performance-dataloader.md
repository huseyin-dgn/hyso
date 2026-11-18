# DataLoader bottleneck analysis — P21

P21 gerçek training sırasında main-process iterator initialization ve `next()`
bekleme aralıklarını ölçer. `measure_training_with_loader` tek bir gerçek fit'ten
P19 timing, P21 loader evidence, P22 throughput ve varsayılan olarak P20 stages
üretir. Dataset, sampler, loader ayarları ve worker ownership değiştirilmez.

**Kapsam partial:** worker-side decode/transform/service trace uygulanmadı.
Main-process wait worker service süresi değildir. Queue occupancy ve prefetch
efficiency ölçülmez; alanlar null/unsupported kalır. Prefetch ile kısa `next()`
süresi worker'ın boşta kaldığını veya verimli çalıştığını kanıtlamaz.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import DataLoaderAnalysisProtocol, measure_training_with_loader

dataset = TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0]))
train = DataLoader(dataset, batch_size=2, shuffle=True,
                   generator=torch.Generator().manual_seed(10), num_workers=0)
model = nn.Linear(3, 2)
trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),
                  loss_fn=nn.CrossEntropyLoss(), accum_steps=2)
result = measure_training_with_loader(trainer, train, epochs=2,
    loader_protocol=DataLoaderAnalysisProtocol(max_records=1000))
data = result.loader.to_dict()
assert [s['samples'] for s in data['summaries']] == [5, 5]
assert [s['batches'] for s in data['summaries']] == [3, 3]
assert [r['samples'] for r in data['observations'] if r['kind'] == 'batch'] == [2, 2, 1, 2, 2, 1]
assert result.throughput.to_dict()['segment']['completed']['samples'] == 10
assert data['worker_service']['value'] is None
```

API:
`measure_training_with_loader(trainer, train_loader, val_loader=None, *, epochs=5,
protocol=None, loader_protocol=None, stages=True, stage_protocol=None,
save_last_to=None, resume_from=None)`.

Standart mevcut `DataLoader` gerektirir; custom DataLoader subclass/list loader
admission kapsamı dışıdır. Standart DataLoader içindeki TensorDataset, ImageFolder
ve custom Dataset çalışabilir; native doğrulanan kapsam audit'te listelidir.
`protocol` mevcut TrainingTimingProtocol, `stage_protocol` TrainingStageProtocol;
`loader_protocol` DataLoaderAnalysisProtocol'dür. Fresh standart Trainer gerekir.
Mevcut `measure_training` ve `measure_training_detailed` observer oluşturmaz;
imzaları ve geçmiş JSON schema'ları değişmez.

`stages=False` seçilirse P20 result `None`; prep/compute stage_links açıkça
null/unsupported/P20_not_requested olur. P19 microbatch süresinden forward,
backward veya loader wait tahmini yapılmaz. Ayrı bir analiz API'si bulunmayan
P21 raw wait verisini geçmiş P19 sonuçlarından uydurmaz.

## Ölçüm scope'ları ve ID ilişkisi

| Alan | Kapsam / birim |
|---|---|
| iterator_init | Gerçek `iter(loader)` çağrısının main host wall aralığı; async worker startup'ın tamamı olmayabilir. |
| batch | Gerçek underlying `next(iterator)` giriş→dönüş; host ms. `delivery_ns=end_ns`. |
| exhaustion | Normal for loop'un son `next()` / StopIteration aralığı; batch/sample üretilmez. |
| samples | `_batch` tarafından doğrulanan gerçek batch dimension; son eksik batch dahil. |
| stage_links.input_preparation | Aynı microbatch ID'nin P20 `_batch` validation/preparation+transfer aralığı. |
| stage_links.compute | Aynı ID'nin ayrı forward/loss_metrics/backward P20 aralıkları; toplam üretilmez. |
| window_id / attempt_id | Mevcut P19 accumulation/update ledger kimlikleri; yeni çelişen training ID sistemi yok. |

Training microbatch IDs global Trainer sayaçlarını, window IDs P19 segmentini
izler. Validation batch'leri training microbatch değildir: phase=validation,
epoch içi batch_index vardır, training IDs null'dır. Validation observer yalnız
varsa `_validate` loop'unda çalışır ve validation süreleri training ratio'ya girmez.
Batch teslimi → P19 microbatch başlangıcı → P20 preparation → forward/backward
sınırları tek host clock domain'inde monotonic olarak doğrulanır.

CPU host clock `perf_counter_ns`, raw ns ve /1e6 host_ms. Worker timestamp veya
worker-main clock mapping yoktur. CUDA event clock yalnız linked P19/P20 selected
stream aralıklarında bulunur; host wait'e eklenmez ve kernel active time değildir.
`_batch` CPU checks ve transferi birlikte içerir; saf CUDA H2D ayrı ölçülmediği
için h2d_transfer null/unsupported (CPU'da not_executed) kalır. Kernel attribution,
worker-specific GPU busy, hidden-stream attribution üretilmez.

Her epoch/phase için delivered wait mean/P50/P95 ve P99 vardır. Quantile yöntemi
linear `(n-1)*q`; P99 için en az1000 delivered batch gerekir, aksi halde
null/insufficient_samples. Segmentteki epoch quantile'ları ortalanmaz.
`training_wall_wait_fraction`: training fazındaki bütün `next()` aralıklarının
(delivered + terminal exhaustion) toplamı / P19 training wall. Iterator init bu
payın dışında, ayrı iterator_init_ms alanındadır. Validation fraction unavailable.
P19 training wall instrumentation overhead içerir; ratio nedensel bottleneck veya
worker utilization teşhisi değildir. Stage/nested/host-device süreler toplanmaz.

## Worker yaşam döngüsü ve Windows spawn

Her epoch/fazda underlying `iter(loader)` tam bir kez; her batch ve normal terminal
exhaustion için gerçek underlying `next()` çağrılır. Dataset örneği ek okunmaz,
loader yeniden kurulmaz, sampler/generator yeniden seed edilmez. Observer batch
tensorlarını saklamaz. Exhaustion/exception sonunda iterator/loader referanslarını
bırakır; `close()` kalan observer referanslarını da temizler. Production kodu
`_shutdown_workers`, worker kill veya persistent reset çağırmaz.

Persistent workers caller-owned DataLoader'da yaşamaya devam eder; bu normal
PyTorch ownership'tir. Caller loader'ı tutarken worker'ların canlı olması leak
sayılmaz. Loader bırakıldığında teardown native PyTorch'a aittir. Mid-epoch hata
failed Trainer'ı retry edilebilir yapmaz; native loader timeout ayarı korunur.

Windows multiworker kodunu bir script içinde main guard ile çalıştırın:

```python
def spawn_example():
    loader = DataLoader(dataset, batch_size=2, num_workers=2,
        multiprocessing_context='spawn', prefetch_factor=2, persistent_workers=True,
        generator=torch.Generator().manual_seed(10))
    model = nn.Linear(3, 2)
    trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                      loss_fn=nn.CrossEntropyLoss())
    return measure_training_with_loader(trainer, loader, epochs=2)

if __name__ == '__main__':
    spawned_result = spawn_example()
```

Worker transform/collate/initialization fonksiyonları spawn-picklable olmalıdır.
P21 bunları değiştirmez. Runtime provenance num_workers, pin_memory/device,
prefetch_factor, persistent_workers, timeout, batch size, sampler/dataset/collate/
worker-init tip/adları, multiprocessing context ve generator presence içerir.

Strict checkpoint/resume mevcut num_workers=0 TensorDataset/ModularCNN profiliyle
sınırlıdır. P21 multiworker strict resume eklemez. Resume yeni segmentin yalnız
yeni epoch'larını ölçer; prefix duration oluşturmaz. Completed checkpoint boş
observations; P22 rate null. EarlyStopping yalnız tamamlanan epoch'ları tutar.

## Bounds, export ve Experiment Manager

`max_records` P21 iterator_init + batch + exhaustion kayıtlarının sert sınırıdır
(default100000, tavan1000000). P19/P20 raw/event bütçeleri ayrıca geçerlidir.
3 batch'li bir train epoch5 kayıt; validation varsa faz başına ayrıca5 kayıt.
Limitte kayıtlar sessizce atılmaz: explicit TrainingTimingError ve cleanup;
ilerleyen training state rollback edilmiş sayılmaz. Worker IPC/queue/trace buffer
eklenmedi. Mevcut sampler/prefetch dışında ekstra iş veya worker başlatılmaz.

DataLoaderAnalysisResult immutable schema1'dir. Embedded canonical P19/P20
source SHA/segment association, raw lifecycle/sample/ID/clock/order ve derived
metrics deserialize/export sırasında doğrulanır. JSON canonical tam kayıttır;
CSV raw observer satırlarıdır. Atomic/no-clobber export, `overwrite=True` opt-in,
CSV formula guard mevcut ortak export helper'ından gelir.

```python
import hashlib
from pathlib import Path
import tempfile
from hyso.experiments import ExperimentManager

# Ölçüm bitti; export/tracking hatasında fit tekrar çağrılmaz.
with tempfile.TemporaryDirectory(prefix='hyso-p21-') as directory:
    root = Path(directory)
    manager = ExperimentManager(root / 'store')
    manager.close()
    ExperimentManager.migrate_store(root / 'store')  # mevcut artifact store akışı
    manager = ExperimentManager(root / 'store')
    run = manager.create_run(manager.create_experiment('P21')['id'], 'training', {})
    run.start()
    for suffix in ('json', 'csv'):
        path = root / ('loader.' + suffix)
        getattr(result.loader, 'to_' + suffix)(path)
        artifact = run.add_artifact(path, kind='report', ownership='copy',
            metadata={'source_segment_id': result.timing.to_dict()['segment_id']})
        assert artifact['expected_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
    for summary in data['summaries']:
        if summary['phase'] == 'training':
            run.log_metric(name='loader.next_wait', value=summary['next_wait_ms'], unit='ms',
                           source='P21', epoch=summary['epoch_id'])
    run.finish()
    identity = run.id
    manager.close()
    manager = ExperimentManager(root / 'store')
    assert len(manager.get_run(identity).read_metrics()) == 2
    assert len(manager.list_artifacts(identity)) == 2
    manager.close()
```

Unsupported worker/queue/isolated-H2D metrikleri sıfır olarak loglanmaz. Yeni
SQLite schema/run kind yoktur; ölçüm recorder'ı DB/dosya yazmaz. Opt-in checkpoint
I/O ve caller callback I/O mevcut Trainer davranışıdır.

Native kanıt ve sınırlar: [P21 audit](../audit/m2-p-b10-loader-bottleneck.md).
B9 korundu; P21 worker trace kapsamı partial, B10 P23/P25 olmadan full GO değildir.
