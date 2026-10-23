# P21 opt-in worker task kanıtı — v2

Bu ek, çalışan v1 IPC/recorder altyapısını kullanır. V1 loader ve worker JSON
kayıtları değişmez. Public API: `WorkerTaskDataLoader`, `analyze_worker_tasks`,
`WorkerTaskAnalysisResult`.

## Kullanım

```python
from hyso.performance import (
    WorkerTaskDataLoader, measure_training_with_loader, analyze_worker_tasks,
)
from hyso.performance._loader_worker_lifecycle import shutdown_traced_workers

# Windows spawn: script girişi if __name__ == "__main__" altında olmalı.
loader = WorkerTaskDataLoader(dataset, batch_size=8, num_workers=2,
    multiprocessing_context="spawn", persistent_workers=True)
try:
    try:
        training = measure_training_with_loader(trainer, loader, epochs=2)
    finally:
        if hasattr(loader, "trace_iterator"):
            shutdown_traced_workers(loader.trace_iterator, timeout=8)
    tasks = analyze_worker_tasks(training.loader, loader)
    tasks.to_json("worker-tasks.json")
finally:
    loader.trace_channel.close()
```

Her fit bir kez çalışır. Modele giden batch özgün collate çıktısıdır. Dataset
wrapper'ı worker_info içinde görünür; worker_init_fn/dataset/collate kodu wrapper
ile uyumlu olmalıdır. Eski dataset sınıf kimliğine bağımlı kullanıcı kodunun
genel şeffaflığı iddia edilmez.

## Açık kimlik ve gerçek ölçümler

Ölçüm başlangıcında tek kullanımlık execution UUID oluşturulur. Gerçek P19
recorder epoch ID'si parent iterator başlamadan loader'a aktarılır. PyTorch'un
gerçek `_send_idx` görev numarası batch ID'dir. Execution/epoch/batch ve ayrı
token gerçek index queue mesajında taşınır. Sample trace ID, task token + fetch
pozisyonudur; sample ID veya PID tekrarından ilişki çıkarılmaz.

Worker'ın döndürdüğü batch zarfı aynı receipt'i taşır. Parent teslimi P21 batch
observation nesnesine bağlanır. Başarılı fit sonunda kaynak hash ve segment ID
execution binding'e yazılır. Yanlış kaynakla analiz, kanal tüketilmeden
reddedilir. JSON importer zinciri yeniden doğrular. Hash bütünlük denetimidir;
imzalı üretici kimlik doğrulaması değildir. Nonpersistent akışta her gerçek
epoch'un ayrı process kayıtları tutulur; persistent akışta aynı süreçler
doğrulanır. Final analizden önce worker'lar durdurulmalıdır.

| Alan | Gerçek kapsam |
|---|---|
| dataset_fetch/decode/transform | Mevcut worker-local recorder; sample/task kimliği |
| collate | Worker'daki gerçek default veya uyumlu custom collate çağrısı |
| dispatch | Gerçek parent `index_queue.put()` çağrısı; feeder/pipe/worker result dispatch dahil değil |
| submitted_not_received | Parent submit/receive görev korunumu ve high-water; fiziksel occupancy değil |
| received_not_delivered | Parent'a ulaşmış, henüz kullanıcıya teslim edilmemiş görevler |
| ordered_buffer_hits/fraction | Talep başladığında gerçek `_task_info` ordered-result buffer'ında hazır bulunan sonuçlar; prefetch efficiency değil |
| clock association | Nedensel offset gözlem aralığı; nokta tahmini/cross-process duration yok |

Parent submit başlangıcı `p0`, worker fetch girişi `w1`, worker collate bitişi
`w2` ve parent sonucu alma zamanı `p3` için, aynı clock rate varsayımı altında
`p0-w1 <= parent_offset <= p3-w2` gözlem aralığı elde edilir. Genişlik ölçülen
belirsizliktir; simetrik transport veya midpoint varsayılmaz. Clock metadata'sı
eşleşmelidir. Aralık mutlak donanım accuracy/drift garantisi değildir; resolution
accuracy yerine geçmez. Worker/parent süre çıkarma ve toplama unsupported kalır.
[CPython clock belgesi](https://docs.python.org/3.10/library/time.html#time.perf_counter).

## Somut unsupported kapsamlar

`Queue.put`, feeder'a kabulden sonra döner. Worker `index_queue.get` ve
`data_queue.put` sınırları `_worker_loop` içindedir; fetch/collate callback'leri
bu transport sınırlarını içermez. Fiziksel queue occupancy ve worker result
serialization/dispatch tamamlanması, feeder/pipe veya worker loop müdahalesi
gerektirir. `qsize()` kullanılmaz. Collate bitişi result dispatch bitişi değildir.
[DataLoader kaynağı](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/dataloader.py),
[fetcher kaynağı](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/_utils/fetch.py).

Parent buffer hit'leri opaque result queue'da hazır veri veya ideal prefetch
kapasitesini göstermez. Bu nedenle tam efficiency null/unsupported kalır.
Cross-process nokta süreleri için nedensel aralık dışında accuracy/drift hata
sınırı yoktur; parent wait'ten worker service çıkarılmaz.

IterableDataset, bulk `__getitems__`, pin_memory, batching kapalı akış, farklı
multiprocessing context, task-traced validation loader ve checkpoint/resume
reddedilir. Plain validation loader kullanılabilir; normal ölçüm yolları aynı
kalır. Adaptör yalnız PyTorch 2.5.1 ve dört iterator metodunun beklenen source
SHA256 değeriyle açılır. Base worker loop, Queue, sampler ve shutdown algoritması
yeniden yazılmaz; global monkeypatch yoktur. Test edilen runtime Windows CPU'dur.

Parent ledger 100000 olayla, kanal mevcut slot/paket bütçeleriyle sınırlıdır.
Final analiz kanalı bir kez tüketir. Çok uzun çalışmada kapasiteyi aşan paketler
reddedilir ve incomplete üretilir; truncate/yeniden eğitim yoktur. `complete`,
v2 companion'ın desteklenen alt kapsamı içindir; P21 genel durumu **partial**.
