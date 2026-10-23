# P21 feature completion — 23 Ekim 2026

**Nihai feature durumu: partial.** Önceki v1 kabulü ve çalışan IPC/recorder
altyapısı korundu. Bu çalışmada gerçek execution/epoch/batch bağlama, collate,
parent dispatch çağrısı ve task yaşam döngüsü işlevleri eklendi. Fiziksel queue,
tam prefetch efficiency, worker result dispatch ve cross-process nokta süreleri
tamamlanmış gösterilmedi. Başka P özelliğine geçilmedi; commit/push yapılmadı.

## Gerçekten supported/measured olanlar

| Yetenek | Kanıt ve ölçülen kapsam |
|---|---|
| Execution association | Ölçüm girişinde tek kullanımlık UUID; gerçek task mesajı ve worker result receipt'i; başarılı aynı fit'in P21 source hash/segment binding'i |
| Epoch/batch association | Gerçek P19 recorder epoch ID'si, PyTorch task index'i ve token; parent tesliminin gerçek P21 batch observation nesnesine bağlanması |
| Tekrarlı sample ID | Task token + fetch pozisyonu; sample sırası/PID tekrarından ilişki tahmini yok |
| Persistent/nonpersistent spawn | İki worker, iki epoch; nonpersistent süreçler gerçek epoch ID ile ayrı kaydedilir |
| Fetch/decode/transform | Mevcut WorkerTimingRecorder/TracedDataset kullanılır; değişken stage sayısı ve worker-local clock |
| Collate | Gerçek worker collate çağrısı; default veya uyumlu custom operation; failed collate complete olamaz |
| Dispatch | Gerçek parent `index_queue.put()` çağrısının start/end/duration değerleri; scope açıkça API çağrısı |
| Task yaşam döngüsü | Submit→parent receive→delivery token zinciri, korunumu ve high-water değerleri |
| Prefetch gözlemi | Talep anındaki gerçek ordered-result buffer hit sayısı ve oranı; efficiency adıyla sunulmaz |
| Clock association | Parent/worker metadata kontrolü ve nedensel offset gözlem aralığı; genişliği saklanır, midpoint veya nokta süre çıkarımı yapılmaz |
| Sonuç entegrasyonu | Yeni `WorkerTaskAnalysisResult` schema v2; embedded DataLoaderAnalysisResult v1, P19/P20/P22 kaynakları aynı fit'ten |

Yanlış source result, kanal tüketilmeden reddedilir. Eksik/duplicate/bozuk task,
run/epoch/batch/PID uyuşmazlığı, sayaç kaybı, drop, collate failure, sıra veya
clock uyuşmazlığı `complete` üretemez. Incomplete sonuçlarda süre ortalamaları
ve kullanılabilir lifecycle özetleri null kalır. Hash bütünlük denetimidir;
imzalı/uzak üretici kimlik doğrulaması iddia edilmez.

## Somut API sınırları ve kalan engeller

1. **Fiziksel queue occupancy:** PyTorch `_try_put_index` yalnız
   `index_queue.put((send_idx, index))` çağırır. Python multiprocessing Queue
   put çağrısı feeder serialization/pipe transferi bitmeden döner. Worker
   `index_queue.get()` sınırı `_worker_loop` içinde, fetch callback'inden
   öncedir. Yeni adaptörün worker fetch girişi gerçek dequeue anı olarak
   etiketlenmez. Parent submit/receive sayıları görev yaşam döngüsüdür, fiziksel
   queue occupancy değildir. `qsize()` kullanılmaz.
2. **Worker result dispatch:** `_MapDatasetFetcher.fetch` collate sonucunu
   döndürür; `_worker_loop` daha sonra `data_queue.put((idx, data))` çağırır.
   Collate bitişi bu sonraki işlemin tamamlanmasını ölçmez. Feeder/pipe veya
   worker loop değiştirilmeden tam serialization/dispatch tamamlanma receipt'i
   yoktur. Bu işte çalışan transport/worker loop yeniden yazılmadı.
3. **Tam prefetch efficiency:** `_task_info` cached-result hit'i ölçülür. Bu,
   parent'ın henüz okumadığı opaque result queue'da hazır veri olup olmadığını
   göstermez; ideal prefetch kapasitesi/yararı için kesin bir payda sağlamaz.
   Bu nedenle efficiency ve physical occupancy null/unsupported kalır.
4. **Cross-process nokta süreleri:** Python 3.10 Windows'ta perf_counter ortak
   clock kullanır; parent ve worker metadata'sı kontrol edilir. Yine de clock
   resolution toplam accuracy/drift hata sınırı değildir. `p0-w1` ile `p3-w2`
   arasındaki nedensel offset aralığı ve genişliği saklanır; bu gözlem aralığı
   mutlak donanım hata garantisi sayılmaz. Worker service parent wait'ten
   çıkarılmaz veya parent wait'e eklenmez. Nokta duration unsupported kalır.
5. **Destek kapsamı:** Adaptör PyTorch 2.5.1 ve dört iterator metodunun source
   SHA256 değeriyle gate edilir. IterableDataset, bulk `__getitems__`,
   pin_memory, batching kapalı akış, task-traced validation ve checkpoint/resume
   reddedilir. Plain validation kullanılabilir. Dataset wrapper'ını görünmez
   varsayan kullanıcı init/collate/dataset koduna genel şeffaflık garantisi yoktur.

Bu sınırları kaldırmak, mevcut opt-in callback kapsamından daha derin native
transport/worker-loop instrumentation ve ayrıca accuracy kanıtı gerektirir.
Desteklenmeyen alanlara tahmini değer yazmak veya mevcut 17 implemented
özelliğin davranışını değiştirmek tercih edilmedi. Genel P21 bu nedenle
implemented olarak yeniden etiketlenmedi.

Kaynaklar: [PyTorch 2.5.1 DataLoader](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/dataloader.py),
[worker fetcher](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/_utils/fetch.py),
[CPython 3.10 perf_counter](https://docs.python.org/3.10/library/time.html#time.perf_counter).
Kurulu kaynakların gerçek kodu da `torch_gpu` interpreter'ında inspect ile
okundu; adaptör fingerprint'i
`e59741680fd0aeaba408f39f5b560bca47d3c3759fabd6d6c428e4f165e5acf3`.

## Dosyalar ve geriye uyumluluk

- Yeni uygulama: `hyso/performance/_loader_task_trace.py`,
  `hyso/performance/worker_task_result.py`.
- Additive bağlama: `hyso/performance/dataloader.py` ve public export için
  `hyso/performance/__init__.py`. Normal DataLoader yolu task hook'u çalıştırmaz.
- Yeni kabul: `tests/p21_task_acceptance.py`, `tests/test_worker_tasks.py`.
- Kullanım/scope: `docs/design/p21-worker-tasks.md`; eski iki kabul/tasarım
  belgesine follow-up bağlantısı; sözleşmede yalnız P21 satırı güncellendi.

Önceki yerel değişiklikler ve scratch dosyaları korundu. Mevcut
`WorkerEventChannel`, worker recorder, stage recorder, cleanup helper ve v1
worker result bu çalışma sırasında yeniden yazılmadı. Önceki 31 + 222 test
dosyalarına dokunulmadı. DataLoaderAnalysisResult schema v1'in zorunlu
partial/null alanları değiştirilmedi. Yeni yetenekler ayrı v2 companion'dadır;
v1 JSON importer ve eski prototype JSON round-trip aynı kalır.

## Kabul komutları ve gerçek sonuçlar

Windows, `torch_gpu`, Python 3.10.18, PyTorch 2.5.1+cu121; CPU,
`CUDA_VISIBLE_DEVICES=-1`. Yeni native testler mevcut Windows Job supervisor
altında, 180 saniye dış deadline ile izole çalışır.

```powershell
& 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe' -m pytest tests/test_worker_tasks.py -q --tb=short
& 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe' -m pytest tests/test_worker_tracing.py tests/test_worker_tasks.py tests/test_dataloader_protocol.py tests/test_performance_dataloader.py tests/test_performance_training.py tests/test_training_details.py tests/test_training_throughput_protocol.py -q --tb=short
```

İlk bağımsız yeni işlev turu: **17 PASS / 0 FAIL / 0 SKIP**, 62,72 s.
Epoch bazlı nonpersistent process receipt ve gerçek failure kapsamları
tamamlandıktan sonraki tur: **21 PASS / 0 FAIL / 0 SKIP**, 208,93 s.

**Son birleşik koşu: 274 PASS / 0 FAIL / 0 SKIP**, **419,05 s**.
Bu sayı önceki 253 testi ve 21 yeni testi kapsar; önceki turlar tekrar eklenmedi.
Mevcut GradScaler API kullanımından 199 deprecation warning alındı. Başarısız
test veya dış Job timeout yok. Son kaynak validator'ı ile arşivlenen persistent,
nonpersistent ve saturation artifact'ları ayrıca tekrar import edilip türetilen
analizleri doğrulandı. Bool worker ID ve source clock penceresi dışındaki
delivery için iki bağımsız negatif validator kontrolü de incomplete verdi;
bunlar 274 pytest sayısına eklenmedi.

| Native v2 senaryosu | Son koşu kanıtı |
|---|---|
| Persistent, iki epoch | 8 task paketi / 16 sample; attempted=accepted=received=8, rejected=0; execution/epoch/batch verified |
| Nonpersistent, iki epoch | Aynı 8/16 korunum; dört gerçek epoch/worker process receipt'i; verified |
| Collate/dispatch | Persistent: collate 8 çağrı, mean 11,0497375 ms; parent queue put 8 çağrı, mean 0,2337875 ms. Bunlar bu koşunun ham ortalamalarıdır, performans garantisi değildir |
| Task lifecycle | Her iki normal koşuda submitted-not-received high-water 4; received-not-delivered high-water 2; ordered buffer hit 1/8 |
| Clock aralığı | Persistent koşuda belirsizlik genişliği 1.270.100–7.952.812.300 ns; cold worker startup geniş aralık üretir, hassas calibration iddia edilmez |
| Crash | Worker exit 0/7; cleanup 1,171 s; source result oluşturulmadı |
| Hang | Forced exit -15/-15; cleanup 1,266 s (2 s bütçe); source result oluşturulmadı |
| Collate exception | Asıl exception korundu; cleanup 1,000 s; worker exit 0/0; source result oluşturulmadı |
| Saturation | attempted=8, accepted/received=1, rejected=7; incomplete; süre ortalamaları ve lifecycle özetleri null |

Tüm native launcher'lar exit 0 ve `timed_out=false` verdi. Kontrollü worker
crash ve forced termination bu senaryolarda beklenen failure kanıtıdır;
başarılı eğitim olarak raporlanmadı.

Kanıtlar: [ilk yeni işlev turu](p21-feature-evidence/initial-new-tests.txt),
[genişletilmiş yeni işlev turu](p21-feature-evidence/expanded-new-tests.txt),
[274 test birleşik log](p21-feature-evidence/combined-final.txt),
[tüm native receipt'ler](p21-feature-evidence/native-summary.json),
[persistent v2 artifact](p21-feature-evidence/persistent-tasks.json),
[nonpersistent v2 artifact](p21-feature-evidence/nonpersistent-tasks.json),
[saturation artifact](p21-feature-evidence/saturation-tasks.json).

Yeni gerçek spawn kabulünde default sampler/shuffle, iki epoch persistent ve
nonpersistent worker'lar, stochastic custom collate (Python/NumPy/Torch RNG),
model/optimizer/gradient/RNG/generator state ve input sırası bit düzeyinde
karşılaştırılır. Her baseline ve observed fit ayrı karşılaştırma çalışmasıdır;
her biri yalnız bir kez çağrılır. Aynı observed fit'in P19/P20/P21/P22 kaynakları
eşleşir. Cross-source giriş, yeniden execution kullanımı ve tüketilmiş kanalla
tekrar analiz de başarı sayılmaz.

Crash ve hang sırasında geçerli source result üretilmez; zorunlu kapanış temiz
receipt sayılmaz. Collate exception asıl hata mesajıyla korunur. Gerçek
saturation'da 8 task attempt, 1 accepted/received, 7 rejected → incomplete.
Raw identity/timing/clock/counter/drop/duplicate/missing negatifleri ve
JSON/hash/immutable/no-clobber testleri de korunur.

CUDA ve başka platform/sürüm kabulü **NOT_RUN**. Bu kapsamlar PASS sayısına
eklenmez. Kanal/parent ledger bounded bütçeleri dolduğunda eksik kanıt başarıya
çevrilmez; tek kullanımlık final analiz ve finite capture bütçesi korunur.

## Git diff --stat

Bu çıktı yalnız tracked dosyaları gösterir ve başlangıçta mevcut olan yerel
değişiklikleri de içerir. Yeni/untracked dosyalar yukarıda ve status'ta listelenir.
`git diff --check` temiz; HEAD `2366607` olarak korundu.

```text
 docs/design/hyso-analysis-optimization-contract.md |   2 +-
 hyso/performance/__init__.py                       |   5 +
 hyso/performance/_loader_dataset_trace.py          |  50 +++-
 hyso/performance/_loader_event_channel.py          | 333 ++++++++++++++++++---
 hyso/performance/_loader_trace_groups.py           | 121 ++++++--
 hyso/performance/_loader_trace_validation.py       |  22 ++
 hyso/performance/dataloader.py                     |  32 +-
 7 files changed, 482 insertions(+), 83 deletions(-)
```

## Git status --short

```text
 M docs/design/hyso-analysis-optimization-contract.md
 M hyso/performance/__init__.py
 M hyso/performance/_loader_dataset_trace.py
 M hyso/performance/_loader_event_channel.py
 M hyso/performance/_loader_trace_groups.py
 M hyso/performance/_loader_trace_validation.py
 M hyso/performance/dataloader.py
?? docs/audit/p21-feature-completion.md
?? docs/audit/p21-feature-evidence/
?? docs/audit/p21-worker-tracing-acceptance.md
?? docs/audit/p21-worker-tracing-evidence/
?? docs/design/p21-worker-tasks.md
?? docs/design/p21-worker-tracing.md
?? hyso/performance/_loader_task_trace.py
?? hyso/performance/_loader_worker_lifecycle.py
?? hyso/performance/worker_task_result.py
?? hyso/performance/worker_trace_result.py
?? scratch_p21_dataloader.py
?? scratch_p21_delivery_audit.py
?? scratch_p21_failure.py
?? scratch_p21_integrity.py
?? scratch_p21_ipc.py
?? scratch_p21_ipc_v2.py
?? scratch_p21_live_delivery.py
?? scratch_p21_multistage.py
?? scratch_p21_multistage_spawn.py
?? scratch_p21_result_integration.py
?? scratch_p21_stages.py
?? scratch_p21_trace_integration.py
?? scratch_p21_variable_stages.py
?? scratch_p21_worker_core.py
?? scratch_p21_worker_shutdown.py
?? tests/p21_task_acceptance.py
?? tests/p21_worker_acceptance.py
?? tests/test_worker_tasks.py
?? tests/test_worker_tracing.py
```
