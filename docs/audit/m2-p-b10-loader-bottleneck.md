# M2-P-B10 / P21 — DataLoader bottleneck analysis

2026-10-21, Europe/Istanbul. Bu görev yalnız P21'i uygular; P23/P25 geliştirilmez.
**P21 PARTIAL:** gerçek main-process iterator/wait, native spawn ve P19/P20/P22
integration vardır; worker-side service trace yoktur. **B10 PARTIAL / full-scope
GO yok.** B9'un tarihsel doğrulanmış kapsamı ve eski audit kapıları değiştirilmedi.

Bu görevde üretilmiş yerel kanıt dizini **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p21-20261021`.
XML, stdout/stderr, native JSON/receipt, archive/protection ve final özet buradadır.
Önceki görevlerin sonuçları yeni test sayısına eklenmez.

## 1. HEAD / branch / protection

Başlangıç branch `feature/hyso-models-next`, HEAD
`24001db75cf3f6973a7a0a106fc49a0039d2b9b7` beklenen değerlerle eşleşti.
Index ve untracked envanteri boştu; tek unstaged fark kullanıcı gradient testiydi.
E/initial-status.txt. Stage/commit/push/reset/checkout/rebase/merge/clean yapılmadı.
Torch/CUDA/driver/dependency veya model/weight/dataset indirilmedi/güncellenmedi.
Yerel oluşturulmuş küçük PNG fixture'ları gerçek ImageFolder girdisidir.

## 2. Tasarım ve API

Yeni `hyso.performance`: DataLoaderAnalysisProtocol, DataLoaderAnalysisResult,
LoaderTrainingResult, measure_training_with_loader. Yeni production modülü
`hyso/performance/dataloader.py`. Bir actual fit'ten timing/P19, loader/P21,
optional stages/P20 ve derived throughput/P22 döner. stages=True default;
False durumunda prep/compute stage_links null/unsupported/P20_not_requested.

P19/P20/P22 public imza/schema'ları değişmez. `_measure_training` yalnız observer
sunabilen private recorder'ı geçici bağlar; `_stage_result` mevcut P20 construction
kodunun private reuse'udur. Yeni SQLite/benchmark/checkpoint schema yoktur.

## 3. Loader timing scope

`_ObservedIterator` yalnız opt-in branch'te kullanılır. Underlying loader veya
dataset yeniden oluşturulmaz/değiştirilmez. İlk `__iter__` tam bir native
iter(loader), her next bir gerçek underlying next yapar; normal terminal
StopIteration da gözlenir. Dataset/batch tensorları collector'da tutulmaz.
Default Trainer/P19/P20 branch'i underlying loader üzerinde normal for iteration
yapar; observer/timer/synchronization eklenmez. Loader konfigürasyonu caller-owned.

## 4. Iterator initialization ve next wait

iterator_init, delivered batch ve exhaustion raw kayıtları ayrıdır. Her biri
main perf_counter_ns başlangıç/bitiş ve host_ms içerir; batch delivery_ns=end_ns.
Iterator init native construction/launch/bookkeeping aralığıdır; worker'ların
asenkron import/ilk veri hazırlığının bütünü olduğu iddia edilmez. Terminal next
PyTorch worker teardown maliyeti içerebilir; delivered wait dağılımına karışmaz.

## 5. Worker service ve main wait ayrımı

Worker decode/transform/service trace kurulmadı. Ek dataset adapter, multiprocessing
IPC, worker event queue veya clock mapping yoktur. Worker service, queue occupancy
ve prefetch efficiency null/unsupported/no_verified_worker_or_queue_trace'dir.
Main-process next wait worker CPU/service süresi değildir. Prefetch nedeniyle kısa
wait, worker'ın çalışmadığını/verimli olduğunu kanıtlamaz. Worker service coverage
olmadığı için P21 complete/implemented ilan edilmez. Bu bir bütün platformlarda
worker trace imkânsızlığı iddiası değildir; bu implementation'ın açık sınırıdır.

Resmi davranış referansları:
[Torch2.5.1 DataLoader ownership/prefetch](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/dataloader.py),
[native worker RNG initialization](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/data/_utils/worker.py).

## 6. Batch / sample ledger

Training batch'leri mevcut global microbatch_id, segment window_id ve global
attempt_id'ye bağlanır. 5 sample/batch2/accum2 → 3 delivered batch, sample[2,2,1],
window sample[4,1], attempt[1,1,2]. Her epoch training count P19 ve P22 ile aynı.
Validation ayrı phase/batch_index kullanır; training ID'leri null'dır. Iterator
init/exhaustion kayıtlarında sample/batch/attempt uydurulmaz. Config provenance
num_workers/pin_memory/device/prefetch/persistence/timeout/sampler/collate/
worker-init/generator presence/multiprocessing context içerir.

## 7. P19 / P20 / P22 entegrasyonu

P21 raw evidence embedded canonical P19 + SHA256/segment association içerir;
P20 istendiyse source P20 SHA ve aynı P19 eşitliği doğrulanır. Prep ile ayrı
forward/loss/backward P20 kayıtları microbatch IDs üzerinden ilişkilidir; duration
toplamı üretilmez. Model compute P19 microbatch time'dan tahmin edilmez. P22 pure
analysis gerçek source P19 ledger'ını kullanır; yeni training çalıştırmaz.
Loader wait geçmiş P19/P20 JSON'undan retrospektif tahmin edilmez.

## 8. num_workers=0 native kanıtı

Yeni CPU suite gerçek TensorDataset/DataLoader üzerinde1/2/3 epoch,
shuffle/validation/stages açık-kapalı combinations ve25-case native ModularCNN
replay matrisi içerir. Init/next/StopIteration sayıları, tail sample'lar, monotonic
clock ve delivered→prep→compute sırası doğrulanır. Timer testlerinde sabit ms
beklentisi yok; fake-clock testleri exact100ns aralıklarla çalışır.

## 9. num_workers=2 gerçek Windows spawn

`tests/loader_native_validation.py`, DataLoader num_workers=2, explicit spawn,
prefetch_factor=2; shuffle ve persistent_workers senaryoları. TensorDataset native
worker startup, gerçek sample/batch order ve state eşitliği doğrulanır. Persistent
fixture'ları fit sonrasında tam2 canlı worker PID'sini gözler; caller loader
bırakıldıktan sonra native teardown ve active_children=0 gerektirir. Production
worker shutdown/terminate/private reset çağırmaz. Worker PID gözlemi test-only'dir;
worker service timing trace'i olarak sunulmaz.

Native supervisor90s deadline, Windows Job Object kill-on-close, 8 process,
4GiB/process ve12GiB/job private-commit tavanı kullanır. Gerçek interpreter doğrudan
başlatılır; venv site yolu child env'de korunur. Job admission marker'ı workload
ve spawn'dan önce yazılır. Final child process_pid=receipt.launcher_pid doğrulanır;
venv redirector'ın assignment öncesi escape riski bu final gate'te yoktur.

## 10. ImageFolder / augmentation

Beş yerel4x4 RGB PNG, iki sınıf, gerçek torchvision ImageFolder. Deterministic
ToTensor (validation dahil), main-process random transform ve spawn2 persistent
random transform senaryoları vardır. RandomHorizontalFlip + Python random,
NumPy RNG ve Torch RNG katkıları kullanılır. Gerçek CPU forward/backward/Adam
update çalışır; dataset/model/dataset download yoktur.

## 11. Dataset RNG / order

Baseline ve P21 aynı seed, aynı config ve aynı native DataLoader davranışından
başlar. Model input pre-hook'ları test-only batch içeriklerini kaydeder; paired
tensor byte eşitliği, content SHA, shuffle/generator state, Python/NumPy/Torch RNG
ve history karşılaştırılır. Observer sample kimliği/seed eklemez. Worker RNG'nin
gözlenen augmentation sonuçları exact eşittir; gizli worker clock veya final RNG
state'inin tamamı okunmuş gibi iddia edilmez.

## 12. CUDA / CPU clock

P21 host-only perf_counter_ns; ns→ms /1e6. Mean/P50/P95 linear quantile;
P99 n>=1000 delivered batches, aksi halde null/insufficient_samples. Training wait
fraction bütün main next aralıkları (delivered+exhaustion) / training wall;
init ayrı, validation hariç. Validation ratio unavailable. Bu gözlemsel oran
nedensel bottleneck teşhisi değildir; instrumentation overhead içerir.

P19/P20 selected-stream events ayrı domain; host wait'e eklenmez. `_batch` validation/
preparation+transfer birlikte olduğundan izole CUDA H2D süresi null/unsupported;
CPU'da H2D not_executed. GPU kernel attribution, worker GPU busy veya unmeasured
queue/utilization üretilmez. Default/non-default stream ve pinned/pageable input
native CUDA fixtures içinde ayrı doğrulanır.

## 13. Training state exact karşılaştırma

Normal ve P21 eğitim parameters/buffers/gradients, optimizer/scheduler/EMA/scaler,
history/counters/modes, RNG ve gerçek batch içerikleri için exact karşılaştırılır.
CPU dropout, shuffle, FP32/FP64, farklı attention/optimizer/scheduler combinations
ve native CUDA AMP/skip/recovery vardır. CUDA byte-equality kanıtı mevcut
deterministic/CUBLAS/TF32 koşulları içindir; production global flags değiştirmez.

## 14. Resume / EarlyStopping sınırları

CPU ve native CUDA epoch2 boundary kesintisi, yeni Trainer epoch3/4/5 segmenti ve
exact suffix state testli. Completed checkpoint boş P21 observations ve null P22
rate üretir; prefix time uydurulmaz. EarlyStopping2 completed epoch'la sonuçlanır;
spawn/persistent ve single-process senaryoları var. Strict resume hâlâ mevcut
ModularCNN+num_workers=0 TensorDataset profiliyle sınırlı; multiworker strict
checkpoint/resume desteği eklenmedi. `_data_plan` orijinal loader'ı görür.

## 15. Error / cleanup

Iterator exception/exhaustion ve recorder.close observer'ın underlying iterator/
loader referanslarını bırakır; remaining live observers set'i temizlenir.
Native PyTorch teardown ve persistent caller ownership korunur. Loader, model,
invalid/empty loader, cap exhaustion ve native worker exception/timeout fixture'ları
testlidir. Worker failure testleri expected Python RuntimeError/cleanup'tır;
beklenmeyen native crash olarak sayılmaz. Failed training rollback/retry iddiası
yoktur. Buffer limiti incomplete successful result'a dönüştürülmez.

## 16. CPU testleri

E/unit.xml ilk deterministic8 PASS. E/new-verified.xml/log final **78 PASS**,
0 failure/error/skip/xfail,15.88s:8 unit +70 real/contract tests. Protocol/cap,
known quantile1000/999, exact iter/next arithmetic, weakref cleanup, fake full
pipeline clock, exact5-record boundary, state/replay/validation/resume/ES,
immutable deserialize/tamper/CSV formula/no-clobber, Manager ve guide kapsar.
Önceki66/74/77 subset tekrarları78'e eklenmez. İlk iki spawn smoke probe PASS;
formal final supervisor/metadata gate'leri ayrı dosyalarda tekrarlandı, distinct
sayısı artırılmadı. Bu görevde gizlenen failure/skip/xfail yoktur.

## 17. Native CUDA ve loader gate'leri

Source formal gate:7 loader senaryosu (tensor_spawn, image_deterministic,
image_random_cpu, image_spawn_random, early_spawn, exception, timeout) ve20 CUDA
case (18 optimizer/AMP/stream/overflow paired +2 checkpoint/resume). Her child ayrı
bounded Job altında; E/source-verified-*.json/.receipt.json. Native CUDA allocator
fraction0.15, CPU thread1. Source CUDA15.375s/exit0; source early_spawn30.00s/exit0.
Installed gate aynı senaryoları tekrarlar; distinct case sayısı ikiye katlanmaz.
Package source path installed target içinde doğrulanır.

B9 korunması için ayrıca bu source üzerinde mevcut native P19 timing20,
P20 detailed21 ve deterministic replay10 gate'leri yeniden çalıştı; E/legacy-*
JSON/receipt/commands. Bunlar önceki görevlerden taşınmış test sayıları değildir.
Beklenmeyen native crash gözlenmedi; B4 reliability riski kapanmaz.

## 18. Regresyon

E/regression-files.txt / regression.xml / regression.log. Trainer classification/
replay/gradient accumulation/scaler/scheduler/EMA/checkpoint, P19/P20/P22,
Experiment Manager, benchmark state safety ve mevcut performance modülleri
seçildi; yeni P21 testleri bu sayıdan ayrı tutulur. **1971 PASS, 0 failure/error/
skip/xfail**,575.67s; verification-summary ile kaydedilir. Geçmiş1898/1971/2313
sonuçları bu yeni koşu sayısına eklenmez.

## 19. Minimum Torch

Mevcut Python3.10.18 / Torch2.2.2+cpu / uyumlu torchvision ortamında isolated
installed wheel smoke, P21 guide ve num_workers=0 checkpoint/resume exact PASS.
CUDA initialized=False kaldı. E/minimum-p21.json / minimum-*.stdout.log.
Yalnız HYSo local wheel --no-index/--no-deps/--target ile TEMP'e kuruldu; mevcut
dependency METADATA/RECORD before/after eşit, E/minimum-environment.json.
Minimum runtime için native spawn2 kanıtı bu görevde ilan edilmez; spawn2 ana
Torch2.5.1 runtime'da doğrulandı.

## 20. Wheel / sdist

E/packaging.xml/log: **4 PASS**,174.77s. Explicit inventory40→41; yeni production
modülü archive'larda var. Source/wheel/sdist/installed bütün41 modül byte-equal,
E/archive-integrity.json. Test worker/helper/dataset artifacts distribution'a
eklenmedi. Version/schema/migration değiştirilmedi.

## 21. Export / Experiment Manager

DataLoaderAnalysisResult immutable JSON-backed schema1; defensive to_dict.
Canonical JSON tam evidence; CSV raw observer satırları; atomic/no-clobber ve
CSV injection guard mevcut ortak helper ile. Malformed/tampered duration/ID/
delivery/count/hash/stage/source/worker payload reddedilir. Ölçümden sonra JSON/
CSV copied artifact + SHA verify + source association metadata, generic epoch
metrics, finish/close/reopen/read geçer. Null worker/queue/H2D metrikleri sıfırla
loglanmaz. Tracking failure fit'i tekrar çağırmaz; recorder DB/file yazmaz.
Yeni SQLite table/run kind yoktur.

## 22. Değişen dosyalar

Yeni8: `hyso/performance/dataloader.py`, `tests/test_dataloader_protocol.py`,
`tests/test_performance_dataloader.py`, `tests/loader_native_validation.py`,
`tests/cuda_loader_validation.py`, `tests/p21_native_supervisor.py`, bu audit ve
`docs/guides/performance-dataloader.md`.
Değişen6: `hyso/training/trainer.py`, `hyso/performance/training.py`,
`hyso/performance/training_details.py`, `hyso/performance/__init__.py`,
`tests/test_packaging.py`, `docs/roadmap/hyso-performance-backlog-v1.1.md`.
Toplam14 görev dosyası; kullanıcı gradient farkı ayrı15'inci working-tree farkıdır.
Eski audit belgeleri, resume.py, result.py, checkpoint ve Experiment Manager
production kodları değişmedi. Final git/signature/schema/protection E/protection.json.

## 23. Korunan SHA256

`tests/test_gradient_accumulation.py` başlangıç/final:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Dosya stage edilmedi/yazılmadı/geri alınmadı. Branch/HEAD ve boş index korunur.

## 24. Güncel35-feature sayımı

P21 planned→partial, **16 implemented /6 partial /13 planned =35**.
P19/P20/P22 implemented kalır; B9 verified GO geçmişi korunur. P18 planned/blocked,
B4 OPEN/UNRESOLVED, B6 profiling PARTIAL, B7 overlap PARTIAL. P23/P25 planned.
Post-Performance Developer Experience notu korunur; Hyso facade geliştirilmedi.

## 25. P21 PASS / PARTIAL / FAIL

**PARTIAL.** Native main-process iterator/wait, actual loader semantics, spawn2,
ImageFolder/augmentation ve source association/counts kapsamı doğrulandı. Worker
decode/transform/service trace ve queue metrics yoktur. Native spawn/P19/P20/P22
başarısı worker trace varlığı diye gösterilmez; complete/implemented iddiası yoktur.

## 26. B10 GO / PARTIAL / NO-GO

**PARTIAL / full-scope GO yok.** P21 service trace eksiği yanında P23 training-memory
ve P25 diagnostics uygulanmadı. Gerekli native testleri geçmek bu kapsamları
otomatik tamamlamaz. Stable release onayı verilmez.

## 27. P23 readiness ve sonraki adım

Training/loader phase ve source ID temeli hazır. P23 ayrı opt-in native memory
probe, allocator/RSS ownership/reset/completion ve state replay kanıtı gerektirir;
loader wait veya worker PID'den memory çıkarımı yapılmaz. Sonraki task P23'ü ayrı
kapsamda tasarlayıp native acceptance ile geliştirebilir. Bu görev P23/P25 veya
model mimarisi/facade geliştirmesi başlatmadı.
