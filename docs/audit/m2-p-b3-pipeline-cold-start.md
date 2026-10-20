# M2-P-B3 — Pipeline, stage timing ve cold start

Tarih: 2026-10-20. Branch `feature/hyso-models-next`, HEAD
`e17a6d2dd56cc0a463c5bae6a13086b158667eae` (B2 committed).
**P03 PASS, P04 PASS, P05 PASS; M2-P-B3 PASS**, aşağıdaki doğrulanmış kapsamda.
M2-P-B4'e geçilebilir; B4 uygulaması bu değişikliğe dahil değildir.

## 1. İnceleme ve değişiklik sınırı

M2-P-A tasarımı, B1/B2 audit'leri, backlog, performance modülleri, Benchmark
runner/result, Experiment Manager ve packaging guard incelendi. Hibrit mimari
korundu: bağımsız versioned results ve explicit ölçüm API'leri. BenchmarkResult
schema1, ThroughputResult/MemoryResult/FootprintResult sözleşmeleri değişmedi.
Runner, eski inference/memory, Trainer/checkpoint, Manager/schema2 ve önceki
audit/roadmap dosyaları byte-identical kaldı. Generic plugin/scheduler/web katmanı,
conversion engine, camera/YOLO adapter, yeni dependency veya version bump yoktur.

Kullanıcıya ait `tests/test_gradient_accumulation.py` farkı korunur; SHA256
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Branch/HEAD değiştirilmedi, index boş bırakıldı; commit/push yapılmadı.

## 2. Yeni public API ve results

`hyso.performance` additive API:

```python
PipelineSample(sample_id, payload)
PipelineStage(name, function, device='cpu')
PipelineProtocol(device='cpu', precision='fp32', samples=10, repeats=1,
                 warmup_samples=1, queue_depth=2)
measure_pipeline(model, source, *, preprocess=None, transfer=None,
                 postprocess=None, deliver=None, protocol=PipelineProtocol())
analyze_stages(pipeline_result)
measure_stages(model, source, **kwargs)
PipelineResult
StageTimingResult
ColdStartRecipe(factory, *, model_kwargs=None, input_shape=(1,1,5,7),
                device='cpu', precision='fp32', artifact_path=None,
                artifact_sha256=None, factory_search_path=None, seed=0)
ColdStartProtocol(trials=3, timeout_seconds=60., warmup_iterations=2, warm_samples=3)
measure_cold_start(recipe, *, protocol=ColdStartProtocol())
ColdStartResult
```

Formats `hyso.performance.pipeline`, `.stage_timing`, `.cold_start`, schema_version=1;
feature IDs P03/P04/P05. Existing frozen JSON `_Result`/export primitives reused.
Results model/tensor/handle/callable tutmaz; `to_dict` defensive copy'dir. UUID,
protocol/scope, requested/actual counts, units/status/reason, device/runtime/model/
source/artifact provenance ve raw evidence bulunur. JSON finite/allow_nan=False;
CSV formula-safe, atomic/no-clobber publication ve temporary cleanup korunur.
Eksik phase null+reason'dır. Stage CSV per-ID/stage boundaries; cold CSV phase
satırlarıdır. JSON tam kanonik kayıttır. Mevcut B1/B2 export branches korunmuştur.

## 3. P04 stage saatleri

Explicit source/read, preprocess, transfer, inference, postprocess, optional
delivery sırası vardır. Adapters aynı-ID `PipelineSample` döndürür. Shape/dtype/
device input-output boundaries, stage identity/order/count, host start/launch-end/
completion ns ve elapsed ms kaydedilir. CPU stage synchronous caller sözleşmesidir.
CUDA stage selected captured current-stream üzerinde event pair ve end-event sync
kullanır; event ms host launch/completion ms ile ayrı tutulur. Async kernel launch
süresi completion diye sunulmaz. Hidden cross-stream/other-device işleri ve async
CPU callbacks desteklenmez. CUDA transfer explicit CUDA stage olmalıdır.

`analyze_stages` raw evidence'ı tekrar doğrular, yeni UUID ve source_result_id ile
P04 üretir; model tekrar çalışmaz. Stage toplamı E2E hesabında kullanılmaz. Bu
instrumented serial isolation overhead'i bare latency veya overlap kapasitesi
değildir; BenchmarkRunner ölçüm penceresine eklenmedi.

## 4. P03 aynı-ID E2E ve queue

Her source çağrısı warmup/repeat dahil unique ID ve dense CPU tensor üretir.
Caller source tensor clone edilir; model input ayrıca ordinary tensor olarak
guard edilir. Bounded deque admission doluyken yeni source okumaz: blocking serial
policy, overlap/drop yoktur. Queue wait, actual max_pending ve completed/dropped/
pending ledger kaydedilir. Başarılı sonuçta dropped=0/pending=0 gerçek invariant;
failure/incomplete/duplicate-ID exception üretir, sahte başarılı result üretmez.

E2E `perf_counter_ns` ile aynı ID'nin source/read-start→postprocess completion
host wall farkıdır. CUDA terminal event synchronization bu bitişten öncedir.
Optional delivery E2E dışında ayrı timestamp/stage'dir. Source clone, queue wait,
guards ve stage instrumentation E2E'ye dahildir; snapshot setup/export/tracking
dışarıdadır. Queue wait diğer read/clone ve önceki guard sürelerini içerebilir.
P50/P95 descriptive; P99 B1 TailPolicy n≥1000 ve tail≥10 şartıyla, aksi null/
insufficient_samples. Pooled samples bağımsızlık/CI garantisi değildir. Sabit
latency/FPS eşiği kullanılmadı. Bilinen clock reference testinde E2E70ns,
stage sum40ns ve queue wait10ns ayrımı doğrulanır.

## 5. P05 fresh-process ve clock domain

Her trial `sys.executable -I -B _cold_worker.py` ile yeni interpreter'dır; Windows
spawn gereksinimini importable recipe/plain JSON ile karşılar. Parent'tan canlı
model/tensor/CUDA handle taşınmaz. Trusted factory `module:qualified_name`, plain
kwargs ve optional explicit search directory kullanır; factory sandbox değildir.

Worker heavy imports'tan önce ready marker gönderir. Parent startup kendi
monotonic domain'inde launch→ready **receipt** süresidir; launcher/bootstrap/pipe/
reader scheduling overhead dahildir. Parent total output publication/shutdown'ı
da içerir. Worker timestamps ayrı domain'dedir, parent'tan çıkarılmaz. Windows
venv redirector launcher_pid ile gerçek worker_pid farklı olabilir; ikisi ayrı
kaydedilir ve ready/output worker identity doğrulanır.

Worker phase sırası: module_import → CPU model construction/input creation →
artifact_load → runtime_prepare → first_inference → warmup → warm_reference.
Her phase clock/timestamp/ms/status/reason/included operations/device/backend
taşır. Runtime preparation CUDA context init/model-input movement/drain içerir;
factory import/construction sırasında CUDA init veya CUDA state kabul edilmez.
First completion host timing ve selected-stream event drain ile gerçek completion;
output validation sonrasındadır. Warm reference ayrı raw samples/phase'dir.
Forward count=1+warmup+warm samples; import/prepare ilk inference'a karıştırılmaz.
CPU fp32 ve native CUDA fp32/amp_fp16 scope'u doğrulandı. Protocol CPU fp64'ü de
kabul eder; bu milestone'da ayrı fp64 integration kanıtı üretilmedi.

## 6. Artifact ve başarısız trial

P11 `_file_info` byte/SHA primitive reused. Artifact verilen recipe expected SHA
zorunlu tutar; worker regular-file size/hash doğruladıktan sonra yalnız
`torch.load(weights_only=True, map_location='cpu')` ve strict state_dict load yapar.
Unsafe fallback yoktur; tests/guide kendi ürettiği trusted fixtures kullanır.
Artifact yoksa load phase disabled/null'dır. File read/hash/deserialization bu
phase'in açık maliyetidir. OS page cache, controller/disk cache veya driver state
coldluğu kanıtlanmaz: `unknown_not_flushed` kaydedilir, cache flush yapılmaz.

Failed/unsupported/timed_out/crashed terminal trial outcomes ve exit codes raw
evidence'dır. Başarısız trial başarılı first-inference average'a girmez; bütün
trial'lar başarısızsa metric unavailable/null. Import/construction errors, missing/
corrupt/hash mismatch artifact, crash, startup exception, timeout ve unsupported
device testlidir. Unexecuted phases not_observed/null; sıfırla doldurulmaz.

Timeout Windows'ta hâlâ canlı ve sahip olunan launcher PID'sinin `/T /F` process
ağacını sonlandırır; wait, reader join, stdout/stderr ve TemporaryDirectory cleanup
finally içindedir. POSIX owned-process kill path vardır; native POSIX validation
iddiası yoktur. Hidden descendants/breakaway factory davranışı destek dışıdır.

## 7. State, RNG ve lifetime

Existing `_profile`, `_Snapshot`, descriptor, output/input guards ve shared run
lock reused. Caller model device/dtype önceden doğru olmalı; pipeline taşımaz.
Registered parameter/buffer/gradient/mode korunur; mutation detected olursa hata
ve best-effort value rollback vardır. Python/NumPy/Torch CPU/selected CUDA RNG,
selected stream/device finally restore edilir. Exact warmup/measured forward
counts testlidir; fazladan preflight forward yoktur. Context/lock/error cleanup
ve GC-disabled weakref lifetime testleri vardır; recursive self-capturing closure
yoktur. Arbitrary Python/external side effects transaction garantisi verilmez.
Caller-exclusive kullanım gerekir. P05 worker-owned state parent'a sızmaz.
P09 allocator reset/empty_cache davranışına ve global GC policy'ye dokunulmadı.

## 8. Experiment entegrasyonu ve rehber

[Çalışır rehber](../guides/performance-pipeline.md) CPU resize/normalize/post,
explicit CUDA function, actual own-artifact cold recipe ve manual Manager örneği
içerir. Bütün Python blokları source ve wheel/sdist installed tests ile, ayrıca
minimum Torch installed smoke ile execute edildi. CUDA function native script'te
eşdeğer gerçek cihaz senaryolarıyla ayrıca doğrulandı.

Ölçümden sonra versioned JSON/raw stage-phase evidence copy artifact olarak
ilişkilendirilir; SHA/size verify, üç bounded scalar metric, finish/close/reopen
ve persistence verify yapılır. Measurement içine hidden SQLite write eklenmedi.
Existing public API/schema2 artifact ve generic metrics kullanılır; yeni table,
migration veya run kind yoktur. Run UUID/result UUID/model artifact SHA ayrıdır.
Future ONNX/LibTorch/C++/TensorRT/OpenVINO/TFLite/Core ML/YOLO backend-artifact
sınırları korunur; burada yalnız PyTorch eager uygulanır.

## 9. Güncel CPU kanıtı ve hata düzeltmeleri

Evidence kökü **E**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb3-20261020`.
Yeni `test_performance_pipeline.py`: **58 PASS**. Kapsam: actual transforms,
known timestamps, same-ID/invalid/duplicate IDs, queue/tail, warmup/order/count,
real fresh workers, artifact/failure/timeout/crash, schema/frozen/JSON/CSV,
state/RNG/weakref cleanup ve actual Experiment reopen/executable guide.

İlk56 CPU case53 PASS/3 FAIL: unavailable CUDA cleanup gereksiz sync, stage CSV
duplicate sample_id ve aynı CSV hatasına bağlı guide failure. Production düzeltildi.
İlk native8 koşusu7 PASS/1 FAIL: aşırı büyük device index unsupported yerine failed
oluyordu; worker torch.device oluşturmadan index sınırını denetleyecek şekilde
düzeltildi. Sonraki native8/8 geçti.

`targeted-final.xml/log`: **720 PASS/3 FAIL**, toplam723,26 mevcut warning.
Yalnız source guide ve iki installed packaging guide case'i fail oldu:
`exec(...,{})` altındaki source callable.__module__ None idi. Provenance için
unknown_module fallback eklendi. İlk minimum installed smoke da aynı eski wheel
hatasını doğruladı; beforefix logları saklandı. Final wheel yeniden build edildi.
`affected-final.xml/log`: **62 PASS**, 58 yeni +4 packaging;120.66s.
Son gate tüm yeni CPU testlerini (güçlendirilmiş RNG assertions dahil) içerir.

| Güncel doğrulanan distinct kapsam | PASS |
|---|---:|
| Yeni P03/P04/P05 CPU/protocol |58|
| B1 inference / B2 memory |122 +83|
| Benchmark CPU85/state28/result24 (C1 dört lifetime dahil) |137|
| Experiment core/metrics/storage/artifacts/migration/recovery/retry/integration |259|
| Public/published/model-state |60|
| Packaging |4|
| Birleşik son durum |723|

**723 tek bir temiz koşu sayısı değildir**: ilk gate'in değişmemiş legacy661 PASS'i
ve onarımdan sonraki62 PASS birleştirilmiştir. Tekrarlanan case'ler çift sayılmaz.
RNG güçlendirmesi sırasında tek-case PASS ek distinct test değildir. Önceki full
suite sonuçları güncel PASS diye sunulmadı; full CPU/Trainer çalıştırılmadı.

## 10. Native CUDA ve Windows kanıtı

Python3.10.18, Torch2.5.1+cu121, CUDA build12.1, RTX3060 Laptop GPU/capability8.6,
driver591.74; gerçek cihaz, mock native PASS değildir. `native-final.json/log`
**8 PASS**; final wheel installed `installed-cuda-pipeline.json` **8 PASS tekrar**:
`fp32_default`, `fp32_nondefault`, `cold_fp32`, `amp_fp16_default`,
`amp_fp16_nondefault`, `cold_amp_fp16`, `stream_failure_cleanup`,
`cold_unsupported_device`. Her case raw ID/count/timestamps/events/stream/device
kanıtı taşır. İki cold case gerçek ayrı CUDA worker ve SHA-verified own fixture
kullanır; import/construction CUDA false, runtime preparation sonrası true.
Installed tekrar post callback'in Python/NumPy/Torch CPU/CUDA RNG tüketimini de
restore assertions ile sınar. Terminal event delivery öncesi complete olmalıdır.

Current legacy scripts: `native-b1.json/log` **15 PASS**, `native-benchmark.json/log`
**45 PASS** (P09 dahil). Final wheel installed B1 **15 PASS tekrar**, native Windows
P10 `installed-windows.json` **3 PASS**. Bu tekrarlar distinct ID toplamına eklenmez.
C1 dört weakref case'i Benchmark state28 içindedir. Bütün eski162 CUDA script'i
gereksiz tekrar çalıştırılmadı. Native Linux/other GPU/backend iddiası yoktur.

## 11. Minimum Torch ve packaging

Mevcut Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu interpreter kullanıldı:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Final local wheel no-index/no-deps/no-compile ile **E/installed-final** target'a
kuruldu; dependency indirilmedi/kurulmadı. Cwd=E, `-I -B` installed provenance.
P03/P04 CPU transforms/export, gerçek P05 CPU worker (aynı Torch2.2.2), E artifact
association/reopen, B1/B2 ve legacy replay/checkpoint/import smoke PASS.
CUDA initialized false. Minimum prefix distribution METADATA/RECORD hash'leri
önce/sonra eşit; tüm binary environment full-content hash iddiası yoktur.

Explicit inventory25→**28 production modules**: pipeline.py/cold.py/_cold_worker.py.
Guard daraltılmadı. Wheel/sdist complete inventory/source-byte equality, installed
imports, unchanged version/dependency bounds ve tests/docs/temp/cache/weights/DB
exclusion PASS. Final installed28 modül source-byte equality ayrıca doğrulandı.
Final build `E/pytest-affected/packaging0/dist`:

| Artifact | Byte | SHA256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |121372|`e73b475d479fbd8acff04bb2dd58b8776554281a6d9f5d8a178d9d8e516a7bbf`|
| hyso-0.2.0.tar.gz |113532|`072f5e765540af786e53e7422f42fdb108087e93a4cac4c919acdec4adc90694`|

`minimum-smoke.json`, `minimum-environment.json`, `installed-integrity.json`,
`installed-commands.json` ve stdout/stderr E'de korunur. Pre-fix artifacts final
kanıt diye kullanılmadı. Son build sonrası production source değiştirilmedi.

## 12. Backlog ve final sınırlar

P03/P04/P05 implemented yalnız doğrulanan serial CPU/CUDA ve native Windows fresh
interpreter scope'larıyla. Güncel backlog **8 implemented, 2 partial, 25 planned**;
diğer P statüleri ve B2 native Linux sınırı korunur. P06 ve sonrası geliştirilmedi.

Sınırlar: instrumentation overhead, synchronous blocking admission, hidden stream
ve external side effects, OS scheduling/cache etkisi, trusted factory ownership,
fresh process'in cold OS/driver anlamına gelmemesi. Başarısız pipeline exception,
başarısız cold trial explicit evidence üretir. Diğer inference/export backend'leri,
camera/drop/overlap scheduling desteklenmiş gibi raporlanmaz.

`E/verification-summary.json` final test/source/archive/git inventory'sidir.
Başlangıç tracked SHA inventory'sine göre yalnız namespace/result/packaging/backlog
eski dosyaları değişti; yeni production3, tests3 ve guide/audit2 dosya eklendi.
Kullanıcı dosyası korunur. `git diff --check` ve yeni dosya whitespace kontrolü
başarılıdır. B3 kapıları tamamlandığı için **M2-P-B4'e geçilebilir**.
