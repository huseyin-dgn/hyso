# M2-P-B1 — P99 Latency & Real Sustained Throughput

2026-10-20, Europe/Istanbul. **PASS — P01/P02, aşağıda tanımlı resident CPU/CUDA
kapsamında.** Yeni P99 analizi ve gerçek completion-window throughput uygulanıp
doğrulandı. Diğer performans özellikleri, conversion, camera, profiler ve training
analytics bu göreve eklenmedi. Full CPU/162-native birleşik gate çalıştırılmadı;
aşağıdaki sonuçlar bu görevin hedefli ve gerçek yeni koşularıdır.

## 1. Branch, HEAD, kaynak ve changed files

Branch başlangıç/son `feature/hyso-models-next`; HEAD
`a12fef5fbbcc6529a295d7591157e92ba596ccbb`, `docs: design advanced performance engine`.
M2-P-A tasarım belgesi bu HEAD'de commit edilmişti. Index başlangıç/son boş.
Başlangıçta yalnız kullanıcıya ait `tests/test_gradient_accumulation.py` farkı
vardı; dosyaya yazılmadı/stage edilmedi/geri alınmadı. Başlangıç/son SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

Görevin dosyaları:

- Yeni: `hyso/performance/__init__.py`, `inference.py`, `result.py`.
- Yeni: `tests/test_performance_inference.py`, `tests/cuda_performance_validation.py`.
- Yeni: `docs/guides/performance-inference.md`, bu audit raporu.
- Güncellenen: `tests/test_packaging.py` ve yalnız P01/P02/current-count açıklaması
  için `docs/roadmap/hyso-performance-backlog-v1.1.md`.

Diğer bütün başlangıç tracked dosyalar byte-identical: BenchmarkRunner/Result/
namespace, Trainer, checkpoint format1, Experiment Manager, root exports,
metadata/dependency/version ve sabit roadmap v1.0 korunur. Commit/push, branch
değişimi, checkout/reset/rebase/merge/revert, dependency/dataset/weights kurulumu
veya indirme yapılmadı. Kendi final HYSo wheel'i yalnız explicit isolated install
target'larına kuruldu; mevcut ortam bağımlılıkları kurulmadı/değiştirilmedi.

Temel kaynaklar M2-P-A, backlog, D-C2/C1 ve istenen production/test kontratlarıdır.
Önceki 1995 full CPU/1 SKIP, Benchmark310, Experiment259 ve native162 sonuçları
tarihsel baseline'dır; bu görevde çalıştırılmış full gate gibi sunulmaz.

## 2. P01 implementation ve matematik

`analyze_latency(BenchmarkResult, quantiles=(.99,), tail_policy=TailPolicy())`
validated source'un defensive payload'unu yeniden doğrular; model/Runner.run
çağırmaz. Raw R×N positive finite samples, source UUID/SHA256 ve source model/
input/environment/configuration/provenance yeni sonuçta korunur. Eski P50/P90/P95,
Result schema1 ve derived rate hesapları değişmez.

Linear interpolation `position=(n−1)q`, komşu iki order statistic arasında yapılır.
Known-array [1..1000] için P99=**990.01**; test ayrıca P50=500.5, P90=900.1,
P95=950.05'i doğrular. Varsayılan minimum n=1000 ve (1−q)n≥10; q=.99/n=1000
sınırında floating-point yuvarlamasıyla false insufficient üretmemek için tail
eligibility Decimal(str(q)) üzerinden hesaplanır. n=999 ve küçük örneklerde
value=null, status=insufficient_samples, reason=tail_policy, unit=ms olur.

TailPolicy iki pozitif integer eşiği explicit değiştirebilir; bu descriptive
exploration'dır. Ham pooled samples correlated olabilir; repeat sayısı bağımsız
trial veya effective sample size sayılmaz, confidence interval üretilmez. Gerçek
native CUDA BenchmarkResult'tan R=2/N=500 ile P99 ayrıca doğrulandı.

## 3. P02 completion-window protokolü

`measure_throughput(model, input_source, protocol=ThroughputProtocol(...))` ilk
kapsamda bir resident contiguous NCHW Tensor kabul eder, defensive clone alır.
Caller model/input taşınmaz; CPU FP32/FP64 ve CUDA FP32/native AMP FP16 desteklidir.
Concurrency=1; queue_depth≥2; CPU synchronous actual queue depth=1. Batch cardinality
input N'dir. Work planında positive batch_count veya duration_seconds'dan yalnız
biri bulunur. Warmup/repeat explicit; her repeat ayrı warmup ve window'dur.

Önce model/output preflight, snapshot/descriptor/input hash, warmup ve CUDA drain
yapılır. Host `perf_counter_ns` t0'dan sonra sürekli admission başlar. Süre deadline'ı
admission'ı kapatır; GPU işi o anda tamamlanmış varsayılmaz. Admission-end timestamp
ayrı, terminal completion sonrası t1 ayrı kaydedilir. Throughput:

`sum(completed_images) / sum((t1−t0)/1e9)`.

Per-repeat actual batches/images, contiguous completed ID range, completion groups,
admitted/pending/max-pending, tail seconds, wall duration ve rate tutulur. IDs yalnız
completion kanıtından sonra ledger'a commit edilir; duplicate/negative/out-of-order
ID reddedilir. Completed=admitted, pending=0, images=batch×completed başarılı sonuç
invariant'larıdır. No-work, inconsistent clock/count, incomplete tail veya nonfinite
output/rate successful result üretmez. Numune yerine admission sayısı kullanılmaz.

## 4. CPU/CUDA sync ve measurement scope

CPU synchronous forward completion loop ve monotonic host clock kullanır.
CUDA captured current stream'de queue_depth kadar önceden initialized event pool ve
bounded output ring kullanır. Ring dolduğunda son completion event henüz bitmediyse
tek bounded-group backpressure wait yapılır; her forward'a sync eklenmez. Grubun
bütün event'leri query ile doğrulanıp IDs commit edilir. Admission sonunda ayrı
terminal event record/synchronize/query ve kalan event checks, tail commit'ten
öncedir. Tail toplam paydada bulunur. Native default/non-default stream doğrulandı.

Scope `resident_forward_completion_wall`; clock `perf_counter_ns`. Dahil:
forward, device-side finite reductions, host output metadata/context checks,
loop/queue bookkeeping, bounded backpressure, terminal tail. Hariç: input clone/
transfer, preflight/warmup, state/hash inspection, validity scalar readback,
JSON/CSV/SQLite/network işleri. Finite flag CUDA'da device-side biriktirilir;
her forward'da .item()/D2H yoktur, flag t1 sonrasında okunur. Instrumentation
overhead gerçek ölçüme dahildir, çıkarılmaz; sonuç bare inference latency değildir.
Terminal ve output finite kontrollerinden geçmeyen çalışma fake rate üretmez.

Güçlendirilmiş native assert'ler, successful run'da device synchronize sayısının
yalnız repeat sayısı kadar warmup drain olduğunu ve event wait'lerinin grup/
terminal/preparation sınırında kaldığını kanıtlar. Native async event'ler gerçektir;
iki terminal failure senaryosu gerçek workload üstünde kontrollü fault injection'dır,
gerçek GPU arızası veya poisoned-context recovery kanıtı değildir.

## 5. Yeni public API ve schema

Yeni namespace `hyso.performance`, altı export:
`TailPolicy`, `ThroughputProtocol`, `analyze_latency`, `measure_throughput`,
`LatencyAnalysis`, `ThroughputResult`. Yeni üç küçük modül dışında framework,
plugin dispatcher, generic orchestration veya public tracking adapter yoktur.

Independent format'lar `hyso.performance.latency_analysis` ve
`hyso.performance.throughput`, ikisinde schema_version=1. Bu version BenchmarkResult
schema1 değildir. UUID result_id, feature IDs, separate source/experiment identity,
versioned protocol/scope/configuration, model/input/device/runtime provenance,
requested/actual counts, unit/status/reason, raw evidence ve diagnostics zorunludur.
Eksik/bilinmeyen schema/invalid fields, nonplain/NaN/Infinity ve evidence ile
uyumsuz quantile/rate/count/sync/timestamp reddedilir. Throughput runtime device
configuration/input/environment ile tutarlı olmalıdır. Unsupported device/precision/
input validation'da explicit hata verir; runtime/measurement failure başarılı
unsupported result gibi gösterilmez ve sessiz CPU fallback yoktur.

Frozen/slots sonuç içinde yalnız immutable JSON string bulunur; to_dict defensive
copy'dir, model/tensor/event tutulmaz. to_json canonical tam kayıt, to_csv latency
için quantile, throughput için window satırlarıdır; raw samples JSON'da kalır.
UTF-8/allow_nan=False, no-clobber, atomic replacement/publication, temporary cleanup
ve CSV formula-text safety mevcut export primitive'lerini internal reuse ile
korur. Export helper/guard private reuse bağımlılığı legacy regression ile testlidir;
Benchmark koduna herhangi bir değişiklik yapılmadı.

## 6. State, RNG, lifetime ve C1

Mevcut Benchmark validation, registered-state Snapshot ve ortak non-reentrant
process lock internal reuse edildi; latency timer veya allocator probe kullanılmadı.
Parameter/buffer identity/values, gradient reference/values, mixed train/eval modes,
input, Python/NumPy/Torch CPU/selected CUDA RNG ve current stream/device restore
edilir. No optimizer/backward; outer inference/autocast scope'u korunur. Failure
yollarında mode/RNG/context/lock cleanup vardır; mutation successful result üretmez.
Güvenli ölçüde best-effort registered value rollback yapılır; tensor version counter,
structural/runtime/Python/external state için arbitrary transaction iddiası yoktur.

C1'in `_profile` closure-cycle fix'i byte-identical korunur. Yeni recursive
self-capturing closure yok; result, logger veya recorder model/tensor saklamaz.
GC disabled CPU success/rejected paths ile native model/weight/output/event weakref
lifetime testleri geçer. Native queue_depth=4 için live outputs≤4, live events≤5;
scope sonunda weakref'ler GC gerektirmeden ölür. Error paths queue referanslarını
finally bırakır. P09 allocator reset, peak counters, empty_cache ve global GC
politikasına müdahale yoktur; native reset spy reset çağrısında test fail ettirir.

## 7. Experiment Manager integration

[Çalıştırılabilir rehber](../guides/performance-inference.md) P99, sustained CPU,
explicit native CUDA function, JSON/CSV ve manuel E akışını içerir. CPU guide
bütün Python bloklarıyla hem source hem installed package'ta execute edildi.

Ölçüm sonrası explicit export → schema2 copy artifact association → sınırlı iki
scalar metric → verified artifact → finish → close/reopen/query/verify. Yeni store
schema1 kalır, mevcut offline migration explicit çağrılır; implicit migration,
new run kind veya SQLite schema değişikliği yoktur. kind=benchmark, axes mevcut
observation step=0; unit/source series sözleşmesi korunur. Run UUID/source benchmark
UUID/performance UUID ayrıdır. Raw samples metric_events'e çoğaltılmaz. Managed
artifact'ın yeniden açılan JSON'u üretilen ThroughputResult'a tam eşittir.

Window içinde file/database/network/hash çağrısı yasağı ayrı testle denetlendi.
Native mevcut Experiment integration guard'ları tracking-only forward/Event/sync/
allocator reset/RNG yasağını yeniden doğruladı. Yeni result association CPU/installed
guide'da doğrulandı; native eski E iki senaryosu bu yeni CPU guide testine eklenmez.

## 8. Yeni CPU testleri

`tests/test_performance_inference.py`: **122 PASS**. Known-array/P99/linear method,
insufficient policy, corrupt source/missing samples, requested/actual repeat
provenance, invalid protocol/device/precision/input, finite/immutable schema,
different batch/dtype cardinalities, exact clock/tail arithmetic, deterministic
duration admission, gerçek CPU duration, no-work/duplicate/negative ledger,
success/failure/state/RNG/gradient/mode safety, GC lifetime, JSON/CSV/formula safety,
atomic publish failure ve managed artifact reopen equality kapsanır.

İlk 109-test koşusu ardından schema/negative coverage genişletildi. Nihai 122 test
aşağıdaki tek targeted gate içindedir; ilk koşu toplama yeniden eklenmez.

## 9. Targeted legacy regression

Evidence kökü **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb1-20261020`.
`targeted-final.xml` / `targeted-final.log`: **1153 PASS, 0 FAIL/ERROR/SKIP/XFAIL/XPASS**,
pytest **250.34s**, 406 warning (log'da mesaj/kaynakları saklı). Test toleransı/assertion
gevşetilmedi, skip/xfail eklenmedi. JUnit classname/name identity'leri tekildir.

| Güncel targeted alt küme | PASS |
|---|---:|
| Yeni performance |122|
| Benchmark CPU85/protocol33/result24/state28/export66/comparison74 |310|
| Experiment core60/metrics59/storage28/artifacts36/migration11/recovery17/retry19/integration29 |259|
| Training checkpoint34/save135/restore179/transaction13/strict replay37 |398|
| Public API41/published API1/model state18 |60|
| Packaging |4|
| Toplam |1153|

Gerçek interpreter `C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`,
Python3.10.18/Torch2.5.1+cu121; CPU gate CUDA_VISIBLE_DEVICES=-1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, MPLBACKEND=Agg, `-B -m pytest -c tests/pytest.ini`.
Nihai argv/file list/identity inventory `E/verification-summary.json` içindedir.
Checkpoint/public API kapsamı full Trainer suite diye sunulmaz; full CPU koşulmadı.

## 10. Native CUDA evidence

Gerçek RTX3060 Laptop, capability8.6, Python3.10.18/Torch2.5.1+cu121/CUDA12.1;
native legacy report driver591.74/6144MiB kaydeder. Ayrı fresh interpreter'larda
sıralı, CUDA_VISIBLE_DEVICES=0, MPLBACKEND=Agg, primary native komutlarında
CUBLAS_WORKSPACE_CONFIG=:4096:8 kullanıldı. CPU CUDA-shaped mocks native sayılmaz.

| Güncel native script / kanıt | Farklı PASS senaryosu |
|---|---:|
| `tests/cuda_performance_validation.py` → native-strengthened.json/log |15|
| `tests/cuda_benchmark_validation.py` → native-benchmark.json/log |45|
| `tests/cuda_benchmark_export_validation.py` → native-export.json/log |8|
| `tests/cuda_experiment_integration_validation.py` → native-experiment/report.json |2|
| Hedefli distinct toplam |70|

Yeni 15 ID: FP32/AMP×default/non-default, FP32/AMP duration, outer inference,
bounded lifetime, preflight/warmup/measured exceptions, injected async terminal
failure, injected incomplete tail, native raw P99 ve mutation rollback. Son iki
fault-injection durumu native events/workload kullanır ama cihaz arızası yaratmaz.
Runtime/stream/count/timestamp/wait/lifetime ve outcomes raw JSON'dadır.

İlk native protokol koşusundan sonra result provenance/schema kontrolleri
güçlendirildi; final source koşusu başarılıdır. Sonrasında yalnız native testin
device-drain/event-weakref assert'leri güçlendirilerek source/installed 15'li tekrar
geçti; production source değişmedi. Aynı ID tekrarları distinct sayısına eklenmez.
Her script argv biçimi `python -B tests/<script> --output <E/path>`; installed
destekli script'ler ayrıca `--installed-target <E/installed-minimum>` aldı.

## 11. Torch2.2.2 minimum installed smoke

Mevcut temiz interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Python3.10.18, Torch2.2.2+cpu, torchvision0.17.2+cpu. Final wheel local/no-index/
no-deps/no-compile ile E/installed-minimum'a kuruldu; prefix/site-packages değiştirilmedi.
Smoke repo dışı cwd=E, `-I -B`, target import provenance ile çalıştı.

**PASS:** mevcut installed legacy/Trainer/checkpoint1/strict replay/Benchmark/E
smoke'una ek yeni namespace/typed results, gerçek CPU1000-sample P99, actual
80-image sustained throughput, JSON/CSV, E association/reopen guide. CUDA initialized
false; bu Torch2.2.2 native CUDA kanıtı değildir. Minimum environment bütün mevcut
distribution METADATA/RECORD hash'leri önce/sonra aynıdır; tüm dependency binary
dosyalarının ayrı full-content audit'i yapıldığı iddia edilmez.

`minimum-smoke.py/json`, minimum stdout/stderr, minimum-version stdout,
`minimum-environment.json`, `installed-commands.json` ve
`installed-source-integrity.json` kanıttır. Son dosya 24 installed production
modülünün source-byte equality/hash envanterini doğrular.

## 12. Wheel/sdist ve installed native

Targeted packaging fixture final production source'lardan bir wheel+sdist üretti;
gereksiz ikinci final build yoktur. Explicit inventory21→24 modül güncellendi,
source inventory equality guard kaldırılmadı. Wheel/sdist complete module inventory,
source-byte equality, metadata/dependency bounds, legacy/new exports, installed
imports ve tests/docs/cache/DB/weights exclusion dört packaging testiyle PASS.

| Artifact | Byte | SHA256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |98500|`046f062f75a384271d0536f53157ccc86d6c8f011d03cfddf15f307b190f3935`|
| hyso-0.2.0.tar.gz |93920|`f94ed08a5b03ea68606b1d6bae11b4a0bf0a02163d9ae139b465a14766d7bc96`|

Yollar `E/pytest-final/packaging0/dist`; wheel/sdist installed CPU smoke JSON'ları
aynı packaging0 altında. Aynı wheel target'ı native Torch2.5.1 runtime ile repo
dışı cwd'den ayrıca yeni performance15+Benchmark45+export8=**68 PASS tekrar** verdi.
Strengthened performance15 tekrarları bu68'e yeniden eklenmez. Installed imports
target altında doğrulanır; 70 distinct primary toplamına68 eklenmez.

## 13. Known limitations

Resident eager single-Tensor, fixed batch/input ve tek selected stream/concurrency
desteklidir; input iterator/decode/pre/post/camera/display/ORT/INT8/BF16/compiled/
distributed/cross-stream çalışma bu API'de yoktur. Unsupported profile açık
exception verir, missing değeri0 ile doldurmaz. Model/input transfer ve validation
pipeline throughput'u ölçülmez. Finite/queue/context instrumentation overhead'i
çıplak latency ile karıştırılmaz; clock/thermal/power kontrol edilmez.

Automatic driver/source-revision/thermal telemetry alanları null+not_collected
olarak explicit'tir; descriptor/weights/input hashes ve runtime/device bilgisi
vardır. Process/caller exclusivity assertion gerekir; cross-process isolation,
arbitrary Python/external side effects, structural mutation veya poisoned CUDA
context recovery garantisi verilmez. GPU hard watchdog yoktur; duration admission
sınırı terminal completion için hard timeout değildir. Windows/RTX3060 native
kanıtı Linux/başka accelerator desteği diye genellenmez. P99 tail eligibility
bağımsızlık/CI garantisi değildir.

## 14. P01/P02 backlog kararı

P01 **implemented**: ayrı versioned P99 contract/API, raw samples/linear math/tail
policy, real CPU/native source, provenance/export/installed integration doğrulandı.
P02 **implemented**: continuous CPU/CUDA completion-window, bounded ring/tail,
actual counts, state/RNG/native/installed/minimum/package ve E association geçti.
Statüler yalnız bu ilan edilen scope içindir, gerçek camera FPS veya başka
precision/backend completion iddiası değildir. P09 implemented, P08/P11/P29 partial
ve diğer29 planned durumu değişmez. Güncel toplam3 implemented/3 partial/29 planned.
Sabit roadmap v1.0 ve M2-P-A tarihsel statüleri değiştirilmedi.

## 15. Son karar ve freeze

**PASS.** 122 yeni test dahil1153 targeted PASS; native yeni15+legacy55;
minimum Torch2.2.2 CPU installed smoke; 24-module wheel/sdist ve installed native68
başarılı. Hiçbir gerekli kabul maddesi yalnız mock veya eski baseline'a dayanmadı.
İki fault-injection senaryosunun gerçek cihaz arızası kanıtı olmadığı açıkça ayrıldı.

`E/initial-tracked-hashes.json` ile final hash karşılaştırmasında yalnız backlog ve
packaging testi farklıdır; bütün diğer eski dosyalar, kullanıcı farkı dahil aynı
byte'lardadır. New production inventory/hash, JUnit ID/count ve native IDs,
package hashes/status counts `E/verification-summary.json` içinde. Index boş,
branch/HEAD aynı, `git diff --check` başarılı; commit/push yok. Yeni helper/build/
install/log/evidence çıktıları repository dışında E'dedir.

## 16. M2-P-B2 önkoşulları

**M2-P-B2'ye geçilebilir:** P10 native OS RSS ve P11 gerçek serialized footprint.
Yeni versioned result/protocol/export ve mevcut E association örüntüsü hazırdır.
B2, unique logical bytes ile disk artifact byte'ı ve sampled RSS ile OS HWM'yi
ayırmalı; Windows/Linux adapter kanıtını ayrı tutmalı. P09 reset/GC/lifetime,
P01/P02 state/scope, Benchmark schema1 ve checkpoint1 regresyonları korunmalıdır.
Bu görev B2 implementasyonu veya tüm35 özelliğin tamamlanması değildir.
