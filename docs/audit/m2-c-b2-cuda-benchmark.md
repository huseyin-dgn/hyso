# M2-C-B2 — Native CUDA Inference Benchmark & GPU Memory

Tarih: 2026-10-19. **Karar: PASS — M2-C-B2.**
**1551 farklı hedefli CPU testcase PASS; 45 farklı native CUDA testcase PASS.**
Torch 2.2.2 CPU installed-wheel FP32/FP64 **2/2 PASS**; final wheel/sdist
**19 production modülüyle PASS**. Aynı45 GPU testi installed wheel üzerinden
de geçti; ikinci kez toplam sayılmadı. Bu hedefli kapıdır; full-suite veya
M2-C-C final kalite kapısı iddiası değildir.

## 1. Git / HEAD / working tree

Branch `feature/hyso-models-next`; başlangıç/son HEAD
`a0672d5b22996f6dc2e6f26fd3063c76db8275f5`,
`feat: add CPU inference benchmark engine`. B1 commit edilmişti.
Devam oturumunda mevcut runner/result kısmi değişiklikleri korundu ve tamamlandı.
Başlangıçtaki kullanıcı dosyası `tests/test_gradient_accumulation.py` hiçbir
şekilde yazılmadı/stage edilmedi; başlangıç/son SHA-256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Index boş; commit/push/branch/reset/checkout/version/dependency değişikliği yok.

Kanıt kökü E=`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2cb2`.
`verification-summary.json` tekil kimlikleri, sınıf sayımlarını, freeze/hash,
backlog ve artifact doğrulamalarını içerir; `summarize.py` yeniden denetler.

## 2. B2 public API

```python
from hyso.benchmark import BenchmarkRunner, BenchmarkResult

# Caller modeli önceden CUDA:0 üzerinde FP32 olarak yerleştirir.
data = BenchmarkRunner(
    model, device="cuda:0", input_shape=(1, 3, 224, 224),
    precision="amp_fp16", warmup_iterations=10, measure_iterations=50,
    repeats=3, measure_cuda_memory=True,
).run().to_dict()
```

Constructor ve public export isimleri değişmedi. `input_shape` veya
`example_input` seçeneklerinden tam biri gerekir. CPU FP32/FP64 korunur;
CUDA FP32/amp_fp16 eklendi. Runner caller modelini taşımaz/dönüştürmez.
Result immutable plain schema_version=1 kalır; model/Tensor referansı taşımaz.
CUDA metadata, resident input/output, event method/scope/sync, memory/count ve
reset-warning tutarlılığı doğrulanır. JSON/CSV dosya export/comparison yoktur.

## 3. CUDA FP32/FP16 precision contract

İki CUDA profilinde de floating parameter/buffer ve input **FP32** olmalıdır.
FP32 native autocast disabled; amp_fp16 `torch.autocast(device_type="cuda",
dtype=torch.float16, enabled=True)` kullanır. Master weights `.half()` yapılmaz.
Output gerçek dtype ile kaydedilir; her operatorün FP16 olduğu varsayılmaz.
Conv/BN/Linear native AMP testinde output torch.float16 doğrulandı.
FP32/AMP için cuDNN benchmark/deterministic/TF32, matmul TF32 ve float32
matmul precision ayarları okunur; benchmark bunları değiştirmez.

CPU AMP, CUDA FP64/native half model, yanlış input dtype/device, mixed-device
model, unavailable index, sparse/meta/lazy/quantized ve önceki unsupported
hook/parallel/compiled/scripted profiller reddedilir. CPU fallback yoktur.

## 4. Event timing ve synchronization

Event çifti `enable_timing=True` ile warmup sonrasında, ilk sample öncesinde
yaratılıp record/end-sync ile lazy initialize edilir. Her repeat warmup'ından
sonra `torch.cuda.synchronize(selected_device)` vardır. Sample sırası:
`start.record(captured_stream) → model(input) → end.record(captured_stream)
→ end.synchronize() → start.elapsed_time(end)`.

GPU sample scope'unda perf_counter, input generation/transfer, state guard,
finite check, statistics veya event creation yoktur. Her elapsed ms finite ve
pozitif olmalı; zero/NaN/Inf/bool reddedilir, clamp/epsilon yoktur.

| Alan | CUDA değeri |
|---|---|
| timing_method | cuda_event_elapsed |
| timing_scope | serial_current_stream_forward |
| synchronization_policy | repeat_device_drain_and_end_event_sync |
| clock_resolution_ns | null; host clock çözünürlüğü GPU event'e atanmaz |

CPU doubles yalnız wiring/order kanıtıdır; gerçek event süreleri native GPU
scriptinde ölçüldü. Her sample end sync ile tamamlandığından sonraki forward
önceki output'u tutmaz; native weak-reference testi bunu ayrıca doğrular.

## 5. Stream/device policy

`cuda` explicit index'e resolve edilir. Runner seçili device context'inde o
device'ın caller current stream'ini yakalar; event ve forward aynı stream'dedir.
Non-default stream gerçek GPU'da çalıştı; forward current-stream assertion'ı,
result stream_id ve success/error sonrası stream/device restore doğrulandı.
Kalıcı stream/device değişimi context guard ile reddedilir; context manager'lar
finally kapanır. CPU protocol double farklı caller device7/target0 restorasyonunu
da sınar; bu iki fiziksel GPU testi değildir. Mevcut donanım tek GPU'dur.

Gizli başka-stream işi yapıp context'i geri alan forward güvenilir biçimde
tespit edilemez; **destek dışıdır**. Event sonucu bütün GPU işlerinin süresi
olarak sunulmaz. Başka GPU'ya iş gönderen forward da ilk profile dahil değildir.

## 6. Warmup/repeat/count

Normal count `1 + R*(W+N)`; memory açıkken ayrıca1 forward. Preflight1,
her repeat W warmup, N measured; event initialization forward yapmaz.
R×N nested samples, per-repeat/pooled mean, median/p50, p90/p95, min/max,
sample std ddof1 ve linear `(n-1)*q` quantile kuralları CPU ile aynıdır.
Tek sample std=null/insufficient_samples; warmup0 warning üretir.

Native W1/N3/R2 + memory örneğinde toplam10 forward, yalnız6 latency sample
doğrulandı. CPU double W2/N2/R2 için9/10 toplam ve4 sample'ı sınar.
`images_per_sec=B*(R*N)*1000/fsum(samples_ms)`; method
`derived_serial_service_rate`, source `cuda_event_elapsed`. Sustained throughput,
end-to-end latency veya kamera FPS iddiası değildir.

## 7. CUDA memory probe

Yalnız opt-in; latency loop'undan sonra aynı inference/autocast context'inde
**separate_single_forward** scope'u. Sıra: selected-device sync; baseline
allocated/reserved; önceki iki peak; reset; ayrı forward; output tutulurken
sync; peak ve current allocated/reserved; CPU output validation; output release;
sync; current allocated/reserved yeniden okuma. `empty_cache()` çağrılmaz.

| Alan | Anlam |
|---|---|
| baseline_allocated_bytes / baseline_reserved_bytes | Probe öncesi canlı allocation/cache |
| previous_peak_allocated_bytes / previous_peak_reserved_bytes | Reset öncesi process/device peak geçmişi |
| peak_allocated_bytes / peak_reserved_bytes | Ayrı probe sırasında mutlak allocator peak |
| incremental_peak_bytes | peak_allocated - baseline_allocated, negatif olamaz |
| current_allocated_after_probe_bytes / current_reserved_after_probe_bytes | Output tutulurken current değer |
| current_allocated_after_output_release_bytes / current_reserved_after_output_release_bytes | Output bırakılıp sync sonrası current değer |
| counter_reset_performed | measured:true, disabled:false |
| value | Measured peak_allocated_bytes; disabled:null |

Native allocator gerekir; cudaMallocAsync memory probe reddedilir. Timing yolu
bu allocator ölçümüyle eş anlamlı değildir. Allocated<=reserved, peak>=baseline/
gözlenen current ve incremental eşitliği schema guard'larıyla doğrulanır.

**Reset geri alınamaz:** process/device genelindeki peak sayaçları etkilenir.
Measured result'ta zorunlu `cuda_peak_reset` warning; reset flag, scope, önceki
peak değerleri ve caller/process exclusivity sınırı bulunur. Probe hatasında
da reset'in geri alınamadığı açık hataya eklenir. Başka thread/stream'in aynı
process allocation'ları sonucu kirletebilir. Process lock dış müdahaleyi engellemez.

CUDA memory disabled ise11 byte alanı ve value null, status=disabled, probe0,
reset0. CPU'nun mevcut not_applicable/value:null/reason şeması korunur.
Native spy kapalı modda reset yapılmadığını; eski peak geçmişinin silinmediğini
doğruladı. Snapshot değerleri CPU'da kalır; baseline büyük GPU snapshot içermez.
Baseline model/input, caller tensor'ları, workspace ve autocast cache içerebilir.
Sayaçlar modelin tek başına VRAM ihtiyacı veya bütün fiziksel VRAM değildir;
diğer process ve allocator dışı driver/library allocation'larını kapsamaz.

## 8. Model/RNG/state safety

B1 guard'ları CUDA'ya genişletildi: registered key/reference, parameter/buffer
byte değerleri, storage/alias/shape/stride/device/dtype, version, existing gradient
reference/değerleri, input ve per-module mixed train/eval flags. Parameter,
gradient ve input değer snapshot'ları CPU'dadır; success'te canlı tensor'a no-op
copy yapılmaz. Integer BN counter'ları değişmez. Tied parameter/storage view
topolojisi native GPU'da korundu. Caller custom input'u kopyalanır/değişmez.

Python/NumPy/Torch CPU ve **yalnız selected CUDA device** RNG get/set ile
korunur. all-device RNG API'leri kullanılmaz. Generated input local CPU
Generator ile üretilip timing öncesinde hedefe taşınır; repeatable hash ve global
RNG korunumu native testte doğrulandı. CPU run CUDA'yı initialize etmez.

Preflight/warmup/measured exception'larında modes, RNG, stream/device,
autocast/inference context ve process lock finally geri döner. Outer CUDA
autocast/inference context'leri ayrıca sınandı. Value mutation başarılı result
üretmez; StateMutationError ve güvenli ölçüde best-effort value restore vardır.
Storage/dtype/topology restore mümkün değilse fatal rollback error verilir.
Version counters, arbitrary Python/external state veya OOM/kernel/allocator
runtime etkileri için tam transaction/temiz runtime garantisi yoktur.
Optimizer/scheduler/GradScaler/backward/zero_grad/Trainer.fit/EBO.fit eklenmedi.

## 9. CPU protocol tests

`tests/test_benchmark_cuda_protocol.py`: **33 PASS**, hardware iddiası yoktur.
CPU Tensor + CUDA API double ile event initialization/record/end-sync/elapsed
sırası, stream binding, repeat drain, warmup/sample ayrımı, forward/probe count,
host clock yasağı, native-autocast kwargs, memory read/reset/output-release sırası,
disabled null/no-reset, invalid configs, selected-only RNG, error cleanup/lock
reuse ve measured/disabled/schema/reset-warning invariants doğrulandı.
GPU sonucu gibi raporlanmaz; native timing aşağıdaki bağımsız scriptin işidir.

## 10. Gerçek NVIDIA GPU senaryoları

`tests/cuda_benchmark_validation.py` standalone unittest; conftest'in CPU
visibility ayarını kullanmaz. **45/45 PASS, 0 failure/error/skip**.
GPU kapalı çalıştırıldığında setup error + exit1 verir; CPU fallback/skip yoktur
(`unavailable.json`/log; bu beklenen negatif kontrol PASS toplamına eklenmez).

Python3.10.18; Torch2.5.1+cu121; CUDA build12.1;
**NVIDIA GeForce RTX3060 Laptop GPU, 6144MiB, driver591.74**.
Native Conv/BN FP32/AMP, attention none/SE/CBAM/ECA/Coordinate × iki precision
× batch1/3 (20 case), non-default stream, generated/custom input, exact counts,
CPU snapshots, four-stage exception, value/topology/stream mutation ve negative
dtype/device/profile senaryoları çalıştı. Test kimlikleri `native.json` içindedir.

Temsilî küçük Conv/BN/Dropout/Linear model, input=(2,1,7,9), W1/N3/R2:

| Precision | Pooled event mean ms | Baseline allocated byte | Peak allocated byte | Incremental byte | Current held / released byte |
|---|---:|---:|---:|---:|---:|
| fp32 |0.281253333|42614784|43664384|1049600|42615296 / 42614784|
| amp_fp16 |0.358159999|8530432|9580032|1049600|8530944 / 8530432|

FP32 baseline/peak reserved121634816; AMP baseline/peak reserved23068672 byte.
Önceki peak'ler, bütün raw samples ve alanlar `native.json` içinde korunur.
Bu kısa correctness smoke performans üstünlüğü/accuracy-drop deneyi değildir;
test sırası, cache ve process baseline farklıdır, precision'ları memory/speed
winner diye karşılaştırmak için kullanılmaz. Sabit latency/FPS eşiği yoktur.

## 11. B1 CPU regression

Mevcut üç B1 dosyası ve assertion'ları değiştirilmedi; Git blob eşliği ve eski
JUnit kimlik altkümesi doğrulandı: CPU85 + Result24 + state24 = **133 PASS**.
FP32/FP64, beş attention, batch1/3, generated/custom input, outer inference,
hash/statistics/rate, invalid profiles, mutation ve state/RNG tests korundu.
Yeni33 protocol ile `benchmark.xml` toplam166 PASS.

## 12. Hedefli Trainer/legacy/M1 regression

CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
tests/pytest.ini; CPU test fixture tek Torch thread. Gerçek pytest/JUnit sayımı:

| Grup | Farklı PASS |
|---|---:|
| Sekiz Generic Trainer classification/update/validation/lifecycle/save/restore/transaction/replay dosyası |706|
| P0 training checkpoint / EMA / scheduler / OneCycle compatibility |141|
| M1 blocks / attention / ModularCNN |356|
| Model state / models / public API / published API |66|
| EBO resume / gradient accumulation / CV-dataset resume |112|
| Final packaging |4|
| B1 benchmark + yeni CPU CUDA protocol |166|
| **Tekil CPU toplam** |**1551**|

`regression.xml`1269, `legacy-resume.xml`112, `packaging-final.xml`4,
`benchmark.xml`166. (classname,name) kimlikleri tekildir; eski B1'in1425 kimliği
bu toplamın altkümesidir. Ara/repeat packaging veya GPU installed tekrarları
eklenmez. 0 failure/error/skip/xfail; mevcut deprecation/undefined metric
warning'ları legacy'den gelir. Full-suite/full CUDA regression iddiası yoktur.

Legacy16 production modülü, checkpoint v1, Trainer lifecycle, bütün models/
callbacks, root/model exports, metadata, sabit roadmap v1.0 ve önceki audit'ler
HEAD blob'larıyla aynıdır. Kullanıcı dosyasının28 testi geçti; dosya hash'i aynı.

## 13. Torch 2.2.2 installed-wheel

Mevcut temiz Python3.10.18 / Torch2.2.2+cpu / torchvision0.17.2+cpu ortamında
son yerel wheel `pip --isolated --no-index --no-deps --no-compile --target`
ile E/installed-minimum-final'a kuruldu. Dependency kurulmadı/değişmedi.
Repo dışından `-I -B` smoke; Runner/Result provenance target içinde,
**FP32/FP64 2/2 PASS**, finite R×N samples/statistics/rate ve CPU state/RNG
korunumu; CUDA initialized=false. `minimum-gate.py`, installed-minimum.json,
stdout/stderr ve clean-before/after.json kanıtları mevcut. Ortamın35 distribution
sürüm/METADATA/eski HYSo kaynak hash'leri önce/sonra aynı.

`minimum-cuda-api.json`: Event/record/synchronize/elapsed_time, device/stream,
selected RNG, allocator reads/reset/backend ve autocast signature inspection
uyumlu. **CPU build'de GPU event çalıştırıldı denmez**; native CUDA2.2.2 runtime
ayrıca denenmedi. Gerçek GPU kanıtı2.5.1+cu121 ortamındadır.

## 14. Wheel/sdist packaging

Gerçek filesystem ve MODULES envanteri **19**. Son kaynaklardan wheel/sdist
üretildi; arşivlerde19 modül için source-byte equality, tests/docs/cache/weights
exclusion, legacy public imports ve metadata koruması geçti. İki arşiv de
checkout dışında kuruldu; installed CPU FP32/FP64 benchmark smoke'ları geçti.
Final wheel'den native script ayrıca45/45 geçti, import target provenance doğru.

| Final artifact | Byte | SHA-256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |69313|65c602e7724071b3bcd1b656328e79a55830169ea2f38a646b9d13dc6fed241a|
| hyso-0.2.0.tar.gz |66825|afac686fbcdc9334d9961200481625ce64860a9436d824eea82ef381d659e53d|

Final artifact konumu `E/pytest-final/packaging0/dist/`; build.log,
wheel-smoke.json/sdist-smoke.json ve installed target'lar aynı packaging0 altında.
Önceki E/pytest-packaging ara build'i final artifact değildir.
HYSo0.2.0, Python>=3.10, Torch>=2.2.2, torchvision>=0.17.2 ve sekiz dependency
korunur. PyPI/TestPyPI/GitHub Release veya başka upload yapılmadı.

## 15. 35-feature backlog status

P01–P35 eksiksiz/tekil. Yalnız P01/P08/P09 satırları güncellendi;
diğer32 satırın metni HEAD ile aynı. **31 planned, 3 partial, 1 implemented.**
P01 CPU/CUDA latency var, P99 yok → partial. P08 CPU32/64 ve CUDA32/AMP16 var;
BF16/INT8/conversion cost/accuracy-drop yok → partial. P09 native allocator
baseline/peak/increment/release, opt-in/reset/no-reset sözleşmesi gerçek GPU'da
doğrulandı → implemented; total VRAM kapsamı değildir. P11 partial değişmedi.
Quality/sweeps/export/comparison/profiling/training/deployment uygulanmış sayılmaz.

## 16. Değişen dosyalar

- `hyso/benchmark/runner.py`: CUDA precision/device/stream/event, CPU snapshots,
  selected RNG, opt-in memory ve diagnostics; mevcut CPU protokolü korunur.
- `hyso/benchmark/result.py`: schema1 CUDA timing/device/count/memory doğrulaması.
- `hyso/benchmark/__init__.py`: yalnız CPU/CUDA docstring; exports aynı.
- `tests/test_benchmark_cuda_protocol.py`:33 CPU wiring/schema/error case.
- `tests/cuda_benchmark_validation.py`:45 standalone native GPU case.
- `docs/roadmap/hyso-performance-backlog-v1.1.md`: doğrulanmış B2 durumları.
- `docs/audit/m2-c-b2-cuda-benchmark.md`: bu rapor.

Git'teki gradient_accumulation farkı önceden mevcut kullanıcı değişikliğidir.
Yeni production modülü, dependency, metadata veya legacy değişikliği yoktur.
`git diff --check` başarılı; staged dosya yoktur.

## 17. Known limitations

Event elapsed stream timeline'dır; host wall-clock veya kernel sürelerinin
toplamı değildir. Host launch gaps/contention yansıyabilir. Per-sample sync
serial profili etkiler, deployment pipeline/concurrency throughput'u ölçmez.
Fixed warmup backend/autotune/clock stabilization garantisi değildir; validation,
CPU transfer/state checks zaman sınırı dışında olsa da cache/launch koşullarını
etkileyebilir. Clocks/thermals/power/background load milisaniyeleri değiştirir.

Hidden cross-stream/other-device iş unsupported; yalnız kalıcı context değişimi
yakalanabilir. Fiziksel multi-GPU restore testi yapılmadı. AMP mixed ops kullanır;
FP16 output bütün ops FP16 demek değildir. Accuracy karşılaştırması yoktur.
Memory reset process/device genelinde geri alınamaz. Allocator allocated/reserved
physical VRAM değildir; model attribution ve başka process'leri kapsamaz.
Native allocator dışı bellek probe desteklenmez. OOM, arbitrary Python/external
state ve CUDA context/cache etkileri için tam runtime rollback garantisi yoktur.
Minimum CPU API inspection native minimum CUDA testi yerine geçmez.

## 18. PASS / PARTIAL / FAIL kararı

**PASS — M2-C-B2.** Native FP32/FP16 event benchmark, selected current stream,
sync/count, opt-in allocated/reserved memory ve reset diagnostics, state/RNG
korunumu, bütün133 B1 testi, hedefli1551 CPU testcase,45 native CUDA testcase,
minimum2/2 installed smoke ve final19-module wheel/sdist kapıları geçti.
Backlog partial satırları B2'den geniş gelecek kapsamı gösterir; B2'yi
full-suite/M2-C-C veya bütün35 özelliğin tamamlanması olarak yorumlamamak gerekir.

## 19. M2-C-B3 önerileri

**B3'e geçilebilir.** JSON/CSV export ve koşul eşliğini denetleyen comparison
ayrı görev olmalı: CPU host / CUDA event scope karıştırılmamalı; device/runtime,
precision/TF32/backend, batch/shape/input, model/weights, repeat/warmup ve memory
probe policy farkları görünür olmalı. Missing/disabled metrics null+reason ile
korunmalı. Bu görevde B3 API veya ileri profiler/sweep/GUI geliştirilmedi.
Ardından M2-C-C full CPU ve bütün gerekli CUDA regression final gate'i uygulanır.
