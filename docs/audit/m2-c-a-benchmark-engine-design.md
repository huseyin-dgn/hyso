# M2-C-A — Benchmark Engine Architecture Design

Tarih: 2026-10-19. **Karar: M2-C-A DESIGN GO.**
Bu belge önerilen sözleşmedir; BenchmarkRunner henüz implement edilmemiştir.
Bu görevde benchmark, full pytest veya CUDA regression çalıştırılmadı.
Sonraki implementasyon görevi **M2-C-B1 — Minimal Inference Benchmark** olabilir.

## 1. Branch / HEAD / working tree

Branch `feature/hyso-models-next`; HEAD
`e64269e7156ca9b37fb9eaf78b75f2a5c5fa0190`,
`docs: record M2 generic trainer final quality gate`. B5 raporu commit edilmiş.
Başlangıçta yalnız `tests/test_gradient_accumulation.py` kullanıcı değişikliği
vardı; dosyaya yazılmadı/stage edilmedi/geri alınmadı. Başlangıç/son SHA-256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

Dayanaklar: [sabit roadmap](../roadmap/hyso-architecture-roadmap-v1.md),
[M2-A](m2-a-generic-trainer-design.md),
[M2-B5](m2-b5-final-generic-trainer-quality-gate.md).
B5'in1426 PASS/1 SKIP,107 farklı CUDA senaryosu, installed minimum ve paket
sonuçları **geçmiş kanıttır**; bu dokümantasyon görevinde yeniden üretilmedi.
Commit/push/branch/version/dependency/roadmap değişikliği yapılmadı.

## 2. Mevcut HYSo mimarisi

Gerçek kaynak incelemesi ve yeniden kullanım kararları:

| Kaynak | Bulgular ve tasarıma etkisi |
|---|---|
| hyso/models/modular.py | Standalone nn.Module; stem→registered residual stages→global pool→linear logits. forward_features NCHW doğrular; fit/device seçimi yok. Runner normal model(x) çağırır. Yedi constructor alanı descriptor olarak okunabilir. |
| hyso/models/blocks.py | ConvNormAct, residual/projection, batch/group/None norm ve registered attention. Parametreler forward'da yaratılmaz. NCHW ve standart parameter/buffer traversal uygundur. |
| hyso/models/attention.py | ECA Conv1d/pooling, Coordinate directional pooling ve kendi BatchNorm'u. Conv norm=None olsa bile bütün alt modüller eval yapılmalıdır. |
| hyso/training/trainer.py | Caller device/parameter ownership, native validation ve mixed-mode restoration örnekleri var. Trainer.fit optimizer/update/loss/EMA/ES sahibidir; inference benchmark bunu çağırmamalı. |
| hyso/training/resume.py | Plain finite descriptor, schema ve state/plan doğrulama yaklaşımı yararlı. Mevcut private helper'lar ModularCNN/checkpoint profile bağlıdır; yeni namespace bunları import etmeyecek. |
| setup.cfg / pyproject.toml | HYSo0.2.0; Python>=3.10, Torch>=2.2.2, torchvision>=0.17.2; setuptools find:hyso.*. Yeni dependency gerekmez. |
| tests/test_packaging.py | Gerçek16-module inventory, source-byte/metadata/exclusion ve isolated wheel/sdist smoke'ları mevcut. Gelecek benchmark modülleri gerçek envantere eklenecek. |

Bağımlılık yönü: `hyso.benchmark → torch + standard library`; mevcut NumPy
yalnız RNG koruması için kullanılabilir. EBO, Trainer, augmentation veya
checkpoint yükleme/kaydetme bağımlılığı yok. Model, eğitim kodu taşımadan
ölçülür. EBO wrapper'ını otomatik unwrap etme yok; caller uygun nn.Module'ü
(örneğin ELO/CEMO veya wrapper.model) kendisi seçer.

Önerilen küçük paket:

```text
hyso/benchmark/__init__.py   # yalnız BenchmarkRunner / BenchmarkResult exports
hyso/benchmark/runner.py     # validation, state guard, CPU/CUDA timing, memory
hyso/benchmark/result.py     # immutable result, statistics/schema; B3 export/comparison
```

B1'de üç modül yeterlidir; ihtiyaca dayanmayan registry/plugin/config framework,
model factory, ikinci Trainer veya root model export değişikliği yok.
Gelecekteki ResNet/MobileNet/GhostNet uygun nn.Module forward sözleşmesini
sağladığında mimari ismine göre dispatch gerektirmeden ölçülebilir.

## 3. Benchmark Engine kapsamı

İlk sürüm: tek process, tek CPU/CUDA device, eager dense NCHW CNN inference,
tek positional Tensor giriş/tek dense floating Tensor çıkış. Çıkışın batch
boyutu girişle aynı; feature tensor veya classification logits olabilir.
Dynamic shapes, arbitrary dict/list/tuple I/O, targets/DataLoader, DDP/DataParallel,
compiled/scripted/CUDA graph, quantized/sparse/meta/lazy/uninitialized modeller
ve custom stream kullanan forward ilk profile dahil değildir; açık hata verir.
Custom Python model desteklenmesi arbitrary mutable/external state'in güvenli
olduğunun iddiası değildir (§8).

| Metrik | İlk karar |
|---|---|
| Latency samples, mean/median=p50, p90/p95, sample stddev | CPU wall-clock veya CUDA stream event; yöntemle birlikte raporlanır |
| Throughput images/sec | İlk sürüm latency'den türetilmiş serial service-rate; sustained/end-to-end ölçüm diye sunulmaz |
| Total/trainable parameter count | Unique Parameter nesnelerinin numel toplamı |
| Parameter/buffer footprint | Logical tensor byte sayısı; resident/disk model size değildir |
| CUDA allocated/reserved/peak | B2'de ayrı opt-in memory probe; process allocator ölçümü |
| Device/runtime/config/repeat/provenance | Zorunlu/explicit unknown alanlarıyla |
| Accuracy | İlk engine dışında; labels/dataset/task protocol ile ayrı değerlendirme gerekli |
| FLOPs/MACs | Desteklenmiyor. Operator kapsamı ve counting convention olmadan sayı yok; bir MAC her bağlamda bir FLOP değildir |
| Serialized disk size | İlk kapsam dışında. İleride yalnız gerçekten yazılan belirli artifact'ın byte sayısı; weights vs full checkpoint ayrılmalı |
| CPU RSS | İlk kapsamda desteklenmiyor; portable/dependency-free güvenilir process RSS yöntemi seçilmedi |
| Training-step timing | Sonraki ayrı tasarım/alt görev; ilk inference engine'in parçası değil |

tracemalloc toplam PyTorch CPU tensor/RSS ölçümü değildir; CUDA allocated,
reserved ve nvidia-smi process/device VRAM aynı metrik değildir.

## 4. Public API önerisi

**BenchmarkRunner** seçildi: aynı doğrulanmış konfigürasyonun run() ile yeniden
çalışması ve immutable sonuç üretimi açık olur. Ek BenchmarkConfig zorunlu değil.
Aşağıdaki kod **gelecek API örneğidir**, bugün çalıştığı iddia edilmez:

```python
from hyso.benchmark import BenchmarkRunner

# Model caller tarafından önceden doğru device/dtype'a yerleştirilir.
runner = BenchmarkRunner(
    model=model,
    device="cuda:0",
    input_shape=(1, 3, 224, 224),
    precision="fp32",
    warmup_iterations=10,
    measure_iterations=50,
    repeats=3,
    seed=0,
    measure_cuda_memory=False,
)
result = runner.run()             # BenchmarkResult; I/O yapmaz
plain = result.to_dict()          # güvenli plain veri
```

Constructor: `model` positional; diğerleri keyword-only. Device default cpu;
örnekteki değerler generic defaults olup warmup yeterliliği garantisi değildir.
`model_name=None`, `model_config=None` isteğe bağlı plain descriptor alanlarıdır.
`input_shape=None` ve `example_input=None` seçeneklerinden **tam biri** verilir.
Model device'ı parametre/buffer'lardan tahmin edilip taşınmaz; CUDA ismi explicit
index'e resolve edilerek sonuçta kaydedilir. Ulaşılamayan device fallback yapmaz.

Validation: model nn.Module; tek cihazda initialized dense parametre/buffer'lar;
N,C,H,W pozitif Python int (bool kabul edilmez). Warmup>=0, measured>=1,
repeats>=1; sayılar exact int, seed geçerli nonnegative int. Batch size yalnız
resolved input_shape[0]'dan gelir; ikinci, çelişebilecek batch_size argümanı yok.
State/model finite kontrolleri ve untimed output preflight vardır.
Parameterless model cihazı input/device ile belirlenir; integer buffer olabilir.

| Precision | Desteklenen ilk profil |
|---|---|
| fp32 | CPU/CUDA: floating model parameters/buffers ve input FP32; autocast explicit disabled |
| fp64 | B1 CPU: floating model/input FP64; autocast disabled |
| amp_fp16 | B2 CUDA: FP32 model/master weights/input; native torch.autocast('cuda', dtype=float16) |

CPU FP16/BF16, native half model ve CUDA FP64 ilk profilde reddedilir.
AMP bütün output/operasyonların half olacağını garanti etmez; gerçek output
dtype kaydedilir. TF32/backend ayarları ayrıca kaydedilir; fp32 etiketi onları
gizlemez. Caller'ın dış autocast/grad context'i context manager ile geri döner.
Runner model.to/.cuda/.half çağırmaz, optimizer/GradScaler oluşturmaz.

Generated input: local CPU torch.Generator(seed), uygun dtype ile randn,
sonra engine'in kendi input'u seçili device'a taşınır; üretim/transfer timing
dışındadır. Bütün iterasyonlarda aynı resident tensor tekrar kullanılır;
profil `fixed_resident_input` ve input hash/shape/stride/layout ile kaydedilir.
Global RNG seed edilmez. Cross-Torch random bitwise eşlik garantisi yoktur.

Custom example_input: plain dense contiguous finite floating Tensor, NCHW,
requires_grad=False, zaten doğru device/dtype. Sessiz cast/transfer/normalize
yok; runner kendi detach+clone kopyasını timing dışında oluşturur. Caller
input'u değişmez. Kopyada forward mutation tespit edilirse profil reddedilir.
Custom generators, input factories ve multi-input adaptörleri ilk API'de yok.

Model descriptor güvenli class qualified-name, registered tensor şeması,
opsiyonel model_config ve source/weights fingerprint'tir. ModularCNN'nin
gerçek alanları okunabilir; arbitrary object.__dict__/repr/pickle serialize
edilmez. Model_config annotation'dır; arbitrary mimari semantiğinin kanıtı
değildir. Kaynağı okunamayan modelin hash'i null/explicit reason olur.

## 5. Timing metodolojisi

**CPU:** inference/autocast context'i loop dışında açılır. Her örnek için
`t0=perf_counter_ns(); output=model(x); t1=perf_counter_ns()`;
`ms=(t1-t0)/1_000_000`. Bu synchronous Python→forward→return wall-clock scope'udur;
Python dispatch dahildir. Input generation/copy, correctness checks, snapshot,
output export, warmup ve istatistikler dahil değildir. Clock resolution metadata'dır.

**CUDA:** CPU perf_counter ile asenkron forward GPU latency olarak raporlanmaz.
Seçili `torch.cuda.device(device)` ve o device'ın captured current stream'i
kullanılır; context dışına çıkış önceki current device/stream'i korur.
Multi-stream forward ve başka stream'e iş kaçıran custom operator profile dışıdır.

1. Input/model resident; correctness forward ve warmup tamamlanır.
2. Timing event çifti `enable_timing=True` ile önceden yaratılır/lazy initialize
   edilir; event yaratma maliyeti örneklerin parçası olmaz.
3. Repeat sınırında seçili device synchronize edilir; eski queued iş dışarıdadır.
4. Her örnekte aynı stream'e start.record(stream), model(x), end.record(stream).
5. end.synchronize() sonrası start.elapsed_time(end) **ms** kaydedilir.
   Senkronizasyon host bekleme maliyeti event interval'in dışında kalır.
6. Output bırakılır; sonraki forward önceki output'u tutarak ekstra peak üretmez.

Event aralığı **stream timeline elapsed** süresidir; saf kernel süreleri toplamı
değildir. Host launch boşlukları veya contention interval'e yansıyabilir;
host call/dispatch/terminal synchronize wall maliyetiyle aynı şey değildir.
Her örneğin sync edilmesi serial isolated-service profilidir; async deployment
pipeline'ı veya concurrency throughput'u değildir. Ölçüm sırasında .item(),
CPU transferi, profiler veya tensor finite-reduction timing scope'unda çalışmaz.

Bir untimed correctness forward output type/device/shape/finite kontrolü yapar;
son ölçüm output'u da scope dışında kontrol edilir. Her ölçüm finite ve >0
olmalıdır. Zero/resolution altı sonuçlar epsilon'a yuvarlanmaz; açık measurement
error verilir ve daha büyük workload/batch önerilir. Kötü örnekler sessizce atılmaz.

## 6. Warmup, repeats ve throughput

Her repeat'te W warmup + N measured forward; R repeat. Bir ayrı preflight
forward vardır, memory opt-in ise ayrıca bir memory forward vardır.
Toplam forward count `1 + R*(W+N) + memory_probe_count`; bütün sayılar sonuçta
kayıtlıdır. Preflight/warmup/memory forward latency samples'a eklenmez.
Warmup=0 geçerlidir ama cold/steady-state karışımı uyarısı zorunludur.
İlk sürüm fixed warmup kullanır; otomatik convergence/clock stabilization iddiası yok.

Samples R×N nested list olarak repeat sınırlarıyla korunur. Her repeat ve pooled
özet: mean, median/p50, p90, p95, min/max; sample stddev ddof=1 (n<2 ise null,
insufficient_samples). Quantile sorted samples'da `(n-1)*q` konumunda linear
interpolation'dır; algoritma şemada sabittir. Outlier trim/winsorization yok.
Repeat means/range ayrıca verilir; autocorrelated iterasyonlar bağımsız örnek
sayılıp güven aralığı veya istatistiksel üstünlük iddiası üretilmez.

İlk throughput: `B * (R*N) * 1000 / sum(all_latency_samples_ms)`.
`method=derived_serial_service_rate`, unit=images/sec ve latency timing basis'i
zorunludur. CPU'da forward wall service-rate; CUDA'da event service-rate'tir.
`mean(B/time)` veya p50'nin tersine çevrilmesi kullanılmaz; pooled ve per-repeat
rate'ler ayrı hesaplanır. Bu, measured-window sustained throughput değildir.

Uzun pencere yöntemi değerlendirmesi: K forward'ı çevreleyen monotonic host
timer + CUDA baş/son sync, `B*K/window_seconds`; Python dispatch/output release
ve end wait overhead'i içerir, per-iteration sync içermez. Farklı scope olduğu
için ilk sürümde **deferred/null** kalır. Sonradan ayrı measurement mode ve
versioned timing policy ile eklenir; derived rate alanının anlamı değiştirilmez.

Thread/interop thread sayısı, seed, input hash, device, precision/autocast,
backend flags, cuDNN version/config, TF32, float32 matmul precision, deterministic
algorithms/warn_only ve sample protocol kaydedilir. Runner process thread veya
backend flags'ini değiştirmez; thread sweep ayrı fresh process'lerde caller işidir.
CPU/GPU clock, sıcaklık, OS scheduler, arka plan yükü ve güç politikası latency'yi
etkiler; seed ve tekrarlar aynı milisaniyenin yeniden çıkmasını garanti etmez.

## 7. CPU / CUDA ayrımı

| Politika | CPU | CUDA |
|---|---|---|
| timing_method | perf_counter_ns | cuda_event_elapsed |
| timing_scope | synchronous_host_forward | serial_current_stream_forward |
| sync | none | repeat device drain + end-event sync/sample |
| memory | logical footprint; RSS unsupported | logical footprint + opt-in allocator probe |
| precision | fp32/fp64 | fp32/amp_fp16 |
| transfer/dataset/I/O | excluded | excluded; inputs already resident |
| throughput | derived host service-rate | derived event service-rate |

CPU wall ve GPU event sonuçları aynı koşul sayılmaz; generic latency_ms sütunu
yöntemsiz export edilmez. CPU result GPU metadata'ya0 yazmaz: not_applicable/null.
CUDNN autotune seçilirse cost warmup'ta kalabilir; caller ayarı kaydedilir,
ilk engine otomatik tuning veya deterministic flag değişikliği yapmaz.
CPU-only ortam CUDA isteğini veya AMP'yi CPU fallback ile karşılamaz.

## 8. Model/mode/state güvenliği

Caller sahipliği seçildi: canlı modelin cihazı/dtype'ı, Parameter identity,
optimizer bağlantıları ve mevcut gradients korunur. Otomatik clone model veya
sessiz .half/.cuda yok; büyük model copy'nin OOM ve allocator baseline etkisi
gizlenmez. Runner aynı process'te reentrant/concurrent run'ı reddeder; caller
aynı modeli/device'ı başka thread/stream'de değiştirmemelidir.

Supported normal inference akışı:

1. Preflight'ta initialized tensor topology/alias/reference şeması, per-submodule
   training flag'leri, registered parametre/buffer değerleri ve mevcut gradients
   timing dışında snapshot edilir; tensor değer snapshot'ları **CPU'da** tutulur.
2. Python/NumPy/Torch CPU ve seçili initialized CUDA device RNG snapshot edilir.
   CPU run CUDA context'i başlatmaz; generated input local generator kullanır.
3. Native train/eval semantiği olan bütün alt modüller eval yapılır; BN stats
   ve Dropout inference davranışı böyle korunur. Custom train/eval overrides,
   side-effect hooks ve mutable custom forward ilk safe profile dahil değildir.
4. Bütün forward'lar `torch.inference_mode()` ve explicit precision context'inde;
   optimizer/loss/backward/zero_grad/EMA/scheduler yok. Önceden var olan grads
   temizlenmez; yeni gradient üretilmez. Output tensor'ları result'a taşınmaz.
5. Preflight sonrası ve run sonunda registered state/topology/input invariants
   kontrol edilir; timing/peak scope'ları dışında yapılır.
6. `try/finally` ile her module.training bayrağı **tek tek** geri atanır. Yalnız
   root.train(previous) nested mixed-mode'u yanlış düzeltir; kullanılmaz.
   RNG ve context'ler success/exception durumunda geri yüklenir.

Native pure CNN forward'larında parametre/buffer/gradient byte'ları aynı kalmalı;
no-op restore için canlı tensor'a gereksiz copy_ yapılmaz. Value mutation
tespitinde sonuç üretilmez; original reference'lara timing dışında no_grad copy
ile best-effort rollback yapılır ve açık StateMutationError yükseltilir.
Rollback failure ayrıca fatal bildirilir. Yapısal mutation, unregistered Python
cache/external I/O veya arbitrary operator yan etkileri için transaction
garantisi yoktur. Autograd version counters/key identity gibi etkiler arbitrary
in-place mutation sonrası bütünüyle geri alınamaz; böyle bir model desteklenen
safe inference profile kabul edilmez ve caller runtime'ı kullanılabilir diye
raporlanmaz. Bu güvenlik sınırı custom nn.Module desteğiyle karıştırılmamalıdır.

CUDA context/kernel/allocator warmup cache'leri inference'ın süreç etkisidir;
bütün runtime'ı bitwise geri alma garantisi verilmez. Device RNG guard tek
hedefli profildir; başka cihazı kullanan forward destek dışıdır. Context/snapshot
restore hata verirse başarılı BenchmarkResult dönmez; export çalışma dışında kalır.

## 9. Bellek ve parametre ölçümü

`parameters_total=sum(p.numel() for p in unique_parameters)`;
`parameters_trainable=sum(p.numel() for p in unique_parameters if p.requires_grad)`.
Traversal remove_duplicate=True. Aynı Parameter nesnesi
birden çok yerde registered ise bir kez sayılır; distinct tied views farklı
logical parametreler olabilir. Trainable=false olsa bile total'a dahildir.

Parameter/buffer byte metriği `numel()*element_size()` ile unique Tensor object
bazında logical footprint'tir. Parametre ve buffer alt toplamları, cross-category
object alias'ları ve combined deduplicated toplam açık tutulur. Distinct tensor
views aynı storage'ı paylaşırsa logical toplam physical storage kadar olmak
zorunda değildir; alias warning/metrik tanımı kaydedilir. İlk sürüm storage-
resident/disk footprint iddiası yapmaz. Gradients, optimizer, activations,
allocator fragmentation, Python object ve CUDA context dahil değildir.

CUDA memory **measure_cuda_memory=True** ile B2'de açılır; CPU'da bu istek
preflight hatasıdır. Model ve owned input resident, warmup/timing tamamlanmış,
latency output'ları bırakılmış ve hedef device synchronize edilmiş olmalıdır.
State guard snapshot'ları CPU'dadır; büyük GPU state clone'u baseline'a eklenmez.

Memory scope'u ayrı tek inference probe'dur:

- Baseline memory_allocated/memory_reserved ve önceki peak okunur.
- reset_peak_memory_stats(device) çağrılır; probe forward inference/autocast ile
  çalışır; output tutulurken device sync ve peak/current allocated/reserved okunur.
- `peak_allocated_bytes` mutlak process allocator peak; `incremental_peak_bytes
  = peak_allocated_bytes-baseline_allocated_bytes`; output dahil gözlenen artıştır.
- Output bırakılıp sync sonrası current allocated ayrıca kaydedilir. Empty-cache
  çağrısı yok; reserved-cache davranışı gizlenmez. Peak reserved varsa kendi adıyla
  kaydedilir; allocated ile eşit kabul edilmez.

Peak modelin tek başına VRAM ihtiyacı değildir: baseline mevcut model/input,
caller'ın diğer tensor'ları ve persistent allocations'ı içerir. Probe artışı
first-run/global peak veya model size değildir; cached workspace/reuse etkileri
vardır. Bu sayaçlar aynı process'in PyTorch allocator'ını görür; diğer GPU
process'leri veya bütün driver/library allocation'larını kapsamaz. Başka process
load'u available memory/latency'yi etkileyebilir, counter'a doğrudan eklenmez.

Peak reset process/device genelinde yan etkidir ve eski peak değerini public
API ile geri yazma garantisi yoktur. Bu yüzden opt-in, previous_peak metadata ve
`counter_reset_performed=true` warning zorunludur; mode/RNG restoration gibi
tam geri alma iddiası kurulmaz. Concurrent same-process memory kullanımı sonucu
kirletir; caller exclusivity sözleşmesi ve ayrı fresh-process ölçümü önerilir.

## 10. Result schema

`BenchmarkResult` immutable/frozen snapshot; public model/Tensor referansı yok.
`to_dict()` defensive copy, yalnız dict/list/string/bool/int/finite float/null.
İlk schema_version=1, benchmark_type=inference; checkpoint FORMAT_VERSION ile
ilgisi yoktur. Successful run zorunlu ölçümler eksikse result üretmez; warning
desteklenmeyen optional metriği başarılı ölçülmüş gibi göstermez.

| Grup | Zorunlu alanlar / tip ve anlam |
|---|---|
| identity | schema_version:int, benchmark_type:str, run_id:str, started_at_utc/finished_at_utc:ISO8601; timestamp latency saati değildir |
| model | model_name, qualified_class:str; plain model_config veya null; descriptor_version, tensor_schema, source_sha256/weights_sha256 ve unavailable nedenleri |
| input | shape:list[int], strides:list[int], layout, batch_size, dtype, source:generated/custom_clone, seed:int/null, content_sha256; fixed_resident_input |
| configuration | warmup_iterations, measure_iterations, repeats, preflight_forward_count, memory_probe_count; precision/autocast; forced_model_mode=eval; protocol_version |
| environment | python/torch/torchvision/hyso version; OS/CPU identity/count, intra/inter-op threads; resolved device/index; GPU name/capability/total_memory veya null; CUDA build/cuDNN; backend flags/TF32/determinism; source revision supplied/known ise |
| provenance | source-tree/installed origin, available source hashes; optional caller environment label. Driver/clock/temperature/power bilgisi güvenli API yoksa not_collected; NVML/shell dependency varsayılmaz |
| timing | timing_method, timing_scope, synchronization_policy, clock_resolution_ns veya null; latency_samples_ms:list[list[float]]; per_repeat ve pooled stats |
| stats | count, mean_ms, median_ms/p50_ms, p90_ms, p95_ms, std_ms, min_ms, max_ms; quantile_method, std_ddof; std n<2 için null+reason |
| throughput | images_per_sec, method=derived_serial_service_rate, source_timing_method; per_repeat rates; window_wall_images_per_sec=null/deferred |
| parameters | total:int, trainable:int, parameter_bytes:int, buffer_bytes:int, parameter_buffer_bytes:int; byte_accounting=unique_tensor_object_logical; alias notes |
| cuda_memory | status, baseline_allocated/reserved, previous_peak, peak_allocated, incremental_peak, current_after_probe/output_release, reserved/peak_reserved: int/null; scope=separate_single_forward; counter_reset_performed |
| optional_metrics | accuracy, flops, macs, disk_artifact_bytes, cpu_rss_bytes: {status, value:null, reason}; sahte0 yok |
| diagnostics | warnings:list[{code,message}], limitations:list[str], state_guard policy; unknown metadata'nın nedenleri |

Optional status değerleri: measured, not_applicable, disabled, unsupported,
not_collected, insufficient_samples. Örnek **schema fragment'i**, gerçek sonuç değil:

```json
{
  "schema_version": 1,
  "benchmark_type": "inference",
  "cuda_memory": {"status": "not_applicable", "peak_allocated_bytes": null},
  "optional_metrics": {
    "cpu_rss_bytes": {"status": "unsupported", "value": null, "reason": "no_validated_portable_backend"}
  }
}
```

to_dict B1'de gerekir; B3'te explicit to_json(path) ve CSV exporter eklenir.
JSON allow_nan=False, unknown object/string fallback yok. Plain descriptors
key/type/finite/size sınırlarıyla doğrulanır; arbitrary model/checkpoint pickle
yoktur. Export run() içinde otomatik I/O yapmaz. CSV kararı: B3'te standart csv
ile summary row; raw samples gerekiyorsa ayrı long-format CSV (run/repeat/iteration/
ms). Nested veriler kayıpla sessizce flatten edilmez; JSON canonical full result'tır.
Timestamp/run_id farklı ölçümleri ayırır; provenance aynı koşulların kanıtıdır,
seed/timestamp aynı latency veya aynı accuracy garantisi değildir.

## 11. Model comparison kuralları

B3 `compare_results(a,b)` plain compatibility report döndürür:
compatible / incompatible / insufficient_metadata, mismatched_fields ve warnings.
Leaderboard, otomatik quality ranking veya istatistiksel significance testi yok.

Karşılaştırma anahtarı: schema/protocol, benchmark_type, resolved input shape/
batch/layout/dtype/content policy+hash, device/hardware identity, precision/
autocast/TF32, timing method/scope/sync, W/N/R, forced eval/inference policy,
thread/interop settings, Torch/CUDA/cuDNN/OS environment ve backend flags.
CPU↔CUDA event/wall veya throughput method farkı doğrudan incompatible'dır.
Farklı makine/unknown hardware kimliği sessiz eşdeğer sayılmaz; gerekli alan
bilinmiyorsa insufficient_metadata olur. Device index tek başına physical GPU
kimliği kanıtı değildir; GPU name/capability ve environment provenance gerekir.

Model class/config/parameter count/weights hash aynı olmak zorunda değildir;
bunlar karşılaştırılan değişkenlerdir. Input/precision/runtime koşulları eşleşse
bile model output semantiği ve accuracy aynı kabul edilmez. Aynı model için
regression analizi yapılacaksa weights/config hash eşliği ayrıca aranır.
Memory probe ve baseline/persistent allocation farkları memory kıyasına ayrı
uyarı üretir; process absolute peak'leri model-only ranking'e çevrilmez.

## 12. Torch 2.2.2 uyumluluğu

Mevcut clean Python3.10.18/Torch2.2.2+cpu ortamında yalnız import/inspect ve yerel
API docstring kontrolü yapıldı; eğitim/forward/timing benchmark çalıştırılmadı.
CUDA görünürlüğü kapalıydı ve CUDA initialized=False kaldı. İncelenen API'ler:

- torch.inference_mode(mode=True), torch.autocast(device_type,dtype,enabled,
  cache_enabled), torch.Generator; Module.named_parameters/named_buffers
  remove_duplicate=True ve Tensor.untyped_storage erişimi.
- time.perf_counter: monotonic=True, adjustable=False; bu Windows ortamında
  QueryPerformanceCounter, resolution1e-7s. Bu resolution benchmark latency değildir.
- torch.cuda.Event(enable_timing=False,blocking=False,interprocess=False),
  record(stream), elapsed_time(end_event), synchronize; current_stream/device/
  stream context ve device-specific synchronize/memory/reset fonksiyon imzaları.
- get_float32_matmul_precision; memory_allocated docstring'i tensor allocator
  byte'larını ve nvidia-smi farkını, reset_peak docstring'i global peak reset'i
  açıklar; elapsed_time birimi ms'dir.

CPU build'de CUDA sınıf/fonksiyonunun import edilebilmesi **CUDA çalıştığının
kanıtı değildir**. Event/stream/memory/FP16 gerçek doğrulaması B2'nin native GPU
gate'ine aittir. Minimum API temeli yeni torch.compile/CPU AMP/özel profiler
API'sine dayanmaz. Sürüm string'ine göre sessiz fallback yerine capability
validation ve desteklenen iki runtime'da installed test gerekir. Yeni dependency yok.

## 13. B1/B2/B3 implementasyon planı

| Alt görev | Çıktı | Kabul kapısı |
|---|---|---|
| M2-C-B1 Minimal Inference Benchmark | Üç küçük modül, proposed API, CPU fp32/fp64, validation/local input, perf_counter_ns, W/N/R/stats/derived rate, state guard, logical counts, schema1/to_dict | Deterministic protocol/statistics tests; native CPU nn.Module+ModularCNN; exception/mixed-mode/RNG/grad preservation; unchanged training/legacy tests |
| M2-C-B2 Native CUDA Benchmark | Event/stream/device/sync scope, native amp_fp16, opt-in separate memory probe, metadata/errors | CUDA yoksa FAIL veren bağımsız gerçek GPU script'i; FP32/AMP, non-default caller stream, iki precision/device hataları, memory counter semantics, state safety; CPU regressions |
| M2-C-B3 Result Export and Comparison | JSON/summary+optional long CSV, schema validation, compatibility report, provenance ve installed smoke | JSON finite/plain round-trip, incompatible timing/precision/input/runtime rejection, unknown metadata policy, wheel/sdist imported runner/result smoke |
| M2-C-C Final Benchmark Quality Gate | Yeni full CPU, bütün native GPU regressions, benchmark smoke, minimum2.2.2 installed CPU, wheel/sdist/freeze/identity audit | Yeni gerçek sonuçlarla GO/PARTIAL/NO-GO; geçmiş Trainer/B5 PASS'lerini yeni benchmark başarısı saymama |

Training-step ölçümü M2 içindeki **ayrı sonraki destek kararı/alt göreve** bırakılır;
roadmap dışında faz yaratılmaz ve inference B1/B2 kapsamına sızmaz. Gerekirse
ileride caller-owned loss/optimizer/scaler ile ayrı function/protocol;
forward, loss, backward, optimizer step, zero_grad ve full-step sınırları açık
olmalı. Training doğal olarak ağırlık/optimizer/RNG değiştirir; replica/reset
ve applied-vs-overflow politikası ayrıca tasarlanmalıdır. Trainer.fit çağırarak
bir epoch süresine training-step latency adı verilmez. Şu an bu API yoktur.

## 14. CPU/GPU test matrisi

| Alan | Kararlı doğrulama |
|---|---|
| Native CPU | Linear/small Conv+BN+Dropout ve5 ModularCNN attention; fp32/fp64; NCHW batch1/>1, odd rectangular; finite positive timings ve R×N sample |
| Protocol | Test-only controlled clock/events ile doğru order/call counts; preflight1, R*(W+N); warmup/probe/snapshot/transfer hariç; gerçek native forwards da ayrı smoke |
| Statistics | Known sample list ile linear quantiles/ddof1, n=1 missing std, doğru pooled sum throughput; NaN/Inf/0 rejection |
| Safety | Nested mixed flags, BN counters/weights/grads/reference identity/RNG önce=sonra; preflight/warmup/measured exception; no optimizer/backward; mutable model açık error/no result |
| Inputs | Generated local generator global RNG'yi tüketmez; custom clone/original unchanged; invalid bool/count/shape/dtype/device/output/hook/profile preflight red |
| Native CUDA | FP32/amp_fp16, real events, synchronization, correct explicit device/current stream; model master dtype/grad/BN korunur; finite positive ms, correct count |
| CUDA memory | Opt-in reset/order/baseline/probe/output release, allocated/reserved ayrı, peak>=baseline ve increment tanımı; disabled/not-applicable/null; OOM/error cleanup |
| Export/comparison | Pure JSON/CSV, null+reason, no pickle, schema rejection, fieldwise mismatch/insufficient metadata, statistics equivalence after round-trip |
| Runtime/package |2.5.1 checkout +2.2.2 installed CPU; future2.2.2 CUDA yalnız gerçek GPU varsa; exact module inventory/source bytes/metadata/legacy API |

Unit testlerde sabit ms eşiği, GPU'nun CPU'dan hızlı olması, p95/throughput
üstünlüğü veya magic VRAM sayısı beklenmez. Timer/event test doubles sadece
protocol wiring içindir; native CUDA kanıtının yerine geçmez. CUDA CI unavailable
ise ayrıca açık platform durumu; sahte PASS/CPU fallback yok. State guard cost,
output checks ve per-iteration sync'in scope'u testlerle sabitlenir.

## 15. Packaging/legacy compatibility

Bugün16 production modülü aynıdır; yalnız tasarım belgesi eklenir. B1'de önerilen
üç modül gerçekten yaratılırsa MODULES gerçek dosya envanterine göre güncellenir
(mevcut16+3); B3 mevcut result.py'de export/comparison ekleyebilir. Gereksiz
ayrı export/comparison modülü veya dependency yok. Setuptools hyso.* namespace'i
bulur; packaging testleri bütün source byte'larını ve exclusion'ları doğrular.

HYSo0.2.0/Python>=3.10/Torch>=2.2.2/torchvision>=0.17.2, legacy exports ve
state_dict, checkpoint v1 ve Trainer lifecycle freeze kalır. B1/B2/B3 installed
smoke'ları ayrı namespace import eder; Trainer/EBO.fit veya setup'da runtime
benchmark side effect eklenmez. Bu tasarım görevinde wheel build/install de yoktur.

## 16. Açık riskler

| Risk | Teknik önlem / test |
|---|---|
| Asenkron CUDA under-report | Aynı device/stream start/end + end sync; protocol order testi ve real CUDA smoke |
| Synchronization/launch overhead scope | Event ve host wall açık etiket; derived rate sustained rate'ten ayrı; scope mismatch comparison red |
| Kernel/autotune/warmup cache | Fixed W her repeat, preflight count ayrı, flags/shape sabit; warmup samples dışlama; steady-state iddiası yok |
| Threads/clocks/temperature/background load | Ayarları kaydet/değiştirme, repeats/raw data, caller isolation; variance gözlemi, fixed latency threshold yok |
| Model mutation/BN/Dropout/mixed state | Eval+inference_mode, registered-state guard ve finally exact flags; mutation/exception tests, arbitrary external state support yok |
| CUDA peak contamination/reset side effect | Ayrı probe/CPU snapshots/baseline, opt-in reset warning, same-process exclusivity, no empty_cache; ordering/accounting test |
| Shared storage logical bytes | Unique object sayımı ve alias note; tied/view test; physical/disk/RSS iddiası yok |
| Farklı batch/device/precision/env ile adaletsiz kıyas | Versioned compatibility key; unknown metadata explicit; CPU/CUDA scope mismatch reddi |
| FLOPs/MACs/RSS/accuracy yanlış iddiası | Unsupported/deferred null+reason; unit/disk/allocator ayrımı; schema tests |
| Trainer/legacy lifecycle coupling | Ayrı namespace, private checkpoint/Trainer dependency yok; freeze/API/regression gates |
| İleri Torch API/backend farkı |2.2.2 inspection + future installed tests; capability error, no silent fallback; protocol version/fingerprint |
| Çok kısa işte timer resolution | Zero/invalid sample error, epsilon uydurmama; workload artırma ayrı kullanıcı config'i |

Açık platform gereksinimleri implementasyonda native CUDA/memory gate'i ve
minimum installed CPU ile kapanmalıdır. Inspection bu riskleri test etmiş sayılmaz.

## 17. DESIGN GO / NEEDS REVISION

**M2-C-A DESIGN GO.** İlk inference sınırı, ownership/state koruması, iki timing
basis'i, throughput formülü, warmup/repeat/statistics sözleşmesi, memory opt-in
yan etkisi, versioned plain schema ve compatibility/red politikası kararlaştırıldı.
Doğrulanmayan metrik uydurulmaz; training/accuracy/FLOPs/RSS/serialized size
kapsamı ayrıldı. Gelecek native test kapıları açık ve uygulanabilir.
Bu karar henüz engine implementasyonu, benchmark sonucu veya final platform GO
değildir. Bilimsel adalet koşul eşliğine ve açık ölçüm kapsamına dayanır;
model kalitesi üstünlüğü iddiası üretmez.

Bu görevde repository'de yalnız bu yeni rapor oluşturuldu. Production/test/
roadmap değişmedi; kullanıcı dosyası aynı hash ile korundu. Full pytest/CUDA
benchmark, dependency kurma, veri/model indirme, version bump, commit/push veya
yayın yapılmadı. Küçük2.2.2 kontrolü yalnız API inspection'dır.

## 18. Sonraki görev önerisi

**M2-C-B1 — Minimal Inference Benchmark** ile başla: CPU fp32/fp64 protokolü,
state guard, parameter/logical-byte sayımı, schema1 ve strict unit/native CPU
testleri. Native CUDA/memory B2; export/comparison B3; final kalite kapısı C-C.
Benchmark Engine dışında GUI veya Experiment Manager geliştirmesine başlanmaz.
