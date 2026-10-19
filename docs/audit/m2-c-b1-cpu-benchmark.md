# M2-C-B1 — Minimal CPU Inference Benchmark

Tarih: 2026-10-19. **Karar: PASS — CPU BenchmarkRunner ve schema v1.**
Yeni benchmark testleri **133 PASS**; mevcut hedefli regression **1292 PASS**.
Final packaging dört testi bu1292 kimliğe dahildir, tekrar eklenmez:
**1425 farklı CPU testcase PASS, 0 failure/error/skip/xfail**.
Minimum Torch2.2.2 installed-wheel FP32/FP64 **2/2 PASS**.
Bu full-suite veya CUDA gate değildir; M2-C-C final kalite kapısı ayrıdır.

## 1. Git/HEAD/working tree

Branch `feature/hyso-models-next`, başlangıç/son HEAD
`979e9285cd9f839e408e3058fed42de3fefb1ed3`,
`docs: design HYSo benchmark engine architecture`. M2-C-A commit edilmişti;
belge kullanıcı kapsamı olarak korunmuş ve değiştirilmemiştir.
Başlangıçta yalnız `tests/test_gradient_accumulation.py` kullanıcı değişikliği
vardı. Bu dosyaya yazılmadı/stage edilmedi/geri alınmadı; başlangıç/son SHA-256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Commit/push/branch/version/release/dependency değişikliği yapılmadı.

## 2. 35-feature roadmap addendum

Yeni [performance backlog v1.1](../roadmap/hyso-performance-backlog-v1.1.md)
kullanıcının onayladığı35 özelliği **P01–P35**, eksiksiz ve tekil olarak kaydeder.
Her satır tam isim, Türkçe açıklama, birim/rapor, araç/donanım/veri,
önkoşul, aşama, durum ve ölçüm kabul kriteri içerir. Sekiz öncelik sırası
korunur. Script ile35 ID ve dokuz alanlı satırlar doğrulandı.

P01 latency, P08 precision, P11 footprint yalnız B1 CPU alt kapsamı nedeniyle
**partial**; kalan32 satır **planned**, tam implemented satır yoktur.
GPU/hardware counters, RSS, Roofline, energy ve provider fallback gerekli
kanıt/araçlarıyla planlıdır. Sabit v1.0, M2-C-A ve önceki milestone durumları
değişmedi. v1.1 belge sürümüdür; HYSo paket sürümü0.2.0 kalır.

## 3. Uygulanan B1 özellikleri ve hâlâ planned olanlar

CPU eager nn.Module/ModularCNN inference, FP32/FP64, sabit resident NCHW input,
local generation/custom clone, monotonic timing, warmup/repeats, mean/p50/p90/
p95/std, derived service-rate, logical counts, state guard ve plain schema v1
uygulandı. Modelin zaten doğru CPU/dtype'ta olması gerekir.

CUDA event/AMP/memory **B2**, JSON/CSV dosya export/comparison **B3**;
P99, sustained/window throughput, end-to-end/camera FPS, sweeps, profiler,
FLOPs/MACs, RSS, disk size, training/accuracy/robustness, deployment ve energy
bu görevde geliştirilmedi. Uygulanamayan metrik null+status+reason'dur;
derived rate gerçek sustained FPS olarak sunulmaz.

## 4. Yeni public API

```python
from hyso.benchmark import BenchmarkRunner, BenchmarkResult

# model caller tarafından CPU FP32'de oluşturulmuş olmalıdır.
runner = BenchmarkRunner(
    model, device="cpu", input_shape=(1, 3, 32, 32), precision="fp32",
    warmup_iterations=5, measure_iterations=20, repeats=3, seed=0,
)
result = runner.run()
data = result.to_dict()
```

FP64 için caller modeli önceden double dtype'ta kurar/yerleştirir ve
precision="fp64" verir. Benchmark modeli .to/.cuda/.half ile değiştirmez.
input_shape veya example_input'tan tam biri gerekir; batch ilk NCHW boyutudur.
Python bool sayısal count/shape/seed olarak kabul edilmez. Seed0..2^63-1,
warmup>=0, measured/repeats>=1; bütün model/input dtype/device/finite
kontrolleri yapılır. Integer/bool buffer'lar ve BN sayaçları geçerlidir.

Custom input no-grad, dense contiguous finite CPU floating Tensor'dır;
runner'ın clone'u kullanılır. Inference Tensor örneği ve caller'ın açık
inference_mode context'i de desteklenir: kendi input'u inference_mode(False)
altında normal/versioned clone olarak hazırlanır; caller context'i korunur.
Meta default-device bağlamı CPU üretimini değiştirmez; device="cpu" explicit'tir.

Multi/nested I/O, dynamic output shape/dtype, parallel/scripted/compiled/
quantized/sparse/meta/lazy profile, unsafe hooks/custom train/eval ve CUDA
istekleri açık hata verir; fallback yoktur. Normal/versioned registered model
state gerekir; inference-mode içinde yaratılmış version-counter'sız registered
parametre/buffer'lar güvenli profile dahil değildir. Arbitrary unregistered
Python state ve external I/O yan etkileri desteklenmez.

## 5. CPU timing doğruluğu

Measured scope tam olarak `start=time.perf_counter_ns(); output=model(inputs);
end=time.perf_counter_ns()`; ms=(end-start)/1_000_000. Python synchronous
forward wall-clock ölçülür. Context kurma, input generation/copy, warmup,
snapshot/guard/output validation, statistics ve result conversion scope dışıdır.
Clock reading exact int ve end>start olmalı; latency finite/>0 değilse açık
FloatingPointError, epsilon/clamp veya sample silme yoktur.

timing_method=perf_counter_ns, timing_scope=synchronous_host_forward,
synchronization_policy=none. Clock resolution metadata'dır, latency değildir.
Kontrollü clock yalnız protocol/statistics testinde kullanıldı; gerçek native
CPU forwards ayrıca çalıştı. Sabit ms/speed-up threshold assertion'ı yoktur.

## 6. Warmup/repeat protokolü

Bir untimed correctness preflight; her repeat W warmup ve N measured forward;
R repeat. Toplam **1+R*(W+N)**; memory_probe_count=0.
Nested samples R×N listesi olarak saklanır. Testte W2/N2/R2 için9 gerçek
forward ve yalnız8 timer okumasının clock/forward sırası doğrulandı.
Warmup samples'a girmez. Warmup0 açık warning üretir.

Bütün forward'lar forced eval + torch.inference_mode + CPU autocast disabled
altındadır. Caller'ın outer grad/autocast/inference context'leri korunur.
Her output tek dense finite floating CPU Tensor, batch uyumlu ve ilk output
shape/dtype'ıyla aynı olmalıdır; kontrol timer dışında yapılır. Son measured
output da böyle doğrulanır. Önceki output sonraki çağrıdan önce bırakılır.
Guard/validation CPU cache koşullarını etkileyebilir; bu sınırlama diagnostics'te
açıktır. Fixed warmup'un steady state'e yettiği otomatik iddia edilmez.

## 7. Statistics ve throughput

Per-repeat ve pooled count/mean/median=p50/p90/p95/min/max/std verilir.
Quantile sorted samples `(n-1)*q` index'inde linear interpolation;
sample std ddof1. Tek sample'da std_ms=null ve std_status=insufficient_samples;
aynı örneklerden gerçek std0 çıkması geçerli ölçümdür.

Bilinen [1,2,3,4] ms örneğinde mean/p50=2.5, p90=3.7, p95=3.85,
std=sqrt(5/3) doğrulandı. Gerçek timer sonuçları için finite/positive ve
sample/statistics/rate tutarlılığı denetlenir; outlier clipping yapılmaz.

`images_per_sec = batch_size * measured_forward_count * 1000 / fsum(samples_ms)`.
method=derived_serial_service_rate; source_timing_method=perf_counter_ns.
Per-repeat oranlar ayrı verilir. Batch2/four samples/10ms örneği800 images/s
çıkar. Bu sentetik arithmetic reference'dır; gerçek model performans iddiası
değildir. Sustained/window rate null/unsupported/deferred olarak kalır.

## 8. Model state safety

Registered modules/parameters/buffers doğrudan traversal ile doğrulanır;
custom train/eval/traversal override ve local/global module hooks önceden reddedilir.
Snapshot kayıtlı key/reference yapısı, tensor shape/stride/storage offset/
storage identity/dtype/device/requires_grad, byte değerleri ve version counters,
mevcut gradient referans/değerleri, per-module training bayrakları ve input'u
kapsar. Generator global seed'i değiştirmez. Python/NumPy/Torch CPU RNG
try/finally ile success/exception sonunda geri yüklenir; CPU run CUDA başlatmaz.

Preflight ve repeat sonlarında full byte invariants; çağrılar arasında
registration/version/input/gradient/mode guard'ları vardır. .data ile version
counter bypass edilen value write da full guard'da yakalanır. Başarıda canlı
tensor'a gereksiz copy yapılmaz, parametre/gradient identity ve version aynı kalır.
Modes root.train(old) ile değil, her alt modülde tek tek restore edilir.

Mutation sonucu başarılı BenchmarkResult üretilmez.
`hyso.benchmark.runner.StateMutationError`, original registries/references ve
değerler için best-effort rollback sonrası yükseltilir. Structural/storage/dtype
mutation güvenle düzeltilemiyorsa açık fatal `caller runtime may be mutated`
hatası vardır. Version counters ve arbitrary external state için transaction
garantisi verilmez. Reentrant/concurrent benchmark run'ları nonblocking process
lock ile reddedilir; caller'ın dışarıdan aynı modele müdahalesi desteklenmez.

Gerçek testler: BN/Dropout/mixed-mode, mevcut gradients ve context preservation;
preflight/warmup/measured exception; parameter/buffer/gradient/input/reference/
key/mode/.data mutation; fatal rollback ve lock release. Optimizer/backward/
zero_grad/EMA/scheduler oluşturma veya çağırma yoktur.

## 9. Parameter/byte counting

Unique Parameter object: total/trainable; frozen=total-trainable.
Logical bytes=numel×element_size; parametre ve buffer alt toplamları ayrı,
combined toplam her Tensor object'ini kategori boyunca bir kez sayar.
Aynı Parameter'ın iki isimden paylaşımı iki kere sayılmaz. Distinct view'lar
aynı storage'ı paylaşsa da ayrı logical tensor olabilir; alias flags/warning
physical storage veya RSS iddiasını engeller.

Native alias testinde iki Linear aynı9-element frozen Parameter'ı paylaşır;
o Parameter buffer olarak da kayıtlıdır, ayrıca storage-sharing view ve int64
counter vardır. Total=frozen9/trainable0, parameter36 byte, buffer80 byte,
combined80 byte olarak doğrulandı; bütün referans/alias'lar korundu.
Bu sayılar optimizer/activation memory, CPU RSS, GPU allocator veya disk
serialized model size değildir. Eksik ölçümler sahte0 almaz.

## 10. Result schema

BenchmarkResult frozen/slots sınıfıdır; içeride immutable plain JSON string
tutar, public model/Tensor referansı taşımaz. to_dict her çağrıda defensive copy
verir; constructor plain type/finite/key/depth, schema ve protocol/sample/
statistics/throughput/count tutarlılığını doğrular. Dosya export API'si yoktur.

schema_version=1, benchmark_type=inference. Identity/model/input/configuration/
environment/provenance/timing/stats/throughput/parameters/cuda_memory/
optional_metrics/diagnostics grupları vardır. Metadata class/registered tensor
ve native structure, input hash/shape/stride/dtype, timestamps/run ID, source
origin, precision, threads/backend settings ve limits'i kapsar. Model config
plain defensive annotation'dır; arbitrary object/repr/pickle kabul edilmez.

Weights hash gerçek registered byte/schema'dan hesaplanır. Source hash
toplanmaz: source_sha256=null + not_collected + reason; uydurma hash yoktur.
CUDA metadata/memory CPU'da not_applicable/null. Accuracy/FLOPs/MACs/disk/RSS/
optimizer/activation memory unsupported/null+reason. Vision metadata
toplanmadığında explicit not_collected'dır. Measured arithmetic finite olmalıdır.

JSON string yalnız immutable internal storage'dır; run otomatik dosya yazmaz,
to_json/to_csv veya compare_results eklenmedi. Backend/thread ayarları okunur,
değiştirilmez. Latency'nin başka sistemde aynı olması garanti değildir.

## 11. Gerçek CPU testleri

Python3.10.18/Torch2.5.1+cu121; CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, tests/pytest.ini ve tek Torch CPU thread.
Önce yeni benchmark testleri, sonra ilgili regression çalıştırıldı.

| Son test dosyası/grup | PASS | Süre |
|---|---:|---:|
| test_benchmark_cpu.py |85| |
| test_benchmark_result.py |24| |
| test_benchmark_state_safety.py |24| |
| Yeni benchmark ortak son koşu |133|6.04s|
| Hedefli mevcut Trainer/legacy/M1/API/packaging |1292|129.84s|
| Son kaynaklarla packaging tekrar doğrulaması |4 (yukarıdaki4 kimliği yeniler)|28.13s|
| **Farklı JUnit identity toplamı** |**1425**| |

Regressions: M2 classification116, updates141, validation32, lifecycle53,
save135, restore179, transaction13, strict replay37; ModularCNN102,
blocks135, attention119; public API41/published1/model-state18;
native checkpoint34/EBO resume45/OneCycle20/accumulation28/CV resume39;
packaging4. Hepsi yeni gerçek koşudur; B5 sonucu kopyalanmadı.

Final classname/name identity birleştirmesi130→133 ara testleri veya packaging
tekrarlarını iki kez saymaz. Son paket kaynak byte'ları son checkout ile eşittir.
0 failure/error/skip/xfail; mevcut regression GradScaler ve yeni testteki
CPU-autocast introspection deprecation uyarıları kayıtlıdır.

İlk yeni koşuda95 PASS sonrası scalar int64 BN sayaç test yardımcısının byte-view
hatası görüldü; flatten+shape/dtype check ile düzeltildi, assertion gevşetilmedi.
Ek profil/context kontrolleri ve CPU explicit allocation incelemesi sonrası
etkilenen yeni testler ve packaging tekrar çalıştırıldı. Önceki başarısız/ara
koşular final PASS toplamına eklenmez. Production freeze dosyalarında değişiklik yok.

Bu hedefli1425 sonucu **full CPU suite değildir**; tüm CPU/GPU final gate C-C'ye
kalır. Gerçek CUDA benchmark çalıştırılmadı ve CUDA PASS iddiası yoktur.

## 12. Minimum Torch 2.2.2 installed-wheel

Mevcut clean Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu ortamı kullanıldı.
Son wheel, repo dışındaki izole `installed-minimum-final` target'ına
`pip --isolated --no-index --no-deps --no-compile` ile kuruldu. Cwd kanıt
dizini, çocuk süreç `-I -B`; bütün HYSo import'ları installed target altında
assert edildi. Test-only replay helper path'i açıkça verilir; benchmark
production kaynağı checkout'tan import edilmez.

**2/2 gerçek FP32/FP64 Conv+BN+Dropout inference PASS**. Her biri W1/N2/R2,
dört native latency sample; schema1/result round-trip, finite statistics,
bağımsız fmean/throughput reference, parameter count, modes/BN/RNG korunumu
ve CUDA initialized=False denetlenir. Aynı installed smoke eski model/Trainer/
checkpoint/replay kapılarını da korur; bunlar iki benchmark senaryosuna ek
benchmark case diye sayılmaz. Clean35 distribution sürüm/METADATA/eski HYSo
source hash'leri önce/sonra aynıdır; yeni dependency kurulmadı.

## 13. Wheel/sdist

MODULES envanteri gerçek filesystem ile **19** olarak doğrulandı: mevcut16 ve
yeni3 benchmark modülü. Yeni son wheel/sdist gerçek build edildi; ikisi de
checkout dışında kurularak Runner/Result import ve FP32/FP64 native smoke geçti.
Kaynak byte eşliği iki arşivde bütün19 modül için denetlendi. Metadata0.2.0,
Python>=3.10/Torch>=2.2.2/torchvision>=0.17.2 ve sekiz dependency aynı kaldı.
Tests/docs/cache/weights/eski artefakt exclusion assertion'ları korundu.

| Son arşiv | Byte | SHA-256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |66412|eb19800211b757eac92835fb5e1748155830109220f9629917c3f8135f875cdd|
| hyso-0.2.0.tar.gz |63691|d10f0165fd045059aa34a67da052ad4f3a47264aab7e5090859a29cced803519|

Kanıt kökü E=`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2cb1`.
Son arşivler `E/pytest-final-verified/packaging0/dist/` altındadır;
önceki ara build'ler final artefakt değildir. Bu hash'lerin önceki build'lerle
aynı olması aranmaz; gerçek source bytes/metadata temel doğrulamadır.

## 14. Legacy compatibility

Git blob eşliğiyle mevcut16 production dosyasının tamamı HEAD ile aynı.
checkpoint.py, Trainer/resume, bütün models/callbacks, public exports ve
FORMAT_VERSION=1 değiştirilmedi. Roadmap v1.0, M2-C-A tasarımı, setup.cfg,
pyproject.toml ve MANIFEST.in aynı. Benchmark namespace eğitim lifecycle'ına
bağlı değildir; EBO.fit/Trainer.fit çağırmaz. Mevcut testlerden yalnız
packaging'e additive inventory/import/smoke assertion'ları eklendi.

## 15. Değişen dosyalar

- `docs/roadmap/hyso-performance-backlog-v1.1.md`:35 onaylı özellik.
- `hyso/benchmark/__init__.py`: yalnız Runner/Result public export.
- `hyso/benchmark/runner.py`: CPU protocol, validation/state guard/metadata.
- `hyso/benchmark/result.py`: frozen plain schema/statistics/derived rate.
- `tests/test_benchmark_cpu.py`:85 native/protocol/profile case.
- `tests/test_benchmark_result.py`:24 schema/immutability/arithmetic case.
- `tests/test_benchmark_state_safety.py`:24 state/RNG/failure case.
- `tests/test_packaging.py`:19-module inventory ve installed benchmark smokes.
- `docs/audit/m2-c-b1-cpu-benchmark.md`:bu rapor.

Git'teki gradient_accumulation farkı başlangıçtaki kullanıcı değişikliğidir;
aynı SHA ile korunur. Kanıtlar: benchmark-junit.xml, regression-junit.xml,
packaging-final-junit.xml, installed-minimum.json, minimum-gate.py,
minimum stdout/stderr, clean-before/after.json, summarize.py ve
verification-summary.json (1425 unique identity, freeze/hash/inventory dahil).
Final packaging alt ağacında build log, installed wheel/sdist smoke JSON'ları
ve target'lar vardır. `git diff --check` başarılıdır.

## 16. PASS / PARTIAL / FAIL

**PASS — M2-C-B1.** CPU FP32/FP64 çalışan public benchmark, state/RNG guard,
schema/statistics/throughput,35-feature backlog, hedefli native testler,
minimum installed runtime ve son wheel/sdist kapıları başarılıdır.
Partial backlog satırları B1 başarısını aşan geniş kapsamı ifade eder;
CUDA/export/profiling/quality/deployment uygulandı anlamına gelmez.
Commit/push/version bump/release/dependency veya veri/model indirmesi yoktur.

## 17. M2-C-B2 için CUDA event/memory gereksinimleri

B2'ye geçilebilir: explicit device/current stream, önceden initialized timing
event'leri, end-event synchronize, repeat boundary drain, native FP32/FP16
autocast ve gerçek GPU gate gerekir. CPU perf_counter scope'u GPU latency
diye yeniden etiketlenmemeli. CUDA memory ayrı opt-in baseline/reset/single
probe/current/reserved/peak protokolüyle; process peak reset'in geri alınamayan
yan etkisi ve başka process/caller allocations sınırı açık olmalıdır.
State safety ve CPU tests korunmalı; CUDA yoksa gate gerçek FAIL/platform
eksikliği olarak raporlanmalı. JSON/CSV/comparison B3'te kalır.
