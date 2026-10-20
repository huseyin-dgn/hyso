# M2-P-B4 — Batch, resolution, precision ve birleşik kalite kapısı

2026-10-20, Europe/Istanbul. Branch `feature/hyso-models-next`, başlangıç HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c` (`feat: add pipeline timing and cold start analysis`).
B3 committed doğrulandı; index boş, başlangıçtaki tek fark kullanıcı gradient testi.
Evidence **E**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb4-20261020`.

**P06 PASS, P07 PASS, P08 PASS (aşağıdaki recipe kapsamları). B4 birleşik kalite
kapısı FAIL; B5 NO GO.** Yeni implementation ve package kapıları geçti, fakat
eski native trainer-replay process crash nedeniyle bütün eski native baseline
başarısı doğrulanamadı. Başarılı retry ile bu hata gizlenmedi; retry yapılmadı.

## 1. Mimari ve public API

M2-P-A tasarımı, B3 audit, backlog, performance kaynakları, Benchmark runner/result
ve packaging guard incelendi. Tek yeni production modülü `hyso/performance/sweeps.py`.
P06/P07/P08 aynı finite point orchestration'ını kullanır; scheduler/registry/plugin
framework, dependency veya conversion engine eklenmedi. Public API:

```python
SweepProtocol(device='cpu', precision='fp32', warmup_iterations=2,
              measure_iterations=10, repeats=1, sustained_batches=None,
              queue_depth=4, continue_on_error=True, measure_cuda_memory=False)
run_sweep(model, input_factory, *, axis, points, input_shape=(1,1,5,7),
          protocol=SweepProtocol())
SweepResult
PrecisionResult
```

Model ordinary dense **CPU FP32 master** olmalıdır. Her nokta bağımsız deepcopy
alır; module/storage aliases caller ile paylaşılırsa reddedilir. Copy ayrı
device/dtype/recipe hazırlığı yapar. Caller üzerinde `.half()/.bfloat16()/.to()`
yapılmaz. Factory `(exact_nchw_shape, dtype, device)` alır; contiguous exact tensor
döndürür. Shape/device/dtype mismatch failed point'tir, silent resize/cast yoktur.

## 2. P06/P07 nokta sözleşmesi

Axis batch: explicit positive unique integers; N gerçek cardinality'dir. Axis
resolution: explicit unique (H,W), positive dimensions; batch/channel/precision
sabit kalır. H/W ve W/H aspect ratio ayrı kaydedilir; model-specific shape
constraints gerçek forward ile sınanır. İlk başarısız shape için interpolation
veya output shape uydurma yoktur. Noktalar caller sırasıyla yürür.

Her nokta kendi warmup/repeat/sample penceresini kurar. FP32/FP64/AMP FP16 latency
eski BenchmarkRunner, sustained rate istenirse mevcut P02 API'sidir. Latency
ms/batch, sustained completed-images/host-wall-second ayrı alanlardır. Eski
BenchmarkResult schema1 payload'ı değiştirilmeden embedded evidence olur. Memory
explicit opt-in P09 separate single-forward probe'udur; default disabled/null.
P10 RSS ve P11 logical/serialized semantikleri yeniden yorumlanmaz.

Point UUID, observation_index, exact shape/aspect, precision, status/reason,
independent PrecisionResult UUID ve actual measurement refs manifest'tedir.
Source master hash her noktada aynıdır; input content SHA exact real tensor'dan
alınır. Batch/shape/precision değişimleri eski compare_results ile otomatik
compatible ilan edilmez. Gerçek farklı batch Benchmark comparison testi
incompatible sonucunu doğrular; sweep `cross_point_compatibility=not_asserted` taşır.

## 3. P08 doğrulanan precision profilleri

| Seçenek | Gerçek kapsam ve kanıt |
|---|---|
| CPU fp32 | Owned FP32 copy/input; mevcut host latency/P02 completion |
| CPU fp64 | Owned FP64 copy/input; mevcut host latency/P02 completion |
| CUDA fp32 | Owned FP32 copy/input; mevcut selected-stream event latency/P02 |
| CUDA amp_fp16 | FP32 master/input, FP16 autocast; existing native protocol |
| CPU bf16 | FP32 master/input, BF16 autocast ve actual BF16 output; CPU backend execution |
| CUDA bf16 | Capability≥8 ve is_bf16_supported; actual BF16 output, selected-stream events/completion |
| CPU int8_dynamic | Active quantized engine, dynamic qint8 Linear conversion ve actual quantized::linear_dynamic |

CPU BF16 dtype execution backend availability kanıtıdır; hardware BF16 instruction
veya acceleration iddiası değildir, backend emüle edebilir. BF16 output FP32 ise
unsupported; CUDA capability/runtime yoksa unsupported, CPU fallback yoktur.
CUDA fp64/INT8 ve CPU AMP FP16 bu API'de unsupported+reason'dır.

INT8 recipe yalnız exact `Sequential` içinde Flatten/Linear/ReLU/Identity whitelist
kabul eder. Linear'lar gerçek dynamic quantized modules olmalı, packed weights
qint8 olmalı, ölçüm dışı proof preflight profiler'ında `quantized::linear_dynamic`
gözlenmelidir. Bu üç şart yoksa INT8 PASS üretilmez. Diğer allowed ops floating
point kalır. Global quantized engine değiştirilmez. Dynamic activation quantization
kalibrasyon dataset'i gerektirmez; calibration not_applicable/reason kaydedilir.
Bu sınırlı recipe conv/detection/whole-graph INT8 kapsamı değildir.

Master/input/output dtype, autocast dtype, backend/recipe, source/input hashes,
runtime/device/stream ve execution proof sonuçtadır. Copy+conversion+device move+
completion maliyeti ayrı host ms scope'unda ölçülür; load/artifact maliyeti
not_measured/null'dır, cold load diye etiketlenmez. P28 accuracy not_measured;
performans ölçümü kalite korunumu kanıtı değildir.

BF16/INT8 özel timer'ı mevcut latency/P02 sonuçlarını değiştirmez. Proof preflight
ölçüm dışı, her repeat warmup bağımsız; CPU host clock ve CUDA event clock ayrıdır.
CUDA repeat drain ve end-event synchronize gerçek completion bekler. P02 `_window`
aynı bounded completion/finite/tail hesabı için reused. Special forward ledger
1 preflight + R×(W+N) + optional R×(W+sustained_batches) kaydeder. Legacy latency
ve P02 ayrı birer preflight'a sahiptir; exact-count testleri bu farkı doğrular.

## 4. Failure, RNG, ownership ve lifetime

Ordinary ValueError/RuntimeError ve unsupported continuation explicit flag'i izler.
OOM admission'ı her zaman durdurur; failed point korunur, kalanlar not_run/null'dır.
CUDA gerçek OOM testi 4TiB allocation talebinin cihaz tarafından reddedilmesini
doğrular; capacity search veya OOM sonrası güvenli retry iddiası değildir. State/
stream mutation ve drain failure fatal exception üretir. Error diagnostic string
plain-data'dır; traceback/model/output retained result'a konmaz.

Her nokta aynı caller Python/NumPy/Torch CPU/selected CUDA RNG başlangıcını alır;
finally caller RNG/state/mode/gradients/current device/stream restore edilir.
Source/input registered-state guard kullanılır; source mutation fatal, best-effort
value rollback yapılır. Owned ordinary/BF16 modeller existing guards ile kontrol
edilir; quantized graph whitelist/factory ownership ile sınırlıdır. External Python
side effects transaction garantisi ve concurrent caller desteği yoktur.

No recursive self-capturing closure; GC-disabled weakref model/output lifetime
testleri vardır. Output retention P02 bounded ring'ini izler. Global GC policy,
empty_cache veya allocator reset eklenmedi. Existing P09 opt-in reset geri alınamaz;
BF16/INT8 allocator probe unsupported/null/reason'dır. C1 dört eski regression
ve P09 native invariant'ları birleşik kapıda ayrıca korunur.

## 5. Versioned result/export

Formats `hyso.performance.sweep` ve `.precision`, schema_version=1; bağımsız frozen
plain JSON results. Existing result/export primitive'leri reused; eski B1–B3
branches/schema değiştirilmedi. Required envelope/protocol, requested/actual
counts, units/status/reason, raw evidence ve diagnostics bulunur. Manifest point
association/ref/count/shape/precision/source-hash doğrulanır; precision raw
Benchmark/Throughput result'ları kendi validator'larından geçer. Special BF16/
INT8 output/operator ve forward/completion ledger doğrulanır.

Defensive copy, finite values/allow_nan=False, unknown schema rejection,
atomic/no-clobber exports ve formula-safe CSV korunur. Failed/unsupported/not_run
point result/metric alanları null'dır; eksik ölçüm numeric zero ile doldurulmaz.
CSV per-point bounded summary, JSON full canonical evidence'dır.

## 6. Experiment Manager ve executable guide

[performance-sweeps.md](../guides/performance-sweeps.md) actual CPU batch/resolution/
precision ve typed JSON/CSV export örneklerini içerir. Ölçüm sonrası manifest,
JSON/CSV ve point results existing schema2 copy artifacts olarak ilişkilendirilir.
Manifest observation_index→point/result refs eşlemesini taşır; dört bounded scalar
metric, artifact size/SHA verify ve close/reopen/query doğrulanır. Unsupported/null
metric numeric zero yapılmaz. Measurement windows içine SQLite/disk/network
tracking yazısı konmaz; yeni migration/table/run kind yoktur.

Guide source/full CPU, wheel/sdist installed ve minimum Torch installed ortamında
execute edilir. CUDA örneği native gate'teki equivalent gerçek selected-stream
workload ile doğrulanır. Backend/artifact alanları future adapter sınırını taşır;
ONNX, LibTorch/C++, TensorRT, OpenVINO, TFLite, Core ML ve YOLO engine geliştirilmedi.

## 7. Geliştirme doğrulaması

İlk smoke ve CPU collection unmatched closing-parenthesis syntax errors yakaladı;
gate öncesi düzeltildi. İlk işlevsel yeni CPU koşusu50 PASS/1 FAIL: deepcopy
izleyici fixture string'e weakref oluşturuyordu; fixture yalnız Module kopyalarını
izleyecek şekilde düzeltildi. Assertions/tolerances gevşetilmedi. Sonra exact
forward/output-release ve formula-safe error CSV testleri eklendi.
`sweep-freeze.xml`: **55 PASS**. Native development B4: **11 PASS**, failed native
execution yok. Bu geliştirme tekrarları primary gate'e ikinci kez eklenmez.

## 8. Final kalite kapısı

Production/test SHA freeze `gate-source-freeze.json`; final kaynaklar üzerinde
tek full CPU koşusu yapıldı. `full-cpu-junit.xml`, `full-cpu-results.json`,
`cpu-collection.json` ve command/stdout/stderr authority'dir.

| Collected | PASS | SKIP | FAIL/ERROR/XFAIL/XPASS | pytest süre |
|---:|---:|---:|---:|---:|
|2314|2313|1|0|350.01s|

Mevcut Python3.10.18/Torch2.5.1+cu121 runtime, CUDA_VISIBLE_DEVICES=-1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, MPLBACKEND=Agg, `-B -m pytest -p gate_capture
-c tests/pytest.ini tests -q`; process wall/argv/env command JSON'dadır.
cuda_initialized=False. Tek skip tam eski
`test_device_precision.py::test_r12_r13_real_cuda_device_smoke`,
`R12/R13: CUDA hidden/unavailable in the CPU-first test environment`.
1043 warning eski mesaj kategorileridir; bütün nodeid/phase/message kayıtları
saklanır. Fail/error/xpass ve yeni skip/xfail yoktur.

D-C2 **1996/1996 exact nodeid/status korundu**, kayıp/değişim0. B1 **122**, B2
**83**, B3 **58**, B4 **55** yeni kimlik: toplam **318**. JUnit classname/name,
collection ve phase inventory aynı2314 kimliği verir, duplicate yoktur. Eski
Generic Trainer/checkpoint v1, Benchmark schema1, Experiment259, C1 dört lifetime,
public/model-state/legacy davranışlar bu güncel full gate içindedir; eski PASS
sonuçları yeni koşu yerine kullanılmadı. Full CPU gate tekrar çalıştırılmadı.

Final native/minimum/package ve identity/source/archive karşılaştırmaları
`E/verification-summary.json` içinde birleştirilir.

### 8.1 Gerçek native CUDA primary gate

Python3.10.18/Torch2.5.1+cu121/torchvision0.20.1+cu121, CUDA12.1, RTX3060 Laptop,
capability8.6, API total6441926656 byte, driver591.74. CUDA_VISIBLE_DEVICES=0,
CUBLAS_WORKSPACE_CONFIG=:4096:8. Her script ayrı process'te, sırayla bir kez
çalıştı; unavailable skip veya CPU fallback yoktur. `gpu-runtime.json`,
`cuda-script-inventory.json` (source SHA/argv), `cuda-runs.json` ve ayrı stdout/
stderr/raw JSON bütün kanıtı saklar.

| Primary script | Observed PASS execution | Exit |
|---|---:|---:|
| cuda_runtime_validation |7|0|
| cuda_imagefolder_validation |10|0|
| cuda_modular_validation |18|0|
| cuda_trainer_updates_validation |40|0|
| cuda_trainer_validation_lifecycle |16|0|
| cuda-save (eski B4-A helper) |3|0|
| cuda_trainer_checkpoint_restore |5|0|
| cuda_trainer_replay_conformance |0 observed|3221226505|
| cuda_benchmark_validation |45|0|
| cuda_benchmark_export_validation |8|0|
| cuda_experiment_integration_validation |2|0|
| cuda_performance_validation (B1) |15|0|
| cuda_pipeline_validation (B3) |8|0|
| cuda_sweeps_validation (B4) |11|0|

**14 process:13 success/1 crash;188 PASS execution,188 distinct PASS,0 duplicate
observed execution.** Expected complete gate198 executions/196 distinct/2
duplicates olacaktı. Replay'nin10 execution'ı successful report üretmedi; bunlar
PASS sayılmadı. Restore'da zaten geçen rollback-False/True iki kimliği replay'de
yeniden gözlenmediği için current duplicate0'dır. Eski162 distinct'ten154 PASS;
**8 eski unique identity eksik**, B1/B3/B4 yeni34 distinct PASS. Exact missing IDs
`cuda-baseline-comparison.json` içindedir: replay fp32/16×ema False/True dört,
native-overflow-recovery ema False/True iki, negative-plan False/True iki.
Eksik sonucu failed assertion olarak uydurmuyoruz: **1 native process crash ve
unobserved scenario outcomes** var; complete native protection gate başarısızdır.

Replay23.515s sonra **0xc0000409** ile sonlandı; `cuda-replay.json` publication
yoktur, stdout boş ve stderr yalnız existing GradScaler FutureWarning içerir.
Windows Application Error1000/Record159956, 2026-10-20 17:22:13 yerel saat kaydı
aynı aralıkta real conda Python PID0x3D1C, faulting module **nvcuda64.dll**, version
32.0.15.9174 ve exception0xc0000409 bildirir (`windows-application-error.json`).
Bu kayıt root cause'u veya B4'ün ilgisizliğini tek başına kanıtlamaz. Legacy
training/checkpoint/replay script/support source byte'ları değişmedi. Driver/
dependency güncellemesi, legacy test değişikliği, tolerance gevşetme veya retry
yapılmadı. Önceki D-C2 replay PASS'i bu crash yerine güncel PASS diye kullanılmadı.

B4'ün11 IDs'i `batch_fp32`, `resolution_fp32`, `batch_amp_fp16`,
`resolution_amp_fp16`, `nondefault_fp32`, `nondefault_amp_fp16`, `bf16_capability`,
`bf16_nondefault`, `actual_cuda_oom_stop`, `invalid_shape_continuation`,
`stream_failure_cleanup`. Her başarı raw per-point shape/sample/event/completion/
RNG/state kanıtını taşır. BF16 capability bu cihazda true; actual BF16 workload
geçti. B3'ün iki fresh CUDA cold worker'ı da güncel primary gate içinde geçti.

### 8.2 Final installed native ve Windows

Tek final wheel installed target üzerinde B1 **15 PASS**, B3 **8 PASS**, B4
**11 PASS tekrar**. Installed import provenance ve29-modül source-byte equality
doğrulandı. Bu34 execution primary distinct toplamına yeniden eklenmez; eski
replay crash'inin retry'si değildir ve onu kapatmaz. Installed Windows P10 native
touch/failure/crash **3 PASS**; process ownership, RSS/HWM/cache/lifetime ve cleanup
koruma kontrolleridir (`installed-windows*.json`). Native Linux iddiası yoktur.

### 8.3 Minimum Torch ve final package

Mevcut Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Final wheel E/installed-final'e local/no-index/no-deps/no-compile kurulmuştur;
dependency indirilmedi veya kurulmadı. Cwd=E, `-I -B` installed imports; P06/P07 ve
CPU FP32/FP64/BF16/INT8 gerçek ölçümler, typed JSON/CSV, E association/reopen,
P01–P05/P09–P11 CPU-compatible smoke, legacy Trainer/Benchmark/checkpoint/replay
smoke PASS. CPU process cuda_initialized=False. Dependency distribution
METADATA/RECORD SHA önce/sonra eşittir (`minimum-environment.json`); bütün environment
binary'leri için full-content hash audit'i iddia edilmez.

Installed example runs ayrıca gerçek **batch1/2/4/8/16/32**, **320×320,416×416,
640×640,640×384,960×960** çözünürlükleri CPU minimum ve CUDA runtime'ında doğrular.
Bu noktalar küçük Conv2d/adaptive-pool fixture için geçerlidir, arbitrary model
shape garantisi değildir. Minimum precision raw evidence CPU BF16 output ve
**pytorch_quantized_x86**, iki qint8 Linear/quantized::linear_dynamic kanıtını
taşır (`sizing-cpu-precision.json`). Extra example executions primary test-ID
toplamına eklenmez.

Explicit production inventory28→**29**; guard korunur. Full CPU içindeki packaging4
PASS, complete module inventory/source-byte equality, wheel/sdist installed
imports, optional feature execution, unchanged0.2.0/Python≥3.10/Torch≥2.2.2 bounds
ve tests/docs/cache/temp/DB/weights exclusion integrity doğrular. Tek final build,
production kaynakları bundan sonra değiştirilmedi. E/pytest-full/packaging0/dist:

| Artifact | Byte | SHA256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |129357|`ab1a0fdbf1ef8603222801c679bea9a022182569363f388a56b8e5badd005e49`|
| hyso-0.2.0.tar.gz |120636|`41ec2472600f13b049d9c93b6f87f672f856c0791181f8175624436f5085070e`|

Commands/stdout/stderr, minimum-smoke JSON, installed-integrity hashes ve current
source freeze E'dedir. Full CPU ve packaging gereksiz tekrar edilmedi.

## 9. Kapsam, backlog ve B5 önkoşulları

P06/P07 explicit finite CPU/CUDA points; P08 yalnız yukarıdaki gerçek backend/
precision recipe'leri kapsar. BF16 CPU acceleration, genel INT8 conv graph veya
P28 doğruluk garantisi verilmez. P17 maximum-feasible-batch araması uygulanmadı.
B5 ayrı worker isolation/timeout, budget/next-OOM sınır kanıtı ve nonmonotonic
workload politikası gerektirir; sweep OOM stop sonucu global maximum değildir.

P06/P07/P08 verified feature sözleşmeleri implemented; **11 implemented,1 partial,
23 planned**, diğer statüler değişmedi. Bu feature statüleri **B4 milestone GO
değildir**. B4 full quality gate native crash nedeniyle **FAIL**, B5'e geçilemez.
Önce replay/nvcuda crash'in kök nedeni araştırılmalı ve hata kaydı korunarak temiz
bir complete native gate kanıtlanmalıdır. Bu görev driver/runtime/legacy training
bug'ını varsayımsal bir patch veya başarılı retry ile kapanmış saymaz.

Görevde4 existing dosya değişti: performance namespace/result, packaging test,
backlog. Yeni5: sweeps production, CPU tests/native CUDA script, guide ve bu audit.
Başlangıç tracked SHA karşılaştırması diğer bütün eski dosyaları korur;
gate-source-freeze production/test kaynakları finalde aynı kalır. Index boş,
branch/HEAD aynı; whitespace/doc links/source-installed integrity doğrulanır.

Kullanıcı gradient testi SHA256
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`
başlangıç/final karşılaştırılır. Eski runner/result schema1, inference/memory/
pipeline/cold, Trainer/checkpoint, Manager ve önceki audit/roadmap değişmez.
Commit/push, branch değişimi, dependency/model/data indirme ve version bump yoktur.
