# M2-P-B6 — FLOPs/MACs, layer ve operator profiling

2026-10-20, Europe/Istanbul. Başlangıç branch `feature/hyso-models-next`, HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`; index boş. B4/B5'in uncommitted
değişiklikleri geliştirmeden önce kullanıcıya bildirildi; stage/commit yapılmadı.
Evidence **E**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb6-20261020`.

## 1. Sınır ve mevcut risk

M2-P-A, backlog/dependency DAG, B1–B5 audit/rehberleri, performance/result/export,
Benchmark/P09, ModularCNN/attention ve packaging/minimum testleri incelendi.
P14 → P12/P13 hard dependency korundu: tek gerçek capture, sonra pure derivation.
Yeni bağımsız latency engine, Trainer instrumentation veya B7 özelliği yoktur.

[B4 koşullu kabul](m2-p-b4-conditional-acceptance.md) yalnız development continuation
içindir. Orijinal 0xC0000409 STACK_COOKIE_CHECK_FAILURE geçerli tarihsel failure'dır;
HYSo/Torch/NVIDIA kök nedeni **UNRESOLVED**, reliability risk **OPEN** kalır.
Bu görev sürücü/dependency değiştirmedi ve crash fix'i ilan etmez. Native fail-fast
tekrarlarsa admission/gate durdurulur; OOM veya sıradan profiler hatası sayılmaz.
Stable/public/PyPI release onayı verilmez.

Tarihsel CPU2313, targeted1595, native196/86 sayıları yeni koşuya taşınmadı.
Aşağıdaki E6 kanıtları bu görevin kaynakları üzerinde üretilmiştir.

## 2. Public API ve ownership

```python
ProfilingProtocol(warmup_iterations=1, measure_iterations=3, timeout_seconds=60,
    cpu_memory_limit_bytes=3*1024**3, input_limit_bytes=32*1024**2,
    cuda_memory_fraction=.1, max_events=20000, max_trace_bytes=16*1024**2)
profile_operators(ColdStartRecipe(...), protocol=..., trace_path=None)
profile_layers(operator_result_or_recipe, protocol=...)
analyze_flops(operator_result_or_recipe, protocol=...)
```

Frozen independent schema1 OperatorProfileResult/P14, LayerProfileResult/P13 ve
OperationCountResult/P12 eklendi. Recipe mevcut plain importable factory/SHA
state_dict sözleşmesini kullanır; exact NCHW ones input, CPU FP32/FP64 veya CUDA
FP32/AMP FP16 inference kapsamıdır. Caller'ın canlı modeli, Tensor/hook/stream/
device handle'ı taşınmaz. Worker kendi modelini eval/inference mode'a alır; caller
mode'u değişmez. Factory trusted, CPU construction ve dış yan etkisiz olmalıdır.

`profiling.py` worker admission/typed result/derivation; `_profile_worker.py`
Torch profiler + worker-local annotations/ATen observation; `_operation_models.py`
pure analitik maliyet politikalarıdır. B5 Windows Job Object private-commit guard'ı
ve B3 owned-process-tree termination yeniden kullanıldı. Guard Torch importundan
önce uygulanır; active-process limit1, explicit input/event/trace/forward/timeout
sınırları vardır. Linux guard unsupported; native Linux doğrulanmadı.

Factory-owned existing module hook registries korunur; yalnız capture handle'ları
finally kaldırılır. BenchmarkRunner'ın hook rejection kuralı değiştirilmedi veya
bypass edilmedi. Shared/nested module path aliases ve dynamic invocation-parent
ağacı ayrı saklanır. Canonical module identity kullanılır; aynı objeye ait iki
attribute alias'ından hangisinin Python call site'ta seçildiği iddia edilmez.

Derived API result girdisinde tekrar forward yapmaz; yeni UUID, source UUID ve
canonical sorted/compact JSON source fingerprint'i taşır. Bu fingerprint, pretty
JSON artifact byte SHA'sıyla aynı şey değildir. Trace/EManager SHA alanları gerçek
dosya byte SHA'sıdır. JSON/CSV ve explicit trace yayını atomic/no-clobber'dır.
Benchmark schema1, checkpointv1 ve EManager schemas değişmez.

## 3. P12 maliyet ve coverage politikası

Torch profiler `with_flops=False`; otomatik with_flops total'i kullanılmaz.
Versiyonları doğrulanan internal TorchDispatchMode gözlenen dış ATen çağrısının
gerçek operand/output shape ve scalar parametrelerini alır; backend'in nested
CPU profiler event'leri tekrar FLOP hesabına eklenmez.

- Dense Conv1d/2d, groups/depthwise: MAC=output.numel×weight.Cin_per_group×kernel
  elements; output shape gerçek forward'dan gelir. FMA için1 MAC=2 FLOPs;
  bias output başına ayrı1 addition.
- Linear/mm/matmul/bmm ve alpha1/beta0–1 addmm/baddbmm: gerçek K/output
  cardinality, ayrı beta1 bias/addition. Arithmetic add/sub/mul/div broadcast
  output cardinality ile sayılır.
- Adaptive avg pool integer bin sınırlarıyla; fixed avg/max pool yalnız
  nonpadded/non-ceil scope'ta modellenir. ReLU/max karşılaştırmaları ayrı sayılır.
- Layout/view/copy/cast için aritmetik FLOPs explicit0'dır; memory/time maliyeti0
  denmez. Custom namespace aynı adlı ATen operatorü sayılmaz.
- Normalization, SiLU/sigmoid/softmax, fused attention, transposed conv/einsum ve
  diğer unmodeled işlemler null/reason üretir. Unknown=0 dönüşümü yoktur.

Full scoped total yalnız bütün **gözlenen** çağrılar modellenmişse vardır;
aksi durumda model total null, supported subtotal ayrı kalır. Coverage ratio
gözlenen call-count oranıdır, FLOP yüzdesi veya hardware/kernel coverage değildir.
Her measurement_index için batch total ayrıca kaydedilir; per_batch tüm measured
forward'ların mean'i, per_sample bunun input batch cardinality'ye bölümüdür.
Dynamic control flow yalnız gerçekleşen input/seed/yollar için gözlenmiştir.

Gerçek örnekler: N2/C2/H5/W7, Conv2d2→4/kernel3/pad1/groups2/biasTrue için
**2520 MAC, 5320 FLOPs/batch, 2660 FLOPs/sample**. Shared Conv2d2→2/kernel1'in
üç çağrısı + ReLU + residual add örneği1820 FLOPs/batch'dir. Native ECA'lı
ModularCNN N1/C1/H5/W7 örneğinde gözlenen42 çağrının28'i modellenmiştir;
supported subtotal13682 FLOPs/batch, fakat model total **null**'dır. Bu bir
physical GPU instruction ölçümü değildir.

## 4. P13 CPU layer ve P14 CPU/CUDA ayrımı

Torch record_function annotation başına CPU inclusive event süresi alınır.
Module exclusive = inclusive − doğrudan child-module inclusive toplamı; bu,
ATen operator self CPU süresiyle aynı tanım değildir. Shared module tekrarları
ayrı invocation_id taşır, alias weights/calls kaybolmaz. Unexecuted module count0
olabilir; süre null/not_executed'dır. Sıralama instrumented CPU exclusive veya
operator self CPU sürelerine göredir; inclusive parent/child süreleri toplanıp
model latency yapılmaz. Input/output shape metadata gerçek çağrılardan gelir.

Yerel native probe ve source test: Windows, RTX3060 Laptop GPU6GB/capability8.6,
driver591.74, Python3.10.18/Torch2.5.1+cu121/CUDA build12.1. Gerçek CUDA forward
başarılıdır; `supported_activities()` yalnız CPU bildirir ve exported trace'te
kernel activity yoktur. Legacy FunctionEvent.kernels içinde `aten::conv2d` gibi
CPU etiketleri görüldü; bunlar kernel adı olarak alınmadı. Bu gözlem bir driver
kusuru veya fiziksel CUPTI kurulum tanısı değildir. Fallback ayrımı
[resmî profiler belgesinde](https://docs.pytorch.org/docs/stable/profiler.html)
açıklanır; çalışan build'in gerçek yeteneği yerel kanıtla sınırlandırıldı.

P14 CPU self/total/count/shapes/ranking gerçek profiler event'leridir.
CUDA kernel name/duration yalnız Chrome trace `cat=kernel, ph=X` gözlenirse
parse edilir; bu build'de **UNSUPPORTED**, kernel count/duration null'dır.
Başka build'de gerçek kernel activity okuyan koşullu parser yolu burada native
doğrulanmış değildir. Raw correlation id varsa yalnız raw id olarak saklanır;
güvenilir CPU→CUDA veya kernel→module attribution ilan edilmez. P13 CUDA layer
activity süresi bu yüzden null/unsupported'dır. Tam NVIDIA coverage iddiası yoktur.

Capture ayrı warmup/active schedule kullanır. Per-layer synchronize yoktur;
yalnız warmup sonrası/capture sonunda seçilen stream drain edilir. Shape capture,
hooks ve dispatch observation overhead yaratır. Collector lifecycle wall ms ve
instrumented host-forward windows ölçülür; bunlar uninstrumented latency veya
GPU kernel duration değildir. Matched baseline olmadan perturbation delta
**not_measured/null** kalır. Export/trace flush/SQLite model forward timing'ine
eklenmez. Per-layer allocation ölçümü yoktur (`profile_memory=False`).

## 5. Yeni testler ve hata kayıtları

Önce yeni B6 testleri çalıştı: final **38 CPU PASS**,0 skip,191.58s;
E/b6-final.xml. Grouped/depthwise Conv1d/2d+bias, matrix operations, nontrivial
pool bins, batch/resolution, custom namespace ve unknown normalization/activation,
shared/nested/repeated modules, units, active-profiler exception/state cleanup,
timeout/evidence cap, parent RNG/state/hook, immutable finite JSON/CSV, trace
SHA/no-clobber ve EManager copied artifact/metrics/SHA/reopen kapsandı.

İlk CPU koşusu13 PASS ardından trace SHA failure verdi: read_text Windows CRLF'yi
LF'ye normalize ediyordu. Özgün trace bytes decode edilerek mevcut newline-preserving
atomic exporter'a aktarıldı; helper/Benchmark export sözleşmesi değiştirilmedi.
Sonraki30 PASS ve final38 PASS bu düzeltmeyi içerir. Ön sonuçlar toplanıp38'e
eklenmez; E/b6-first.*, b6-second.* korunur. Test toleransı/skip/xfail gevşetilmedi.

Yeni native source doğrulamasında **6 scenario PASS** (E/cuda-b6-source.json):
FP32, AMP FP16, ECA ModularCNN, gerçek active-profiler forward exception cleanup,
profiler_active phase'ta15s timeout ve sonraki fresh capture. Parent parameters/
buffers/gradients/mode, Python/NumPy/CPU+CUDA RNG, non-default stream/device,
hook registries ve allocated/reserved/peak counters eşit kaldı. Worker/launcher
PID ölüm kontrolleri geçti. Exception normal exit0 diagnostic, intentional timeout
forced exit1'dir; bunlar native fail-fast sayılmaz. Beklenmeyen native crash yoktur.

Native scenario testlerinin PASS olması CUDA kernel özelliğinin PASS olduğu
anlamına gelmez: activity yokluğu doğru unsupported davranışla doğrulanmıştır.

## 6. Regresyon ve packaging

Yeni testlerden sonra ilgili B1–B5/Benchmark/P09/Trainer-checkpoint/EManager/
public/package regression selection çalıştı. İlk pytest1593 PASS,1 FAIL verdi;
sdist installed case henüz çalışmadı. Failure yeni rehberin collector metriğinde
observation axis eksikliğiydi (`invalid_axis`); EManager API/schema değiştirilmedi.
Rehbere explicit step0/observation eklendi. **Production code değişmedi.**

Aynı final archives ile mevcut dört test_packaging test fonksiyonu hiçbir assertion
değiştirilmeden TEMP runner'da yeniden çalıştı: configuration/archive/wheel/sdist
**4 PASS**. İki önceki packaging PASS duplicate execution'dır. Sonuç bu görevin
kaynakları için **1595 distinct regression checks +38 yeni =1633 CPU controls PASS**;
tek pytest invocation1633 PASS diye sunulmaz. E/regression.xml ilk failure'ı,
packaging-reuse.json düzeltme sonrası original-function sonuçlarını korur.

Wheel/sdist34 production modülünün tam envanterini içerir; tüm `.py` dosyaları
final kaynakla byte-equal. Yeni rehber gerçek CPU FP32 ModularCNN ve FP64 grouped
Conv profillerini, raw trace SHA ve EManager akışını installed wheel/sdist'te yürüttü.
E/archive-integrity.json:

- Wheel151073 byte, SHA256
  `bd0540fb2fe3367e30118e547e2bb01bb9e061b6c8ba6a932f3e24599659c084`.
- Sdist137512 byte, SHA256
  `c2addf2f522638d099fe1e538d6b7028a40ec197ffa5c585a8f118d55f71fe7e`.

Tarihsel full CPU2313/native196 gate yeniden koşuldu sayılmaz.

## 7. Minimum installed smoke

Mevcut Python3.10.18 / Torch2.2.2+cpu / torchvision0.17.2+cpu ortamında aynı final
wheel ile **PASS**. Rehberin FP32 ModularCNN ve FP64 grouped Conv capture'ları,
P12/P13 pure derivation, trace byte SHA, atomic exports, EManager metrics/artifact
SHA/close/reopen ve önceki B1–B5/Trainer smoke'u çalıştı. Parent CUDA initialized
False kaldı; worker package_source installed target altında doğrulandı.

Tüm34 installed production modülü kaynak/archives ile byte-equal;
minimum environment METADATA/RECORD hash envanteri önce/sonra eşit.
Dependency kurulumu/güncellemesi yoktur: yalnız kendi yerel HYSo wheel'i TEMP
target'a --no-index/--no-deps/--no-compile ile kurulmuştur. Kanıtlar
E/minimum-smoke.json, minimum-environment.json, installed-integrity.json ve
installed-commands.json içindedir.

Minimum installed CPU rehberinin gerçek capture'ından örnek sıralama (iki active
forward boyunca aggregate instrumented CPU µs): layer exclusive stem.conv 2705,
stem.norm 1527, stages.0.0.attention.pool 1237; operator self CPU
aten::mkldnn_convolution 1627, aten::conv2d 1408, aten::batch_norm 1078.
Kaynak E/minimum-smoke.stdout.log'dur. Bunlar bare model latency veya GPU bottleneck
değildir; başka run/runtime ile hız karşılaştırması olarak kullanılmaz.

## 8. Final installed-native sonuçları

Aynı final wheel'den tüm seçili native çağrılar normal outer exit0 ile tamamlandı.
E/installed-commands.json exact argv/cwd/exit/wall sürelerini; ilgili report.json
ve trace dosyaları gerçek event/candidate kanıtını taşır.

| Native doğrulama grubu | Scenario PASS |
|---|---:|
| Yeni B6 | 6 |
| B5 batch capacity | 5 |
| Benchmark/P09 | 45 |
| B1 | 15 |
| B3 | 8 |
| B4 | 11 |
| Experiment Manager CUDA | 2 |
| Toplam CUDA validation scenario | 92 |
| Ayrıca Windows P10 | 3 |

B6 source6 ve installed6 aynı logical ID'lerin iki execution'ıdır; toplam12 diye
sayılmaz. CUDA rehberi de final installed target'ta çalıştı; Conv1×1 için252
logical FLOPs/batch ve actual GPU output doğrulandı. Demonstration ek scenario
sayılmadı. Kernel activity dört completed B6 capture'ında da unsupported'dır;
6 scenario PASS doğru missing-activity/cleanup davranışının PASS'idir.

Aktif profiler timeout'u ve controlled forward exception negatif testleridir;
beklenmeyen native crash, model corruption veya eski davranış regresyonu
saptanmadı. Bu, orijinal B4 fail-fast olayını geçersiz veya çözülmüş yapmaz.
E/verification-summary.json final kaynak/hash/protection/test birleşimidir.

## 9. Durum, koruma ve sonraki sınır

- **P12 PARTIAL:** doğrulanan analitik Conv/Linear/matrix/pool/arithmetic alt
  kapsamları PASS; genel model coverage unmodeled işlemler yüzünden partial
  olabilir. Unsupported maliyet sıfır değildir.
- **P13 PARTIAL:** CPU hierarchy/shared/nested/inclusive/exclusive/shapes/ranking
  PASS; CUDA layer activity ve per-layer allocation ölçümü unsupported/not measured.
- **P14 CPU PASS; P14 CUDA kernel UNSUPPORTED.** Kernel visibility/correlation
  doğrulanmadığından bütün P14 implemented diye işaretlenmedi.
- **B6 PARTIAL; tam milestone için NO-GO.** M2-P-A B6 kapısındaki "PARTIAL GPU
  activity eksik" koşulu gerçekleşmiştir. Desteklenen CPU/analitik API'lerin
  doğrulanmış olması tam GPU kernel scope'u veya stable release onayı değildir.

Backlog P12/P13/P14 partial, toplam **12 implemented /4 partial /19 planned**.
B7 uygulanmadı ve tam B6 phase gate kapanmadan B7'ye geçiş önerilmez. En küçük
eksik native kanıt, gerçek CUDA kernel activity üreten destekli bir profiler
backend'inden küçük Conv/Linear trace'i ve varsa güvenilir correlation doğrulamasıdır.
Bu görevde SDK/driver/dependency değiştirilmedi; unsupported alan workaround veya
sahte isim/süreyle doldurulmadı. Normalization/nonlinear maliyet modelleri ve
uninstrumented overhead attribution da daha geniş scope için ayrı kanıt ister.

B4 reliability risk **OPEN**, root cause **UNRESOLVED**; ayrı stable release gate
zorunludur. Hiçbir vendor/HYSo kusuru veya fix kanıtı üretilmedi.

Gradient dosyası başlangıç/final SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Başlangıçtaki kullanıcı farkı aynen korundu. B1–B5 audit/guide/ölçüm çekirdekleri,
B4 conditional/R1/R2/R3 kayıtları ve eski testler byte-identical. Başlangıçta var
olan dosyalardan yalnız performance exports/result dispatch, packaging ve backlog
additive değişti. Branch/HEAD/index korunur; commit/push/stage/checkout/reset/
rebase yapılmadı. Model/weights/dataset/dependency/driver indirmesi/güncellemesi,
dump upload veya B7+ geliştirmesi yoktur.
