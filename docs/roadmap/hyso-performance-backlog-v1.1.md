# HYSo Performance Backlog v1.1 — Onaylı genişleme kapsamı

Tarih: 2026-10-19. Branch `feature/hyso-models-next`, kaynak HEAD
`979e9285cd9f839e408e3058fed42de3fefb1ed3`.
Kullanıcı aşağıdaki **35 özelliğin gelecekte geliştirilmesini** açıkça onayladı.
Bu belge yeni performans kapsamını ve önceliklerini taşır; paket sürümü değildir.
HYSo0.2.0 ve [sabit roadmap v1.0](hyso-architecture-roadmap-v1.md) değişmez.
[M2-C-A](../audit/m2-c-a-benchmark-engine-design.md) ve önceki milestone durumları
geriye dönük değiştirilmez. M2-C-B1 CPU inference çekirdeğidir; M2-C-B2
native CUDA event/FP16 autocast ve opt-in allocator memory probe ekler.
M2-C-B3 explicit JSON/summary CSV/sample CSV ve koşul uyumluluğu raporu ekler;
quality/regression/Pareto sistemi değildir.

Durumlar: `planned`=henüz benchmark özelliği olarak uygulanmadı;
`partial`=belirtilen alt kapsam var; `implemented`=bütün satırın doğrulanmış
sözleşmesi var; `optional-hardware`=donanım bağımlı uygulama ayrıca onaylanmışsa
kullanılacak durum. B3 sonunda P01/P08/P11/P29 partial, P09 sınırları belirtilmiş
native allocator sözleşmesiyle implemented, kalan30 satır planned'dır.
Donanım opsiyonelliği araç sütununda ayrıca belirtilir.

M2-P-B1 (2026-10-20) sonrasında P01/P02 aşağıdaki doğrulanmış resident inference
kapsamıyla implemented'dır. B1 sonrası sayım: **3 implemented, 3 partial, 29 planned**.
B1'de P09 ve P03–P35'in diğer durumları değişmedi. Kanıt ve sınırlar:
[P99/throughput audit](../audit/m2-p-b1-latency-throughput.md),
[kullanım rehberi](../guides/performance-inference.md).

M2-P-B2 (2026-10-20): P10 **native Windows current-process RSS/HWM** ve P11
**dense CPU logical bytes + gerçek artifact bytes/SHA** kapsamıyla implemented.
Linux adapter'ı parser/protocol testlidir; native Linux/Torch doğrulaması yoktur,
fully verified platform kapsamı Windows'tur. B2 sonrası sayım:
**5 implemented, 2 partial, 28 planned**. Diğer P durumları değişmedi.
[Memory/footprint audit](../audit/m2-p-b2-memory-footprint.md),
[kullanım rehberi](../guides/performance-memory.md).

M2-P-B3 (2026-10-20): P03/P04 **bounded blocking serial CPU/CUDA pipeline ve
ayrı host/event stage clocks**, P05 **fresh interpreter CPU/CUDA cold phases**
kapsamıyla implemented. Native doğrulama Windows/RTX3060 üzerinde; overlapped
scheduling, kamera/drop policy ve cold OS page-cache garantisi kapsam dışıdır.
B3 sonrası sayım: **8 implemented, 2 partial, 25 planned**. Diğer statüler değişmedi.
[Pipeline/cold-start audit](../audit/m2-p-b3-pipeline-cold-start.md),
[kullanım rehberi](../guides/performance-pipeline.md).

M2-P-B4 (2026-10-20): P06/P07 explicit finite batch/resolution sweeps ve P08
doğrulanan CPU FP32/FP64/BF16/dynamic-qint8 Linear (x86), CUDA FP32/AMP FP16/BF16
profilleriyle implemented. Native CPU BF16 instruction acceleration veya genel
INT8 graph desteği iddia edilmez; P28 accuracy ayrı kalır. Güncel sayım:
**11 implemented, 1 partial, 23 planned**. **Birleşik B4 kalite kapısı geçmedi:**
eski native trainer-replay süreci 0xc0000409 ile çöktü; B5'e geçiş kapalıdır.
Feature statüleri kendi doğrulanmış sözleşmelerini gösterir, milestone GO değildir.
[Sweep/precision final gate audit](../audit/m2-p-b4-sweeps-precision-final-gate.md),
[kullanım rehberi](../guides/performance-sweeps.md).

B4 koşullu kabul (2026-10-20): kullanıcı kök nedeni bilinmeyen native crash
riskini açıkça kabul etti. Güncel development kararı **B4: CONDITIONAL GO —
development continuation only**; yukarıdaki tarihsel strict NO-GO kanıtı geçerlidir.
Yalnız geliştirmeye devam kapısı açıldı; crash çözülmüş veya public stable release
onaylanmış değildir. **Risk OPEN / root cause UNRESOLVED.** Tekrarlayan native
crash/regresyon/state corruption kapıyı durdurur; stable öncesi ayrı release gate
zorunludur. [Koşullu kabul](../audit/m2-p-b4-conditional-acceptance.md).

M2-P-B5 (2026-10-20): P17 **explicit finite ordered inference capacity search,
Windows bounded fresh CPU/CUDA workers** kapsamında implemented. CPU FP32/FP64,
CUDA FP32/AMP FP16, typed OOM/unknown crash ayrımı, stop policies ve sparse-list
observed-max/exhaustive-range proof ayrımı doğrulandı; optional adaptive search,
Linux guard, training ve physical/global maximum iddiası kapsam dışıdır.
Güncel sayım **12 implemented, 1 partial, 22 planned**. Final hedefli CPU: 1595 PASS;
installed wheel CUDA: 86 scenario PASS (B5'in 5'i dahil); Windows RAM: 3 scenario PASS.
Minimum Torch 2.2.2+cpu ve 31 modül wheel/sdist byte-equality PASS. Bunlar tarihsel
full CPU 2313/native 196 sonuçlarının otomatik tekrarı değildir. **B5: PASS within
verified scope, B4 risk inherited.** Stable release onayı yoktur; B4 risk OPEN.
B6 uygulanmadı. [P17 audit](../audit/m2-p-b5-max-feasible-batch.md),
[kullanım rehberi](../guides/performance-max-batch.md).

M2-P-B6 (2026-10-20): P12/P13/P14 **partial**. P12 gözlenen ATen için doğrulanmış
Conv/Linear/mm/bmm/arithmetic/pool modelleri sunar; normalization/SiLU/sigmoid/
softmax/custom maliyetleri unknown ve model total null kalabilir. P13 worker-only
CPU invocation hierarchy/shared aliases/inclusive-exclusive timing doğrulandı;
CUDA layer activity ve per-layer allocation ölçülmez. P14 gerçek CPU operator
trace/self-total/count/shapes üretir; bu Windows/Torch build'inde gerçek CUDA
kernel activity **UNSUPPORTED**, correlation doğrulanmadı. Legacy CPU etiketleri
kernel sayılmaz. Güncel sayım **12 implemented, 4 partial, 19 planned**.
B6'nın desteklenen CPU/analitik kapsamı kullanılabilir; GPU activity eksikliği
nedeniyle tam B6 milestone **PARTIAL / full-scope NO-GO**. B7'ye geçilmedi.
B4 reliability risk OPEN, root cause UNRESOLVED ve ayrı release gate korunur.
[B6 audit](../audit/m2-p-b6-flops-layer-operator-profiling.md),
[kullanım rehberi](../guides/performance-profiling.md).

## Uygulama sırası

M2-P-B7 (2026-10-20): P15 **partial**; Windows bounded fresh worker'da gerçek
pageable/pinned, blocking/nonblocking H2D/D2H, exact logical bytes, ayrı host
enqueue/CUDA event interval ve terminal equality doğrulandı. P14 gerçek copy/
compute timeline sağlamadığı için overlap null/unsupported; saf PCIe peak veya
kanıtlanmış async overlap iddiası yok. P16 **implemented within verified scope**:
Windows process kernel/user ve ayrı system CPU counters; logical-core açık
normalizasyon; kurulu read-only nvidia-smi/NVML ile physical-device busy,
memory-controller busy ve VRAM timeline. Process GPU/Tensor Core counters ve
kesin sensör pencere sınırı unsupported. Güncel toplam **13 implemented,
5 partial, 17 planned**. B7 **PARTIAL / full-scope NO-GO**, yeni conditional
acceptance ilan edilmedi. B4 OPEN/UNRESOLVED ve B6 partial sınırları korundu;
B8'e geçilmedi. [B7 audit](../audit/m2-p-b7-transfer-utilization.md),
[rehber](../guides/performance-transfer-utilization.md).

| Öncelik/aşama | Kapsam ve kapı |
|---|---|
| Ö1 | Temel CPU/CUDA Benchmark: M2-C-B1 CPU; M2-C-B2 native event/AMP/memory |
| Ö2 | JSON/CSV ve adil comparison: M2-C-B3; ardından M2-C-C kalite kapısı |
| Ö3 | Batch/Resolution/Precision Sweep; cold-start ve ayrı zaman penceresi protokolleri |
| Ö4 | Layer-wise/Operator Profiling ve Roofline; profiler/counter araç doğrulaması |
| Ö5 | Training Performance Analysis; inference'tan ayrı state/update sözleşmesi |
| Ö6 | Model Quality/Robustness Evaluation; sabit veri/split/metric protokolleri |
| Ö7 | Real-world/Edge/Deployment Analysis; cihaz/provider/camera adapter kanıtları |
| Ö8 | Long-running/Regression/Cost Analysis; sürüm/provenance ve donanım telemetrisi |

Bu sıra genişlemenin geliştirme önceliğidir; v1.0 dışında otomatik yeni mimari
fazı, kurulmuş dependency veya gerçekleştirilmiş deney anlamına gelmez.

## A. Inference ve hız analizi

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P01 | Inference Latency — P50/P95/P99 | Mevcut schema1 mean/p50/p90/p95/std korunur; hyso.performance.analyze_latency aynı raw R×N'den ayrı versioned P99 analizi üretir | ms/batch, raw samples/quantiles, tail status | CPU monotonic clock; native CUDA event kaynağı | Sabit shape/device/precision, warmup, repeat/sync scope; default n≥1000 ve (1−q)n≥10 | Ö1 / M2-P-B1 | implemented | Linear [1..1000] P99=990.01; n=999 null/insufficient_samples; no rerun/mutation; repeat/source provenance, gerçek CPU/CUDA sample ve installed minimum/package kanıtı; CI/independence iddiası yok |
| P02 | Real Throughput / FPS | hyso.performance.measure_throughput: fixed resident NCHW tensor, continuous completion-window; mevcut derived rate değişmez; camera/display/E2E kapsam dışı | images/s; completed images / host wall seconds | CPU FP32/FP64; native CUDA FP32/AMP FP16, bounded event/output ring | Explicit batch-count veya duration admission; concurrency=1, queue depth, warmup/repeat, tail completion; finite-check/queue overhead scope'ta | Ö1 sonrası ayrı protokol / M2-P-B1 | implemented | Gerçek synchronous CPU ve CUDA terminal drain/non-default stream; count ledger, pending=0, tail paydada, bounded lifetime/no per-forward sync; state/RNG, export/artifact reopen ve installed minimum/package PASS; hidden cross-stream/device unsupported |
| P03 | End-to-End Latency | measure_pipeline: aynı ID'nin read-start→postprocess completion host wall süresi; optional delivery ayrı | ms/ID, P50/P95 ve eligible P99 | Synchronous CPU; native CUDA FP32/AMP FP16, terminal event | Bounded blocking queue, unique ID, warmup/repeat ve explicit stage/device | Ö3 / M2-P-B3 | implemented | Gerçek CPU/CUDA completion, queue wait/count/tail ve same-ID reference; dropped=0/pending=0 serial policy, stage sum E2E değildir; hidden streams/overlap/camera/drop unsupported; minimum installed/package PASS |
| P04 | Preprocessing / Postprocessing Time | PipelineStage adapters; analyze_stages raw evidence'dan ayrı result üretir | ms/stage, host launch/completion ve CUDA event ayrı | Caller CPU transforms / selected-stream CUDA stages | Ordered boundaries, execution counts, explicit overhead/sync | Ö3 / M2-P-B3 | implemented | Gerçek resize/normalize/post transform, stage order/count/boundaries; native event/default/non-default stream ve terminal completion; model tekrar çalışmaz; host/event clocks toplanmaz |
| P05 | Cold Start & Model Loading | measure_cold_start: her trial fresh interpreter; import/construction/load/prepare/first/warm ayrı | ms/phase, raw trial outcomes | Trusted importable recipe; SHA-verified weights_only state_dict | Parent receipt ve worker clock domain ayrı; page-cache unknown | Ö3 / M2-P-B3 | implemented | Gerçek fresh CPU/CUDA worker; FP32/AMP, safe artifact load, timeout/tree cleanup/crash/missing/corrupt/unsupported kanıtı; parent state ve process ownership; native Windows ve minimum Torch installed PASS, OS cache cold garantisi yok |
| P06 | Batch Size Sweep | run_sweep axis=batch: explicit noktalar, independent owned copies ve warmup | ms/batch, optional completion images/s ve P09 ref | Gerçek CPU/CUDA FP32/AMP, exact NCHW input factory | Immutable SweepResult, point/source/result IDs, continuation flag; OOM stop | Ö3 / M2-P-B4 | implemented | Batch 1/2/4/8 ve installed 1–32 gerçek forward; cardinality/W/N/R, failure/not_run, state/RNG/lifetime, export/E reopen/minimum/package kanıtı; cross-batch compatibility otomatik değil |
| P07 | Resolution Sweep | Aynı sweep altyapısı; explicit H,W ve W/H aspect, fixed batch/model/precision | Shape→ms/rate; quality not measured | CPU/CUDA, exact shape factory ve gerçek model constraints | Her noktada bağımsız warmup; silent resize/interpolation yok | Ö3 / M2-P-B4 | implemented | Dikdörtgen 5×7/9×13 ve installed 320/416/640/640×384/960 noktaları; actual forward, invalid shape continuation/OOM stop, source hash ve export/minimum/native kanıtı |
| P08 | Precision Benchmark | Isolated PrecisionResult: CPU FP32/FP64/BF16/dynamic INT8 Linear; CUDA FP32/AMP FP16/BF16 | ms/batch, completion images/s, ayrı copy/conversion ms | Native-verified CPU PyTorch eager/quantized x86 ve RTX3060 CUDA; diğer engine/platform doğrulanmış sayılmaz | Master/input/output/autocast dtype, recipe/calibration not_applicable, actual operator proof | Ö1/Ö3 / M2-P-B4 | implemented | BF16 actual output/capability, qint8 packed Linear ve quantized::linear_dynamic preflight, no fallback; source state/RNG, minimum Torch2.2.2 installed PASS; CPU acceleration/general INT8 graph ve P28 accuracy iddiası yok; genel B4 legacy replay gate başarısızlığı ayrı |

## B. Donanım, bellek ve darboğaz analizi

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P09 | GPU Peak Memory | Hedef process/device native allocator baseline, önceki/mutlak peak, incremental peak ve output release sonrası değerler; total VRAM değildir | allocated/reserved byte, separate_single_forward scope | Native CUDA allocator API; RTX3060 üzerinde doğrulandı | Model/input resident, sync, opt-in geri alınamayan peak reset ve caller/process exclusivity | Ö1/B2 | implemented | Native baseline/reset/probe/output-release; peak>=baseline; disabled no-reset/null; prior peaks ve reset diagnostics; allocator vs driver/other process ayrımı |
| P10 | CPU Peak RAM | measure_memory: current-process baseline/current RSS, timestamped samples, sampled peak ve ayrı lifetime HWM; native Windows working set doğrulandı; Linux adapter native-unverified | byte, ns timestamps/query/gaps, interval/window seconds | Windows GetProcessMemoryInfo stdlib ctypes; Linux /proc/self/status parser/protocol testli, native Linux kanıtı yok | Caller-exclusive synchronous workload; sampled peak mutlak peak değil; lifetime HWM reset yok; thread/GIL/native-query overhead explicit | M2-P-B2 / Ö4/Ö8 | implemented | Gerçek Windows CPU tensor allocate/touch/release, bounded sample budget, sampler/process failure cleanup, ctypes cache-stability, immutable export/E reopen, minimum Torch2.2.2 installed/package PASS; unsupported null+reason, tracemalloc fallback yok |
| P11 | Model Footprint | measure_footprint: mevcut unique logical parameter/buffer convention; tied/view/cross-category diagnostics; explicit state_dict veya opaque caller artifact için gerçek stat/SHA; physical memory unsupported | parameter/buffer elements ve object count, logical byte, serialized artifact byte/SHA/format | Ordinary dense CPU tensors, mixed dtype/int/bool buffers; stdlib streamed SHA/stat; Torch own state_dict serialization | No-clobber temporary/atomic publication; optional weights_only round-trip yalnız API'nin ürettiği dosyada; caller artifact format/model match unverified | M2-P-B2 / Ö1/Ö3 | implemented | Native Windows tied/view/mixed dtype/nonpersistent/shared-module tests; actual file stat/SHA, corrupt/mutation/failure cleanup; model/state/RNG, E association/reopen, minimum installed/package PASS; existing Benchmark logical result unchanged |
| P12 | FLOPs / MACs | analyze_flops: actual observed ATen shape/scalar logical models, unknown coverage explicit | MAC/FLOP batch/sample/per-forward, supported subtotal, unknown/null model total | Windows CPU FP32/FP64; native CUDA FP32/AMP observed inference | FMA1MAC=2FLOPs; bias/arithmetic/comparisons ayrı; actual paths/recipe only | Ö4 / M2-P-B6 | partial | Conv1d/2d grouped/depthwise, Linear/mm/matmul/bmm/default addmm, pool/add gerçek analytic references PASS; normalization/nonlinear/fused/custom cost unmodeled; hardware FLOP counter değil |
| P13 | Layer-wise Profiler | profile_layers: worker-only canonical module/aliases + dynamic invocation tree | instrumented CPU inclusive/exclusive us, call count/shapes/ranking | Torch profiler module annotations; no caller hook mutation | Direct-child subtraction; existing worker hooks preserved; no per-layer sync | Ö4 / M2-P-B6 | partial | Nested/shared/repeated CPU calls ve exception cleanup PASS; CUDA layer activity null/unsupported; per-layer allocation ve uninstrumented perturbation ölçülmedi |
| P14 | Operator / Kernel Profiler | profile_operators: actual CPU events + optional exported kernel activity | CPU self/total us/count/shapes, immutable result/trace SHA | Windows Torch2.5.1 CUDA legacy/CPU; minimum Torch2.2.2 CPU verified | Only actual cat=kernel trace qualifies; no legacy label as kernel; raw correlation unverified | Ö4 / M2-P-B6 | partial | CPU profiler PASS, real GPU forwards PASS; this build supported_activities=CPU, CUDA kernel count/time null/UNSUPPORTED; full NVIDIA coverage veya reliable attribution iddiası yok |
| P15 | CPU–GPU Transfer Analysis | measure_transfers: explicit owned dense contiguous cuda:0 buffers | logical byte, host enqueue ms, CUDA stream interval ms, completed byte/s | Windows guarded fresh worker; pageable/pinned ve blocking/nonblocking | Exact numel×element_size, preallocated buffers, warmup/repeat, terminal/equality; P14 overlap hard dependency korunur | Ö4 / M2-P-B7 | partial | Gerçek H2D/D2H FP32/uint8 completion/equality, installed/minimum/export PASS; host/event clocks ayrı; overlap null/unsupported, event interval CUPTI memcpy veya saf PCIe peak değildir |
| P16 | GPU / CPU Utilization | measure_utilization: bounded OS/sensor timestamped samples | process kernel/user delta, single-core/machine %, ayrı system %, device GPU busy/controller busy/VRAM | Windows GetProcessTimes/GetSystemTimes; optional installed nvidia-smi/NVML read-only | Deadline/sample/query budget, raw times/gaps/reset/unavailable; workload label attribution değildir | Ö4/Ö8 / M2-P-B7 | implemented | Native CPU idle/load ve physical GPU idle/controlled-load samples, parsed UUID/PCI/driver, missing query/null ve installed/minimum/E SHA PASS; process GPU/Tensor Core counters unsupported; GPU busy occupancy'den türetilmez; Linux native scope doğrulanmadı |
| P17 | Maximum Feasible Batch Size | Explicit finite ordered inference candidates; observed max ile exhaustive 1..cap proof ayrı | per-point actual batch/latency/memory/outcome, observed/proven-within-range | Windows fresh bounded CPU FP32/FP64 ve CUDA FP32/AMP FP16 worker | Hard private-commit/input budget, CUDA allocator fraction, total-worker timeout, stop policies; no monotonicity assumption | Ö3 / M2-P-B5 | implemented | Gerçek CPU/CUDA forward, controlled allocator OOM→fresh feasible worker, runtime-prepared timeout/PID cleanup; immutable export/E reopen/minimum Torch/package PASS; sparse list global max değildir; adaptive search/Linux/training scope dışı; B4 risk OPEN |
| P18 | Arithmetic Intensity & Roofline Analysis | Gerçek compute/traffic verisiyle bounded teşhis; tahmini parametre bytes trafik sayılmaz | FLOP/byte, FLOP/s, GB/s, roofline grafiği | Opsiyonel doğrulanmış hardware counters/traffic ve compute ölçümü | P12/P14; aynı memory hierarchy, kalibre bandwidth/compute tavanı | Ö4 | planned | Güvenilir DRAM/cache traffic ve FLOP scope'ları eşleşmeli; sayaç yoksa yalnız tahmin etiketi, gerçek Roofline teşhisi yok |

## C. Training Performance

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P19 | Epoch / Training Step Time | measure_training ile gerçek Trainer epoch/microbatch/window/attempt timing | Host ms ve epoch s; CUDA selected-stream event ms ayrı | Native CPU FP32/FP64 ve doğrulanan CUDA FP32/AMP; bounded recorder | Existing Trainer lifecycle/counters; ayrı validation ve checkpoint I/O exclusion | Ö5 / M2-P-B9 P19 alt kapsam | implemented | Gerçek uneven sample ledger, applied/skip, exact state/RNG/history/replay, bounded events, export/EManager ve minimum/package gate; kernel attribution/P20 stages/P22 rates yok; B4 riski açık |
| P20 | Forward / Backward / Optimizer Breakdown | Gradient step aşamalarının ayrı süreleri | ms/stage, full-step ratio | Native autograd/optimizer/scaler | P19; clipping/unscale/loss/zero_grad boundaries | Ö5 | planned | Gerçek backward/update; AMP skipped/applied ayrımı, profiler overhead ve stage overlap açıklanır |
| P21 | DataLoader Bottleneck Analysis | Load/decode/transform/queue wait ile compute ayrımı | ms/batch wait, samples/s, timeline | Gerçek dataset, worker/pin settings | Worker RNG/epoch order/profiling adapter | Ö5 | planned | Worker ve ana süreç scope'ları ayrılır; bekleme ile CPU compute eşit sayılmaz |
| P22 | Training Throughput | Gerçek training penceresinde tamamlanan örnek ve applied update hızı | samples/s, applied updates/s | Loss/data/optimizer/scaler | P19; accumulation ve overflow count | Ö5 | planned | Sample-weighted gerçek count/window; attempt ile applied update ayrı |
| P23 | Training Memory Analysis | Params/gradients/optimizer/activation ve measured peak ayrımı | logical byte + process/allocator byte | Training state ve platform memory araçları | P09/P10/P19; gradient lifecycle | Ö5 | planned | Aynı training boundary'de native snapshot/probe; component logical toplam process RSS/VRAM diye sunulmaz |
| P24 | Time to Target Accuracy | Tanımlı kalite eşiğine ilk ulaşma zamanı | s/epoch/update to threshold, censored runs | Labeled train/validation data, repeated seeds | P19/P26; metric/threshold/eval cadence | Ö5/Ö6 | planned | Aynı split/target; ulaşılamayan run censored; validation/clock scope'u ve run variance açık |
| P25 | Gradient / AMP Diagnostics | Gradient norm/finite, scale growth/backoff ve applied ownership analizi | norm, scale, skip/update count | Native optimizer/GradScaler; uygun CUDA | P19/P20, güvenli gözlem ve overhead | Ö5 | planned | Gerçek overflow/recovery; hook/diagnostic zamanı ölçümle ayrılır; mevcut Trainer testleri benchmark uygulaması sayılmaz |

## D. Accuracy, Optimization ve Comparison

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P26 | Accuracy / Precision / Recall / F1 | Model kalitesinin ayrı evaluation protokolü | %, oran, confusion matrix, averaging policy | Labeled veri, class mapping, predictions | Split/label/threshold/macro-micro tanımı | Ö6 | planned | Bilinen confusion-matrix reference; missing class/zero division policy; latency kalite kanıtı değildir |
| P27 | Robustness Benchmark | Belirli perturbation/domain değişiminde kalite dayanıklılığı | metric drop/curve, corruption severity | Labeled robustness set/transforms | P26, perturbation/seed/severity protokolü | Ö6 | planned | Clean ve perturbed paired samples; reproducible order; accuracy değerlendirmesinden ayrı protocol |
| P28 | Quantization / Precision Accuracy Drop | FP32→FP16/INT8 kalite farkı | yüzde puan ve göreli % değişim, baseline=0 durumu | Aynı labeled evaluation set; uygun backend/calibration data | P08/P26, aynı preprocessing/output semantics | Ö6 | planned | pp=100×(new-base) oran metrikleri için; relative%=100×(new-base)/base; base0 için null+reason, ikisi karıştırılmaz |
| P29 | Accuracy–Speed–Memory Trade-off & Regression | B3 JSON/CSV ve timing/memory koşul eşliği temeli var; quality/Pareto/regression henüz yok | koşul eşliği raporu; curves/deltas sonraki kapsam | Ö2 schema1 export/comparison; quality metrics ileride | P01/P09/P10/P26; uyumlu input/runtime/precision | Ö2 temeli, Ö6/Ö8 analiz | partial | Compatible/incompatible/insufficient_metadata, raw samples korunumu; memory ayrı eligibility; accuracy/significance/ranking/regression iddiası yok |

## E. Gerçek dünya, edge ve deployment

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P30 | Real Camera FPS & Dropped Frames | Capture/inference/display kuyruğunda gerçek frame hız/kaybı | FPS, dropped/late count, latency | Opsiyonel kamera ve capture adapter | Capture timestamps, queue/backpressure/drop policy | Ö7 | planned | Frame ID conservation: received=processed+dropped+pending; derived service-rate camera FPS sayılmaz |
| P31 | Long-Running Stability | Uzun süreli hata/gecikme/throughput kararlılığı | time series, failure/recovery report | Uzun-run harness, sabit workload | Ö1/Ö2; duration/health/thermal politikası | Ö8 | planned | Kesintisiz süre ve tüm failures görünür; kısa smoke uzun-run kanıtı sayılmaz |
| P32 | Memory Leak / Memory Growth Analysis | Warmup sonrası canlı allocation/RSS trendi | byte/time slope, snapshots | OS RSS ve/veya CUDA allocator araçları | P09/P10/P31; cache plateau ve ownership | Ö8 | planned | Cache/reserved artışı leak ilan edilmez; canlı referans ve tekrar/profiling kanıtı gerekir |
| P33 | Power & Energy per Inference | Zamanla entegre güçten iş başına enerji | watt, joule/inference, sampling error | Opsiyonel güç sensörü/NVML/RAPL/harici sayaç | Counter domain, baseline idle ve sample interval | Ö8 | planned | Gerçek güç integrali ve completed count; cihaz toplamı vs model attribution açıklanır; sensör yoksa null |
| P34 | Multi-Device Comparison | Aynı protokolün cihazlar arasında koşullu kıyası | device→latency/quality/memory/energy table | Gerçek farklı cihazlar ve provenance | Ö2; supported runtime/precision/scope mapping | Ö7 | planned | Her cihazda gerçek run; farklı precision/backend koşulları görünür; eksik cihaz için tahmin/PASS yok |
| P35 | Execution Provider Fallback Detection | Deployment node'larının beklenmeyen provider'da çalışmasını saptama | node→provider execution evidence, fallback report | Gelecekte onaylanacak ONNX Runtime/profiling adapter | Gerçek model graph/session trace ve node mapping | Ö7 | planned | Gerçek node execution/profiling kanıtı; yalnız provider listesine bakarak fallback ilan edilmez |

## Ortak güvenilirlik kuralları

Derived throughput ve gerçek sustained throughput farklı protokollerdir.
Peak allocated total VRAM değildir; reserved/driver/OS RSS ayrı tutulur.
FLOPs/MACs convention ve covered-operator listesiyle; Roofline ancak güvenilir
compute/traffic verisiyle raporlanır. Hardware counter, Tensor Core utilization,
enerji ve OS RSS araç/donanım desteği olmadan uydurulmaz. Desteklenmeyen alan
`null + status + reason` alır; sahte0 yazılmaz. Accuracy/robustness kendi veri ve
evaluation protokollerini gerektirir. P99/sweeps/export/comparison/GUI/deployment
B1'e sessizce eklenmez. Yeni dependency/adapter/uzun-run ancak ilgili alt görevin
açık kapsamı içinde uygulanır; bu belge bunları bugün kurulmuş saymaz.

B1 kanıtı ve dosyalar: [CPU benchmark audit](../audit/m2-c-b1-cpu-benchmark.md).
B2 kanıtı ve sınırlar: [CUDA benchmark audit](../audit/m2-c-b2-cuda-benchmark.md).
B3 kanıtı ve sınırlar: [Export/comparison audit](../audit/m2-c-b3-benchmark-export-comparison.md).

## M2-P-B8 capability / B9 P19 güncellemesi — 2026-10-21

P18 **planned / blocked**: mevcut Windows Torch `LIBKINETO_NOCUPTI` build'i,
eşleşmiş FLOP/DRAM counter ve calibration kanıtı sağlamıyor; Roofline API yok.
[B8 capability gate](../audit/m2-p-b8-roofline-capability-gate.md).
P19 doğrulanan native CPU ve selected-stream CUDA event kapsamıyla implemented.
Gerçek fit, ayrı train/validation/total epoch clocks, microbatch/window/attempt
ledger, applied/skipped ownership, exact deterministic replay, checkpoint segments
ve export/storage sınırları [B9 audit](../audit/m2-p-b9-training-timing.md) ve
[rehber](../guides/performance-training-timing.md) içinde açıklanır.
Güncel 35-feature sayım: **14 implemented, 5 partial, 16 planned**.
P20–P25 durumları değişmedi; özellikle P20 stage breakdown ve P22 rates yoktur.
M2-P-A'nın tam B9 kapsamı P19+P20+P22 olduğundan **B9 PARTIAL**; full B9 GO yoktur.
B4 OPEN/UNRESOLVED, B6 kernel profiling ve B7 overlap/full-scope NO-GO geçmiş
kapıları aynen korunur. Bu güncelleme stable release onayı değildir.

## Post-Performance Developer Experience

35 performans özelliği tamamlandıktan sonra kullanıcıya yönelik sade HYSo API'si
geliştirilmesi gelecekteki tasarım hedefidir. Örnek **mevcut olmayan** kullanım:

```python
from hyso import Hyso

app = Hyso(model="modular_cnn", device="auto")
app.train(train_loader, epochs=10)
app.benchmark()
app.report("report.html")
```

`Hyso` facade API'si şu anda mevcut değildir ve bu görevde uygulanmadı.
Mevcut Trainer/BenchmarkRunner/performance public API'leri korunacak.
Bu not yeni performans özelliği eklemez; toplam özellik sayısı **35** kalır.
