# HYSo Performance Backlog v1.1 — Onaylı genişleme kapsamı

Tarih: 2026-10-19. Branch `feature/hyso-models-next`, kaynak HEAD
`979e9285cd9f839e408e3058fed42de3fefb1ed3`.
Kullanıcı aşağıdaki **35 özelliğin gelecekte geliştirilmesini** açıkça onayladı.
Bu belge yeni performans kapsamını ve önceliklerini taşır; paket sürümü değildir.
HYSo0.2.0 ve [sabit roadmap v1.0](hyso-architecture-roadmap-v1.md) değişmez.
[M2-C-A](../audit/m2-c-a-benchmark-engine-design.md) ve önceki milestone durumları
geriye dönük değiştirilmez. M2-C-B1 CPU inference çekirdeğidir; M2-C-B2
native CUDA event/FP16 autocast ve opt-in allocator memory probe ekler.

Durumlar: `planned`=henüz benchmark özelliği olarak uygulanmadı;
`partial`=belirtilen alt kapsam var; `implemented`=bütün satırın doğrulanmış
sözleşmesi var; `optional-hardware`=donanım bağımlı uygulama ayrıca onaylanmışsa
kullanılacak durum. B2 sonunda P01/P08/P11 partial, P09 sınırları belirtilmiş
native allocator sözleşmesiyle implemented, kalan31 satır planned'dır.
Donanım opsiyonelliği araç sütununda ayrıca belirtilir.

## Uygulama sırası

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
| P01 | Inference Latency — P50/P95/P99 | Aynı resident batch için CPU/CUDA mean/p50/p90/p95/std var; P99 henüz yok | ms, raw samples/quantiles | CPU monotonic clock; native CUDA event | Sabit shape/device/precision, warmup, repeat/sync scope | Ö1 | partial | B1/B2 finite positive R×N; known samples linear quantiles ve native event/sync; P99 ancak versioned protokol kararı ve yeterli örnekle |
| P02 | Real Throughput / FPS | Uzun ölçüm penceresinde gerçekten tamamlanan örnekler; B1 derived rate bu özellik değildir | images/s veya frames/s; window süresi | CPU/CUDA, tamamlama/sync sayacı | Window başlangıç/sonu, pipeline/concurrency ve I/O scope'u sabit | Ö1 sonrası ayrı protokol, Ö3 | planned | Completed images/window seconds; native sync, pipeline tail flush ve bağımsız count reference |
| P03 | End-to-End Latency | Capture/read→preprocess→transfer→inference→postprocess zincirinin gecikmesi | ms/frame, percentiles | Gerçek/sentetik veri akışı ve host clock | Stage boundary, queue/drop policy, ortak timestamp domain | Ö3/Ö7 | planned | Aynı örneğin uçtan uca izlenmesi; async completion dahil, device event ile host wall karıştırılmaz |
| P04 | Preprocessing / Postprocessing Time | Resize/normalize/decode ve output dönüşümü sürelerini ayırma | ms/stage, breakdown | Tanımlı transforms/postprocessor, CPU/GPU | Gerçek veri, stage device/sync ve işlem listesi | Ö3 | planned | Stage count/order ve wall/event scope reference; toplamla overlap ilişkisi açıklanır |
| P05 | Cold Start & Model Loading | Fresh process import/load/ilk inference maliyeti | ms veya s, ayrı cold/warm rapor | Yerel artifact, fresh subprocess | Artifact format/bytes, cache/page-cache koşulu ve init sınırı | Ö3 | planned | Gerçek fresh process; model load, context init ve ilk forward ayrı; disk cache durumu kaydedilir |
| P06 | Batch Size Sweep | Sabit model/resolution/precision için batch ekseninde ölçüm | batch→latency/rate/memory tablosu | CPU/CUDA; Ö1 ölçümleri | Aynı timing method; batch başına warmup, order/provenance | Ö3 | planned | Her nokta bağımsız valid result; batch değişimi comparison etiketi ve başarısız noktalar görünür |
| P07 | Resolution Sweep | Sabit batch/model için spatial resolution ekseninde ölçüm | H×W→latency/memory/quality | CPU/CUDA; geçerli model input'ları | Aspect ratio, preprocessing ve task tanımı | Ö3 | planned | Her shape gerçek forward, ayrı warmup ve sample sayısı; interpolate edilmiş sahte sonuç yok |
| P08 | Precision Benchmark | CPU FP32/FP64 ve CUDA FP32/native AMP FP16 inference var; BF16/INT8, dönüşüm maliyeti ve accuracy-drop sonraki kapsam | precision→ms/bytes ve policy | CPU; native CUDA autocast, quantization backend ileride | Master/input/output dtype ve autocast policy | Ö1/Ö3 | partial | Gerçek CPU fp32/fp64 ve GPU fp32/amp_fp16; FP32 master/input korunur, silent cast/fallback yok; diğer precision ve accuracy ayrı gate ister |

## B. Donanım, bellek ve darboğaz analizi

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P09 | GPU Peak Memory | Hedef process/device native allocator baseline, önceki/mutlak peak, incremental peak ve output release sonrası değerler; total VRAM değildir | allocated/reserved byte, separate_single_forward scope | Native CUDA allocator API; RTX3060 üzerinde doğrulandı | Model/input resident, sync, opt-in geri alınamayan peak reset ve caller/process exclusivity | Ö1/B2 | implemented | Native baseline/reset/probe/output-release; peak>=baseline; disabled no-reset/null; prior peaks ve reset diagnostics; allocator vs driver/other process ayrımı |
| P10 | CPU Peak RAM | Process CPU RSS/peak; Python heap'ten ayrı | byte/MiB ve sampling interval | OS-specific RSS backend veya ileride onaylı araç | Güvenilir platform adapter, baseline, sampling/peak tanımı | Ö4/Ö8 | planned | Native tensor allocation/free reference; tracemalloc'a toplam tensor/RSS adı verilmez |
| P11 | Model Footprint | B1 unique logical parameter/buffer bytes ve alias notes; physical/storage/disk kapsamı yok | count, logical byte; ayrı disk artifact byte ileride | Tensor numel/element_size; disk için gerçek dosya | Dedup/alias convention ve artifact türü | Ö1/Ö3 | partial | Tied Parameter/cross-category/view testleri; numel×element_size; disk sayı ancak gerçekten yazılan artifact'tan |
| P12 | FLOPs / MACs | İşlem sayımı; operator coverage ve convention zorunlu | FLOP/MAC count, coverage raporu | Gelecekte seçilecek trace/profiler adapter | Sabit input graph, desteklenen ops ve sayım tanımı | Ö4 | planned | Küçük analitik Conv/Linear reference; MAC ile FLOP farklı alanlar; unsupported ops görünür |
| P13 | Layer-wise Profiler | Modül bazlı süre/shape/memory breakdown | ms/module, çağrı ağacı | PyTorch profiler/measurement adapter | Hook overhead, shared module ve nesting politikası | Ö4 | planned | Ortak/tekrar çağrılar doğru; inclusive/exclusive süre ayrılır; profiler latency baseline'dan ayrı |
| P14 | Operator / Kernel Profiler | Operator ve kernel execution breakdown | op/kernel süre, count, trace | PyTorch profiler; opsiyonel GPU profiler araçları | CUDA sync/trace, profiler overhead ve sürüm | Ö4 | planned | Gerçek trace/correlation; CPU launch süresi GPU kernel süresi sayılmaz; missing events açıklanır |
| P15 | CPU–GPU Transfer Analysis | H2D/D2H ve overlap/PCIe etkileri | ms, byte/s, overlap timeline | CUDA; pageable/pinned host tensors | Transfer direction/size, pinning/nonblocking ve sync | Ö4 | planned | Gerçek copy completion; bandwidth numerator gerçek byte; overlap çift sayılmaz |
| P16 | GPU / CPU Utilization | Zaman örneklemeli kullanım ve destek varsa compute-unit counters | %, sample timeline, available counters | Opsiyonel OS/NVML/profiler/hardware counter erişimi | Sample interval, counter tanımı/izinler | Ö4/Ö8 | planned | Query availability ve interval kaydı; GPU busy% Tensor Core utilization değildir; unsupported alan null |
| P17 | Maximum Feasible Batch Size | Belirli input/mode/precision için bellekçe çalışabilir batch sınırı | max batch, failure/baseline raporu | CPU/CUDA, izole worker | OOM recovery/fresh process, arama limiti ve precision | Ö3 | planned | Geçen noktanın gerçek forward'u, sonraki fail/budget kanıtı; sonsuz arama veya device reset yok |
| P18 | Arithmetic Intensity & Roofline Analysis | Gerçek compute/traffic verisiyle bounded teşhis; tahmini parametre bytes trafik sayılmaz | FLOP/byte, FLOP/s, GB/s, roofline grafiği | Opsiyonel doğrulanmış hardware counters/traffic ve compute ölçümü | P12/P14; aynı memory hierarchy, kalibre bandwidth/compute tavanı | Ö4 | planned | Güvenilir DRAM/cache traffic ve FLOP scope'ları eşleşmeli; sayaç yoksa yalnız tahmin etiketi, gerçek Roofline teşhisi yok |

## C. Training Performance

| ID | Tam isim | Türkçe açıklama | Birim/rapor | Donanım/veri/araç | Önkoşullar | Aşama | Durum | Doğrulanma kriteri |
|---|---|---|---|---|---|---|---|---|
| P19 | Epoch / Training Step Time | Inference'tan ayrı training-step/epoch latency | ms/step, s/epoch | Model, data, loss, optimizer, CPU/CUDA | Update/zero_grad/accumulation ve validation scope'u | Ö5 | planned | Gerçek training state/update; epoch toplamı ve excluded I/O açık; inference latency yeniden etiketlenmez |
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
| P29 | Accuracy–Speed–Memory Trade-off & Regression | Uyumlu sonuçlarda Pareto/regression değerlendirmesi | koşul eşliği raporu, curves, deltas | Ö2 versioned results, quality+memory metrics | P01/P09/P10/P26; uyumlu input/runtime/precision | Ö2 temeli, Ö6/Ö8 analiz | planned | Koşul uyuşmazlığı flag/red; raw repeats/uncertainty; karşılaştırılamayan modeller otomatik quality winner olmaz |

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
