# HYSo analiz, optimizasyon ve kullanıcı kontrolü sözleşmesi

Tarih: 2026-10-21, Europe/Istanbul. **Tasarım sözleşmesi; yeni facade, adapter,
evaluation veya rapor motoru implementasyonu değildir.** Kaynak branch
`feature/hyso-models-next`, HEAD `b0e48e98366bcf8c0054e561434bdf35f358dc74`.
Bu belgede gelecek zamanla tanımlanan veri tipleri henüz public Python API değildir.

## 1. Ürün kapsamı ve mevcut gerçeklik

HYSo iki giriş yolunu destekleyecek: mevcut HYSo yapı taşlarıyla model oluşturma
ve kullanıcının kendi PyTorch `nn.Module` modelini getirmesi. İkisinde de
model/eğitim tercihlerinin sahipliği kullanıcıdadır. Analiz gerçek workload,
veri, cihaz ve desteklenen ölçüm protokolü üzerinde yapılır; öneriler kanıta bağlanır.

Bugünkü kod: `models/bcam.py` ELO/CBAM ve EBO eğitim wrapper'ı;
`WithSE.py` CEMO/SE/EMA; `fit.py` CNNSENormalWithFit;
`blocks.py` ConvNormAct/ResidualBlock; `attention.py` ECA/Coordinate;
`modular.py` ModularCNN. `hyso.models.__all__` legacy beş export'u korur;
ModularCNN ve yeni bileşenlerin kendi modül importları vardır. ModularCNN
width/repeat/stride/attention seçimleriyle composition yapar; genel aile registry'si
veya bütün gelecek mimarileri kapsayan builder değildir.

`training.Trainer`, caller-owned model/optimizer ve multiclass/binary mean-loss
classification sözleşmesini uygular; modeli otomatik başka device'a taşımaz.
Her `nn.Module` her API'ye sınırsız uygun değildir: tensor/layout/output,
optimizer, loader ve lifecycle admission koşulları geçerlidir.
`BenchmarkRunner` desteklenen resident input ve tensor output profiliyle inference
ölçer; detection dict/list çıktılarının varlığı otomatik destek anlamına gelmez.
`ExperimentManager` explicit persistence sağlar; training veya benchmark çalıştırmaz.
`Hyso` facade, YOLO/detection ve segmentation adapter'ları mevcut değildir.

[Mimari roadmap v1.0](../roadmap/hyso-architecture-roadmap-v1.md) **97 başlıktır**;
[performans backlog v1.1](../roadmap/hyso-performance-backlog-v1.1.md) **35 ID'dir**.
Mimari roadmap'in tarihsel M2 planned satırları, sonradan yazılan Trainer/
Benchmark/Manager kodlarını yok saydırmaz; bu belge sabit roadmap'i yeniden yazmaz.
Güncel performans sayımı **17 implemented / 7 partial / 11 planned**.
Implemented, yalnız audit'te doğrulanan kapsamdır; platform/genel release onayı değildir.
Yerel P23 ekleri uncommitted'dir ve P23'ü implemented yapmaz.

## 2. Kullanıcı tercihi ve çalışma sözleşmesi

Model mimarisi, katman/attention sayısı, batch, epoch, optimizer, precision,
device ve eğitim konfigürasyonu kaydedilen `requested_configuration` içindedir.
`effective_configuration` ayrıca tutulur; fark varsa gerekçe ve izin referansı
zorunludur. Sessiz batch küçültme, epoch kesme, precision fallback, model
pruning veya optimizer değişimi yasaktır. OOM, timeout ve unsupported gerçek
sonuç olarak döner; ikinci fit veya değiştirilmiş recipe ile otomatik retry yapılmaz.

|İşlem|Sözleşme|
|---|---|
|Mevcut immutable evidence'dan analiz|Model çalıştırmadan yapılabilir; source ID/hash korunur|
|Opt-in instrumentation|Ölçüm/sync/scratch/I/O overhead ve clock kapsamı açıklanır; ücretsiz gözlem sayılmaz|
|Mimari/eğitim semantiği değişimi|Önce açıklanan tam değişiklik ve kullanıcı onayı; kalite ve state etkisi test planında|
|İzinli çalışma optimizasyonu|Amaçlanan semantik aynı olsa da tolerans, RNG/order, backend ve gerçek karşılaştırma doğrulanmalı|
|Yeni workload/karşılaştırma|Explicit execution plan/bütçe ve yetki; recommendation engine tek başına başlatamaz|
|Kaynak bütçesi|İzole probe'da kullanıcıya görünür, kullanıcı tarafından seçilebilir; sonuç/provenance'a yazılır|

Bir önerinin kabulü, bütün gelecekteki model mutation veya training run'ları için
genel yetki değildir. Kullanıcının mevcut açık izinleri tekrar sorulmaz; izin
kapsamı değişen recipe/semantik/snapshot için somut yeni plan üzerinden alınır.
Ölçülmemiş hızlanma, accuracy, VRAM veya enerji tasarrufu garanti edilmez.

## 3. Dört farklı GPU politikası

|Bağlam|Utilization|Bellek/timeout/ownership|
|---|---|---|
|Normal Trainer ve in-process gözlem|Gözlemsel; %35/%80/%100 tek başına eğitim reddi değildir|Caller device ve gerçek allocator davranışı; P23 never-reset gözlemi kullanıcının limitini değiştirmez|
|Opt-in isolated probe|Telemetri; sessiz otomatik kapasite politikası yok|Protocol'da görünür kullanıcı bütçesi; Job private commit, timeout ve cleanup; native CUDA allocator fraction ayrı|
|B4 forensic tek diagnostic|Yeni v3 adayında utilization hard gate değil; background activity kaydı|RAM>=4 GiB, disk>=12 GiB, üç örnekte GPU free>=2048 MiB, identity, workload review, Job ve tek-deneme kuralları|
|Performans karşılaştırması|Background activity ölçümün adilliğini etkileyebilir|Önceden tanımlanan kalite/exclusivity protokolü; veri confounded/insufficient olabilir, training yasaklanmaz|

GPU busy zaman oranı; memory-controller busy bellek erişim zaman oranıdır.
Tensor allocated, allocator reserved, fiziksel device free/used ve gerçek OOM
ayrı ölçüm alanlarıdır. Utilization'dan VRAM yeterliliği veya Tensor Core kullanım
oranı türetilmez. Native crash güvenliği yalnız düşük utilization ile kanıtlanamaz.
[NVIDIA nvidia-smi tanımları](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

P23 isolated `.15` varsayılanı forensic'e özel değildir: yerel production
`IsolatedTrainingMemoryProtocol` alanıdır, `(0,1]` seçilebilir. 6 GiB visible
memory için yaklaşık921.6 MiB allocator bütçesi eder; bu bir rezervasyon veya
fiziksel VRAM sınırı değildir. Bütçe OOM'u device tamamen dolmadan oluşabilir.
Normal Trainer/in-process P23'e bu fraction uygulanmaz. P17/P12–14/P15 owned
worker protocol'larında da `.1` varsayılanları vardır. Bu görev bunları değiştirmez.
[PyTorch v2.5.1 allocator kaynağı](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py).

İleri API işi ayrı kabul kapısıdır: documentation/example'da etkin fraction ve
yaklaşık allocator bytes görünür olmalı; aynı `(0,1]` explicit seçim korunmalı.
Varsayılanı `None`/1 yapmak veya budget seçimini zorunlu kılmak, API/result
validation ve geriye uyumluluk review'u gerektirir. `None` bugün desteklenmez.
Bütçe OOM'u ile fiziksel exhaust ayrımı kesin belirlenemiyorsa neden
`unknown` kalır; otomatik fraction artırımı veya batch düşürümü yapılmaz.

## 4. Workload ve adapter capability sözleşmesi

Adapter tasarımı model sınıf adından görev tahmin etmez. Caller task beyanı ve
adapter'ın input/output/evaluation kabiliyeti ayrı doğrulanır. Gelecekteki adapter
yalnız sahip olduğu capability'yi ilan eder; `supported=false + reason` bir sonuçtur.

|Task|Girdi ve kimlik|Uygulanabilir analiz|Ek kalite verisi ve mevcut sınır|
|---|---|---|---|
|Image classification/CNN|NCHW, class map, split, transforms, model/config hash|Mevcut latency/throughput/footprint/memory, kapsanan conv/operator ve training analizleri|Labeled targets ve averaging policy olmadan P26 yok; legacy accuracy ortak evaluation sistemi değildir|
|Object detection/YOLO|Gelecek adapter'da image sizes, boxes/classes, coordinate/resize policy, output/NMS recipe|Preprocess/inference/postprocess ayrı, batch/resolution, NMS scope|mAP için ground truth, matching/IoU ve evaluator sürümü; adapter bugün unsupported|
|Segmentation|Gelecek adapter'da image/mask resolution, logits/mask encoding, ignore labels|Mask output maliyeti, memory, stage latency|IoU/Dice için ground truth ve class/aggregation policy; adapter bugün unsupported|
|Custom PyTorch|Caller-declared input/output signatures, task unknown olabilir|Yalnız ilgili API'nin admission profilinden geçen generic ölçümler|Task-specific metric ve desteklenmeyen output/loader profilleri null/unsupported|

Adapter'ın ölçüm sonucu yayımlaması input normalization veya postprocess'i
otomatik değiştirme hakkı vermez. Detection/segmentation için classification
accuracy'yi yeniden adlandırmak yasaktır. Pre/post stage'leri caller tanımlı
pipeline olarak ölçmek, tam YOLO adapter/evaluator implementasyonu sayılmaz.

## 5. Öneri veri sözleşmesi (önerilen version1)

Her recommendation kaydı aşağıdaki alanları taşımalı. Bu aşamada serializer veya
rule engine yazılmadı. Confidence'in sayısal olasılık olduğu iddia edilmez;
kalibrasyon yoksa `value=null`, gerekçeli low/medium/high ordinal etiket verilebilir.

```json
{
  "recommendation_id": "stable-rule-id@version/source-measurement-id",
  "workload_id": "caller-workload-id",
  "task_type": "image_classification",
  "applicability": {"status": "unsupported", "reason": "required_evidence_missing"},
  "evidence": [],
  "bottleneck": {"value": null, "status": "unknown", "reason": "no_matching_measurement"},
  "proposed_change": {"kind": "configuration", "patch": null, "description": null},
  "expected_effect": {"value": null, "unit": null, "status": "unknown", "reason": "not_estimated"},
  "verified_effect": {"value": null, "unit": null, "status": "not_measured", "reason": "no_authorized_comparison"},
  "tradeoffs": {
    "speed": {"value": null, "status": "unknown", "reason": "no_comparison"},
    "memory": {"value": null, "status": "unknown", "reason": "no_comparison"},
    "quality": {"value": null, "status": "unknown", "reason": "no_evaluation"}
  },
  "training_semantics_risk": {"level": "unknown", "reason": "change_not_selected"},
  "approval": {"required": true, "scope": null, "reference": null},
  "confidence": {"value": null, "status": "uncalibrated", "reason": "no_calibration", "basis": []},
  "limitations": ["No model mutation or additional run is authorized by this record."]
}
```

`applicability.status` yalnız applicable/not_applicable/unsupported;
not_applicable=geçerli evidence kuralın koşulunu sağlamıyor,
unsupported=gerekli backend/task/data/capability yok. Kanıt girdisi en az
result ID, measurement/field path, source SHA256, feature ID, scope ve clock
taşır. `estimated` etki için yöntemi/varsayımları/birimini; `measured` etki için
baseline/candidate ID/hash, uyumluluk kararı, gerçek sample sayısı ve kalite
karşılaştırması gerekir. Eksik payda, incompatible ortam veya farklı görev
metriği için yüzde/speedup üretilmez.

Örnek kural: P21'de yüksek main-process next-wait payı **worker decode yavaştır**
sonucunu tek başına vermez. Kanıtlı bulgu delivery beklemesidir; decode/transform/
queue nedenleri worker trace gelene kadar unknown. Öneri önce bu boşluğu ölçmektir;
`num_workers`, prefetch veya pinned memory'yi otomatik değiştirmek değildir.
P14 CPU self-time sıralaması GPU kernel bottleneck sıralaması olarak sunulmaz.
P12 unknown FLOPs varken model toplamı veya Roofline classification çıkarılmaz.

## 6. Birleşik rapor veri sözleşmesi ve mevcut API'ye eklenme planı

Önerilen `format=hyso.analysis_report`, `schema_version=1` ayrı bir üst belge
olacak. Mevcut Benchmark/schema1, checkpoint/v1, performance result formatları
ve Manager DB şeması değişmeyecek. Kaynak sonuçlar opaque/versioned referans
olarak korunacak; raw ledger yeniden yazılmayacak.

|Alan|Zorunlu içerik|
|---|---|
|identity|report/workload/model/run ID, model origin HYSo/custom, task, input/output signature, config/source/artifact hashes|
|provenance|device UUID/PCI, OS/Python/Torch/backend/precision, seed, clocks, collection times, source revision ve dirty-state etiketi|
|configuration|requested/effective config, izinli farklar, explicit budgets, observation/reset ownership|
|training_and_evaluation|Trainer history/counters/segment refs; ayrı evaluator/split/class map, eval cadence, unsupported alanlar|
|feature_results|P01–P35 için 35 slot; status, scope, reason ve sıfır/çok sayıda source result refs; uygulanmayan slot ölçülmüş sayılmaz|
|resources_and_timing|CPU/RAM/GPU/VRAM domain ayrımı, sampled/lifetime/window peak ve host/event/sensor clocks|
|bottlenecks|Kanıtlı tespit, alternatif açıklamalar ve belirsizlik; unknown geçerli sonuç|
|recommendations|Bölüm5 sözleşmesi; versioned kurallar, izin kapsamı, task capability|
|optimization_comparisons|Gerçek baseline/candidate çiftleri; compatible/incompatible/insufficient, ölçülen kalite/hız/bellek ve limitations|
|next_actions|Bağımlılıklara göre uygulanabilir, kullanıcıya görünür işler; kendiliğinden çalıştırılmaz|

Her scalar: `{value, unit, status, reason, evidence_ref}`; ölçülen/tahmin edilen/
unsupported durumlar ayrıdır. `status=measured` yalnız gerçek sample/domain
ile; `estimated` yöntemi zorunlu. Null sahte0'a çevrilmez. Rapor genel
`outcome=success` yanında ayrı `coverage=partial` taşıyabilir. Bir feature'ın
planned durumu, o raporda measured0 olarak dışa aktarılmaz.

İlk uygulanabilir ürün adımı, **mevcut immutable sonuçları alan saf bir evidence
index/report-envelope validator** olacaktır: source ID/hash/path/clock/task
association, 35 coverage slot'u, null/status/reason doğrulaması. Başlangıçta
UI veya yeni ölçüm gerektirmez. Bu görev yalnız sözleşmesini tanımlar.

Entegrasyon sırası:

1. Caller mevcut Trainer/BenchmarkRunner/performance API'sini açıkça çalıştırır;
   örneğin P22 mevcut P19'dan analiz edilir, ikinci fit başlatılmaz. Aynı fit'te
   birlikte desteklenmeyen recorder'ları otomatik birleştirmek yasaktır.
2. Envelope kaynak sonuçların native JSON/CSV exports, segment/measurement
   ID'leri ve SHA'larını referanslar; farklı fit'ler aynı run diye birleştirilmez.
3. Mevcut `ExperimentManager.create_run/start`, `ExperimentRun.add_artifact`
   üzerinden JSON rapor `kind=report`, `ownership=copy` artifact olur; reopened
   artifact SHA kontrolü korunur. Structured null alanlar JSON'da kalır.
4. `log_metric(s)` yalnız finite numeric değer, mevcut epoch/step_kind ekseni ve
   açık unit/source kabul eder. Null veya unsupported scalar metric'e gönderilmez;
   axis olmayan özet için hayalî epoch/step yaratılmaz, artifact yeterlidir.
5. `finish/fail` ve workload/tracking outcome ayrımı korunur. Persistence
   hatası retrain/benchmark retry tetiklemez; önce üretilmiş evidence korunur.

Kabul: fixture source referansları/tamper/clock/task mismatch, 35-slot completeness,
unknown/null export round-trip, no-clobber ve Manager close/reopen/hash doğrulama;
validator sırasında factory import/fit/device query sayısı0. Bu bir sonraki ayrı
ürün görevidir; burada rapor motoru veya hayalî facade eklenmedi.

## 7. 35 ID'nin kaynak ve acceptance matrisi

Tablodaki implemented/partial/planned, backlog'un güncel satırlarıyla birebir
aynıdır. Native sütunları **tarihsel audit + bugünkü kaynak** kapsamıdır; bu görev
bütün native testleri yeniden çalıştırmış değildir. E/M sütununda “mevcut” native
result export ve audit'teki explicit Manager artifact/metric testlerini; “plan”
gelecek bağımsız kabul kapısını anlatır. Ortak kabul koşulu: immutable JSON,
unit/domain/source ID/hash, null/status/reason, tamper ve Manager reopen;
platform/installed/minimum/package doğrulaması yeni kaynağa göre ayrıca gerekir.

|ID / durum|Mevcut native kapsam / kaynak|Eksik davranış|Veri/donanım ve bağımlılık|Bağımsız acceptance işi|Export/Manager|Kullanıcı yararı|
|---|---|---|---|---|---|---|
|P01 implemented|CPU/CUDA resident latency; inference.py, B1|Genel bağımsızlık/CI iddiası yok|Raw benchmark samples, warmup, same scope|P99 n/tail eligibility, sample korunumu, state/RNG|Mevcut LatencyAnalysis|Kuyruk gecikmesini doğru örnek sayısıyla görür|
|P02 implemented|CPU FP32/64, CUDA FP32/AMP completion throughput; B1|Camera/E2E ve hidden streams yok|Resident input, bounded completion queue; P01 altyapısı|Completed/pending/tail conservation, terminal drain|Mevcut ThroughputResult|Gerçek completed images/s|
|P03 implemented|Blocking serial CPU/CUDA pipeline; pipeline.py/B3|Overlap/camera drop yok|Explicit stages ve frame IDs; P01/P02|Aynı ID read→completion, raw clocks|Mevcut PipelineResult|Uçtan uca gecikme|
|P04 implemented|Host/event pre/post stages; B3|Her task'a hazır adapter yok|Caller stages, ordered boundaries; P03|Order/count/reference equality, host/device ayrımı|Mevcut StageTimingResult|Model dışı maliyeti ayırır|
|P05 implemented|Windows fresh CPU/CUDA interpreter cold phases; cold.py/B3|OS page-cache cold garantisi yok|Trusted factory ve hashli weights_only artifact|Import/load/first/warm ayrımı, timeout/cleanup|Mevcut ColdStartResult|İlk isteğin maliyeti|
|P06 implemented|Explicit finite batch sweep CPU/CUDA; sweeps.py/B4|Training batch veya global optimum yok|Explicit shapes/owned copy; P01/P02/P08/P09|Her noktada gerçek forward; OOM stop/state korunumu|Mevcut SweepResult|Kendi batch tercihini kanıtla kıyaslar|
|P07 implemented|Explicit resolution/aspect sweep CPU/CUDA; B4|Otomatik resize/quality yok|Caller shape factory; P06 çekirdeği|Dikdörtgen/invalid shape, no silent interpolation|Mevcut SweepResult|Çözünürlük maliyetini görür|
|P08 implemented|CPU FP32/64/BF16/dynamic-qint8 Linear; CUDA FP32/AMP16/BF16; B4|Genel INT8 graph/quality yok|Uygun backend ve gerçek dtype/operator; P01|No fallback, conversion scope, output/state; kalite P28 ayrı|Mevcut PrecisionResult|Precision hız/bellek koşullarını ayırır|
|P09 implemented|Native CUDA allocated/reserved forward probe; benchmark/P09|Total physical VRAM değil|Exclusive reset izni; CUDA allocator|Baseline/prior/peak/incremental/output release ve no-reset path|Mevcut benchmark memory|Allocator tepe kullanımını anlar|
|P10 implemented|Windows RSS/HWM native; memory.py/B2|Linux native doğrulama yok; sampled peak mutlak değil|OS backend, bounded sampler|Allocate/touch/release, gaps ve failure cleanup|Mevcut MemoryResult|Process RAM gözlemi|
|P11 implemented|CPU dense logical footprint + actual artifact bytes/SHA; B2|Physical/device footprint attribution yok|Ordinary tensors ve explicit artifact; serialization|Tied/view dedup ve stat/hash/no-clobber|Mevcut FootprintResult|Model ve dosya boyutunu ayırır|
|P12 partial|Observed ATen Conv/Linear/mm/bmm/pool/arithmetic modelleri; profiling.py/_operation_models.py, B6|Nonlinear/norm/fused/custom maliyetler unknown|Gerçek trace shapes; P14|Yeni operator başına analitik referans; unknown total null|Mevcut OperationCountResult; genişleme plan|Kapsanan hesap maliyeti|
|P13 partial|CPU module invocation hierarchy/aliases/timing; B6|CUDA per-layer activity/allocation yok|P14 correlation + module IDs|Nested/shared/repeated calls, exclusive/inclusive; GPU coverage ayrıca|Mevcut LayerProfileResult; GPU plan|Katman maliyetini doğru domain'de görür|
|P14 partial|CPU operator traces; mevcut Windows build CUDA kernel activity unsupported; B6/R1|Gerçek kernel/correlation yok|CUPTI etkin backend/izin ve native cihaz|Önce gerçek kernel trace/time/name/correlation; labels proof değil|Mevcut OperatorProfileResult/trace SHA; GPU plan|Kernel darboğazının güvenilir temeli|
|P15 partial|Windows H2D/D2H pageable/pinned/blocking/nonblocking completion; transfers.py/B7|Gerçek copy/compute overlap yok|P14 common timeline, CUDA, owned buffers|Bytes/equality/terminal + copy/kernel timeline eşliği|Mevcut TransferResult; overlap plan|Transfer seçeneklerini doğru yorumlar|
|P16 implemented|Windows process/system CPU + physical NVIDIA sensors; utilization.py/B7|Process GPU/Tensor Core ve Linux native yok|OS counters, optional installed CLI|Delta/gap/reset/unavailable ve UUID provenance|Mevcut UtilizationResult|Donanımı gözler; yüzdeyle eğitim engellenmez|
|P17 implemented|Windows finite fresh worker capacity CPU/CUDA; capacity.py/B5|Training/adaptive/global max yok|Explicit budgets/candidates; P06/P08/P09|OOM→fresh candidate, timeout/Job cleanup; observed/proven ayrımı|Mevcut CapacityResult|Bütçe içinde denenmiş batch kapasitesi|
|P18 planned|API/counter acceptance yok; B8 capability audit|Gerçek Roofline/traffic/ceiling|P12/P14, matched FLOP/DRAM counters ve calibration|Aynı kernel/hierarchy/time; gerçek byte+FLOP ve empirical ceilings|Plan Roofline artifact; counter yoksa unsupported|Compute/bandwidth teşhisi için sağlam kanıt|
|P19 implemented|CPU ve selected-stream CUDA gerçek fit ledger; training.py/B9|Kernel attribution yok|Trainer lifecycle/counters, finite loader|Epoch/step/window/attempt clock, exact replay/RNG|Mevcut TrainingPerformanceResult|Eğitim zamanını doğru scope'ta görür|
|P20 implemented|Host/event forward/backward/optimizer stages; training_details.py/B9|Kernel attribution/overhead correction yok|P19, native optimizer execution hooks|Applied/skip/EMA/scheduler ledger, inclusive scopes|Mevcut TrainingStageResult/DetailedTrainingResult|Hangi eğitim aşaması pahalı|
|P21 partial|Windows main iterator init/next/delivery, spawn workers; dataloader.py/B10|Worker decode/transform/service/queue trace yok|CPU labeled veya synthetic loader, spawn IDs; P19/P20/P22|Bounded worker events, clock join, conservation, exception/timeout/cleanup|Mevcut DataLoaderAnalysisResult; worker extension additive|Bekleme nedenini tahmin yerine ölçer|
|P22 implemented|P19 raw ledger'dan pure throughput; B9|Yeni training/quality analizi yok|P19 sample/segment/skip ledger|Ratio of sums, uneven tail, resumed/no-op/null|Mevcut TrainingThroughputResult|İkinci fit olmadan samples/update rate|
|P23 partial|Legacy never-reset; yerel Windows isolated/CPU saved-payload, historical CUDA window scope; B10/R1|CUDA activation disabled; final native/install gates açık|P09/P10/P19, ownership ve B4 gate|CPU replay/mutation/GC; native ayrıca açık onay ve final matrix/packaging|Mevcut iki ayrı result formatı; genişleme koşullu|Eğitim belleğini domain/ownership ile anlar|
|P24 planned|API yok|Censored time-to-target, variance|P19/P26, labeled fixed split/target/cadence, repeated seeds|Known first-crossing, unmet censoring, eval clock exclusion|Plan evaluation/training refs + JSON|Kalite hedefine gerçek süre|
|P25 implemented|Dense CPU/CUDA pre/post clip norm + real AMP ledger; gradients.py/B10|Sparse/gapped/complex/custom unsupported|P19/P20 ve gerçek FP16 GradScaler|Known norm, skip→recovery/scale, exact state/RNG|Mevcut GradientDiagnosticsResult|Gradient/AMP davranışını gözler|
|P26 planned|Ortak evaluation API yok; Trainer/legacy accuracy mevcut|Precision/recall/F1/confusion shared evaluator|Labels/class map/split; future task adapter|Known confusion/zero division/missing classes/averaging; task capability|Plan EvaluationResult artifact; finite metrics|Hızın kaliteyle birlikte değerlendirilmesi|
|P27 planned|API yok|Paired robustness evaluation|P26, labeled clean/perturbed pairs, seed/severity|Sample-ID paired deltas/order ve corruption provenance|Plan evaluation refs/artifacts|Dağılım değişiminde kalite|
|P28 planned|API yok; P08 hız sonuçları kalite değildir|Precision/quantization accuracy delta|P08/P26, identical labeled eval/calibration|pp vs relative %, baseline0 null, preprocessing equality|Plan baseline/candidate evaluation refs|Hızın kalite bedelini gösterir|
|P29 partial|Benchmark schema1 comparison eligibility; benchmark/result.py ve B3|Quality/Pareto/regression/significance yok|P01/P09/P10/P26; matched protocol/runtime/input|Incompatible/insufficient guards; quality constraints ve bounded comparison|Mevcut comparison JSON/CSV; quality plan|Uygun alternatifleri gerçek ödünleşimle kıyaslar|
|P30 planned|Kamera adapter yok; P03 seri sentetik pipeline mevcut|Capture/drop/late/backpressure|P03/P04, gerçek kamera/timestamps/queue|received=processed+dropped+pending; late policy/terminal|Plan frame ledger artifact + finite rates|Canlı kamerada gerçek FPS/kayıp|
|P31 planned|Uzun-run harness yok|Uzun süreli stability/failure timeline|P01/P02/P16; explicit duration/thermal/health|Gerçek tam süre, tüm failures ve gaps, cleanup|Plan chunked artifacts/reopen; bounded metrics|Saatler içindeki bozulmayı görür|
|P32 planned|Trend/leak API yok; P09/P10 counters mevcut|Warmup sonrası ownership/growth ayrımı|P09/P10/P31, live refs ve snapshots|Plateau/cache vs retained refs; slope uncertainty; leak kanıtı|Plan trend/raw snapshots artifact|Büyüme ve leak iddiasını ayırır|
|P33 planned|Enerji API/sensor integrali yok|Joule/inference ve sampling uncertainty|Güç sensörü, P02/P16/P31 clock/count; idle baseline|Gerçek power integral/completed count, device attribution|Plan sensor/raw energy artifact; yoksa null|Ölçülmüş iş başına enerji|
|P34 planned|Multi-device orchestration/comparison yok|Gerçek farklı cihazlarda eşleşen runs|P01/P08/P26/P29; her cihaz/backend erişimi|Per-device native run, explicit precision/runtime differences|Plan immutable per-device refs + comparison|Cihaz seçimi için adil kanıt|
|P35 planned|Deployment provider adapter yok|Node-level fallback tespiti|Gelecek ONNX Runtime graph/session trace ve node mapping|Gerçek node execution provider proof; provider listesi yetmez|Plan deployment trace/result artifact|Beklenmeyen CPU/provider fallback'ini görür|

## 8. Uygulanabilir geliştirme sırası ve küçük kapılar

1. **Öncelikli recovery: B4/P23.** Bu görevin ayrı forensic v3 preflight'ı CPU/
   statik olarak hazırlanır. Yeni native kontrol ancak ayrı açık onayla, yeni
   run-ID/manifest, tek hooks-off/reset-on fit ve mevcut limitlerle mümkündür.
   Tek PASS root cause veya P23 completion değildir. CUDA activation guard
   mevcut kalır; hook noninterference CPU düzeltmesi crash çözümü sayılmaz.
2. **İlk eksik performans işi: P21 worker trace.** B4 native onayına bağlı
   olmayan CPU alt görevi olarak planlanır. Önce worker-local monotonic
   receive/decode/transform/collate/dispatch tanımları ve bounded event channel;
   ardından spawn0/2, persistent/prefetch, order/RNG, teardown/error/timeout.
   Saatler host clock'la körlemesine çıkarılmaz; explicit calibration/clock
   mapping veya unsupported. Tam kimlik/count/cleanup olmadan queue/prefetch
   efficiency raporlanmaz. Bu görev P21 kodunu implement etmez.
3. **Evidence envelope ve capability validator.** Bölüm6'nın saf importer'ı,
   ardından yalnız mevcut kanıtla applicable/not-applicable kuralları. Model
   veya factory çalıştırmadan JSON/Manager acceptance ayrı tamamlanır.
4. **İlk tamamen planned feature: P26 classification evaluation.** Known
   confusion tests → gerçek labeled CPU evaluation → immutable result/Manager.
   Sonra P24 target/censoring, P27 paired robustness ve P28 precision quality;
   bunlar P29 quality-aware comparison/Pareto/regression için veri sağlar.
   Detection/segmentation metrikleri ancak gerçek adapter sözleşmeleriyle gelir.
5. **P12–P15 completion.** P12'ye küçük operator maliyet modelleri ayrı referans
   testleriyle; P14 native backend capability/trace gate'i ayrı. Kernel
   correlation doğrulanınca P13 CUDA layer attribution, sonra P15 copy/compute
   overlap. Mevcut Windows CPU-only Kineto activity kernel proof yerine geçmez.
6. **P18 counter gate.** Uyumlu ortam/izin/counter/replay/traffic ve calibration
   önce; ardından arithmetic intensity/roofline. Sayaç yoksa planned/blocked
   kalır; model bytes veya sensor busy ile doldurulmaz.
7. **Gerçek deployment ve uzun çalışma.** P35 node/provider adapter ile P30
   gerçek kamera ayrı alt görevlerdir. P31 bounded long-run health/timeline
   kurulur; sonra P32 growth attribution ve P33 sensor+integral. P34 her cihazda
   gerçek run ve protocol mapping tamamlandıkça eklenir. Uzun süre, enerji veya
   ikinci cihaz bu görevden otomatik yetki almaz.

Her alt görev önce contract/unsupported sınırını, sonra CPU/reference tests'i,
gerekiyorsa ayrıca yetkili native/device tests'i ve export/Manager/tamper/
minimum/package kapısını tamamlar. Donanım/counter/data eksiği üst katmanlarda
uydurulmuş sonuçla telafi edilmez. B4 OPEN/UNRESOLVED, P23 PARTIAL, B10 PARTIAL
ve R1 NO-GO bu tasarımla değişmez.
