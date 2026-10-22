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

### 7.1. Güncel kabul matrisi — CUDA ve Windows/CPU — 2026-10-22

Güncel 17 implemented özellik için son dar kapsamlı native kanıt aşağıdadır.
Yukarıdaki implementation durumları değişmez. Execution 1 kaynağı:
[Execution 1 audit ve ham ölçümler](../audit/native-cuda-execution-1-p01-p02-p09-p16-20261022.md).
Önceki auditler değiştirilmedi. Execution 1'de P02 yalnız derived service rate
nedeniyle PARTIAL kaldı. Sonraki ayrı yetkili tek `conv2d_fp32_p02_completion`
execution gerçek bounded completion-window API'siyle başarılı oldu; aşağıdaki
P02 kaydı bu yeni sonucu taşır. Retry/baseline/karşılaştırma yapılmadı.

| ID | Güncel native kalite sonucu | Kanıt / açık sınır |
|---|---|---|
| P01 | PASS, dar scope | 3 CUDA event sample; mean 0.161557337 ms; P99 null/insufficient_samples |
| P02 | PASS, tek bounded window scope | 5 completed batch / 10 image; 0.0166102 s gerçek wall-clock; 602.0397105393072 images/s; pending0, group3, terminal drain1; warmup ölçüm dışında |
| P03 | PASS, dar serial pipeline scope | 3 measured + 1 warmup frame; ID eşleşmesi; E2E raw 1.2436/2.6465/2.5686 ms, mean 2.1529 ms; read-start→post-completion host wall |
| P04 | PASS, aynı P03 execution | Preprocess host mean 0.0409 ms; postprocess host mean 0.098866667 ms, CUDA event mean 0.067168001 ms; clock domain'leri ayrı |
| P05 | PASS, tek fresh FP32 interpreter | Import 6459.0377 ms; model construction 19.1007 ms; hashli artifact load 2.4305 ms; runtime prepare 179.6437 ms; first inference 320.2875 ms; warm samples 0.1436/0.1387 ms; OS page-cache cold garantisi yok |
| P06 | PASS, iki batch noktası | Batch1/2: latency mean 0.147264004/0.141263999 ms/batch; gerçek completion throughput 288.811445/6268.937415 images/s |
| P07 | PASS, iki resolution noktası | Batch2, H×W 7×9/9×11: latency mean 0.136191994/0.163952000 ms/batch; throughput 571.706257/6595.580961 images/s; exact shape, resize yok |
| P08 | PASS, FP32/AMP FP16 profili | FP32/AMP FP16: latency mean 0.146384001/0.182592005 ms/batch; throughput 465.784264/4734.474868 images/s; çıktı float32/float16, pytorch_eager; P28 accuracy ölçülmedi |
| P09 | PASS, dar scope | Owned allocator baseline 2048 B, peak 3584 B, release sonrası 2048 B; fiziksel VRAM değildir |
| P10 | PASS, mevcut B2 Windows native kanıtı | GetProcessMemoryInfo RSS baseline588197888 B, sampled peak655634432 B; lifetime HWM588283904→655634432 B; 6 observation/0.0273762 s; mutlak/model-only peak değil, Linux native doğrulanmadı |
| P11 | PASS, mevcut B2 dense CPU/artifact kanıtı | Parameter48 B + buffer24 B = logical union72 B; state_dict2820 B, SHA yeniden doğrulandı; tied/shared object dedup testi PASS, distinct views ayrı sayılır; physical memory unsupported |
| P16 | PASS, dar scope | İki read-only physical-device NVML sample; GPU busy 39/13%; worker attribution yok |
| P17 | PASS, yalnız explicit candidate scope | Batch1/2 ayrı fresh CUDA worker'da feasible; latency mean 0.201952003/0.100559998 ms/batch, her noktada2 measured event sample; observed maximum2, global/proven maximum null; throughput disabled/not_requested |
| P19 | PASS, tek FP32 training scope | Epoch training 0.2719047 s, total 0.2719245 s; 5 sample, 3 microbatch, 2 window/attempt/applied update, skipped0 |
| P20 | PASS, aynı P19 execution | Forward/backward/optimizer host toplamları 43.0251/24.6650/33.3581 ms; ayrı selected-stream CUDA event kayıtları; stage toplamları epoch yerine kullanılmaz |
| P22 | PASS, gerçek P19 ledger türetimi | 5/0.2719047 = 18.388795780 samples/s; 2/0.2719047 = 7.355518312 applied updates/s; ikinci fit yok |
| P25 | PASS, FP32 dense norm/clipping; AMP kapsam dışı | L2 pre→post: 0.348718226→0.099999707 ve 1.522389875→0.099999942; threshold0.1, 26 gradient element, iki gerçek applied update; AMP/overflow/skip recovery ölçülmedi |

P10/P11 son kabulü mevcut [B2 native Windows/CPU kanıtını](../audit/m2-p-b2-memory-footprint.md)
yeniden kullanır; yeni test/workload0, CUDA0, production değişikliği0. Güncel
memory.py SHA256 `cbc831a6a8fa8db6121c107a0bdb077d04bfa39c1b85885476a2f62e99f5995a`
B2 final installed source ile byte-identical; inference.py de aynı. Footprint
bağımlılıkları _counts/_descriptor/_bits/_type AST-identical. Ortak result schema'da
diğer feature branch'leri eklenmiştir; B2 ham sonuçları güncel MemoryResult ve
FootprintResult validator'larıyla Torch import edilmeden doğrulandı.
P10 source result `47276b5b-cae9-4aef-a393-c200caac6cb0`: perf_counter_ns query
start/end/midpoint ve5 actual gap, interval0.002s, held64MiB ordinary CPU tensor,
release weakref=True; lifetime HWM reset=False. Başarılı worker exit0; failure
worker exit0/expected exception, sampler_threads0; controlled crash exit7 ve
başarılı result yok. Tüm üç worker terminated; pointer cache23→23.
P11 source result `90711438-484a-4e28-be5f-ded2f32224ed`: gerçek artifact2820 B,
SHA256 `3c461bcc170f76254663564083b77e27ef43cb091102476d7776e10c113f9df5`,
weights_only roundtrip_verified. Ayrı installed final2 sonuçları da güncel schema,
artifact stat/SHA ve cleanup bakımından doğrulandı. Tied/view/cross-category
testinin mevcut byte'ları B2 commit e17a6d2 ile aynı; final JUnit83 memory testinde
PASS: parameter64 B, buffer54 B, object-union94 B. Aynı object dedup edilir;
farklı view object'leri storage paylaşsa da logical olarak ayrı sayılır.
Ham kanıtlar `C:/Users/hdgn5/AppData/Local/Temp/hyso-m2pb2-20261020/`
altında `native-windows-cachefix-touch.json`, `native-windows-cachefix.json`,
`installed-windows-final2-touch.json`, `targeted-final2.xml` ve artifact dosyalarıdır.
Bu son kontrolün kaydı:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p10-p11-acceptance-20261022/acceptance-verification.json`.
17 implemented feature için kayıtlı dar kapsamların tümü PASS:15 CUDA +2 Windows/CPU.
Önceki15 CUDA sonucu tekrarlanmadı. Bu tam precision/platform acceptance veya release
GO değildir: P25 AMP ve Linux native gibi ölçülmeyen kapsamlar açık kalır.
B4/P23/B10 ve eski R1 release NO-GO kapanmadı; güncel tüm source değişikliklerini
kapsayan final wheel/sdist/installed/minimum release gate bu kabulden çıkarılamaz.

P19/P20/P22/P25 kanıtı: tek `linear_fp32_p19_p20_p22_p25_training` execution,
run ID `be71ecb2-b397-4365-ad3f-a3d6412e0e0d`, public P19 segment ID
`1e96848e-ebd0-4c78-8a19-158e44277e59`; P20/P22/P25 aynı source segment ve
aynı P19 payload/hash üzerinden doğrulandı. Public formatlar sırasıyla
`training_timing`, `training_stages`, `training_throughput`, `gradient_diagnostics`;
tam result kimlik/hash kayıtları ve ham ölçümler:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-training-native-20261022/metrics.json`,
`training-candidate/native-execution-1/result.json`, `postflight.json`.
Gerçek Trainer + AdamW, Linear(3,4)/Dropout(0.2)/Linear(4,2), seed71,
batch2/2/1, accum2, lr0.01, grad_clip0.1, amp=False, epoch1; retry/baseline0.
Model/optimizer ve Python/NumPy/Torch CPU/CUDA RNG öncesi/sonrası bounded
state/hash kanıtı ölçülen fit dışında kaydedildi; rollback veya ikinci fit yok.
P19 host microbatch süreleri 117.7554/8.7820/8.4390 ms, optimizer attempt
süreleri 125.6789/10.2167 ms; P20 CUDA forward 39.057407/0.317440/3.608576 ms,
backward 13.880448/7.223136/3.476192 ms, optimizer 32.874687/0.419040 ms.
Host clock perf_counter_ns, CUDA event selected default stream0; event drains17,
pending0, warmup0, validation yok. P25 norm query senkronizasyonu ve overhead'i
P19/P20 ölçümlerini etkiler; stage scope'ları inclusive, çapraz stage toplam yok.
CUDA initialization gerçek model/backward/optimizer/event sonuçlarıyla doğrulandı.
Controller/parent/worker exit0, crash/OOM/timeout yok; child/outer active0,
altı owned/controller PID exited/absent. Source51/runtime26 hash, Job exact
membership/readback, process4GiB, child8GiB/4, outer12GiB/8, worker60s/outer90s,
allocator0.15, kill-on-close ve bounded logs/artifacts doğrulandı.
GPU context unavailable warning-only; ölçüm güvenilirliği `unqualified`, kontrollü
karşılaştırma yok. P23 activation hooks kapalı; B4 hâlâ OPEN/UNRESOLVED.

P05/P17 sonraki native kabulü: public `measure_cold_start` ve
`find_max_batch_size`, private scoped launcher üzerinden aynı forensic supervisor'a
bağlandı; public imza/result schema değişmedi, ölçüm worker'ları yeniden yazılmadı.
Bu sealed native yolda fresh interpreter'lar `-I -S -B` ile inherited owned Jobs
içinde başlar; ayrıca nested Job assignment receipt, exact membership ve hard-limit
readback tamamlanır. Torch importundan önce full source/runtime seal, request SHA,
controller/parent/worker zinciri, token ve canlı parent process handle üzerinden
PID + creation-time doğrulanır. PID numarasıyla kill/reopen kullanılmaz.
Kill-on-close, bounded stdout/stderr/result ve owned-Job cleanup korunur.
12 hedefli stdlib/offline test geçti: pre-import ret, parent creation identity,
request/runtime mismatch, missing receipt/limit, timeout/crash/simulated OOM,
orphan cleanup, public failure schema ve mevcut scenario contract uyumu.

P05 run `783be515-4547-4a8b-b683-7d652bd01503`, public ColdStartResult
`ee019edc-e445-48c0-a944-83d53e72d25f`; manifest SHA256
`19a30f0fc1fa5f693f0cc5b1fb66f8ec6c43884e00e74825ec193128cb487e71`.
Tek fresh worker; CPU Conv2d state_dict SHA256 doğrulanarak weights_only CPU load,
sonra CUDA:0 FP32 preparation. Module import ve model construction sonunda CUDA
initialized=False, runtime preparation sonunda True. İlk forward1, warmup1,
warm reference2, toplam4 forward. Tüm fazlar gerçek worker perf_counter_ns
completion wall süreleridir; startup receipt ayrıca guard/pipe/reader overhead içerir.
Warm reference toplam0.4815 ms; ayrı warm inference samples0.1436/0.1387 ms.
OS page cache ve GPU driver state cold değildir/garanti edilmez.

P17 run `50f2ca0b-789a-4aed-a6cf-8e821af636a3`, public CapacityResult
`130f5f98-f31a-48ac-a070-7fc93300d7e7`; manifest SHA256
`a44fe0d497fef4c62fbcd1ff85e037033c8092dee41fbb51db08f1cb67288dac`.
Batch1/2 iki ayrı fresh worker'da
gerçek shape1×1×7×9 / 2×1×7×9, FP32 pytorch_eager, CUDA default stream0.
Her adayda public latency protokolü warmup1 + measured2; CUDA allocator probe
ayrı single forward'dur ve latency sample sayısına eklenmez. Probe peak3072/3584 B,
release sonrası allocated2048 B; fiziksel VRAM veya P10/P11 acceptance değildir.
İki aday feasible, OOM/unsupported0; search_bound4, explicit_points_only,
range_exhaustive=False, maximum_observed_feasible_batch2,
maximum_proven_within_range=null. OOM sınırı/global maksimum ölçülmedi;
throughput null/disabled/not_requested olarak korundu.

Her özellikte ayrı manifest/seal/marker, bir üst düzey native execution; toplam2,
retry0/baseline0. P05 nested40s, P17 her candidate20s; worker60s/outer90s,
process4GiB, child8GiB/4, outer12GiB/8, CUDA allocator0.15. Controller/parent/worker
ve üç nested workload worker exit0; crash/OOM/timeout yok. Outer/child/nested
active-process0 ve her run'da kaydedilmiş10 owned/controller PID exited/absent.
Her run'da source51/runtime26 hash önce/sonra doğrulandı. GPU telemetry unavailable
warning-only, measurement reliability unqualified; kontrollü karşılaştırma yok.
P23 activation hooks kapalı; B4 OPEN/UNRESOLVED. Önceki audit/marker'lar değişmedi.
Ham public sonuçlar, phase/candidate/provenance ve güvenlik kanıtları:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p05-p17-native-20261022/p05-metrics.json`,
aynı dizinde `p17-metrics.json`, `p05-postflight.json`, `p17-postflight.json`,
`p05-candidate/native-execution-1/result.json`,
`p17-candidate/native-execution-1/result.json` ve `preservation-final.json`.

Execution 1'de GPU paylaşımı, unknown PID, WDDM N/A ve import sırasında eskiyen capture yalnızca
warning/unqualified metadata üretti. Source/runtime, Job ownership/readback,
60/90s deadline, bounded artifacts ve active-process=0 cleanup doğrulandı.
Controller/parent/worker exit0; OOM/crash/timeout yok. **B4 OPEN/UNRESOLVED**;
bu tablo 17/17 native acceptance veya release GO değildir.

P02'nin yeni tek execution kanıtı:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p02-native-execution-20261022/execution-result.json`.
Sealed run ID `39c35fb7-b570-4777-a6c0-7fa050a5d2e3`;
public result ID `6e706916-8546-4f3c-b144-131fe2aaf3bb`.
Ledger: admitted5=completed5+pending0, image10=2×5, ID0–4,
max pending2, backpressure sync1, terminal drain1; conservation PASS.
Clock `perf_counter_ns`, scope `resident_forward_completion_wall`, default stream0,
synchronization `bounded_group_completion_and_terminal_event`; warmup1 hariç.
Controller/parent/worker exit0; OOM/crash/timeout yok; outer/child active0 ve
tüm owned PID'ler exited/absent. Source51/runtime26 hash doğrulandı.
GPU context toplanmadığı için unavailable warning'leri ve `unqualified` ölçüm
güvenilirliği korunur; bu kısa tek pencere kontrollü karşılaştırma veya sürdürülen
throughput iddiası değildir. Diğer özelliklerin kanıt kapsamı genişletilmedi.

P03/P04 yeni kanıtı: tek `measure_pipeline` çağrısı ve onun sonucuna uygulanan
`analyze_stages`; ikinci pipeline veya baseline yok. Kaynak:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-validation-completion-20261022/result-summary.json`
ve aynı dizindeki `metrics.json` / `pipeline-candidate/native-execution-1/result.json`.
Run ID `fc21f1a4-2a80-4c32-9942-a812fc5b1c25`; manifest SHA256
`b6d93e2e52d23ac2da4ded87a9f3e4b891365a3d31deeae15b466b6f65c70235`.
Measured frame ID'leri `frame-1, frame-2, frame-3`; warmup `frame-0`;
source/delivered sırası aynı, pending0/dropped0, forward count4.
P03 `perf_counter_ns` E2E scope'u kaynak okuma, bounded queue beklemesi,
declared stages ve instrumentation/terminal completion içerir; delivery hariçtir.
P04 CPU preprocessing için yalnız host süresi, CUDA postprocessing için ayrı
host elapsed ve `end_event_sync` selected-stream event süresi verir. Stage
süreleri toplanarak E2E üretilmedi. Üç örnekte P99 null/insufficient_samples.
Controller/parent/worker exit0; OOM/crash/timeout yok; outer/child active0 ve tüm
owned PID'ler exited/absent; source51/runtime26 doğrulandı. GPU context unavailable
warning'leri nedeniyle quality `unqualified`; exclusivity veya karşılaştırma yok.
Bu P03/P04 kabulü sırasında P05/P17 nested-worker admission eksikleri nedeniyle
BLOCKED idi; diğer yeni runner bağlantıları o aşamada yalnız offline READY idi.
Sonraki P05/P17 native kabulü aşağıda ayrıca kaydedilmiştir.

P06/P07/P08 için sonraki yetkili kabul: her senaryo ayrı run ID/manifest/seal,
worker, execution marker ve cleanup ile bir kez çalıştı; toplam3 execution,
retry0 ve ek baseline0. Her noktanın latency geçişinde2 gerçek CUDA event sample;
ayrı completion geçişinde3 completed batch, pending0, terminal drain1 kaydı var.
Completion image sayıları P06 batch1/2 için3/6, P07 ve P08 noktalarında6'dır.
Latency seçili default stream'de end-event sync ile ms/batch ölçülür; throughput
gerçek `perf_counter_ns` completion window süresinden images/s hesaplanır.
Her geçişte1 warmup ve1 untimed preflight vardır; bunlar measured count'a katılmaz.
Bu iki geçiş aynı forward olarak sunulmaz. Public raw schema'lar, point/result ID
bağları, precision/backend, requested/actual shape ve sample sayıları korunur.
Üç run'da crash/OOM/timeout yok; controller/parent/worker exit0, outer/child active0
ve owned PID'ler exited/absent; source51/runtime26 hash doğrulandı. Warning-only
GPU context unavailable ve quality `unqualified` kaydı korunur. Kısa pencerelerin
oranları speedup karşılaştırması, global optimum veya P28 accuracy kanıtı değildir.
Ham kanıt ve her noktanın ayrı ledger/provenance kaydı:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p06-p07-p08-native-20261022/P06-result.json`,
aynı dizinde `P07-result.json`, `P08-result.json` ve `summary.json`.
Bu sonuçlar yalnız seçilen iki batch, iki çözünürlük ve iki precision profilini
kapsar; diğer senaryolar çalıştırılmadı ve B4 OPEN/UNRESOLVED kaldı.

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
