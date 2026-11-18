# P01–P35 native analiz yürütücüsü

`hyso.analysis` içindeki `native` modu, operasyon başına kaynak ve native policy
ile çalışır. Uygulama sırası ve teknik envanter mevcut 10A planındadır.
[10A entegrasyon planı](../roadmap/stage10a-executor-integration-plan.md)

## Plan, kaynak, inceleme ve execution

```python
import torch
from hyso.analysis import (
    ModelContext, NativeBinding, InferenceSource, OperationPolicy,
    plan_analysis, preflight_analysis, execute_analysis,
)

model = torch.nn.Linear(4, 2)
binding = NativeBinding(
    operation_id='logical-footprint', model_ids=('model-A',), features=('P11',),
    source=InferenceSource(model), policy=OperationPolicy(allow_capture=True),
)
plan = plan_analysis(ModelContext('model-A'), features='P11', execution_mode='native')
review = preflight_analysis(plan, (binding,))
print(review.to_dict())
result = execute_analysis(review, authorized_features=('P11',), explicit_authorization=True)
print(result.to_dict()['status_counts'])
```

`NativeBinding` bir operation ID, bir veya iki model etiketi, özellik listesi,
tipli source, execution device, operation policy ve isteğe bağlı pinned dosyalar
taşır. Aynı operation içindeki seçili/yetkili projeksiyonlar tek capture kullanır.
P29 için baseline/candidate kaynakları ve iki model etiketi gerekir.
Önkoşul satırları görünür ve `SKIPPED` kalır. G1 saf planının blocker'ları native
dry-run içinde ayrı çözümleme kayıtları taşır. Native uygunluk execution sırasında
üretici tarafından doğrulanır.

`CPUAnalysisPolicy`, `trusted_cpu` ve `isolated_cpu` mevcut girişleri sürdürür.
Yeni kaynak politikaları `OperationPolicy` içindedir. Capture, training, CUDA,
kamera, uzun çalışma, power sampling ve kalıcı dosya yazımı ayrı alanlardır.
`peak_reset_pid` açık izin verilen mevcut allocator process PID'sini belirtir.
Native binding metadata'sı değişirse yeni dry-run gerekir. Ticket tek kullanımlıdır.

## Kaynak seçimi

| Source | Özellikler ve native kaynak |
|---|---|
| `InferenceSource` | P01/P02 resident model/input, P06/P07/P08 CPU master + input_factory + explicit points, P09 CUDA allocator, P10 process workload/snapshot, P11 logical/artifact footprint |
| `PipelineSource` | P03/P04 model + sample source + native PipelineProtocol/stage seçenekleri |
| `RecipeSource` | P05 ColdStartRecipe, P12/P13/P14 ortak profiler capture, P17 bounded BatchSearchProtocol, P18 açık yeni profile capture, P23 TrainingMemoryRecipe |
| `EvaluationSource` | P26 model/sequential labeled loader veya targets/predictions + class_names/split |
| `QualitySource` | P24 fresh Trainer factory/cohort/seeds/target, P27 perturbation, P28 precision candidate, P29 explicit model çifti/cohort/bounds |
| `TrainingSource` | P19/P20/P21/P22/P23/P25 fresh Trainer/loaders ve epoch/checkpoint seçenekleri |
| `ProtocolSource` | P15 TransferProtocol, P16 UtilizationProtocol |
| `RooflineSource` | `counter` için P18 mevcut target/reference session'ları. `nsys_modules` P13, `nsys_operators` P14 pinned dosya okuyucuları |
| `RuntimeSource` | P30 StreamSource + trained processor recipe, P31 WorkloadRecipe + süre/journal policy, P34 physical device listesi, P35 ONNX bytes/feed recipe + provider options. recipe=None mevcut P31 dizinini açar |
| `RetainedSource` | P01–P35 doğrulanabilir native payload. P03→P04, P14→P12/P13/P18, P19→P22, P31→P32/P33 türetmeleri |

Source `options` sözlüğü seçilen native public API'nin parametrelerini taşır.
Native protokol sınıflarını kullanın. Bilinmeyen anahtarlar execution sırasında
imza kontrolünden geçemez. `RecipeSource.analysis_options` aynı profiler
capture'ından türetilecek özelliklerin seçeneklerini ID başına taşır.
`RetainedSource.options` tek analizde doğrudan native kwargs, çoklu analizde
`{'P32': {'warmup_seconds': 60}, 'P33': {'max_gap_seconds': 2}}` olabilir.
Runtime capture'ın türetme seçenekleri `options['analysis_options']` alanındadır.
`EvaluationSource` içinde CPU targets/predictions sağlandığında işlem retained
analiz olarak yürür ve capture izni gerektirmez.

## Retained kaynak ve evidence kimliği

```python
from hyso.analysis import ArtifactSource, NativeBinding, RetainedSource
from hyso.analysis._codec import read_json

reference = ArtifactSource.from_path('C:/measurements/P31.json')
payload, _, _ = read_json(reference.path, reference.sha256)
binding = NativeBinding('trend', ('model-A',), ('P32',),
    RetainedSource(payload, options={'warmup_seconds': 60}), artifacts=(reference,))
```

Offline execution device `cpu` seçilir. Native payload içindeki CUDA cihazı ve
precision kendi kaynak kapsamını korur. Hash kimliği canonical native JSON'a
veya açık dosya byte'larına aittir. Model state, checkpoint ve dataset kimlikleri
kendi alanlarında kalır.

`AnalysisExecutionResult.from_dict()` ve `.from_json()` Manager'ın mevcut
snapshot okuyucusunu kullanır. Her evidence native constructor ile yeniden
açılır. P21 worker_trace v1 ve worker_tasks v2 `RetainedSource.companions`
içinde ayrı payload olarak verilir. Companion'ın `source_p21` bağlantısı ve
kendi native SHA kapıları doğrulanır. Aux kaynaklar da Manager reopen sırasında
kapalı reader listesi üzerinden doğrulanır.

Nsight P13/P14 sonuçları mevcut read-only native dosya okuyucusuna yeniden
delege edilir. Pinned SQLite/report/capture ve receipt'in bağlı kaynakları
erişilebilir olmalıdır. Sadece NSYS operator ledger'ından FLOPs türetme
sunulmaz. ATen tabanlı P14 bunun kendi source sözleşmesini kullanır.

EvidenceRecord native payload sınırı 64 MiB, Manager/presentation snapshot
sınırı 128 MiB kalır. Büyük ownership kayıtları açık size-limit hatası verir.
Yan dosyalar source receipt alır. `artifact_sources()` ile bu dosyaların
referanslarını, `verify_artifacts()` ile güncel missing/mismatch durumlarını alın.

## TrainingSession ve tek episode

```python
from hyso.analysis import TrainingSource, NativeBinding, OperationPolicy

# session, TrainingConfig.prepare() ile hazırlanmış kullanıcı session'ıdır.
source = TrainingSource.from_session(session)
binding = NativeBinding('episode-1', ('model-A',), ('P19', 'P20', 'P22'), source,
    policy=OperationPolicy(allow_capture=True, allow_training=True))
```

Session mevcut prepared kontrolü ve running/completed/failed geçişlerini
paylaşır. Native wrapper bir fit çalıştırır. UX TrainingResult aux kaydında,
native timing/stage/throughput kendi evidence formatlarında saklanır.
P22 tek başına retained P19 ister. P21/P23/P25 recorder birleşimleri için
ayrı fresh Trainer episode'ları gerekir. Hata sonrası Trainer state'i ilerlemiş
olabilir ve yeni episode için fresh kaynak sağlanır.

P23 Basic in-process ve isolated zarfları ayrı native constructor kullanır.
`basic_metrics()` ve `diagnostics()` aux görünümde bulunur. Entegrasyondaki yeni
isolated capture Basic kapsamını kullanır. Kaydedilmiş historical Extended
kanıt kendi orijinal formatıyla okunabilir. CUDA saved-activation guard'ı korunur.

## Cihaz, runtime ve donanım kaynakları

CUDA capture için explicit indexed device ve `allow_cuda=True` gerekir. Recipe,
protocol ve resident input/model cihazları eşleşmelidir. P09 mevcut process
allocator reset'ini açık `peak_reset_pid=os.getpid()` ile kabul eder. Bu kaynak
eşzamanlı başka allocator kullanımıyla paylaşılmamalıdır. Owned recipe worker
seçenekleri kendi native süreç sınırlarını kullanır.

P15 cuda:0 native profilini kullanır. P16'nın sampler PID'si operation receipt'te
görünür. GPU selector fiziksel sensör kimliğidir. P18 retained counter sessions
ve analytic model kendi native sonuçlarını korur. Calibration ve Nsight capture
için kaynakları kullanıcı hazırlar.

P30 CPU processor ve açık video/kamera ister. Kamera ayrıca `allow_camera`
gerektirir. P31 capture explicit StabilityProtocol ve yeni runtime dizini ister.
7200 saniye ve üzeri için `allow_long_run` gerekir. GPU P31 native olarak power
sample aldığı için `allow_power_sampling` de gerekir. UUID görünürlüğü altındaki
worker cihazı cuda:0'dır. P32/P33-only yetkisi yeni P31 capture başlatmaz.

P34 en az iki farklı physical target kullanır. P35 seçili runtime, mevcut ONNX
bytes/feed factory ve provider/options alır. Windows owned launcher kullanılır.
Canlı common ORT admission CPU/Azure CPU ve CUDA/TensorRT provider profillerini
kapsar. Diğer provider'lar için kaynak admission'ı unsupported kalır. Retained
native provider kayıtları kendi gözlenen provider adlarıyla okunur.
Provider availability, Node execution, fusion/source mapping ve expectation
mismatch native kayıttaki anlamlarını korur. CUDA kabulü kendi 3+10 ledger
kapısına tabidir.

## ExperimentManager, rapor ve karşılaştırma

Schema2 artifact işlemlerinden önce store migration'ını mevcut API ile açıkça
yapın. Review'daki `tracking_config()` authorization bayraklarını taşımaz.
Analysis workflow mevcut benchmark kind eşlemesini sürdürür.

```python
run = manager.create_workflow_run(experiment_id, 'analysis',
    model_ids=['model-A'], analysis=review)
run.start()
input_ids = []
for reference in result.artifact_sources():
    if reference.verify()['status'] != 'verified':
        raise ValueError('source file changed or became unavailable')
    artifact = run.add_artifact(reference.path, kind='analysis_source', role='input',
        ownership='reference', retain_locator=True)
    if artifact['expected_sha256'] != reference.sha256:
        raise ValueError('registered source identity changed')
    input_ids.append(artifact['id'])
artifact = run.record_result(result, input_artifact_ids=input_ids)
run.finish()
opened = manager.open_result(artifact['id'])
```

`to_evidence_index()`, `to_unified_report(context=...)`, `advisor.recommend`,
`reporting.create_report`, `ModelComparisonSource.from_execution` ve
`compare_models` aynı akışta kullanılabilir. Scalar Advisor ve iki model
karşılaştırması mevcut destekledikleri metrik kümesini korur. Native partial,
unsupported, missing ve failed alanları raporda görünür. Yeni executor coverage
tek başına yeni optimizasyon kuralı oluşturmaz.

Çalıştırılabilir komutlar ve gerekli kaynaklar örnek dizinindedir.
[Native workflow örnekleri](../../examples/analysis/README.md)
