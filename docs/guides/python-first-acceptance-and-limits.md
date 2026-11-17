# Python-first kullanım, kabul ve kanıt sınırları

Bu rehber mevcut Python API'lerini bir araya getirir; yeni runtime kabulü veya
otomatik optimizasyon tanımlamaz. 2026-11-16 tarihli [13.5.16 kabulü](../reports/stage13-5-16-cpu-cuda-performance-acceptance.md)
**PARTIAL — CPU VERIFIED / CUDA BLOCKED** durumundadır. CPU kabulü kayıtlı iki
sentetik fp32 fixture kapsamındadır. CUDA admission **BLOCKED**, execution
**NOT_TESTED**, B4 **ROOT_CAUSE_UNRESOLVED** olarak kalır.

<a id="urun-akisi"></a>
## Ürün akışı ve sorumluluklar

HYSo, PyTorch üzerinde Python-first model geliştirme, eğitim, değerlendirme ve
performans analizi framework'üdür. Studio/web ürünü kaldırılmıştır; HTML rapor
exportu bir web uygulaması değildir.

| İş | Gerçek giriş noktası | Sorumluluk ve sınır |
| --- | --- | --- |
| Model oluşturma/düzenleme | `hyso.ModelBuilder`, `hyso.GraphModelBuilder` | Sequential schema1/2 veya DAG schema3; config ağırlık dosyası değildir |
| Graph Tensor çıktısı | `hyso.models.output.GraphTensorOutput` | Tek giriş veya CPU sınıflandırmada iki açık named input; aynı Graph ve parametreleri paylaşır |
| Eğitim | `hyso.TrainingConfig.prepare`, `hyso.TrainingSession.fit`, `hyso.Trainer.fit` | Açıkça istenen eğitim ağırlıkları günceller; Session bir mevcut Trainer.fit çağrısını yönetir |
| Değerlendirme | `hyso.evaluate_predictions`, `hyso.evaluate_classification` | Hazır prediction skorlama ile gerçek model inference ayrı işlemlerdir |
| Benchmark | `hyso.BenchmarkRunner.run`, `hyso.performance.measure_throughput` | Latency ve tamamlanan iş penceresi; ayrı zamanlama kapsamları |
| Analysis / Advisor | `hyso.analysis.EvidenceIndex`, `hyso.advisor.advise`, `hyso.advisor.recommend` | Mevcut kanıtı yorumlar; eksik ölçümü veya optimizasyonu başlatmaz |
| Karşılaştırma | `hyso.architecture_comparison.compare_architectures`, `hyso.compare_model_structures`, `hyso.analysis.compare_models` | Mimari, registered yapı ve uygun native metrikler ayrı kaynaklardır |
| Raporlama | `hyso.reporting.create_report`, `hyso.reporting.report_experiment` | TR/EN, JSON/Markdown/HTML; mevcut typed sonuçların sunumu |
| Deney kaydı | `hyso.ExperimentManager` | Açık lifecycle, metrik ve artifact kaydı; model registry veya iş yürütücüsü değildir |

Analyzer/Advisor kullanıcı modelinin kaynak kodunu, mimarisini, hiperparametrelerini
veya ağırlıklarını otomatik değiştirmez. Yeni model oluşturma ve düzenleme Builder
sorumluluğudur; öneriyi uygulamak kullanıcıya aittir. `analysis.execute_analysis`
ayrı, açıkça yetkilendirilen workload yoludur; salt-okunur advice çağrısı değildir.
Benchmark state guardları da keyfi Python yan etkilerine genel transaction garantisi
vermez. [Python-first iş akışı](python-first-workflow.md) bu sınırları ayrıntılandırır.

<a id="destek-matrisi"></a>
## Gerçek destek matrisi

| Alan | Mevcut destek | Sınırlı veya desteklenmeyen kapsam |
| --- | --- | --- |
| CPU model iş akışı | Bounded Builder/Graph forward-backward, sınıflandırma eğitimi, evaluation, strict state restore ve kayıtlı CPU ölçümleri | Bütün component/topoloji/shape birleşimlerinin kabulü değildir; 13.5.12nin sınırlı 54 kombinasyonundan 36sı NOT_TESTED |
| Graph | Raw DAG; tek veya iki açık named input ve iki classification loss için Trainer/TrainingSession; tek seçilmiş Tensor için GraphTensorOutput | Tensor adapterında iki input için named_inputs=True ve açık task gerekir; genel multi-task eğitim yok; tüm Graph çıktıları yürütülür |
| Task/head | Classification, regression ve embedding head oluşturma; mevcut Session/evaluation ile binary/multiclass sınıflandırma | Head oluşturabilmek genel regression/embedding/segmentation eğitim kabulü değildir |
| Checkpoint | Strict ModularCNN v1 / seçilmiş GraphTensorOutput v2; tek-input CPU fp32 iki-loss raw Graph v2, exact model_config ve NamedTargetDataset | İki-input/tek-output için NamedTensorDataset; iki-input/iki-output automatic resume yok; BuiltClassifier desteklenmez |
| İki-head evaluation | analysis.evaluate_graph_heads ile ayrı binary/multiclass metrikler, label şemaları, stream ve protokoller; generic artifact roundtrip | Tek-input CPU fp32 NamedTargetDataset; iki-input evaluation yok; global accuracy veya native performans receipt üretmez |
| Graph checkpoint güveni | Eşleşen wrapper/state yapısı ve desteklenen açık receiver sözleşmeleri | `graph.` key prefixi korunmalı; import edilen dosya process-local live receiver otoritesini yeniden yaratmaz |
| CUDA | Bazı public API'lerde CUDA profili bulunur; tarihsel scoped native kanıtlar korunur | 13.5.16 fixture CUDA kabulü yok; üretim CUDA saved-activation instrumentation kapalı, B4 açık |
| CLI | `models`, `train`, `analysis`, `compare`, `advisor`, `experiments`, `reports` | Python API ile tüm girdi/politika seçenekleri eşit değil; [CLI ayrımı](#cli) geçerli |

13.5.18 Python örnekleri ve kesin resume/loss sınırları için
[Graph iş akışına](python-first-workflow.md#graph-training-and-saved-results) ve
[training rehberine](training-config.md#selected-graph-automatic-resume) bakın.
Bu dar CPU geliştirmesi CUDA/B4 kabul durumunu değiştirmez.

13.5.16 kayıtlarında geliştirme kaynağında **174**, commercial wheel/sdist
allowlistinde **164** Python modülü bulunuyordu; on development dosyası bilinçli
olarak dışlanmıştı. Bu tarihsel envanter yeni modül kabulü değildir. 13.5.12deki Registry
envanteri source için **85**, commercial için **57** kayıt içerir. Modül sayısı,
Registry kaydı, Graph operatorü ve çalıştırılmış component sayısı farklıdır.
Kaynakta bulunan named ailelerin hepsi kurulu commercial pakette bulunmaz.
[Component matrisi](../reports/stage13-5-12-component-matrix.md) ve
[commercial allowlist](../../commercial-profile.json) profil sınırlarının kaynağıdır.
Sabit tarihsel [roadmap](../roadmap/hyso-architecture-roadmap-v1.md) güncel kabul
tablosu olarak okunmamalıdır; geçmiş Planned/Implemented satırları geriye dönük
yeniden yazılmaz.

<a id="karsilastirma"></a>
## Model karşılaştırması: amaç, politika ve kimlik

Karşılaştırma yalnız latency/accuracy değildir. `compare_architectures` input/output
kontratları, nodes/connections/branches ve parametre sınırlarını karşılaştırır;
`exact_parameters=True` açık CPU model oluşturması ister. Registered yapı
snapshotları parametre, modül, tensor, alias, dtype ve logical byte farklarını
sunar. Dosya boyutu `measure_footprint` artifact gözlemidir; state kimliği ayrı
benchmark/evaluation/checkpoint bağlarından gelir. Registered modül sayısı
çalıştırılmış operator sayısı değildir. [İki-model rehberi](two-model-comparison.md)
metrik ve eligibility kapsamını, [yapı karşılaştırması](model-structure-comparison.md)
registered snapshot sınırını açıklar.

| Amaç | Kimlik kuralı | Sonucun anlamı |
| --- | --- | --- |
| Aynı model / aynı state replikasyonu veya cihaz paritesi | Mimari, state SHA ve input kimliği eşleşmeli; cihaz değişikliği parite protokolünde açık olmalı | İki rastgele model parite kanıtı değildir; matching state tek başına eş süre/çıktı garantisi vermez |
| Eski/yeni checkpoint sürümü | İki state/checkpoint kimliğini ayrı tut; izinli model değişimini ve sabit diğer koşulları belirt | State SHA farklı olabilir; bütün sürüm karşılaştırmalarına SHA eşitliği dayatılmaz |
| Farklı mimariler | Her mimari ve state ayrı tanımlanır; ortak ölçüm koşulları ve değişken ekseni açık olur | Gözlenen fark genel winner, nedensel iyileşme veya istatistiksel üstünlük değildir |

`model_id` etiketi state kimliği değildir. Architecture/config identity modeli
tarif eder; state SHA ilgili producer/fingerprint konvansiyonundaki ağırlık ve
buffer durumunu tanımlar; artifact SHA kaydedilmiş dosyanın tam byte'larını korur.
Aynı state farklı serialization byte'ları üretebilir. Farklı fingerprint
konvansiyonlarını doğrudan eşitlik testi gibi kullanmayın. Dosya SHA doğrulaması
model eşliği, veri kaynağı veya üretici kimliğinin bağımsız doğrulaması değildir.

### Legacy ve açık strict seçimi

`compare_models(..., context_policy="legacy")` mevcut varsayılandır ve schema1
üretir. `context_policy="strict"` açık opt-in ile schema2yi seçer; P26 seçili
olmalıdır. Dataset ID, split ve preprocessing ID beyanlarının selected record ve
report context içinde mevcut ve tutarlı olmasını ister; native task/class/stream
koşulları ayrıca geçerlidir. Graph için mevcut
`hyso.analysis.graph_evidence.compare_graph_evaluations(..., context_policy="strict")`
strict schema3 yoludur. Strict, beyan tutarlılığını denetler; veri kaynağı veya
preprocessing uygulamasının otoritesini yükseltmez.

Aşağıdaki tarif hazır, doğrulanmış analysis raporlarını karşılaştırır; model veya
checkpoint yüklemez. Model etiketleri raporlardaki ilişkilendirmeyle eşleşmeli;
birden fazla capture varsa `ModelComparisonSource(records=...)` ile kayıt seçimi
ayrıca yapılmalıdır. Eksik bağlar bu fonksiyon tarafından doldurulmaz.

```python
from hyso.analysis import (
    ModelComparisonSource, UnifiedAnalysisReport, compare_models,
)

def compare_saved_p26(baseline_path, candidate_path, baseline_id, candidate_id):
    baseline = ModelComparisonSource(
        baseline_id, UnifiedAnalysisReport.from_json(baseline_path))
    candidate = ModelComparisonSource(
        candidate_id, UnifiedAnalysisReport.from_json(candidate_path))
    return compare_models(
        baseline, candidate, features="P26", context_policy="strict")
```

Bilinen context uyuşmazlığı `INCOMPATIBLE`, eksik gerekli kanıt
`INSUFFICIENT_EVIDENCE` sonucudur. Strict ret halinde quality difference alanları
nested sonuçlarda da null kalır. Legacy varsayılanın dört strict-beklenti kaydı
13.5.14 kapanışında hâlâ FAIL olarak korunmuştur; strict yolun PASS olması
varsayılanı strict yapmaz. [Strict kapanış](../reports/stage13-5-14-strict-comparison-closure.md)
bu uyumluluk kararının kanıtıdır.

Native performance comparison eligibility ayrıca producer-bound capture,
clock/protocol, batch/shape/device/precision ve ilgili kaynak bağlarını ister.
Tek bir metrik gözlenebilir olsa da iki kaydın farkı uygun olmayabilir.
`hyso.benchmark.compare_results` içindeki compatible tanısı daha katı native
performance gate'inin yerine geçmez. Caller beyanı eksik producer capture'ı
tamamlamaz; eksik kanıtla `BEST_MODEL`, genel üstünlük veya optimizasyon başarısı
üretilmez.

<a id="kanit-protokol"></a>
## Kanıt, protokol ve sunum otoritesi

| Kimlik / nesne | Anlamı | Kullanım kuralı |
| --- | --- | --- |
| External QA protocol SHA | Tüm dış deney planının byte/content kimliği | Açıkça adlandırılmış generic run config/provenance/metadata alanında veya ayrı artifact olarak saklayın |
| Native producer protocol SHA | Producerın kendi ölçüm protokolünün kimliği | `EvidenceScope.protocol_sha256` native değerle çelişemez; dış zarf SHA'sıyla değiştirmeyin |
| `EvidenceRecord.from_native` | Mevcut native payloadı kendi reader sözleşmesiyle doğrular | Artifact kimliği canonical in-memory JSON içindir; dosya yazıldığını iddia etmez |
| `EvidenceRecord.from_json` | Dosyayı `feature_id` ve `expected_sha256` ile okur | Byte doğrulaması ile native/declared bağlar ayrı kalır |
| `EvidenceIndex` / `AnalysisContext` | Recordları birleştirip açık caller context ile yorumlar | AnalysisContext beyanları ölçüm veya attestation değildir |
| `PresentationReport` | Mevcut typed sonuçların sunumu | `presentation_only=true`, `authoritative_evidence=false` korunur |

13.5.16daki external/native SHA karışıklığı doğru biçimde
`native_scope_mismatch:protocol_sha256` ile reddedilmişti. Guard gevşetilmedi;
external kimlik ayrı QA/store metadata'sına taşındı. Hashleri yeniden adlandırarak
authority üretmeyin. Mismatch, invalid payload, eksik kanıt ve native safety
block ayrı durumlardır; birini diğerinin yerine PASS göstermeyin.

<a id="evaluation-raporlama"></a>
## Tarif: standard DataLoader, evaluation ve Reporting adapterı

Kod blokları fonksiyon tanımlar; çağrılmadıkça model işi veya dosya/store yazımı
başlatmaz. İlk fonksiyonun açık çağrısı küçük CPU inference yapar; öğrenme veya
gerçek veri kalitesi iddiası yoktur. Bu belgelendirme görevinde örnekler yalnız
statik olarak kontrol edilmiştir, çalıştırılmamıştır.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso import ModelBuilder, evaluate_classification

def evaluate_cpu_fixture():
    builder = ModelBuilder((1, 8, 8), num_classes=2).add(
        "conv2d", in_channels=1, out_channels=2, kernel_size=3, padding=1)
    model = builder.build().eval()
    dataset = TensorDataset(
        torch.zeros(4, 1, 8, 8), torch.tensor([0, 1, 0, 1], dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)
    return evaluate_classification(
        model, loader, class_names=["a", "b"], split="qa",
        source_id="synthetic-qa", capture_evidence=True, device="cpu")
```

`evaluate_classification` bir `nn.Module` ve standard `DataLoader` ister; tensor
batch listesi eşdeğer değildir. Bu profil finite sequential map-style dataset,
workers=0 ve drop_last/pin_memory kapalı kullanımını ister. Isolated deepcopy
üzerinde eval/inference yürür; `capture_evidence=True` aynı geçişten schema2
producer evidence üretir. Ordered input/label hashleri veri kaynağını doğrulamaz.
Default schema1dir; structure/receiver capture ayrı opt-in ve destek şartlarına
bağlıdır. CUDA evaluation capability, CPU producer-evidence kapsamını genişletmez.

`create_report(EvaluationResult)` doğrudan desteklenmez. Mevcut adapter yolu:

```python
from pathlib import Path
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord
from hyso.reporting import create_report

def export_evaluation(evaluation, output_dir):
    record = EvidenceRecord.from_native(evaluation.to_dict(), feature_id="P26")
    analysis = EvidenceIndex([record]).analyze(context=AnalysisContext(
        workload_id="qa-evaluation", task="multiclass",
        model={"label": "qa-model"}, dataset={"split": "qa"}))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # Yeni, ayrı çıktı dizini.
    for language in ("tr", "en"):
        report = create_report(analysis, language=language, level="expert")
        report.to_json(output / f"report-{language}.json")
        report.to_markdown(output / f"report-{language}.md")
        report.to_html(output / f"report-{language}.html")
    return output
```

İkinci fonksiyon aynı evaluationdan TR/EN çıktılar üretir; inference tekrarlamaz.
Context içindeki model etiketi veya split beyanı eksik native dataset/preprocessing
bağlarını tamamlamaz; bu tarif tek başına strict comparison eligibility kurmaz.
`create_report` seviyeleri `beginner`, `technical`, `expert`tir. Tam ham kaynak
ve metric interchange için JSON veya expert kaynak bölümlerini kullanın.
`PresentationReport` tekrar `create_report` girdisi değildir; kendi
`from_json` / export metotları vardır. [Typed girdi matrisi](../reports/stage13-5-15-reporting-format-matrix.md)
diğer desteklenen sonuç türlerini ve authority sınırlarını listeler.

<a id="artifact-migration"></a>
## Tarif: yeni QA storeunda explicit artifact migration

Yeni store schema1 snapshot/metric desteğiyle açılır; artifact için store kapalıyken
`ExperimentManager.migrate_store` ve yeniden açma gerekir. Aşağıdaki fonksiyon
yalnız yeni bir store kökü kabul eder. `store_root` OneDrive/network dışında,
storea ayrılmış yerel bir dizin olmalıdır. Mevcut kullanıcı veya eski Studio
SQLite dosyası bu tarife verilmez.

```python
from pathlib import Path
from hyso import ExperimentManager

def archive_qa_report(store_root, report_path, external_qa_protocol_sha256=None):
    root = Path(store_root)
    if root.exists():
        raise FileExistsError("Yeni, storea ayrılmış QA dizini gerekli")
    manager = ExperimentManager(root)
    manager.close()
    ExperimentManager.migrate_store(root)
    manager = ExperimentManager(root)
    try:
        experiment = manager.create_experiment("QA evaluation report")
        run = manager.create_run(experiment["id"], "evaluation", config={
            "external_qa_protocol_sha256": external_qa_protocol_sha256})
        run.start()
        artifact = run.add_artifact(report_path, kind="report", ownership="copy")
        verification = manager.verify_artifact(artifact["id"])
        if verification["status"] != "verified":
            raise RuntimeError(verification["status"])
        run.finish()
        return artifact, verification
    finally:
        manager.close()
```

Buradaki external SHA caller metadata'sıdır; native protocol identity değildir.
`ownership="copy"` mevcut report byte'larını managed artifacta kopyalar;
`verified` kayıtlı size/SHA eşliğidir, typed native authority değildir. Başarısız
kayıt lifecycleı otomatik tamamlanmaz; storeu incelemeden tekrar çalıştırmayın.
Typed `open_result` whitelist'i TrainingResult, AnalysisExecutionResult ve
ModelComparisonResult ile sınırlıdır. Graph/Evaluation sonuçları generic artifact
olarak saklanabilir; typed workflow `architectures` alanı GraphModelBuilder kabul
etmez. [Experiment iş akışı](experiment-workflow.md) mevcut generic rotaları açıklar.

<a id="cli"></a>
## CLI keşfi ve Python API farkları

Aşağıdaki komutlar katalog/plan/metadata preflight içindir; burada çalıştırılmış
testler değildir. Son satırda `--execute` yoktur; gerçek workload başlatmaz.
Kaynak veya önkoşul eksikse dry-run BLOCKED/UNSUPPORTED sonucu verebilir.

```powershell
python -m hyso analysis list
python -m hyso analysis show P01
python -m hyso analysis plan --features P01 P02 --model-id qa-model --device cpu
python -m hyso analysis run --features P01 --model-id qa-model --device cpu
```

| İş | Python | CLI sınırı |
| --- | --- | --- |
| Strict P26 karşılaştırma | `compare_models(..., context_policy="strict")` | `compare --context-policy strict` aynı native policyyi seçer; flag yoksa veya `legacy` ise mevcut default korunur |
| Hazır strict sonuç sunumu | Typed reader + `create_report` | `reports render` kaydedilmiş ModelComparisonResult strict schema2yi sunabilir |
| Graph sonucu sunumu | Typed Graph readerları + `create_report` | `read_result` GraphEvaluation/GraphComparison dispatch sunmaz |
| PresentationReport yeniden açma | `PresentationReport.from_json` ve kendi exportları | `reports render` PresentationReport girdi dispatchi sunmaz |
| EvaluationResult sunumu | Yukarıdaki EvidenceRecord/Index adapterı | EvaluationResult doğrudan CLI/report girdisi değildir |
| Analysis execution | Açık policy/preflight ile ayrı yürütme API'leri | `run --execute` ayrıca exact `--authorize` ve gerekli workload izinlerini ister; plan/list feature kabulü değildir |

Hazır analysis raporlarıyla gerçek strict CLI sözdizimi:

```powershell
python -m hyso compare baseline-analysis.json candidate-analysis.json --baseline-model A --candidate-model B --features P26 --context-policy strict --format json --output strict-comparison.json
```

`--context-policy {legacy,strict}` yalnız mevcut generic iki-model karşılaştırma
yoluna aktarılır; ek `--strict` bayrağı yoktur. Strict için P26 seçilmemişse native
validation hatası CLI exit 2 üretir. Geçerli `INCOMPATIBLE` veya
`INSUFFICIENT_EVIDENCE` sonucu ise exit 0 ile dönebilir; komutun tamamlanması
kanıtın karşılaştırılabilir olduğu anlamına gelmez. Eksik context tamamlanmaz.
Python ve CLI aynı strict eligibility/schema2 kurallarını kullanır; CLI girdileri
hazır UnifiedAnalysisReport veya AnalysisExecutionResult ile sınırlıdır. Pythonun
explicit record/standalone structure/benchmark kaynak seçenekleri ve Graph
comparison readerları bu bayrakla CLIye eklenmez; CLI `include_structure=False`
varsayılanını korur. Kaydedilmiş strict sonucu render etme yolu değişmez.
[Native executor rehberi](native-analysis-executor.md) kaynak/izin/protokol
kontratlarını açıklar. Eski P01–P35 A/B 35/35 sonuçları tarihsel declared scope
kabulüdür; CLI kataloğunda görünmek yeni cihaz veya workload kabulü sağlamaz.

<a id="olcum-sinirlari"></a>
## Bellek, tail ve throughput yorumlama

[Bellek alanları](performance-memory.md#bellek-alanlari) process RSS, sampled
maximum, lifetime HWM, logical parameter byte, serialized checkpoint byte ve CUDA
allocator sayaçlarını ayırır. [Latency/tail ve throughput](performance-inference.md#latency-tail-throughput)
descriptive percentile, default TailPolicy eligibility, bağımsız örnek ve clock
kapsamlarını açıklar. Bu metrikler birbirinin yerine geçirilmez.

13.5.16 fixture FLOPs/MACs, CPU utilization ve model-exclusive peak RSS
**NOT_MEASURED** kalır. Reporting'in yaklaşık 3,5–3,7 s p50 kaydı yalnız
16-record expert create + JSON/Markdown/HTML bileşiğidir; aşama profili veya
kanıtlanmış darboğaz nedeni değildir. CUDA guardları, admission kimlik açığı ve
eksik eski diagnostic locatorı bu rehberle kapanmaz; [kayıtlı backlog](../reports/stage13-5-16-performance-usability-backlog.md)
geçerli sınırları korur.
