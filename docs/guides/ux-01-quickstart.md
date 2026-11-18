# Kısa Python API ve CLI kullanımı — HYSo v0.2.0

İşe `hyso --help` ile başlayın. Yedi grubun her birinde `--help`, yapılabilecek
işleri ve gerçek komut örneklerini gösterir. `python -m hyso` aynı giriş noktasıdır.

## Kaynaktaki public girişler

| İş | Python API | CLI |
|---|---|---|
| Model tanımlama | `ModelBuilder`, `GraphModelBuilder`; `to_config`, `from_config`, `build` | `hyso models list`, `show`, `validate`, `summary`, `preview`, `inspect` |
| Eğitim hazırlama | `TrainingConfig`, `TrainingSession`; `prepare_datasets` veya `make_loaders` + `prepare` | `hyso train options`, `validate`, `run` |
| Açık eğitim çağrısı | `session.fit()` veya mevcut `Trainer.fit(...)` | `hyso train run ... --execute` ve gerekli izinler |
| Analiz | `hyso.analysis.plan_analysis`, `preflight_analysis`, `execute_analysis`, `UnifiedAnalysisReport` | `hyso analysis list`, `show`, `plan`, `run` |
| Karşılaştırma | `hyso.analysis.ModelComparisonSource`, `compare_models` | `hyso compare` |
| Advisor | `hyso.advisor.advise`, `recommend` | `hyso advisor recommend` |
| Raporlama | `hyso.reporting.create_report`, `report_experiment` | `hyso reports render`, `experiment` |
| Deney kayıtları | `ExperimentManager`; `list_experiments`, `get_experiment_history`, `record_result`, `open_result` | `hyso experiments list`, `history`, `run`, `artifacts`, `verify`, `open` |

Analyzer/Comparison/Advisor/Reporting başlıklarının mevcut public girişleri bu
işlevler ve namespace'lerdir. Root convenience import'ları ve eski modül yolları
korunur; yeni bir eğitim veya analiz algoritması eklenmez.

## Model oluşturma ve eğitim hazırlama

```python
from hyso import ModelBuilder, TrainingConfig

builder = ModelBuilder((1, 4, 4), 2).add("conv2d", in_channels=1, out_channels=2)
model = builder.build()                  # Yeni, eğitilmemiş CPU modeli.
config = TrainingConfig(epochs=1, batch_size=2)
```

`dataset`, kullanıcının verdiği mevcut, sonlu PyTorch Dataset'idir. Eski yol aynen
geçerlidir:

```python
train_loader, val_loader = config.make_loaders(dataset)
session = config.prepare(model, train_loader, val_loader)
```

Yeni kısa yol aynı loader ayarlarını ve aynı hazırlama kontrollerini kullanır:

```python
session = config.prepare_datasets(model, dataset)
print(session.settings()["state"])       # prepared; eğitim başlamadı.
# result = session.fit()                 # Yalnız kullanıcı eğitim istediğinde.
```

Validation için açıkça `TrainingConfig(validation=True, ...)` seçip
`prepare_datasets(model, train_dataset, val_dataset)` kullanın. Ayarlardan farklı
bir validation verisi sessizce kabul edilmez. Kısayol `model_config`, `loss_fn`,
`loss_weights`, `on_epoch` seçeneklerini mevcut `prepare` sözleşmesine geçirir;
desteklenmeyen Graph/checkpoint/loss kombinasyonları yine reddedilir. Özel
DataLoader'lar için eski `prepare(model, train_loader, val_loader)` yolunu kullanın.
Kısayol modelin device/dtype'ını, seed'i veya config varsayılanlarını değiştirmez.

Graph için mevcut `GraphModelBuilder(...).add(...).outputs(...).build()` yolunu
ve eğitimde [açık çıktı seçimini](python-first-workflow.md) kullanın. Session
hazırlama ham Graph/topoloji/task kombinasyonlarına yeni destek sağlamaz.

## CLI ile açık önizleme

```text
hyso models summary model.json
hyso models inspect model.json --dry-run
hyso train validate training.json
hyso train run training.json --dry-run
hyso analysis plan --features P01 P02
hyso analysis run --features P01 --dry-run
```

`--dry-run`, `models inspect`, `train run`, `analysis run` için açık önizlemedir;
`--execute` ile aynı komutta kullanılamaz. Factory import edilmez ve workload
başlatılmaz. Açık `--output` istenirse önizleme raporu dosyaya yazılabilir.

`train run --dry-run`, geçerli ayarları yürütme izni istemeden gösterir; sonuçtaki
`blockers`, gerçek yürütmede hâlâ gereken factory/izinleri listeler.
`execution_authorized=false` kalır. Eski bayraksız `train run` davranışı değişmez.
`models inspect --dry-run` model oluşturmadan mimariyi doğrular. Analiz ise eksik
kaynak/kanıt veya desteklenmeyen seçim nedeniyle dry-run'da da blocked/unsupported
dönebilir; önizleme bunları başarıya dönüştürmez.

Gerçek CLI eğitimi, kurulu ve güvenilen
`factory(config: TrainingConfig, **kwargs) -> hazırlanmış TrainingSession`
işlevi ister. Örneğin:

```text
hyso train run training.json --factory my_project:session_factory --execute --trust-factory --allow-training
```

Bu komut eğitim talebidir. Checkpoint yazımı için `--allow-file-writes`, açık CUDA
device için ayrıca `--allow-cuda` gerekir. Bu bayraklar herhangi bir GPU profilinin
kabul edildiğini kanıtlamaz. Factory'nin kendisi `fit()` çağırmamalıdır.

## Kayıtlı analizden Advisor, karşılaştırma ve rapor

```python
from hyso import analysis, advisor, reporting

source = analysis.UnifiedAnalysisReport.from_json("analysis.json")
advice = advisor.recommend(source)
text = advice.to_markdown(language="tr", level="beginner")
report = reporting.create_report(source, language="tr", level="beginner")
html = report.to_html()
```

İki kayıt için `ModelComparisonSource("A", report_a)` ve
`ModelComparisonSource("B", report_b)` oluşturup `analysis.compare_models(...,
features=("P01", "P02"))` kullanın. Etiketler kanıtla eşleşmelidir; eksik kanıt
ölçüme veya kazanana çevrilmez. Strict karşılaştırma gerekiyorsa mevcut
`context_policy="strict"` ve P26 bağlam gereksinimleri geçerlidir.

```text
hyso compare a.json b.json --baseline-model A --candidate-model B --features P01 P02
hyso advisor recommend analysis.json --language tr --format markdown
hyso reports render result.json --format html --output report.html
hyso experiments list --root C:/local-hyso/store
```

Bunlar kayıtlı sonuçları okur; model kodunu değiştirmez, eğitim/capture başlatmaz.
CLI compare, kayıtlı UnifiedAnalysisReport/AnalysisExecutionResult girdilerini
kabul eder; Pythonun bütün source türlerine eşit değildir. EvidenceIndex ve
execution raporlamasında gerektiğinde `--context` verin. Ham multi-head evaluation
JSON için Python reader/ExperimentManager kullanın; genel CLI reader yeni destek
kazanmadı. Python `ExperimentManager(root)` yeni store oluşturabilir; CLI yalnız
mevcut store'u açar. Export varsayılan olarak var olan dosyayı ezmez.

## Hata nedeni ve çözüm önerisi

Bilinen TensorDataset/model uyuşmazlığında hata artık loader adını, beklenen ve
verilen örnek şeklini, batch ekseninin hariç tutulduğunu açıklar. ModularCNN için
yalnız bilinen CHW/kanal sözleşmesi belirtilir; bilinmeyen H/W değeri uydurulmaz.
Python `TrainingConfigError.suggestion` çözüm önerisini ayrı taşır. CLI'de hata
metni ve `Suggestion:` ayrı satırlardadır; varsayılan traceback gösterilmez.
`--debug` veya mevcut `--verbose` teknik traceback'i stderr'e ekler. Tanınmayan
bir RuntimeError şekil, CUDA veya veri hatası olarak tahmin edilmez.

`--json` için stdout tek `hyso.cli` schema1 envelope'udur; stderr tanı içindir.
Mevcut alan adları, hata kodları ve ModelConfigError diagnostics yapısı korunur.
İnsan tarafından okunan hata cümlelerini makine protokolü gibi ayrıştırmayın.

## v0.2.0 uyumluluğu ve kabul sınırları

| Alan | Mevcut sözleşme / kanıt |
|---|---|
| Paket | HYSo 0.2.0; metadata Python >=3.10, Torch >=2.2.2, torchvision >=0.17.2 |
| Somut CPU beta kanıtı | [13.5.22 raporu](../reports/stage13-5-22-cpu-beta-readiness.md): Windows x64, Python 3.10.18, Torch 2.5.1+cu121 / torchvision 0.20.1+cu121 üzerinde bounded CPU akışları |
| CPU kapsamı | Belgelenmiş Builder/Graph classification, açık named-input/loss profilleri, strict v2 resume, retained analiz/karşılaştırma/Advisor/rapor ve store roundtrip |
| GPU | B4 ROOT_CAUSE_UNRESOLVED; CPU beta kabulü ve CUDA cihaz seçeneği GPU production kabulü değildir |
| Şemalar | TrainingConfig schema1, eski model konfigürasyonları, checkpoint ve rapor reader'ları aynı; otomatik migration yok |

Metadata alt sınırları bütün Python/Torch/işletim sistemi kombinasyonlarının test
edildiği anlamına gelmez. Bu UX değişikliği yalnız ilgili CPU/API/CLI regresyonları
ile doğrulanır; yeni GPU kabulü veya yeni paket/runtime kombinasyonu iddia etmez.
UX-01 kısayolu güncellenmiş kaynak sürümdedir; önceki wheel/sdist artifact'ları bu
görevde yeniden üretilmedi. Kurulumunuzun kullandığınız kaynak sürümle eşleştiğini
kontrol edin.
Detaylı sınırlar: [training config](training-config.md),
[kabul rehberi](python-first-acceptance-and-limits.md).
