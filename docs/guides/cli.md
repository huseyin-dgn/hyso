# HYSo terminal arayüzü

Python 3.10+ ve mevcut HYSo bağımlılıkları yeterlidir; CLI `argparse` kullanır.
Uyumlu Torch/torchvision kurulumundan sonra repo kökünde
`python -m pip install -e .` çalıştırın. Wheel kurulumu da aynı `hyso` girişini
sağlar. `hyso --help` ve `python -m hyso --help` eşdeğerdir. Yardım ve CLI
import'u model oluşturmaz, CUDA sorgulamaz, eğitim veya benchmark başlatmaz.
Paketin mevcut üst düzey import'u Torch'u yükleyebilir.

## Komut ağacı

```text
hyso [--json] [--debug | --verbose] [--version]
  models      list | show NAME | validate FILE | summary FILE | inspect FILE
  train       options | validate CONFIG | run CONFIG
  analysis    list | show Pxx | plan | run
  experiments list | history ID | run ID | artifacts | verify ID | open ID
  compare     BASELINE CANDIDATE
  advisor     recommend FILE
  reports     render FILE | experiment ID
```

Her yaprak komutta `--help`, `--json`, `--debug`, `--verbose`, `--output FILE`
ve `--overwrite` bulunur. Tanılama seçenekleri komut grubundan önce veya sonra
verilebilir. JSON stdout'a tanılama/progress yazılmaz; bunlar stderr'e yönlenir.
Factory'nin OS seviyesinde doğrudan stdout'a yazması veya alt süreç açması
güvenilen uygulama kodunun sorumluluğundadır.

## Modeller ve eğitim

```console
hyso models list --json
hyso models show conv2d
hyso models validate examples/cli/architecture.json
hyso models summary examples/cli/architecture.json
hyso models inspect examples/cli/architecture.json --execute
hyso train options
hyso train validate examples/cli/training.json
hyso train run examples/cli/training.json --factory my_app.resources:training
```

`validate` ve `summary` yalnızca ModelBuilder sözleşmesini doğrular ve normalize
edilmiş mimariyi sunar. Mimari JSON ağırlık içermez. `inspect --execute` yeni,
eğitilmemiş CPU modeli kurar; parametre sayısı ve parametre/buffer shape'lerini
okur. Forward, aktivasyon shape profili ve checkpoint yükleme yapmaz.

TrainingConfig tam schema1 dosyasıdır; `train options --json` varsayılan tam
şemayı gösterir. Optimizer `sgd/adam/adamw`, scheduler
`null/step/cosine/onecycle/plateau`, precision `fp32/fp64/amp_fp16`, device
`cpu/cuda:INDEX` seçenekleri mevcut API doğrulamasından geçer. CPU AMP ve
`num_workers > 0` bu profilin kapsamı dışındadır.

Eğitim factory sözleşmesi:

```python
def training(config, **kwargs):
    # Uygulamanızın gerçek modelini, ağırlıklarını ve gerçek veri kaynaklarını
    # yükleyin. Modeli config'in device/dtype sözleşmesine göre yerleştirin.
    model, train_loader, val_loader = load_application_resources(**kwargs)
    return config.prepare(model, train_loader, val_loader)
```

Bu örnekte `load_application_resources` kullanıcı uygulamasına aittir; CLI demo
verisiyle gerçek kullanıcı eğitimi yapmaz. Factory import edilebilir kurulu bir
modülde olmalıdır. `--factory-args args.json` plain JSON keyword nesnesidir;
JSON içinden class/import/eval/pickle çözümlemesi yapılmaz.

```console
hyso train run training.json --factory my_app.resources:training --factory-args data.json --execute --allow-training --trust-factory --output training-result.json
```

Dry-run factory import etmez, loader oluşturmaz/iterate etmez ve checkpoint
okumaz. Canlı çalışmada CLI factory'nin döndürdüğü prepared TrainingSession'ın
ayarlarını incelenen config ile birebir karşılaştırır ve `fit()`i bir kez çağırır.
CUDA için ayrıca `--allow-cuda`, `save_last_to` için `--allow-file-writes`
gerekir. Mevcut checkpoint hedefi reddedilir; CLI `--overwrite` yalnızca rapor
export'una uygulanır. Resume Trainer'ın mevcut doğrulamasına bırakılır; hata
sonrası otomatik retry yoktur. Eğitim/natif checkpoint sözleşmeleri değişmez.

## Analiz: seçim, preflight ve açık execution

```console
hyso analysis list --json
hyso analysis show P23
hyso analysis plan --features P01 P02 P11
hyso analysis plan --profile full --device cuda:0
hyso analysis run --features P01 P02 --factory my_app.resources:bindings
```

`plan` saf seçimdir ve runtime hazır olma iddiası taşımaz. `run` varsayılan
preflight'tır. Her satır native destek, blocker, dependency, maliyet sınıfı ve
kapsam içerir; plan ayrıca CPUAnalysisPolicy bütçesini taşır. `--policy` mevcut
CPUAnalysisPolicy keyword JSON'unu alır. Native protokol bütçeleri typed binding
içindedir; metadata resource reservation veya keyfi Python süresi/RAM sınırı
değildir. Factory dry-run'da yüklenmediği için canlı kaynak satırları
`MISSING_PREREQUISITE` kalır; gerçek binding preflight'ı execution sırasında,
factory hazırlığından sonra ve producer çağrısından önce yapılır.

`--models contexts.json`, ModelContext keyword nesnelerinden oluşan nonempty
bir JSON array'i kabul eder; aksi halde `--model-id model` etiketi kullanılır.
Task/backend/capability beyanları ölçülmüş gerçek veya cihaz yetkisi değildir.

Canlı analiz factory sözleşmesi `factory(**kwargs) -> list[NativeBinding]`dir.
Mevcut `InferenceSource`, `PipelineSource`, `TrainingSource.from_session`,
`RecipeSource`, `RuntimeSource`, `ProtocolSource`, `EvaluationSource`,
`QualitySource`, `RooflineSource` ve `RetainedSource` tipleri aynen kullanılır.
Gerçek model, dataset, Trainer, mevcut ağırlık/ONNX ve protokolleri uygulama
hazırlar; factory kendi içinde forward/fit/capture başlatmamalıdır.
`examples/cli/bindings.py` yalnızca mevcut retained JSON'a bağlanan çalışan
factory örneğidir. CLI bütün factory policy'lerini kendi açık izinleriyle
değiştirir; kaynak factory kendine izin veremez.

```console
hyso analysis run --features P01 P02 --factory my_app.resources:bindings --execute --authorize P01 P02 --trust-factory --allow-capture --output execution.json
```

`--execute` ve ayrı `--authorize` zorunludur. Yetkili ID'ler açık seçimin benzersiz
alt kümesi olmalıdır. Seçim, profil ve dependency otomatik iş başlatmaz.
`--allow-capture`, `--allow-training`, `--allow-cuda`, `--allow-camera`,
`--allow-power-sampling`, `--allow-long-run`, `--allow-file-writes` ayrıdır.
`--allow-peak-reset` yalnızca bu CLI sürecinin allocator'ına reset izni verir ve
mevcut PID'yi native policy'ye taşır. `--peak-reset-pid PID` aynı sahipliği açık
PID ile belirtmek içindir; native admission başka bir sürecin PID'sini kabul
etmez. İki seçenek birbirini dışlar; varsayılan reset yoktur.
Native executor narrowed authorization, ownership, receipt ve CUDA saved
activation guard'larını korur. Kabul durumu `NOT_EVALUATED` kalır.

Factory import'u keyfi güvenilen Python çalıştırabilir; bir sandbox değildir.
CLI bu yüzden import öncesi capture iznini, seçilmiş P19–P25 için training,
CUDA device için CUDA, P30 için camera, P31 için long-run iznini konservatif
olarak ister. Kayıtlı video için de P30 factory sınırı camera bayrağını ister.
Native admission protokol ve sensor/file-write izinlerini ayrıca denetler.
Factory hazırlığı cihazları kendiliğinden sorgulamamalı, native recipe'deki
deferred factory'ler mümkün olduğunca korunmalıdır.

### Retained ve offline derive

PowerShell hash: `(Get-FileHash -Algorithm SHA256 source.json).Hash.ToLower()`.
Linux hash: `sha256sum source.json`. İncelediğiniz hash'i sabit girdi olarak
verin; CLI sessizce yeni bir expected hash üretmez.

```console
hyso analysis run --features P22 --source p19.json --sha256 EXPECTED_SHA256
hyso analysis run --features P22 --source p19.json --sha256 EXPECTED_SHA256 --execute --authorize P22 --output execution.json
hyso analysis run --features P32 P33 --source p31.json --sha256 EXPECTED_SHA256 --execute --authorize P32 P33 --source-options derive.json
```

`derive.json` örneği: `{"P32":{"warmup_seconds":0}}`. P22 mevcut P19'dan,
P32/P33 mevcut P31'den kapalı native derive yollarıyla üretilir; fit veya yeni
7200 saniye/sensor işi başlamaz. Asıl süre, sensor ve kapsam eksikleri korunur.
P21 ek kaynakları için `--companions refs.json` içeriği
`[{"path":"worker-trace.json","sha256":"..."}]` biçimindedir. Relative yollar
manifest dizinine göre çözülür. Ekler bağımsız hash taşır ve base evidence gibi
sunulmaz. Retained kaynak limiti 64 MiB, genel kayıtlı rapor limiti 128 MiB'dir.

## Deneyler, karşılaştırma, Advisor ve raporlar

```console
hyso experiments list --root STORE --json
hyso experiments history EXPERIMENT_ID --root STORE --limit 100
hyso experiments run RUN_ID --root STORE
hyso experiments artifacts --root STORE --run-id RUN_ID
hyso experiments verify ARTIFACT_ID --root STORE
hyso experiments open ARTIFACT_ID --root STORE --json
hyso compare baseline.json candidate.json --baseline-model A --candidate-model B --features P01 P02 P11 P26 --format markdown --output comparison.md
hyso advisor recommend analysis-report.json --language tr --level expert --format html --output advisor.html
hyso reports render training-result.json --language en --level beginner --format html --output training.html
hyso reports experiment EXPERIMENT_ID --root STORE --format html --output history.html
```

STORE mevcut dedicated local ExperimentManager dizinidir; synced/network yol
kuralları değişmez. Listeleme/history/run sayfalama cursor'larını korur;
`--after-seq` ile devam edilir. CLI store oluşturmaz veya migrate etmez.
Schema1 history okunabilir; artifact işlemleri schema2 gerektirir. `verify`
mevcut API üzerinden **verification observation kaydı ekler**; expected hash,
artifact veya run durumunu değiştirmez. `open` hash/schema/native reader
doğrulamasından geçmeden sonuç döndürmez. Silme/repair/migration komutu yoktur.

Compare execution veya UnifiedAnalysisReport dosyaları alır; ModelComparison
kuralları aynı kalır. Uygunsuz kapsam `INSUFFICIENT_EVIDENCE`/native comparability
bilgisiyle kalır, global winner üretilmez. Bu komut P29 paired canlı deneyini
çalıştırmaz. JSON ve Markdown mevcut Comparison API'sinden export edilir.

Advisor `recommend()` kullanır; giriş UnifiedAnalysisReport, EvidenceIndex veya
ModelComparisonResult'tır. EvidenceIndex için `--context context.json`
zorunludur: `{"workload_id":"study","task":"classification","model":{},"dataset":{}}`.
Context association caller-declared kalır. Yeni ölçüm, model optimizasyonu veya
kanıtsız sayısal kazanç üretilmez. JSON export RecommendationReport ham
sözleşmesidir; `--json` stdout görünümü seçilen dil/seviyedeki display modelidir.

Reports TrainingResult, UnifiedAnalysisReport, execution/EvidenceIndex (açık
context ile), ModelComparisonResult ve RecommendationReport destekler. Ham
AdvisorReport için orijinal UnifiedAnalysisReport `--advisor-source FILE` ile
verilmelidir; mevcut `validate_advisor_report` doğrulaması uygulanır. Tüm render
işleri `hyso.reporting` API'sini kullanır; `tr/en`, `beginner/technical/expert`,
`json/markdown/html` desteklenir.

## Çıktı ve hata sözleşmesi

`--json` stdout her sonuç/hata için tek JSON envelope üretir:

```json
{"format":"hyso.cli","schema_version":1,"status":"blocked","exit_code":3,"result":{"blockers":["--allow-training"]}}
```

Envelope dışındaki `--output` dosyası ilgili native/result/presentation API'nin
ham JSON/Markdown/HTML biçimidir ve yeniden okunabilir. `--format` dosya ve
insan stdout formatını seçer; `--json` varsa stdout her zaman JSON envelope'dur.
Başarılı render/compare exit 0, raporun üretildiğini ifade eder; kanıt
yeterliliği veya karşılaştırılabilirlik sertifikası değildir.

| Kod | Anlam |
| --- | --- |
| 0 | Başarılı metadata/rapor veya hazır dry-run; yetkili execution satırları tamamlandı |
| 1 | Beklenmeyen teknik hata |
| 2 | Eksik/yanlış argüman, dosya, JSON, schema veya output conflict |
| 3 | Eksik kaynak/izin veya doğrulanamayan artifact; sıfır başarılı execution |
| 4 | Desteklenmeyen runtime/schema operasyonu; sıfır başarılı execution |
| 5 | Execution/training hatası; sıfır başarılı execution |
| 6 | Başarılı ve başarısız/blocked/unsupported execution satırları birlikte |
| 130 | Kullanıcı kesintisi; native state ilerlemiş olabilir |

Native `SKIPPED` dependency ve yetkisiz satırlar başarı sayılmaz. Yalnızca
başarısız satırları olan bir çalışma başarılı sunulmaz. `--debug` veya
`--verbose` traceback'i stderr'e açar. Varsayılan hata eksik girdiyi ve
`--help`/`--debug` düzeltme yolunu söyler.

Output dizini önceden var olmalıdır. Var olan dosya varsayılan olarak
reddedilir; `--overwrite` açıkça verilmelidir. Input dosyasının yerine export,
symlink/dizin hedefi ve Manager store içine export reddedilir. Export mevcut
atomic/no-clobber yazıcıyla yapılır. Export ön kontrolü iş başlamadan uygulanır;
sonradan oluşan dosya yarışlarında atomic no-clobber son kararı verir. Tamamlanan
native iş, sonradan export başarısız olursa geri alınmaz; retry kendiliğinden
yapılmaz. Windows/Linux için `pathlib` kullanılır; boşluklu yolları shell'de
tırnaklayın, JSON Windows yollarında `\\` veya `/` kullanın.

Bu teslim Windows ve Python 3.10.18 üzerinde doğrulanmıştır; Linux üzerinde
ayrı bir kurulum çalıştırılmamıştır. CLI path işlemleri `pathlib` kullanır;
native owned-worker akışları mevcut Windows/platform sınırlarını aynen korur.
CLI ek bir Linux worker backend'i veya otomatik cihaz fallback'i eklemez.

Çalışan offline senaryolar ve factory Python örneği:
[examples/cli](../../examples/cli/README.md).
