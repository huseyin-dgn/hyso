# Aşama 13-R ticari kaynak dağıtımı doğrulaması

Tarih 2026-11-12. Branch `feature/hyso-models-next`, başlangıç ve son HEAD
`9ef0205210bb926b51441f81a8f5e969b36f158a`.

Ticari HYSo kaynak profili oluşturuldu. Hazır aileler ve on geliştirme modülü
wheel/sdist payload'ından ayrıldı. Normal kurulu ticari paket 16 Registry girdisi
sunuyor. Source checkout ve editable kurulum eski kullanıcı modelleri ve M3/M4
testleri için geliştirme profilini koruyor. Tam ticari ürünün bağımlılık binary
dağıtımı ve hukuki uygunluğu için otomatik onay verilmez.

## Kaynak ve profil kararı

152 Python modülü pozitif `commercial-profile.json` izin listesinde.
`setup.py` build_py ve sdist yollarında aynı listeyi uygular. Geliştirme kaynakları
build fixture'ında fiziksel olarak mevcutken ticari arşivler üretildi. Böylece
yalnız eksik dosyalı yapay kaynak ağacı üzerinden doğrulama yapılmadı.

Ticari kapsam dışında ResNet, ResNeXt, DenseNet, PreActResNet, WideResNet,
MobileNetV2/V3, EfficientNetV2, GhostNet, ShuffleNetV2, RepVGG, MobileOne ve
önceki kararla FasterNet/PartialConv bulunur. M3 residual/attention adaptörleri,
M4 aileye özgü blokları ve fusion yardımcıları aynı geliştirme kapsamına ayrıldı.

Paketlerde `architectures`, `m3_residual`, `m3_attention`, `m4_blocks`,
`m4_config`, `m4_registry`, `m4_reparam`, `mobile_architectures`,
`reparam_architectures` ve `_development` modülleri yoktur. Installed testler
hem find_spec hem gerçek import üzerinden erişimin başarısız olduğunu doğruladı.
Runtime export'ları ve JSON kayıtları dışlanan modelleri açamıyor.
Ortam değişkeni kullanılması ticari profili değiştirmiyor.

## Korunan işlevler

15 ortak primitive/blok ve configurable ModularCNN, toplam 16 Registry girdisi
sunulur. Convolution, grouped/depthwise groups parametreleri, mevcut residual
projection, SE, CBAM, ECA, Coordinate Attention, BN/GroupNorm, ReLU/SiLU/HardSwish,
Identity, ChannelShuffle ve DropBlock kullanılabilir.

ChannelShuffle, DropBlock2D ve probability doğrulaması eski M4 dosyasından
`regularization.py` içine çıkarıldı. Üç AST gövdesi başlangıç HEAD ile aynı.
Yeni convolution veya model ailesi uygulanmadı.

Builder list/get/validate/create, add/build/modular ve schema1 JSON roundtrip
yolları korunur. Named hazır aileler ticari Registry'de hata üretir. Validation
ve summary metadata üzerinden çalışır. Kullanıcının kendi nn.Module modeli
TrainingSession'a verilebilir. EMA ve loader-plan eski import yollarındadır.
ELO/EBO, CEMO ve fit wrapper'ları repository geçmişi belgelenen HYSo configurable
modelleri olarak korunur. Sınıflandırmanın kanıt sınırı
[risk raporunda](stage13-commercial-risk-audit.md) açıklanır.

## Test sonuçları

**1.276 benzersiz pytest testi ve 129 alt test başarılı.** Failure/error sayısı
0, skipped sayısı 0. Tekrar alınan ilk paketleme sonuçları benzersiz sayıya
yeniden eklenmedi.

| Kanıt | Sonuç | Kapsam |
|---|---|---|
| development-models.xml | 678 passed | M3/M4 geliştirme API'leri, Builder, attention/modular, eski model/state regresyonları |
| api-regression.xml | 588 passed, 129 alt test, 7 deselected | CLI, TrainingConfig/Session, native execution, reporting, Manager, Advisor, comparison ve saved P01–P35 |
| commercial-packaging-final.xml | 10 passed, 2 deselected | Wheel/sdist içerikleri, kurulu API/CLI ve ticari bloklar |

Eski M3/M4 testleri silinmedi. `development_profile` marker'ıyla 256 eski vaka
ayrıca collection kontrolünden geçti. Yeni ticari dağıtım vakaları
`commercial_profile` marker'ı taşır. Diğer mevcut ortak regresyonlar 678 ve 588
testlik gruplarda yer alır.

Yedi workload/worker/README örneği ve iki uzun installed native smoke vakası
seçim dışındadır. CUDA benchmark veya yeni donanım kabulü çalıştırılmadı.
P01–P35 saved native okuma testleri mevcut kanıtları yeniden okur.

Test ortamı Windows, Python 3.10.18, torch 2.5.1+cu121, torchvision 0.20.1+cu121.
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`CUDA_VISIBLE_DEVICES=-1`, `MPLBACKEND=Agg` kullanıldı. Test giriş noktası
`python -B -m pytest -c tests/pytest.ini -q -p no:cacheprovider --tb=short`.
Mevcut GradScaler deprecation uyarısı sürer.

## Kurulu wheel ve sdist doğrulaması

Her arşiv ayrı target'a offline/no-deps kuruldu ve checkout dışındaki isolated
Python süreçlerinde çalıştırıldı. Import kaynağının kurulu target olduğu kontrol
edildi. Commercial Registry tam olarak 16 girdiyi içerir.

İki dağıtımın her birinde 15 blok float32 ve float64 ile çalıştırıldı. Shape,
finite output/gradient ve weights-only strict state_dict roundtrip geçti.
Toplam 60 operator/dtype kontrolü iki pytest smoke vakasının içindedir ve
benzersiz pytest sayısına ayrı testler olarak eklenmedi.

Custom Builder zinciri üzerinde iki sentetik örnekle bir SGD optimizer adımı
tamamlandı. JSON ve state_dict ile klon çıktıları eşleşti. Eski EMA ve
loader-plan girişleri çalıştı. CLI list/show/validate/summary/inspect ve gerçek
console / python -m hyso giriş noktaları kontrol edildi. Son durumda CUDA
context başlatılmadığı doğrulandı.

Bütün paket Python dosyaları güncel kaynaklarla byte düzeyinde karşılaştırıldı.
Arşivler açıldı ve dışlanan class implementasyonları AST üzerinden de arandı.
Wheel 152 HYSo Python dosyası içerir. Sdist aynı 152 dosyaya ek olarak build
destek kaynağı setup.py içerir. Dışlanan hazır mimari, test, örnek, checkpoint, pretrained
ağırlık, veri kümesi ve forensic kaynakları ticari payload'a eklenmedi.
THIRD_PARTY_NOTICES.md hem wheel metadata'sında hem sdist içinde byte düzeyinde
doğrulandı. Python 3.10 AST denetimi bütün 162 geliştirme modülünü kapsadı.

| Arşiv | Boyut | SHA256 |
|---|---|---|
| hyso-0.2.0-py3-none-any.whl | 681728 byte | `a61b331b179ae7fe11055c50f59656bf4be21ffdf220080474fb6dd7bed024ef` |
| hyso-0.2.0.tar.gz | 598751 byte | `75657619c7d0196237bc7ee32c0417e3f8bc3d9e25cc12f01d5b3dc972110477` |

Arşivler `%LOCALAPPDATA%/hyso-stage13r-20261112/dist/` altındadır.
`distribution-final.json` tam archive member listelerini saklar.

## Lisans ve kapsam sınırları

98 model/helper sembolü, 32 runtime distribution ve dört ek optional deployment
distribution için envanter kaydedildi. Her model/blok kaynağı, lisans,
tutma/çıkarma gerekçesi, kalan belirsizlik ve Builder durumu risk raporundadır.

Ticari kaynak payload'ının dışlanmış kodları içermediği doğrulanmıştır. Ayrı
kurulan bağımlılıklardaki karma binary lisansları ve ürün yeniden dağıtım
yükümlülükleri ayrıca değerlendirilir. PyTorch, NumPy, SciPy, Pillow, tqdm ve
torchsummary toplam dağıtım statüleri için UNRESOLVED sınırları kaydedilmiştir.
Bu paketler HYSo arşivlerine vendored edilmez. Yeni bağımlılık eklenmedi.
Patentsiz kullanım, marka izni ve pretrained/data hakları NOT_EVALUATED.

## Bütünlük ve Git durumu

Başlangıçta kayıtlı 623 tracked dosya SHA256 ile karşılaştırıldı. Görev kapsamında
13 mevcut dosya değişti. Önceki 47 forensic/debug/ilgisiz dosyanın SHA256 değerleri
47/47 eşleşti. Üç checkpoint ve üç harici kabul/index kaydı 6/6 eşleşti.
Native performance/benchmark ve Trainer/resume/checkpoint kaynaklarından oluşan
71 dosyanın tamamı başlangıçla eşleşti. Mevcut 70/70 kabul kanıtı korundu.

HEAD `9ef0205210bb926b51441f81a8f5e969b36f158a`, staged alan boş.
Görev dosyaları çalışma ağacında kullanıcı incelemesine hazırdır.
Aşama 13.5 için yalnız
[girdi envanteri](../guides/stage13-5-builder-input-inventory.md) hazırlanmıştır.
Yeni geniş kapsamlı Builder API geliştirmesine geçilmedi.

Kanıt dizini `%LOCALAPPDATA%/hyso-stage13r-20261112/` altında protection,
component-classification, dependency-inventory, optional-dependency-inventory,
test-summary-final, distribution-final ve integrity-final JSON kayıtlarını içerir.

## Commercial Python payload

Each row below is present in both wheel and sdist with bytes matching the checkout. The sdist additionally contains setup.py as build support. The wheel includes no setup.py.

| Python source | Wheel | Sdist |
|---|---|---|
| `hyso/__init__.py` | included | included |
| `hyso/__main__.py` | included | included |
| `hyso/_architecture_coverage.py` | included | included |
| `hyso/_architecture_rules.py` | included | included |
| `hyso/_evaluation_evidence.py` | included | included |
| `hyso/_evaluation_structure.py` | included | included |
| `hyso/_quality_memory.py` | included | included |
| `hyso/_structure_native.py` | included | included |
| `hyso/advisor/__init__.py` | included | included |
| `hyso/advisor/_json.py` | included | included |
| `hyso/advisor/_locales.py` | included | included |
| `hyso/advisor/_p26_comparison.py` | included | included |
| `hyso/advisor/_recommendation_text.py` | included | included |
| `hyso/advisor/comparison.py` | included | included |
| `hyso/advisor/contracts.py` | included | included |
| `hyso/advisor/engine.py` | included | included |
| `hyso/advisor/goals.py` | included | included |
| `hyso/advisor/model_comparison_presentation.py` | included | included |
| `hyso/advisor/performance.py` | included | included |
| `hyso/advisor/planning.py` | included | included |
| `hyso/advisor/presentation.py` | included | included |
| `hyso/advisor/profiles.py` | included | included |
| `hyso/advisor/recommendations.py` | included | included |
| `hyso/advisor/render.py` | included | included |
| `hyso/advisor/resolver.py` | included | included |
| `hyso/advisor/validation.py` | included | included |
| `hyso/analysis/__init__.py` | included | included |
| `hyso/analysis/_analysis_resources.py` | included | included |
| `hyso/analysis/_codec.py` | included | included |
| `hyso/analysis/_isolated_worker.py` | included | included |
| `hyso/analysis/_native.py` | included | included |
| `hyso/analysis/_windows_job.py` | included | included |
| `hyso/analysis/catalog.py` | included | included |
| `hyso/analysis/evidence.py` | included | included |
| `hyso/analysis/execution.py` | included | included |
| `hyso/analysis/feature_plan.py` | included | included |
| `hyso/analysis/isolated_bridge.py` | included | included |
| `hyso/analysis/model_comparison.py` | included | included |
| `hyso/analysis/native_adapters.py` | included | included |
| `hyso/analysis/native_bridge.py` | included | included |
| `hyso/analysis/native_execution.py` | included | included |
| `hyso/analysis/report.py` | included | included |
| `hyso/analysis/sources.py` | included | included |
| `hyso/architecture_audit.py` | included | included |
| `hyso/augmentation/__init__.py` | included | included |
| `hyso/augmentation/augmentation.py` | included | included |
| `hyso/benchmark/__init__.py` | included | included |
| `hyso/benchmark/result.py` | included | included |
| `hyso/benchmark/runner.py` | included | included |
| `hyso/callbacks/__init__.py` | included | included |
| `hyso/callbacks/callbacks.py` | included | included |
| `hyso/checkpoint.py` | included | included |
| `hyso/cli/__init__.py` | included | included |
| `hyso/cli/analysis.py` | included | included |
| `hyso/cli/common.py` | included | included |
| `hyso/cli/experiments.py` | included | included |
| `hyso/cli/main.py` | included | included |
| `hyso/cli/models.py` | included | included |
| `hyso/cli/presentation.py` | included | included |
| `hyso/cli/train.py` | included | included |
| `hyso/evaluation.py` | included | included |
| `hyso/experiments/__init__.py` | included | included |
| `hyso/experiments/manager.py` | included | included |
| `hyso/experiments/workflow.py` | included | included |
| `hyso/models/WithSE.py` | included | included |
| `hyso/models/__init__.py` | included | included |
| `hyso/models/_profile.py` | included | included |
| `hyso/models/attention.py` | included | included |
| `hyso/models/bcam.py` | included | included |
| `hyso/models/blocks.py` | included | included |
| `hyso/models/builder.py` | included | included |
| `hyso/models/fit.py` | included | included |
| `hyso/models/modular.py` | included | included |
| `hyso/models/registry.py` | included | included |
| `hyso/models/regularization.py` | included | included |
| `hyso/performance/__init__.py` | included | included |
| `hyso/performance/_batch_worker.py` | included | included |
| `hyso/performance/_cold_worker.py` | included | included |
| `hyso/performance/_cpu_layer_memory.py` | included | included |
| `hyso/performance/_device_diagnostics.py` | included | included |
| `hyso/performance/_energy_diagnostics.py` | included | included |
| `hyso/performance/_layer_allocator.py` | included | included |
| `hyso/performance/_layer_memory.py` | included | included |
| `hyso/performance/_loader_dataset_trace.py` | included | included |
| `hyso/performance/_loader_diagnostics.py` | included | included |
| `hyso/performance/_loader_event_channel.py` | included | included |
| `hyso/performance/_loader_stage_trace.py` | included | included |
| `hyso/performance/_loader_task_iterator.py` | included | included |
| `hyso/performance/_loader_task_trace.py` | included | included |
| `hyso/performance/_loader_trace_groups.py` | included | included |
| `hyso/performance/_loader_trace_validation.py` | included | included |
| `hyso/performance/_loader_worker_lifecycle.py` | included | included |
| `hyso/performance/_loader_worker_trace.py` | included | included |
| `hyso/performance/_memory_trend_diagnostics.py` | included | included |
| `hyso/performance/_module_activity.py` | included | included |
| `hyso/performance/_operation_models.py` | included | included |
| `hyso/performance/_operator_activity.py` | included | included |
| `hyso/performance/_profile_evidence.py` | included | included |
| `hyso/performance/_profile_worker.py` | included | included |
| `hyso/performance/_provider_diagnostics.py` | included | included |
| `hyso/performance/_runtime_worker.py` | included | included |
| `hyso/performance/_saved_activation.py` | included | included |
| `hyso/performance/_stability_diagnostics.py` | included | included |
| `hyso/performance/_stability_windows.py` | included | included |
| `hyso/performance/_stream_diagnostics.py` | included | included |
| `hyso/performance/_stream_resources.py` | included | included |
| `hyso/performance/_telemetry.py` | included | included |
| `hyso/performance/_training_memory_worker.py` | included | included |
| `hyso/performance/_training_process.py` | included | included |
| `hyso/performance/_transfer_activity.py` | included | included |
| `hyso/performance/_transfer_worker.py` | included | included |
| `hyso/performance/_windows_camera.py` | included | included |
| `hyso/performance/capacity.py` | included | included |
| `hyso/performance/cold.py` | included | included |
| `hyso/performance/dataloader.py` | included | included |
| `hyso/performance/devices.py` | included | included |
| `hyso/performance/energy.py` | included | included |
| `hyso/performance/gradients.py` | included | included |
| `hyso/performance/inference.py` | included | included |
| `hyso/performance/isolated_training_memory.py` | included | included |
| `hyso/performance/layer_overhead.py` | included | included |
| `hyso/performance/memory.py` | included | included |
| `hyso/performance/memory_trend.py` | included | included |
| `hyso/performance/pipeline.py` | included | included |
| `hyso/performance/profiling.py` | included | included |
| `hyso/performance/providers.py` | included | included |
| `hyso/performance/result.py` | included | included |
| `hyso/performance/roofline.py` | included | included |
| `hyso/performance/roofline_counters.py` | included | included |
| `hyso/performance/runtime_result.py` | included | included |
| `hyso/performance/stability.py` | included | included |
| `hyso/performance/streaming.py` | included | included |
| `hyso/performance/sweeps.py` | included | included |
| `hyso/performance/training.py` | included | included |
| `hyso/performance/training_details.py` | included | included |
| `hyso/performance/training_memory.py` | included | included |
| `hyso/performance/transfers.py` | included | included |
| `hyso/performance/utilization.py` | included | included |
| `hyso/performance/worker_task_result.py` | included | included |
| `hyso/performance/worker_trace_result.py` | included | included |
| `hyso/quality.py` | included | included |
| `hyso/reporting/__init__.py` | included | included |
| `hyso/reporting/catalog.py` | included | included |
| `hyso/reporting/render.py` | included | included |
| `hyso/reporting/report.py` | included | included |
| `hyso/structure.py` | included | included |
| `hyso/structure_comparison.py` | included | included |
| `hyso/training/__init__.py` | included | included |
| `hyso/training/config.py` | included | included |
| `hyso/training/resume.py` | included | included |
| `hyso/training/session.py` | included | included |
| `hyso/training/trainer.py` | included | included |
| `setup.py` | excluded | build support |

## Task file scope

The task changes 22 files, including the three requested reports/guides. The original 47 protected dirty files remain outside this scope.

- [MANIFEST.in](../../MANIFEST.in)
- [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
- [commercial-profile.json](../../commercial-profile.json)
- [docs/guides/stage13-5-builder-input-inventory.md](../../docs/guides/stage13-5-builder-input-inventory.md)
- [docs/reports/stage13-commercial-distribution-validation.md](../../docs/reports/stage13-commercial-distribution-validation.md)
- [docs/reports/stage13-commercial-risk-audit.md](../../docs/reports/stage13-commercial-risk-audit.md)
- [hyso/models/__init__.py](../../hyso/models/__init__.py)
- [hyso/models/_development.py](../../hyso/models/_development.py)
- [hyso/models/_profile.py](../../hyso/models/_profile.py)
- [hyso/models/blocks.py](../../hyso/models/blocks.py)
- [hyso/models/builder.py](../../hyso/models/builder.py)
- [hyso/models/m4_blocks.py](../../hyso/models/m4_blocks.py)
- [hyso/models/registry.py](../../hyso/models/registry.py)
- [hyso/models/regularization.py](../../hyso/models/regularization.py)
- [setup.py](../../setup.py)
- [tests/pytest.ini](../../tests/pytest.ini)
- [tests/test_commercial_distribution.py](../../tests/test_commercial_distribution.py)
- [tests/test_m3_architectures.py](../../tests/test_m3_architectures.py)
- [tests/test_m4_architectures.py](../../tests/test_m4_architectures.py)
- [tests/test_m4_blocks.py](../../tests/test_m4_blocks.py)
- [tests/test_m4_examples.py](../../tests/test_m4_examples.py)
- [tests/test_packaging.py](../../tests/test_packaging.py)
