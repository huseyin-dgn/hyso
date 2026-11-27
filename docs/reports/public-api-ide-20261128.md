# HYSo 0.2.0 — Final Public API Documentation & IDE Experience

2026-11-28 / Europe-Istanbul. Base: `feature/hyso-models-next`, `6b6001e`. Version **0.2.0** korunmuştur. Yeni commit, push, tag veya yayın yapılmamıştır.

## Nihai kapsam ve sayılar

1011 public tanım incelendi: 288 class, 137 module function, 478 method/property/call ve 108 explicit constructor. Alias/re-export tekrarları ve 173 statik overload bu sayıya tekrar eklenmedi.

2843 parametre/record-field girdisi gözden geçirildi; 607 açıklama eklendi, 736 açıklama düzeltildi/genişletildi: toplam **1343**. Eksik 105 docstring tamamlandı. Son envanterde eksik docstring veya açıklanmamış açık parametre **0**. Sayımlar `counts.json`, tam envanter `after.json` içindedir.

Kapsam, commercial profile içindeki 190 Python modülünün kullanıcıya açık tanımlarını ve public namespace üzerinden sunulan private-module re-exportlarını içerir. Public protocol/config/result dataclass alanları da sayılmıştır. CLI/worker/admission iç yardımcıları ve paketlenmeyen development aileleri kapsam dışıdır. Inherited PyTorch API’leri yeniden yazılmadı; upstream yardım ve tipleri korunmuştur. Parametre varlığı kontrolleri, her olası kullanıcı girdisinin dinamik kabul testi anlamına gelmez.

Google docstring standardı kullanıldı: summary, **Args (parameters)**, Returns, Raises, Notes ve gerektiğinde Examples. İngilizce açıklamalar signature/default/validation ile eşleştirildi. Mevcut dört dilli offline guide, API topic kayıtları ve rapor yerelleştirmeleri değiştirilmedi.

## Builder katman ve parametre kapsamı

[Tam parametre referansı](../guides/public-api-parameters.md): **57 registry bileşeni / 222 parametre girdisi**. Convolution, attention, activation, normalization, pooling, regularization, residual, resize/pixel işlemleri ve terminal head’ler dahil. Her satır tip, required/default ve native range/cross-parameter koşullarını açıklar.

ModelBuilder `.add` için **53** non-model/non-head bileşen overload’ı; registry `.create` için **57** overload. Schema2 sequential factory için classification/regression/embedding **3** overload vardır. Whole-model ModularCNN ayrı named/modular/modular_cnn yollarında kalır. Fluent convolution `in_channels=None` ile kanal çıkarımı yapabilir; generic add gerekli in_channels değerini açıkça ister.

GraphModelBuilder `.add` için **56** registry bileşeni + add/concat/multiply/weighted_sum **4** fusion overload’ı vardır. Girişler 1..8 named C/CHW shape veya shape/layout/dtype/device spec’idir. Node ID ve referanslar, channel/group eşleşmesi, fan-in/out, topology, batch-preserving shape kuralları ve geçersiz kullanımlar açıklanmıştır. Graph input-spec anahtarları TypedDict ile sunulur. GraphModel.forward dönüşü `dict[str, torch.Tensor]` olarak belgelenmiştir.

Convolution açıklamalarında kernel_size/stride/padding/dilation/groups/bias, transposed output_padding ve özel çok aşamalı blokların intermediate geometry kuralları vardır. ConvNormAct gibi bir API’de desteklenmeyen dilation parametresi eklenmemiştir. Standalone legacy constructor sınırları, Registry/Builder’ın 8192 profile sınırından ayrılmıştır. DropBlock runtime block-size clipping davranışı açıklanmıştır.

TrainingConfig optimizer/scheduler/early-stopping sözlükleri private TypedDict’lerle desteklenir; optimizer’a özgü izinli anahtarlar, native default’lar ve karşılıklı koşullar açıklanır. Callback mutable lifecycle ve checkpoint yazma davranışı, FP16/INT8/calibration kapsamı ve export validation aşamaları da tamamlanmıştır.

## Gerçek kurulu paket / IDE sonuçları

| Kontrol | Nihai sonuç |
|---|---|
| Wheel kurulumundan import/signature/docstring/origin | 1.011/1.011 PASS |
| Sdist kurulumundan import/signature/docstring/origin | 1.011/1.011 PASS |
| Python help(), pydoc, py.typed | İki kurulumda PASS |
| Tiny ModelBuilder / GraphModelBuilder örnekleri | İki kurulumda PASS |
| Gerçek VS Code/Pylance provider probe | **27/27 PASS** |
| Hover ve public import paths | Builder + 13 ek domain entry point PASS; kaynak URI’leri site-packages altında |
| Builder / Graph member autocomplete | PASS |
| Builder named/generic ve Graph generic keyword completion | ke → kernel_size PASS |
| Signature help ve parametre açıklaması | Gerçek uygun overload, default ve dedicated kernel_size documentation PASS |
| Return type inference | ModelBuilder ve GraphModel PASS |
| Son hedefli unit/packaging turu | 12/12 PASS |

Gerçek host sürümleri: VS Code 1.148.0, Python extension 2026.10.0, Pylance 2026.4.1. Yeni IDE veya büyük bağımlılık kurulmadı. Mevcut uzantılar, geçici VS Code extension test host içinde doğrudan çalıştırıldı; standalone Pylance kullanımı veya statik analizi gerçek IDE yanıtı gibi sunma yoktur.

Probe repository dışında, mevcut CPU release venv’inin interpreter/site-packages yoluna açıkça bağlandı. Hover/signature/completion yanıtları `ide-result.json` içinde saklandı ve final wheel SHA256 ile ilişkilendirildi. Geçersiz/eksik `ke` metni completion cursor fixture’ıdır; geçerli kernel_size=3 çağrısı signature ve parametre documentation kontrolüdür. Generic add çok sayıda overload döndürebilir; odaklı kullanım için named `.conv2d` gibi metotlar tercih edilebilir.

Bu yeni gerçek provider kanıtı önceki F01 NOT_TESTED snapshot’ını tamamlar; eski rapor değiştirilmedi. Fareyle manuel ekran/tooltip görsel QA yapılmadı (**NOT_TESTED**); burada PASS verilenler gerçek editor/language-server provider yanıtlarıdır. Python 3.11/3.12 ve başka OS/IDE için yeni sertifikasyon iddiası yoktur.

## Testler ve davranış güvenliği

İlk tur ve hedefli düzeltmelerin JUnit kayıtları testcase kimliğiyle birleştirildiğinde **626 farklı ilgili test PASS**. Tekrar çalıştırılan testler ikinci kez PASS toplamına eklenmedi. Builder, GraphBuilder, registry completion, smart/pythonic builders, TrainingConfig, contextual help, public doc coverage ve paketleme kontrolleri kapsandı.

İlk turlardaki hard-coded overload sayımları ve eski 70-topic assert’i düzeltildi; sonuncusu gerçek guide content kataloğuyla karşılaştırılır. Çok satırlı örnek okuma ve overload/implementation ayrımı için contextual-help testleri güncellendi. Eski başarısız XML/log kayıtları silinmedi. Son testler bu düzeltmeleri doğrular.

190 modül için executable AST karşılaştırması **PASS**: runtime parametre adları, positional/keyword düzeni, default ifadeleri ve algoritma gövdeleri değişmedi. Docstring, annotation, statik overload ve dört private TypedDict ayrıldı; ek import’lar yalnız reviewed type/mandatory-Torch imports. `behavior-contract.json` ayrıntıları içerir. Annotation düzeltmelerinde, legacy augmentation None değerini zaten reddettiği için Literal da bu gerçeğe uyarlandı; runtime davranışı değiştirilmedi.

P01–P35, U01–U18 ve E01–E13 toplu kabul kampanyaları çalıştırılmadı. Optional deployment performans/kalite kabulü yeniden ilan edilmedi. TensorRT/native GPU çağrısı, paket indirme veya torch_gpu/hyso_trt ortamlarına paket değişikliği yapılmadı. Yerel source unit test interpreter’ı kullanmak environment package değişikliği değildir. `full_cuda_acceptance=false`.

## Kullanım ve yardım örnekleri

```python
from hyso import ModelBuilder, GraphModelBuilder, ModelRegistry
model = (ModelBuilder((1, 8, 8), 2)
         .conv2d(out_channels=4, kernel_size=3, padding=1)
         .relu().build())
graph = (GraphModelBuilder({"image": [1, 8, 8]})
         .conv2d("stem", ["image"], out_channels=4)
         .outputs(features="stem").build())
help(ModelBuilder.conv2d)
help(GraphModelBuilder.conv2d)
print(ModelRegistry().get("conv2d").to_dict())
```

## Final artifactler ve SHA256

Yalnız aşağıdaki yeni `outputs/public-api-ide-20261128/dist/` dosyaları yayın assetidir. Eski release-preparation ve bu görevin superseded-v1/v2 dosyaları kullanılmamalıdır.

- `outputs/public-api-ide-20261128/dist/hyso-0.2.0-py3-none-any.whl` (1213228 bytes)
  SHA256: `3bb652b93d4705c24e8382cd3e346a249b4a0c3ba783bf66c5e12ca37a9fc626`
- `outputs/public-api-ide-20261128/dist/hyso-0.2.0.tar.gz` (1099991 bytes)
  SHA256: `b296b522784c6fb4282b70293ffca4a83507841ad57787ed45466252c83a1b57`

Checksum asseti: `outputs/public-api-ide-20261128/dist/SHA256SUMS.txt`. Wheel/sdist metadata ve 190-modül allowlist PASS; sdist→wheel dosya içerik eşitliği PASS. `sdist-wheel/` wheel’i yeniden üretim kanıtıdır ve ayrı yayın asseti değildir. İki kurulum offline --no-deps/--no-build-isolation ile yenilendi, pip check PASS. direct_url kaynak yolu ve kurulu wheel payload dosyalarının içerik hashleri final artifact ile eşleştirildi; bu pip sürümü direct_url archive_info içinde SHA kaydetmiyor. Kanıt `installed-artifact-provenance.json` içinde.

## Eksikler ve kalan yayın adımları

Envantere dahil public API’de eksik açıklama veya gerekli dokümantasyon/IDE teknik engeli kalmadı. Dinamik result payload’larında Any/dict türleri bilinçli olarak kalır; statik shape ispatı vaat edilmez. ModelBuilder.build tüm named/sequential türleri kapsadığı için nn.Module dönüşünü korur; Graph builder daha belirli GraphModel tipini sunar.

Manuel tooltip görsel QA ve farklı IDE/OS/Python sürümleri NOT_TESTED. Önceki Flatten+Dynamo provider-coverage sınırı ve deneysel/doğrulanmamış TensorRT destek sınırları aynı kalır; bu görev bunları PASS’e yükseltmez.

PyPI için account/project yetkisi ve 0.2.0 sürüm kullanılabilirliği bu görevde doğrulanmadı. Yeni source değişikliklerinin owner review/commit/push/tag işlemleri bekler; GitHub Release ve PyPI yayını yapılmadı. F07/F08 artifact ve belge hazırlığı günceldir, yayın işlemleri ayrıca yürütülmelidir.

## Kesin değişiklik listesi

115 dosya bu görevin review/commit kapsamındadır. Mevcut branch/base commit korundu; staged dosya yok. Önceden var olan .vscode, scratch/probe/patch ve .bak dosyalarına dokunulmadı. Tam git status `outputs/public-api-ide-20261128/git-status-final.txt`, aynı kesin liste `changed-files.txt` içindedir. Artifact/evidence klasörleri ignored kalır.

```text
CHANGELOG.md
RELEASE_NOTES.md
docs/guides/public-api-parameters.md
docs/guides/python-api.md
docs/reports/public-api-ide-20261128.md
hyso/advisor/_json.py
hyso/advisor/comparison.py
hyso/advisor/contracts.py
hyso/advisor/cpu_comparison.py
hyso/advisor/engine.py
hyso/advisor/goals.py
hyso/advisor/model_comparison_presentation.py
hyso/advisor/p12_observation.py
hyso/advisor/performance.py
hyso/advisor/planning.py
hyso/advisor/presentation.py
hyso/advisor/profiles.py
hyso/advisor/recommendations.py
hyso/advisor/render.py
hyso/advisor/resolver.py
hyso/advisor/validation.py
hyso/analysis/_codec.py
hyso/analysis/catalog.py
hyso/analysis/evidence.py
hyso/analysis/execution.py
hyso/analysis/feature_plan.py
hyso/analysis/graph_evidence.py
hyso/analysis/graph_heads.py
hyso/analysis/isolated_bridge.py
hyso/analysis/model_comparison.py
hyso/analysis/native_bridge.py
hyso/analysis/native_execution.py
hyso/analysis/report.py
hyso/analysis/sources.py
hyso/architecture_audit.py
hyso/architecture_comparison.py
hyso/augmentation/augmentation.py
hyso/benchmark/result.py
hyso/benchmark/runner.py
hyso/callbacks/callbacks.py
hyso/checkpoint.py
hyso/datasets.py
hyso/evaluation.py
hyso/experiments/manager.py
hyso/experiments/workflow.py
hyso/export/contracts.py
hyso/export/converters.py
hyso/export/deployment.py
hyso/export/native.py
hyso/export/precision.py
hyso/export/presentation.py
hyso/export/runtime.py
hyso/export/validation.py
hyso/model_checks.py
hyso/model_observations.py
hyso/models/WithSE.py
hyso/models/advanced.py
hyso/models/attention.py
hyso/models/bcam.py
hyso/models/blocks.py
hyso/models/builder.py
hyso/models/components.py
hyso/models/fit.py
hyso/models/graph.py
hyso/models/modular.py
hyso/models/output.py
hyso/models/registry.py
hyso/models/regularization.py
hyso/models/sampling.py
hyso/performance/_loader_dataset_trace.py
hyso/performance/_loader_stage_trace.py
hyso/performance/_loader_task_trace.py
hyso/performance/capacity.py
hyso/performance/cold.py
hyso/performance/comparison_capture.py
hyso/performance/dataloader.py
hyso/performance/devices.py
hyso/performance/energy.py
hyso/performance/gradients.py
hyso/performance/inference.py
hyso/performance/isolated_training_memory.py
hyso/performance/layer_overhead.py
hyso/performance/memory.py
hyso/performance/memory_trend.py
hyso/performance/pipeline.py
hyso/performance/profiling.py
hyso/performance/providers.py
hyso/performance/result.py
hyso/performance/roofline.py
hyso/performance/roofline_counters.py
hyso/performance/runtime_result.py
hyso/performance/stability.py
hyso/performance/streaming.py
hyso/performance/sweeps.py
hyso/performance/training.py
hyso/performance/training_details.py
hyso/performance/training_memory.py
hyso/performance/transfers.py
hyso/performance/utilization.py
hyso/performance/worker_task_result.py
hyso/performance/worker_trace_result.py
hyso/quality.py
hyso/reporting/catalog.py
hyso/reporting/render.py
hyso/reporting/report.py
hyso/structure.py
hyso/structure_comparison.py
hyso/training/config.py
hyso/training/data.py
hyso/training/session.py
hyso/training/trainer.py
tests/test_contextual_help.py
tests/test_public_api_documentation.py
tests/test_pythonic_builders.py
tools/api_documentation_audit.py
```
