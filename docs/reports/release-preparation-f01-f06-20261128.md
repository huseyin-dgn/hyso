# HYSo 0.2.0 — F01–F06 nihai yayın hazırlığı

Tarih: 2026-11-28 (Europe/Istanbul). Durum: CPU yayın adayı hazır; yayın yapılmadı.

| Aşama | Nihai durum | Kanıt / sınır |
|---|---|---|
| F01 | PARTIAL | Kurulu wheel: 26 public API import/signature/docstring ve py.typed PASS. Gerçek Pylance completion/signature NOT_TESTED. |
| F02 | PASS | Dört dil çevrimdışı rehber, yeniden üretim, kurulum/export belgeleri ve destek sınırları; önceki doğrulamalar korundu. |
| F03 | PASS | 0.2.0 dependency/extras, ayrı ORT CPU/GPU politikası, açık Torch dağıtım seçimi. |
| F04 | PASS | 190 Python modülü; wheel/sdist allowlist, veri/metadata, sdist→wheel içerik eşitliği. |
| F05 | PASS (Windows x64 / Python 3.10.18 / Torch 2.5.1+cpu) | İki izole ortam, repository dışından temel ve optional CPU smokes. GPU/TRT optional kurulumu ve smoke NOT_TESTED. |
| F06 | PASS | Version 0.2.0 korundu; changelog/release notes sonlandırıldı. |

## Devralınan durum ve tamamlanan işler

İlk incelemede önceki göreve ait çalışan Python/pip işlemi yoktu. İki task-local venv ve artifact klasörleri yerindeydi. Önceki temel smoke kayıtları 16/16 PASS idi; erken optional kurulumlar geri alınmıştı. Eksik sdist CLI export-help/version/guide tamamlandı. Gerçek IDE araştırması yapılmadı.

İlk optional CPU smoke, `hyso._native_identity` modülünün commercial allowlist içinde bulunmadığını gösterdi. Bu modül CPU ONNX loaded-module doğrulamasında gereklidir. Paket listesine eklendi; inventory testi 190 modüle güncellendi. Başarısız testin logu saklandı. Kaynak algoritması değiştirilmedi. Bu gerçek paketleme düzeltmesi ve changelog/release notes değişikliği nedeniyle wheel/sdist ve sdist→wheel yeniden üretildi; içerik karşılaştırması PASS. Önceki artifactler `outputs/release-preparation-20261127/superseded/` altında tutuldu ve yayın asseti değildir.

Düzeltilmiş artifactler mevcut izole task ortamlarına --no-deps ile yeniden kuruldu. Wheel ortamındaki optional paketler önce kaldırıldı; her iki temel smoke bitmeden hiçbir optional kurulum başlatılmadı. Yeni venv oluşturma veya büyük bağımlılıkları tekrar indirme gerekmedi. Bütün kurulumlar yerel wheelhouse üzerinden --no-index ile yapıldı. torch_gpu ve hyso_trt ortamlarına kurulum/değişiklik yapılmadı. Venvin taban interpreter yolunun torch_gpu olması system-site-packages erişimi veya o ortamın değiştirildiği anlamına gelmez.

## Temiz kurulum ve smoke sonuçları

| Artifact | Temel smoke | Optional smoke | pip check |
|---|---|---|---|
| wheel | 16/16 PASS | 11/11 PASS | Temel + optional PASS |
| sdist | 16/16 PASS | 11/11 PASS | Temel + optional PASS |

Temel kapsam: installed origin, signature/docstring/py.typed, eksik ONNX bağımlılığının çıktı yayımlamadan reddi, ModelBuilder/GraphBuilder, weights=None torchvision, küçük CPU training/checkpoint/resume, evaluation, bounded benchmark/analysis, Advisor/comparison, dört dil JSON/Markdown/HTML, TorchScript ve ExportedProgram. CLI module/console help, export help, version ve offline guide kayıtları korundu/tamamlandı.

Optional kapsam: deployment-cpu + onnx-dynamo + precision; legacy ve Dynamo ONNX export/runtime eşdeğerliği, sayısal validation, bounded benchmark/comparison, dört dil export raporu/Advisor, FP16, dynamic INT8, static QDQ calibration ve model/RNG korunumu. Bunlar sentetik kurulum smoke testleridir; tarihsel kabul testleri veya performans/kalite sertifikası değildir. Runtime smoke sırasında ağ bağlantısı ve CUDA initialization engellendi. TensorRT import edilmedi.

deployment alias metadata ile doğrulandı; GPU ve TensorRT dağıtımları kurulmadı. Python 3.11/3.12 ve diğer OS NOT_TESTED. P01–P35, U01–U18, E01–E13 kabul kampanyaları tekrar çalıştırılmadı; önceki evidence hashleri değişmedi. `full_cuda_acceptance=false`.

Önceki paketleme/rehber test turu 19 PASS + README newline karşılaştırması nedeniyle 1 FAIL idi; test düzeltmesinden sonraki hedefli tekrar 1/1 PASS. Bu ilk başarısızlık gizlenmedi. Son paketleme düzeltmesi ayrıca `resume-repaired-content.log` ile final artifactler üzerinde doğrulandı.

## Yayına ayrılan artifactler

- `outputs/release-preparation-20261127/dist/hyso-0.2.0-py3-none-any.whl` (1159730 bytes)
  SHA256: `5c0c81be58295f780f18357fc475626a21d8796b6f5a55af883d7ab6d1203da5`
- `outputs/release-preparation-20261127/dist/hyso-0.2.0.tar.gz` (1051774 bytes)
  SHA256: `5d331d813cf2574f112b7afb8dc36affd8fb31d603bcfb2d7f6965a54a74f7c7`

Checksum asseti: `outputs/release-preparation-20261127/dist/SHA256SUMS.txt`. `sdist-wheel/` yeniden üretim kanıtıdır; ikinci bir wheel yayınlanmamalı. Final hashler `final-artifacts.json` ile eşleşir.

## Git ve kesin commit listesi

Git: 30 tracked dosya değişmiş; staged dosya yok. Aşağıdaki 80 dosya mevcut kaynak/belge/test değişikliklerinin review/commit kapsamıdır. Commit yapılmadı. `.vscode`, kök scratch/probe/patch dosyaları ve `.bak` yedekleri kapsam dışı bırakıldı ve silinmedi. Tam git status ve kapsam dışı liste kanıt klasöründe `git-status-final.txt`, `excluded-local-files.txt` içinde. Artifactler outputs altında ignored; binary asset olarak tutulmalı.

```text
CHANGELOG.md
MANIFEST.in
README.md
RELEASE_NOTES.md
commercial-profile.json
docs/guides/cli.md
docs/guides/first-release.md
docs/guides/model-export.md
docs/guides/python-api.md
docs/guides/python-first-acceptance-and-limits.md
docs/guides/python-first-workflow.md
docs/guides/release-support.md
docs/reports/export-development-e03-e12-20261127.md
docs/reports/release-preparation-f01-f06-20261128.md
docs/reports/tensorrt-closure-e13-readiness-20261127.md
docs/user-guide/README.md
docs/user-guide/content.json
examples/export/libtorch/CMakeLists.txt
examples/export/libtorch/main.cpp
hyso/__init__.py
hyso/_native_identity.py
hyso/advisor/__init__.py
hyso/cli/export.py
hyso/cli/main.py
hyso/data/api-topics.json
hyso/data/user-guide-topics.json
hyso/data/user-guide.html
hyso/export/__init__.py
hyso/export/_common.py
hyso/export/_dll_identity.py
hyso/export/_evidence.py
hyso/export/_model.py
hyso/export/_worker.py
hyso/export/artifacts.py
hyso/export/contracts.py
hyso/export/converters.py
hyso/export/dependencies.py
hyso/export/deployment.py
hyso/export/native.py
hyso/export/precision.py
hyso/export/presentation.py
hyso/export/runtime.py
hyso/export/validation.py
hyso/performance/_runtime_worker.py
hyso/performance/_training_process.py
hyso/performance/providers.py
hyso/performance/stability.py
hyso/reporting/__init__.py
run_export_dll_diagnostic.ps1
run_final_acceptance.ps1
setup.cfg
tests/test_contextual_help.py
tests/test_export_acceptance_scenarios.py
tests/test_export_admission.py
tests/test_export_blocker_resolution.py
tests/test_export_development.py
tests/test_export_development_safety.py
tests/test_export_dll_diagnostic.py
tests/test_export_dll_manifest.py
tests/test_export_engineering.py
tests/test_export_final_readiness.py
tests/test_packaging.py
tests/test_product_experience.py
tests/test_user_guide.py
tools/cuda_acceptance/native_identity.py
tools/cuda_acceptance/runner.py
tools/export_acceptance/__init__.py
tools/export_acceptance/cpu_worker.py
tools/export_acceptance/development.py
tools/export_acceptance/diagnostic_gate.py
tools/export_acceptance/dll_diagnostic.py
tools/export_acceptance/evidence.py
tools/export_acceptance/final_profile.py
tools/export_acceptance/gpu_worker.py
tools/export_acceptance/guide_metadata.py
tools/export_acceptance/run_development_tests.py
tools/export_acceptance/runner.py
tools/export_acceptance/scenarios.py
tools/export_acceptance/validation.py
tools/product_acceptance/scenarios.py
```

## Gerçek kalan yayın adımları ve F07/F08

F07/PyPI: artifact ve checksum hazırlığı tamam. Account/project yetkisi, 0.2.0 sürüm kullanılabilirliği ve yayın kararı henüz doğrulanmadı; kimlik bilgisi kullanılmadı, upload yapılmadı.

F08/GitHub Release: release notes ve assetler hazır. İncelenmiş commit, push ve 0.2.0 tag/release henüz yok. Bu adımlar kullanıcı tarafından ayrıca yürütülmeli.

F01 gerçek IDE deneyimi açık NOT_TESTED maddesidir. Tam F01 veya GPU/TRT doğrulanmış destek iddiası için kanıt yoktur. Bunlar belgelenmiş CPU kapsamındaki bu artifactlerin test sonucu olarak sunulmaz. TensorRT deneysel/doğrulanmamış desteğin önceki engelleri korunmuştur; yeniden inceleme yapılmadı. Native C++, AOTInductor, OpenVINO ve TFLite kapsam dışıdır.

Bu görevde commit, push, tag, PyPI upload veya GitHub Release oluşturma işlemi yapılmadı.

## Optional smoke kapsam sınırı

İlk düzeltilmiş optional testin Flatten+Linear Dynamo örneği `PROVIDER_EXECUTION` ile reddedildi: Tam original-node coverage sağlanmadı; shape düğümlerinin runtime dönüşümü olası nedendir, kök neden bu görevde kesinleştirilmedi. Bu sonuç PASS olarak yeniden etiketlenmedi; `resume-wheel-repaired-optional.log` korundu. CPU legacy ONNX aynı modelde geçti. Dynamo bağımlılık/kurulu paket smoke testi ayrıca mevcut engineering örneğindeki küçük Linear model ile yapıldı. Runtime admission veya mapping algoritması değiştirilmedi. Son tabloda PASS sayıları bu açıkça daraltılmış smoke kapsamına aittir; Flatten+Dynamo kombinasyonu bu ortamda doğrulanmış destek değildir. Bu sınırlama release notes içindeki genel kapsam sınırına ek olarak okunmalıdır.
