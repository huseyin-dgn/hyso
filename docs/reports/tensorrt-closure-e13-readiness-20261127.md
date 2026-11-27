# E03/E04 TensorRT closure ve E13 nihai kabul hazırlığı — 27 Kasım 2026

**Sonuç: kabul altyapısı çalıştırılabilir; mevcut TensorRT dependency readiness
BLOCKED.** Paket metadata'sının tamamlanması doğrulandı. Dört TensorRT DLL'inin
RECORD boyutu ve SHA256 uyuşmazlığı sürüyor. NumPy metadata envanteri belirsiz.
GPU inference, engine build, native import testi veya gerçek CUDA acceptance
başlatılmadı. Gerçek kabulü kullanıcı başlatacaktır.

## Bağımlılık ve DLL sonucu

Salt okunur envanter: Python 3.10.18, Torch 2.5.1+cu121, ONNX 1.20.0;
`tensorrt-cu12`, `tensorrt-cu12-libs`, `tensorrt-cu12-bindings` **10.13.3.9**;
`nvidia-cuda-runtime-cu12` **12.9.79**. Önceki eksik üst paket metadata engeli kalktı.

| DLL | RECORD bayt | Gerçek bayt | SHA256/size sonucu |
| --- | ---: | ---: | --- |
| nvinfer_10.dll | 434626560 | 434636320 | FAIL / FAIL |
| nvinfer_builder_resource_10.dll | 1823544320 | 1823554080 | FAIL / FAIL |
| nvinfer_plugin_10.dll | 55148544 | 55158304 | FAIL / FAIL |
| nvonnxparser_10.dll | 3035648 | 3045408 | FAIL / FAIL |

Dört dosyanın NVIDIA Authenticode imzası Valid. Her PE sertifika tablosu **9760
bayt** ve RECORD'daki dosya boyutu sınırında başlıyor. Boyut farkı sertifika tablosu
ile tam örtüşüyor; bu, imzalama/RECORD üretim sırasıyla tutarlı bir bulgudur.
Paketin nasıl üretildiği bu incelemede kanıtlanmadı. Geçerli imza, whole-file
RECORD SHA256 yerine geçirilmedi. RECORD güncellenmedi, sertifika çıkarılmadı,
dosyalar değiştirilmedi; engel görmezden gelinmedi.

Binding `.pyd` ve iki cudart kopyası kendi RECORD/hash/size kontrollerini geçiyor.
Nihai hazırlık seçilen Torch'un ilgili CUDA/cuBLAS/cuDNN/NVRTC/nvJitLink DLL'lerini
de RECORD, canonical path, SHA256, boyut, file identity ve inceleme sırasındaki
değişmezlik açısından tarar. Hatalı dosya varsa kullanılabilir kısmi allowlist
üretmez. Tam dosya kayıtları:

- `outputs/export-closure-20261127/dependency-audit.json`
- `outputs/export-closure-20261127/signatures.json`
- `outputs/export-closure-20261127/pe-certificates.json`
- `outputs/export-final-acceptance-20261127/plan.json`: `dependency_readiness.files`

## CUDA, API ve NumPy

Torch'un cudart DLL'i **527872**, NVIDIA CUDA 12.9 runtime DLL'i **583680** bayt;
adları aynı (`cudart64_12.dll`), SHA256 değerleri farklı. İki dosya da kendi
dağıtımının RECORD kaydını geçiyor. TensorRT paketinin loader kaynağı, NVIDIA CUDA
runtime dizinindeki DLL'leri tam path ile yüklemeyi deniyor. Dolayısıyla yalnız
PATH veya paket sürümüne bakılarak hangi DLL'in kullanıldığı söylenemez.

Nihai profil seçilen Torch cudart'ını kabul eder, alternatif cudart'ı native
allowlist'e eklemez. Seçilen Torch site'ı daima ilk sıradadır. TensorRT loader
importundan sonra gerçek loaded-module path/hash/file identity kontrol edilir;
beklenmeyen DLL varsa **Builder/engine çağrısından önce** owned worker durur. Bu
gerçek yükleme kontrolü henüz çalıştırılmadı; Windows'un iki kopyayı hangi şekilde
yükleyeceği ölçülmüş gibi raporlanmıyor.

[NVIDIA 10.13.3 release notları](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html)
CUDA 12.1 Update 1 ve 12.9 Update 1'i, Windows R537+ ve PyTorch >=2.0'ı listeler.
Driver 591.74 bu minimumu karşılar. Aynı notlarda test edilen ONNX **1.16.0**;
mevcut ONNX 1.20.0 ile gerçek parser/engine uyumluluğu NOT_TESTED.
[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
driver uyumluluğunu açıklar; bu, farklı userspace DLL'lerinin seçimini kanıtlamaz.
[Windows LoadLibraryEx](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa)
yüklenen modülün başka modülleri de yükleyebildiğini belirtir.

Adaptör TensorRT 10.x isim tabanlı I/O, `build_serialized_network`,
`deserialize_cuda_engine`, optimization profile, `set_tensor_address` ve
`execute_async_v3` kullanır. Seçilen 10.13 binding sözleşmesiyle statik uyumluluk
incelendi. `IGpuAllocator` beş callback'i açıkça tanımlar; sıfır/null integer
pointer, alignment, resize reddi, thread-safe sayaç ve OOM davranışı önceki
kaynak-bağlı CPU simülasyon kanıtlarında mevcuttur. C++ binding'in integer pointer
dönüşü ve callback belgeleri tekrar kontrol edildi:
[binding](https://github.com/NVIDIA/TensorRT/blob/release/10.13/python/src/infer/pyCore.cpp),
[callback sözleşmesi](https://github.com/NVIDIA/TensorRT/blob/release/10.13/python/docstrings/infer/pyCoreDoc.h).
Gerçek TRT allocator callback çalışması NOT_TESTED; eski 45 test yeniden çalıştırılmadı.

NumPy **2.2.2** kendi doğru module konumundan yükleniyor; metadata lookup da 2.2.2
döndürüyor. Küçük CPU `arange/reshape/sum` kontrolü **15.0** verdi; Torch/TRT import
edilmedi. Yanında `~umpy-2.2.6.dist-info` ve `~umpy.libs` duruyor; distribution
tarama iki NumPy sürümü görüyor. Gözlenen etki: basit NumPy CPU hesabı çalışıyor,
paket/provenance envanteri belirsiz. Bu, Torch↔NumPy veya TensorRT inference
uyumluluğunun kanıtı değildir. Metadata kalıntıları silinmedi; strict readiness
belirsizliği engel olarak tutuyor.

## Kod düzeltmeleri ve korunan sözleşmeler

- Mevcut `runner.py --final`, mevcut Windows Job/controller/journal/checkpoint
  motorunu E13 profiliyle yapılandırır. Paralel bir kabul motoru yazılmadı.
- `run_final_acceptance.ps1 -ExportFinalAcceptance` eklendi; PrepareOnly, Status,
  Resume ve açık Revalidate aynı girişten geçer.
- Seçilen interpreter ve Torch site'ı önceliği doğrulanır. Kayıtlı ORT referans
  site'ı controller'dan worker'a ve nested worker'a açıkça aktarılır. Önceki tek
  site aktarımı ayrı TRT ortamında ORT'yi görünmez bırakabiliyordu.
- CPU interpreter/CPU Torch kimliği ve eksik bağımlılıklar PrepareOnly planına
  kaydedilir. ORT CPU veya GPU dağıtımından yalnız biri kabul edilir; CPU ORT
  metadata'sının GPU paket adı üzerinden aranması düzeltilir. Referans module ve
  extension kimliği, kaynak site ve sürüm başlatmadan önce bağlanır.
  Korunan shared ORT gerçekte onnxruntime-gpu **1.20.2**'dir; önceki kabul edilmiş
  profil kaynak/binary kimlikleriyle kullanılabilir. Optional dependency profili
  1.23.2 ayrıca tanımlıdır. Hazırlık için gereksiz sürüm yükseltmesi yapılmaz;
  bu iki açık profil dışındaki sürümler BLOCKED kalır.
- Native worker girişi, nested worker environment flag'i yerine mevcut outer
  controller'ın gerçek Job/PID/nonce/readback receipt'ini kullanır. Asıl GPU worker
  kendi tam admission kontrollerini de yürütür.
- E03 dynamic case min/opt/max batch **1/2/4** çalıştırılmasını gerektirir; üç kez
  aynı sabit batch kanıt olarak kabul edilmez. E04 FP16 ayrı zincir olarak doğrulanır.
- Status keşif/import/çalıştırma yapmaz. Dependency BLOCKED durumunda gerçek
  acceptance motoruna/admission'a ulaşılmaz. Başarısız native case otomatik
  yeniden denenmez. Bilinmeyen cleanup veya güvenlik ihlalinde mevcut motor durur.
- Native limitler değişmedi: workspace **64 MiB**, allocation budget **384 MiB**,
  allocator fraction **0.15**, process **3 GiB**, job **6 GiB**, native process
  limiti **4**, timeout **90 s**. Mevcut outer campaign limitleri de artırılmadı.
- Atomic artifact publication, failure retention, kill-on-close, no-breakaway,
  fiziksel UUID ve loaded DLL/file identity denetimleri korunur. Makine path'i,
  UUID, interpreter konumu veya hash kaynak koda eklenmedi; değerler keşif ve
  yerel kanıt dosyalarında tutulur.

Değişen/eklenen dosyalar:

1. `run_final_acceptance.ps1`
2. `tools/cuda_acceptance/runner.py`
3. `tools/export_acceptance/runner.py`
4. Yeni `tools/export_acceptance/final_profile.py`
5. Yeni `tests/test_export_final_readiness.py`
6. `docs/guides/model-export.md`
7. Bu rapor ve `outputs/export-closure-20261127` kanıtları.

Ürün model/precision/benchmark modülleri bu görevde değiştirilmedi. İlgisiz mevcut
untracked dosyalara dokunulmadı. Görevle ilgili export dosyaları için kullanıcı
istisna izni alındı. Paket kurulmadı/güncellenmedi/kaldırılmadı; `torch_gpu`
değiştirilmedi. Commit/push veya yayın yapılmadı.

## E13 matrisi ve önceki kanıtlar

Önceki 45 CPU engineering JUnit case'i son düzeltme kanıtlarıyla birleştirilip
kayıtlı kaynak hash'lerine bağlandı: **45 PASS yeniden kullanıldı, yeniden
çalıştırılmadı**. Bunlar yeni owned kabul receipt'i olarak sunulmuyor. Kaynak
eşleşmesi bozulursa PASS terfi ettirilmez. Önceki 4 CPU ve ADMISSION/E02_CUDA/
E05_CUDA 3 GPU PASS kayıtları tarihsel scope'larıyla korunuyor; yeni TRT
interpreter admission'ı yerine geçirilmiyor.

| Alan | CPU kanıtı | Kalan gerçek GPU kabulü |
| --- | --- | --- |
| E01/E02: dependencies, ONNX, integrity | Kaynak-bağlı önceki CPU kanıtı | E03_TRT: ONNX→TRT FP32 |
| E03/E04 | Native test yok | E04_TRT: PyTorch→ONNX→TRT FP32 |
| E05/E06: TorchScript / torch.export | Önceki reload/equivalence testleri | C++/AOT kapsam dışı |
| E07: FP16 | Gerçek graph/dtype/provenance/numerical testleri | E07_TRT_FP16, E04_TRT_FP16 |
| E08: INT8 dynamic/static, calibration | MatMulInteger, QDQ ve invalid input testleri | E08_TRT_QDQ; mevcut açık signed symmetric QDQ kapsamı |
| E09: numerical ve labelled task quality | Metrikler, çoklu çıktı, ayrı kalite kararı | E09_TRT_COHORT |
| E10: deployment benchmark | Küçük CPU ölçüm/kapsam testleri | E10_TRT_BENCHMARK |
| E11: comparison | Uyumluluk ve yanıltıcı speedup reddi | E11_TRT_COMPARISON |
| E12: API/CLI/Advisor/raporlar | tr/en/de/es × JSON/Markdown/HTML, saved records, optional deps | Ölçülmemiş performans iddiası yok |
| Dynamic profile | CPU sınır/shape sözleşmeleri | E03_TRT_DYNAMIC: min/opt/max 1/2/4 |
| Native safety | Yeni metadata/dispatch/negative-input testleri | Yeni selected-interpreter ADMISSION ve bütün native cleanup receipt'leri |

Plan ayrıca yeni **24 CPU closure/dispatch/safety case'i**, **9 native GPU case'i**
ve **1 yeni ADMISSION** içerir. E02_CUDA/E05_CUDA, 658 CPU, 36 GPU, U01–U18 30/30,
P01–P35 ve uzun P31 bu seçimde yeniden çalıştırılmaz.

CPU P35 FP16 cast coverage BLOCKED, mevcut test fixture'ında sayısal ürün hatası
olarak sınıflandırılmadı: ORT CPU precision cast yeniden yazımlarında her orijinal
node'un yürütülmesini P35 tam eşleyemiyor. Gerçek sayısal çıktı karşılaştırması
ayrıdır. Desteklenmeyen attestation profili BLOCKED kalır; node coverage denetimi
gevşetilmez, deployment PASS üretilmez. Gerçek numerical drift yine FAIL olur.

## Çalıştırılan doğrulamalar

- İlk yeni koşu: 11 PASS / 1 FAIL. PowerShell parse testinin argüman aktarım
  fixture'ı düzeltildi; gerçek acceptance çalıştırılmadı.
- Genişletilmiş koşu: 17 PASS / 1 FAIL. Windows path ayıracı yüzünden mock'un
  kaynak değişikliğini tetiklemeyen test fixture'ı düzeltildi; hash denetimi korunur.
- Hedef düzeltme koşusu: 2 PASS.
- Tam yeni closure/dispatch/safety koşusu: **21 PASS**.
- Yeni CPU ORT/reference negative testleri: **2 PASS**. Ayrı CPU/reference
  kimliği değişikliğinin yürütmeyi durdurması da hedef testle doğrulandı. Toplam **24 farklı yeni
  senaryo** doğrulandı; koşuların sayıları toplanarak test sayısı şişirilmedi.
- Python AST kontrolü ve gerçek PowerShell **PrepareOnly** çalıştı. Gerçek Status
  girişi salt okunur kontrol edildi. Gerçek Start/Resume çalıştırılmadı.
- Önceki 45 case, genel kabul kümeleri ve GPU acceptance tekrar çalıştırılmadı.

JUnit kayıtları `outputs/export-closure-20261127/tests-*.xml` altında; ilk
başarısızlıklar ve düzeltme koşuları da korunur. GPU/engine/native allocator/
runtime-loader doğrulaması: **NOT_TESTED**.

## Kullanıcının çalıştıracağı komutlar

```powershell
# Hazırlık: bu görevde çalıştırıldı; engine/inference başlatmaz.
.\run_final_acceptance.ps1 -ExportFinalAcceptance -PrepareOnly

# Gerçek kabul: yalnız kullanıcı çalıştırır.
.\run_final_acceptance.ps1 -ExportFinalAcceptance

# Salt okunur durum.
.\run_final_acceptance.ps1 -ExportFinalAcceptance -Status

# Tamamlananları koruyarak kalanları çalıştırır; native hata/engeli kendiliğinden denemez.
.\run_final_acceptance.ps1 -ExportFinalAcceptance -Resume
```

Ortam değiştirilecekse `-RuntimeEnvironment <Conda-adı>` veya `-RuntimePython
<seçilen-python.exe>`; kanıt hedefi için `-AcceptanceRoot <klasör>` kullanılabilir.
Varsayılan ortam `hyso_trt`, kanıt kökü `outputs/export-final-acceptance-20261127`.
Bağımlılıklar değişirse önce PrepareOnly gerekir. Başarısız/blocked native case
ancak neden incelendikten sonra `-Resume -Revalidate <tam-case-adı>` ile açıkça
yeniden doğrulanır. Safety hold ve cleanup belirsizliği normal Resume ile aşılmaz.

PrepareOnly sonucu: **PREPARED**, native dependency **BLOCKED**, reused CPU **45**,
GPU executed **false**, automatic retries **0**. Eksikler giderilmeden Start
bağımlılık kontrolünde durur; GPU admission veya engine build'e geçmez.

Kalan engeller: dört DLL'in whole-file RECORD uyuşmazlığı; NumPy metadata
belirsizliği; gerçek owned loader'da tek doğru cudart seçiminin henüz doğrulanmamış
olması; ONNX 1.20/TRT parser ve bütün yeni native case'lerin gerçek kabulü.
Yerel RECORD yeniden yazımı veya hash istisnası çözüm olarak uygulanmadı.

## F01–F08 yayın hazırlığı ön envanteri

| Aşama | Mevcut kanıt / kalan iş |
| --- | --- |
| F01 IDE/autocomplete | Type hints, py.typed ve statik rehber/API kayıtları var. Gerçek IDE içinde kurulu paket autocomplete/signature doğrulaması henüz yok. |
| F02 dokümantasyon/örnekler | Dört dil ve Python/CLI örnekleri var. Native closure sonrası destek/limitation tabloları ve örneklerin gerçek kabul sonuçlarıyla son eşlemesi gerekli. |
| F03 dependency/packaging | CPU/GPU/TRT/Dynamo/precision extras ayrı. ORT namespace çakışması, Torch wheel seçimi ve desteklenen Python/platform aralığı release politikasıyla netleştirilmeli. TRT RECORD/NumPy engelleri açık. |
| F04 wheel/sdist | Önceki wheel 17 export/CLI kaynak dosyasını ve optional metadata'yı doğruladı. Bu görevde yeniden build/install yapılmadı; güncel final release wheel/sdist payload ve sdist→wheel yeniden üretim kontrolü gerekli. |
| F05 temiz kurulum | Bu görevde kuruluma izin verilmedi. Ayrı temiz CPU ve TRT ortamlarında kurulum/import/CLI/optional-dependency negatif doğrulaması gerekli. |
| F06 version/changelog | setup.cfg ve CLI 0.2.0. E01–E12'nin gerçek kabul sınırlarını içeren final changelog/release notes ve version tutarlılığı henüz onaylı değil. |
| F07 PyPI | Native/E13 engelleri, final artifacts ve temiz kurulum kapanmalı; proje/sürüm kullanılabilirliği ve yayın yetkisi ayrıca doğrulanmalı. Yayın yapılmadı. |
| F08 GitHub Release | E13 sonucu, release notes, wheel/sdist/checksum asset'leri ve hedef commit/tag hazır olmalı. Commit/push/tag/release yapılmadı. |

Native C++, AOTInductor, OpenVINO ve TFLite kapsam dışı. Bellek veya timeout
artırılarak PASS aranmadı. `full_cuda_acceptance=false`.
