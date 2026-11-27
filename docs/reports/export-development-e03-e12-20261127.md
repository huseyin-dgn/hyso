# HYSo E03–E12 geliştirme sonucu — 27 Kasım 2026

Branch: `feature/hyso-models-next`. Bu rapor CPU mühendislik doğrulamasını ve native
kabul hazırlığını kapsar. Gerçek CUDA/TensorRT kabulü yapılmadı. Paket kurulmadı,
güncellenmedi veya kaldırılmadı. `torch_gpu` değiştirilmedi. Commit, push, PyPI ve
Release yayını yapılmadı. İlgisiz mevcut untracked dosyalara dokunulmadı; mevcut
export dosyaları kullanıcının verdiği özel izinle genişletildi.

| Aşama | Sonuç | Kanıt ve sınır |
| --- | --- | --- |
| E03 | BLOCKED | Ayrı interpreter seçimi ve fail-closed RECORD/DLL inventory hazır; TRT kurulumu/kimliği hazır değil. |
| E04 | BLOCKED | PyTorch→retained ONNX→TRT zinciri korunuyor; aynı native bağımlılık engeli. |
| E07 | CPU sayısal PASS; deployment coverage BLOCKED olabilir | Gerçek ONNX FP16 dönüşümü, FP32 I/O/operation koruma, provenance ve çıktı karşılaştırması test edildi. TRT FP16 NOT_TESTED. |
| E08 | CPU PASS | Gerçek dynamic MatMulInteger ve static symmetric QDQ üretimi/calibration test edildi. TRT QDQ native NOT_TESTED. |
| E09 | CPU PASS / GPU NOT_TESTED | Native PyTorch, ORT, TorchScript ve ExportedProgram; çoklu çıktı, hata metrikleri, ayrı etiketli kalite kararı. TRT owned runtime yolu hazır. |
| E10 | CPU PASS / GPU NOT_TESTED | Mevcut benchmark istatistikleri, p99, yükleme/forward kapsamları; owned TRT ölçüm altyapısı hazır. |
| E11 | CPU PASS / GPU NOT_TESTED | Kaynak/girdi/donanım/protokol eşleşmesi, uyumsuz koşullarda speedup reddi. |
| E12 | Mühendislik PASS | Public API, CLI, kayıt okuma/yazma, Advisor/reporting köprüleri, dört dil, optional extras ve wheel envanteri. |

## TensorRT ortamı

Salt okunur tespit: Python 3.10.18 x64, Torch 2.5.1+cu121, ONNX 1.20.0.
Başlangıçta TRT metadata yoktu. Daha sonraki kontrolde `tensorrt-cu12-bindings` ve
`tensorrt-cu12-libs` 10.13.3.9 metadata'sı bulundu; üst `tensorrt-cu12` dağıtımı
yoktu. Denetimler sırasında çalışan bir pip/TRT installer saptanmadı. Bu görev
installer başlatmadı; dışarıdaki kurulumun tamamlandığı varsayılmadı.

NumPy 2.2.2 metadata'sı yanında `~umpy-2.2.6.dist-info` ve `~umpy.libs` kalıntıları
bulunuyor. Bunlar invalid distribution uyarısını açıklıyor; otomatik silinmedi.

| TRT DLL | RECORD bayt | Gerçek bayt |
| --- | ---: | ---: |
| nvinfer_10.dll | 434626560 | 434636320 |
| nvinfer_builder_resource_10.dll | 1823544320 | 1823554080 |
| nvinfer_plugin_10.dll | 55148544 | 55158304 |
| nvonnxparser_10.dll | 3035648 | 3045408 |

Her DLL 9760 bayt daha büyük. Bunun nedeni kanıtlanmadı. RECORD uyuşmazlığı
allowlist genişletilerek kabul edilmedi; native readiness **BLOCKED**.

[NVIDIA 10.13.3 notları](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html)
CUDA 12.1 Update 1, Windows R537+ ve PyTorch >=2.0 listeliyor. Driver 591.74 bu
driver koşulunu karşılıyor. NVIDIA'nın bu sürümde test ettiği ONNX **1.16.0**;
buradaki ONNX 1.20.0 ile gerçek parser/engine uyumluluğu henüz kanıtlanmadı.
Önceki 10.15.1.29 önerisine otomatik geçiş veya ONNX downgrade yapılmadı.

[10.13 allocator binding kaynağı](https://github.com/NVIDIA/TensorRT/blob/release/10.13/python/src/infer/pyCore.cpp)
ve [callback sözleşmesi](https://github.com/NVIDIA/TensorRT/blob/release/10.13/python/docstrings/infer/pyCoreDoc.h)
incelendi. Beş callback açıkça tanımlandı; integer pointer/null, thread-safe sayaç,
alignment, zero-size, resize reddi, async wrapper ve OOM latch simülasyonları test
edildi. Bunlar GPU allocator'ın gerçek native kabulü değildir.

## API ve CLI

```python
from hyso import export as ex
source = ex.onnx(model, inputs, "model.onnx")
half = ex.fp16(source, "half.onnx", example_input={"input0": inputs})
quantized = ex.quantize(source, "int8.onnx", example_input={"input0": inputs})
quality = ex.validate(ex.Runtime(model), ex.Runtime(quantized), labelled_cohort,
                      precision="int8", labelled=True, dataset_id="my-cohort-v1")
a = ex.benchmark(ex.Runtime(model), inputs, warmup=2, iterations=10)
b = ex.benchmark(ex.Runtime(quantized), inputs, warmup=2, iterations=10)
comparison = ex.compare(a, b, validation=quality)
ex.save(comparison, "comparison.json")
report = ex.render(ex.load("comparison.json"), format="html", language="tr")
```

Static quantization: `ex.quantize(..., mode="static", calibration=iterator,
dataset_id="...", representative=True)`. Representativeness yalnız kullanıcı
beyanıdır. Mühendislik testleri açıkça temsilî olmayan sentetik veriler kullandı.

```powershell
hyso export --help
hyso export dependencies
hyso export fp16 model.onnx.export.json --example inputs.npz --output half.onnx
hyso export quantize model.onnx.export.json --mode dynamic --example inputs.npz --output int8.onnx
hyso export benchmark model.onnx.export.json --example inputs.npz --execute --trust-artifact --iterations 10
hyso export validate model.onnx.export.json --candidate int8.onnx.export.json --example inputs.npz --precision int8 --execute --trust-artifact
hyso export report comparison.json --format html --language es --output comparison.html
hyso export advise comparison.json --language de
```

Native CLI build: `hyso export tensorrt --help`. Native benchmark/validation için
`--device cuda --native-config admitted.json --evidence-directory ...` gerekir.
Bu seçenekler sahiplik/bellek/DLL denetimlerini atlamaz ve bu görevde çalıştırılmadı.
`hyso.advisor.export_advice` ve `hyso.reporting.render_export` aynı kanıta bağlı
öneri/renderer katmanını kullanır; model veya ağırlık değiştirmez.

## Optional dependencies

| Extra | Doğrudan paketler | Not |
| --- | --- | --- |
| deployment / deployment-cpu | onnx 1.20.0, onnxruntime 1.23.2 | CPU referans |
| deployment-gpu | onnx 1.20.0, onnxruntime-gpu 1.23.2 | Ayrı ORT GPU ortamı |
| tensorrt | onnx 1.20.0, tensorrt-cu12 10.13.3.9 | Tam builder/runtime |
| onnx-dynamo | onnx 1.20.0, onnxscript 0.5.6 | Açık Dynamo exporter |
| precision | onnx 1.20.0, onnxruntime 1.23.2 | ORT FP16 ve INT8 araçları |

ORT CPU ve GPU aynı ortamda birlikte kurulmamalı. GPU ORT dağıtımı dönüşüm
araçlarını da içerir; bu ortamda `precision` extra'sının CPU ORT'si eklenmemeli.
Torch CPU/CUDA wheel seçimini kullanıcı yapar. Temel HYSo gereksinimlerine TRT
veya ORT eklenmedi. Import sırasında kurulum veya internet erişimi yok.
Kurulum örnekleri ve platform sınırları [model-export rehberinde](../guides/model-export.md).
Bu görevde fresh-install doğrulaması yapılmadı; paket kurulumları yasaktı.

## Gerçek test sonuçları

- İlk yeni özellik koşusu: 30 PASS, 1 FAIL. FP16 dönüştürücü cast topolojisi
  düzeltilip strict ONNX checker korundu. Sonraki P35 cast eşleme engeli gerçek
  çıktı kanıtı korunarak BLOCKED olarak modellendi.
- İlk ilgili regresyon koşusu: 129 PASS, 3 FAIL. Bir hedef FP16 testi, yeni API
  rehber kayıtları ve alignment simülasyon fixture'ı düzeltildi. Hedef koşusu:
  **3 PASS**. Bu koşuda 131 eski export regresyonu ve bir yeni FP16 testi vardı.
- Genişletilmiş yeni özellik/sözleşme koşusu: **43 PASS**.
- Sonraki güvenlik/DLL/regresyon koşusu: **64 PASS**.
- Son yeni özellik koşusu: **44 PASS, 1 FAIL**. Karşılaştırma negatif testinin
  fixture'ı `sum` ile throughput yeniden hesaplarken yuvarlama farkı üretiyordu;
  geçerli fixture `math.fsum` ile kuruldu, üretim bütünlük denetimi gevşetilmedi.
- Düzeltme ve RECORD hash/size hedef koşusu: **6 PASS**. Böylece yeni dosyalardaki
  45 farklı CPU senaryosu geçmiştir; sayılar koşular arasında toplanıp test sayısı
  şişirilmemelidir.
- CLI convert→inspect→benchmark→validate uçtan uca küçük CPU smoke testi geçti.
- Wheel üretimi, optional dependency metadata/env marker ve 17 export/CLI kaynak
  dosyasının paket içinde olması denetlendi. AST ve dört dilli rehber build/check
  geçti. Wheel kurulmadı veya yayımlanmadı.

JUnit kanıtları: `outputs/e13-development-tests-20261127.xml` (ilk son koşudaki
44 PASS/1 FAIL korunmuştur), `outputs/e13-development-corrections-20261127.xml`
(6 PASS), `outputs/e13-development-cli-final-20261127.xml` (son CLI kontrolü).
Native GPU, TRT engine, CUDA acceptance, GPU benchmark: **NOT_TESTED**.

## Dosyalar

Yeni ürün dosyaları: `hyso/export/{precision,validation,deployment,presentation,dependencies}.py`,
`hyso/cli/export.py`. Genişletilen export dosyaları:
`hyso/export/{__init__,contracts,native,runtime,_worker}.py`.

Entegrasyon: `hyso/cli/main.py`, `hyso/advisor/__init__.py`,
`hyso/reporting/__init__.py`, `setup.cfg`, `commercial-profile.json`,
`tools/cuda_acceptance/runner.py` (request'e bağlı interpreter seçimi).

Kabul araçları: `tools/export_acceptance/{runner,scenarios,evidence,validation,guide_metadata}.py`;
yeni `development.py`, `run_development_tests.py`.
Testler: yeni `tests/test_export_development.py`,
`tests/test_export_development_safety.py`; mevcut export admission testinde
alignment simülasyonu düzeltildi.

Belgeler: `docs/guides/model-export.md`, bu rapor, `docs/user-guide/content.json`,
`hyso/data/{api-topics.json,user-guide-topics.json,user-guide.html}`.
Önceden değiştirilmiş diğer dosyalar bu görevin değişikliği olarak sahiplenilmedi.

## Önceki kanıtlar, E13 ve yayın sınırı

658 CPU / 36 GPU ve U01–U18 30/30 tarihsel kabul kayıtları yeniden çalıştırılmadı.
Mevcut ExportAcceptance dosyaları değiştirilmedi: 4 CPU PASS, ADMISSION/E02_CUDA/
E05_CUDA PASS; safety hold null. E02_CUDA/E05_CUDA tekrar çalıştırılmadı.
Yeni kaynaklar için bu kayıtlar otomatik güncel kabul sayılmaz. E13 ayrı çıktıda
etkilenen export kaynaklarını ve gerekçeli revalidation ihtiyacını kaydeder;
ilgisiz P01–P35/U01–U18 için toplu tekrar talep etmez.

```powershell
python -B tools/export_acceptance/development.py --prepare-only --conda-env hyso_trt --output outputs/e13-next-review
```

Bu PrepareOnly çalıştırıldı; ilk ve son snapshot'lar ayrı tutuldu. Native senaryolar:
E03_TRT, E04_TRT, E07_TRT_FP16, E08_TRT_QDQ, E09_TRT_COHORT,
E10_TRT_BENCHMARK, E11_TRT_COMPARISON. Hepsi şu ortamda bağımlılık/kimlik nedeniyle
BLOCKED; çalıştırılmadılar. Ayrı interpreter ile owned campaign hazırlama komutu
rehberde bulunuyor. Kurulum/provenance sorunu çözülmeden ve kaynak/kanıt geçişi
incelenmeden `Resume` kullanılmamalı; taze kampanya eski PASS'i otomatik benimsemez.

Yayın öncesi zorunlu işler: kullanıcı tarafından yetkilendirilmiş paket onarımı,
NumPy/ORT tek-dağıtım tutarlılığı, eksiksiz TRT RECORD provenance, seçilen ONNX
parser uyumluluğu, mevcut limitlerle gerçek owned E03/E04 ve yeni GPU senaryoları,
E13 kaynak/kanıt geçişi ve izin verilirse ayrı ortamda wheel fresh-install kontrolü.
CPU FP16 P35 cast coverage ve ölçülemeyen fiziksel VRAM açık sınırlamalardır.
Native C++, AOTInductor, OpenVINO ve TFLite eklenmedi. `full_cuda_acceptance=false`.
