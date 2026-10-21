# HYSo — P23 R1 Safe Consolidation / CPU Release Preflight

2026-10-21, Europe/Istanbul. **Commit / release: NO-GO.** Yeni P23 CPU
coverage ve packaging/minimum smoke başarılı; full legacy CPU koşusu temiz
geçmedi. **B4 OPEN / root cause UNRESOLVED; P23 PARTIAL; B10 PARTIAL.**
Eski R1 native crash / NO-GO kaydı korunur. Bu görevde yeni CUDA workload,
CUDA smoke/regression veya guard bypass çalıştırılmadı.

## 1. Branch, HEAD ve kesin dirty envanteri

Branch `feature/hyso-models-next`; HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`. Index boş. Başlangıçta
**18 dirty dosya: 15 P23 görev dosyası + 2 forensic rapor + kullanıcı gradient
farkı** vardı. Hepsinin başlangıç/son SHA256 eşliği doğrulandı. Bu görevde
repository'ye yalnız bu rapor eklendi; son envanter **19 dosya (5 M, 14 ??)**.

| Git status | Başlangıç dosyası; byte'ları korundu |
|---|---|
| ` M` | `docs/guides/performance-training-memory.md` |
| ` M` | `docs/roadmap/hyso-performance-backlog-v1.1.md` |
| ` M` | `hyso/performance/__init__.py` |
| ` M` | `tests/test_gradient_accumulation.py` |
| ` M` | `tests/test_packaging.py` |
| `??` | `docs/audit/m2-p-b10-p23-isolation-activation.md` |
| `??` | `docs/audit/m2-p-b4-p23-native-crash-investigation.md` |
| `??` | `docs/audit/m2-p-b4-p23-single-forensic-reproduction.md` |
| `??` | `docs/guides/performance-training-memory-isolated.md` |
| `??` | `hyso/performance/_saved_activation.py` |
| `??` | `hyso/performance/_training_memory_worker.py` |
| `??` | `hyso/performance/_training_process.py` |
| `??` | `hyso/performance/isolated_training_memory.py` |
| `??` | `tests/cuda_isolated_training_memory_validation.py` |
| `??` | `tests/fixtures/training-memory-p23-schema1.json` |
| `??` | `tests/test_isolated_training_memory.py` |
| `??` | `tests/test_saved_activation.py` |
| `??` | `tests/training_memory_factories.py` |
| `??` | `docs/audit/m2-p-b10-p23-cpu-release-preflight.md` — yalnız bu görevde eklenen dosya |

`tests/test_gradient_accumulation.py` SHA256, başlangıç ve son:

`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`

Dosyaya yazılmadı; kullanıcıya ait mevcut fark korundu. Stage/commit/push,
reset/checkout/clean, dependency/Torch/driver değişikliği yapılmadı.

## 2. Kaynak dondurma ve yeni kanıt kökü

Bu görevin evidence kökü **Ecpu**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-cpu-preflight-20261021T161847`.

`source-manifest.json`: **243** kaynak/test/fixture/config/dokümantasyon dosyası;
SHA256 `3d5c93b79d2e23180bbcc7025bbd7c05987cd18ef288b7a8364dcf1613b2f478`.
`frozen-source/` aynı byte'ların bağımsız kopyasıdır. Testler mevcut checkout ve
fresh installed target'larda çalıştı; mevcut kaynakların testlerden önce/sonra
manifest eşliği doğrulandı. Bu kontrol, her canlı native frame'in ayrı bir
memory attestation'ı olarak sunulmaz. Kaynak veya test düzeltmesi yapılmadı.

`start-protection.json`, `provenance-comparison.json`, XML/loglar,
`cpu-result-evidence.json`, `archive-integrity.json`, `all-module-imports.json`,
`legacy-failures.json`, `legacy-profile-unavailable.json` ve
`preflight-summary.json` yeni sonuçların kanıtlarıdır. Tarihsel 2303 PASS bu
kaynak sürümünün final sonucu olarak kullanılmadı.

## 3. Başarısız ve başarılı CUDA forensic denemelerin karşılaştırması

İncelenen tarihsel raporlar:

- [P23 R1 isolation/activation](m2-p-b10-p23-isolation-activation.md).
- [Native crash investigation](m2-p-b4-p23-native-crash-investigation.md).
- [Tek bounded diagnostic](m2-p-b4-p23-single-forensic-reproduction.md).

Eski kök `C:/Users/hdgn5/AppData/Local/Temp/hyso-p23r1-20261021`;
başarılı kopya `single-repro-20261021T125035Z`. Receipt, tamamlanmış12 sonuç,
manifest/import identity ve observed-summary birlikte karşılaştırıldı.

| Konu | Başarısız eski C | Başarılı tek forensic |
|---|---|---|
| Kaynak kimliği | **unverified_source_identity**; mutable checkout/factory yolu; crash öncesi C manifest/import attestation yok | Bağımsız read-only post-stop kopya,147 source +25 runtime dosyası;8 pre/post identity checkpoint |
| Manifest | Frozen B manifesti C'yi kanıtlamaz; B'de saved-activation modülü yoktu | `3b24a24a053c0ffd90125fa29952212bb412cd0a1a90eaeae31c5ee4b4835b21` |
| Scope/sıra | SGD6 + Adam6 başarılı child sonucu, ardından AdamW13. case native failure |1 attempt /1 child fit; prefix0, baseline fit0; başka case yok |
| Ortak recipe alanları | AdamW FP32, default stream, overflow=false, reset=true, activation=true,seed71,epochs2 | Aynı dar recipe alanları;5 sample,batch2,accum2,Linear–Dropout fixture |
| Fixture/output farkı | Receipt `state_path` digest export wrapper seçiyor; beklenen state/result yayınlanmadı | State digest wrapper, baseline replay, checkpoint/resume yok; yalnız diagnostic ledger |
| CUDA initialization | Failed child CUDA library loading/dropout dispatcher stack'ine ulaştı; kesin initialization/phase/window/reset ledger yok | Parent CUDA initialized=false; admitted child allocator fraction.15 modelden önce; stream0 durably kayıtlı |
| Hook davranışı | Failed worker pack/unpack/release sayıları ve son payload durumu bilinmiyor; önceki12 PASS onun cleanup kanıtı değil |60 pack,48 unpack,60 native release,0 forced cleanup,0 live payload; callback'ler iki native thread'de |
| Allocator | Reset izni var; kaç resetin tamamlandığı bilinmiyor |4 reset attempt /4 completion; retry/OOM delta0;4 window |
| Instrumentation | Child stdout boş; stderr yalnız warning; worker receipt/measurement yok |499 fsync metadata satırı; phase/callback/cleanup/identity ledger, bounded çıktılar |
| Exit / cleanup | PID19264:3221226505 /0xC0000409; parent19548 exit1 assertion; child Job active0; timeout yok | Child22448,parent18532,controller7496 exit0; inner/outer Job active0; timeout yok |

Eski `c-cuda-fixed.stderr.log` matrix `run_case` çağrısını ve başarısız
assertion'ı gösterir. Bugünkü harness'te parent baseline kodu bulunması,
eski process'in yüklediği bütün byte'ları veya parent CUDA state'ini kanıtlamaz.
Yeni koşunun cold tek-case olması prefix/parent/timing history'sini tekrar
etmez. Metadata I/O'su scheduling/destruction timing'ini değiştirebilir.

Eski dump: `nvcuda64.dll`32.0.15.9174, offset0x18fd6e5, fastfail subcode2
(STACK_COOKIE_CHECK_FAILURE); dar immediate stack dropout/library-load →
`cuLibraryLoadData` → NVIDIA fail-fast. Bu, driver veya HYSo/PyTorch root
cause'unu tek başına belirlemez. Dump/failure kayıtları silinmedi veya PASS
olarak yeniden etiketlenmedi. Başarılı yeni manifest, eski C kaynak kimliğini
gerçekte doğrulamıyor. Mevcut17 dosyanın başarılı diagnostic'in post-stop
baseline hashleriyle eşliği ayrıca doğrulandı; bu da geçmiş crash attestation'ı
değildir.

## 4. Production guard ve CPU-only admission

`isolated_training_memory.py::_run_recipe` içinde
`saved_activation_lifetime && device.type == cuda` için `NotImplementedError`
koruması etkin. AST kontrolü bu reddin CUDA availability/device/fraction
admission'ından önce olduğunu doğruladı. Dosya SHA256 değişmedi; forensic
kopyanın guard istisnası checkout'a taşınmadı.

Test/factory/conftest ve worker bootstrap çağrıları çalıştırmadan önce tarandı
(`cuda-test-static-audit-utf16.txt`). Test runtime **Python3.10.18 /
Torch2.2.2+cpu**, `torch.version.cuda=None`; görünürlük ayrıca `-1`.
Session başı/sonu initialized=false doğrulandı. P23 worker runtime path'larında
CPU site-packages ilk sırada; worker parent Torch sürümünü doğruluyor.
Saklanan8 isolated export'un provenance'ı Torch2.2.2+cpu/device=cpu,
exit0/Job active0 gösteriyor.

`test_r12_r13_real_cuda_device_smoke` explicit deselect edildi. CUDA adını
taşıyan şema fixtures/protocol doubles/unsupported admission ve sentetik
native-exit classification testleri CPU kontrolleridir; native CUDA başarı
kanıtı sayılmadı. `cuda_*.py` validation script'leri çalıştırılmadı. CPU Torch
build'i native CUDA initialization sağlayamaz. Build/pip araçları mevcut
Torch2.5.1 environment interpreter'ını yalnız build/no-deps target install
için kullandı; eğitim/import smoke CPU interpreter'ındaydı.

## 5. CPU isolated-worker sonucu

`p23.xml`: **58 PASS /0 failure /0 error /0 skip**,279.156s.
`test_isolated_training_memory`: **29 PASS**.

SGD/Adam/AdamW tek-fit exact state/parent RNG, constructor/forward exception,
real Job memory/log/result limits, forward timeout, child/grandchild cleanup,
GC-disabled weakrefs, immutable result/JSON/CSV, schema1 compatibility,
tamper ve raw Windows RSS unsupported davranışı mevcut kaynakta doğrulandı.
`cuda=None` CPU CSV yolu da fresh export testleri ve installed smoke içinde
geçti. Native crash classification testi sentetik receipt'tir; gerçek crash
üretimi değildir.

## 6. CPU saved-activation sonucu

`test_saved_activation`: **29 PASS**. SGD/Adam/AdamW,tied-storage,ModularCNN,
early-stop exact state/RNG; interrupted checkpoint → resume → completed no-op;
real worker forward/backward failure ve success GC-disabled cleanup;
direct exception/budget unwind weakrefs/hook cleanup; retained graph,
custom autograd ve checkpointing unsupported admission; alias/original-object
identity/storage deduplication doğrulandı.

JSON/CSV no-clobber, Experiment Manager copied artifact SHA verification,
metric finish/close/reopen, tracking error'da retrain olmaması ve9 rehashed
tamper varyantı PASS. Bu CPU gözlemleri genel CUDA/destructor güvenliği veya
physical VRAM/storage-free lifetime kanıtı değildir. Önceki static payload
cycle/reentrant-destruction riskleri root cause kanıtlanmadan kapatılmadı.

## 7. Full CPU legacy regression ve açık hatalar

Bütün `tests/test_*.py` dosyaları seçildi; P23 iki dosyası ayrı58-test koşusunda,
packaging ayrı kontrol edildi. Legacy **55 test dosyası** seçimi,
`legacy.xml`: **2825 executed =2814 PASS /11 FAIL /0 error /0 skip**;
1 gerçek CUDA smoke deselect;563.979s. Bu tam koşu **FAIL** olarak korunur.

| Failure grubu | Adet | Gözlenen kanıt / sınır |
|---|---|---|
| Benchmark comparison Torch-version mismatch fixture |1 | Fixture2.2.2+cpu'yu farklı sürüm sanıyor; gerçek runtime da2.2.2+cpu olduğundan karşılaştırma compatible dönüyor |
| OneCycle non-legacy round-trip |4 | Test2.5.1 native state biçiminin aynılığını bekliyor;2.2.2 native `anneal_func` ile canonical `_anneal_func_type` key sırası farklı |
| CPU GradScaler overflow scheduler |2 |2.2.2 `torch.amp.GradScaler` API'si yok; test eğitim başlamadan AttributeError |
| Fused Adam unsupported-profile fixture |1 |2.2.2 native Adam CPU'da fused=True construction'ı HYSo validation öncesi reddediyor |
| Profiling → Manager metric |1 |Saklanan profile partial; `aten::_convolution.self_cpu_us=null`; test bunu numeric metric olarak gönderiyor, Manager invalid_metric_value ile reddediyor |
| Recursive pytest quality harness |2 |İlk external PYTHONPATH Python3.13 Pillow'u CPU runtime'ın önüne koydu; collection ImportError `_imaging` |

Son iki kalite testi yalnız dış harness path sırası CPU site-packages ve
Python3.10 dependency yolları önce olacak biçimde düzeltildikten sonra
`quality.xml`: **2 PASS /0 fail**,27.544s. Üretim/test kaynağı değiştirilmedi;
ilk11 failure sonucu üzerine yazılmadı. Başarılı bu targeted koşu tam suite'i
sonradan temiz PASS yapmaz; kalan **9 failure** hâlâ gate açığıdır.

İlk8 failure'ın runtime/test varsayımı sınırları doğrudan traceback/kodla
gösterildi; bütün minimum-version functional contract'larının güvenli olduğu
sonucu çıkarılmadı. Profiling integration'da partial/null timing politikası
test/consumer düzeyinde ayrıca ele alınmalı; null değer0'a çevrilmedi, metric
validation gevşetilmedi, timing clamp/retry uygulanmadı. Bu görevde bu legacy
kod/test dosyaları değiştirilmedi. Eski2303 PASS veya2 targeted PASS ile yeni
temiz full regression toplamı türetilmedi.

## 8. Minimum installed runtime

Mevcut `hyso-p06c-oi2koqex/clean/Scripts/python.exe`: Python3.10.18,
Torch2.2.2+cpu. Yeni wheel ve sdist ayrı fresh `--target` dizinlerine
`--no-index --no-deps --no-build-isolation --no-compile` ile kuruldu.
Existing runtime'a HYSo/dependency kurulmadı veya upgrade yapılmadı.

Mevcut packaging testindeki SMOKE gövdesi iki installed target'ta repo dışında
çalıştı; başta exact CPU Torch sürümü/build assert edildi. **2/2 installed
smoke PASS**, `isolated_activation=true`, `cuda_initialized=false`.
Model/API/Trainer/checkpoint replay/state-RNG, performance guides,
benchmark exports/comparisons, Manager reopen ve AdamW CPU isolated saved
activation bu smoke kapsamındadır. Sonuçlar `smoke-wheel.json` ve
`smoke-sdist.json`. Minimum smoke PASS, full minimum-runtime regression'ın
PASS olduğu anlamına gelmez; bölüm7'deki hatalar ayrı tutulur.

## 9. Packaging / source equality

`test_packaging` canonical configuration ve archive content/metadata testleri:
**2 PASS**,9.928s;2 installed parametrization deselect, onların SMOKE
workload'ları yukarıda doğrudan CPU driver ile çalıştırıldı. Bu2 doğrudan
smoke pytest PASS sayısına eklenmedi. Mevcut build/wheel araçları kullanıldı;
missing build-tool path ilk hazırlıkta düzeltildi, dependency indirilmedi.

47/47 Python modülü için:
**checkout == wheel == sdist == installed-wheel == installed-sdist byte eşliği**.
Python3.10 AST parse PASS. Wheel target'taki **47/47 modül import/origin**
kontrolü PASS; bütün module yolları installed target altında.
Arşivlerde docs/tests/sentinel/checkpoint/db/pyc/build çıktıları dışlandı;
metadata/dependency/Python floor kontrolleri PASS.

- Wheel SHA256 `1ee0a614562802959a684c837a1972e231c8f086ab5f009a7fd6f99abfc920cf`.
- Sdist SHA256 `1fade631440efc884962008d48f9e9de51db355d74f9f86f7175a3b9988e784d`.
- Existing minimum environment'ın 70 METADATA/RECORD dosyasının önce/sonra SHA256 eşliği PASS; bu bütün binary belleğinin attestation'ı olarak sunulmaz.

Arşiv byte eşliği, bütün modüllerin bütün runtime/native yollarının functional
acceptance'ı değildir. Özellikle mevcut-source CUDA release acceptance yok.

## 10. Eksik native release gate'leri ve kararlar

| Gate | Güncel durum |
|---|---|
| Tarihsel A/B |A29 ve frozen B21 sonuçları korunur; yeni final CUDA acceptance değildir |
| Güncel CPU isolation/activation |58 PASS; gözlenen Windows/minimum CPU kapsamı doğrulandı |
| Full CPU legacy |FAIL;2 harness failure targeted çözülmüş,9 failure açık |
| Minimum installed / package bytes |PASS; kapsamı bölüm8–9 |
| B4 |**OPEN / UNRESOLVED / NO-GO**; eski0xC0000409 failure geçerli |
| P23 |**PARTIAL**; CUDA saved-activation disabled; implemented değil |
| B10 |**PARTIAL**, full-scope GO yok; P21 worker service trace açığı da korunur |
| R1 |Eski native crash / **NO-GO** silinmedi |
| Commit/stable release |**NO-GO**; tüm mevcut-source functional/native gate'ler kapanmadı |

Final-source ve installed CUDA acceptance, saved-activation Conv/tied/early/
resume/GC/exception kapsamı, güvenli hook/reset etkileşimi ve ilgili mevcut
native regression gate'leri eksik. Tek cold forensic PASS bunların yerine
geçmez. Production guard'ını kaldırma veya genel CUDA matrix yetkisi yok.

## 11. Dosyaların commit'e hazır olma durumu

Yeni audit ve mevcut üç tarihsel rapor incelemeye hazır kanıt belgeleridir;
önceki15 görev dosyası byte'ları korundu. Beş production P23 dosyası ile CPU
test/fixture'ları gözlenen CPU kapsamı ve packaging bakımından doğrulandı,
ancak bu **commit/release GO** değildir. Genel legacy gate9 açık failure ve
native release açıkları nedeniyle bütün task değişiklik seti commit'e hazır
ilan edilmez. CUDA validation script'i bu görevde çalıştırılmadı. Önceki
guide/roadmap'in pending tarihsel ifadeleri yeniden yazılmadı; bu rapor yeni
sonuçların tarihli devam kaydıdır. Kullanıcının gradient dosyası task commit
kapsamına alınmamalı; dosyaya dokunulmadı. Index boş, stage/commit/push yok.

## 12. Sonraki CPU ve native kabul koşulları

Önce mevcut failure kanıtlarıyla version-aware test varsayımları ve partial
profiler → metric consumer davranışı karara bağlanmalı; source/test değişirse
yeni manifestten uygun hedefli doğrulama ve temiz full CPU regression gerekir.
Bu rapor legacy hataları gizlemek için skip/xfail veya dependency değişikliği
getirmedi.

Yeni native test için **ayrı açık kullanıcı onayı**, önceden yazılmış tek
hipotez/karşılaştırma ve execution bütçesi gerekir. Örneğin aynı sealed
post-stop source üzerinde hook-off/reset-on tek control, ayrıca onaylanırsa
bir değişkeni ayırabilir; eski unverified C'nin exact replay'i sayılamaz.
Bu görev o workload'u veya prefix tekrarını başlatmadı.

Onaylı sonraki planın minimum güvenlik/kabul koşulları:

1. Checkout guard korunur; bypass yalnız bağımsız sealed forensic kopyada,
   manifest/run-ID/tam recipe eşleşmesiyle. Parent/child pre/post source,
   interpreter/Torch/runtime/DLL identity doğrulaması; mismatch admission reject.
2. Çakışan yoğun workload yok; Job admission importtan önce. Child60s,
   outer90s;4GiB/process,8GiB/childJob,4 child process; outer12GiB/8 process;
   allocator fraction.15; bounded stdout/stderr/result ve process-tree kill-on-close.
3. Tensor/graph referansı tutmayan fsync metadata: callback thread/PID,
   epoch/microbatch/window/attempt,last durable phase,pack/unpack/release,
   reset/stream/cleanup entry-exit. Copy/empty_cache/GC/workaround/extra step yok.
4. WER/dump toplama zamanı worker timeout'tan ayrı kaydedilir; global registry/
   dump policy değişikliği veya dump upload yok. Gerekirse ayrıca onay alınır.
5. Native crash'te derhal stop; exit/Windows evidence/source snapshot ve Job
   active0/PID cleanup doğrulanır; otomatik retry veya başka case yok.
6. PASS yalnız tam ledger/identity, önceden belirlenmiş state/RNG/fixture
   karşılaştırması, allocator/reset invariants ve process cleanup ile dar
   scenario acceptance olur. Tek PASS root cause resolved/P23 implemented/
   B4 release GO sayılmaz; final-source/installed kapsam ve native gate'ler
   ayrıca review/acceptance gerektirir.

## 13. Koruma ve kapanış

Başlangıçtaki18 dirty dosya ve243 manifest girdisi değişmedi. Yalnız bu audit
repository'ye eklendi. Korunan SHA256 beklenenle aynı; production CUDA guard
etkin; yeni CUDA workload sayısı **0**. Son inventory/hash/index/cleanup
kanıtları `final-protection.json` ve evidence manifestinde tutulur.
