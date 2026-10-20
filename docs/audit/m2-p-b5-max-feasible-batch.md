# M2-P-B5 — Maximum feasible batch size / P17

2026-10-20, Europe/Istanbul. Başlangıç branch `feature/hyso-models-next`, HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`; index boş. Önceden uncommitted
B4/R1/R2/R3 dosyaları ve kullanıcı gradient farkı korunmuştur. Evidence **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb5-20261020`.

## 1. Ön koşul ve açık güvenilirlik riski

Yeni feature kodundan önce [B4 conditional acceptance](m2-p-b4-conditional-acceptance.md)
oluşturuldu. Kullanıcı bilinmeyen native crash riskini açıkça kabul ederek yalnız
development continuation yetkisi verdi. **B4: CONDITIONAL GO — development continuation only.**
Eski strict gate PASS değildir; public/PyPI stable release onayı yoktur.

Tarihsel P06/P07/P08 scoped PASS, CPU 2313 PASS/1 expected SKIP ve R1 native 196
distinct PASS/198 successful executions korunur; B5'in yeni test sayıları değildir.
Orijinal 0xC0000409 reason2/nvcuda64.dll olayı geçerlidir. SiLU forward → Torch
CUDA dispatch → NVIDIA driver stack'i corruption origin veya kusurlu vendor
kanıtı değildir. **Root cause UNRESOLVED; reliability risk OPEN.** Production
crash fix'i, dependency/driver güncellemesi veya workaround uygulanmadı.

## 2. API ve kapsam

Additive `hyso.performance.find_max_batch_size(recipe, *, protocol)`, frozen
`BatchSearchProtocol` ve independent schema1 `CapacityResult` eklendi.
Recipe mevcut `ColdStartRecipe`: plain importable CPU FP32 model factory, seed,
exact NCHW ones input, optional SHA-verified weights_only CPU state_dict load.
Canlı model/Tensor/stream/handle parent'tan aktarılmaz. Bu, caller'ın mevcut model
ağırlıklarını otomatik kullanan API değildir; böyle bir kullanım explicit artifact
gerektirir. Trusted factory external side effect/subprocess üretmemelidir.

Scope Windows CPU FP32/FP64 ve CUDA FP32/AMP FP16 eager single-Tensor inference.
Diğer precision/model/shape/profiles explicit rejection/reason üretir; CPU
fallback yoktur. Training capacity, accuracy, global/physical capacity ve uzun
vadeli stability iddiası yoktur. Linux resource guard unsupported'dır.

Explicit unique finite candidates (1..512 nokta), user maximum, caller order ve
`exhaust_candidates`/`first_infeasible` stopping desteklenir. Adaptive/binary
search bu optional ilk-sürüm kapsamına dahil değildir. İkinci timing framework
yoktur: worker mevcut B4 `sweeps._point` → owned copy/precision hazırlığı →
legacy BenchmarkRunner warmup/repeat/real forward/timer → P09 ayrı memory probe
yolunu kullanır. B1–B4 ve Benchmark schema1/checkpointv1 değişmez.

## 3. Arama kanıtı ve bellek sınırları

Her candidate fresh interpreter ve CUDA için fresh context kullanır. CPU hard
private-commit bütçesi, Torch importundan önce stdlib ctypes Windows Job Object
ile process/job memory limit olarak uygulanır. Active-process limit1 trusted
factory'nin child process açmasını da sınırlar; guard başarısızsa işlem durur.
Input tek-dense-tensor byte cap allocation öncesi kontrol edilir. Total worker
timeout startup/import dahil bounded'dır; Windows venv launcher ile actual worker
PID ayrıdır. Mevcut B3 owned process-tree termination yardımcısı yeniden kullanılır.

CUDA `set_per_process_memory_fraction` yalnız child'a uygulanır; driver/context ve
başka process VRAM'ini sınırlamaz, fiziksel VRAM kapasitesi değildir. Parent P09
counter reset sözleşmesi değişmez. P10 RSS/HWM queries ölçüm öncesi/sonrası ayrı
gözlemdir; exact inference-window CPU peak değildir. P09 single-forward allocator
peak, process RSS, private commit bütçesi ve physical VRAM birbirine eşitlenmez.

Typed CUDA OOM ve CPU MemoryError, normal exit 0 receipt ile `infeasible_oom`
olabilir. Unexpected exit code `crashed/unknown`, timeout `timed_out/unknown`,
ordinary exception `failed/unknown`, input policy rejection `resource_limit/unknown`
olur. Fatal/unknown outcome admission'ı durdurur, sonraki adaylar not_run kalır.
OOM sonrası aynı context kullanılmaz; exhaust policy yalnız yeni worker açar.
Global reset/cache flush/GC veya kontrolsüz retry yoktur.

`maximum_observed_feasible_batch` completed gerçek forward'lardan gelir.
`maximum_proven_within_range` yalnız 1..maximum_batch tüm integer değerleri
classified olarak gözlenirse yayımlanır. Sparse powers-of-two hiçbir ara batch'i
kanıtlamaz; monotonluk varsayılmaz. Aborted arama successful maximum yayımlamaz;
önceki başarılar yalnız `prior_observed_feasible_batch` ve raw satırlarda kalır.
Kanıt recipe/seed/bütçe/çalışma anı kapsamındadır, evrensel maksimum değildir.

## 4. Result ve Experiment Manager

Format `hyso.performance.capacity`, schema_version=1, feature_ids=["P17"], result UUID,
protocol/recipe, candidate UUID/PID/shape/requested+actual cardinality, precision,
backend/device, forward completion, independent W/N/R, execution wall duration,
latency, memory scopes, embedded PrecisionResult/model+input hashes ve runtime
provenance saklanır. Eksik/failure ölçümler numeric0 yapılmaz. Frozen/plain/finite
validation ve mevcut atomic/no-clobber JSON/CSV export kullanılır.

[Rehber](../guides/performance-max-batch.md) CPU/CUDA örneklerini ve ölçüm sonrası
mevcut schema2 benchmark run + generic metrics + copied artifacts → SHA verify →
finish/close/reopen akışını içerir. Yeni table/migration/run kind eklenmez;
SQLite/network/artifact write ölçüm penceresinde yoktur.

## 5. Doğrulama

E/initial.json başlangıç
hash/status envanteri; E/gate-source-freeze.json final production/tests/guide
hash'leridir. İlk B5 CPU koşusu 23 PASS/2 test-expectation hatası verdi: embedded
PrecisionResult'ın yanlışlıkla SweepResult sanılması ve mevcut EManager create_run
imzasının yanlış kullanılması. Test beklentileri düzeltildi; bu başarısızlıklar
E/b5-first.* içinde korundu. Sonraki bağımsız B5 koşusu32 PASS verdi. Daha sonra
timeout kanıtına atomic runtime-prepared receipt ve actual-PID cleanup kontrolleri
eklendi; final gate bu son kaynakları ayrıca doğrular.

Son production kaynaklarında hedefli CPU gate **1595 PASS, 0 SKIP, 0 FAIL**;
570 mevcut API/deprecation warning, 452.93 saniye. B5 testleri önce yürütüldü,
ardından aşağıdaki seçili regresyonlar çalıştı; `-x` ilk başarısızlıkta durmayı
sağladı. E/regression-final.xml ve regression-counts.json exact sayımlardır.
İlk regression invocation iki yanlış test dosyası adı nedeniyle test toplamadan
durdu; düzeltilmiş selection E/regression-selection.json içinde korunur.

| Final CPU grup | PASS |
|---|---:|
| Yeni B5 CPU | 32 |
| B4 sweeps | 55 |
| B1 latency/throughput | 122 |
| B2 RAM/footprint | 83 |
| B3 pipeline/cold | 58 |
| Benchmark/P09 protocol/state/result/export/comparison | 310 |
| Trainer update/lifecycle/checkpoint/save/restore/replay/transaction | 612 |
| Experiment Manager | 259 |
| Public API/model-state/published compatibility | 60 |
| Packaging | 4 |
| Toplam | 1595 |

Yeni 32 test gerçek CPU FP32/FP64 forward/cardinality/W/N/R, nonmonotonic
1PASS/2controlled-MemoryError/3PASS, caller order/cap/gaps/partial stop, ordinary
exception, controlled os._exit(73), runtime-prepared timeout, actual worker ve
launcher PID ölüm kontrolü, input-budget rejection, invalid model/output,
caller mode/parameter/gradient/RNG/CUDA-init preservation, real Job Object
allocation rejection, SHA load/rejection, immutable finite serialization ve
EManager metric/artifact SHA/reopen yollarını kapsar. Kontrollü MemoryError
fixture'ı fiziksel RAM exhaustion deneyi diye sunulmaz; ayrıca stdlib hard-guard
testi 256 MiB private-commit limitiyle 300 MiB allocation'ı gerçek MemoryError ile
reddetmiştir. Yeni skip/xfail/tolerans gevşetmesi yoktur.

Final wheel ve sdist 31 production modülünü tam envanterle içerir; her iki
archive'daki tüm `.py` byte'ları final kaynakla aynıdır. Wheel/sdist installed
CPU smoke, eski API/replay/rehberler ve yeni CPU kapasite rehberini çalıştırdı.
Artefact kayıtları E/archive-integrity.json:

- Wheel 136997 byte, SHA256
  `7d7ad80f52373784da3ec014d4f1b85f94f86e394877418bb17cdebeea2c495a`.
- Sdist 125707 byte, SHA256
  `bf31a3573823a37340ab94b46438c83db0d7b96ba4043584e4e3c974a12e72ae`.

Bu 1595 seçili test, tarihsel full CPU 2313 sonucu veya bütün repository için yeni
full gate değildir; native 196 da otomatik yeni PASS sayılmaz.

Minimum installed smoke da aynı final wheel ile **PASS**: mevcut Python 3.10.18,
Torch 2.2.2+cpu/torchvision 0.17.2+cpu. Yeni CPU rehberi 1/2/3 real forward ve
maximum_proven_within_range 3, eski inference/memory/pipeline/sweep/EManager ve
Trainer replay smoke birlikte çalıştı. Parent CUDA initialized=False kaldı;
worker package_source kurulu target altında doğrulandı. Tüm31 installed source
byte'ı checkout/final archives ile aynıdır. Minimum environment METADATA/RECORD
hash envanteri önce/sonra eşittir; dependency kurulmadı/güncellenmedi.
E/minimum-smoke.json, minimum-environment.json, installed-integrity.json ve
installed-commands.json bu sonuçların kanıtıdır. Local HYSo wheel yalnız TEMP
target'a `--no-index --no-deps --no-compile` ile kurulmuştur.

## 6. Final installed-wheel native CUDA kanıtı

Windows, NVIDIA GeForce RTX 3060 Laptop GPU 6GB, capability 8.6, driver 591.74;
Python 3.10.18, Torch 2.5.1+cu121, CUDA build 12.1. Torch total_memory 6441926656 byte,
nvidia-smi 6144 MiB; ölçüm anındaki free memory ayrı ortam gözlemidir.
E/installed-cuda-b5.json final31-modüllü wheel'den **5 native scenario PASS**
kanıtıdır. Her completed satır CUDA input/output, actual batch, event timing ve
package_source installed target doğrulamasını geçti. Precision FP32 outputFP32,
AMP FP16 outputFP16; hidden CPU fallback yoktur.

| Native B5 senaryo | Gözlem |
|---|---|
| FP32 | batch 1/2/4 feasible, observed max 4, proven-within-range null |
| AMP FP16 | batch 1/2/4 feasible, actual FP16 output, observed max 4 |
| Controlled allocator budget | caller order 1/2/4/8/32/16; 32 typed CUDA OOM, diğerleri feasible; observed max 16, proven null |
| Runtime-prepared CUDA timeout |15s cap; worker runtime_ready=True; timed_out/unknown, kalan2 aday not_run; successful max null |
| Fresh after OOM/timeout | yeni aramada1/2/4 feasible |

OOM testi Conv2d 16→16/kernel1, N×16×128×128 input ve **yalnız worker allocator
fraction .02** ile yürüdü. Error 122.87 MiB allowed ve yaklaşık4.87GiB device free
bildirdi; fiziksel VRAM exhaustion değildir. Batch 16 ayrı P09 probe peak 50332672
byte gözledi; bu bütün candidate yaşam süresinin peak'i değildir. Search feasibility
B4 hazırlık/copy/preflight/timing/probe protokolünün tamamını bitirmeyi gerektirir;
yalnız minimal forward kapasitesine eşitlenmez. Batch 17–31 denenmedi; max 16 bu
bütçe/protokolde yalnız gözlenmiş feasible değerdir, fiziksel/global maximum değildir.

Bu5 senaryoda 18 requested satır/16 worker admission: 14 feasible, 1 expected OOM,
1 intentionally terminated timeout ve 2 not_run. OOM receipt exit 0'dır; timeout'un
forced exit 1 değeri native crash diye sayılmaz. Tüm actual-worker/launcher PID
ölüm kontrolleri, parent non-default stream/device, parameter/grad/mode,
Python/NumPy/CPU+CUDA RNG, allocated/reserved/peak counter eşitliği geçti.
OOM 32'den sonra 16 yeni context'te başarılıdır; hasarlı context reuse edilmedi.
Beklenmeyen native crash gözlenmedi; bu orijinal B4 crash'in fix kanıtı değildir.

E/cuda-b5-source.json içindeki önceki B5 revision'ının 5 CUDA smoke'u yardımcı
kanıttır; atomic runtime-ready receipt ve güçlendirilmiş timeout/PID kontrolünden
öncedir. Final wheel 5 senaryosuna ek distinct başarı diye sayılmaz.

Final installed-wheel regresyonları da PASS:

| Native grup | Başarılı senaryo |
|---|---:|
| Yeni B5 | 5 |
| Benchmark/P09 | 45 |
| B1 | 15 |
| B3 pipeline/cold | 8 |
| B4 sweeps/precision | 11 |
| Experiment Manager CUDA integration | 2 |
| Toplam CUDA | 86 |
| Ayrıca native Windows P10 | 3 |

CUDA rehberinin ayrı block'u da final installed target'ta gerçek AMP FP16
1/2/4 forward ile çalıştı (E/installed-cuda-guide.json); ek 86 distinct scenario
sayısına eklenmedi. CPU rehberi wheel/sdist/minimum smoke içinde çalıştırılmıştır.
Testleri geçen native process'ler tekrar çalıştırılmadı. EManager script'i
`--installed-target` desteklemediği için ilk invocation argparse exit 2 ile CUDA
koduna girmeden durdu; hata E/installed-commands.json içinde korunur. TEMP'teki
wrapper installed hyso/benchmark/experiments'i önce import edip değişmemiş test
script'ini gerçek argümanlarıyla yürüttü; tüm loaded hyso modül yollarının final
installed target altında olduğunu ayrıca doğruladı. Bu CLI invocation hatası
native crash, OOM veya production regresyon değildir.
E/installed-cuda-experiment/report.json iki gerçek scenario PASS kanıtıdır;
kalan native çağrılar tamamlandı.

## 7. Koruma ve final karar

Kullanıcı gradient dosyası başlangıç/final SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Eski B4/R1/R2/R3 audit hash'leri ve B4 sweep source/test/guide dosyaları aynıdır.
Önceden var olan dosyalarda yalnız performance exports/result dispatch,
packaging inventory/smoke ve backlog additive genişletildi; Trainer, checkpoint,
Benchmark/P09 implementation, EManager storage ve B1–B4 ölçüm çekirdekleri değişmedi.
Başlangıç branch/HEAD/index korunur; commit/push/checkout/reset/rebase/merge yoktur.
Dependency/driver/CUDA/Torch/model/weights/dataset indirmesi/güncellemesi yapılmadı;
dump okunmadı/kopyalanmadı/yüklenmedi. Production inventory 31 modüldür.

**P17 PASS — yalnız doğrulanan Windows explicit finite inference scope.**
**B5: PASS within verified scope, B4 risk inherited.**
Bu başarı stable/public/PyPI release onayı değildir. B4 root cause **UNRESOLVED**,
risk **OPEN** kalır. Orijinal native failure geçerlidir; yeni koşularda görülmemesi
fix kanıtı değildir. Ayrı stable release gate zorunludur. Tekrarlayan native crash,
reproduced regression veya model/state corruption development gate'i durdurur.

B6 için yalnız koşullu development continuation bakımından kapı açıktır;
bu görevde B6 veya sonraki özellikler uygulanmadı. Mevcut doğrulanmış P17 scope
için ilave full CPU/native 196 replay tekrarı gerekli değildir. Scope genişletilirse
(adaptive search, Linux guard, başka precision/model/training profili) ayrı kanıt
gerekir. B4 riskinin kapanması için R3'teki offline exact-build symbol/map eşlemesi
veya ayrıca tanımlanmış reliability/release-gate çalışması gerekir; burada fix
veya vendor kusuru ilan edilmez. E/verification-summary.json final hashes/counts
ve koruma kontrollerini bir araya getirir.
