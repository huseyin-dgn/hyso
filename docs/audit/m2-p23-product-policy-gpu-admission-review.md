# HYSo P23/B4 — ürün politikası ve GPU admission incelemesi

2026-10-21, Europe/Istanbul. **Yeni native CUDA workload0.**
Normal Trainer/Benchmark için utilization yüzdesiyle eğitim reddi bulunmadı.
Forensic r2'deki üç örneğin her biri<%35 kuralı, NVIDIA zorunlu güvenlik
sınırı değildir. Eski r2 korunarak ayrı v3 politika/harness ve mühürlü r3
adayı hazırlandı; **NOT APPROVED / NOT EXECUTED**.

Yeni stdlib politika testleri **14 PASS**. Mevcut P23 minimum-CPU suite'i
**61 PASS /1 FAIL**; temiz genel CPU PASS değildir. Failure timeout fixture'ın
forward-start kanıtında; timeout/Job cleanup assertion'ları geçti. Test, limit
ve sonuç değiştirilmedi; retry yapılmadı. **B4 OPEN/UNRESOLVED, P23 PARTIAL,
B10 PARTIAL, R1 NO-GO** korunur. Commit/release GO yoktur.

## 1. Başlangıç ve dosya koruması

Branch `feature/hyso-models-next`; beklenen HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş.
Başlangıç **29 dirty dosya**, önceki22:57 tesliminin envanteriyle path/status/
SHA256 aynı: fark0. Sayı sapması yoktur. `start-protection.json` tam envanteri
saklar. Bu görevin sonunda29 başlangıç dosyası byte/status eşit; yalnız5 yeni
dosya, toplam34 dirty. Mevcut audit raporları da değiştirilmedi.

Korunan gradient SHA256 başlangıç/son:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Production guard dosyası `hyso/performance/isolated_training_memory.py` SHA256:
`9bacc67691ab032dfa9b6202f5ea3c4bd8b78e0653aca40d53b2fa6e86f85a0d`.
Saved activation SHA256:
`0b3519a32a9a3fc015e003e8ced99167d5a41c1037ce755b165fa9a08cd2d654`.
Bu dosyalara veya diğer mevcut P23/CPU düzeltmelerine yazılmadı.

Evidence **E**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-product-policy-20261021T232140`.
Reset/restore/clean/rebase/merge/stage/commit/push yok; dependency/driver/Torch
kurulumu veya güncellemesi yok. Kullanıcı uygulaması kapatılmadı veya process'i
sonlandırılmadı. CPU negative testlerinin kendi Job tree cleanup'ı test kapsamıdır.

## 2. İncelenen kaynak, kapsam ve tarihsel kanıt

README, sabit97 başlıklı architecture roadmap,35 ID'li performance backlog;
`models/{bcam,WithSE,fit,blocks,attention,modular,__init__}.py`, Trainer lifecycle,
BenchmarkRunner/result ve ExperimentManager persistence/metric/artifact yolları
incelendi. `source-scope-audit.json` üretim modüllerinin AST class/function
envanterini, SHA256'larını ve35 backlog satırını saklar.

Performans sayımı makineyle doğrulandı: **17 implemented /7 partial /11 planned**.
Yeni tasarım tablosu her mevcut ID/status'u bir kez içerir. Yerel uncommitted
P23 kodu completion/commit kabulü sayılmadı;97 mimari başlık35 performans
özelliğiyle birleştirilmedi. YOLO/detection/segmentation adapter'ı ve `Hyso`
facade yoktur. Custom nn.Module, yalnız her API'nin desteklediği profile göre
admit edilir; generic Trainer multiclass/binary classification kapsamındadır.

Özellikle korunan kanıtlar:

- [P23 R1 isolation/activation](m2-p-b10-p23-isolation-activation.md),
  [native inceleme](m2-p-b4-p23-native-crash-investigation.md),
  [tek başarılı hook-on diagnostic](m2-p-b4-p23-single-forensic-reproduction.md).
- [CPU release preflight](m2-p-b10-p23-cpu-release-preflight.md),
  [minimum CPU recovery](m2-p-b10-p23-minimum-cpu-regression-recovery.md),
  [version-counter blocker düzeltmesi](m2-p-b4-p23-blocker-resolution.md).
- [RAM rejection](m2-p-b4-p23-hook-off-reset-on-control.md),
  [GPU31/35/71 rejection](m2-p-b4-p23-hook-off-reset-on-control-continuation.md),
  [GPU21/32/35 rejection](m2-p-b4-p23-hook-off-reset-on-control-preflight-2257.md).
- [B6 profiling](m2-p-b6-flops-layer-operator-profiling.md),
  [B7 transfer/telemetry](m2-p-b7-transfer-utilization.md),
  [B8 Roofline capability](m2-p-b8-roofline-capability-gate.md).

Tarihsel2825 minimum legacy,874 development CPU, packaging4 ve sonraki
blocker62/108/13 sonuçları yeni test toplamına eklenmedi. Eski C loaded-source
identity **unverified_source_identity** kalır. İki eski dump'ın benzer
`0xC0000409 / FAST_FAIL_STACK_COOKIE_CHECK_FAILURE` forward/library-load imzası
corruption origin'i veya driver kusurunu belirlemez. CPU version-counter
noninterference düzeltmesi CUDA crash çözümü olarak sunulmadı.

## 3. Kod düzeyinde GPU/bütçe/admission envanteri

|Kod yolu|Gerçek davranış|Bağlam ve gerekçe|
|---|---|---|
|`training/trainer.py::__init__`, `_validate_setup`, `fit`|Device availability/index, model/optimizer/precision/loader/lifecycle admission; utilization sorgusu/gate'i ve allocator fraction setter yok|Normal eğitim; caller modelini taşımaz; user config korunur|
|`models/bcam.py`, `WithSE.py`, `fit.py`|Legacy model/training/device tercihleri; utilization<35 gate yok|Bazı legacy default device seçimleri availability'ye bağlı; bu yeni utilization politikası değildir|
|`benchmark/runner.py::BenchmarkRunner`, `_cuda_memory_probe`|Explicit precision/device/state/input admission; ayrı opt-in peak reset; utilization hard gate/fraction setter yok|Benchmark güvenilirliği ve allocator peak ownership; gerçek OOM ayrı|
|`performance/utilization.py::measure_utilization`, `_utilization_metrics`|Bounded CPU/system ve optional physical-device GPU telemetry; percent'ler ölçüm alanı|Training başlatmaz/reddetmez; GPU process/Tensor Core attribution unsupported|
|`performance/_telemetry.py::gpu_observe`|Installed CLI ile read-only, timeout'lu sorgu; UUID/domain/timestamps, N/A→null|Sensor timeout/unavailable, workload admission değildir; unexpected native sensor crash stop'tur|
|`performance/training_memory.py::TrainingMemoryProtocol`, recorder|In-process never-reset logical/RSS/allocator gözlemi; fraction setter yok|Normal kullanıcı eğitimini bütçelendirmez; activation/window/physical sınırları explicit|
|`performance/isolated_training_memory.py:75–113`, `_run_recipe:204–216`|Opt-in trusted recipe/Windows worker; `.15` configurable fraction, Job/private-commit/timeouts; CUDA activation guard setup'tan önce|Kullanıcı modelini/Trainer'ı parent'tan serialize etmez; owned kaynak bütçesi; genel normal-fit politikası değil|
|`performance/_training_process.py::_Job`, `_launch` ve `_training_memory_worker.py`|Job assignment/admission Torch/factory importundan önce; capped log/result, timeout/tree termination, active0 receipt|Owned probe containment; RSS/physical VRAM sandbox'ı değildir|
|`performance/capacity.py::BatchSearchProtocol`, `_batch_worker.py:81`|P17 `.1` configurable allocator fraction, explicit candidate/input/commit/time budgets|Maximum batch, bu bütçe içindeki gözlenen/kanıtlanan kapasite; fiziksel/global maximum değil|
|`performance/profiling.py::ProfilingProtocol`, `_profile_worker.py:119`|P12–14 `.1` configurable fraction; input/commit/trace/event/timeout sınırları|Owned profil capture; GPU activity capability'si ayrı ve mevcut build'de eksik|
|`performance/transfers.py::TransferProtocol`, `_transfer_worker.py:36`|P15 `.1` configurable fraction; owned-buffer/input/commit/timeout sınırları|Transfer probe; normal eğitim veya tüm uygulama için geçerli limit değil|
|`performance/cold.py`, `sweeps.py`|Explicit trial/timeout/point budgets, error/OOM stop; utilization hard gate yok|Cold/sweep API admission'ı yüzdeye göre sessiz model mutation yapmaz|
|`tests/cuda_isolated_training_memory_validation.py:112`|Test harness `.15` setter|Production varsayılanının tek kaynağı değildir; bu görevde çalıştırılmadı|
|r2 `source/forensic_controller.py`|Üç örneğin tamamı<35 + GPUfree>=2048 MiB + RAM>=4 GiB + disk>=12 GiB + workload0 +<=30s evidence|Yalnız tarihsel B4 forensic deney şartı; normal API'ye import edilmez|

Exact allocator setter/default path/line envanteri
`snapshot-static-verification.json::allocator_code_inventory` içindedir.
Repository genel aramasında normal Trainer/Benchmark'i `%35` ile engelleyen
bir yol bulunmadı. “HYSo hiçbir kaynak sınırı uygulamıyor” sonucu da yanlıştır:
explicit isolated/profiling/capacity/transfer probe bütçeleri gerçek hard
sınırlardır; başlangıç değerlerinin kullanıcıya açıklanması gerekir.

## 4. %35 teknik değerlendirmesi ve dört bağlam

NVIDIA utilization.gpu'yu örnekleme döneminde GPU'da kernel yürütülen zamanın
oranı olarak tanımlar. Bu, boş VRAM veya native crash güvenliği ölçüsü değildir;
bu tanımda %35 zorunlu güvenlik eşiği yoktur.
[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
%35'in bu stack-cookie fail-fast'i önlediğine dair yerel kontrollü kanıt da yok.
Bu nedenle eski kuralın **yerel konservatif deney tercihi** olduğu sonucuna
varıldı; bilimsel/vendor güvenlik garantisi diye sunulamaz.

Normal eğitimde %35/%80/%100 meşguliyet tek başına red sebebi olmamalıdır.
P16 telemetri olarak kalır. Opt-in isolated API'de kullanıcı kaynak bütçesi
ayrı seçilir. Forensic diagnostic'te GPUfree/RAM/Job/timeout/identity ve workload
incelemesi gerçek admission koşullarıdır. Benchmark karşılaştırmasında ise
shared/background activity, zamanların adilliğini bozabilecek confounder'dır;
sonuç unqualified/insufficient olabilir, kullanıcının eğitimi yasaklanmaz.

Meşgul GPU'daki diagnostic yalnız bu ortamda dar native gözlemdir; düşük busy
bile exclusivity kanıtı değildir. Yeni v3 kararında performance-comparison
alanı **null/unsupported** kalır. Bu sonuçlardan speedup veya causality çıkarılmaz.
Free VRAM minimumunun da yerel headroom koşulu olduğu, gelecekteki allocation
başarısını veya native safety'yi garanti etmediği açıklanır. RAM/Job/timeout ve
memory bütçeleri bu görevde keyfî biçimde kaldırılmadı.

## 5. Allocator fraction=.15 incelemesi ve uyumluluk kararı

P23 isolated production protocol varsayılanı `.15`; yalnız forensic değildir.
Validation numeric finite `(0,1]` kabul eder; bool/None/0/>1 geçmez. Kullanıcı
explicit `IsolatedTrainingMemoryProtocol(cuda_memory_fraction=...)` ile
değiştirebilir; recipe/memory_protocol device eşleşmesi gerekir. Setter yalnız
admitted CUDA child'da model factory'den önce çalışır. Parent/in-process Trainer'a
fraction uygulanmaz. Bütçe result protocol/provenance'ında kaydedilir.

PyTorch2.5.1 kaynağında fraction total visible memory üzerinden allocator'a
sınır koyar; sınırı aşmak allocator OOM oluşturabilir. Bu, system/device
memory reservation veya bütün CUDA allocations için fiziksel cap değildir.
[PyTorch v2.5.1 memory.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py).
Örnek aritmetik:6144 MiB×.15=921.6 MiB. Bu bir ölçüm veya bu workload'un
gerektirdiği bellek tahmini değildir; küçük VRAM'de fiziksel boş bellek varken
budget OOM üretme olasılığını açıklar. Global user workload'a uygun universal
fraction olduğuna dair kanıt bulunmadı.

Bu görev production varsayılanını/signature/result schema'sını değiştirmedi.
`.1` kullanan diğer owned protocol'lar da değişmedi. Güvenli sonraki API işi:
explicit budget'ın örneklerde/raporda görünürlüğü, kullanıcı seçim provenance'ı
ve budget-vs-device OOM nedeninin kanıt varsa ayrılması. Varsayılanı artırmak,
None eklemek veya seçim zorunluluğu getirmek ayrı compatibility/güvenlik
review'u gerektirir: mevcut serialized protocol'lar, validator'lar, installed
tests ve kullanıcı bütçe beklentileri etkilenir. Otomatik fraction artırımı,
batch düşürme veya OOM retry önerilmedi/uygulanmadı.

## 6. Gerçek yeni uygulama ve bağımsız snapshot

Yeni repository dosyaları:

1. `tools/forensics/p23_preflight.py`: stdlib-only evaluator ve bounded JSON CLI;
   sensor/torch/HYSo importu, training veya process launch yolu yok.
2. `tools/forensics/prepare_p23_policy_snapshot.py`: exact r2 manifest/runtime
   doğrulaması ardından **yalnız ayrı, mevcut olmayan output** içine kaynak
   kopyası; yeni run-ID/manifest/seal üretir. Workload çalıştırmaz.
3. `tests/forensics/test_p23_preflight.py`: offline unittest suite; Torch
   conftest'ini çalıştırmayan stdlib discovery ile doğrulandı.
4. [Ürün/öneri/rapor sözleşmesi](../design/hyso-analysis-optimization-contract.md).
5. Bu audit. Mevcut29 dosya değiştirilmedi.

**Eski r2 T**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-b4-blocker-20261021T173434/proposed-hook-off-control-r2`.
Manifest SHA256 **aynı**:
`2be2ebe1562bfc0863b16dc6d1bb093f91f32f1dd9ab8b456ed94387cedbd78d`.
251 source+26 runtime başlangıçta ve sonda hash eşit. Eski approval/preflight/
attempt marker'ları yok; worker/events file0. Eski execution hakkı bu görevde
kullanılmadı. Eski onay yeni kimliğe taşınmadı.

**Yeni r3**: `E/proposed-hook-off-control-r3`.
Run ID `f6ff1fa5-60d4-41b0-963f-d6459856c479`.
Manifest SHA256:
`660a53dbdde6a21901bf6ad2743a3fad9bb51bad57ebd25a5a5dee46ad2444b8`.
252 source+26 runtime; source/manifest/seal read-only.
Yalnız controller gate/proposal/approval açıklaması değişti ve policy module
eklendi; diğer248 kaynak dosyası eski snapshot ile byte eşit. Parent,
forensic_support, worker, `_Job` ve recipe/limits değişmedi. Controller farkı
tersine uygulanınca r2 controller string'i birebir elde edildi.

Yeni controller manifest/policy module hash'ini importtan önce bağlar;
v3 evaluator kararından sonra mevcut full source/runtime verify ve Job readback
yoluna gider. Approval/run-ID/manifest eşleşmesi ve exclusive attempt/child
marker'ları korunur. Policy decision yalnız plain metadata olarak yayımlanır.
Identity doğrulaması disk/pre-post import kapsamıdır; tüm native belleğin
attestation'ı veya adversarial sandbox olduğu iddia edilmez.

V3 evidence sözleşmesi (`p23_preflight.evaluate`):

- Policy/run/manifest identity; collection başlangıcından admission'a<=30s;
  future/reversed/stale time red. Üç GPU sample aynı UUID, strictly ordered ve
  collection aralığında olmalı. Üç örnek bir kez alınır; admission polling yok.
- Host>=4 GiB, disk>=12 GiB, **her** sample GPUfree>=2048 MiB. Missing veya
  malformed memory alanları fail-closed. Utilization0–100 veya null telemetri;
  %35/%80/%100 admission blocker değildir, eksik telemetri sıfır yapılmaz.
- Raw process/compute query success; raw PID listesi ve structured PID listesi
  eşit. Her compute PID process inventory'de bulunur ve gerekçeli
  desktop/compute/unknown review alır. Incomplete/duplicate/missing review,
  unknown/compute veya ayrıca bilinen conflicting workload STOP'tur.
- Review, raw inventory canonical SHA256'sına bağlıdır. Process adına göre
  dar desktop allowlist kullanılmaz: WhatsApp gibi yeni desktop uygulaması
  gerekçeli identity review ile desktop olabilir. İzin/N/A verisi ham saklanır.
  Bu reviewer beyanıdır; digest executable güvenliğini kanıtlamaz. Yetersiz
  kimlik çözümü halinde unknown ve STOP gerekir.
- CLI input1 MiB, duplicate JSON key reddi, dosya/parsing hatasında
  eligible=false/exit2. CPU eligible kararı native onayı veya Job PASS değildir.

Bu görevde **canlı GPU preflight alınmadı**; eski ölçümler v3 passed kaydına
dönüştürülmedi. Approval/fresh-preflight dosyaları yeni snapshot'ta yok;
attempt/child marker0, worker/events file0. Yalnız bir stdlib approval-absent
refusal probe'u çalıştı: `APPROVAL_REQUIRED_NO_WORKLOAD_STARTED`, exit1;
Torch import, Job veya native workload başlamadı. Bu, yetkili execution değildir.

## 7. Güncel CPU/statik sonuçları ve açık failure

|Doğrulama|Sonuç|Kanıt|
|---|---|---|
|Yeni policy unit suite|**14 PASS**,0 FAIL;0.397s|`preflight-unittest.log`|
|r2/r3 kaynak/runtime/manifests, controller diff ve35-feature eşliği|PASS|`snapshot-static-verification.json`, `source-scope-audit.json`|
|Approval-absent CPU stdlib probe|Beklenen red,exit1; marker/Job/worker0|`approval-refusal-receipt.json`, stderr|
|Mevcut P23 isolation/activation suite|**62 executed =61 PASS /1 FAIL**,0 error/skip;453.654s|`p23-minimum.xml`, log/runtime/exit JSON|

Unit tests yüksek utilization ve exact memory eşiklerini, her sample'da low
GPUfree, RAM/disk eksikliği, stale/future/order/UUID/run/manifest mismatch,
unknown compute process, missing review/identity/query success, digest tamper,
malformed/null telemetry, duplicate/oversized JSON ve no-input-mutation yollarını
kapsar. Subtest'ler ayrı pytest case sayısına dönüştürülmedi. Unit inputs
sentetiktir; bunlar live GPU veya native CUDA acceptance değildir.

Mevcut suite gerçek **Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu**
ortamında koştu. Runtime origins ve sys.path kaydedildi. Worker base executable
GPU environment yolunda görünse de admission sonrası explicit runtime paths
CPU Torch'u seçer; version eşleşmesi üretim worker tarafından doğrulanır.
CUDA visibility=-1, Torch CUDA build=None, session final initialized=false;
CUDA validation script'i çalıştırılmadı.

Tek failure:
`tests/test_isolated_training_memory.py::test_real_owned_failure_limits_and_cleanup[timeout-timeout]`.
Bu mevcut CPU negative fixture özel olarak **15s** timeout kullanır.
`outcome=timeout`, `measurement=None`, `Job.active_processes=0` assertion'ları
geçti; son assertion stdout içinde `training_forward_started` aradı ve boş
stdout nedeniyle başarısız oldu (15.071s). Fixture bu çıktıyı forward-pre-hook'ta
flush edip120s bekler. **Forward'a ulaşıldığının kanıtı yoktur.**
Toplam runtime'ın tarihsel suite'den uzun olması import/startup gecikmesiyle
uyumlu olabilir; eldeki receipt/çıktı kesin gecikme nedenini veya son worker
phase'ini kanıtlamaz. “Yalnız flake” veya production defect çözümü ilan edilmedi.

Korunan testi, timeout'u, fixture'ı veya assert'i değiştirmedim; failure'ı
skip/xfail yapmadım; tekrar koşmadım. Native60s/outer90s sınırları bunun için
değiştirilmedi. **Temiz CPU gate yoktur.** Sonraki ayrı CPU işi, bu timeout
case'inin import/admission/forward phase kanıtını mevcut limitler içinde nasıl
sağlayacağını incelemek ve gerekirse kullanıcıca yetkili test-harness revizyonunu
ayrı doğrulamaktır. Bu audit eski temiz CPU recovery sonucunu silmez, bugünkü
suite'i de o eski sonuçla PASS'a dönüştürmez. Saved-activation suite33 case'in
tamamı ve isolation28 case'i geçti; toplam61 budur.

## 8. Sonraki native kontrolün somut sınırı ve karar

Bu görev native execution'ı açıkça yasakladı; önceki r2 onayları yeni r3 için
geçerli sayılmadı. Native çalıştırma için **yeni run-ID/manifest ve v3 politika
değişikliği üzerinden ayrı açık kullanıcı onayı gerekir**. Bugünkü CPU timeout
kanıtı açığı review/disposition almadan temiz CPU/native-readiness PASS verilmez.

Onay verilirse gelecekteki tek aday yine AdamW FP32,default cuda:0 stream,
hooks OFF/reset ON,seed71,epochs2,5 sample,batch2,accum2,küçük Linear–Dropout,
tek child fit; baseline/prefix/retry0'dır. Production CUDA saved-activation
guard değişmeyecek. Taze v3 preflight, source/runtime/import kimliği, empty-child
Job ve outer/inner Job admission readback'leri zorunludur.

Limitler **aynı**: worker60s, outer90s; process4 GiB private commit,
child Job8 GiB/4 process, outer Job12 GiB/8 process; allocator fraction.15;
kill-on-close; stdout/stderr64 KiB/channel,result2 MiB,
metadata2000 row/1 MiB per process. Job limits RSS/physical VRAM limitleriyle
karıştırılmaz. Failed admission veya native crash/timeout/missing identity/
diagnostic durumda STOP; exclusive attempt yeniden kullanılmaz, başka case yok.
Native exit/WER/dump/log/last durable phase korunur; owned PID/Job active0
doğrulanır. Crash sonrası ayrı bounded30s gözlem; dump policy/registry edit
ve upload yoktur. Tek diagnostic PASS, B4 root cause/P23 release kabulü değildir.

## 9. Ürün mimarisi sonucu ve sıradaki küçük işler

[Tasarım belgesi](../design/hyso-analysis-optimization-contract.md) iki model
giriş yolunu, kullanıcı requested/effective config sahipliğini, dört GPU
politikasını, task capability/recommendation/report sözleşmesini ve35 satırın
native kapsam/eksik/veri/donanım/dependency/acceptance/export/Manager/kullanıcı
yararı alanlarını tanımlar. Rapor/recommendation motoru implement edilmedi.

İlk eksik performans alt işi **P21 worker-side DataLoader trace**: CPU'da
bounded worker phase/ID/clock/queue kanıtı, exact order/RNG ve cleanup; bugün
next-wait'ten decode bottleneck uydurulmasını önler. Sonraki yeni planned feature
P26 classification evaluation'dır; P24/P27/P28 ve P29 quality karşılaştırması
ona dayanır. P14 kernel/correlation olmadan P13 CUDA/P15 overlap; gerçek
FLOP/traffic counters olmadan P18; gerçek cihaz/sensor/deployment olmadan
P30–35 completion ilan edilmeyecek.

Birleşik rapor için ilk uygulanabilir adım mevcut immutable result'ların
ID/hash/task/clock ilişkisini doğrulayan **saf evidence envelope validator**:
35 coverage slot, null/status/reason ve copied JSON artifact. Manager'a yalnız
gerçek finite numeric metric, mevcut geçerli epoch/step ekseniyle gönderilir;
structured unsupported alanlar JSON'da kalır. Tracking failure retrain tetiklemez.
Bu adım da sonraki ayrı görevdir; bu teslim otomatik başlamaz.

Son koruma ve kanıt hashleri `final-repository-protection.json` ve
`evidence-manifest.json` içinde. **B4 OPEN/UNRESOLVED; P23 PARTIAL; B10 PARTIAL;
R1 NO-GO.** Bu aşama burada durur; commit/push veya sonraki feature/native
execution'a geçilmez.
