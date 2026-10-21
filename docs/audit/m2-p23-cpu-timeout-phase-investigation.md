# HYSo — P23 CPU timeout / forward-start evidence investigation

2026-10-21–22, Europe/Istanbul. **CPU timeout evidence gate PASS.**
Güncel tam P23 CPU suite **63 PASS**,0 FAIL/ERROR/SKIP. Yeni CUDA workload0;
native reproduction0. r3 snapshot ve production CUDA guard değiştirilmedi.
Bu kayıt [önceki politika incelemesindeki](m2-p23-product-policy-gpu-admission-review.md)
61 PASS/1 FAIL CPU sonucunun devamıdır; eski failure kanıtı korunur.

## 1. Koruma ve kaynak kimliği

Branch `feature/hyso-models-next`; HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş.
Başlangıç34 dirty dosya önceki teslimin path/status/SHA256 envanteriyle eşleşti;
fark0. Yeni görev kanıt kökü **E**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-cpu-timeout-20261021T234559`.

İzinli değişiklik yalnız mevcut `tests/test_isolated_training_memory.py`
üzerindeki timeout testidir. Önceki test byte'ları E'de ayrı kopyada tutulur;
`timeout-test-change.patch` farkı saklar. Diğer33 başlangıç dosyası byte/status
eşit korunur. Yeni iki repository dosyası `tests/training_timeout_evidence.py`
ve bu audit'tir; final36 dirty. Mevcut P23/CPU recovery değişiklikleri geri
alınmadı veya üzerlerine eski sürüm yazılmadı.

|Korunan veya değişen kaynak|Başlangıç / son SHA256|
|---|---|
|`tests/test_gradient_accumulation.py` — değişmedi|`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`|
|Production guard dosyası — değişmedi|`9bacc67691ab032dfa9b6202f5ea3c4bd8b78e0653aca40d53b2fa6e86f85a0d`|
|Isolation testi — eski|`c81ce37242015a8b0a366a1cf0c339a50221fa7860e938315c9dba91521a4c4f`|
|Isolation testi — yeni|`b54955811e7e866eabdc49c410b557a02d9b1e999c7f8808fea350054ac5aaef`|
|Yeni test-only phase helper|`ab4b2f1c7c8359589063cd479e7f78b50e02478cfa910f10bd63247bc25d103e`|

47 production Python modülünün tamamı başlangıçla byte eşittir; production
timeout/Job/classifier/worker/Trainer/saved-activation kodu değiştirilmedi.
150 code/test/fixture/config dosyalık güncel test manifesti
`test-source-manifest.json`, SHA256:
`4b90c4437f71eacf5030cbc6540d9e7119be1b8b4490d3fc49ca840f2261708a`.
Bu test kaynağı kimliğidir; native r3 manifestinin yerine geçmez.

## 2. Eski failure'dan gerçekten bilinenler

Önceki Epolicy:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-product-policy-20261021T232140`.
Eski XML/log62 executed=61 PASS/1 FAIL,453.654s olarak korunur.
Başarısız case
`test_real_owned_failure_limits_and_cleanup[timeout-timeout]`,15.071s.
Son stdout assertion'ından önceki `outcome=timeout`, `measurement=None`,
`Job.active_processes=0` kontrolleri geçmişti. Stdout boştu.

Eski API temporary worker dizinini sonuçtan sonra temizlediğinden eski
ready/request/factory/import fazları geri kazanılamadı. Eski koşunun gerçekten
forward'a ulaştığı veya tam kaç saniyeyi importta harcadığı **bilinmiyor**.
Eksik marker'dan tek başına flush defect'i veya forward girişi çıkarılmadı.

Kaynakta `_training_process._launch` saati `Popen` öncesinde başlatır;
`protocol.timeout_seconds` bütün process lifecycle'ını kapsar. Import/factory
sonrasında ikinci sayaç başlatmaz. Worker `ready.json` dosyasını Job admission
doğrulamasından sonra, HYSo/Torch importundan **önce** yayımlar.
Fixture'ın `training_forward_started` çıktısı gerçek model forward-pre-hook'u
içinde `print(...,flush=True)` ile yazılır; sonra120s uyur. Bu hook girişidir,
modelin gerçek forward hesaplamasının tamamlandığı anlamına gelmez.

## 3. Yeni araştırma koşuları: aynı failure, daha güçlü faz kanıtı

Özgün assertion korunarak önce hedefli CPU probe yapıldı:
**1 FAIL**,27.83s test süresi; launcher elapsed15.031s.
Timeout/cleanup geçti, stdout yine boştu. Bu kez request/admit/ready/receipt
temporary dizin silinmeden E'ye alındı. `ready.json` token/PID eşleşmesi
`job_admitted_before_import` noktasını kanıtladı; factory/forward kanıtı yoktu.

İlk probe'da test-only argv router `-I -S -B` dizisinde script indeksini yanlış
seçtiği için wrapper devreye girmedi; parent/child phase listeleri boştu.
Bu gözlem hatası düzeltildi ve ilk sonuçlar üzerine yazılmadan ayrı
`instrumented-target.*` dosyalarıyla ikinci araştırma yapıldı. İlk probe
import timing ölçümü gibi sunulmaz.

Instrumented özgün test **1 FAIL**,23.23s; launcher elapsed15.047s.
PID932 için gerçek son phase `hyso_torch_import_enter`; completed/factory/
Trainer/forward phase yok. Tek süreçteki sequential çağrı yolunda factory,
observed import dönüşünden sonra çağrılabilir; bu probe factory/forward'a
ulaşmadı. Marker eksikliği burada forward'a ulaşılmamasıyla açıklanır;
stdout flush defect'i gözlenmedi.

|Probe fazı|Launch-enter'a göre gözlenen zaman|
|---|---:|
|Popen döndü|0.015s|
|Parent gerçek Job assignment tamamlandı|0.031s|
|Child Python bootstrap|0.187s|
|Child admit token/PID doğrulandı|0.234s|
|HYSo/Torch import çağrı sınırı|0.265s|
|Launcher timeout+cleanup elapsed|15.047s|

Kalan yaklaşık14.7s import çağrısı/completion beklenen bölgede geçti; tamamlanmış
import süresi ölçülmedi. Monitor'un tam termination anı ayrıca event olarak
kaydedilmediğinden cleanup zamanı saf import süresine eklenmez. Yavaşlığın
alt dependency/DLL/disk/OS nedeni bu kanıttan belirlenmedi. Bu tablo instrumented
probe'a aittir; eski failed worker'ın geçmiş zaman çizelgesi değildir.
Araştırma helper'ı bu probe'lar sırasında geliştirildi; ara probe'ların helper
byte'ları nihai150-file test manifestiyle attested sayılmaz. Nihai hedefli ve
tam suite kaynağı ayrı manifest/hash eşliğiyle korunur.

Python3.10 `monotonic_ns` sistem çapındaki monotonic saat kullanımıyla
Windows process'leri arasında faz hizalaması yapıldı; çözünürlük ve I/O
overhead nedeniyle sonuçlar yaklaşık gözlem süreleridir.
[Python3.10 time sözleşmesi](https://docs.python.org/3.10/library/time.html#time.monotonic).

## 4. Neden ve en küçük düzeltme

**Kanıtlanan hata test varsayımıdır:** startup-inclusive15s deadline'ın her
koşulda forward girişinden sonra dolacağını beklemek production sözleşmesine
uygun değildir. Sözleşme startup/import sırasında da timeout'a izin verir.
Yeni phase probe bu geçerli yolu gerçek CPU worker'da gösterdi. Production
timeout classification veya process cleanup defect'i bulunmadı.
Eski run'ın exact fazı ve import yavaşlığının OS/library alt nedeni unresolved
olarak kalır; bunlar doğru timeout sözleşmesini değiştirmeyi gerektirmez.

Uygulanan test düzeltmesi:

1. Özgün parametrized total-lifecycle testi **15s** kalır. Gerçek son phase,
   launcher/ready/PID/Job receipt'i kaydedilir. Forward-pre-hook gözlenmişse
   özgün gerçek stdout marker assertion'ı zorunludur; gözlenmemişse forward
   kapsamı iddia edilmez ve stdout marker'ın da bulunmadığı doğrulanır.
2. Ayrı `test_default_deadline_timeout_proves_forward_entry_and_cleanup`,
   mevcut production default **60s** bütçesiyle tek gerçek CPU worker çalıştırır.
   Factory→Trainer/loader→fit→gerçek forward-pre-hook phase ve özgün stdout
   marker'ı **koşulsuz** ister; ardından timeout ve Job active0 doğrular.
   Parent Python/NumPy/Torch RNG eşliği kontrol edilir. Bu,15s testinin
   başarısızlığına göre açılan fallback/retry değildir; bağımsız sabit case'tir.
3. Test-only `training_timeout_evidence.py`, aynı production worker'ı
   `runpy` ile çalıştırır; gerçek Popen ve Job.assign dönüşünü gözler, child'da
   admit token/PID doğrular, gerçek import ve function giriş/dönüşlerini kaydeder.
   Factory/Trainer/loader/forward kaydı mevcut fonksiyon çağrılarından gelir;
   stdout marker üretmez. Production timer/clock/recipe/Job kodu değiştirilmez.

Helper her process için ayrı append/fsync JSONL kullanır;64KiB/file ve
32 profiled function event sınırı vardır. Partial son satır tamamlanmış event
sayılmaz. Ready dosyası bound4096B ile okunur. Test çıktı/phase/result dosyaları
tmp_path altında kalır; assert başarısız olsa da incelemeye açıktır. JUnit
properties son phase/forward-observed/budget/active0 alanlarını saklar.

Instrumentation faz sınırlarına I/O ekler ve import/function gözlem overhead'i
yaratır; timing/scheduling'i etkileyebilir. Bu performans karşılaştırması veya
etkisiz tracing iddiası değildir. Kayıtlar tensor/graph/value kopyası içermez.
Yeni helper yalnız timeout testlerinde aktiftir; production telemetry/API değildir.

Test skip/xfail yapılmadı, stdout/phase uydurulmadı. Özgün15s artırılmadı;
ayrı forward-coverage case'i zaten var olan60s varsayılanını kullanır. Hiçbir
sayaç import/forward girişinde resetlenmez. Native forensic worker60s/outer90s,
memory/Job ve CUDA guard sınırları aynıdır.

## 5. Hedefli güncel sonuç ve faz dağılımı

Güncel hedefli koşu **2 PASS /28 deselected**,79.51s. Bu iki test tam suite'in
alt kümesidir; sonraki toplamına eklenmez.

|Faz|15s total-deadline case|Ayrı default60s forward-timeout case|
|---|---:|---:|
|Python bootstrap|0.156s|0.125s|
|Job admission child'da gözlendi|0.187s|0.156s|
|HYSo/Torch import enter|0.218s|0.187s|
|Import completed|13.750s|11.890s|
|Factory enter|13.781s|11.922s|
|Trainer initialization enter/return|13.828/13.859s|11.937/11.953s|
|Factory return, loader'lar kurulmuş|13.937s|11.968s|
|Trainer.fit enter|13.984s|12.015s|
|Gerçek timeout forward-pre-hook enter|14.140s|12.031s|

Import aralığı13.532s ve11.703s; instrumentation overhead dahildir. 15s case'in
bu kez forward'a çok az süre kala ulaşması, önceki import-time timeout'ı
geçersiz kılmaz; faz değişkenliğini gösterir. İki güncel case'te de gerçek
`training_forward_started` stdout marker'ı ve Job active0 doğrulandı.
Forward hook120s sleep'i bitmeden ilgili15/60s total deadline devreye girdi.
Model forward hesaplaması tamamlandı veya optimizer step uygulandı denmez.

## 6. Tam güncel P23 CPU gate

Güncel tam suite **63 PASS =30 isolation +33 saved-activation**,0 FAIL/ERROR/SKIP,
**533.380s**. Kanıt: `p23-minimum.xml`, `p23-minimum.log`, runtime/exit JSON ve
worker phase/result dosyaları. Önceki61 PASS veya hedefli2 PASS bu toplamın
parçası olarak ayrıca eklenmedi. Ara özgün-test failure'ları da korunur.

|Tam suite timeout case'i|Gerçek son child phase|Stdout marker|Launcher elapsed|Job active|
|---|---|---|---:|---:|
|Özgün15s total-deadline|`hyso_torch_import_enter`|Yok; forward iddia edilmez|15.062s|0|
|Ayrı default60s forward-coverage|`forward_hook_enter`|Gerçek marker var|60.047s|0|

Tam suite iki farklı timeout fazını birlikte doğruladı:15s import tamamlanmadan,
60s gerçek forward-pre-hook girişinden sonra. Böylece forward kapsamı koşulsuz
ayrı bir testte korunurken startup timeout doğru sınıflanır. JUnit test süreleri
sırasıyla15.143s/60.135s; launcher elapsed ile test/fixture overhead aynı şey değildir.

Constructor/forward exception, memory/log/result sınırları, child/grandchild
tree cleanup, üç optimizer için tek-fit exact state ve parent RNG kontrolleri,
activation/tied/ModularCNN/early-stop replay, interrupted/resumed/completed
checkpoint, GC-disabled/exception/reentrant/version-counter davranışları,
JSON/CSV/Manager/no-retrain/tamper kapsamı mevcut testleriyle geçti.
Bu test kapsamı genel native CUDA güvenliği veya bütün RNG/backend/platform
kombinasyonlarının kabulü değildir.

Runtime gerçek Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu;
CUDA visibility=-1. Parent origins/sys.path ve final initialized=false kaydı
saklandı: hedefli ve tam koşu exit0, CUDA initialized=false. Worker base executable GPU environment yolunda olsa da admitted
bootstrap CPU runtime site-packages yollarını seçer ve Torch sürümünü doğrular.
Yeni phase helper requested device=cpu koşulunu ayrıca zorunlu tutar.

Production değişmediğinden yeni packaging veya geniş legacy suite bu görevde
gerekmedi:47 module bytes, exports, archive config ve result/schema aynı.
Yeni helper tests altında ve mevcut packaging kapsamı dışındadır. Bu karar
packaging testleri yeniden geçmiş gibi sunulmaz; geçmiş sayılar eklenmez.

## 7. r3 koruması, native öncesi açık işler ve duruş

Salt okunur doğrulanan r3:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-product-policy-20261021T232140/proposed-hook-off-control-r3`.
Run ID `f6ff1fa5-60d4-41b0-963f-d6459856c479`;
manifest SHA256
`660a53dbdde6a21901bf6ad2743a3fad9bb51bad57ebd25a5a5dee46ad2444b8`.
252 source+26 runtime ve manifest/seal dosyaları başlangıçta/sonda doğrulandı;
254 snapshot dosyasının byte ve dosya kümesi eşliği ile read-only nitelikleri
korundu. Kaynaklara yazılmadı.
Approval/fresh-preflight/attempt/child marker'ları yok; worker/events file0.
Bu test değişikliği r3 içine taşınmadı; r3 yeniden mühürlenmedi.

CPU timeout test sorununun çözülmesi yeni native yetkisi değildir. Eski native
`0xC0000409` root cause'u ve eski C source identity açığı, CUDA activation
safety/noninterference/native coverage, final-source/installed/package release
kapıları açık kalır. Tek hook-off/reset-on r3 diagnostic'i ayrıca açık kullanıcı
onayı, taze v3 preflight ve mevcut identity/Job/memory/timeout/cleanup koşullarını
gerektirir. Bu görevde CUDA workload veya reproduction başlatılmadı.

Production CUDA saved-activation guard etkin kalır. **B4 OPEN/UNRESOLVED;
P23 PARTIAL; B10 PARTIAL; R1 NO-GO** korunur. Dependency/Torch/NVIDIA driver
ve dump policy değişmedi; stage/commit/push/reset/restore/clean yapılmadı.
Yeni audit ile test-only düzeltme raporlandıktan sonra bu aşama durur.
Son koruma `final-repository-protection.json`, faz/test özeti
`test-result-summary.json`, r3 karşılaştırması `r3-before.json`/`r3-after.json`,
process gözlemi ve kanıt hashleri `evidence-manifest.json` içinde saklanır.
