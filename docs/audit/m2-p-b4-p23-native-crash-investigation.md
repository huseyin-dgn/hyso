# B4 Native Crash Investigation / P23 R1 — ilk forensic aşama

2026-10-21, Europe/Istanbul. **B4 NO-GO / OPEN; root cause UNRESOLVED.**
Bu aşamada yeniden üretim veya test çalıştırılmadı. Mevcut dump, Windows olayları,
receipt/loglar ve kaynaklar salt okunur incelendi. CUDA saved-activation admission
koruması yerinde; production/test dosyaları değiştirilmedi. Yeni repository
dosyası yalnız bu rapordur. P23 PARTIAL, R1 NO-GO ve B10 PARTIAL kararları korunur.

## 1. Koruma, kanıt zinciri ve kesin chronology

Branch `feature/hyso-models-next`; HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`. Başlangıçtaki 15 görev dosyası ve
kullanıcıya ait ayrı gradient farkı korundu. Index boş; stage/commit/push,
reset/checkout/clean, dependency/Torch/driver değişikliği yapılmadı.
Korunan `tests/test_gradient_accumulation.py` SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

Eski evidence kökü **E**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23r1-20261021`.
Bu incelemenin ek kanıtları **F**:
`E/forensic-phase1-20261021T123056Z`. Önceki kanıtlar üzerine yazılmadı.
F/start-protection.json başlangıç hash/status kaydıdır. Windows XML/Report.wer,
hash envanterleri, CDB çıktıları ve türetilmiş JSON özetleri F içindedir.
Dump kopyalanmadı veya dışarı yüklenmedi; rapora tensor/veri belleği aktarılmadı.

Aşağıdaki saatler **21 Ekim 2026, Europe/Istanbul / UTC+03:00**:

| Zaman | Olay ve kanıt niteliği |
|---|---|
| 15:05:24–15:09:28 | Önceki B koşusu frozen kaynakta 21/21 tamamlandı; `b-rss-fixed.*`. Bu C acceptance değildir. |
| 15:17:55.483335 | C parent harness için admission dosyası yazıldı; dosya mtime'ı, kesin process-start saati değildir. Supervisor actual launcher PID19548, timeout600s. |
| 15:18:12.722085–15:19:57.150001 | SGD6 ardından Adam6: 12 başarılı `.result.json` üretildi. Sonuçların checksum'ları geçerli; mtime'lar publication zamanına işaret eder. |
| 15:19:57.2120877 | PID19264 başlangıcı: Windows Event1000 içindeki process-start FILETIME. Dump MiscInfo saniye çözünürlüğünde15:19:57 ile uyuşur. |
| 15:20:05.9079777 | Application Error1000, RecordId160426: PID0x4B40/19264, nvcuda64.dll, `0xC0000409`, offset`0x18fd6e5`. Bu Windows olayının kaydedilme zamanıdır; memory corruption başlangıcı değildir. |
| 15:20:16 | Dump header zamanı; dump process uptime19s. WER toplama gecikmesi nedeniyle fault olayından sonradır. |
| 15:20:16.1814329 | Windows Error Reporting1001, RecordId160427, BEX64; aynı integrator Report ID ve exception alt verisi2. |
| 15:20:17.901463 | Parent stderr mtime: assertion, production tarafından sınıflandırılan native-crash payload'ını içeriyor. |
| 15:20:18.738341 | Parent supervisor receipt mtime; parent exit1, elapsed143.266s, timed_out=false. |
| 15:22:47.686399 | `c-native-crash.json` stderr'deki assertion payload'ından sonradan türetildi. |
| 15:26:40.190511 | `resume-protection.json` post-stop hash kaydı; crash öncesi manifest değildir. |

Başarılı 12 sonucun PID/saat/activation özeti F/completed-C-cases.json'dadır.
Başarısız13. case: AdamW, FP32, default stream, overflow=false, reset=true,
saved_activation_lifetime=true, seed71, epochs2; modular/early/tied=false.
Child receipt elapsed20.672s'dir; bu import/çalışma/Windows hata raporlaması ve
cleanup süresini kapsar, saf training zamanı değildir. Başarısız child için state,
result ve aggregate `c-cuda-fixed.json` dosyaları yoktur. Hangi epoch/microbatch/
attempt'e ulaşıldığı bilinmiyor; “ilk dropout” veya “ilk optimizer step” denemez.

## 2. Exit code, Windows evidence ve mevcut dump

Üç bağımsız kayıt katmanı uyuşuyor: parent-observed child exit, Windows
Application Error/WER ve dump exception stream. `c-native-crash.json` bağımsız
bir child publication değildir: stderr'den türetildiği byte içeriğiyle kontrol
edildi; canonical checksum da doğrulandı. `worker_receipt=null`,
`measurement=null`, `monitor_outcome=null`.

| Alan | Doğrulanmış değer |
|---|---|
| Worker PID / exception thread |19264 /21692 (`0x54bc`), dump thread0 |
| Exit / exception |3221226505 /`0xC0000409` |
| Exception parameter0 |2, CDB: `FAST_FAIL_STACK_COOKIE_CHECK_FAILURE` |
| Exception flags / instruction |1 / `int 29h` |
| Faulting module |`nvcuda64.dll`, file version32.0.15.9174, driver591.74 |
| Fault offset |`0x18fd6e5` |
| Windows integrator Report ID |`91c91be6-82ea-4559-b5be-4d327839f692` |
| WER archive report ID |`2711c31c-0989-4db3-b115-d591c85c00c7` |
| Dump |`C:/Users/hdgn5/AppData/Local/CrashDumps/python.exe.19264.dmp` |
| Dump boyutu / SHA256 |295308564 byte /`384412ccf6cb95b6505b2ce97052d1742e505fb6f45df9de8272818d8140eb45` |

WER archive'da Report.wer var; SHA256
`ca1517c56234692406abb5e8c71008b45e01d2e25640b7747b1c0aab771f88f6`.
Dosya F'ye kopyalandı; Windows Event160426/160427 XML'leri de korundu.
15:17–15:23 System log penceresinde seçilen Display/nvlddmkm/WHEA/Dxg
provider'ları için eşleşme bulunmadı. Bu dar sorgu GPU/hardware sorunu olmadığını
kanıtlamaz. LocalDumps ayarları yalnız okundu, değiştirilmedi.

Bu hata sıradan Python exception değildir. Microsoft'un tarif ettiği fast-fail,
standart exception handling ile kurtarılabilir bir yol sayılmaz; ilk exception
parametresi alt kodu taşır. Alt kod2 tespit edilen stack-cookie kontrol hatasıdır,
bozucu yazmanın adresini, zamanını veya kusurlu bileşeni belirlemez.
[Microsoft __fastfail](https://learn.microsoft.com/en-us/cpp/intrinsics/fastfail?view=msvc-170).
`except BaseException`, Python `finally` ve `atexit` çalıştı varsayılmadı.

## 3. Native çağrı yolu ve sembol güvenilirliği

Mevcut Microsoft.WinDbg1.2610.1001.0 paketinin amd64 CDB10.0.29661.1004 aracı
yalnız `-z` ile orijinal dump'a açıldı. `.exr`, `.ecxr`, `kn`, 37 thread için
sınırlı `~*kn 8`, `lmvm`, `ln`, `.frame`, `dv`, `dt` sorguları kullanıldı.
Live target, `.call`, `.restart`, CUDA probe veya hedef kod yürütülmedi.
Symbol path boş yerel dizine sabitlendi; symbol-server yolu yapılandırılmadı.

Çağırandan fault'a doğru daraltılmış zincir:

```text
Python interpreter / torch_python binding
  -> torch_cpu at::_ops::dropout::call +0x176
  -> torch_cpu at::native::dropout +0x48
  -> torch_cpu at::_ops::native_dropout::call +0x17f
  -> torch_cpu at::_ops::native_dropout::redispatch +0xa5
  -> torch_cuda internal frames
  -> nvcuda64 cuLibraryLoadData +0x4f
  -> NVIDIA internal frames
  -> nvcuda64 +0x18fd6e5, int29h / fast-fail
```

Dört dropout frame'i ve cuLibraryLoadData için `ln` komşu export sınırları
kontrol edildi: kısa offset'ler sonraki export öncesinde kalıyor. Bu, dropout
forward/library-load immediate yolunu güçlü biçimde destekliyor; tam private
işlev sınırı veya exact Python satırı kanıtı değildir. `cuLibraryLoadData` bir
CUDA code/library yükleme API'sidir; yüklenen PTX/cubin/image ve JIT nedeni
dump'tan belirlenmedi.
[NVIDIA Library Management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__LIBRARY.html).

Büyük offset'li `unique_dim_cuda`, `bucketize_outf`, `GraphRoot::apply`,
`cuProfilerStop` ve `native_dropout_backward_cuda` yakın-export etiketlerinden
unique/bucketize/backward/profiler-stop yürütüldüğü sonucu çıkarılmadı. Özellikle
backward adını taşıyan bir export, doğrulanmış forward dispatcher zincirini
tek başına backward kanıtına çevirmiyor. Immediate stack'te payload destructor,
saved-hook callback, peak reset veya Job termination çağrısı gösterilemedi.
Önceki yürütmelerin etkisi bu anlık stack ile dışlanamaz.

Torch/NVIDIA sembolleri export düzeyinde; python310 için yerel private PDB
çözüldü. Dump module timestamp/SizeOfImage ile mevcut python310, torch_python,
torch_cpu, torch_cuda ve nvcuda64 PE header'ları eşleşiyor. Mevcut dosyaların
SHA256'ları F/minidump-summary.json'a kaydedildi. Header eşleşmesi crash anındaki
tam binary byte veya private-PDB identity attestation değildir. CDB'nin Torch
checksum doğrulayamama uyarısı checksum mismatch kanıtı olarak kullanılmadı.

Dump flags`0x200121`, full-memory flag kapalı;37 thread/324 module içeriyor.
En yakın `_PyEval_EvalFrameDefault` frame'inden Python frame pointer'ı bulundu,
ancak f_back/f_code/f_lasti/f_lineno okumaları eksik heap nedeniyle başarısız.
MemoryList kontrolü de ilgili adres aralığının capture edilmediğini doğruladı.
Exact Python filename/line ve canlı payload/graph state kurtarılamadı.

Önceki B4 PID15644 dump'ı da salt okunur karşılaştırıldı: eski hash
`c68a154e30fd66b6408f68aa88afc7d41db550bec0b6ee756ad1c14292da4ef1`, aynı code,
parameter2, driver timestamp/image size ve offset`0x18fd6e5` var. Önceki
[R3 raporunun](m2-p-b4-r3-native-callpath-investigation.md) dar çağrı yolu SiLU
forward idi; yeni dump dropout forward gösteriyor. **Benzer fail-fast imzası
doğrulandı; aynı root cause veya C adapter'ın kusurluluğu kanıtlanmadı.**

## 4. Kaynak provenance: `unverified_source_identity`

F/source-provenance.json dosya bazında current/B/post-stop hashlerini içerir.
B frozen manifestindeki **138 dosyanın tamamı** tekrar hash'lendi ve eşleşti.
Bu, B snapshot bütünlüğünü doğrular; mutable checkout'tan yürüyen C child'ın
aynı byte'ları yüklediğini doğrulamaz. C recipe, factory_search_path ve parent
argv checkout yollarını gösteriyor; başarısız child import hash'i yayınlamadı.

| Dosya grubu | Kanıt ve güvenilirlik sınırı |
|---|---|
| `_training_memory_worker.py`, `_training_process.py` |Current = frozenB = post-stop hash; değişmeyen dosya için güçlü disk sürekliliği kanıtı. C process import byte hash'i yok. |
| `training_memory.py`, `training.py`, `trainer.py`, supervisor |Current=frozenB; legacy dosyalar HEAD/post-stop kontrolüyle de tutarlı. Loaded-byte kimliği ayrıca attest edilmemiş. |
| `_saved_activation.py` |B snapshot'ında yok. Current hash`e66b0ddf…75235c1b`, post-stop ile aynı; current mtime15:21:01, fault'tan sonra. C için frozen kaynak yok. |
| `isolated_training_memory.py` |B ile farklı; current hash`9bacc676…6f85a0d`. mtime15:22:44. Current guard aynı protocol'ü reddederdi; crash öncesi sürümle eşit kabul edilemez. |
| CUDA harness / factory |B ile farklı; current post-stop hashleriyle aynı. mtime'ları crash öncesi olsa da çalışan process'in source byte kimliğini tek başına kanıtlamaz. |

Mevcut15 görev dosyasının hashleri `resume-protection.json` post-stop kaydıyla
eşleşiyor. Bu kayıt crash sonrasına aittir. Mtime, source path, version string,
HEAD ve sonraki hash eşleşmesi import-time byte hash'inin yerine geçmez.
Sonuç: **C crash kaynak seti için `unverified_source_identity`**. Aşağıdaki statik
bulgular mevcut kaynak için geçerlidir; crash process'inde aynı satırların
bulunduğu kesinleştirilmemiştir. Eksik kaynak kimliğini tahmini ters patch ile
“verified” hale getiren bir reconstruction yapılmadı.

## 5. Saved-tensor hook ownership ve autograd analizi

Mevcut `_saved_activation.py:15–51,61–114` şu ownership zincirini kuruyor:

```text
autograd SavedVariable / native hook holder
  -> packed _Payload -> original Tensor -> grad_fn -> saved slots
  -> bound pack/unpack methods -> recorder
recorder.slots -> weakref(_Payload) + storage identity metadata
_Payload.owner -> recorder
```

`slots` weakref kullanıyor olsa da bütün ownership zinciri weak değildir.
Payload tensora güçlü referans tutuyor; native holder da payload ve bound
method'lar üzerinden recorder'ı tutabiliyor. Tensor bir graph çıkışıysa kendi
grad_fn'ına geri dönerek cycle oluşturabilir. Bu statik olarak mümkün bir
retention/cleanup riskidir; crash'in kanıtlanmış nedeni değildir. Hooks içinde
clone/detach/CPU kopyası görülmedi. Önceki görevdeki detach yasağını değiştiren
bir çözüm bu forensic aşamada uygulanmadı.

PyTorch v2.5.1 kaynakta default hook yolu orijinal packing davranışından ayrılır;
normal output-saving yolunda cycle önleyici tensor-data saklama bulunur.
Hook açılması ayrıca unpack version-counter denetiminin semantiğini değiştirir.
Bu nedenle aynı sayısal tensorun dönmesi bütün autograd ownership/detection
davranışının aynı kaldığını kanıtlamaz.
[SavedVariable v2.5.1](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/autograd/saved_variable.cpp).

Native Python hook holder pack/unpack sırasında GIL alır, packed Python nesnesini
saklar ve destructor'da GIL altında callback/data referanslarını azaltır.
Dolayısıyla `_Payload.__del__` normal backward saved-data release sırasında da
çağrılabilir; yalnız dış Python scope kapanışında çalıştığı varsayılamaz.
[PySavedVariableHooks v2.5.1](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/autograd/python_saved_variable_hooks.cpp).
Bu kaynak incelemesi exact DLL/PDB satır eşlemesi değildir.

Güncel PyTorch açıklaması graph output'unu hook return nesnesinde canlı tutmanın
cycle riskini de açıklar; bu güncel doküman tek başına2.5.1'e özgü sözleşme
değişikliği kanıtı sayılmadı.
[Autograd saved-tensor mekanikleri](https://docs.pytorch.org/docs/stable/notes/autograd.html#hooks-for-saved-tensors).

## 6. Destructor, weakref ve hook cleanup etkileşimleri

`_Payload.clear:21–26` önce owner'ı yerel değişkene alıyor, `self.owner=None`
yapıyor, sonra `self.tensor=None` ve `owner.release()` çağırıyor. Owner'ın önce
temizlenmesi aynı payload için yeniden girişte ikinci release riskini azaltır.
Ancak tensor decref'i native graph/storage destruction ve başka payload
finalizer'larını **release bookkeeping tamamlanmadan önce** tetikleyebilir.
Bu reentrant destruction tasarımının değişik release sıralarıyla ayrıca
kanıtlanması gerekir. Statik kodda manuel native-pointer free veya kesin
double-free görülmedi.

`__del__:28–29` içindeki metadata işlemleri istisna atabilir; `release()` slots
ve storage kayıtlarının varlığını varsayıyor. Olası KeyError/teardown hatası bir
Python unraisable exception olur; bunu doğrudan`0xC0000409` açıklaması saymak
yanlıştır. Child stderr yalnız GradScaler deprecation warning içeriyor; bundan
bütün destructor yollarının güvenli olduğu sonucu da çıkarılamaz.

`close:131–138` weakref registry'nin snapshot'ından erişilebilen slotları forced
clear yapıyor. Tensoru bırakarak graph cycle'ını kırabilir, fakat native holder
henüz payload nesnesini tutarken de bu işlem yapılabilir. Bu yüzden forced
cleanup “native payload disposal” veya physical CUDA free kanıtı olamaz.
`result()` live slotları reddediyor; başarısız child için zaten result yok.

`saved_tensors_hooks.__exit__` default hook pair'ini pop eder; mevcut native
saved-slot holder'larının tamamının yok olduğunu garanti etmez. `_run_recipe`
finally'si recorder.close ardından activation.close uygular; `_measure_training`
ayrıca recorder'ı kendi finally'sinde kapatır. Memory recorder idempotent closed
kontrolü ve sampler join içerir. Bu sıra statik olarak incelendi; fast-fail anında
bu finally'lerin çalıştığına dair kanıt yok. Mevcut immediate stack forward'dadır,
teardown veya Python finalization'da değildir.

`_cdata` yalnız storage identity metadata'sı olarak kullanılıyor; pointer'a
Python'dan yazma/free görülmedi. Kayıtların private identity'ye dayanması veya
callback thread/ordering bilgisinin eksikliği ayrı diagnostic eksiklerdir.
GIL Python refcount/dict işlemlerini koruyabilir; CUDA/asynchronous native
ownership güvenliğini tek başına kanıtlamaz. GC-disabled CPU başarıları CUDA
destruction güvenliğine genellenemez. Başarısız recipe'de lifetime_path yok;
test fixture'ın özel `gc.disable()` yolu veya `atexit` lifetime probe'u seçili
değil. Crash anındaki gerçek GC durumu ayrıca kaydedilmemiş.

## 7. CUDA allocator peak reset etkileşimi

Mevcut `_IsolatedRecorder.microbatch_begin:146–171` yeni window'da selected
stream synchronize -> pre-reset snapshot -> reset_peak_memory_stats -> after
snapshot sırasını izliyor. attempt_end:181–197, synchronize ardından actual
peak/delta okuyor. Default-stream case de recorder tarafından seçilmiş stream'e
bağlı. Parent reset'i, empty_cache, context reset veya hook içinde CUDA copy yok.

Native allocator v2.5.1 `resetPeakStats` kilit altında peak counter'larını
sıfırlar; cache boşaltma/free yolu değildir. “Peak reset live payload belleğini
serbest bıraktı” hipotezi bu kaynak tarafından desteklenmiyor.
[CUDACachingAllocator v2.5.1](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDACachingAllocator.cpp#L1521-L1537).
Senkronizasyon ve ilave queries lazy-loading/timing'i etkileyebilir; bu dolaylı
etki burada ölçülmedi. Hidden/concurrent stream davranışı kanıtlanmadı.

Başarısız child'ın protocol'ünde reset izni var; fakat hangi reset çağrısına
gerçekte ulaşıldığını gösteren crash öncesi durable window ledger yok. B21 PASS
reset mekanizmasının başka frozen/no-activation kaynakta gözlenen başarısıdır;
C'deki reset/hook etkileşimini aklamaz veya suçlamaz. Dump'ta reset çağrısı yerine
dropout/library-load yolu bulunuyor.

## 8. Worker, Job Object ve process cleanup bulguları

`_training_process._launch:79–152` actual interpreter'ı `-I -S -B` ile açar;
Job assignment/admission'dan sonra worker HYSo/Torch import'una geçer.
Monitör timeout/log/result ceiling durumlarında Job'u sonlandırır; leader
normal veya anormal kapanınca da descendant temizliği yapılır. Kapatma çağrısı
her durumda bulunması nedeniyle tek başına crash nedeni sayılamaz.

Crash receipt: admitted=true, monitor_outcome=null, exit`0xC0000409`, total
processes4, active0. Peak process private commit2015567872 ve Job2016907264 byte;
configured4GiB/process ve8GiB/Job altında. OOM/resource-limit/timeout termination
kanıtı yok. Windows exception kaydı actual worker native failure'ını bağımsız
destekliyor; bunu parent assertion veya Job cleanup exit'iyle karıştırmıyoruz.

Outer supervisor PID19548 exit1/timed_out=false; assertion child crash payload'ını
yazmış. Bu parent native crash değildir. Receipt active0 ve post-stop PID
kontrolleri cleanup'ı destekler; bu incelemede PID19264/19548 için canlı process
de görülmedi. Child/grandchild PID listesi crash kaydında yok; total4'ün her
üyesinin kimliği çıkarılamıyor. Process-tree cleanup, Python hook cleanup veya
GPU tensor lifetime doğrulaması değildir. Bootstrap'ın `except BaseException`
bloğu native fail-fast'i completed Python outcome'a dönüştüremedi; worker receipt
yayınlanmadı. Temporary request/ready/result dosyalarının kalıcı failure copy'si
olmadığından import/phase kanıtı eksik kaldı.

## 9. Doğrulanmış bulgular ve kanıtlanmamış hipotezler

Doğrulananlar: actual worker native fail-fast, alt kod2, driver fault offset,
dropout/library-load immediate stack,12 tamamlanmış sonuç, parent exit1,
active-process0, failed measurement yokluğu, B snapshot bütünlüğü, C source
identity eksikliği ve mevcut guard'ın korunmuş olmasıdır. Root cause belirlenmedi.

| Hipotez | Destek / sınır |
|---|---|
| Driver/library-loading yolunda hata |Immediate path ve önceki B4 ile benzer imza destekler; NVIDIA kusuru veya code-image/JIT kaynaklı defect kanıtlanmadı. |
| Packed original Tensor cycle / reentrant deallocation etkisi |Mevcut kaynakta gerçek statik risk; crash source identity, canlı payload state ve destructor stack yok. Causal değil. |
| Peak reset / synchronize timing etkisi |İlave ordering/timing mümkün; doğrudan tensor free iddiası desteklenmiyor; failed window reset kanıtı yok. |
| AdamW optimizer math/step hatası |Case etiketi AdamW; stack optimizer step göstermiyor. Önceki step'ler dışlanamaz, herhangi bir step'e ulaşıldığı kanıtlanmadı. |
| Parent Job'un erken kill'i |Monitor nedeni yok, Windows native exception ve forward stack var. Direct termination açıklaması desteklenmiyor; bütün çevresel etkiler dışlanmış değil. |
| Genel runtime/driver/hardware veya önceki async corruption |Mevcut veri ayırt etmiyor. Aynı offset veya sınırlı System log yokluğu birini root cause yapmaz. |

## 10. Eksik diagnostic evidence

- Import-time exact source manifest/immutable C snapshot; failed worker'ın
  package/factory/runtime-path receipt'i ve kaynak bytecode kimliği.
- Son durable phase, epoch/microbatch/attempt/window ID; pack/unpack/release
  sayaçları, callback thread kimliği ve ordered cleanup giriş/çıkış kayıtları.
- Python frame/code/payload heap alanlarını içeren full user-mode dump;
  mevcut minidump bu adresleri dışarıda bırakıyor.
- Exact Torch/NVIDIA private symbols veya exact-build map; loaded CUDA
  library/code/options kimliği. Current PE header eşleşmesi bunların yerine geçmez.
- Crash öncesi allocator snapshots ve yürütülmüş reset sayısı; native GPU hata
  timeline'ı. B ve diğer12 sonuç failed worker'ın geçmişi değildir.
- Child/grandchild creation/exit listesi, sonlanan worker'ın preserved request/
  ready dosyaları ve gerçek GC/finalization state.

## 11. Riskleri azaltılmış bounded yeniden üretim planı — ÇALIŞTIRILMADI

Bu bir sonraki aşama önerisidir; exact geçmiş crash replay'i olduğu iddia
edilemez. İmport edilen C source kimliği eksik olduğu için yeni diagnostic
variant ayrı adlandırılmalı ve eski failure kanıtı korunmalıdır.

1. Önce bu rapor ve F kanıtları gözden geçirilir. Yeni kod/fix veya otomatik
   matrix başlatılmaz. Production guard yerinde tutulur. Kullanıcıdan **tek
   bounded diagnostic koşu** ve gerekirse **yalnız adlandırılmış forensic
   kopyada activation guard bypass** için açık, kapsamı belirli onay alınır.
2. Ayrı immutable kaynak kopyası hazırlanır; HEAD, dirty dosyalar, factory,
   bootstrap, interpreter, Torch/DLL identity ve bütün kaynaklar hash'lenir.
   Parent/child aynı manifesti import öncesi ve import sonrası doğrular; mismatch
   workload admission'ını durdurur. Checkpoint/resume/state transfer eklenmez.
3. İlk onaylı aday tek AdamW FP32/default-stream/no-overflow case olur:
   seed71,2 epoch,5 sample, microbatch2/uneven accumulation2, tiny Linear–Dropout
   modeli.12-case prefix tekrarı veya sonsuz retry yoktur. Cold case'in prefix
   koşulundan farklı olduğu raporlanır; tek PASS eski crash'i kapatmaz.
4. Mevcut Job admission düzeni korunur: child60s,4GiB/process,8GiB/Job,4 process,
   CUDA fraction.15; outer90s,4GiB/process,12GiB/Job,8 process üst sınırı.
   Dump capture overhead'i timeout kararından ayrı kaydedilir. WER child/helper
   process'lerin Job limitine etkisi preflight'ta değerlendirilir; otomatik
   resource-limit gevşetmesi yapılmaz.
5. İşe başlamadan disk/host/GPU kapasitesi ve ilgisiz concurrent işler okunarak
   kontrol edilir; kullanıcı process'leri öldürülmez. Full dump için yeterli
   disk ve capture boyut sınırı üzerinde anlaşılır. Dump ayarı gerektiğinde
   global Python politikasını sessizce değiştirmek yerine scoped capture ve
   geri alma planı ayrıca onaylanır. Dump dışarı yüklenmez.
6. Ayrı diagnostic runner, bounded ve tensor tutmayan metadata kaydeder:
   admission/import IDs, phase ve source ledger ID'leri, hook sayaçları,
   callback thread IDs, finalization giriş/çıkışları. Tensor repr/clone/CPU copy,
   global GC/empty_cache, senkronizasyon workaround'u veya üretim matematiği
   değişikliği eklenmez. Ölçümün timing'i değiştirdiği açıkça kaydedilir.
7. Her native failure'da derhal stop: actual exit, process tree, WER/report ID,
   dump ve failure kaynak snapshot'ı korunur; kalan senaryolar NOT RUN kalır.
   Bir sonucun başarısızlığı üzerine otomatik başka optimizer/AMP/stream,
   reset-off, hook-off veya destructor variant'ı denenmez. Bunlar gerekirse
   sonraki ayrı onaylı kontrollü karşılaştırmalardır.

Güvenlik koşulları: workload başlamadan Job/timeout/size sınırları, kaynak
identity doğrulaması, tek counter owner/selected stream, rollback gerektirmeyen
izole kaynak, full failure evidence capture hazırlığı, protected hash kontrolü
ve native-crash stop davranışı hazır olmalıdır. CPU/static diagnostic preflight
bir CUDA acceptance PASS'i olarak sayılmamalıdır. Driver/Torch yükseltmesi,
production guard kaldırılması ve geniş release matrix bu plana dahil değildir.

## 12. B4 gate kararı ve ilk aşama sonu

**B4 NO-GO / OPEN, root cause UNRESOLVED. C native acceptance FAIL; P23 PARTIAL.**
B21/A29 ve önceki2303 regresyon geçmiş kanıttır; bu forensic görevde yeni PASS
üretildiği veya C gate'inin kapandığı iddia edilmez. B10 full GO yoktur.

Bu aşama kanıt toplama/statik analizle burada biter. Rapor planındaki tek bounded
yeniden üretim ve forensic kopyada gerekli guard bypass için ayrıca kullanıcı
onayı gerekir. Onay alınmadan mevcut CUDA admission koruması değiştirilemez,
crash yeniden çalıştırılamaz. Bu rapor herhangi bir yeniden test onayı değildir.
