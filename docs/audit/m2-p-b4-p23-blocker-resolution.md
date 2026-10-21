# HYSo — P23 / B4 Blocker Resolution, ilk aşama

2026-10-21, Europe/Istanbul. **B4 OPEN/UNRESOLVED; P23 PARTIAL; B10 PARTIAL.**
Native crash root cause'u belirlenmedi. Bununla birlikte saved-tensor hook'un
native in-place mutation error boundary'sini atlayan **kanıtlanmış bir HYSo
noninterference defect'i** CPU'da bulundu ve düzeltildi. Bu, driver fail-fast'in
nedeni veya çözümü olarak sunulmaz. CUDA saved-activation admission koruması
etkin; bu aşamada yeni CUDA workload/reproduction/bypass sayısı **0**.

## 1. Koruma, kaynak ve önceki kanıtlar

Branch `feature/hyso-models-next`; HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş.
Başlangıç25 dirty dosya `start-protection.json` / `start.diff` ile kaydedildi.
Önceki CPU recovery test düzeltmeleri ve bütün eski raporlar korundu.
Yetkili minimal düzeltme yalnız mevcut `_saved_activation.py` ve
`test_saved_activation.py` üzerinde; diğer23 başlangıç dirty dosyası byte eşit.
Yeni repository dosyası yalnız bu audit'tir. Final26 dirty (10 modified,
16 untracked); kesin inventory/hashes `final-protection.json` içindedir.

Korunan `tests/test_gradient_accumulation.py` SHA256 başlangıç/son:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Dosyaya yazma/stage/restore yapılmadı. Reset/clean/restore/rebase/stage/commit/
push, dependency/Torch/driver değişikliği yok. Commit/release hazırlığı,
packaging yeniden üretimi veya genel native regression başlatılmadı.

Evidence **Eb**: `C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-b4-blocker-20261021T173434`.
Güncel CPU test source manifesti245 dosya:
`a968a19388bd7c60be2a150d972e777635b7110510d951119c33fd4fe03288a2` (`source-manifest.json`).
`phase1-source.patch`, pre-fix probe JSON'ları, XML/loglar,
`blocker-phase1-summary.json`, `offline-evidence-integrity.json`,
`prepared-native-proposal.json` ve `snapshot-validation.json` burada.

İncelenen ve değiştirilmeyen raporlar:

- [P23 R1 isolation/activation](m2-p-b10-p23-isolation-activation.md).
- [İlk native forensic](m2-p-b4-p23-native-crash-investigation.md).
- [Tek sealed diagnostic](m2-p-b4-p23-single-forensic-reproduction.md).
- [CPU preflight](m2-p-b10-p23-cpu-release-preflight.md).
- [Minimum CPU recovery](m2-p-b10-p23-minimum-cpu-regression-recovery.md).
- [Önceki SiLU R3 callpath](m2-p-b4-r3-native-callpath-investigation.md).

Önceki2825 minimum legacy,874 development CPU,58 P23 CPU,4 packaging ve47
production module byte-equality sonuçları **geçmiş kanıt** olarak korunur.
Özel activation kaynağı bu aşamada değişti; bunların bütünü yeni final-source
acceptance olarak yeniden etiketlenmedi. Çözülmüş9 minimum-runtime fixture
hatası yeniden araştırılmadı veya geri alınmadı.

## 2. İki native failure'ın ortak ve ayrışan imzası

|Kanıt|B4 SiLU olayı|P23 C dropout olayı|
|---|---|---|
|PID / dump|15644;295261910 byte|19264;295308564 byte|
|Exit/exception|0xC0000409,parameter2 STACK_COOKIE_CHECK_FAILURE|0xC0000409,parameter2; decimal3221226505|
|Driver|nvcuda64.dll32.0.15.9174 / driver591.74|Aynı version/header ve fault offset|
|Fault offset|0x18fd6e5,int29h|0x18fd6e5,int29h|
|Dar immediate path|SiLU forward dispatcher → cuLibraryLoadData → driver fail-fast|Dropout forward dispatcher → cuLibraryLoadData → driver fail-fast|
|P23 saved-hook gerekliliği|Bu olay P23 saved-activation implementasyonundan önce; o özellik çağrı zincirinde yok|Hook-enabled case etiketi var; failed callback/phase ledger yok|

İki original dump SHA256'sı yeniden salt-okunur doğrulandı:

- PID15644: `c68a154e30fd66b6408f68aa88afc7d41db550bec0b6ee756ad1c14292da4ef1`.
- PID19264: `384412ccf6cb95b6505b2ce97052d1742e505fb6f45df9de8272818d8140eb45`.

Dump kopyalanmadı/yüklenmedi; yeni native process veya CDB replay başlatılmadı.
Mevcut R3 ve C CDB/WER kayıtları kullanıldı. Kısa dispatcher/library export
frame'leri immediate path'i destekler; büyük-offset GraphRoot/CompileAndLaunch/
cuProfilerStop etiketleri backward,JIT veya profiler-stop kanıtı sayılmadı.
Aynı offset ortak root cause kanıtı değildir. Fail-fast stack cookie kategori
bilgisidir; memory corruption origin veya driver kusurunun tespiti değildir.

SiLU olayı, P23 hook'unun bu imza için genel olarak gerekli olmadığını gösterir;
farklı C olayında hook/timing etkisini dışlamaz. Exact native input image/PTX/
cubin,Python heap frame'leri,ilk bozulan write ve async önceki işler bilinmiyor.

## 3. Başarısız C / başarılı sealed diagnostic farkları

Eski C kaynağı **unverified_source_identity** kalır. Frozen B138-file manifesti
C source'ünü kanıtlamaz; `_saved_activation.py` B'de yoktu. C recipe/argv mutable
checkout yolu verir; import-time source hash yoktur. Sonradan alınmış source
hash/mtime/HEAD aynı loaded bytes varsayımı kurmak için kullanılmadı.

|Alan|Eski başarısız C|Sonraki başarılı tek diagnostic|
|---|---|---|
|Kaynak|Unverified mutable C|147 source +25 runtime hash;8 identity checkpoint|
|Sıra/process history|12 başarılı child sonucu (SGD6,Adam6),13. AdamW failure;parent harness history|Cold tek child fit;prefix0,baseline0;bağımsız Job/parent|
|Recipe|AdamW FP32/default,overflow false,reset true,seed71/epochs2;state_path digest wrapper seçili|Aynı dar model/batch alanları;state digest/checkpoint wrapper yok|
|Initialization|Failed child dropout/library-load'a ulaştı;pre/post CUDA/phase/window ledger yok|Parent initialized=false;child allocator.15 owned setup,stream0 kayıtlı|
|Hook/cleanup|Failed pack/unpack/release/forced/live sayıları bilinmiyor|60 pack/48 unpack/60 native release,0 forced/live;iki callback native thread|
|Reset|İzin var;kaç resetin tamamlandığı bilinmiyor|4 attempt/4 completion,retry/OOM delta0|
|Instrumentation|Worker stdout boş;warning dışında receipt/measurement yok|499 fsync plain metadata satırı;source/phase/cleanup/identity|

Yeni metadata I/O'su scheduling/destruction timing'ini etkileyebilir.
Cold tek PASS eski prefix/process history'sinin birebir replay'i değildir;
eski C import kimliğini veya root cause çözümünü kanıtlamaz.

## 4. Torch runtime / driver / lazy library-loading analizi

Yerel Torch2.5.1+cu121 CUDA build12.1 ve mevcut NVIDIA591.74 dosyaları önceki
sealed diagnostic'in25 runtime file hash'iyle eşleşiyor. Bu, şu anki disk
identity sürekliliğidir; eski C loaded-byte attestation veya bütün runtime/driver
kombinasyonlarının safety/compatibility kanıtı değildir. Driver/runtime değişimi
veya sürüm tavsiyesi bu veriden çıkarılmadı.

Yerel `torch/cuda/__init__.py:317–319`, `CUDA_MODULE_LOADING` unset ise
`LAZY` atayıp `_cuda_init()` çağırıyor. Bugünkü task host environment kaydı
unset; bu crash anındaki environment değildir. C için gerçek policy,driver'ın
ilk initialization zamanı veya yüklenen library identity kaydı yoktur.
`cuLibraryLoadData` library/code-loading yoludur; lazy-loading policy veya
başarılı kernel/JIT'in tek başına ispatı değildir. Yeni öneri bu Python kaynağını
runtime manifestine dahil eder ve CUDA admission/setup öncesi/sonrası environment
string'ini metadata olarak kaydeder. Policy değiştirilmez; EAGER,launch-blocking,
CUDA cache/driver/sync workaround uygulanmadı.

## 5. Ownership, destructor, weakref ve reentrancy

Native saved slot → `_Payload` → original Tensor → grad_fn → saved slot
çevrimi mümkündür. Weak `recorder.slots`, bu bütün zinciri weak yapmaz.
Yeni CPU testi GC-disabled iken abandoned sigmoid output'un normal Python
referansları bırakıldıktan sonra native cycle ile kaldığını; `result()` bunu
retained-graph unsupported diye reddettiğini ve `close()`'un modeli/leaf'i
serbest bıraktığını gösterdi. Forced cleanup native payload disposal veya
physical storage/device-free kanıtı sayılmadı.

`clear()` owner'ı önce None yapar, sonra original tensor'u bırakır, sonra
metadata release yapar. Tensor destruction sırasında callback ile nested
`close()` tetikleyen CPU testinde slot/storage counters0, bir native release
ve bir forced cleanup; tekrar close/clear idempotent; unraisable exception0.
`clear()` ve `__del__` AST'si başlangıçla aynıdır. Bu kanıtta double-release
veya cleanup accounting defect'i bulunmadığı için sırayı tahminen değiştirmek
uygun görülmedi. Bir CPU callback case'i bütün CUDA graph lifetime/reentrant
native destruction yollarının güvenliği değildir.

Native hook scope exit mevcut saved slots'u yok etmek zorunda değildir.
Worker finally recorder.close → activation.close sırasını korur; başarısız
native fail-fast'te Python finally çalıştığı iddia edilmez. `_cdata` yalnız
storage identity metadata'sıdır; manuel pointer free/write bulunmadı.

## 6. Kanıtlanmış HYSo defect: saved-version rejection bypass

**Native crash root cause'undan bağımsız, gerçek production adapter defect'i.**
Modüllerin `inplace=True` flag'ini reddetmek, functional in-place mutation veya
sonradan parameter mutation'ını kapsamaz. Hook return yolu native saved-variable
version rejection'ını garanti etmiyor; HYSo unpack bu sınırı kontrol etmiyordu.

Her iki runtime CPU probe'unda iki case:

1. Sigmoid forward output → no-grad in-place add → backward.
2. Linear forward'da saved weight view → no-grad parameter add → backward.

Baseline native autograd `modified by an inplace operation` RuntimeError verdi;
aynı mutation HYSo hook açıkken **hatasız backward/gradient creation** yaptı.
`pre-fix-version-minimum.json` ve `pre-fix-version-development.json` input/version
artışlarını ve sonuçları kaydeder. Yeni regression testleri patch öncesi
**2 FAIL /2 PASS** verdi; bu yeni defect önce CPU'da gösterildi.

Bununla birlikte failure dump immediate stack'leri forward/library-load'dadır;
failed C kaynak kimliği ve saved tensor mutation kanıtı yoktur. Bu bug'ın
`0xC0000409` ürettiği veya düzeltmenin driver crash'i giderdiği **kanıtlanmadı**.

## 7. Uygulanan en küçük güvenli düzeltme

`_saved_activation.py`:pack admission'da tensor `_version` integer'ı accounting
mutasyonundan önce alınır; `_Payload.version` içinde saklanır. Unpack'te current
version eşit değilse RuntimeError verilir. Normal orijinal tensor aynı nesne
olarak döner. Native TensorImpl header'ındaki atomic version-counter yapısı da
salt-okunur incelendi; bu DLL/PDB satır attestation'ı değildir.

Yeni alan yalnız internal integer; wire schema/public API değişmedi. Tensor
clone/detach/copy,weak-owner değişimi,global production GC,empty_cache,ek
synchronize,optimizer step veya RNG/gradient matematiği değişikliği yok.
Mutation içermeyen exact state/RNG/resume kontrolleri aşağıda yeniden geçti.
Düzeltme tensor corruption'ı onarmaya çalışmaz; native error boundary'yi korur.

Yeni4 test:output/parameter mutation rejection2,GC-disabled abandoned-cycle
cleanup1,reentrant idempotent release1. GC kontrolü yalnız bounded test scope'unda
restore edilerek uygulanır; production GC workaround'u değildir.

## 8. Güncel kaynak CPU sonuçları ve bağımsız A/B/C kararları

Source freeze sonrası245 girdinin pre/post hashleri eşit. Minimum
Python3.10.18/Torch2.2.2+cpu;development2.5.1+cu121 runtime'ında yalnız CPU yolları.
Torch/torchvision origin/sys.path ayrı runtime JSON'larında;development native
CUDA discovery/init girişleri fail-closed;initialized=false ve blocked entry0.

|Koşu|Sonuç / kanıt|
|---|---|
|Yeni hedefli4,minimum|4 PASS;3.451s|
|Yeni hedefli4,development CPU|4 PASS;3.021s|
|Tam P23 CPU isolation/activation|**62 PASS =29 isolation +33 saved activation**;242.831s;p23-minimum.xml|
|Development doğrudan activation/lifetime|**13 PASS**;3.920s;c_direct-development.xml|
|İlgili legacy CPU memory/replay|**108 PASS =71 legacy memory +37 replay**;44.585s;related_legacy-minimum.xml|

Final koşularda FAIL/ERROR/SKIP0; hedefli scope dışı deselect'ler aggregate PASS'a
eklenmedi. CPU native exception,GC-disabled weakrefs,exact state/RNG,
checkpoint interrupted/resume/completed,JSON/CSV/Manager/immutable/tamper
kontrolleri P23 suite içinde yeniden doğrulandı. Tarihsel2825/874/4 packaging
sonuçları yeni sonuçlara eklenmedi veya yeniden çalışmış gibi sunulmadı.

|Aşama|Karar|
|---|---|
|A isolation|Gözlenen Windows CPU worker scope PASS (fresh29);genel native acceptance ayrı|
|B gerçek CUDA window peaks|Frozen historical21 PASS korunur;bu görevde yeni CUDA gate yok|
|C activation|CPU33 PASS ve version defect'i düzeltilmiş;CUDA native failure OPEN,admission disabled,feature PARTIAL|
|B4|OPEN/UNRESOLVED;native root cause yok|
|P23/B10|PARTIAL/PARTIAL;R1 NO-GO korunur|

## 9. Allocator reset / stream / Job analizi

Owned reset sırası selected-stream synchronize → previous snapshot → peak
counter reset → after snapshot;attempt end synchronize → actual peaks/delta.
Peak reset counter domain'idir;Tensor free/empty_cache yolu olarak yorumlanmadı.
Yeni version fix reset/stream/measurement sırasına dokunmaz. Timing/lazy-loading
etkileşimi olasılığı native execution olmadan doğrulanmadı veya dışlanmadı.

C native receipt admitted=true,monitor=null,~2.016GB peak process commit4GiB
altında;Job active0. Parent assertion exit1 ayrı olaydır. Windows exception,
WER ve dump fail-fast'i destekler;resource-limit/timeout/Job kill root cause'u
kanıtlanmadı. Process-tree cleanup Python payload veya CUDA free lifetime
kanıtı sayılmadı. `_training_process.py` ve Job class byte'ları değişmedi.

## 10. Hazırlanan tek sonraki native control — ONAY YOK / ÇALIŞMADI

**Tek hipotez:**Shared forward/library-load fail-fast, saved-tensor hook'ları
olmadan da oluşabilir. Yeni control **hook-off / reset-on**'dur. Bu, eski C'nin
birebir veya tek-değişkenli paired replay'i değildir. Crash aynı dar path ile
olursa yeni gözlem için hook gerekliliği dışlanabilir;PASS yalnız cold hook-off
case acceptance olur,root cause çözümü veya eski C hook causality kanıtı olmaz.

Read-only hash-sealed snapshot:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-b4-blocker-20261021T173434/proposed-hook-off-control-r2`.
**251 source/test/factory/bootstrap/config/doc +26 runtime/interpreter/DLL file**.
Manifest SHA256:
`2be2ebe1562bfc0863b16dc6d1bb093f91f32f1dd9ab8b456ed94387cedbd78d`.
Current patched payload ve Job source byte'ları checkout ile aynı. Isolation
metadata çağrıları strip edildiğinde AST checkout ile aynı;guard raise aynı,
**bypass yok**. Parent/child aynı manifesti pre/post import/run ve factory-import
noktalarında doğrular;source/runtime mismatch admission reject.

Controller ayrı `explicit-user-approval.json` ve<=30s gerçek live-preflight
kaydı olmadan ilerlemez. Approval absent refusal CPU stdlib-only kontrolü
`APPROVAL_REQUIRED_NO_WORKLOAD_STARTED` verdi;Torch import/Job/child başlamadı.
Attempt marker0,worker file0,event row0,CUDA workload0. Önceki tek diagnostic
onayı yeniden kullanılmadı. Ön hazırlık snapshot r1 de değiştirilmeden korunur;
r2 current CUDA init policy source identity/metadata'sını ekleyen aktif öneridir.

|Alan|İzin istenecek tek control|
|---|---|
|Recipe|AdamW FP32,default cuda:0 stream,AMP/overflow false,seed71,epochs2;5 sample,batch2,accum2;Linear–Dropout fixture|
|Saved hooks / peak reset|False /True;production guard aynen etkin|
|Execution bütçesi|1 child fit,0 baseline,0 prefix,0 retry;attempt marker exclusive-create|
|Child sınırları|60s;4GiB/process,8GiB/Job,4 process;allocator fraction.15|
|Outer sınırları|90s;4GiB/process,12GiB/Job,8 process;kill-on-close|
|Output|64KiB/channel stdout/stderr;2MiB result;process başına2000 metadata row/1MiB;fsync plain metadata|
|Live preflight|Yeni3 GPU utilization sample<%35,free GPU>=2048MiB,host RAM>=4GiB,disk>=12GiB,conflicting workload0;limit artırma yok|
|Admission|Empty child Job/readback ve outer Job assignment Torch/workload importundan önce;failed admission stop|
|Evidence|PID/thread,phase,epoch/microbatch/window/attempt/stream,reset attempt/complete,cleanup;hook-off configured counters0;pre/post loader env string|
|Stop|Native crash veya missing identity/diagnostic/timeout →STOP;exit/WER/dump/log/snapshot koru,owned PID/Job active0 doğrula;ikinci case/retry yok|

WER/dump toplama workload timeout'tan ayrı,bounded30s post-observation olarak
planlandı. Global dump policy/registry değişikliği veya dump upload yok;ek dump
policy gerekirse ayrıca onay alınır. Tensor/graph referansı tutan diagnostic,
copy,ek optimizer step veya synchronization workaround yok. Snapshot read-only
attributes+hash admission ile sabittir;adversarial security sandbox değildir.
Sistemin live durumu bu ilk aşamada ölçülüp geleceğe geçerli sayılmadı;onaydan
sonra fresh preflight ve gerçek kernel limit readback şarttır.

## 11. İlk teslim sorularının yanıtı ve commit öncesi eksikler

1. **Native root cause belirlendi mi?Hayır.** Library-load immediate path ve
   ortak fail-fast imzası var;corruption origin/source identity eksik.
2. **Production HYSo defect bulundu mu?Evet.** Opt-in saved hook native
   in-place saved-version rejection'ını atlıyordu;CPU2.2.2/2.5.1'de gösterildi.
3. **Düzeltme?** Internal pack-version snapshot/unpack validation;ownership,
   destructor,allocator,training matematiği ve production CUDA guard aynı.
4. **CPU sonuçları?** Fresh P2362 PASS,related legacy108 PASS,development
   direct13 PASS;new regression4 her iki runtime'da PASS;örtüşen sayılar toplanmaz.
5. **Sonraki tek native hipotez?** Fail-fast saved hooks olmadan da oluşabilir;
   yalnız hazırlanmış sealed hook-off/reset-on control için ayrı onay gerekir.
6. **Kararlar?** B4 OPEN/UNRESOLVED,P23 PARTIAL,B10 PARTIAL,R1 NO-GO.
7. **Eksik zorunlu gate'ler?** B4 causality/safety acceptance;current final-source
   ve installed CUDA B/C matrix,saved activation native noninterference/lifetime,
   Conv/tied/early/resume/GC/exception coverage ve ilgili native regression'lar.
   Version fix sonrası47-module wheel/sdist/installed equality,min installed
   smoke ve final frozen-source full CPU regression hâlâ final commit/release
   gate'leridir;bu blocker aşaması onları yapmış gibi sunmaz. P21/B10 diğer
   partial sınırları da durur.

Bu aşama burada durur. Genel commit/release GO yok;stage/commit/push yapılmadı.
Yeni native workload veya guard değişikliği ancak kullanıcının ayrıca açık
onayı ve ilgili preflight ile mümkündür.
