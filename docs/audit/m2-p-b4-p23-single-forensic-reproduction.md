# B4 / P23 R1 — tek bounded forensic diagnostic sonucu

2026-10-21, Europe/Istanbul. **Tek yeni diagnostic: PASS; native crash yok.**
**Root cause UNRESOLVED; B4 NO-GO / OPEN; P23 PARTIAL; B10 PARTIAL.**
Bu sonuç önceki failure'ı kapatmaz. Production CUDA saved-activation guard
korundu. Genel CUDA matrix, regresyon, P23 completion veya feature geliştirmesi
başlatılmadı. Kullanıcının verdiği tek-deneme onayı tüketildi; ikinci koşu yok.

## 1. Kapsam, başlangıç ve preflight

Branch `feature/hyso-models-next`, HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş. Önceki15 görev dosyası,
[ilk forensic rapor](m2-p-b4-p23-native-crash-investigation.md) ve korunan
gradient dosyası dahil17 mevcut worktree dosyası başlangıçta hash'lendi.
İlk forensic raporun ve önceki failure/dump evidence'ının hashleri doğrulandı.
Production guard'ın mevcut olması ve source byte'ları ayrıca kontrol edildi.

Yeni evidence dizini **R**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23r1-20261021/single-repro-20261021T125035Z`.
Önceki evidence değiştirilmedi. Kaynak hazırlama, statik parse/hash incelemesi ve
salt-okunur host/GPU sorguları CUDA workload sayısına dahil değildir; bu aşamalarda
HYSo eğitimi veya CUDA allocation probe'u çalıştırılmadı.

Preflight: yaklaşık42GiB boş disk, toplam15.36GiB host RAM ve son admission'da
4644896KiB (yaklaşık4.43GiB) boş RAM. Python/torchrun/ollama/ffmpeg/cdb/blender
yoğun workload'u bulunmadı. GPU RTX3060 Laptop,6144MiB, driver591.74. GPU'daki
process listesi masaüstü uygulamalarını gösteriyordu; izin nedeniyle adı
okunamayan PID1992 ayrıca `dwm.exe` olarak doğrulandı. Önceki bir desktop ölçümü
%65 transient spike içeriyordu; son üç admission ölçümü %9/%9/%9, kullanılan
GPU belleği860/600/872MiB idi. Son admission, GPU<%35 ve boş RAM>=4GiB/disk>=12GiB
gibi konservatif koşullarla geçti. Bu ölçümler genel bir sistem izolasyonu
garantisi değildir. Kullanıcı process'i öldürülmedi.

Windows Job kontrollerinin hazır olduğu, CUDA başlamadan boş child Job'un
gerçek kernel limit readback'iyle doğrulandı. Dump registry/politikası
değiştirilmedi. Mevcut WER minidump mekanizması korundu; full Python heap dump
elde edileceği varsayılmadı. Bu denemede crash olmadığı için dump toplama gereği
doğmadı. Preflight kayıtları `preflight-live.json`, `final-live-admission.json`
ve `empty-child-job-readback.json` içindedir.

## 2. Gerçekten çalıştırılan senaryo sayısı

**Bir attempt, bir CUDA workload, bir child recipe, bir gerçek Trainer.fit.**
Parent baseline fit sayısı0; retry,12-case prefix veya başka karşılaştırma yok.
`attempt-consumed.json` ve `child-launch-consumed.json` exclusive-create ile
ikinci başlatmayı reddedecek biçimde yazıldı; ledger'da yalnız bir
`single_fit_enter` ve bir `single_fit_returned` var.

| Parametre | Gerçek koşu |
|---|---|
| Optimizer / precision |AdamW / FP32, AMP=false |
| Device / stream |cuda:0 / default stream, native stream ID0 |
| Overflow |false |
| Peak reset / saved activation |explicit enabled / yalnız forensic kopyada enabled |
| Seed / epoch |71 /2 |
| Dataset / batch / accumulation |5 sample /2 /2 |
| Model |Linear(3,4) → Dropout(.2) → Linear(4,2) |
| Native fixture davranışı |CrossEntropy, EMA.8, OneCycle4 steps, grad_clip.1, shuffle ve validation korunmuş |
| Gerçek ledger |6 microbatch,4 attempt,4 applied,0 skipped,4 EMA update; window samples `[4,1,4,1]` |

Bu koşuda ModularCNN, EarlyStopping, checkpoint/resume veya state digest export'u
yok. Instrumentation için tensor/gradient CPU copy'si eklenmedi. Önceki fixture'ın
state_path digest wrapper'ı seçilmedi; extra eğitim veya baseline yapılmadı.

## 3. Immutable kopya, SHA256 manifesti ve import identity

Çalışma yalnız R/source altındaki bağımsız kopyada yürüdü. Kaynak dosyalarına
read-only attribute uygulandı; bytecode yazma kapalıydı (`-I -S -B`). Bu,
adversarial bir güvenlik sandbox'ı değil, doğrulanan sabit diagnostic snapshot'tır.

Manifest: **147 kaynak/test/factory/bootstrap/config dosyası +25 runtime dosyası**.
Runtime listesinde Python interpreter/DLL'leri, Torch version/graph/memory/functional
kaynakları, Torch native DLL'leri, c10/CUDA runtime/cuBLAS/cuDNN ve NVIDIA driver
DLL identity hashleri ile mevcut Torch/NumPy metadata/RECORD dosyaları bulunur.
Manifest SHA256:

`3b24a24a053c0ffd90125fa29952212bb412cd0a1a90eaeae31c5ee4b4835b21`

Parent/child Python3.10.18, Torch2.5.1+cu121, CUDA build12.1 kullandı. Native
interpreter `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe` idi; venv redirector
üzerinden kontrolsüz yeni interpreter açılmadı.

Toplam8 identity checkpoint'i aynı manifesti doğruladı:

- Controller: prelaunch.
- Parent: pre-import, post-import, post-run.
- Child: pre-import, post-import, factory-import ve post-run.

Her checkpoint147 source ve25 runtime dosyasını yeniden hash'ledi. Import sonrası
HYSo/forensic/factory module yollarının snapshot içinde ve manifestte olması
kontrol edildi. Request aynı run-ID ve manifest hash'ini taşıdı; bir mismatch
workload admission'ını reddedecekti. Identity JSON'ları ve immutable kaynaklar
R'de korunuyor. Dosya pre/post hashleri ve module paths güçlü snapshot kimliği
kanıtıdır; her runtime binary'nin canlı bellekteki bütün byte'larının ayrı
attestation'ı veya bütün dependency tree'nin manifesti olduğu iddia edilmez.

Forensic guard istisnası bu run-ID, manifest ve tam izin verilen recipe/protocol
eşleşmesiyle sınırlıydı; production checkout'ta bypass yapılmadı. Metadata çağrıları
çıkarıldığında payload class yöntemlerinin AST gövdelerinin orijinallerle aynı
olduğu, Job class gövdesinin de değiştirilmediği statik olarak kontrol edildi.
Patch farkı R/forensic-source.patch'tedir. Metadata logging timing'i etkiler;
eski crash'in birebir replay'i değildir. **Eski C kaynak kimliği hâlâ
`unverified_source_identity`**; yeni manifest geçmişte yüklenen kaynakları kanıtlamaz.

## 4. PID, Job admission ve timeout kontrolleri

| Rol | PID / sonuç |
|---|---|
| Stdlib controller |7496 / exit0 |
| Outer Job içindeki parent |18532 / exit0 |
| Inner Job training worker |22448 / exit0 |

Parent admission'dan önce Torch import etmedi. Child Job admission'ı da
HYSo/Torch import ve recipe'den önceydi. Readback flags8968: kill-on-close,
process-count ve private-commit limitleri etkin.

| Limit | Child Job | Outer Job |
|---|---|---|
| Hard timeout |60s |90s |
| Private commit / process |4GiB |4GiB |
| Private commit / Job |8GiB |12GiB |
| Active process ceiling |4 |8 |
| Gözlenen elapsed |19.75s |34.0s |
| Monitor outcome |null |null |
| Peak process commit |2545172480 byte |2545172480 byte |
| Peak Job commit |2546495488 byte |4174798848 byte |
| Total processes / final active |4 /0 |7 /0 |

Total process sayıları import/helper/descendant accounting'idir;4 veya7 eğitim
senaryosu değildir. CUDA native allocator fraction.15, model construction öncesi
worker'da ayarlandı. Limitler yükseltilmedi. Private commit physical RAM değildir;
allocator fraction physical VRAM limiti değildir.

Stdout/stderr tutulması kanal başına64KiB; child result2MiB ile sınırlıydı.
Metadata dosyası process başına2000 record/1MiB, identity/diagnostic publication
2MiB ile sınırlıydı. Metadata append'leri fsync edildi; request/ready/result
dosyaları R/worker altında kalıcı korundu. Child result332086 byte,
diagnostic-result713435 byte; child stdout0/stderr305 byte, yalnız mevcut
GradScaler deprecation warning'i var. Outer stdout/stderr0 byte.

## 5. Son durable phase ve yürütme zamanları

21 Ekim 2026, Europe/Istanbul / UTC+03:00:

| Durable olay | Saat |
|---|---|
| Child Job admitted, import öncesi |16:00:01.858694 |
| Source/import doğrulamasından sonra recipe giriş |16:00:13.058629 |
| Yalnız kopyadaki authorized guard override |16:00:13.068060 |
| Factory construction giriş/çıkış |16:00:15.581271 /15.595271 |
| Tek gerçek fit giriş/dönüş |16:00:15.600062 /18.393466 |
| Activation cleanup giriş/çıkış |16:00:18.407007 /18.418012 |
| Recipe dönüş |16:00:18.422011 |
| Son child durable record: worker_publication_exit |16:00:20.615316 |

Crash gerçekleşmedi; dolayısıyla “crash öncesi son phase” yoktur. Son eğitim
semantik phase'i `single_fit_returned`, son child durable phase
`worker_publication_exit` idi. Son source context: epoch2/microbatch6/window4/
attempt4/stream0. Bu context son gözlenen eğitim ID'leridir; publication'ın
kendisine yeni bir training microbatch atfetmez. Metadata499 record/200631 byte
ile sınırlar içinde kaldı. Durumlar tam ledger'da R/events/child-22448.jsonl'dedir.

## 6. Saved-tensor event ledger

| Olay | Sayı |
|---|---:|
| Pack |60 |
| Unpack |48 |
| Native payload release |60 |
| Forced cleanup release |0 |
| End live payload |0 |

12 saved record unpack edilmeden native release edilmiş; pack/unpack sayısının
eşit olmaması tek başına eksik lifecycle değildir. Bu12 record'un da release
kanıtı var. Parent, mevcut plain activation/window validator'larını ve source
sample/update ledger'ını doğruladı; bu full public result/package acceptance
suite'i olarak sayılmadı.

Callback'ler child PID22448 içinde iki native thread'de gözlendi:

- Thread2620:60 pack ve12 release.
- Thread22380:48 unpack ve48 release.

Bu ölçüm native backward sırasında callback/release'in tek main thread'e
indirgenemeyeceğini gösterir. Tek başına data race veya crash nedeni kanıtı
değildir. Metadata yalnız plain scalar/string/ID/dict tuttu; tensor, graph,
exception traceback veya Python frame nesnesi diagnostic registry'ye eklenmedi.
Original payload tensor ownership sırası, unpack dönüşü ve optimizer matematiği
değiştirilmedi. Global GC, empty_cache veya ek optimizer step yoktu.

Total saved logical bytes880, peak live saved logical152, unique saved
storage136 ve non-parameter storage104 byte idi. Bunlar bu tiny workload'un
saved-payload logical/storage kayıtlarıdır; activation peak VRAM veya fiziksel
storage free/lifetime ölçümü değildir. Lifetime kapsamı host autograd packed
payload retention olarak partial sınırlarını korur. Bare baseline çalıştırılmadığı
için logging toplam overhead'i veya numerical noninterference için yeni replay
kanıtı üretilmedi.

## 7. CUDA allocator ve reset evidence

Reset attempt4, completed reset4. Bütün windows worker PID22448/run-ID'ye ait,
stream0, reset_performed=true; source samples `[4,1,4,1]`. Mevcut selected-stream
synchronize/read/reset sırası korundu. Metadata dışında ek CUDA synchronization
workaround'u eklenmedi.

| Window / attempt | Sample | Actual peak allocated byte | Peak reserved byte | Incremental allocated byte |
|---|---:|---:|---:|---:|
|1 /1 |4 |68166656 |69206016 |68162560 |
|2 /2 |1 |68168704 |69206016 |1050112 |
|3 /3 |4 |68170752 |69206016 |1052160 |
|4 /4 |1 |68168704 |69206016 |1050112 |

Her window allocation-retry delta0, OOM delta0. Reset öncesi/current/after-reset/
end allocator tuple'ları ve completion wait süreleri tam worker result'ta
korunuyor. Bunlar allocator domain'idir; physical/device-wide VRAM veya leak
attribution iddiası yoktur. Parent CUDA initialized=false kaldı; parent baseline
ve parent allocator reset'i yoktu.

## 8. Native exit, WER ve dump

Native crash **gerçekleşmedi**. Child exit0, parent exit0, controller exit0;
child/outer timeout yok. Outcome `success`; unhandled Python exception veya
native-crash classification yok. 12:59:48.2149776Z–13:01:24.2470516Z Application
Error/WER1000/1001/1002 sorgusunda bu child/parent PID'leri için eşleşme0.
Bu dar sorgunun sınırı R/windows-outcome.json'da kaydedildi.

CrashDumps envanterinde yeni Python dump yok; yalnız eski PID15644 ve19264
dump'ları bulunuyor. Eski dump hashleri korunuyor. Yeni WER/dump capture süresi
**uygulanamaz**; elapsed19.75/34.0s değerleri bir crash capture süresi olarak
sunulmaz. Registry/genel dump politikası değişmedi; dump dışarı yüklenmedi.

## 9. Parent/child cleanup

Inner ve outer Job accounting final active_processes0. Kaydedilmiş kill-on-close
termination/close giriş-çıkışları ve receipt'ler R'de. Sonradan PID22448/18532/7496
Windows process sorgusunda canlı eşleşme yoktu. Parent flag-only CUDA initialized
kontrolü false; GPU training yalnız child'daydı.

Child hook/recorder/activation cleanup giriş ve çıkışları durable. Native release60,
forced release0, live payload0. Process cleanup ile tensor/storage fiziksel
lifetime birbirinin kanıtı sayılmadı. Descendant'ların ayrı creation PID listesi
yakalanmadı; kernel Job active0 bütün owned process accounting'ini kapsıyor.

## 10. Crash gerçekleşti mi ve root cause durumu

**Hayır. Bu diagnostic PASS; önceki crash çözüldü demek değildir.** Yeni koşu
sabit, post-stop C source + ek durable instrumentation ile yapıldı; eski mutable
C byte identity doğrulanamıyor. Prefix12 yeniden çalıştırılmadı. Cold tek-case
koşulu, önceki failure'ın parent/prefix history'si ve timing'iyle eşdeğer değil.
Ek metadata I/O'su da scheduling/destruction timing'ini etkileyebilir.

Önceki `_Payload` graph-cycle/reentrant destructor riskleri statik olarak
yerinde; bu koşunun başarıyla bitmesi onların bütün CUDA yollarında güvenli
olduğunu kanıtlamaz. Driver/library-loading defect'i veya HYSo/PyTorch kusuru
kanıtlanmadı. **Root cause UNRESOLVED**; önceki`0xC0000409` failure kayıtları aynen
korundu, sonradan PASS olarak yeniden etiketlenmedi.

## 11. Sonraki güvenli karşılaştırma önerisi

Bu görev kapsamında yeni koşu yok. Önce mevcut iki crash dump'ının
cuLibraryLoadData/dropout–SiLU return-address zincirleri exact Torch build
symbol/map verisiyle offline daraltılabilir. Yeni ledger'daki iki callback
thread'i ve native release sırası da yalnız metadata üzerinden incelenebilir.

Runtime kontrol gerekli görülürse aynı sealed source/limitlerle yalnız bir
değişkeni ayıran ek diagnostic, örneğin hook-off/reset-on control, ayrı bir
hipotez ve ayrı kullanıcı onayıyla tasarlanmalıdır. Bu rapor o karşılaştırmaya,
prefix tekrarına, guard değişikliğine veya genel matrix'e onay değildir.
Şu an tek başarılı ölçümden yeni runtime denemesi zorunluluğu çıkarılmadı.

## 12. B4 / P23 / B10 kararları

| Alan | Güncel karar |
|---|---|
| Bu tek diagnostic |PASS,1 execution, crash/timeout yok |
| Önceki B4 / C failure |FAIL kaydı korunur; root cause UNRESOLVED |
| B4 |NO-GO / OPEN |
| P23 |PARTIAL; production CUDA activation guard etkin |
| B10 |PARTIAL; full GO yok |

P23 implemented veya stable release ilan edilmedi. Genel regresyon, packaging,
minimum runtime, AMP/başka optimizer/stream/seed senaryosu çalıştırılmadı.

## 13. Koruma ve kesin değişiklik kapsamı

Repository'deki yeni görev dosyası yalnız bu devam raporudur. Önceki15 dosya,
ilk forensic rapor ve user gradient farkı byte seviyesinde korundu. Forensic
source/metadata/runtime manifestleri Temp altındaki R'de; production source
dosyalarına diagnostic patch veya guard bypass uygulanmadı. Stage/commit/push,
reset/checkout/clean ve dependency/driver değişikliği yapılmadı.

Manifest/identity/ledger/Job/Windows/cleanup özetleri R/observed-summary.json ve
diğer named evidence dosyalarındadır. Başlangıç/final protection ve evidence
hash manifesti de R'de tutulur; önceki source/dump kanıtı güncellenmedi.

## 14. Korunan kullanıcı dosyası SHA256

`tests/test_gradient_accumulation.py` başlangıç/son:

`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`

Beklenen hash ile aynı. Tek-deneme onayının kapsamı tamamlandı; yeni CUDA
workload'u veya production guard değişikliği için yeni açık onay gerekir.
