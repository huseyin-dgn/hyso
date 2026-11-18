# Stage 13.5.23 — CUDA B4 incelemesi ve savunmacı düzeltme

2026-11-18, Europe/Istanbul. **ROOT_CAUSE_UNRESOLVED / DEFENSIVE_FIX_ONLY /
GPU_ADMISSION_BLOCKED.** Gerçek CUDA workload veya saved-activation hook bağlamı
çalıştırılmadı. CPU beta ve tarihsel A/B 35/35, P23 Basic CUDA ve hooks-OFF
başarıları korunur; bu rapor onları yeniden test edilmiş saymaz.

| Aşama | Sonuç | Kanıt / kalan eksik |
| --- | --- | --- |
| A — Root cause investigation | PARTIAL | Immediate native sınır biliniyor; ilk bozucu işlem ve kesin failed-source kimliği bilinmiyor |
| B — Source remediation | PASS; DEFENSIVE_FIX_ONLY | Owned pipe reader hatası CPU mock ile üretildi ve düzeltildi; B4 ile nedensel bağı yok |
| C — CPU-only validation | PASS | 42 pytest + 25 stdlib unittest, 0 final FAIL; CUDA doğrulaması değildir |
| D — Fresh GPU admission | BLOCKED | Güncel PID 1968 kimlik kısıtı ve compute/C/G farkı; gerçek GPU execution yok |

## A — Hatanın sınırı ve hipotezler

[2026-11-03 audit'i](../audit/p23-native-crash-root-cause-audit-20261103.md),
[P23 native incelemesi](../audit/m2-p-b4-p23-native-crash-investigation.md),
[erken B4 çağrı yolu](../audit/m2-p-b4-r3-native-callpath-investigation.md) ve
[hooks-OFF kontrolü](../audit/m2-p23-b4-r3-wrapper-fix-cuda-diagnostic.md) okundu.
Bunların native stack/WER yorumları bu aşamada rapor kaynaklı kanıttır; yeni
dump/debugger analizi değildir. T07'nin kayıp beş ham dosya ve üç dump araması
tekrarlanmadı; kayıp dosyalar mevcut gösterilmedi.

P23 başarısız worker PID 19264: AdamW fp32, default stream, hooks ON/reset ON,
seed 71, iki epoch; exit `3221226505 / 0xC0000409`, monitor null, measurement ve
worker publication null. Tarihsel WER/dump incelemesi subcode 2 ve
`dropout forward -> cuLibraryLoadData -> nvcuda64+0x18fd6e5` immediate yolunu
bildirir. Fail-fast tespit noktası ilk bozucu yazmanın yeri değildir. Hangi
epoch/microbatch/step olduğu bilinmez; ilk dropout veya optimizer step denemez.
Python/native sınırı bu yola daralır; kesin Python satırı/private frame yoktur.

Erken B4 PID 15644'te SiLU/library-load imzası P23 hook implementasyonundan
öncedir. İki olayın aynı kök nedeni olduğu kanıtlanmaz. Hooks-OFF/reset-ON
kontrolünün başarısı yalnız o koşulu doğrular; eşlenmiş nedensel deney değildir.

| İncelenen mekanizma | Kaynak ve değerlendirme |
| --- | --- |
| Hook registration/removal, tensor lifetime, reentrant/double cleanup | `hyso/performance/_saved_activation.py`: native slot payload'ı, payload özgün tensorü güçlü referansla tutar; recorder weakref tutar. Owner-null idempotence, alias ve reentrant release doğrudan CPU testleri geçti. CUDA callback/dangling-reference güvenliği kanıtlanmadı. |
| Hook scope, kapanış sırası, exception sonrası cleanup | `isolated_training_memory.py:_run_recipe`, `training.py:_measure_training`, `training_memory.py:_MemoryRecorder`: fit scope, sampler kapanışı, seçilmiş stream completion ve outer cleanup incelendi. Native fail-fast'te Python finally çalışmış varsayılmaz. |
| Device/stream, async execution, GPU memory ownership, reset | `_IsolatedRecorder`: açık worker device/stream; reset öncesi/sonrası seçilmiş stream sınırları. Peak-stat reset tensor/context destruction değildir. Hidden stream ve fiziksel VRAM serbestliği gözlenmez. Kanıtsız sync/allocator/reset değişikliği yapılmadı. |
| Child startup, context ve Job/PID sahipliği | `_training_memory_worker.py`, `_training_process.py`: `-I -S` bootstrap, Job assignment sonrası token/PID admission, sonra HYSo/Torch importu; yalnız oluşturulan Popen/Job cleanup. Envanter PID'lerine termination yok. |
| Timeout/cancellation, native exit, parent communication | `_launch`, `isolated_training_memory.py:_classify`: deadline/log/result sınırları ve Job cleanup ayrı; abnormal exit success publication ile örtülemez. Aşağıdaki pipe-error iletim hatası bulundu. |
| Native extension/driver boundary, invalid cleanup state | Mevcut immediate stack teardown/hook destructor/Job termination göstermiyor. Önceki async iş veya cleanup etkileşimi dışlanamaz; driver/PyTorch kusuru atanamaz. |

**B4'ün kesin kök nedeni bulunmadı.** Başarısız mutable kaynak için pre-crash
attestation, son durable faz/hook/reset sayacı, yüklenen library image/options ve
yeterli native/Python heap/private-symbol bağı eksik. Güncel CPU davranışı bu
tarihsel boşluğu doldurmaz. Production CUDA saved-activation guard'ları değişmedi.

## B/C — Kanıtlanan ayrı hata ve CPU doğrulaması

`_training_process.py:_launch` içindeki daemon stdout/stderr drain thread'leri,
`read` veya `close` exception'ını parent'a taşımıyordu. Thread sonlanabildiği için
`is_alive()` kontrolü geçiyor, exit 0 gözlenen bir child için normal launcher
receipt'i dönülebiliyordu. Bu, eksik capture'ın gizlenmesi riskidir; tarihsel B4
koşusunda böyle bir I/O hatası olduğuna dair kanıt yoktur.

İlk dört regression case mevcut kaynakta **4 FAIL / 4 unhandled-thread warning**
üretti. Düzeltme yalnız bounded channel/operation/error metni tutar, yaşayan
owned child'ın durdurulmasını tetikler ve mevcut Job cleanup/join/close
kontrollerinden sonra parent `RuntimeError` üretir. Exception mesajı gözlenen
worker exit kodunu da korur. Traceback/tensor tutulmaz; schema, native exit
sınıflandırması, retry, memory/timeout bütçeleri ve hook guard'ları değiştirilmez.
Bu, B4 çözümü değil **DEFENSIVE_FIX_ONLY** durumudur.

| Son test grubu | PASS / FAIL | Kapsam |
| --- | --- | --- |
| CPU pytest | 42 / 0 | İki kanal × read/close × exited/running child × exit 0/native fail-fast; normal constructor-failure worker/cleanup; native/signed exit karşısında sahte publication; recipe/protocol, gerçek Job flags, CUDA guard sırası, doğrudan payload alias/reentrant release |
| Stdlib admission unittest | 25 / 0 | Freshness, UUID/run/manifest bağları, PID review/identity drift, kaynak/runtime, limit readback ve malformed/unknown inventory reddi; Torch import edilmedi |

CPU runner CUDA initialization/query/workload girişlerini ve gerçek
`saved_tensors_hooks.__enter__` çağrısını fence etti. Payload testleri hooks'u
kurmadan doğrudan CPU nesnelerini kullandı. Gerçek owned CPU child yalnız
constructor exception yolunu çalıştırdı; GPU forward/training yoktur. Bir mevcut
JUnit `record_property` uyumluluk uyarısı davranış testini etkilemez. İlk harness
denemesinde hook sınıfını fonksiyonla değiştirmek collection hatası verdi;
sınıf korunup yalnız context entry engellenerek düzeltildi. Assertion gevşetilmedi.

Gerçek CUDA hook callback/lifetime, stream completion, native library-load,
allocator ve crash tekrarlanabilirliği **GPU_VALIDATION_REQUIRED**; CPU PASS bu
alanların PASS'i değildir. P01–P35/P31, full suite veya native crash workload yok.

## D — Tek taze, read-only admission

Karar anı **18 Kasım 2026 17:23:47 +03:00**, capture yaşı **2.318 saniye**.
Mevcut strict `native_admission.evaluate_bound` / `p23_preflight` kullanıldı;
warning-only GPU politikası kullanılmadı. Kaynak/runtime hash'leri, yeni run ID,
nonce ve exact declarative recipe/protocol `QA/admission-proposal.json` içindedir.
Manifest SHA-256: `89a124b8551546f4eb0e99d16f96f1124c781a70806e4f84ec2e2f5f21c420be`.
Bu executable authorization/sealed controller kabulü değildir; nonce execution
için yeniden kullanılamaz. Tam PID creation/parent/owner/command-line kayıtları
yalnız dış QA'da tutulur; process adı otomatik desktop onayı oluşturmaz.

- GPU: RTX 3060 Laptop, UUID `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`,
  driver **591.74**, driver-reported CUDA support **13.1**; Torch **2.5.1+cu121**,
  CUDA build **12.1**, Python 3.10.18. Yeni CUDA context oluşturulmadı.
- Üç free-VRAM örneği **5534/5534/5534 MiB**; ayrı son metadata sorgusu
  total/used/free **6144/470/5525 MiB**. RAM **5.029 GiB**, disk **18.251 GiB**;
  ilgili 2048 MiB / 4 GiB / 12 GiB eşikleri geçti. Utilization telemetry'dir.
- Compute envanteri 11 PID, C/G tablosu 10 PID. **Eski PID 2000 mevcut değil**.
  Güncel fark **PID 1968 (`dwm.exe`)**: compute sorgusunda
  `[Insufficient Permissions], [N/A]`; C/G'de yok; CIM path/command-line null,
  owner sorgusu return 2. Creation UTC `2026-11-18T11:04:35.0190720Z` kaydedildi.
  Adı, desktop-only bağımsızlık kanıtı sayılmadı. Diğer on PID için alınan
  CIM/owner bilgisi de otomatik compute-workload muafiyeti oluşturmadı.
- CPU-only Job readback: kill-on-close/process/job/active-process flags `0x2308`;
  child 4/8 GiB ve 4 process; outer capability 4/12 GiB ve 8 process. Job'lara
  D aşamasında child atanmadı. Önerilen sınır 60 s child / 90 s outer, allocator
  .15, kanal başına 64 KiB, sonuç 2 MiB, metadata 2000 satır/1 MiB; retry yok.

Sonuç **BLOCKED**: `compute_CG_inventory_mismatch`,
`compute_or_unknown_GPU_process`, `incomplete_PID_creation_or_owner_identity:1968`.
Eski PID 2000 engeli güncelmiş gibi taşınmadı. Bu capture 30 saniyelik freshness
penceresi dışında yürütme yetkisi vermez; uygun duruma gelene kadar polling yoktur.

Bir sonraki kontrollü GPU diagnostic için her güncel GPU PID'sinin güvenilir
kimlik/creation/owner ve desktop/compute bağımsızlığı, tam eşleşen fresh inventory,
yeni sealed executable/runtime manifest ve kullanılmamış nonce, gerçek controller/
worker Job admission/readback/deadline/owned-cleanup sözleşmesi ve **ayrı açık
kullanıcı yürütme onayı** gerekir. Bu preflight'ta bağlanan öneri yalnız mevcut
tiny AdamW fp32, default-stream, hooks OFF/reset ON, seed 71/iki epoch kontrolüdür;
B4 hooks-ON yeniden üretimi değildir. B4 için ayrıca tek hipoteze bağlı faz/native
kanıt protokolü ve ayrı onay gerekir; production guard otomatik açılmaz.

## Koruma, commitler ve duruş

QA: `C:/Users/hdgn5/AppData/Local/hyso-stage13-5-23-20261118-171036`.
Yeni kalıcı test/preflight çıktıları repository dışındadır. Başlangıç branch
`feature/hyso-models-next`, HEAD `3842ee40ccc4fe73bf2c5f69b37930306b8b5597`, index boş.
47 kullanıcı dosyası, mevcut 19 artifact, önceki QA manifestleri ve tarihsel
tracked raporlar baseline hash/status ile korundu; eski dump/QA arşivi yazılmadı.

Kaynak/test commit'i **`ec71f37cfd663e48a1a9dc2c83972673c12fd49e`** yalnız
`hyso/performance/_training_process.py` ve `tests/test_isolated_training_memory.py`
içerir. İkinci yerel commit yalnız bu sonuç raporudur; bitiş HEAD ve tam koruma
kaydı `QA/final-state.json` / `QA/preservation-final.json` ve son kullanıcı yanıtındadır.
Final index boş; özgün 47-dosya status'u korunur. Driver/Toolkit/Torch değişikliği,
GPU process kill, GPU diagnostic, saved-hook enable, push veya yayın yoktur.
CUDA B4, daha geniş hooks-enabled activation coverage ve genel CUDA kabulü açık
kalır. Stage 13.5.23 burada durur; T09/T10 veya yeni stage başlatılmaz.
