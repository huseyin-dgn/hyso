# M2-P-B10 / P23 — Training Memory Analysis

2026-10-21, Europe/Istanbul. **P23 PARTIAL; B10 PARTIAL / full-scope GO yok.**
Bu kayıt önceki B4/B6/B7/B9/P21 audit kararlarını değiştirmez. Yeni evidence **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p23-20261021`.
Final receipt/hash/count manifestleri aynı dizindedir; eski task sonuçları yeni
PASS sayısına eklenmez. Kullanım: [rehber](../guides/performance-training-memory.md).
Final manifest: E/verification-summary.json. Yeni CPU71, mevcut regresyon2049,
packaging4 PASS; bu üç gate'te FAIL/ERROR/SKIP=0. Source/installed CUDA aynı23
senaryoyu geçer; distribution tekrarları distinct sayıya eklenmez.

## 1. Branch / HEAD / protection

`feature/hyso-models-next`, beklenen HEAD
`34f56687516ba2bf3d13c7e26e7293a818840311` eşleşti. Başlangıç index boş,
untracked yok; tek mevcut kullanıcı farkı `tests/test_gradient_accumulation.py`.
Bu dosya yalnız okundu/çalıştırıldı. Stage/commit/push/reset/checkout/rebase/merge/
clean, dependency/Torch/CUDA/driver güncellemesi veya veri/model indirmesi yoktur.

## 2. P23 API ve tasarım

Yeni `hyso.performance.training_memory`: `TrainingMemoryProtocol`, immutable
`TrainingMemoryResult`, `TrainingMemoryError`, `measure_training_memory`.
Standard fresh Trainer admission; explicit CPU/indexed CUDA device, scope,
never-reset policy ve sample/boundary/tensor-record hard ceilings. Private P19
one-fit recorder factory ile bir gerçek fit; public Trainer veya eski performance
imzaları değişmez. İkinci forward/probe/training ve state/RNG rollback yoktur.
Embedded P19 schema1, segment UUID ve canonical SHA ile kaynak ilişkilidir.

## 3. Logical domain

Registered tensor metadata'dan `numel * element_size`. Her component içinde
distinct object bir kez; views ayrı logical objelerdir. Summary count/elements/byte,
dtype/device/shape/stride/storage offset, isimler, snapshot-local identity/storage
groups ve deduplicated full-storage extent ayrı raporlanır. Hiçbiri RSS veya fiziksel
VRAM değildir; component toplamı allocator toplamına eşitlenmez.

## 4. Parameter / buffer / gradient

Frozen parameter summary parameter total'ın alt kümesidir. Nonpersistent registered
buffer dahildir; gradient yalnız gerçekten mevcutsa sayılır. Epoch zero-grad ve
attempt finally zero-grad sonrası allocation convention gerçek optimizer davranışını
izler. `set_to_none=False` unit fixture'ı mevcut gradient storage'ının kalmasını
doğrular. EMA/scaler/scheduler tensorları logical profile kapsamına dahil değildir.

## 5. Lazy optimizer

Exact native SGD/Adam/AdamW state okunur; state_dict veya step çağrısı yoktur.
Momentum kapalı SGD sıfır state, momentum açık SGD gerçek buffer, Adam/AdamW
ilk/ikinci applied step ve başlangıç boş state ile testlidir. FP32/FP64 step
tensor dtype'ları varsayılmaz. Non-tensor state plain metadata olarak gözlenir;
Python heap byte maliyeti null'dır. Custom optimizer accounting unsupported.

## 6. Tied / alias / views

Mixed FP64/int16 fixture: tied parameter object tek, frozen view ayrı logical
object; parameter96 byte, frozen subset32, buffer22, union full storage70 byte.
Shared buffer/view aynı storage ID'sine bağlanır. IDs boundary'ler arasında kalıcı
adres değildir. Traversal iteratiftir; cyclic module/state reddedilir; sonuç tensor,
storage handle, model closure veya traceback tutmaz.

## 7. Activation coverage

`post_forward` gerçek Trainer boundary'sidir; activation lifetime ölçümü değildir.
Saved-tensor hooks eklenmedi. `activation.value=null/status=unsupported`,
`saved_tensor_lifetime_adapter_unvalidated`. Saved activation peak, transient
forward allocation veya pack/unpack attribution iddiası yoktur. M2-P-A activation
acceptance tamamlanmadığından P23 implemented ilan edilmez.

## 8. CPU RSS

P10 `_observe` native Windows working-set / Linux proc adapter'ı yeniden kullanılır.
Sampler thread yalnız OS query yapar; tensor/model okumaz. Baseline/end, sampled
peak, OS lifetime HWM, query bounds/midpoints/gaps ve query toplam maliyeti raw
evidence'dır. Windows native doğrulanır; Linux-native test yapılmadı. Backend kaybı
raw prefix'i korur ve summary'yi null/unsupported yapar. Sample budget exhaustion
başarılı incomplete result üretmez. RSS runtime/loader-main/observer dahil process
alanıdır; worker RSS unsupported. Transient kaçabilir; lifetime HWM prefix içerir.

## 9. CUDA allocator

Gerçek allocated/reserved, lifetime peak allocated/reserved, allocation retries,
OOM counter, device name/capability/native backend ve selected stream kaydedilir.
Baseline/end ve boundary sampled maxima ayrıdır. Query'ler host-clock aralığıdır;
per-boundary sync yok, P19 stream drains korunur. Reserved canlı tensor memory
veya physical/device-wide VRAM değildir. Non-native allocator unvalidated hatasıdır.

## 10. Reset ownership

Yalnız `peak_reset_policy="never"` desteklenir; reset opt-in bile bu dar API'de
reddedilir. `reset_performed=false`, ownership none. True window/incremental peak
null/unsupported; lifetime peak minus baseline değeri uydurulmaz. Ortak `_RUN_LOCK`
P09/reentrant ölçüm yarışını reddeder. Caller başka thread'lerde de exclusivity
sağlar. Azalan lifetime peak ledger failed olur; clamp/restoration sözü yoktur.
P09 `_cuda_memory_probe` ayrı single-forward explicit irreversible reset'i aynı kalır.

## 11. Native isolation

Production isolated recipe/factory worker, workload memory guard veya hard timeout
**UNSUPPORTED**. Canlı Trainer/tensor worker'a taşınmaz. Testler ayrı stdlib Windows
Job supervisor kullanır: actual interpreter PID admission, kill-on-close, 8 process,
4GiB/process ve 12GiB/job private commit, hard deadlines; CUDA allocator fraction .15.
Bu test-only güvenlik production API özelliği değildir. Typed OOM/no retry ile
Python exception, expected timeout ve native crash sınıfları ayrılır.

## 12. Accumulation

5 sample/batch2/accum2 gerçek eğitim: 3 microbatch, window samples `[4,1]`,
2 attempt. Pre-microbatch gradient bytes `[0,N,0]`, post-backward `[N,N,N]`.
Microbatch/window/attempt/epoch ID'leri P19 ledger ile yeniden doğrulanır.
Post_scaler_step zero-grad öncesi; post_attempt applied/skipped sonrasıdır.

## 13. AMP overflow / recovery

Gerçek CUDA GradScaler ilk gradient inf fixture: 4 attempt, 1 skip, 3 applied;
SGD/Adam/AdamW ve iki stream modu. Skip Adam state'i allocate edilmiş varsayılmaz;
ilk skipped attempt boş state, sonraki applied state gerçekten doludur. Scaler,
EMA, OneCycle ownership değişmez; exact snapshot eşliği korunur.

## 14. P19 / P20 / P21 / P22 uyumluluğu

Bu task Trainer/training.py/training_details.py/dataloader.py kaynaklarını değiştirmez.
Mevcut private boundaries kullanılır; default Trainer'a sampler/query/reset gelmez.
Memory capture P20 stage timers/hooks kurmaz; embedded P19 projection bunu açıklar.
Eski schemas/checkpoint v1/BenchmarkResult/Manager schemas aynıdır.

## 15. Exact state / RNG

Linear + 25 ModularCNN replay konfigürasyonu: parameters, buffers, gradients,
optimizer, scaler, scheduler, EMA, history, counters, modes ve Python/NumPy/Torch
RNG exact karşılaştırılır. CUDA ayrıca CUDA RNG; deterministic algorithms,
CUBLAS workspace, cudnn deterministic ve TF32 kapalı koşullarındadır. CUDA exact
eşlik tüm nondeterministic workload'lara genellenmez. Resume yeni segment/suffix'tir.

## 16. Lifetime / GC / cleanup

Automatic GC kapalı success, unsupported backend, constructor/boundary/timing/
training/sampler failure ve model/tensor weakrefs; retained-model pozitif kontrol,
thread/recorder cleanup testlidir. CUDA iki ardışık run raw allocator gözlemi ve
test-only scheduled-GC/baseline-reset diagnostic içerir. Production GC/cache reset
yapmaz. Bu diagnostic fresh owned test worker'dadır; reset geçmişi kaydedilir,
geri yüklenmez. Plateau gözlemi leak PASS/root-cause iddiası değildir. Query süreleri
ölçülür; toplam instrumentation overhead null/unknown. Legacy P20 event-error
cleanup native regresyonda ayrıca testlidir.

## 17. Yeni CPU unit gate

Final **71 PASS, 0 FAIL/SKIP**, E/new-ledger-final.xml (23.44s); E/verification-summary.json
aynı sayımı kaydeder. Logical mixed/tied/views,
lazy state, gradient lifecycle, protocol budgets, unsupported tensor/custom
optimizer, RSS ordering/backend loss/budget, defensive-copy/immutable/rehashed
tamper ve atomic/no-clobber export testleri dahildir. İlk koşu20 PASS/46 FAIL:
P23 recorder'ın Trainer private `remove_optimizer_hooks` no-op metodunu sağlamaması
nedeniyle. Metot eklendi; native hook kurulmadığı korunur. Beklenen hata gevşetilmedi.
Ara tekrarlar final PASS sayısına eklenmez.

## 18. Gerçek CPU training

Linear SGD/Adam/AdamW FP32/FP64, dropout/validation/EMA; ModularCNN attention,
scheduler, early stop/best restore ve checkpoint2→suffix3/4/5 kapsamı gerçek fit
ile testlidir. Completed checkpoint yalnız ready/end snapshots ve sıfır yeni epoch
üretir. CPU sample peak için sabit byte veya timing eşiği kullanılmaz.

## 19. Native CUDA gate

RTX3060 Laptop / Torch2.5.1+cu121. Final source ve installed 23-scenario matrix:
18 optimizer/precision/stream/overflow case +2 resume +1 Python failure cleanup
+1 GC-disabled/two-run observation +1 test-only scheduled-GC/reset diagnostic.
Ayrı gerçek training forward içindeki typed allocator OOM ve sampler/recorder
cleanup, fresh child/parent allocator+RNG preservation ve gerçek training forward
içinde runtime-ready timeout/worker termination receipts E'dedir. Source/installed aynı
senaryoların iki distribution doğrulamasıdır; distinct sayımda ikiye katlanmaz.
İlk CUDA harness exit1: Python fixture'da eksik optimizer argümanı; düzeltildi.
Bu native crash değildir. Installed parent fixture ilk kez source child çıktı adıyla
çakıştı; supervisor workload başlamadan fresh-name admission'ı reddetti. Child yolu
parent output stem'inden türetildi; final installed parent/child exit0 ve exact
allocator/RNG eşliğiyle geçti. Beklenmeyen native crash görülmedi.
Final E/release-cuda-source.json (18.516s) ve release-cuda-installed.json (19.266s);
OOM/parent/timeout ayrı release-cuda-* receipt'leridir. Expected timeout25s,
ready PID doğrulanır ve process sonlanır; `timed_out=true`, başarılı result yoktur.
Windows kill-on-job-close exit_code0 döndürebildiği için timeout exit code'dan
başarıya çevrilmez. E/worker-cleanup.json tüm final supervisor PID'lerini ölü doğrular.

## 20. Regresyon

E/regression-files.txt kesin mevcut-test seçimidir; E/regression.xml:
**2049 PASS, 0 FAIL/SKIP**, 579.17s. Yeni P23 testleri bu sayıya dahil değildir.
P09/P10/P11/P17, P19/P20/P21/P22, Trainer/gradients/scaler/EMA/schedulers/ES,
checkpoint/resume/benchmark state ve Experiment Manager dahildir. Yeni P23 testleri
ayrı sayılır. Ayrı bounded native P19(20), P20/P22(21), deterministic replay(10),
P09 benchmark(45), P17 capacity(5), P21 loader(20) ve native RSS/footprint(3)
yeniden çalıştırıldı; her supervisor exit0. Controlled test-only
os._exit ve expected timeout başarı diye gizlenmez; unexpected native failure değildir.

## 21. Minimum runtime

Mevcut Python3.10.18 / Torch2.2.2+cpu runtime, yalnız local HYSo wheel
`--no-index --no-deps --target` kurulumu. Installed smoke + P23 native CPU
checkpoint/resume exact; CUDA initialized=False. Dependency METADATA/RECORD
before/after aynı; environment upgrade yok. Final receipt E/release-minimum-environment.json;
70 dependency metadata/record hash'i aynıdır. Minimum target dahil42 modül
source/wheel/sdist/installed bytes eşittir.

## 22. Wheel / sdist integrity

Explicit production inventory41→42. E/release-packaging.xml **4 PASS**,156.95s; source/wheel/
sdist ve her installed module byte equality E/archive-integrity.json ile kaydedilir.
İlk paket sonrası optimizer traversal, GC kapalı metadata closure cycle'ını da
önlemek için iteratif hale getirildi; final paket/minimum/native bu kaynakla yeniden
doğrulanır. Son validator ayrıca memory boundary query scope ve applied/skipped/sample
alanlarını embedded P19 ledger'a bağlar; yeniden hash'lenmiş attempt tamper da reddedilir.
Bu değişiklik sonrası final source/unit/native/installed/package doğrulaması yenilendi.
Eski paketlerin PASS sonuçları final4'e eklenmez.

## 23. Export / Experiment Manager

Canonical JSON raw RSS+logical+allocator evidence; CSV boundary/component snapshots.
Atomic/no-clobber, schema/source/canonical SHA ve export-time tamper kontrolü.
JSON checksum authentication imzası değildir. Fit sırasında disk/SQLite yazılmaz;
caller explicit checkpoint I/O full-fit memory scope'undadır, P19 timing dışındadır.
Ölçümden sonra copied JSON/CSV artifact + SHA verify, generic RSS metric,
finish/close/reopen/read gerçek store testinde geçer. Tracking failure training'i
tekrar çağırmaz. Null unsupported metrik sıfırla loglanmaz; yeni migration/kind yok.

## 24. Kesin task dosyaları

Yeni5: `hyso/performance/training_memory.py`, `tests/test_training_memory.py`,
`tests/cuda_training_memory_validation.py`, `docs/guides/performance-training-memory.md`,
bu audit. Değişen3: `hyso/performance/__init__.py`, `tests/test_packaging.py`,
`docs/roadmap/hyso-performance-backlog-v1.1.md`. Toplam8 task dosyası; korunan
kullanıcı gradient farkı ayrı9'uncu working-tree farkıdır. Tarihsel auditler değişmez.

## 25. Korunan SHA256

`tests/test_gradient_accumulation.py` başlangıç/son:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Boş index, branch/HEAD ve değişmeyen legacy public/schema kaynak hash'leri
E/protection.json içinde kaydedilir. Dosya stage/write/restore edilmedi.

## 26. 35-feature sayımı

P23 planned→partial: **16 implemented /7 partial /12 planned =35**.
P21 partial, P25 planned; P19/P20/P22 implemented korunur. Facade yapılmadı.

## 27. P23 karar

**PARTIAL within verified observation scope.** Logical/state/RSS/native allocator
gözlemi çalışır. Activation lifetime, true isolated window/incremental peak,
production worker/hard resource budget ve worker-private RSS unsupported.
Bu sınırlar M2-P-A tam acceptance yerine geçmez; implemented/PASS-full iddiası yoktur.

## 28. B10 karar

**PARTIAL / full-scope GO yok.** P21 worker-service eksik, P23 partial, P25 planned.
Native testlerin geçmesi eksik domainleri completed yapmaz. Stable-release onayı yoktur.

## 29. Açık riskler

B4 native reliability OPEN / root cause UNRESOLVED. B6 gerçek CUDA kernel profiling
PARTIAL, B7 copy/compute overlap PARTIAL, P18 planned/BLOCKED ve P21 worker trace
PARTIAL korunur. Yeni in-process snapshot API'si workload OOM/hang'i izole etmez.
Linux native, başka GPU/runtime ve alternatif allocator kapsamları doğrulanmadı.

## 30. P25 readiness

Microbatch/window/attempt ve applied/skipped source ledger hazır; P25 hâlâ PLANNED.
Gradient norm/finite/scale-growth analizi eklenmedi. Sonraki task mevcut ownership'i
koruyarak ayrı opt-in diagnostics tasarlayabilir. P23 için sonraki bağımsız kapsam
trusted factory/OS-guarded worker ve gerçek saved-activation lifetime adapter'ıdır.
