# M2-P-B9 — P19 gerçek training timing

2026-10-21, Europe/Istanbul. Bu görev P19'u uygular; M2-P-A'nın B9=P19+P20+P22
paketinin tamamını uygulamaz. **P19 PASS within verified scope; B9 PARTIAL.**
P20–P25 uygulanmadı. Son native/paket/regresyon kanıtları aşağıda ayrı tutulur.

Yerel yeni kanıt dizini **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb9-20261021`.
CPU test fixture'ları gerçek CPU Torch çalıştırır; native CUDA pytest conftest
dışında 60 saniyelik supervisor altında çalışır. Önceki audit test sayıları bu
görevin sayısına eklenmez. Ham sonuçlar makineye özgü TEMP artifact'larıdır.

## 1. Başlangıç HEAD / branch / protection

`feature/hyso-models-next`, `ef64fd1beec1cde040d503ee8aa03a9d87123e37` doğrulandı.
Başlangıç index boş, tek çalışma ağacı değişikliği kullanıcıya ait
`tests/test_gradient_accumulation.py` idi. Dosya okunup test edildi; yazılmadı,
stage edilmedi/geri alınmadı. Trainer constructor/fit imzaları korunur.

## 2. B8 / P18 gerçek capability sonucu

[B8 capability gate](m2-p-b8-roofline-capability-gate.md) ve E/capability.json:
Torch2.5.1+cu121 / Python3.10.18 / Windows26200 / RTX3060 Laptop / driver591.74;
CUDA hesaplama var, profiler activity yalnız CPU. `LIBKINETO_NOCUPTI` mevcut.

## 3. P18 neden BLOCKED

Aynı kernel/precision/memory hierarchy için gerçek FLOP, DRAM/cache traffic,
device time, replay correlation ve kalibrasyon paketi yok. PATH'te ncu/nsys
bulunamadı; counter izinleri doğrulanmadı. P18 planned/blocked kaldı; Roofline
API/Result yok. Parametre byte, P15 transfer rate veya GPU busy sayaç yerine
kullanılmadı. Alternatif Linux/CUPTI/counter ortamı öneri; test edilmiş destek değil.

## 4. P19 public API

`hyso.performance`: `TrainingTimingProtocol`, `TrainingPerformanceResult`,
`TrainingTimingError`, `measure_training`. Üretim kodu
`hyso/performance/training.py`; public entrypoint doğrudan bir gerçek `Trainer.fit`
çağrısı yapar. Callback eklemez, optimizer.step override/replay yapmaz. Default
Trainer akışında recorder/timer/event/profiler yoktur; private None branch'leri var.

## 5. Gerçek training step / epoch timing

Raw ns timestamps, microbatch host_ms, accumulation window host_ms, inclusive
`_update` attempt host_ms, epoch training_s/validation_s/total_s ayrıdır. Epoch
loader/transfer/check/backward/update/metrics completion kapsamını içerir;
microbatch backward sonunda ve update öncesinde kapanır. Forward/backward stage
decomposition yoktur; P20 tamamlanmış gibi davranılmaz. Scope'lar iç içedir ve
toplanmaz. Rehber bütün başlangıç/bitiş sınırlarını listeler.

## 6. Validation zaman sınırı

Training completion → `_validate` → validation completion ayrı timestamp'larla
ölçülür. Yoksa duration/timestamps null. Total epoch scheduler/ES/history/
epoch_end observer'ı kapsar; checkpoint disk I/O, checkpoint restore, fit başlangıç/
bitiş observer'ları ve best-weight finalization hariçtir. Native/CPU validation ve
no-validation testleri vardır; EMA evaluation target mevcut Trainer'a aittir.

## 7. Accumulation ledger

`test_real_training_exact_state_and_ledger`: 5 sample, batch2 → 3 microbatch;
accum2 → 2 window/attempt; sample ledger `[4,1]`. Her epoch sample/microbatch/
window/attempt/applied/skipped alanları ayrıdır. Global microbatch/attempt/applied
IDs Trainer counter'larıyla; window IDs segment içinde ardışık; epoch IDs resume
sonrasında gerçek epoch numarasıyla başlar. Her result yeni segment UUID taşır.

## 8. Optimizer attempt / applied / skip kanıtı

Recorder `_update` giriş/çıkışında mevcut update_count kararını gözler. CPU gerçek
SGD reference ve normal/instrumented paired state tests; native gradient hook ile
ilk AMP window'a inf verilmesi `[skipped,applied,applied,applied]` üretir. Dört
attempt içinde üç applied update; EMA/OneCycle sayaçları üç kalır. `_update`
ownership/GradScaler/hook/zero_grad kodu değiştirilmedi.

## 9. CPU host / CUDA event clock davranışı

CPU `perf_counter_ns`; açık /1e6 ve /1e9 dönüşümleri. CUDA host wall ve selected
stream event interval ayrı alanlardadır; CPU time device time diye etiketlenmez.
Eventler yalnız synchronize/query sonrası okunur. En fazla event_buffer kapalı
pair + üç açık scope; chunk/epoch terminal drain, validation completion; her
microbatch'te device-wide synchronize yoktur. Küçük buffer overhead'i scope'tadır.
Default/non-default stream testli; gizli streams/concurrent process attribution
unsupported. CUDA event interval kernel active time/CUPTI trace değildir.

## 10. AMP overflow / recovery

`tests/cuda_training_timing_validation.py`: SGD/Adam/AdamW native AMP, ilk backward
gradient inf fixture'ı ve sonraki recovery. Hook sonunda kaldırılır; gerçek native
scaler kullanılır. FP32 master weights/scaler state/history/RNG exact paired
comparison'a dahildir. CPU AMP etkinleştirilmez; mevcut Trainer CPU amp yasağı
değişmez. Minimum runtime yalnız CPU'dur; native AMP kanıtı Torch2.5.1+cu121 içindir.

## 11. EMA ve scheduler uyumluluğu

Yeni CPU tests StepLR/OneCycle, 25-case ModularCNN replay matrisi ayrıca
CosineAnnealingLR/Plateau/none ve beş attention varyantını kapsar. Native skip
fixture'ında EMA ve OneCycle applied updates ile aynı ilerler. Epoch scheduler,
ES ve evaluation_model seçimini instrumentation sürmez.

## 12. State / RNG / history / replay

Normal vs observed eğitim model parameters/buffers/gradients, optimizer/scaler/
scheduler/EMA, modes, history, Python/NumPy/Torch RNG ve native CUDA RNG için
exact karşılaştırılır. CPU dropout/shuffle içerir. Timing eşitliği beklenmez.
Normal training matematik/state akışı aynı kaldı; instrumentation maliyeti süreye
dahildir. Runtime provenance deterministic flags, Torch/Python/platform/device,
model module/tensor schema, input ve optimizer group ayarlarını içerir.

İlk ek CUDA checkpoint fixture'ı deterministik ayarlar olmadan bit eşitliğinde
başarısız oldu (exit1, native crash değil). E/native-nondeterministic-failed.stderr.log
ve receipt korunur. Kontrol E/baseline-checkpoint-repeat.log: instrumentation
**kapalıyken** aynı FP32 checkpoint/resume akışının üç koşusu arasında da model
tensorları farklı; AMP üç koşusunda fark yok. Tek full-fit baseline-repeat ayrıca
fark üretmedi. Bu kanıt genel native crash kök nedeni değildir. Fixture mevcut
`cuda_trainer_replay_conformance.py` determinism/CUBLAS/TF32 precondition'larına
eşitlendi; tolerance=0 kaldı, production ayarları değiştirilmedi. Exact CUDA replay
iddiası bu deterministik scope içindir; nondeterministic CUDA için byte equality
garantisi verilmez. Son deterministik gate başarılıdır.

## 13. Checkpoint / resume

v1 serializer ve `resume.py` değişmedi; ephemeral recorder checkpoint'e yazılmaz.
CPU checkpoint epoch2 kesinti → yeni Trainer epoch3/4/5 timing segmenti ve exact
state; tamamlanmış checkpoint boş/no_new_epochs result. Native FP32 ve AMP aynı
kesinti/restore akışında exact karşılaştırılır. Prefix timings üretilmez;
doğrulanmış eski segment olmadan cumulative duration iddia edilmez.

## 14. Experiment Manager

Yeni tablo/migration/run kind yok. Yeni guide ve testler ölçümden sonra JSON/CSV,
generic report artifact copy/SHA verification, metric, finish/close/reopen/read
akışını çalıştırır. Recorder DB/file yazmaz. Export/tracking hatası fit'i yeniden
çağırmaz; result elde kalır. Ölçüm ve tracking caller tarafından ayrı çağrılır.

## 15. Yeni CPU testleri

E/cpu-verified.xml ve cpu-verified.log: **62 PASS, 0 skip/xfail**, 12.63s.
1/2/3 epoch ve validation çiftleri, exact replay matrisi, uneven reference,
fake-clock 10ns arithmetic, disabled no-timer/no-profiler baseline, early stop,
exception/recorder failure cleanup, record cap ve tam8-record boundary,
immutability/tamper/no-clobber export, resume/completed checkpoint ve rehber/
Experiment Manager akışı. Gerçek timer testleri sabit ms beklemez.

İlk interpreter invocation pytest olmayan base environment nedeniyle test
başlatamadı. İlk yeni suite25FAIL/5PASS, Adam betas tuple provenance sorununun
tespitiydi; mevcut checkpoint plain-config normalizasyonuyla düzeltildi. Sonraki
28PASS/2FAIL CSV helper'a dict yerine list verilmesini yakaladı; düzeltildi.
Sonraki 30/59/62 alt küme tekrarları nihai62'ye eklenmez; doğrulama gevşetilmedi.

## 16. CPU regression testleri

E/regression-files.txt, regression.xml ve regression.log: **1836 PASS, 0 skip/xfail**,
556.19s; özet verification-summary.json içinde kayıtlıdır. Seçim Trainer classification/update/
validation/lifecycle/save/restore/replay, korunan gradient testi, scheduler/EMA,
checkpoint/benchmark state safety, bütün Experiment Manager ve mevcut performance
test dosyalarını kapsar. Tarihsel CPU2313 bu sayıya eklenmez. Son provenance/
linear-time validator düzenlemesi ayrıca final62 P19 ve installed package gate'inde
doğrulandı; Trainer bağlantısı regresyon koşusu boyunca aynı17 additive satırdı.

## 17. Native CUDA kontrolleri

E/native.json ve native-receipt.json: **20 distinct senaryo PASS**, final source
deterministik gate16.00s, timeout60s/exit0; 18 optimizer/AMP/stream/overflow paired
senaryosu +2 checkpoint/resume. Native test subprocess'i allocator fraction0.15
ve CPU thread1 ile bounded; beklenmeyen native crash yok. Installed native aynı
20 senaryoyu ayrı çalıştırır; tekrarlar40 distinct senaryo olarak sunulmaz.
Final installed gate16.14s/exit0; receipt ve package_source aynı dizinde kayıtlıdır.

## 18. Minimum Torch / wheel / sdist

`tests/test_packaging.py` inventory38→39 additive güncellendi; module
`hyso/performance/training.py` wheel/sdist'e dahildir. Existing schema/result
dispatch değişmedi. E/packaging-final.xml/log ve archive-integrity.json production
modüllerinin source/wheel/sdist/installed byte eşitliğini doğrular. Final packaging
**4 PASS, 0 skip/xfail**, 170.71s; bütün39 production modülü eşittir.
Minimum existing Python3.10.18/Torch2.2.2+cpu ortamına yalnız HYSo wheel
`--no-index --no-deps --target` ile kuruldu. Isolated smoke, yeni timing rehberi ve
P19 checkpoint/resume exact testi çalışır; CUDA initialized=False kalır.
Dependency METADATA/RECORD before/after eşitliği E/minimum-environment.json'dadır.
Model/weight/dataset/dependency/Torch/CUDA/driver indirme/güncelleme yoktur.

## 19. Eklenen / değişen dosyalar

Yeni: `hyso/performance/training.py`, `tests/test_performance_training.py`,
`tests/cuda_training_timing_validation.py`, bu B9 audit, B8 audit ve training timing
rehberi. Mevcut değişiklikler: performance namespace, Trainer'da17 additive private
boundary satırı, packaging inventory/smoke, backlog. Toplam10 görev dosyası;
kullanıcının başlangıç gradient değişikliği ayrı11'inci working-tree farkıdır.
Mevcut benchmark/result/checkpoint/Experiment Manager production schema'ları
değişmedi. Index boş; stage/commit/push/reset/checkout/merge/rebase/clean yapılmadı.

## 20. Korunan kullanıcı dosyası SHA256

Başlangıç ve son:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Dosya byte'ları korunur. Final branch/HEAD/index/SHA kanıtı E/protection.json.

## 21. Güncel 35-feature backlog sayımı

**14 implemented, 5 partial, 16 planned =35**. Yalnız P19 planned→implemented
within verified scope. P18 planned/blocked. P20–P25 planned kalır. Backlog sonuna
Post-Performance Developer Experience tasarım notu eklendi; gelecekteki `Hyso`
facade şu anda yoktur. Mevcut public API'ler korunur; toplam35 değişmez.

## 22. B4 / B6 / B7 açık riskleri

B4 reliability OPEN, native crash root cause UNRESOLVED; bu görevin başarılı
event timing'i crash çözümü değildir. B6 CUDA kernel/layer attribution unsupported
ve B7 overlap partial/full-scope NO-GO kayıtları değişmedi. P19 CUPTI kapsamı
iddia etmez. Uzun-run, başka GPU/OS/build, hidden streams veya multiworker strict
replay kabulü yoktur; stable release onayı verilmez.

## 23. P19 PASS / PARTIAL / FAIL

**PASS within verified scope:** gerçek CPU ve doğrulanan selected-stream native
CUDA training timing/count ledger; exact deterministic state/replay; bounded
storage/events; immutable export ve generic tracking. Raw records sınırlı RAM'de
tutulur, event pairs chunk drain ile boşaltılır; disk streaming yoktur. Limit
aşımında başarılı eksik sonuç yerine explicit failure; training rollback iddiası
yoktur. Bu sınırlar API rehberinde açıktır.

## 24. B9 GO / PARTIAL / NO-GO

**PARTIAL.** Kullanıcının bu görevde istediği P19 alt kapsamı teslim edilir;
M2-P-A'nın B9=P19+P20+P22 tam kapısında P20/P22 hâlâ planned. Full B9 GO veya
stable release GO verilmez. P18 counter blokajı bağımsız P19'u engellemez.

## 25. P20 readiness

Private opt-in boundary ve training result temeli hazır; P20 için gerçek
forward/loss/backward/unscale/clip/scaler.step/scaler.update/EMA/scheduler/
zero_grad stage scope'ları ayrıca tasarlanıp test edilmelidir. Mevcut inclusive
attempt timing bu stage decomposition yerine geçmez. Bu görev P20'yi başlatmadı.
