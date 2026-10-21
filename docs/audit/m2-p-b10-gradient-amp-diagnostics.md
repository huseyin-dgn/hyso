# M2-P-B10 / P25 — Gradient & AMP Diagnostics

2026-10-21, Europe/Istanbul. **P25 PASS within verified dense/native scope;
B10 PARTIAL / full-scope GO yok.** Yeni evidence **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p25-20261021`.
E/verification-summary.json, protection.json, archive-integrity.json ve native
receipt'ler final manifestlerdir. Eski task PASS sayıları yeni sonuçlara eklenmez.
[Kullanım rehberi](../guides/performance-gradient-amp.md).
Final gate: **104 yeni CPU +2199 mevcut regresyon +4 packaging PASS**;
bu gate'lerde FAIL/ERROR/SKIP=0. Source/installed25 CUDA senaryosu aynı distinct
settir. E/final_verify.py branch/index/protection, archive/module hashes, namespace
korunumu, minimum runtime ve tüm final native PID'lerinin sonlandığını doğrular.

## 1. Branch / HEAD / protection

Branch `feature/hyso-models-next`, beklenen HEAD
`5ddc0241fa8ee802127fc20cd7bf5f09aedd27cf` eşleşti. Başlangıç index boş/untracked yok;
yalnız kullanıcı `tests/test_gradient_accumulation.py` farkı vardı. Bu dosya yalnız
okundu/çalıştırıldı. Stage/commit/push/reset/checkout/rebase/merge/clean yapılmadı.
Driver/CUDA/Torch/dependency upgrade veya model/weight/dataset indirmesi yoktur.

## 2. Public P25 API

Yeni `hyso.performance.gradients`: `GradientDiagnosticsProtocol`, immutable
`GradientDiagnosticsResult`, `GradientDiagnosticsError`, `measure_gradient_diagnostics`.
Bir fresh standard Trainer/native GradScaler admission; bir gerçek fit. Protocol
attempt/tensor-record/chunk ceilings ve optional postclip read içerir. P19/P20
private recorder factory kullanılır; ikinci eğitim/backward/step/update yoktur.
Result schema1, feature_ids=[P25]; embedded P19/P20 ve canonical SHA association vardır.

## 3. Normalization ve ölçüm sırası

Trainer._update kaynak sırası değişmedi: unscale → finite check → gerçek samples'a
bölme → optional clipping → native scaler.step → scaler.update → applied EMA/
OneCycle → finally zero-grad. P25 snapshot, P20 normalization stage kapandıktan
sonra ve clipping başlamadan önce okunur. Postclip ayrı read'dir; preclip diye
sunulmaz. Diagnostics normalization veya clipping'i yeniden çalıştırmaz.

## 4. Global L2 doğruluğu

Sözleşme `sqrt(sum(abs(unscaled_sample_normalized_g)^2))`. Device-local bounded
FP64 chunks; max-scaled square sum + host math.hypot. Gradient tam olarak CPU'ya
taşınmaz. Known3/4→5, mixed3/4/12→13, zero/empty/scalar/transposed dense, FP16/BF16/
FP32/FP64 ve1e-300/1e200 testlidir. FP64 finite norm representability aşılırsa null/
unrepresentable; partial result. Parameter objesi bir kez, ayrı alias objeleri
logical olarak ayrı sayılır; shared storage fiziksel dedup normu değildir.

## 5. Finite / NaN / Inf

Per-tensor finite/has_nan/has_inf; global NaN/Inf tensor count, with/without_gradient,
elements/dtype/device/layout/shape kaydedilir. NaN/Inf değerler JSON numeric NaN/Inf
değildir: norm null/nonfinite, finite=false. Known NaN ve ±Inf ayrı testlidir.
Unsupported gradient varsa tam global count/norm uydurulmaz. Non-AMP nonfinite,
Trainer'ın mevcut FloatingPointError'ını korur; başarılı diagnostics result yoktur.

## 6. Sparse / diğer coverage

Ordinary real packed strided CPU/CUDA gradients desteklidir. Sparse COO coalesced/
uncoalesced, CSR/diğer sparse, gapped/expanded/overlapping, complex/meta ve custom
Tensor adapter'ları unvalidated/unsupported. P25 coalesce/dense dönüşümü yapmaz;
duplicate-index norm semantiği hakkında iddia yoktur. Native sparse Embedding+SGD
gerçek fit null/unsupported global norm ve partial result üretir. Trainer'ın eski
sparse finite-check coalesce davranışı korunur. Alternatif optimizer/AMP backend
genel destek iddiası yoktur; standard Trainer admission aynı kalır.

## 7. Accumulation ledger

5 sample/batch2/accum2 → 3 microbatch, window samples `[4,1]`,2 attempt. IDs P19
window/attempt ve gerçek microbatch listesiyle doğrulanır. Standart BCEWithLogitsLoss
FP64 analitik fixture'ı ilk normalized gradient `[2,2.5]`, clipping/update sonrası
tek-sample son gradienti bağımsız sigmoid referansıyla doğrular. Samples yerine
accum_steps'e bölme veya postclip normu preclip diye sunma testi geçemez.

## 8. Native scale before / after

Gerçek native get_scale before-attempt ve after-update readout; increased/unchanged/
decreased classification ve continuity validation. Disabled scaler gerçek1 verir.
Scalers mutate/load/update edilmez; update yalnız Trainer tarafından bir kez çağrılır.
Native growth_interval1 fixture actual doubling'i gözler. Scale azalması execution
kararı olarak kullanılmaz; invalid/nonfinite scale readout failure'dır.

## 9. AMP skip / recovery

Gerçek FP16 AMP GradScaler ve ilk backward gradientine kontrollü Inf:4 attempt,
1 skipped,3 applied, enabled EMA3, OneCycle3. Sonraki finite attempt'ler native
execution'a döner. Ayrı native NaN case'i aynı recovery'yi doğrular. Null preclip
norm ve skipped clipping evidence korunur. `recovered_after_skip_attempt_ids=[2]`
fixture sonucu; genel sabit sayı veya resume-prefix recovery iddiası değildir.
Overflow evidence nonfinite+native skip association'dır; root cause unknown kalır.

## 10. Execution ownership

P20 public native optimizer pre/post hooks execution evidence sağlar. Trainer'ın
post-step hook/counter sahipliği değişmez. P25 applied/skipped/update ID'lerini
Trainer counter delta'sından ve embedded P19/P20'dan doğrular; kendi optimizer
step'i veya tahmini applied counter'ı yoktur. Source mismatch reddedilir.
Optimizer.step monkeypatch edilmez. Native execution proof scale-drop değildir.

## 11. EMA / OneCycle

Applied-only mevcut dallar korunur. CPU replay attention/optimizer/scheduler/EMA
matrix ve CUDA skip→recovery exact snapshot; scheduler/EMA counters/history/modes
aynıdır. StepLR/Cosine/Plateau/OneCycle mevcut cadence'i değişmez.

## 12. P19 / P20 / P21 / P22 / P23 uyumluluğu

Trainer.py, resume.py, training.py, training_details.py, dataloader.py ve
training_memory.py bu task'ta değiştirilmedi. Legacy schemas/signatures source
hash eşliğiyle korunur. P25 StageRecorder subclass'ı ayrı opt-in capture'dır;
default yollara gradient traversal/query/sync/profiler/RNG tüketimi gelmez.
Checkpoint v1, BenchmarkResult ve Experiment Manager schemas değişmedi.

## 13. Instrumentation overhead

Raw query ns, chunks ve scalar_reads gözlenir. CUDA .item readout'ları host sync
yapar; FP64 scratch/dispatch/wall/allocator perturbation vardır. P20 stage'leri
arasındaki query gap source P19 attempt/window timing'ine dahildir. P20/P19 bare
training performance değildir; stage total additive değildir. Scale-before read
ve toplam instrumentation overhead ayrı izole edilmediği için null/unknown'dur.
Hidden streams/kernel attribution yoktur. GC/empty_cache/peak-reset çağrısı yoktur.
E/instrumentation-observations.json: guide CPU fixture2 attempt/default chunk65536
query wall1.4986ms/32 scalar read; native ilk CUDA fixture4 attempt/test chunk3
query wall96.0113ms/312 scalar read. Bunlar farklı küçük instrumented workload'lardır;
bare-baseline delta veya genellenebilir overhead yüzdesi değildir. CPU örneğinin
tam result'ı E/example-cpu.json; toplam overhead null olarak kalır.

## 14. Exact state / RNG

Linear SGD/Adam/AdamW FP32/FP64 clip enabled/disabled ve25 ModularCNN replay case'i
normal/P25 aynı başlangıç/seed altında exact karşılaştırılır. Parameters/buffers/
gradients/optimizer/scaler/EMA/scheduler/history/counters/modes ve Python/NumPy/
Torch CPU RNG; CUDA native cases ayrıca CUDA RNG. CUDA deterministic algorithms,
CUBLAS workspace, cudnn deterministic ve TF32 kapalı koşulludur; evrensel
nondeterministic byte-equality garantisi değildir. Fit spy tek çağrıyı doğrular.

## 15. Checkpoint / resume / EarlyStopping

Epoch2 kesinti ve fresh Trainer suffix3/4/5; önceki attempt'e diagnostic uydurulmaz.
Completed/terminal checkpoint observations=[] ve no_new_attempts verir. Early stop
ve EMA best-weight restore mevcut sırayı/target'ı korur. Failure state rollback
vaadi yoktur; yeni diagnostics checkpoint payload'ına girmez.

## 16. Error / cleanup

Automatic GC kapalı success ve forward/reduction/optimizer/event/record-budget/
nonfinite error paths, model/parameter weakrefs ve hook/private recorder cleanup
testlidir. Native query failure native finally zero-grad'ı korur. Önceden var olan
optimizer hooks korunur. Plain result tensor/model/optimizer/traceback tutmaz.
Record limit explicit failure üretir; truncation/retry yoktur. Budget hatası
update'e girmeden olursa state/grads o gerçek failure boundary'sinde kalabilir;
transactional rollback veya fresh-Trainer retry sözü verilmez.

## 17. Yeni CPU testleri

E/cpu-final.xml: **104 PASS,0 FAIL/ERROR/SKIP**,18.06s. İlk E/cpu-first.xml90 PASS/
4 FAIL: fixture custom loss standard Trainer admission'a aykırıydı; monkeypatch
undo listesi model referansını tutuyordu; expected exception FloatingPointError
idi; Manager metric argument'i step/step_kind olmalıydı. Fixture'lar düzeltildi,
production admission/exception gevşetilmedi. Ara94 PASS final104'e eklenmez.
Norm, group/None/shared-storage, finite, sparse, analytical tail, clipping order,
scale, tamper/rehashed evidence, budgets, exports ve native replay CPU kapsamındadır.

## 18. Native CUDA

RTX3060 Laptop / Torch2.5.1+cu121, selected default/non-default stream, deterministic
replay. Bounded stdlib Windows Job: actual interpreter PID admission, kill-on-close,
8 process,4GiB/process/12GiB/job private commit,90s hard timeout; CUDA fraction .15.
Source ve installed **aynı25 senaryo**:18 optimizer/precision/stream/Inf paired case,
2 checkpoint/resume,1 query-failure cleanup,1 scale growth,1 NaN skip/recovery,
1 GPU bounded reduction/sparse capability,1 GC-disabled weakref lifetime.
Distribution tekrarları distinct sayıyı ikiye katlamaz. Receipts E/cuda-source ve
cuda-installed.*. Beklenmeyen native crash görülmedi; B4 OPEN kalır.

## 19. Regresyon

E/regression-files.txt kesin eski-test seçimidir; yeni P25 testleri hariçtir.
P19/P20/P21/P22/P23, Trainer/gradient/scaler/optimizers/EMA/scheduler/ES/checkpoint/
resume/benchmark state/Experiment Manager/performance export dahildir. Final XML
sayısı E/verification-summary.json'dadır: **2199 PASS,0 FAIL/ERROR/SKIP**,683.92s.
P23 training-memory71, full checkpoint ve EBO replay dahil genişletilmiş mevcut
test selection'ıdır; önceki task'ın2049 sayısına ekleme hesabı değildir.
Mevcut legacy CUDA P19(20), P20/P22(21),
P21(20), P23(23), deterministic replay(10) bu task'ta yeniden çalıştırıldı;
supervisor exit0, timeout yok. Tarihsel native sonuçlar eklenmez.

## 20. Minimum Torch

Mevcut Python3.10.18/Torch2.2.2+cpu/uyumlu torchvision ortamında yalnız local HYSo
wheel --no-index/--no-deps/--target kurulur. Installed smoke ve P25 CPU checkpoint/
resume exact kontrolü E/minimum-environment.json'dadır. CUDA initialized=False;
dependency METADATA/RECORD before/after aynıdır. Environment upgrade yoktur.
70 dependency metadata/record hash'i değişmedi; minimum target dahil43 production
modülü source/wheel/sdist/installed byte equality gate'ini geçer.

## 21. Wheel / sdist integrity

Explicit inventory42→43 production module. E/packaging.xml: **4 PASS**,204.51s.
Source/wheel/sdist/both installed ve minimum installed byte eşliği
E/archive-integrity.json'dadır. Test helper, docs, runtime artifact ve dataset'ler
distribution'a eklenmez. Version/mandatory dependencies/schema değişmedi.

## 22. Export / Experiment Manager

Immutable JSON-backed result; defensive copy ve export-time schema/source/SHA
validation. Checksum authentication imzası değildir. JSON canonical tam evidence,
CSV raw attempt summaries; ortak atomic/no-clobber/formula guard, overwrite explicit.
Rehashed norm/finite/sample/ID/execution/scale/recovery/clock/coverage/instrumentation
tamper reddedilir. Ölçümden sonra gerçek store copied JSON/CSV+SHA verify, generic
attempt metric, finish/close/reopen/read testi geçer. Null norm sıfırla loglanmaz.
Tracking failure retraining yapmaz. Checkpoint dışında ölçümde disk/SQLite write
yoktur; checkpoint I/O source P19 epoch timing dışında kalır. Migration/kind yoktur.

## 23. Kesin dosya listesi

Yeni5: `hyso/performance/gradients.py`, `tests/test_gradient_diagnostics.py`,
`tests/cuda_gradient_diagnostics_validation.py`, `docs/guides/performance-gradient-amp.md`,
bu audit. Değişen3: `hyso/performance/__init__.py`, `tests/test_packaging.py`,
`docs/roadmap/hyso-performance-backlog-v1.1.md`. Toplam8 task dosyası; korunan
kullanıcı gradient farkı ayrı9'uncu working-tree farkıdır. Tarihsel auditler değişmez.

## 24. Korunan SHA256

`tests/test_gradient_accumulation.py` başlangıç/son:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
E/protection.json branch/HEAD, boş index, legacy hash'ler ve kesin farkları kaydeder.
Kullanıcı dosyasına write/stage/restore yapılmadı.

## 25. Güncel35-feature sayımı

P25 planned→implemented: **17 implemented /7 partial /11 planned =35**.
P21/P23 partial, P18 planned/BLOCKED; facade notu korunur, facade yapılmadı.

## 26. P25 PASS / PARTIAL / FAIL

**PASS within verified dense/native scope — implemented.** Sparse/unsupported
adapter'lar complete norm değildir; böyle run partial döner. Finite-range dışı norm
null/unrepresentable; root-cause inference unknown kalır. Scaler/optimizer state
drift veya sahte execution attribution görülmedi. Native AMP unverified değildir.

## 27. B10 GO / PARTIAL / NO-GO

**PARTIAL / full-scope GO yok.** P25'in doğrulanması P21/P23 eksiklerini kapatmaz.
Stable-release onayı verilmez; B9 tarihsel verified GO korunur.

## 28. Açık riskler

B4 native reliability OPEN / root cause UNRESOLVED; B6 CUDA kernel attribution
PARTIAL, B7 overlap PARTIAL, P18 GPU-counter BLOCKED; P21 worker service trace ve
P23 activation lifetime/isolated true window peak PARTIAL. Başka runtime/GPU/OS,
hidden stream ve alternatif AMP/allocator/custom optimizer genel desteği yoktur.

## 29. P21 / P23 completion readiness

P19/P20 IDs, native ownership, state replay ve immutable source association temeli
hazırdır. P21 gerçek worker decode/transform/service trace adapter'ı ister. P23
trusted factory + bounded isolated memory probe ve gerçek saved-activation lifetime
adapter'ı ister. Norm, loader wait veya allocator lifetime counter bu boşlukları
otomatik doldurmaz; bu task onları implemented yapmadı.

## 30. Sonraki adım

B10 completion için önce scope'u ayrı tutulan güvenilir worker recipe/OS guard/
timeout ve parent-state ownership tasarımını tamamlamak, ardından P23 activation
lifetime veya P21 worker-service trace acceptance gate'ini ele almak uygundur.
Sparse diagnostics adapter'ı ayrı validated duplicate-index reduction çalışmasıdır;
sessiz densification ile genişletilmemelidir. Yeni facade geliştirmesi başlatılmadı.
