# HYSo — P23 R1 / Minimum Torch CPU Regression Recovery

2026-10-21, Europe/Istanbul. **CPU recovery gate PASS.** Güncel sabit kaynakta
minimum Torch tam seçilmiş legacy CPU regresyonu temiz geçti. Production kodu
ve CUDA saved-activation guard değişmedi. **B4 OPEN/UNRESOLVED, P23 PARTIAL,
B10 PARTIAL; eski R1 native crash / NO-GO korunur. CUDA release acceptance yok;
bütün P23 worktree için genel release/commit GO verilmedi.**

## 1. Başlangıç, koruma ve evidence

Branch `feature/hyso-models-next`; HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş.
Başlangıç19 dirty dosya (önceki P23, iki forensic ve CPU-preflight raporları,
kullanıcı gradient farkı dahil) byte düzeyinde korundu. Önce status, tam git
diff ve index okundu; `start-protection.json` / `start.diff` kaydedildi.
Korunan `tests/test_gradient_accumulation.py` SHA256 başlangıç/son:

`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`

Bu dosyaya yazılmadı/stage/restore yapılmadı. Stage/commit/push/reset/checkout/
clean/rebase/merge, Torch/dependency/driver upgrade/download yapılmadı.

Evidence **Er**: `C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-minimum-cpu-recovery-20261021T165402`.
244 kaynak/test/fixture/config/doc dosyası `source-manifest.json` ve
`frozen-source/` içinde; manifest SHA256:
`5822ec230f1d6d780eb4be2d75ad5d3a4a13ef58f8694c4d30748fd848e593e6`.
Bütün final testler bu5 test düzeltmesinden sonra sabitlenen kaynakla çalıştı;
pre/post source hashleri eşit. Önceki19 dirty dosyaya ve47 production modülüne
göre değişiklik yok. Yeni rapor final manifestin çıktı artifact'ı olarak ayrı
hashlenir. Temp sürücüler/loglar repository'ye eklenmedi.

## 2. Önceki kanıt ve hataların yeniden gözlenmesi

[P23 R1](m2-p-b10-p23-isolation-activation.md),
[native investigation](m2-p-b4-p23-native-crash-investigation.md),
[tek forensic diagnostic](m2-p-b4-p23-single-forensic-reproduction.md) ve
[CPU preflight](m2-p-b10-p23-cpu-release-preflight.md) korundu/karşılaştırıldı.
Eski evidence `hyso-p23-cpu-preflight-20261021T161847` altındaki legacy XML,
failure JSON,partial-profile JSON,summary ve source manifest incelendi.

Dokuz case önce mevcut kaynakta isolated seçildi: **9 FAIL**,
11.919s (`before-minimum.xml`). Traceback/dosya/test/runtime/
beklenen-gerçek/düzeltme ayrıntıları `nine-case-analysis.json` içindedir.
Hataların P23 tarafından oluşturulduğu varsayılmadı.

|#|Dosya / test (ilk traceback minimum runtime)|Sınıf|Beklenen–gerçek / neden|
|---|---|---|---|
|1|`tests/test_benchmark_comparison.py` / `test_cpu_condition_mismatch_reports_exact_field[environment.torch_version-2.2.2+cpu]`|Test fixture varsayımı|Aynı2.2.2+cpu metadata değeri incompatible bekleniyordu; actual compatible doğru.|
|2|`tests/test_onecycle_checkpoint_compatibility.py` / `test_onecycle_round_trip_and_next_step[False-cos-sgd]`|Test representation varsayımı|Canonical _anneal_func_type ile native2.2.2 anneal_func/key order karşılaştırılıyordu.|
|3|`tests/test_onecycle_checkpoint_compatibility.py` / `test_onecycle_round_trip_and_next_step[False-cos-adamw]`|Test representation varsayımı|Canonical _anneal_func_type ile native2.2.2 anneal_func/key order karşılaştırılıyordu.|
|4|`tests/test_onecycle_checkpoint_compatibility.py` / `test_onecycle_round_trip_and_next_step[False-linear-sgd]`|Test representation varsayımı|Canonical _anneal_func_type ile native2.2.2 anneal_func/key order karşılaştırılıyordu.|
|5|`tests/test_onecycle_checkpoint_compatibility.py` / `test_onecycle_round_trip_and_next_step[False-linear-adamw]`|Test representation varsayımı|Canonical _anneal_func_type ile native2.2.2 anneal_func/key order karşılaştırılıyordu.|
|6|`tests/test_performance_profiling.py` / `test_experiment_reopen`|Test consumer hatası|Partial operator self_cpu_us=None numeric metric olarak Manager'a gönderiliyordu; invalid_metric_value rejection doğru.|
|7|`tests/test_scheduler_regression.py` / `test_r11_real_cpu_scaler_overflow_skips_even_delegated_fused_updates[False]`|Native capability varsayımı|torch.amp.GradScaler2.2.2 içinde yok; AttributeError eğitimden önce.|
|8|`tests/test_scheduler_regression.py` / `test_r11_real_cpu_scaler_overflow_skips_even_delegated_fused_updates[True]`|Native capability varsayımı|torch.amp.GradScaler2.2.2 içinde yok; AttributeError eğitimden önce.|
|9|`tests/test_trainer_updates.py` / `test_unsupported_optimizer_profile[fused]`|Boundary varsayımı|2.2.2 native fused Adam CPU constructor HYSo admission öncesi RuntimeError.|

Bütün9 case test fixture/capability/boundary veya test consumer hatasıdır;
bu kanıtlarla bir production HYSo comparison/checkpoint/Trainer/Manager defect'i
saptanmadı. Native API/representation farkları doğrudan iki Torch runtime'ında
gözlendi; yalnız version-string tahmini yapılmadı.

## 3. Runtime origin ve CPU admission

|Runtime|Python / Torch / torchvision|Kapsam|
|---|---|---|
|Minimum|3.10.18 /2.2.2+cpu /0.17.2+cpu|Tam seçilmiş55 legacy dosya,P23,profiling,Manager,packaging installed smoke|
|Development|3.10.18 /2.5.1+cu121 /0.20.1+cu121|Seçilmiş16 CPU test dosyası; GPU workload yok|

Her koşu başlamadan Torch/torchvision sürümü ve module origin primary runtime
site-packages altında assert edildi; sys.path önceliği/executable/base executable
ayrı `*-runtime.json` dosyalarına kaydedildi. Mevcut pytest araçları ilgili
runtime'dan sonra eklendi; dependency kurulmadı. Minimum subprocess PYTHONPATH
CPU site-packages önce olacak şekilde kuruldu; önceki Pillow harness problemi
tekrar etmedi.

Development harness gerçek2.5.1 CPU math/scaler/optimizer'ı kullanır;
CUDA visibility=-1, availability discovery kapalı; `_lazy_init`, native
`_cuda_init` ve `_cuda_getDeviceCount` girişleri workload öncesi fail-closed
engellendi. Final initialized=false, blocked native-entry çağrısı0.
Bunlar runtime sürümü veya scaler sahteciliği değildir. Development seçiminde
unguarded fresh profiler/transfer/GPU worker yolları yok; tam minimum suite
CPU build'inde çalışır. Yeni native CUDA validation/reproduction/smoke yok.

## 4. A — comparison fixture

Observed gerçek Torch provenance değiştirilmedi. Sabit comparison fixture
2.2.2+cpu, observed ile aynı olduğunda karşı taraf2.5.1+cu121 seçiliyor ve
farklı olduğu assert ediliyor. Değişiklik yalnız comparison input metadata'sı;
`torch.__version__` veya gerçek runtime identity değiştirilmedi. Incompatible
status ve exact mismatch field assertion korunur. Production compatibility
mantığı gevşetilmedi; bütün comparison/export/tamper testleri yeniden geçti.

## 5. B — OneCycle semantiği ve exact replay

`capabilities-minimum.json` / `capabilities-development.json` native state
keylerini ve gerçek LR observations'ı gösterir.2.2.2 native callable
`anneal_func`,2.5.1 `_anneal_func_type` string kullanır. Checkpoint'in mevcut
canonical v1 bridge sözleşmesi korunur. Save live scheduler state'ini değiştirmez;
yalnız bilinen representation alanı canonical forma karşı ayrı kontrol edilir,
bütün diğer alanlar/yerleşim tam karşılaştırılır. Unknown/conflicting callable
ve rollback/unsafe-load rejection testleri gevşetilmedi.

Her SGD/AdamW × cos/linear × native/legacy case3 gerçek update sonrası RNG'li
checkpoint üretir. Baseline kalan7 update'i yapar; fresh target checkpoint'ten
resume olur. Her gerçek post-step sayısı7, last_epoch/total_steps10, her LR ve
counter/draw dizisi; final optimizer (LR,momentum/Adam beta1/moments),model,
canonical scheduler ve Python/NumPy/CPU Torch RNG exact eşit kontrol edilir.
Native canonical alan zaten varsa eski native representation tam eşliği de
korunur. Native iki runtime'ın ilk4 LR observation'ı ayrıca aynı çıktı.

Bu within-runtime native baseline/replay karşılaştırmasıdır; farklı Torch
sürümleri arasındaki bütün model aritmetiğinin bitwise aynı olduğu iddia
edilmez. Optimizer state'i elle rewrite edilmedi; production adapter değişmedi.

## 6. C — native CPU GradScaler capability

2.2.2'de `torch.amp.GradScaler` yok;2.5.1'de gerçek native CPU scaler mevcut.
Capability callable yokken iki case, gerçek HYSo `Trainer(...amp=True)` CPU
admission'ının `TypeError: CPU amp is unsupported` verdiğini; disabled native
scaler'ın disabled/empty state olduğunu; optimizer state ve gradientlerin
oluşmadığını doğrular. XML property:
`cpu_native_overflow=unsupported_no_native_CPU_GradScaler`.
**Bu iki PASS native overflow gerçekleşti demek değildir; skip/xfail değil,
açık unsupported-boundary testidir.**

2.5.1'de eski gerçek CPU scaler kontrolü korunur: ilk gradient Inf,
native scale128→64,3 applied update; delegated fused SGD'de4 step invocation
ama ilkinde kernel update yok. LR/scheduler recovery ve finite parameters
kontrolleri PASS. Property `native_CPU_GradScaler`. Sahte scaler eklenmedi.
Bu legacy CPU test control'ü, HYSo Trainer için yeni CPU AMP desteği ilan etmez;
Trainer CPU AMP admission her iki runtime'da da unsupported kalır.

## 7. D — fused Adam boundary

Native `_get_fused_kernels_supported_devices()` minimum runtime'da CPU içermez.
Gerçek Adam(fused=True) constructor rejection'ı exact RuntimeError/device
mesajıyla assert edilir; parameters CPU/gradNone kalır; non-fused Adam HYSo
admission'ı başarılı ve state boş kalır. Hiç fused native update iddiası yok.
2.5.1 native capability CPU içerir, constructor geçer ve HYSo'nun mevcut
NotImplementedError fused admission'ı doğrulanır. XML capability property'leri
hangi boundary'nin gözlendiğini açıkça ayırır; custom optimizer rejection da
korunur. Validation gevşetilmedi veya exception gizlenmedi.

## 8. E — partial profiler / Manager consumer

Worker'ın mevcut `_duration` sözleşmesi invalid/missing native duration'ı
(negative/nonfinite/None dahil) None olarak saklar; operator aggregate partial
olur. Eski native `aten::_convolution.self_cpu_us=null` böyle korunmuştu.
Eldeki export ham geçersiz self-time nedenini tam ayırt etmez; profiler/driver
root cause'u bu veriden ilan edilmedi. Hata, test consumer'ın her row'u numeric
metric sanmasıydı. Manager'ın null rejection'ı doğru ve değişmedi.

Consumer artık None row'un partial olduğunu doğrular, tam artifact'i aynen
saklar ve yalnız mevcut numeric self timing'leri loglar. Reopen sonrası her
axis_index/axis_kind/value birebir kontrol edilir; sahte0/metrik üretimi yok.
Natural timing her koşuda partial olmayabildiğinden bu görevdeki pre-fix gerçek
partial capture, değiştirilmeden güncel consumer'a ayrıca verildi:
`partial_replay-minimum.xml` **1 PASS**. Artifact equality ve Manager metrics
assertion'ları geçti. Bu preserved-data consumer replay'idir, yeni profiling
ölçümü olarak sunulmaz. Null/NaN/Inf/type rejection testleri de full suite'te
PASS. Unsupported domain measured olarak yeniden etiketlenmedi.

İlk hedefli düzeltme denemesinde yeni reopen assertion'ı yanlış `step` read
key'i nedeniyle8 PASS/1 FAIL verdi; persisted API `axis_index` olarak düzeltildi.
`after-initial-consumer-fix.*` korunur. Final manifest bu düzeltmeden sonradır.

## 9. Güncel kaynak test sonuçları

|Koşu|Sonuç|Süre / kanıt|
|---|---|---|
|Minimum isolated9 final|9 PASS,0 FAIL/ERROR/SKIP|10.394s;after-minimum.xml|
|Minimum tam seçilmiş legacy CPU|2825 PASS,0 FAIL/ERROR/SKIP|585.389s;legacy-minimum.xml|
|Development seçilmiş CPU16 dosya|874 PASS,0 FAIL/ERROR/SKIP|109.713s;development-development.xml|
|Yeni P23 isolation/activation CPU|58 PASS (29+29),0 FAIL/ERROR/SKIP|268.569s;p23-minimum.xml|
|Tam packaging testleri|4 PASS,0 FAIL/ERROR/SKIP|163.697s;packaging-minimum.xml|
|Gerçek partial artifact consumer replay|1 PASS|4.581s;partial_replay-minimum.xml|

Full minimum legacy, tüm `test_*.py` dosyalarından ayrı koşulan P23 iki dosyası
ve packaging hariç55 dosyadır; gerçek CUDA smoke explicit **1 deselect**.
Production için blanket skip/xfail eklenmedi. Önceki2814 PASS, intermediate
ve targeted sonuçlar yeni temiz full toplamına eklenmedi; tabloda kapsamlar
ayrı, fresh execution sonuçlarıdır. Dev16 dosya listesi `run_tests.py` içinde;
comparison,OneCycle,scheduler,checkpoint,Trainer updates/restore/replay,
gradient accumulation/diagnostics,training memory/details/throughput,
EBO resume,Experiment integration/metrics,benchmark export/result kapsamı.
Development sonucu tam55-file suite veya CUDA sonucu diye sunulmaz.

P23 yeniden koşusu exact state/RNG, interrupted/resumed/completed checkpoint,
exception/GC-disabled weakref cleanup,immutable-result,JSON/CSV,
Manager artifact verification/reopen/tracking-error ve rehashed tamper
kontrollerini kapsar. Üretim payload/lifetime matematiği değişmedi.

## 10. Minimum installed smoke ve byte equality

Mevcut Python3.10.18/Torch2.2.2+cpu içinde yeni wheel ve sdist ayrı target'larda
installed CPU smoke geçti. Test packaging command adapter'ı build ve no-deps
`--target` install için yalnız mevcut build-tool interpreter'ını kullanır;
installed eğitim/smoke CPU interpreter'ında, repo dışında yürür. Her smoke
başında Torch/torchvision origin/version/sys.path ve CUDA initialized=false
kaydı alınır. Global environment'a HYSo/dependency kurulmadı veya indirilmedi.

47/47 module byte eşliği:
**current source == wheel == sdist == installed-wheel == installed-sdist**.
47 installed modülün tamamı doğru wheel target'tan CPU'da import edildi.
Python3.10 AST,metadata,Pythonspec,dependency/archive exclusions ve config
kontrolleri PASS. Wheel/sdist smoke gerçek AdamW CPU isolated activation,
Trainer/checkpoint/RNG replay,export/Manager ve executable guide kapsamını
korur; her ikisinde `isolated_activation=true`,CUDA initialized=false.

- Wheel SHA256 `03ff4e24b8d0b86abbfc1279f4c1a801a00ebd2a2833d05384d8c5ec6fe38fef`.
- Sdist SHA256 `7086eb613739f32509dd8fe3944643ab157d8f3e0db9d0ab5fa88a737f3e4be1`.
- `archive-integrity.json`:47 module hash ve iki installed target/smoke sonucu.
- Runtime METADATA/RECORD karşılaştırmaları minimum70,development478 dosyada eşit. Minimum70 record önceki preflight finaliyle de eşit; byte-level runtime memory attestation iddiası yok.

## 11. Tam değişiklik listesi ve korunmuş worktree

Bu görevde değişen yalnız5 test dosyası ve bu yeni audit:

- `tests/test_benchmark_comparison.py`
- `tests/test_onecycle_checkpoint_compatibility.py`
- `tests/test_scheduler_regression.py`
- `tests/test_trainer_updates.py`
- `tests/test_performance_profiling.py`
- `docs/audit/m2-p-b10-p23-minimum-cpu-regression-recovery.md` (yeni)

Önceki19 dirty dosya byte'ları aynen korundu. Final toplam25 dirty:
10 modified /15 untracked; bütün inventory ve SHA'lar
`final-protection.json` içinde. `git diff --check` PASS.
Production comparison/checkpoint/Trainer/profiler/Manager ve yeni P23 modülleri
sözleşmeye uygun davrandığından production düzeltmesi gerekmedi. Test-only
scope yeterliydi; production guard ve eski audit/NO-GO kayıtları korunur.

## 12. Kalan FAIL/SKIP/unsupported ve readiness

Final koşularda **0 FAIL /0 ERROR /0 SKIP**. Bir gerçek CUDA smoke deselect;
minimum native CPU GradScaler overflow2 case açık unsupported ve fused Adam
native-constructor unsupported olarak yukarıda ayrıldı. Bunlar native AMP/
fused update başarıları olarak sayılmadı. Custom/checkpointed/retained saved
activation unsupported sınırları ve physical VRAM/storage-free lifetime
iddiası olmaması değişmedi. Uyarılar deprecation/sklearn/lazy/sparse native
uyarılarıdır; dependency upgrade ile giderilmedi.

CPU recovery/test-only değişiklik seti incelemeye hazır; CPU gate **PASS**.
Ancak bütün mevcut P23 worktree için genel commit/stable release GO yok:
**B4 OPEN/UNRESOLVED; P23 PARTIAL; B10 PARTIAL.** CUDA saved-activation
production admission disabled kalır; eski R1 crash ve NO-GO kaydı geçerlidir.
Eski C source identity hâlâ `unverified_source_identity`; tek forensic PASS
root cause çözümü veya final-source/installed CUDA acceptance değildir.
P21/B10 diğer partial sınırları da bu görevle kapanmadı.

Yeni CUDA reproduction,comparison veya guard değişikliği için ayrı açık onay,
sealed source/runtime manifesti ve bounded Job/timeout/metadata/cleanup planı
gerekir. Bu görev kapsamında CUDA workload sayısı **0**. Stage/commit/push
ve diğer yasak git işlemleri yapılmadı.
