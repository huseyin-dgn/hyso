# M2-B3 — Trainer Validation, EMA, EarlyStopping ve Observer Lifecycle

Tarih: 2026-10-19 (Europe/Istanbul). **Karar: PASS — M2-B3.**

Gerçek hedefli CPU koşuları: **707 farklı testcase PASS**, 0 FAIL/ERROR/SKIP/XFAIL.
Yeni bağımsız native CUDA lifecycle gate: **16/16 PASS**; mevcut B2 CUDA update
gate'i güncel kaynakta yeniden **40/40 PASS**. Minimum Torch 2.2.2 CPU ortamında
güncel installed-wheel lifecycle: **26/26 PASS**. Tam CPU/GPU regression B5'e,
full-state checkpoint ve epoch-resume B4'e aittir.

## 1. Branch ve HEAD

Branch `feature/hyso-models-next`, HEAD
`f2633c3b73e41c03aa054466013118eecaa6f36c`
(`feat: add CUDA AMP accumulation and scheduler support`). Başlangıç çalışma
ağacı temizdi; M2-B2 commit edilmişti. Beklenmedik kullanıcı değişikliği yoktu.
Branch ve HEAD korunuyor. Commit/push/checkout/merge/rebase/reset yapılmadı.

Dayanaklar: [Roadmap v1.0](../roadmap/hyso-architecture-roadmap-v1.md),
[M2-A tasarımı](m2-a-generic-trainer-design.md),
[M2-B1](m2-b1-generic-trainer-classification.md),
[M2-B2](m2-b2-trainer-updates-amp-scheduler.md).
Gerçek WithSE.EMA ve callbacks.callbacks.EarlyStopping kodları incelendi;
tasarım önerisindeki algoritmalar yerine mevcut implementasyonlar kullanıldı.

## 2. Yeni Trainer API

```python
Trainer(
    model, *, optimizer, loss_fn, device="cpu", task="multiclass",
    scheduler=None, amp=False, accum_steps=1, grad_clip=None,
    ema_decay=None, early_stopping=None, callbacks=(),
)
trainer.fit(train_loader, val_loader=None, *, epochs=5)
```

B1/B2 default'ları ve strict classification loss/ownership/device/precision
sözleşmeleri korunur. `evaluation_model` property, validation/final inference
hedefini verir: live model veya mevcut EMA'nın `ema_model` shadow'u.
`ema_update_count`, applied optimizer update sayısından ayrı tutulur; EMA
kapalıyken sıfırdır. Yeni framework, registry, callback adapter modülü veya
public model export eklenmedi. Production değişikliği yalnız trainer.py'dedir.

## 3. Validation davranışı

Validation yalnız val_loader sağlandığında training epoch sonrasında çalışır.
Finite/sized/nonempty loader preflight'ı optimizer adımı başlamadan yapılır.
Validation epoch başına bir kez iterate edilir; gizli örnek batch okuması,
dataset/transform değişikliği veya validation verisini yeniden örnekleme yoktur.

Evaluation hedefi eval mode'a alınır; torch.no_grad altında mevcut loss/target
ve AMP precision sözleşmesiyle forward yapılır. Optimizer/scaler/EMA update,
gradient üretimi veya scheduler step validation fazına ait değildir.
Loss `sum(batch_mean_loss * actual_B) / total_samples`, accuracy
`100 * correct / total_samples`. Binary ve multiclass aynı B1 hard-target
kurallarına uyar. Validation yoksa history val_loss/val_acc alanları None kalır.

Her alt modülün önceki training flag'i ayrı kaydedilir. Başarıda ve exception'da
finally ile bu flag'ler doğrudan geri kurulur; yalnız root.train(old_flag)
kullanılmaz. Böylece mixed-mode kaybolmaz. Eğitim başlangıcı da bilinçli eval
çocukları korur: root train veya yalnız classifier train olan mixed yapı
korunur. Tamamen eval olan standart model fit'e girince normal train mode'a alınır.

Gerçek testler validation sırasında BN floating/integer buffer'ları, model
parametreleri ve optimizer state'inin değişmediğini; forward'da autograd'ın
kapalı olduğunu; unequal/tail batch metriğini ve exact mode restore'u denetler.
Invalid target/shape, nonfinite logits/loss, len/iteration uyuşmazlığı ve forward
exception açık hatadır. EMA değerlendirmesi başarısızsa live modele fallback yoktur.

## 4. EMA cadence ve shadow modeli

Doğrudan `hyso.models.WithSE.EMA` kullanılır. Constructor'da live modelden
deepcopy shadow oluşturulur; parametre gradient'leri disabled ve shadow eval'dir.
Model veya shadow sessizce taşınmaz. Parametre/buffer isimleri ve sıraları,
shape/dtype/device eşliği, parametre bağımsızlığı ve shadow requires_grad=False
kontrol edilir; kontroller fit ve update/değerlendirme sınırlarında da yapılır.

`ema_decay=None` kapalı, finite [0,1] (0 dahil) açıktır; bool, NaN/inf ve aralık
dışı değerler reddedilir. Native EMA'nın parametre averaging algoritması ve
isimle tam buffer kopyası değiştirilmedi. Integer BN num_batches_tracked ve
non-persistent registered buffer semantiği mevcut EMA'ya aittir.

Her gerçek applied optimizer update sonrasında tam bir EMA.update çağrısı vardır.
Microbatch başına veya atlanmış AMP attempt'inde çağrılmaz. Native post-step hook
ve scaler backoff birlikte applied-update kapısını korur. EMA update, aynı applied
update'in OneCycle step'inden önce tamamlanır. EMA hatası görünürdür; gradient
temizliği update finally bölümünde kalır. Applied optimizer değişimi geri alınmış
gibi raporlanmaz; B3 rollback transaction değildir.

CPU testinde A=1/2/9 ve decay=0/.8/1 için gerçek optimizer post-hook'unda ayrı
native EMA referansı güncellendi; shadow state tam eşleşti. BN buffer'ları
averaging yerine live'dan kopyalandı. Bu cadence, legacy EBO'nun microbatch EMA
cadence'inden bilinçli olarak farklıdır; EBO kodu değiştirilmedi.

## 5. EarlyStopping snapshot/finalization

Doğrudan standart `hyso.callbacks.callbacks.EarlyStopping` instance'ı kabul edilir.
Gerçek API `patience`, **delta** (min_delta anlamında), `verbose`,
`restore_best_weights` kullanır. Yeni improvement algoritması yazılmadı.
Patience pozitif Python int, delta negatif olmayan finite Real; restore/verbose
bool olmalıdır. Kullanılmış best/counter/terminal state fresh eğitim diye
yeniden kabul edilmez; typed resume bu aşamada yoktur.

EarlyStopping validation gerektirir; yoksa ilk update öncesinde hata. Native
`val_loss < best_loss - delta` improvement, counter/patience ve deepcopy
state_dict snapshot davranışı kullanılır. Snapshot aynı evaluation hedefinden
alınır: EMA varsa shadow, yoksa live. BN running_mean/var ve integer counter
snapshot'a dahildir; sonraki değişimler best_state'i değiştirmez.

Son tamamlanan epoch'un observer ve coherent last-state boundary'sinden sonra,
restore_best_weights=True ise aynı hedefe native restore_weights uygulanır.
EMA best shadow live modele yüklenmez. Live optimizer momentleri geri sarılmaz;
bu final inference seçimi, training continuation restore'u değildir.
Best restore veya early termination sonrasında Trainer terminal'dir; yeni fit
açık hata verir. B1/B2'nin completed-budget no-op davranışı, finalization olmayan
başarılı tamamlanmış lifecycle için korunur.

Testler gerçek kötüleşen validation loss ile iki epoch'ta early termination,
live/shadow best restore, restore kapalı davranış, patience/delta ve gerçek
iyileşmeyi doğrular. Epoch observer son continuing state'i, train_end observer
restore edilmiş best target'ı görür. Epoch observer hatasında finalization'a
geçilmez. Legacy EarlyStopping'e serialization metodu eklenmedi.

## 6. ReduceLROnPlateau stepping

Exact native ReduceLROnPlateau, aynı optimizer identity'siyle kabul edilir.
Val_loader zorunludur; epoch başına bir kez gerçek finite val_loss ile step edilir.
Metric minimize edilen loss olduğu için mode='min' istenir. Factor, patience,
cooldown, threshold/eps ve fresh state kontrolleri açıktır. Scheduler hatası
bastırılmaz veya tekrar denenmez.

Torch 2.2.2 Plateau constructor'ında ne get_last_lr ne de ilk step öncesi _last_lr
vardır. Bu gerçek minimum runtime introspection'ıyla görüldü; ilk LR kontrolü
optimizer group'larını kullanır, _last_lr oluştuğunda o state doğrulanır.
Minimum installed-wheel testleri bu yolu gerçekten çalıştırdı.

OneCycle hâlâ yalnız applied optimizer update başına; StepLR/Cosine completed
epoch başına ilerler. Sıra validation→epoch scheduler→EarlyStopping→epoch_end'dir.
Plateau LR dizisi ayrı native scheduler referansına ve gerçek val_loss'a göre
test edildi; exception sonrası tek çağrı ve mode restore da doğrulandı.

## 7. Callback lifecycle

Observer protokolü `callback(event, logs)`; olaylar train_begin, epoch_end,
train_end. Tuple/list'teki synchronous iki argüman kabul eden callable'lar
desteklenir. Legacy EarlyStopping/ModelCheckpoint/LRSchedulerCallback nesneleri
observer diye tahmin edilmez; yanlış signature/async callback açık hata verir.

Logs yalnız plain bağımsız değerlerdir: son train/val loss/acc veya henüz yoksa
None, epoch (begin=0, sonra completed epoch sayısı), applied_updates,
optimizer_attempts, ema_updates, learning_rate group listesi ve early_stop.
Her observer ayrı deepcopy alır; bir callback'in log/list mutation'ı history'yi,
başka observer'ı veya saklanmış sonraki snapshot'ı değiştirmez. Logs model,
gradient veya optimizer referansı içermez. Callback scheduler/gradient/restore
sahibi değildir; arbitrary Python closure'larının harici yan etkileri sandbox
veya transaction garantisi altında değildir.

```text
train_begin
training → validation → epoch scheduler → EarlyStopping snapshot
→ history/epoch_end observer → coherent last-state boundary
→ stop kararı → final best-target restore → train_end
```

Hata fail-fast'tır: train_begin başarısızsa training başlamaz; validation/scheduler/
epoch_end hatasından sonra sonraki olaylar/finalization çalışmaz. Validation
mode'ları finally ile geri kurulur. Train_end hatası best restore sonrasında
görünürdür; retry veya sahte completed no-op üretilmez. `_finished` yalnız
train_end başarıyla döndükten sonra set edilir. Training update/buffer değişimleri
rollback edilmez; custom callback replay veya dış I/O transaction eklenmedi.

## 8. CPU testleri

Windows / Python 3.10.18 / Torch 2.5.1+cu121 / torchvision 0.20.1+cu121.
Mevcut interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.

| Grup | Nihai farklı testcase sonucu | Süre |
|---|---:|---:|
| Yeni validation | 32 PASS | Dört Trainer dosyasının ortak son koşusu: 8.13 s |
| Yeni lifecycle | 53 PASS | Aynı koşu |
| B1 classification | 116 PASS | Aynı koşu |
| B2 updates | 141 PASS | Aynı koşu |
| P0 scheduler/EMA/EarlyStopping | 94 PASS | 13.22 s |
| Legacy API/model-state/M1-D/checkpoint/resume | 267 PASS | 18.11 s |
| Packaging | 4 PASS | 28.18 s |
| Toplam, JUnit kimliklerine göre ayrık | **707 PASS / 0 SKIP** | |

Yeni validation matrisi multiclass/binary × FP32/FP64 × EMA kapalı/0/.8;
eşit olmayan batch ve son tek örnek, no-grad, BN/state/mode korunumu ve
exception/no-fallback kontrollerini kapsar. Lifecycle testleri native EMA
referansı, schema/alias reddi, best snapshot/finalization, Plateau, callback
isolation/order/exception ve terminal Trainer davranışını kapsar.

P0: test_scheduler_regression **69**, test_ema_regression **18**,
test_early_stopping_regression **7 PASS**. Legacy: test_public_api **41**,
test_published_api_compatibility **1**, test_model_state_compatibility **18**,
test_models **6**, test_modular_model **102**, test_training_checkpoint **34**,
test_ebo_training_resume **45**, test_onecycle_checkpoint_compatibility **20 PASS**.

Kullanıcı, artık açılan validation/Plateau ile çelişen iki eski unsupported testi
yeni davranışı doğrulayan daha güçlü testlerle güncellemeyi açıkça onayladı.
B1 testi exact post-train validation loss/accuracy ve fazladan optimizer update
olmadığını; B2 Plateau case'i validation yokken update öncesi reddi, validation
ile iki step/best metric/update sayısını denetler. Diğer eski test assertion'ları
ve toleranslar aynı kaldı; test dışlama/xfail ile çelişki gizlenmedi.

Ara koşular toplamda tekrar sayılmadı. İlk yeni fixture koşusunda 30 PASS/1 FAIL
vardı: eval BatchNorm'un trainable parametreleri de frozen sanılmıştı; fixture
requires_grad_(False) ile düzeltildi. Son mixed-root test eklemesiyle 32 validation
PASS ve 342 Trainer PASS alındı. Sonrasında source'a bağlı packaging/CUDA/minimum
gates yeniden çalıştırıldı. P0/legacy kaynakları bu son değişiklikten etkilenmez;
Trainer'ı kullanmayan bu grupların başarılı sonuçları tekrar sayılmadı.

Warning'ler native minimum-uyumlu GradScaler deprecation ve mevcut sklearn
undefined metric uyarılarıdır. Tam CPU suite koşulmadı; geçmiş 720/1, 385,
649/1 sonuçları yeni tam suite PASS diye sunulmaz.

## 9. Gerçek CUDA testleri

GPU **NVIDIA GeForce RTX 3060 Laptop GPU**, driver **591.74**, Torch
**2.5.1+cu121**, CUDA build **12.1**. Bağımsız süreçlerde CUDA_VISIBLE_DEVICES=0;
Trainer import yolu checkout altında doğrulandı. CUDA yoksa script açık hata
verir; CPU ortamında sahte PASS veya skip üretmez.

| Yeni cuda_trainer_validation_lifecycle.py | Sayı | Sonuç |
|---|---:|---|
| FP32/FP16 AMP × EMA kapalı/açık × iki mixed-root durumu; validation, ES, Plateau | 8 | PASS |
| SGD/Adam/AdamW × ilk overflow/tüm overflow; EMA applied cadence ve BN copy | 6 | PASS |
| FP32/FP16 EMA validation exception, exact mode restore/no fallback | 2 | PASS |
| Yeni B3 gate toplamı | **16** | **PASS** |

CUDA normal lifecycle: iki gerçek applied update, Plateau last_epoch=2,
validation forward'larında no-grad ve FP32/FP16 çıktılar, best target restore,
canlı model/shadow ayrımı ve terminal davranış gözlendi.
Overflow: dört attempt'te üç applied update/üç EMA update, scale128→64;
all-skipped durumda dört attempt/sıfır applied/sıfır EMA update, scale128→8.
BN live/shadow counter çiftleri toparlanmada `[3,3]` ve `[6,6]`, tüm skip'te
`[3,0]` ve `[6,0]`: atlanmış update'in forward buffer'larının live'da ilerlemesi,
EMA'nın ise yalnız gerçek applied update'te buffer kopyalaması açıkça doğrulandı.
Bağımsız native EMA referansıyla shadow state birebir eşleşti.

Mevcut `cuda_trainer_updates_validation.py` de son production kaynakta yeniden
**40/40 PASS**: native clipping, accumulation, OneCycle/epoch scheduler,
skip/backoff/growth ve optimizer sınırları korunuyor. İki script'in 56 case
kimliği ayrık; CPU pytest sayılarına eklenmedi. Ara 12 B3 CUDA sonucu tekrar
sayılmadı. Bu targeted B3+B2 gate'idir; bütün legacy CUDA script'leri ve B5
final CUDA gate çalıştırılmış sayılmaz.

## 10. Torch 2.2.2 uyumluluğu

Mevcut minimum runtime:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Python3.10.18 / Torch **2.2.2+cpu** / torchvision **0.17.2+cpu**.
Güncel son wheel yeni `installed-minimum-final` target'ına --no-index/--no-deps
ile kuruldu. Repo dışındaki cwd'den isolated -I -B Python ile import provenance,
version0.2.0 ve CUDA initialized=False doğrulandı.

Multiclass/binary × FP32/FP64 × EMA kapalı/0/.8 × Plateau kapalı/açık:
**24 gerçek senaryo**; accumulation, clipping, validation, iki epoch'ta ES,
same-target best restore, observer olay sırası ve terminal state geçti.
İki ek FP32/FP64 BN floating/integer buffer ve mixed-mode validation senaryosu
ile **26/26 PASS**. Bu installed CPU minimum profilidir; 2.2.2 CUDA testi değildir.
Yeni dependency kurulmadı ve eski ortam/kurulu target'lar değiştirilmedi.

## 11. Packaging

Yeni production modülü yok: MODULES **15** kaldı. Gerçek wheel/sdist build,
arşiv byte eşliği/exclusion/metadata testleri ve iki ayrı installed target import
kontrolü geçti. B1/B2 installed smoke'ları korunarak B3 validation+EMA+native
EarlyStopping+Plateau+observer+best-shadow restore smoke'u eklendi.
İki eğitim epoch'u, iki applied/EMA/Plateau step ve live/shadow ayrımı doğrulandı.

Metadata **0.2.0**, Python>=3.10, Torch>=2.2.2 ve torchvision>=0.17.2 sözleşmesi,
diğer dependency listesi aynı kaldı. Arşivler kanıt kökünün
`pytest-packaging-final/packaging0/dist/` dizininde:

| Arşiv | SHA-256 |
|---|---|
| hyso-0.2.0-py3-none-any.whl | 32e0c189e4899c08bbb68677206642395fe4176ba448f0f7fcb2d02b8c1def36 |
| hyso-0.2.0.tar.gz | 4e94eeb8987e34a90fab00bfbc3c1ba322e81195ac29bfe1bb9d9be506cd6fe0 |

## 12. Legacy compatibility

Trainer dışındaki 14 mevcut production Python modülü, paket config dosyaları ve
onaylı üç test dosyası dışındaki mevcut testler, Git çalışma ağacı filtrelerini
gözeten object ID karşılaştırmasıyla HEAD ile aynı doğrulandı. Model public
exports/state_dict yapıları, EBO/ELO/CEMO/CNNS/EMA, legacy callback algoritmaları,
checkpoint format v1 ve EBO epoch-resume kodu aynı. Yeni EMA cadence'i legacy
EBO'ya uygulanmadı. Legacy üretim dosyaları refactor edilmedi.

## 13. Checkpoint/resume için korunan sınırlar

Coherent last-state boundary, epoch scheduler/ES snapshot/history/observer'dan
sonra ve stop/final best restore'dan önce belirgin bırakıldı. B4 burada continuing
model/optimizer/EMA/history/RNG/callback state'ini kaydedebilir. Bu görev dosya
yazma, full-state restore, typed ES serialization veya epoch-resume eklemedi.
save_last_to/resume_from ve model_config henüz açık unexpected-keyword hatasıdır.
İlk B4 exact-resume profili callbacks=() gerektirecek; stateful observer replay
ve dış I/O transaction garantisi bu aşamaya taşınmadı.

## 14. Değiştirilen dosyalar ve kanıtlar

- `hyso/training/trainer.py` — validation/EMA/ES/Plateau/observer lifecycle.
- `tests/test_trainer_validation.py` — 32 yeni gerçek CPU testi.
- `tests/test_trainer_lifecycle.py` — 53 yeni gerçek CPU testi.
- `tests/cuda_trainer_validation_lifecycle.py` — 16 native CUDA senaryosu.
- `tests/test_trainer_classification.py` — onaylı eski unsupported validation testinin güncellenmesi.
- `tests/test_trainer_updates.py` — onaylı eski unsupported Plateau case'inin güncellenmesi.
- `tests/test_packaging.py` — installed B3 smoke eklemesi.
- `docs/audit/m2-b3-trainer-validation-ema-lifecycle.md` — bu rapor.

Kanıt kökü `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b3`:
`trainer-final-junit.xml`, `p0-junit.xml`, `legacy-junit.xml`, `packaging-junit.xml`,
`cuda-lifecycle.json`, `cuda-b2-regression.json`, `installed-minimum.json`,
`verification-summary.json`, `installed-minimum.py`, `summarize.py`.
Final packaging alt dizininde build.log, wheel/sdist-smoke.json, arşivler ve
kurulu hedefler bulunur. Daha eski ara artefact'lar final sonuçla karıştırılmaz.
`git diff --check` başarılı. Commit/push, version bump, release/upload, yeni
dependency veya dataset/pretrained weights indirmesi yapılmadı.

## 15. Karar

**PASS — M2-B3.** Sample-weighted validation, exact mode restore, native EMA'nın
applied-update cadence'i, same-target ES snapshot/finalization, Plateau stepping,
observer isolation ve exception sınırları; mevcut B1/B2 regresyonları, gerçek
CUDA ve installed minimum runtime kapıları başarılı. Tam Trainer resume
veya tam CPU/GPU regression desteği iddia edilmez.

## 16. M2-B4'e geçiş önerileri

M2-B4'e geçilebilir. Existing checkpoint v1 altında generic_trainer metadata,
typed EarlyStopping state adapter'ı, coherent last-before-best kayıt sırası ve
outer restore transaction/RNG/loader plan eşliği uygulanmalı. Completed/terminal
resume ayrı doğrulanmalı; final best weights üzerinden eski optimizer ile
sessiz continuation açılmamalı. Stateful callbacks ve arbitrary external state
ilk exact-resume profiline alınmamalı. B5 tam CPU/GPU identity/regression gate'i
ayrı görev olarak korunmalıdır.
