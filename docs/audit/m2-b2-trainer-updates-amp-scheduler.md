# M2-B2 — Trainer Updates, CUDA AMP ve Scheduler Doğruluğu

Tarih: 2026-10-19 (Europe/Istanbul). **Karar: PASS — M2-B2.**

Nihai kaynaklarda hedefli CPU pytest: **649 PASS / 1 mevcut SKIP**, 0 FAIL/ERROR.
Bağımsız gerçek CUDA gate: **40/40 PASS**. Mevcut minimum Torch 2.2.2 CPU
ortamında güncel installed-wheel: **36/36 PASS**. Bu sonuç M2-B5 tam CPU/GPU
kalite kapısının yerini almaz.

## 1. Git branch ve HEAD

Branch `feature/hyso-models-next`; HEAD
`daa1ffabccbb66f9e6fc080c716d169887d7e722`
(`feat: add minimal generic classification trainer`). Başlangıç çalışma ağacı
temizdi; M2-B1 production/test/raporu commit edilmişti. Branch ve HEAD korundu.
Commit, push, checkout, merge, rebase veya reset yapılmadı.

Dayanaklar: [Roadmap v1.0](../roadmap/hyso-architecture-roadmap-v1.md),
[M2-A DESIGN GO](m2-a-generic-trainer-design.md),
[M2-B1 PASS](m2-b1-generic-trainer-classification.md).

## 2. Mevcut Trainer'da yapılan değişiklikler

Yalnız `hyso/training/trainer.py` production dosyası genişletildi. Küçük private
update helper'ı ve scheduler doğrulamaları eklendi. EBO engine'i ile birleşme,
delegasyon veya kopyalama yapılmadı. Modelin standart forward'u kullanılıyor.

CPU FP32/FP64 korunarak explicit CUDA device, CUDA FP16 AMP, native scaler,
sample-weighted accumulation, gradient clipping ve üç bare scheduler eklendi.
Model taşınmıyor; caller önce device'a taşır, ardından optimizer yaratır.
Parametre/buffer cihazları ve optimizer parameter identity/ownership kontrolleri
constructor ve fit preflight'ta korunuyor. CUDA index varlığı da doğrulanıyor.

## 3. Yeni constructor API

```python
Trainer(
    model, *, optimizer, loss_fn, device="cpu", task="multiclass",
    scheduler=None, amp=False, accum_steps=1, grad_clip=None,
)
trainer.fit(train_loader, val_loader=None, *, epochs=5)
```

`accum_steps` pozitif Python int; bool/type hatası TypeError, sıfır/negatif
ValueError. `amp` exact bool. `grad_clip` bool hariç pozitif finite Real veya None.
CPU amp=True açık TypeError; CPU AMP/BF16 etkinleştirilmez. AMP modelinin
floating parametre/buffer'ları FP32, girişleri FP32 olmalıdır. Forward logits
native autocast nedeniyle FP16 olabilir; master parametreler FP32 kalır.

Desteklenen optimizer profili exact standart non-fused SGD/Adam/AdamW.
Fused/differentiable ayarlar ve custom optimizer subclass'ları açıkça
NotImplementedError alır; LBFGS closure reddi korunur. Bu sınırlama B2'nin
applied-update tespit profilini açık tutar; fused skip davranışı desteklenmiş sayılmaz.

## 4. Accumulation matematiği

Her microbatch için loss gerçek batch_size ile çarpılarak backward yapılır.
Her pencere sonunda gradient gerçek toplam örnek sayısına bölünür:

```text
S = sum(actual_microbatch_sizes)
scaled backward contributions = scaler.scale(mean_loss_i * batch_size_i)
window gradient = unscaled accumulated gradient / S
```

Pencere `accum_steps` microbatch dolunca veya epoch'un son batch'inde flush edilir.
Unequal microbatch, son eksik batch, eksik tail window, A>L ve A=1 aynı kurala
uyar. Epoch başında ve her update girişiminin finally bölümünde optimizer
gradient'leri temizlenir; başarılı epoch'lar arasında gizli accumulation yoktur.

Normalization/clipping yalnız optimizer'ın sahip olduğu parametrelere uygulanır.
Optimizer dışında bırakılmış frozen parametredeki eski gradient, clipping normunu
etkilemez; bunun için ayrı reference regression testi eklendi.

History loss/accuracy gerçek örnek sayısıyla ağırlıklandırılır; accuracy yüzde
[0,100]. B1'in dört history anahtarı, her completed epoch için bir satır ve
None validation alanları aynıdır. Prediction'lar kendi microbatch forward'undan,
ilgili update öncesinden gelir. Weighted CE/ignored target/soft label açılmadı.

## 5. Gradient clipping sırası

```text
scale(loss * B).backward() [her microbatch]
unscale_(optimizer)       [pencere başına bir kez]
grad /= actual_samples
clip_grad_norm_(..., grad_clip, error_if_nonfinite=True)
scaler.step(optimizer)
scaler.update()
zero_grad()
```

CPU FP64 testinde clipping girişindeki gradient bağımsız birleştirilmiş batch
referansına eşit; norm clipping'den önce eşik üstünde, sonra eşik altında.
Native CUDA testinde unscale sonrası sum gradient, clipping girişinde tam olarak
S'ye bölünmüş halde gözlendi; bağımsız FP16 autocast gradient referansıyla da
karşılaştırıldı. Bir pencere için tek unscale ve tek clip çağrısı doğrulandı.

AMP nonfinite gradient'te native unscale found_inf'i kaydeder. Bu durumda clip
uygulanmaz; scaler'ın gerçek skip/backoff yolu çalışır. AMP kapalıysa nonfinite
gradient update öncesinde açık FloatingPointError'dur. Nonfinite logits/loss
ise AMP'de de fail-fast kalır. Finite gradient'lerin norm hesabı overflow olursa
clipping hatası bastırılmaz.

## 6. AMP / GradScaler

Minimum uyumlu `torch.autocast` ve `torch.cuda.amp.GradScaler` kullanılır.
CPU/AMP kapalı yolda native disabled scaler, CUDA AMP'de enabled scaler vardır.
Yeni `torch.amp.GradScaler('cpu')` production gereksinimi eklenmedi.
CPU float32/float64 ve CUDA FP32 forward/backward gerçek testlerden geçti.

CUDA testleri native scaler'ı değiştiren sahte implementasyon kullanmaz.
Deterministik overflow/growth senaryolarında native scaler state_dict'inden
başlangıç scale=128 ayarlanır; gerçek autocast/backward/unscale/step/update
çalışır. Production default native scaler ayarları korunur.

Torch 2.5.1, minimum sürüm için seçilen eski scaler constructor'ına FutureWarning
verir; bu bilinen deprecation'dır. Minimum 2.2.2 profilinde API gerçekten çalıştı.

## 7. Applied optimizer update kanıtı

Üç dar runtime sayaç: `microbatch_count`, `attempt_count`, `update_count`.
Attempt scaler.step denenirken artar. Applied update yalnız optimizer post-step
hook'u çalıştıysa **ve** scaler yeni scale değeri eski değerden küçük değilse artar.
Scaler.step dönüş değeri kullanılmaz. Temporary hook finally'da kaldırılır;
optimizer veya scheduler hata verdiğinde hook/gradient temizliği test edildi.

Üç optimizer × üç scheduler için gerçek CUDA backward hook'uyla ilk pencereye
inf gradient enjekte edildi. Her senaryoda 8 microbatch, 4 attempt, **3 applied
update**, gerçek parametre değişim dizisi `[False, True, True, True]`; scale
**128→64**. OneCycle last_epoch=3, total_steps=4; applied LR dizisi bağımsız
native OneCycle referansıyla aynı. Adam/AdamW state step sayaçları da 3.
Native scaler.step dört attempt'in hepsinde None döndürdü: hem gerçek success
hem skip için None gözlendi.

Tüm pencereleri overflow olan ayrı dokuz CUDA senaryosunda **4 attempt / 0 update**,
parametre değişimi yok; scale **128→64→32→16→8**. OneCycle last_epoch=0.
Growth_interval=1 kullanılan ayrı gerçek senaryoda dört applied update,
OneCycle last_epoch=4 ve **128→2048** scale growth gözlendi.

## 8. Scheduler ownership

Destek exact bare OneCycleLR, StepLR, CosineAnnealingLR. Trainer tek stepping
sahibi; callback üzerinden ikinci scheduler yolu yok. Scheduler.optimizer nesne
identity'si doğrulanır. Fresh fit için last_epoch=0 ve _step_count=1 gerekir;
ileri alınmış scheduler state'i restore yapılmış gibi kabul edilmez.
LR state uzunluğu/finite değerleri ve optimizer group LR eşliği doğrulanır.
StepLR step_size/gamma; Cosine T_max/eta_min; OneCycle phase/boundary ve group
LR planları doğrulanır. Unsupported custom scheduler reddedilir.

StepLR/Cosine her tamamlanan epoch sonunda bir kez ilerler. Tüm update'ler AMP
overflow ile skip olsa bile epoch tamamlanmıştır: iki epoch için last_epoch=2.
Native scheduler bu all-skipped durumda optimizer.step'ten önce scheduler.step
uyarısı verebilir; bu beklenen epoch semantiği ayrı CUDA kaydında görünürdür.
Hata bastırma veya ikinci kez step denemesi yoktur.

ReduceLROnPlateau validation gerektirdiği için açık M2-B3 NotImplementedError alır.

## 9. OneCycle hesaplaması

```text
updates_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps
expected_total_steps = epochs * updates_per_epoch
```

Caller-created total_steps bu bütçeye tam eşit olmalıdır. Kontrol iterator
açılmadan ve optimizer update başlamadan çalışır. Overflow planned attempt
bütçesini büyütmez; scheduler yalnız gerçek applied update başına bir kez ilerler.
CPU testlerinde yanlış bütçe, bozuk phase/LR planı ve ileri scheduler state'i,
loader hiç okunmadan reddedildi. LR sequence/cadence native referanslarla eşleşti.

Tek fit lifecycle korunur: tamamlanmış aynı total epochs çağrısı no-op;
farklı bütçe veya failed-run retry fresh Trainer gerektirir. Tamamlanmış no-op
çağrısında scheduler state'inin değişmediği de doğrulandı.

## 10. CPU test sonuçları

Windows / Python 3.10.18 / Torch 2.5.1+cu121 / torchvision 0.20.1+cu121.
Interpreter: `C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.

| Grup | Gerçek nihai sonuç | Süre |
|---|---:|---:|
| Yeni tests/test_trainer_updates.py | 141 PASS | B1 ile ortak koşu: 9.02 s |
| Mevcut tests/test_trainer_classification.py | 116 PASS | Aynı koşu |
| M1-D tests/test_modular_model.py | 102 PASS | 6.29 s |
| P0 accumulation/scheduler/device precision | 103 PASS / 1 SKIP | 13.76 s |
| Legacy API/state/EMA/checkpoint/resume | 183 PASS | 19.56 s |
| Packaging | 4 PASS | 29.44 s |
| Birbirinden farklı JUnit testcase kimlikleri | **649 PASS / 1 SKIP** | |

Yeni 141 testin 96'sı FP32/FP64 × SGD/Adam/AdamW × A=1/2/4/9 × clip kapalı/açık
× multiclass/binary bağımsız concatenated-batch reference karşılaştırmasıdır.
İki epoch, unequal `[2,1,3,4,1]` batch boyutları, gerçek update gradient/parametre
eşliği, metrikler, tail ve temiz gradient'ler doğrulandı. Kalan testler clipping
sırası, 13 invalid update ayarı, üç scheduler cadence'i, invalid state/config,
exception/hook temizliği, unsupported optimizer ve frozen parametre sınırlarıdır.

P0 ayrı sayımları: test_gradient_accumulation.py **28 PASS**;
test_scheduler_regression.py **69 PASS**; test_device_precision.py **6 PASS/1 SKIP**.
Tek skip mevcut `test_r12_r13_real_cuda_device_smoke`, CPU fixture CUDA'yı gizlediği
için oluştu. Yeni Trainer testlerinde skip/xfail yok. P0'daki mevcut simulated
selection/skip testleri gerçek CUDA kanıtı diye kullanılmadı.

Legacy ayrı sayımları: test_public_api.py **41**, test_published_api_compatibility.py
**1**, test_model_state_compatibility.py **18**, test_models.py **6**,
test_ema_regression.py **18**, test_training_checkpoint.py **34**,
test_ebo_training_resume.py **45**, test_onecycle_checkpoint_compatibility.py **20 PASS**.

Son JUnit dosyaları kimlik bazında ayrık: **650 farklı testcase**, 649 başarılı,
1 mevcut skip. İlk ara koşuların 256 PASS sonucu veya ilk CUDA 32 sonucu nihai
sayılara eklenmedi. Eski test assertion'ları değiştirilmedi; B1 test dosyası HEAD ile aynı.
CPU warning'leri native scaler deprecation ve mevcut sklearn undefined metric
uyarılarıdır; assertion gevşetmesi yapılmadı.

## 11. Gerçek CUDA test sonuçları

`tests/cuda_trainer_updates_validation.py` CPU pytest fixture'ından bağımsız,
CUDA_VISIBLE_DEVICES=0 ile gerçek GPU sürecinde çalıştırıldı. CUDA yoksa script
açık hata verir; sahte PASS veya sessiz skip üretmez.

GPU **NVIDIA GeForce RTX 3060 Laptop GPU**, driver **591.74**, Torch
**2.5.1+cu121**, CUDA build **12.1**. Trainer import provenance checkout altında.

| CUDA senaryosu | Sayı | Sonuç |
|---|---:|---|
| SGD/Adam/AdamW × FP32/native AMP × multiclass/binary | 12 | PASS |
| Üç optimizer × üç scheduler, ilk overflow sonrası toparlanma | 9 | PASS |
| Üç optimizer × üç scheduler, bütün attempt'ler skip | 9 | PASS |
| Native AMP unscale→normalization→clip numeriği | 1 | PASS |
| ModularCNN gerçek CUDA AMP/tail eğitimi | 1 | PASS |
| Native scaler growth ve applied OneCycle eşliği | 1 | PASS |
| Model/device, AMP model/input dtype, invalid CUDA index ve üç fused optimizer reddi | 7 | PASS |
| Toplam (33 pozitif, 7 negatif) | **40** | **PASS** |

Gerçek forward çıktıları FP32 modunda FP32, autocast modunda FP16 gözlendi.
Master parametreler FP32 ve CUDA'da kaldı; identity korundu, gerçek ağırlık
güncellemeleri/clipped normlar gözlendi. Test instrumentation yalnız native
işlemleri gözler; optimizer/scaler/forward yerine simülasyon kullanmaz.

```text
<test-python> -B tests/cuda_trainer_updates_validation.py --output <kanıt>/cuda-results.json
```

Bu yeni B2 gate'tir; tüm legacy CUDA script'leri veya B5 final gate yeniden
koşulmuş sayılmaz. CUDA FP64/custom/fused optimizer için ek doğrulama iddiası yoktur.

## 12. Torch 2.2.2 uyumluluğu

Mevcut minimum runtime:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Python **3.10.18**, Torch **2.2.2+cpu**, torchvision **0.17.2+cpu**.
Yeni dependency kurulmadan üretilen wheel fresh isolated target'a kuruldu.
Repo dışındaki cwd'den `-I -B` ile çalışan installed probe'da FP32/FP64 × üç
optimizer × üç scheduler × A=2/4, iki epoch, unequal `[2,1,2]` batch'ler:
**36/36 PASS**. Native concatenated-batch reference parametre eşliği, clipping,
update sayısı, scheduler LR/counter, history ve temiz gradient'ler doğrulandı.
Trainer import yolu installed target, version=0.2.0; CUDA initialized=False.
Bu CPU minimum profildir; 2.2.2 CUDA AMP doğrulanmış sayılmaz.

## 13. Wheel/sdist packaging

Yeni production modülü oluşturulmadı: MODULES **15** olarak kaldı.
Gerçek `python -m build --no-isolation --sdist --wheel` başarılı. Wheel/sdist
modül byte'ları checkout ile eşleştirildi, metadata ve exclusion kuralları geçti.
İkisi ayrı target'lara `--no-index --no-deps --no-build-isolation` ile kuruldu.
Installed import provenance ve B1 ModularCNN smoke korunarak FP64 AdamW,
accum_steps=2, grad_clip=.1, OneCycle total_steps=4 ile gerçek iki epoch B2
smoke eklendi; dört applied update ve scheduler last_epoch=4 doğrulandı.

Metadata **0.2.0**, Python>=3.10, Torch>=2.2.2, torchvision>=0.17.2 ve mevcut
dependency listesi aynı. Runtime environment dependency'leri değiştirilmedi;
yalnız HYSo artifact'ı yeni target dizinlerine kuruldu.

| Arşiv | SHA-256 |
|---|---|
| hyso-0.2.0-py3-none-any.whl | d89dcaea7d00d3050927d4bfdf7c3b97885644fee9c87c89d59201752f31bc6d |
| hyso-0.2.0.tar.gz | 9ecde2b9d916c0c733f2b9826e13bf45fa449099f1d64fdc494e4358631c0321 |

## 14. Legacy compatibility

Trainer dışındaki 14 mevcut production Python modülü (training/__init__.py dahil),
setup.cfg/pyproject.toml/MANIFEST.in ve B1 sınıflandırma test dosyası Git çalışma
ağacı filtrelerini gözeten object ID karşılaştırmasıyla HEAD ile aynı doğrulandı.
EBO/ELO/CEMO/CNNS/EMA exports, M1 model/attention state_dict yapıları, checkpoint
format v1 ve EBO epoch-resume kodu değiştirilmedi. Gerçek targeted legacy ve P0
koşuları yukarıda ayrı listelendi; eski engine'e refactor/delegasyon eklenmedi.

M1-E'nin 720 PASS/1 SKIP ve M2-B1'in 385 PASS/4 minimum senaryosu geçmiş
baseline kanıtıdır; bunlar yeni tam regression sonucu diye sunulmaz. Tam CPU
collection/sonuç eşliği ve bütün CUDA script'leri M2-B5'e bırakıldı.

## 15. Bilerek desteklenmeyen özellikler

Validation (val_loader None dışında M2-B3 NotImplementedError), EMA,
EarlyStopping, observers, save_last_to/resume_from, checkpoint/resume,
benchmark/experiment/dataset manager, CPU AMP/BF16, fused/custom optimizers,
custom scheduler ve Plateau bu increment'te etkin değildir.
Henüz eklenmeyen keyword'ler açık TypeError üretir. Weighted/ignored/soft-label
loss, Mixup/CutMix ve detection/segmentation sözleşmeleri açılmadı.

## 16. Değiştirilen dosyalar ve kanıtlar

- `hyso/training/trainer.py` — B2 production genişletmesi.
- `tests/test_trainer_updates.py` — 141 yeni CPU reference/boundary testi.
- `tests/cuda_trainer_updates_validation.py` — bağımsız 40 senaryolu native CUDA gate.
- `tests/test_packaging.py` — mevcut assertion'ları koruyan installed B2 smoke eklemesi.
- `docs/audit/m2-b2-trainer-updates-amp-scheduler.md` — bu rapor.

Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b2`.
`trainer-junit.xml`, `modular-junit.xml`, `p0-junit.xml`, `legacy-junit.xml`,
`packaging-junit.xml`, `cuda-results.json`, `installed-minimum.json`,
`verification-summary.json`, `installed-minimum.py`, `summarize.py`.
Packaging altında `pytest-packaging/packaging0/` build.log, wheel-smoke.json,
sdist-smoke.json, dist arşivleri ve kurulu target'ları içerir.

`git diff --check` başarılı. Branch/HEAD/metadata korunuyor. Commit/push,
yayın/release, dependency veya dataset/pretrained weights indirmesi yapılmadı.

## 17. Karar

**PASS — M2-B2.** CPU referans numeriği, gerçek CUDA FP32/FP16 ve native scaler
skip/backoff/growth, optimizer/scheduler ownership, tail/clipping, installed
wheel/sdist ve minimum Torch 2.2.2 CPU kapıları başarılı. Desteklenen profil
standart non-fused optimizer ve belirtilen scheduler/device/precision sınırlarıdır.

## 18. M2-B3'e geçiş önerisi

M2-B3'e geçilebilir. Validation no-grad/mode restore, EMA applied-update cadence,
EarlyStopping ve observer lifecycle bu doğrulanmış update sınırı üzerine eklenmeli.
Plateau yalnız gerçek finite validation metric'i hazır olduğunda etkinleştirilmeli.
Last-before-best-finalization sınırı B4 coherent resume tasarımıyla korunmalı;
checkpoint v1 veya legacy fit kodları değişmeden kalmalı. M2-B5 tam CPU/GPU gate
bu targeted B2 sonucundan bağımsız olarak tamamlanmalıdır.
