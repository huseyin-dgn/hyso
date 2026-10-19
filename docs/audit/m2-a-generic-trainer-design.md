# M2-A — Generic Trainer Mimari Tasarımı

Tarih: 2026-10-19. **Karar: M2-A DESIGN GO.** Bu belge önerilen sözleşmedir;
Trainer implementasyonu veya yeni runtime regression sonucu değildir.
Sabit kapsam: [Roadmap v1.0](../roadmap/hyso-architecture-roadmap-v1.md).
M2-B1'e geçilebilir; Benchmark/Experiment Manager bu implementasyon serisinin dışında.

## 1. Branch, HEAD ve inceleme kanıtı

Branch `feature/hyso-models-next`, HEAD
`757efc985822cc4b254d528ef6c173e28faafbbc`
(`test: complete M1 architecture quality gate`). Başlangıç çalışma ağacı temiz;
M1-E raporu ve bağımsız GPU script'i commit edilmiş. Branch/HEAD korunuyor.

Gerçek kaynak üzerinden incelendi: bcam.EBO.fit/callbacks ve private resume
yardımcıları; callbacks.py; WithSE.EMA/CEMO; fit.CNNSENormalWithFit;
checkpoint.py; augmentation.py; ModularCNN ve ilgili regression testleri.
Paketin mevcut minimumları Python>=3.10/Torch>=2.2.2/vision>=0.17.2, sürümü 0.2.0.

M1-E'nin 720 PASS/1 SKIP, 17 legacy CUDA, 18 M1 CUDA ve 2.2.2 installed-wheel
5/5 sonucu **geçmiş kanıttır**; bu dokümantasyon görevinde yeniden koşulmadı.
Mevcut Torch 2.2.2+cpu ile yalnız import/inspect kontrolü yapıldı: torch.autocast,
torch.cuda.amp.GradScaler, optimizer post-step hook ve load API'si mevcut;
FORMAT_VERSION=1; ModularCNN.fit ve EarlyStopping.state_dict mevcut değil.
CUDA initialized=False. Kanıt kökü:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2a-7465be9c`;
`inspect-apis.py` ve `api-inspection.json`. Bu bir training veya pytest koşusu değildir.

## 2. Mevcut eğitim mimarisi

### Gerçek API ve ownership

`hyso/models/bcam.py:EBO` bir nn.Module wrapper, `self.model` bir ELO'dur.
`callbacks(optimizer='Adam', optimizer_params=None, scheduler=None,
scheduler_params=None, model_checkpoint=False, earlystop=None, lr=0.001,
weight_decay=0.0001)` optimizer ve callback'leri oluşturur.

`fit(train_loader, val_loader=None, epochs=10, lr=0.001,
optimizer_name='adamw', weight_decay=0.0001, use_ema=False, use_amp=True,
grad_clip=None, accum_steps=1, scheduler_name=None, scheduler_params=None,
max_lr=None, *, resume_from=None, save_last_to=None)` asıl lifecycle sahibidir.

| Konu | Şu anki gerçek davranış |
|---|---|
| Model/device | EBO device seçer, child model'i taşır; optimizer child parametreleriyle oluşturulur |
| Optimizer | callbacks veya fit; Adam/AdamW/SGD. Fit mevcut self.optimizer'ı kullanır |
| Loss | num_classes==1: BCEWithLogitsLoss; aksi halde CrossEntropyLoss |
| Scheduler | Aktif self.lr_scheduler.scheduler, fit tek stepping sahibi; eski scheduler callback'leri genel callback döngüsünde ilerletilmez |
| OneCycle | ceil(len(loader)/accum_steps) × total epochs planı; max_lr gerekli |
| Accumulation | loss × gerçek batch_size backward; window sonu unscale, grad / window_samples, clipping, step/update, zero_grad; eksik tail flush edilir |
| AMP | CUDA üzerinde torch.amp.autocast + torch.cuda.amp.GradScaler; CPU'da disabled scaler |
| Applied update | OneCycle için post-step hook; fused skip durumunu scale backoff ile ayırır. Scaler.step dönüş değeri ölçüt değildir |
| EMA | WithSE.EMA(self.model); fit içinde her microbatch sonrasında update çağrılır; bazı EMA hataları bastırılır |
| Validation | self.model.eval(), no_grad; EMA varsa shadow ile inference, hata olursa live modele fallback; train/val metrikleri ayrı |
| Epoch scheduler | StepLR/Cosine bir kez epoch sonunda; Plateau yalnız mevcut finite validation loss ile; OneCycle bu yoldan ilerletilmez |
| Callback | EarlyStopping(val_loss,self.model), ModelCheckpoint(epoch,self.model,val_loss); scheduler ayrıca yönetilir; bazı callback hataları bastırılır |
| Last checkpoint | Tamamlanan epoch sonunda self wrapper + optimizer/scheduler/scaler/EMA/history/RNG, metrics={'ebo_fit':plan} |

Bu farklı ownership/cadence ayrıntıları yeni Trainer'a kör kopyalanmayacak.
Özellikle mevcut EMA microbatch cadence'i, EMA validation + live-model best
snapshot seçimi ve try/except fallback'ler mevcut legacy davranışıdır;
M2 tasarımında bunlar açık ve ayrı politikalarla tanımlanır, legacy değiştirilmez.

`WithSE.CEMO.fit` kendi Adam/SGD/EMA eğitim döngüsüdür;
`fit.CNNSENormalWithFit` CEMO subclass'ıdır ve ayrı fit/callback/device politikası
taşır. Bunlarda EBO'nun AMP/accumulation/full-state epoch-resume API'si yoktur.
`ModularCNN.forward_features/forward` standalone interface'tir; fit/optimizer/
callback/AMP sistemi yoktur. Trainer bunu veya ELO/standart nn.Module'ü doğrudan
kullanmalı; EBO'ya takma, subclass yapma veya mevcut model.fit çağırma yok.

### Mevcut callback ve veri sözleşmesi

- `EarlyStopping.__call__(val_loss,model)` improvement'ta **deepcopy(state_dict)**
  alır; patience/counter/early_stop tutar. `restore_weights(model)` ayrı çağrıdır.
  State_dict/load_state_dict API'si yoktur.
- `ModelCheckpoint.__call__` best_loss tutar, yalnız raw model.state_dict yazar;
  full-state/resume dosyası değildir. `LRSchedulerCallback.step` yalnız wrapper'dır.
- `augmentation.build_transforms` basic/advanced zincirler; `get_loaders`
  validation için ayrı `_validation_transform` kurar. Random split sonrası val
  Subset ayrı ImageFolder/validation transform'u kullanır; class mapping train
  sözlüğüne remap edilir. Trainer dataset/transform yaratmayacak veya değiştirmeyecek.
- `EMA` nn.Module değildir; ema_model deepcopy, parameter decay ve **buffer'ları
  adla tam kopyalama** yapar. Parameter eşleşmesi zip sırasına dayanır; model
  topology'sinin optimizer/EMA kurulduktan sonra değişmemesi gerekir.

## 3. Güvenle yeniden kullanılabilecek parçalar

| Parça | Yeniden kullanım kararı |
|---|---|
| checkpoint.save_training_checkpoint/load_training_checkpoint | Doğrudan mevcut public API; atomik save, weights-only, tip/şema/optimizer-name denetimi ve restore rollback korunur |
| WithSE.EMA | Aynı nesne; yalnız model kopyalanır, Trainer kopyalanmaz. Device/name/schema doğrulaması yeni Trainer sınırında yapılır |
| callbacks.EarlyStopping | Doğrudan best snapshot/counter algoritması; B3 orchestration ve B4 typed serialization adapter'ı yeni training kodunda |
| augmentation.get_loaders/build_transforms | Caller kullanabilir; Trainer bu hazır loader'ları alır, validation transform'larını yeniden yazmaz |
| P0 numerik regression referansları | Accumulation, scheduler, EMA, snapshot ve RNG test fikirleri yeni Trainer testlerine taşınır; eski assertion'lar korunur |
| bcam._resume_loader_plan | B4'te bounded data descriptor için dar internal delegation mümkün; _resume_dataset_plan/_resume_transform_plan'a zaten delegasyon yapar |
| checkpoint._scheduler_state/_rng_state/_restore_rng | Gerekli canonical OneCycle planı ve dış restore transaction için sınırlı internal kullanım; private coupling açıkça test edilir, yeni public API sayılmaz |

**EBO.fit reusable generic engine değildir.** `_resume_plan` exact EBO/ELO type
kontrolü, EBO özel history/plan ve repr bağı taşır. Onu çağırıp type gate'i
gevşetmek çözüm değildir. İlk implementasyon bağımsız, küçük bir lifecycle
olacak; mevcut döngüyü ortak engine'e çıkarma/refactor yok. Matematik sözleşmesi
aynı reference testleriyle denetlenir; 800 satırlık fit kopyalanmaz.

## 4. Riskli bağımlılıklar ve açık sınır

İlk hedef, **sınıflandırma sözleşmesine uyan standart nn.Module'ler**; tek tensor
giriş ve tensor logits. Bu, her arbitrary output/closure/custom task'ın otomatik
destekleneceği anlamına gelmez. Detection tuple/dict batch, segmentation mask ve
pixel-weighted loss sonraki M6 adapter'ıdır. Mimari tipi EBO olmamalı diye kısıt
yok; Trainer model.forward'unu kullanır, model.fit çağırmaz.

Private bcam data helper'ları plotting/sklearn ve legacy modüle import bağı
oluşturur. B4 adapter'ı bu bağı bir dosyada tutmalı, aynı positive/negative
data-plan fixture'larıyla denetlemeli. Güvenlik helper'larını sessizce kopyalayıp
farklılaştırma veya legacy'den taşımak bu kapsamda yoktur.

Arbitrary optimizer/criterion/callback/model'in serialized state dışındaki
Python/harici state'i için resume garantisi yoktur. Stateless sınıflandırma
eğitimi ile desteklenen exact epoch-resume profili iki ayrı kapıdır.

## 5. Önerilen Generic Trainer API

**Henüz mevcut olmayan API önerisi:** yeni namespace `hyso.training`.

```python
# M2-B için öneri; bu kod şu anda çalıştırılabilir bir HYSo API'si değildir.
import torch
from hyso.models.modular import ModularCNN
from hyso.training import Trainer  # Henüz implement edilmedi.

model = ModularCNN(num_classes=3, widths=(4, 6), repeats=(1, 1),
                   strides=(1, 2), attention="eca").to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
trainer = Trainer(
    model, optimizer=optimizer,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
    device="cuda", task="multiclass",
    scheduler=None, amp=True, accum_steps=2, grad_clip=0.5,
    ema_decay=0.999, early_stopping=None, callbacks=(),
    model_config={"input_channels": 3, "num_classes": 3,
                  "widths": [4, 6], "repeats": [1, 1], "strides": [1, 2],
                  "attention": "eca", "norm": "batch"},
)
history = trainer.fit(train_loader, val_loader=val_loader, epochs=5,
                      save_last_to="last.pth", resume_from=None)
```

Sözleşme:

```text
Trainer(model, *, optimizer, loss_fn, device='cpu', task='multiclass',
        scheduler=None, amp=False, accum_steps=1, grad_clip=None,
        ema_decay=None, early_stopping=None, callbacks=(), model_config=None)
fit(train_loader, val_loader=None, *, epochs, save_last_to=None, resume_from=None)
```

- Model önce caller tarafından device'a taşınır, **optimizer bundan sonra**
  kurulur. Trainer model'i sessizce taşıyıp parametre identity'sini riske atmaz;
  parametre/buffer cihazı, scheduler.optimizer identity'si ve optimizer'ın
  modeldeki trainable parametreleri tekil kapsaması doğrulanır. Frozen model
  parametrelerinin optimizer'da bulunması mümkündür; dış parametre/duplicate yok.
- `device` explicit, default cpu; `amp=False`, accum_steps=1, grad_clip=None,
  ema_decay=None, early_stopping=None, callbacks=(), model_config=None varsayılan.
  CUDA istenip yoksa fail; amp=True CPU'da sessizce farklı precision'a dönüştürülmez,
  açık hata. B2 AMP kapsamı CUDA FP16; CPU FP32/FP64 ve disabled scaler.
- `loss_fn` caller-created. B1'de multiclass: (B,C) logits/(B,) long hard labels,
  unweighted mean CrossEntropyLoss, ignored target yok. Binary: (B,) veya (B,1)
  logits/aynı B hedef, unweighted mean BCEWithLogitsLoss; kontrollü flatten ve
  float hedef. Task explicit `multiclass/binary`; model.num_classes okunmaz.
- Accumulation'ın `loss×B` kuralı **per-sample aritmetik mean** gerektirir.
  Weighted/ignored-label CE'nin mean denominator'ı B olmayabilir; böyle loss,
  reduction='sum'/'none', soft labels/Mixup/CutMix B1'de desteklenmiş sayılmaz.
  Yeni sum/denominator sözleşmesi olmadan sessiz genelleştirme yapılmaz.
- B1/B2 kalite profili standard exact SGD/Adam/AdamW; fused/custom optimizer
  state/overflow davranışı test edilmeden destek iddiasına eklenmez. Kullanılmayan
  parametrede grad=None geçerli olabilir; bütün arbitrary model parametrelerine
  zorunlu nonzero gradient şartı konmaz. Var olan gradient'ler denetlenir.
- `train_loader` finite/nonempty ve len() sahibi, (inputs,targets) tensor batch'leri
  verir. Validation optional, nonempty. Genel eğitimde çok worker kullanılabilir;
  bunun güvenli exact resume olduğu varsayılmaz. Trainer loader yaratmaz.
- `epochs` total özgün bütçe; resume'de kalan epoch sayısı değildir. Completed
  bütçe no-op; fit aynı lifecycle'ı gelişigüzel baştan başlatmaz. Yeni eğitim
  veya best-weight finalization sonrası devam için fresh Trainer/explicit last
  restore gerekir; eski optimizer ile sessizce best weights üzerinden devam yok.
- epochs/accum pozitif Python int; amp/restore flag bool; grad_clip varsa pozitif
  finite sayı; ema_decay varsa finite [0,1] (None=kapalı, 0 geçerli/açık).
  EarlyStopping patience pozitif int, delta negatif olmayan finite sayı olmalı.
- History düz dict: train_loss/train_acc/val_loss/val_acc; acc yüzde [0,100].
  Her completed epoch bir satır; validation yoksa val alanı None, fake zero değil.
  Progress'te microbatch/attempt/applied-update sayaçları ayrı tutulur.
- `model_config` normal eğitimde optional; B4 exact resume için doğrulanmış
  descriptor gereklidir. No optimizer factory, model registry, YAML veya TrainerConfig sınıfı.

## 6. Scheduler, AMP ve accumulation kararları

**Tek owner Trainer'dır.** Scheduler bare PyTorch instance olarak verilir;
LRSchedulerCallback callback listesinde kabul edilmez. İlk kapsam OneCycleLR,
StepLR, CosineAnnealingLR, ReduceLROnPlateau. Custom scheduler için otomatik
cadence tahmini yok. OneCycle update bazlı; StepLR/Cosine completed epoch bazlı;
Plateau completed validation epoch'ta finite val_loss ile bir kez. Plateau
ve EarlyStopping validation olmadan constructor/fit preflight'ta reddedilir.

OneCycle caller tarafından optimizer ile yaratılır:

```text
L = len(train_loader) > 0
A = accum_steps > 0, bool değil Python int
U_epoch = ceil(L/A) = (L + A - 1)//A
total_steps = epochs * U_epoch
```

Fit preflight OneCycle.total_steps'i bu özgün bütçeyle eşleştirir; fresh
scheduler last_epoch=0, resume counter state'ten gelir. Overflow planlanan
üst update bütçesini büyütmez; gerçek applied update ve OneCycle sayaçları
total_steps'e ulaşmadan eğitim bitebilir. Resume'de budget/accum/L değiştirilmez.

Her accumulation penceresi:

1. zero_grad; her microbatch için autocast, mean loss×batch_size scaled backward.
2. A batch dolunca veya epoch tail'de **bir kez unscale**; gerçek toplam sample
   sayısına grad böl; ardından clipping. Unequal microbatch ve A>L aynı kurala uyar.
3. Gerçek scaler.step; optimizer post-step hook'u finally içinde kaldırılır;
   scale_before/after ile backoff kontrolü. Scaler.step'in None dönüşüne bakılmaz.
4. scaler.update her denemede; uygulanan update varsa global update_count,
   OneCycle ve **yeni Trainer EMA** bir kez ilerler; sonra zero_grad/tail reset.

AMP overflow'da modelin optimizer parametre update'i, optimizer update sayacı,
update scheduler ve yeni EMA update sayacı ilerlemez. Scaler backoff/growth
state'i, tüketilmiş microbatch/data/RNG ve forward BN buffer'ları ilerleyebilir;
"bütün state değişmez" denmez. Fused optimizer hook'u çağrıldığı halde kernel
skip edebilir; hook tek başına yeterli değil. CPU/CUDA gerçek overflow testleri
ile karar gözlenir; initial non-fused profil dışına destek kanıtsız genişletilmez.

Epoch scheduler, completed epoch semantiğini korur; o epoch'ta bütün updates
skip olsa da bir epoch ilerler. Bu, applied-update sayacından farklıdır ve
all-skipped testiyle belgelenir. Scheduler hatası bastırılmaz/tekrar denenmez;
double-step yok. Loss/metric nonfinite veya invalid config fail-fast'tır.

## 7. Validation, EMA, EarlyStopping ve callback lifecycle

Validation ayrı no_grad fazı; training optimizer/scheduler/EMA update'i yok.
Live veya EMA değerlendirme hedefinin **her alt modülünün önceki training flag'i**
kaydedilip finally'da geri kurulur; yalnız root.train(old_flag) ile mixed/frozen
mode kaybedilmez. Metric hesaplaması saf ve sample ağırlıklıdır. Loss history
ile accuracy aynı evaluation model'inden gelir. Gizli torchsummary, örnek batch
okuma veya otomatik plotting yok; model/loader/RNG sunum için tüketilmez.

Yeni Trainer EMA politikası applied optimizer update başınadır; **legacy EBO'nun
microbatch EMA cadence'i değiştirilmez**. EMA shadow constructor'da aynı modelden
kurulur, eval/no-grad; parameter schema/name sırası doğrulanır, running/int
buffer'lar existing EMA ile kopyalanır. Exception fallback ile live modele
sessiz geçiş yapılmaz; hata görünürdür.

EMA açıksa validation/early-stopping target EMA shadow, değilse live model.
Existing EarlyStopping aynı target üzerinde çağrılır; snapshot deepcopy ve
buffer'ları içerir. Böylece EMA loss'a göre live ağırlık seçilmez. Best model
seçimi, continuing training state'ten ayrılır:

```text
train -> validation -> epoch scheduler -> early-stop karar/snapshot
-> observer epoch-end -> coherent full last checkpoint
-> stop/finalization -> gerekirse seçilen eval target'a best weights restore
```

Last dosyası best-weight restore'dan **önce** coherent model/optimizer/EMA
devam noktasını kaydeder. Restore edilen best weights optimizer momentlerini
geri sarmaz; bu finalization'dır. Trainer bunun ardından normal fit devamını
reddeder; fresh last restore ile devam edilir. EMA varsa best shadow inference
hedefidir, live model ayrı kalır; sonuç/history bu seçimi açıkça belirtir.
Checkpoint early_stop=True taşıyorsa resume terminal kararı korur: yeni iterator/
optimizer adımı yok, history korunur ve aynı best-target finalization uygulanır.
EarlyStopping'i sıfırlayıp sessizce planın kalanına devam etmek resume sayılmaz.

B3 callbacks için tek küçük observer arayüzü: `callback(event, logs)`;
train_begin/epoch_end/train_end olayları, detached plain log snapshot'ları.
Optimizer/scheduler stepping veya loss mutation callback görevi değildir.
Legacy callback nesneleri otomatik tahmin edilerek çağrılmaz; EarlyStopping ayrı
argüman ve typed adapter'dır. Callback hataları görünürdür, sessiz retry yoktur.
B4 ilk exact-resume kapsamı callbacks=() ister; custom stateful/observer callback
replay ve dış I/O transaction garantisi henüz yoktur. Internal history/logging sürer.

## 8. Checkpoint ve epoch-resume güvenliği

### Format v1 ile mümkün olan ve olmayan

`checkpoint._FIELDS` exact set'tir; yeni top-level trainer alanı eklemek v1'i
bozar. **Yeni FORMAT_VERSION veya top-level alan gerekmiyor.** Mevcut history ve
metrics altında ayrı bir Trainer namespace önerilir:

```text
metrics = {
  'generic_trainer': {
    'schema_version': 1,  # yalnız bu metadata namespace'inin sürümü
    'plan': {... plain finite scalar/list/dict ...},
    'progress': {'attempts': ..., 'updates': ..., 'ema_updates': ...,
                 'model_modes': {...}, 'ema_modes': {...}},
    'early_stopping': None veya doğrulanmış config/counter/best snapshot state
  }
}
```

Namespace **önerilen yeni semantiktir, mevcut API değildir**. Plan JSON/plain
normalizasyonundan geçer; early-stop best_state tensorları plan içine değil,
metrics namespace'inin typed state bölümüne girer. Generic serializer _safe
bu plain/tensor yapıları destekler; Python callback/function pickle edilmez.
Best-loss başlangıç infinity'si için metadata'da None kullanılır; runtime adapter
initial state'e çevirir. Format v1 global şeması/weights_only davranışı aynı kalır.

Eski EBO `metrics={'ebo_fit':plan}` ve `_validate_resume_progress` aynen kalır;
yeni namespace EBO resume'ye migration değildir. Namespace'siz component dosyası
generic load ile kullanılabilir; **Trainer epoch-resume** diye kabul edilmez.

### B4 plan ve kapsam

Plan: toplam epochs, task/loss type+settings, model class/config/structure/code
fingerprint, optimizer type/param-name-order/groups/settings, scheduler canonical
config/total budget, accum/clip, device/dtype/AMP/scaler policy, EMA cadence/decay,
EarlyStopping config/target ve loader/dataset/transform descriptor'ları.
OneCycle config için existing _scheduler_state adapter'ı kullanılır; bilinmeyen
şema/callable sessizce normalize edilmez.

Same-shaped stride/activation değişimini weight schema tek başına yakalayamaz.
ModularCNN descriptor'ı mevcut input_channels/num_classes/widths/repeats/strides/
attention_name/norm_name ile caller model_config'ini karşılaştırır; module tree,
conv stride/kernel/groups ve source code fingerprint ek koruma sağlar. Custom
forward/harici state'in semantiği genel introspection ile ispatlanamaz.
İlk B4 exact-resume conformance profili ModularCNN'in beş attention seçeneği ve
doğrulanmış küçük PyTorch reference modelleridir. Diğer nn.Module'ler normal
eğitilebilir/component-load yapılabilir; exact epoch-resume için ayrıca descriptor
ve regression kanıtı gerekir. Global model registry veya arbitrary import yoktur.

Data plan için existing bounded `_resume_loader_plan` internal delegation:
TensorDataset CPU dense tensor hash'leri; PIL ImageFolder/Subset, ordered indices,
class mapping, root/path/size/mtime manifest; standart DataLoader/collate/sampler,
num_workers=0, explicit generator/replacement/custom sampler yok, nonempty loader.
Whitelisted torchvision v1 stateless transform config'i ve runtime sürümleri
eşleşir. Advanced augmentation'ın tümü resume destekli değildir. Manifest
görüntü içerik hash'i değildir; aynı size/mtime ile değişen içerik saptanmayabilir.

get_loaders random split'i yeniden oluşturmak aynı Subset'i garanti etmez;
resume caller'ı özgün ordered indices'i korumalı. Train/val planları ayrı
doğrulanır. Custom dataset/transform, multi-worker, random source/external state,
mid-epoch ve cross-device eşlik ilk B4 kapsamı değildir; normal fit'te kabul
edilmeleri resume garantisi sağlamaz.

### Restore transaction ve RNG

1. Caller model/optimizer/scheduler/EMA/loader'ları kurar; config doğrulanır.
   RNG tüketebilen constructor'lar restore'dan **önce** tamamlanır.
2. Yeni resume adapter'ı beklenen plan/history/progress/EarlyStopping state'ini
   staging nesnelerinde doğrular. `load_training_checkpoint(validate_metadata=...)`
   model/optimizer apply öncesinde namespace/plan uyuşmazlığını reddeder.
3. Existing loader component/EMA/RNG snapshot+rollback ve map_location='cpu',
   weights_only=True güvenli yolunu kullanır. RNG Python/NumPy/Torch CPU ve
   initialize edilmiş CUDA'yı kapsar; ilk yeni iterator restore'dan sonra açılır.
4. Generic loader Trainer counter/history/callback/mode state'ini bilmez.
   B4 dış transaction adapter'ı bunları ve gerektiğinde component/native
   scheduler/EMA/RNG snapshot'larını korur; staged metadata commit'inde sonradan
   enjekte edilen hatada tüm eski state geri alınır. Post-load user callback/I/O
   transaction içinde çalıştırılmaz. Rollback'in kendisi başarısızsa fatal
   açıklayıcı hata; "eski state kesin sağlam" diye sessiz devam edilmez.

Progress denetimi: completed_epochs<=total, history uzunlukları, attempts=
completed_epochs×ceil(L/A), updates<=attempts; OneCycle counter=applied updates,
EMA counter=applied updates, epoch scheduler counter=completed epochs. Parametre
başına optimizer step'i unused gradients nedeniyle global count'tan farklı
olabilir; arbitrary modelde hepsine eşitlik dayatılmaz.
Mode haritaları `named_modules` anahtarlarıyla tam eşleşen bool değerlerdir.
Başarılı restore commit'inde validation/user hook çalıştırmadan uygulanır;
başarısız restore'da önceki mode değerleri korunur. Tamamlanmış/terminal
resume çağrısında da bu mode sözleşmesi geçerlidir.

Stateful custom callback ilk B4'te reddedilir. Existing exact EarlyStopping
adapter'ı patience/delta/restore flag, counter/early_stop/best_loss/best_state
ve seçilen eval target'ın schema'sını typed olarak serialize/validate eder.
Generic loader bunu kendiliğinden restore etmez; yeni adapter'ın transaction
testi zorunludur. Bu iş bitmeden "tam Trainer resume destekli" ilan edilemez.

## 9. Legacy compatibility planı

Yeni namespace/model-agnostic lifecycle, mevcut EBO/ELO/CEMO/CNNS/EMA API'sini
genişletmez/yeniden tanımlamaz. hyso.models.__all__, legacy module/class adları,
state key/order, format v1, EBO exact type/plan gate'i ve M1 implementasyonları
aynen kalır. EBO'ya yeni Trainer delegasyonu yaptırılmaz.

Gerekli yeni production dosyaları yalnız `hyso/training/...` altında. Packaging
MODULES envanteri yeni gerçek dosyalarla büyür, eski assertion'lar silinmez;
setup.cfg/pyproject.toml minimum/sürüm/dependency değişikliği gerekmez. Mevcut
packages=find: alt paketi bulur. Root hyso/__init__ ve models export değişikliği
varsayılan çözüm değildir; `from hyso.training import Trainer` yeni subpackage
arayüzüdür. Legacy callback interface'ine state_dict metodu ekleme gereği yoktur.

## 10. M2-B implementasyon alt görevleri

### B1 — Minimal supervised classification loop

- Yeni production: `hyso/training/__init__.py`, `hyso/training/trainer.py`.
  Mevcut production dosyasına dokunma; tests/test_packaging.py MODULES/import
  eklemesi gerekir. Yeni test: tests/test_trainer_classification.py.
- API: Trainer(model, optimizer, loss_fn, device, task); fit(loader,val_loader=None,
  epochs=...). İlk incrementte amp/accum>1/scheduler/EMA/resume özellikleri henüz
  varsa açık unsupported error, sessiz no-op değil. Validation argümanı B3'e
  kadar unsupported olarak belirtilir; final örnek tüm B1–B4 tamamlanmış API'dir.
- Test: küçük ModularCNN, nn.Sequential ve plain Linear; binary/multiclass,
  farklı B/son batch; true optimizer change, CPU float32/float64, explicit device,
  invalid loss reduction/label/model-output/empty loader/optimizer binding reddi.
- Kabul: bir bağımsız optimizer sahibi, finite mean-loss eğitimi ve history;
  legacy 720/1 baseline'ı etkilemeden yeni hedefli test + installed import.

### B2 — Accumulation, AMP, scheduler doğruluğu

- Production: trainer.py'de private küçük update helper'ları; ayrı engine sınıfı
  veya EBO refactor yok. Yeni test: tests/test_trainer_updates.py.
- API: accum_steps, grad_clip, amp, bare scheduler aktive edilir. Geçici B1
  unsupported kontrolleri yalnız implement edilen özellikler için kaldırılır.
- Test: unequal batch sample mean, tail, A>L, zero/bool rejection;
  unscale→normalize→clip sırası, real CPU/CUDA optimizer counts, overflow/skip,
  fused hook false-positive sınırı, OneCycle budget/exhaustion, epoch cadence.
  Plateau'nun gerçek validation cadence testi B3'te tamamlanır.
- Kabul: OneCycle yalnız gerçek update başına, sayım loss-return değerinden
  çıkmaz; native scaler backoff; clipping reference eşliği, scheduler hatası tek kez görünür.

### B3 — Validation, callbacks, EMA, EarlyStopping

- Production: trainer.py; gerekirse yalnız adapter mantığı için yeni
  `hyso/training/callbacks.py`. Legacy callbacks.py/WithSE.py aynı kalır.
  Tests: tests/test_trainer_validation.py ve tests/test_trainer_lifecycle.py.
- API: val_loader, ema_decay, early_stopping, observer callbacks aktive edilir.
- Test: doğru train/eval/mixed modes finally restore; hiçbir validation optimizer
  update'i yok; EMA applied-update formula/int buffer; independent ES snapshot,
  EMA vs live hedef; finite Plateau metric; missing val fail; callback exception
  görünürlüğü, last-before-best-finalization tasarımını B4'te bağlamak için contract.
- Kabul: reproducible metric/history, callback'ler stepping sahibi değil;
  mevcut get_loaders val transform ayrımı/class mapping regression'ı korunuyor.

### B4 — Safe full-state ve epoch-boundary resume

- Yeni production: `hyso/training/resume.py`; trainer.py orchestration, gerekirse
  yeni callbacks adapter dosyası. checkpoint.py, bcam.py ve format v1 değişmez.
  Tests: tests/test_trainer_resume.py ve tests/test_trainer_restore_transaction.py.
- API: save_last_to/resume_from, original total epochs ve verified model_config.
  Metadata namespace typed validation, stateless loader gate ve exact built-in
  EarlyStopping state adapter'ı bu increment'in parçasıdır.
- Test: interrupted/uninterrupted 5 vs 3+resume model/opt/sched/scaler/EMA/history/
  RNG/dataset-order eşliği, callback counter/best state; schema/config/strides/
  transform/indices/batch/accum/budget mismatch mutation öncesi reddi;
  component apply ve Trainer metadata commit'in geç safhasında fault injection
  ile tam rollback, atomic save failure, unsafe file reddi. Completed total no-op.
- Kabul: component round-trip'ten ayrı **tam Trainer epoch-resume** kanıtı;
  last state coherent, best snapshot finalization'dan ayrılmış, iterator RNG restore sonrası.

### B5 — CPU/GPU, packaging ve compatibility gate

- Production ekleme beklenmez; yalnız doğrulanmış kusur için minimal düzeltme.
  Yeni bağımsız tests/cuda_trainer_validation.py ve kalite raporu; packaging
  MODULES/import envanteri yeni dosyalarla tutarlı olmalı.
- Test: tam CPU collection/JUnit kimlik karşılaştırması, P0 364+M1 356
  baseline'ının korunması; native CUDA overflow/AMP ve Trainer resume; mevcut
  iki legacy + M1 CUDA script'i, yeni Trainer script'i ayrı CUDA-visible süreçte.
  Final wheel/sdist source-byte, imports ve gerçek Torch 2.2.2 installed profile.
- Kabul: baseline kimliği/sonucu korunur, yeni test sayısı collection'dan
  hesaplanır, 0 yeni skip/xfail/tolerans gevşetmesi. GPU erişimsizse PARTIAL;
  eski PASS yeni sonuç diye kullanılmaz. Benchmark/Experiment Manager dahil edilmez.

Her alt görevde kendi hedefli testleri önce çalışır; tam CPU/GPU tekrarını B5'te
toplamak esastır, yeni üretim kusuru/tutarsızlık bulunursa etkilenen kapılar öne alınır.

## 11. Önerilen test matrisi ve 15 kararın özeti

| Profil | Hedef |
|---|---|
| CPU Python 3.10.18/Torch 2.5.1 | FP32/FP64, classification/binary, reference update/loss, negative config, history/mode |
| CUDA RTX 3060/Torch 2.5.1+cu121 | Native FP16 autocast/scaler, actual overflow/skip, EMA/BN, strict/full resume/RNG |
| CPU Torch 2.2.2+cpu/vision 0.17.2+cpu/NumPy 1.26.4 | Installed-wheel Trainer import, gerçek train/StepLR/OneCycle/checkpoint/resume; yeni API'nin minimum uyumu |
| Sentetik TensorDataset + gerçek küçük PNG ImageFolder/Subset | Sampler/transform order ve config-negative; büyük dataset/model indirme yok |
| Legacy frozen export/model-state/v1 | EBO/ELO/CEMO/CNNS/EMA aynen; unsafe load/rollback güvenliği ve packaging |

Torch 2.2.2 için torch.autocast veya torch.amp.autocast ve
torch.cuda.amp.GradScaler(enabled=...) seçilir. Yeni torch.amp.GradScaler('cpu')
API'si minimum harness'e şart koşulmaz; CPU AMP ilk kapsam değildir. Mevcut
2.2.2 ortamı/dependency'leri bozulmaz. Gelecek sürümlerin tamamı doğrulanmış sayılmaz.

| Kritik soru | Tasarım cevabı |
|---|---|
| 1 / Engine reuse seviyesi | Low-level serializer/EMA/snapshot/data descriptor; EBO.fit generic engine değil |
| 2 / EBO'ya dokunmadan | Yeni hyso.training lifecycle; legacy .fit'e delegation/subclass yok |
| 3 / Scheduler sahibi | Tek Trainer; update/epoch/metric cadence ayrımı |
| 4 / Overflow state | Optimizer/applied update/OneCycle/EMA ilerlemez; scaler/data/RNG/BN değişebilir |
| 5 / OneCycle bütçe | epochs×ceil(len(loader)/accum_steps); özgün bütçe resume'de sabit |
| 6 / Validation mode | no_grad ve her submodule training flag'ini finally restore |
| 7 / EMA vs ES | EMA update cadence ayrı; ES validation target'ını snapshot alır; best finalization coherent last'tan sonra |
| 8 / V1 doğrudan? | Component serde evet; epoch-resume yalnız yeni plan/transaction katmanıyla |
| 9 / Yeni format alanı? | Top-level/version değişmez; metrics.generic_trainer altında versioned namespace gerekir |
| 10 / Model config | Explicit config + actual structural/code descriptor + schema; arbitrary semantic eşlik iddiası yok |
| 11 / Data değişimi | Bounded loader/transform/dataset planı, immutable data/indices; unsupported resume reddi |
| 12 / Restore failure | Existing component rollback + yeni Trainer-state outer transaction; rollback failure fatal |
| 13 / CPU/CUDA/AMP | Yukarıdaki gerçek native matrix; skip veya mock'u GPU PASS sayma |
| 14 / Torch 2.2.2 | Yerel API introspection doğrulandı; B5 gerçek minimum installed runtime gate gerekli |
| 15 / Detection/segmentation | M6'da batch/loss denominator/prediction/metrics adapter; M2-B yalnız classification, model core'a task kodu ekleme |

## 12. Açık riskler ve bu görevdeki doğrulama

- İleri sürüm scheduler şemaları, arbitrary custom model/optimizer/loss/callback
  state'i, external randomness ve multi-worker exact resume genel garanti değildir.
- EBO microbatch EMA ve bazı fallback davranışları yeni Trainer'ın applied-step/
  fail-fast politikasından farklıdır; karşılaştırmalı testler bu bilinçli ayrımı
  açık tutmalı, legacy davranışını yeni döngüye göre değiştirmemeli.
- Direct mean-loss accumulation weighted CE/ignored targets/segmentation için
  yanlış olabilir; ilk loss contract dar, genişletme ayrı denominator testi ister.
- Model stride/code değişimi weight-shape eşliğiyle yakalanmaz; B4 descriptor
  ve caller config'in güven sınırı kullanıcıya açıklanmalı.
- Stateful ES snapshot/metadata apply, generic loader'ın kapsamı dışındadır;
  typed adapter/outer rollback tamamlanmadan B4 GO verilemez.
- CUDA adaptive avg/max pool determinism ve GradScaler deprecation sınırlamaları
  M1-E'de belgeli; yeni Trainer bunları ortadan kaldırmış sayılmaz.

**Bu görevde pytest, CUDA script'i veya training deneyi çalıştırılmadı.** Yeni
runtime PASS iddiası yok. Yapılanlar: kaynak/AST incelemesi, mevcut 2.2.2 CPU'da
import/inspect API kontrolü ve iki belgenin sayım/ID/faz/status/link/isim doğrulaması.
Roadmap kategorileri 23/9/12/14/6/6/4/4/7/12, toplam 97 olarak statik doğrulandı:
26 Implemented, 71 Planned. Doğrulama kaydı:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2a-7465be9c\document-validation.json`.
Yalnız iki doküman eklendi; production/test/packaging/metadata farkı yok.

## 13. M2-A kararı ve sonraki adım

**M2-A DESIGN GO.** M2-B1 için model-agnostic classification loop ve koruma
sınırları yeterince belirli. Bu, B2/B3/B4 özelliklerinin implement edildiği veya
runtime gate aldığı anlamına gelmez. İlk iş yalnız yeni training namespace'i,
minimal loss/optimizer lifecycle ve küçük testler; Benchmark/Experiment Manager
veya legacy engine refactor'ı başlatılmayacak.

Değişen dosyalar: `docs/roadmap/hyso-architecture-roadmap-v1.md` ve bu belge.
EBO/ELO/CEMO/CNNS/EMA, exports, legacy state, v1, M1, setup/dependencies korundu.
Commit/push/branch/merge/rebase/reset, version bump, yayın, dependency veya
dataset/pretrained model indirme yapılmadı. Kapsam değişikliği Roadmap v1.1
için açık kullanıcı kararı gerektirir; bu belge yeni maddeleri otomatik eklemez.
