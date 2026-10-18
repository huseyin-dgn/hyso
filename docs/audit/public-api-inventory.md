# HYSo P0.1-A — Public API uyumluluk sözleşmesi

HEAD `b74e3e2cdec527e7bb7b078f69636a27b19923fe`, main; 18 Ekim 2026. İmzalar AST'dendir; self dahil kaynak biçimi gösterilir. Hazır torch_gpu ortamında normal hyso ve on hedef sembol import edildi. **Import başarısı tüm metotların runtime başarısı değildir.** EBO/ELO/CEMO küçük CPU constructor/state denemeleri ve sınırlı gözlemler manifesttedir. fit/callbacks eğitim akışları, CNNS constructor ve dataset akışı çalıştırılmadı.

## Tam imzalar ve tanımlı public metotlar

### build_transforms

Canonical import: `hyso.augmentation.augmentation.build_transforms`, `hyso/augmentation/augmentation.py:12`.

```python
def build_transforms(img_size=(224, 224), augment='basic', extra_ops=None, channels=3): ...
```

### get_loaders

Canonical import: `hyso.augmentation.augmentation.get_loaders`, `hyso/augmentation/augmentation.py:77`.

```python
def get_loaders(train_path, val_path=None, batch_size=32, img_size=(224, 224), augment='basic', extra_ops=None, num_workers=0, channels=3): ...
```

### EarlyStopping

Canonical import: `hyso.callbacks.callbacks.EarlyStopping`. Kaynak `hyso/callbacks/callbacks.py:3`. Kalıtım: `object`.

```python
def __init__(self, patience=7, delta=0.0, verbose=True, restore_best_weights=True): ...  # L4
def __call__(self, val_loss, model): ...  # L14
def restore_weights(self, model): ...  # L26
```

### ModelCheckpoint

Canonical import: `hyso.callbacks.callbacks.ModelCheckpoint`. Kaynak `hyso/callbacks/callbacks.py:31`. Kalıtım: `object`.

```python
def __init__(self, filepath='best_model.pth', verbose=True): ...  # L32
def __call__(self, epoch, model, val_loss): ...  # L37
```

### LRSchedulerCallback

Canonical import: `hyso.callbacks.callbacks.LRSchedulerCallback`. Kaynak `hyso/callbacks/callbacks.py:45`. Kalıtım: `object`.

```python
def __init__(self, scheduler): ...  # L46
def step(self, val_loss=None): ...  # L49
```

### ELO

Canonical import: `hyso.models.bcam.ELO`. Kaynak `hyso/models/bcam.py:137`. Kalıtım: `nn.Module`.

```python
def __init__(self, input_channels=3, num_classes=10, conv_channels=[64, 128, 256, 512], residual_per_block=2, p_drop=0.0, att_reduction=16, norm_type='auto', use_depthwise=False): ...  # L138
def forward(self, x): ...  # L171
```

### EBO

Canonical import: `hyso.models.bcam.EBO`. Kaynak `hyso/models/bcam.py:184`. Kalıtım: `nn.Module`.

```python
def __init__(self, input_channels=3, num_classes=10, conv_channels=[32, 64, 128, 256], residual_per_block=2, p_drop=0.0, att_reduction=16, norm_type='auto', device=None, use_depthwise=False): ...  # L185
def callbacks(self, optimizer='Adam', optimizer_params=None, scheduler=None, scheduler_params=None, model_checkpoint=False, earlystop=None, lr=0.001, weight_decay=0.0001): ...  # L210
def fit(self, train_loader, val_loader=None, epochs=10, lr=0.001, optimizer_name='adamw', weight_decay=0.0001, use_ema=False, use_amp=True, grad_clip=None, accum_steps=1, scheduler_name=None, scheduler_params=None, max_lr=None): ...  # L270
```

### CNNSENormalWithFit

Canonical import: `hyso.models.fit.CNNSENormalWithFit`. Kaynak `hyso/models/fit.py:32`. Kalıtım: `CEMO`.

```python
def __init__(self, *args, **kwargs): ...  # L33
def callbacks(self, optimizer='Adam', optimizer_params=None, scheduler=None, scheduler_params=None, model_checkpoint=False, earlystop=None, lr=0.001, weight_decay=0.0001): ...  # L46
def fit(self, train_loader, val_loader=None, epochs=10, lr=0.001, use_ema=False): ...  # L93
```

### EMA

Canonical import: `hyso.models.WithSE.EMA`. Kaynak `hyso/models/WithSE.py:20`. Kalıtım: `object`.

```python
def __init__(self, model, decay=0.999, device=None): ...  # L21
def to(self, device): ...  # L31
def update(self, model): ...  # L38
```

### CEMO

Canonical import: `hyso.models.WithSE.CEMO`. Kaynak `hyso/models/WithSE.py:128`. Kalıtım: `nn.Module`.

```python
def __init__(self, input_channels: int=3, num_classes: int=10, conv_channels=[64, 128, 256], residuals_per_stage: int=2, se_reduction: int=8, p_drop: float=0.1, dropout_fc: float=0.3, use_ema: bool=False, ema_decay: float=0.999, device: str=None): ...  # L129
def forward(self, x): ...  # L189
def callbacks(self, optimizer='Adam', lr=0.001, weight_decay=0.0001, model_checkpoint=False, earlystop=None, scheduler=None, scheduler_params=None): ...  # L201
def fit(self, train_loader, val_loader=None, epochs=5, lr=0.001, weight_decay=0.0001, use_ema=False): ...  # L251
```

## Export ve alternatif import yüzeyi

- `from hyso.models import EBO, ELO, CEMO, EMA, CNNSENormalWithFit` açık re-export'tur. WithSE modülünün büyük/küçük harfi korunmalı.
- `from hyso.augmentation import build_transforms, get_loaders` açık re-export'tur.
- Callback yolu `hyso.callbacks.callbacks`'tir. `hyso.callbacks` boş __init__ nedeniyle sınıfları re-export etmez. hyso kökü yalnızca models/callbacks/augmentation alt paketlerini bağlar; sınıfları doğrudan export etmez.
- bcam.EMA, fit.CEMO/EMA ve augmentation.augmentation.CEMO kaynakta transitif alias'tır; ayrı implementasyon değildir.
- bcam.__all__ EBO/ELO; augmentation.__all__ iki fonksiyondur. WithSE.py:18 içindeki __all__ ise tanımlı olmayan CNNSENormalWithFit ve EMA'yı listeler; CEMO yoktur. Açık CEMO importu başarılıdır; wildcard import riski statiktir, çağrılmadı.

## Dönüşler ve nn.Module davranışı

| API | Dönüş / yan etki | Çalıştırılan kapsam ve kırılma riski |
|---|---|---|
| EBO | nn.Module wrapper, child model=ELO; callbacks None, fit history | Constructor/state ve child forward doğrulandı. Wrapper forward yok: ebo(x) NotImplementedError. model. prefix sözleşmesi korunmalı |
| ELO | forward → [N,num_classes] logits | İki küçük CPU eval konfigürasyonu doğrulandı; fit/callbacks yok |
| CEMO | forward → [N,num_classes] logits; callbacks None, fit history | EMA açık/kapalı küçük CPU eval doğrulandı; eğitim denenmedi |
| CNNSENormalWithFit | CEMO'yu miras alır, constructor *args/**kwargs aktarır; kendi callbacks/fit'i vardır | Yalnızca import doğrulandı; constructor'da explicit device otomatik cuda/cpu ile ezilir (fit.py:40) |
| EMA | nn.Module değil; deepcopy ema_model; to ve update None döndürür | Parametre güncellemesi ve buffer'ın değişmeden kalması gözlendi; chaining ve parent state traversal'ı yok |
| build_transforms | torchvision.transforms.Compose | basic Compose oluşturuldu; görüntü/advanced pipeline denenmedi |
| get_loaders | (train_loader,val_loader) tuple | Yalnızca import/statik; ImageFolder veya Subset, class map döndürmez |
| EarlyStopping | __call__/restore_weights None; best_loss/counter/early_stop/best_state değişir | __call__ storage alias gözlemi yapıldı; restore çalıştırılmadı; step/restore_best yok |
| ModelCheckpoint | __call__ None; daha küçük val_loss'ta verilen modelin state_dict'ini torch.save ile yazar | Kaydetme denenmedi; on_epoch_end yok; epoch/optimizer metadata yok |
| LRSchedulerCallback | step None; Plateau'ya val_loss, diğerlerine parametresiz step | Yalnızca import/statik; scheduler çalıştırılmadı |

Tüm constructor'lar Python gereği None döndürür. Üç fit implementasyonu `{'train_loss': list, 'val_loss': list, 'train_acc': list, 'val_acc': list}` döndürür; CEMO anahtar ekleme sırası farklıdır. Loss örnek sayısı ağırlıklı ortalama, accuracy yüzde 0–100; sınır koşulları risk raporundadır. Validation yoksa val listeleri boş; early stop aynı history ile erken döner. Normal epoch sınırında best ağırlık restore garantisi yok; yalnızca early-stop dalında denenir.

EBO/ELO/CEMO/CNNS nn.Module üzerinden parameters/named_parameters, buffers/named_buffers, state_dict/load_state_dict, train/eval/to ve __call__ miras alır. EBO __call__ eksik forward nedeniyle çalışmaz; fit child model kullanır. Wrapper eval/to kayıtlı child'a yayılır; fit child train/eval yaparken wrapper training bayrağı ayrı kalabilir. .to uygulamaya özgü self.device alanını güncellemez. EMA sıradan Python nesnesidir; parent to/eval/state_dict gölgeyi otomatik kapsamaz. CEMO'nun inherited stochastic_depth yardımcı metodu üst modelde değil bloklarda tanımlıdır; CNNS doğrudan CEMO forward'ını miras alır.

## Korunacak çağrı semantiği

- EBO varsayılan kanallar [32,64,128,256], ELO [64,128,256,512], CEMO [64,128,256]. residual_per_block ve residuals_per_stage farklı adlardır. Parametre sırası/default'lar kesin imzalardadır. Mutable liste default'ları mevcut kaynakta değiştirilmez.
- ELO norm_type='batch' dışında GroupNorm'a gider; stem her zaman BatchNorm'dur. use_depthwise modül/key/shape yapısını değiştirir. Output softmax/sigmoid uygulanmamış logits'tir.
- EBO callbacks optimizer adları case-insensitive Adam/SGD/AdamW; fit varsayılan AdamW, callbacks varsayılan Adam. Var olan optimizer fit argümanlarıyla yeniden kurulmaz.
- EBO callbacks scheduler adları steplr/reduceonplateau/cosineannealinglr/onecycle. onecycle burada nesne kurmaz. Fit adları onecycle/steplr/reduceonplateau/cosine; max_lr OneCycle için zorunlu. Scheduler_params doğrudan kwargs'tır.
- CEMO callbacks Adam/SGD, StepLR/ReduceOnPlateau kabul eder (lowercase karşılaştırma). Varsayılan criterion label_smoothing=0.1 CrossEntropyLoss; num_classes=1 için ayrı BCE yolu yoktur.
- CNNS callbacks case-sensitive Adam/SGD, StepLR/ReduceLROnPlateau/CosineAnnealingLR kabul eder. Kendi fit'i BCE veya CrossEntropyLoss seçer, CEMO criterion alanını kullanmaz.
- callbacks çağrısı listeyi temizlemeden ekler; idempotent değildir. Dolu optimizer_params lr/weight_decay default'larıyla merge edilmez. EBO/CNNS SGD caller dict'ine momentum ekleyebilir.
- earlystop=None veya boş dict false sayılır; callback kurulmaz. Model-level callback restore_best_weights=True ile kurulur. Checkpoint adları EBO best_bcam_model.pth, CEMO best_cemo_model.pth, CNNS best_model.pth.
- CEMO constructor use_ema ile fit use_ema ayrı ve mevcut kodda tutarsızdır: fit flag EMA yaratmaz; attr varsa update/validation flag'den bağımsızdır. EBO/CNNS fit(use_ema=True) EMA yaratabilir. Bu farklılıklar sessizce tekleştirilmemeli.
- build_transforms channels=1 grayscale ekler; channels=3 ImageNet normalize, diğer değerler doğrulanmadan tek kanallı normalization yoluna gider. Extra_ops normalization öncesi sona eklenir. get_loaders val_path yoksa %20 floor random_split kullanır; seed/stratification API'si yoktur.

Bu belge mevcut davranışın referansıdır; API'deki hatalar düzeltilmiş kabul edilmedi. İsim, import yolu, imza/default, logits, history, cihaz ve checkpoint key değişiklikleri eski çağıranları etkiler.
