# M1-A — Model Architecture Inventory & Modular Design

Tarih: 2026-10-19 (Europe/Istanbul). Bu çalışma hedefli kaynak incelemesi,
gerçek CPU import/forward ölçümü ve tasarım önerisidir; yeni mimari implementasyonu yoktur.

## 1. Git durumu ve kapsam

Branch `feature/hyso-models-next`, HEAD
`a62731fc903a559782c21fb1ddae609930569084` (`docs: complete HYSo 0.2.0 final quality gate`).
Kullanıcının belirttiği başlangıç commit'iyle eşleşiyor. Başlangıç çalışma ağacı temiz.
P0.6-E raporu commit edilmiş; metadata sürümü 0.2.0 korunuyor.

İncelenen ana dosyalar: `hyso/models/bcam.py`, `WithSE.py`, `__init__.py`;
doğrudan bağlı EMA/CNNS subclass ve resume sınırları için `fit.py` ile checkpoint
type/parameter-name kontrolleri. Mimari, frozen model-state, public API ve packaging
testlerinin ilgili beklentileri okundu. Bütün depo/release denetimi tekrarlanmadı.

## 2. Public model envanteri ve sorumluluklar

`hyso.models.__all__` sırası aynen:
`EBO, ELO, CEMO, EMA, CNNSENormalWithFit`.

| Public ad | Gerçek sınıf | Mimari/eğitim sorumluluğu |
|---|---|---|
| EBO | hyso.models.bcam.EBO | nn.Module wrapper; `self.model` bir ELO. Forward child'a delegasyon. Callback/optimizer/AMP/scheduler/EMA/accumulation ve epoch checkpoint/resume API'si burada |
| ELO | hyso.models.bcam.ELO | CBAM CNN; standalone forward. Kendi callbacks/fit veya trainer'ı yok |
| CEMO | hyso.models.WithSE.CEMO | SE CNN ve kendi callbacks/fit'i aynı sınıfta; EBO'nun training core'u kullanılmıyor |
| CNNSENormalWithFit | hyso.models.fit.CNNSENormalWithFit | CEMO subclass'ı; aynı CNN/forward, farklı callbacks/fit ve cihaz seçimi |
| EMA | hyso.models.WithSE.EMA | CNN değildir, nn.Module subclass'ı değildir; deepcopy shadow, parameter average ve buffer copy yardımcısı |

EBO ve ELO aynı constructor varsayılanlarına sahip değildir: EBO genişlikleri
`[32,64,128,256]`, standalone ELO `[64,128,256,512]`. Aynı açık mimari config'i
verilirse wrapper ile child'ın parametreleri aynıdır; default modellerin aynı
ağırlık dosyasını kullanabileceği varsayılmamalı.

### Constructor sözleşmeleri

Gerçek importtan inspect ile doğrulanan varsayılanlar:

```text
EBO(input_channels=3, num_classes=10, conv_channels=[32,64,128,256],
    residual_per_block=2, p_drop=0.0, att_reduction=16, norm_type='auto',
    device=None, use_depthwise=False)
ELO(input_channels=3, num_classes=10, conv_channels=[64,128,256,512],
    residual_per_block=2, p_drop=0.0, att_reduction=16, norm_type='auto',
    use_depthwise=False)
CEMO(input_channels=3, num_classes=10, conv_channels=[64,128,256],
     residuals_per_stage=2, se_reduction=8, p_drop=0.1, dropout_fc=0.3,
     use_ema=False, ema_decay=0.999, device=None)
CNNSENormalWithFit(*args, **kwargs)  # CEMO constructor'ına aktarır
EMA(model, decay=0.999, device=None)
```

CEMO sayısal/cihaz argümanlarında type annotation da taşır; tam inspect çıktısı
kanıt JSON'unda. EBO device=None ile CUDA availability'ye göre seçer ve child'ı
constructor'da taşır. ELO cihaz seçmez. CEMO device tercihini kaydeder, live CNN'i
constructor'da otomatik taşımaz; callbacks/fit taşır. CEMO use_ema=True shadow
yaratır. CNNS constructor CEMO sonrası cihazı availability'ye göre yeniden seçer
ve `.to()` uygular; CEMO'ya verilen device argümanıyla aynı davranış sanılmamalı.
Mevcut bu farkları M1 içinde sessizce standartlaştırma.

### Parametre sayıları

Gerçek `sum(p.numel() for p in model.parameters())`; BN buffer'ları ve ayrı EMA
shadow bu sayıya dahil değildir. Fresh use_ema=False modellerde bütün parametreler
trainable. Default = yukarıdaki mimari varsayılanları, 10 sınıf, CPU; tiny =
3 kanal, 2 sınıf, `[4,8]`, stage başına bir block, attention reduction=2,
p_drop=0, CEMO/CNNS dropout_fc=0, use_ema=False.

| Model | Default parametre | Default state entry | Tiny parametre | Tiny state entry |
|---|---:|---:|---:|---:|
| EBO | 280,190 | 141 | 635 | 45 |
| ELO | 1,103,066 | 141 | 635 | 45 |
| CEMO | 170,650 | 152 | 375 | 58 |
| CNNSENormalWithFit | 170,650 | 152 | 375 | 58 |

Bu sayılar bütün constructor kombinasyonlarının sayısı değildir; sınıf sayısı,
kanallar, norm ve depthwise seçimiyle değişir. EMA nesnesinin shadow'u parent'a
nn.Module olarak register edilmez; CEMO/EBO state_dict'inde otomatik `ema.*`
anahtarları oluşmaz. Full-state checkpoint EMA'yı ayrı component olarak kaydeder.

## 3. CNN ve attention mimarileri

### ELO ve EBO child: CBAM hattı

Stem: `Conv2d(input,C0,3,stride=1,padding=1,bias=False) → BatchNorm2d → SiLU`.
Flat `ModuleList blocks` içinde her stage için residual_per_block adet
`BottleneckResidualBlock`. Sonraki stage'in **ilk** block'u stride=2; ilk stage
stride=1. Channel değişimi veya stride/projection gerektiğinde shortcut
`1×1 Conv(stride,bias=False) → norm`; diğer durumda Identity.

Block: `1×1 Conv → norm → SiLU → 3×3 Conv → norm → SiLU → 1×1 Conv → norm
→ CBAM → stochastic_depth → +shortcut → SiLU`.
Mid kanal `max(1,int(out_channels*0.25))`; ratio block constructor'ında değişebilir,
ELO constructor'ında ayrı exposed değildir.

`use_depthwise=True` orta hattı `depthwise 3×3(groups=mid,stride) → norm → SiLU
→ pointwise 1×1(mid→mid) → norm` ile değiştirir; conv3'e geçmeden ilave SiLU yok.
`conv2/bn2` bu modda Identity placeholder, gerçek parametreler `dw_seq.*` altında.
Depthwise default değildir.

`norm_type='batch'` residual path/shortcut'ta BatchNorm2d; **'auto' ve 'group'
GroupNorm** kullanır. Group sayısı en çok 8, kanal sayısını bölen uygun sayı;
gerekirse 1. Stem her zaman BatchNorm. Kaynakta 'batch' dışındaki başka string'ler
de group yoluna düşer; yeni bileşenlerde bu davranışı genel API diye kopyalama.
`bn1/bn2/bn3` isimleri GroupNorm içerebilir.

Head: `AdaptiveAvgPool2d(1) → flatten → LayerNorm(C_last) → Linear(C,C//2)
→ SiLU → Dropout(0.2) → Linear(C//2,num_classes)`. Sonuç logits; softmax yok.
Head dropout'u ELO p_drop argümanı kontrol etmez; p_drop residual branch içindir.

### CEMO ve CNNS: depthwise SE hattı

Stem: `3×3 Conv(stride=2,padding=1,bias=False) → BatchNorm2d → SiLU`.
`ModuleList stages` içinde her stage bir Sequential; residuals_per_stage adet
`BottleneckSEBlock` içerir. Stage'in ilk block'u **in_ch != out_ch ise** stride=2;
aynı genişlik tekrarında downsample yapılmaz. ELO ise stage indeksine göre
downsample eder; bu kurallar aynı değildir.

Block: `1×1 Conv → BN → SiLU → depthwise 3×3(groups=mid,stride) → BN → SiLU
→ pointwise 1×1(mid→out) → BN → SE → stochastic_depth/+shortcut → SiLU`.
Mid oranı 0.25, en az 1. Shortcut channel/stride değişiminde `1×1 Conv → BN`,
aksi halde Identity. BottleneckSEBlock doğrudan norm_layer ve activation class
kabul eder; CEMO constructor'ı bunları expose etmez ve default BN/SiLU kullanır.

Head: `AdaptiveAvgPool2d((1,1)) → flatten → Dropout(dropout_fc) → Linear(C,num_classes)`.
LayerNorm/iki katmanlı ELO head'i yok. CEMO criterion varsayılanı
CrossEntropyLoss(label_smoothing=0.1); bu mimari değil, training tercihidir.
CEMO fit'inde Adam/SGD callback seçenekleri ve EMA var; EBO'nun AMP/accumulation,
OneCycle ve save_last_to/resume_from kapsamı CEMO'ya ait değildir.

### Gerçek forward boyutları

Default model, eval/no_grad, giriş **(2,3,64,64)**; tabloda batch=2 kısaltıldı.
Her stage içinde ikinci block channel/spatial shape'i korur.

| Nokta | EBO child | Standalone ELO | CEMO / CNNS |
|---|---|---|---|
| Stem | 32×64×64 | 64×64×64 | 64×32×32 |
| Stage 0 | 32×64×64, blocks.0–1 | 64×64×64, blocks.0–1 | 64×32×32, stages.0 |
| Stage 1 | 64×32×32, blocks.2–3 | 128×32×32, blocks.2–3 | 128×16×16, stages.1 |
| Stage 2 | 128×16×16, blocks.4–5 | 256×16×16, blocks.4–5 | 256×8×8, stages.2 |
| Stage 3 | 256×8×8, blocks.6–7 | 512×8×8, blocks.6–7 | Yok |
| Global pool | 256×1×1 | 512×1×1 | 256×1×1 |
| Flatten / norm | 256, LayerNorm | 512, LayerNorm | 256, dropout |
| Logits | (2,10) | (2,10) | (2,10) |

Tiny giriş (2,3,16,16): EBO/ELO `4×16×16 → 4×16×16 → 8×8×8 → 8×1×1 → (2,2)`;
CEMO/CNNS `4×8×8 → 4×8×8 → 8×4×4 → 8×1×1 → (2,2)`.
3×3 padding=1 stride=2 tek boyutlarda ceil(H/2), ceil(W/2) verir. Geçerli config
ve pozitif spatial boyutlarda adaptive pooling sabit görüntü boyutu zorlamaz;
training BN'nin tek değerli batch/spatial sınırları ayrıca dikkate alınmalı.

### SE ve CBAM bağımsızlığı

| Özellik | Mevcut SEBlock | Mevcut CBAMBlock |
|---|---|---|
| Doğrudan import | hyso.models.WithSE.SEBlock | hyso.models.bcam.CBAMBlock |
| Constructor | channels, reduction=8 | channels, reduction=16, kernel_size=7 |
| Channel hattı | H/W mean → Linear(C,r) → SiLU → Linear(r,C) → sigmoid | AdaptiveAvg ve AdaptiveMax → **paylaşılan** iki Linear/SiLU MLP → toplama → sigmoid |
| Spatial attention | Yok | Channel attention sonrası channel mean/max → concat(2 kanal) → Conv2d(2,1,k,pad=k//2) → sigmoid |
| Uygulama | x × kanal kapısı | x × kanal kapısı, sonra × spatial kapı |
| Shape | N,C,H,W → aynı | N,C,H,W → aynı; shape koruyan pozitif tek k, default 7 |

İkisi de normal nn.Module, model instance'ına/callback'e bağlı değiller; ayrı bir
CNN feature map'ine doğrudan takılabilirler. Ancak `hyso.models` public export
listesinde değiller; model paketinin eager import'u ağır plotting/sklearn gibi
mevcut bağımlılıkları da yükler. Bağımsız tensor kullanımı, hafif dependency-isolated
import API'si anlamına gelmez.

Kanal sayısı constructor C ile eşleşmeli; rank=4 NCHW beklenir, NHWC/generic
sequence kabulü yok. Reduced kanal `max(1,C//reduction)`; C<reduction çalışır.
Reduction pozitif olmalı; tam bir input/config validation katmanı mevcut değil.
Standart device/dtype `.to()` yönetimi caller'a aittir.

Bağımsız gerçek testte ikisi C=1 ve C=8, H=11/W=13 ile shape korudu, finite
forward ve input/parameter backward verdi. Bu, tüm dtype/kernel/config matrisi
test edildi demek değildir. SE yalnız squeeze channel; CBAM buna spatial gating
ekler. SiLU ve bias'lar mevcut kodun tercihidir; makaledeki her ayrıntının birebir
uygulandığı varsayılmamalı.

## 4. Yapısal sınırlamalar ve tekrarlar

Attention sınıfları yeniden kullanılabilir, fakat legacy residual constructor'ları
`CBAMBlock`/`SEBlock` yaratmayı sabitliyor; attention factory/none seçeneği yok.
Üst modelde norm/activation/block/head seçimleri de kısmen sabit. Block/stage
adları ve görevleri farklı; yeni bir ortak container'a taşımak ilk adım olmamalı.

Tekrar eden yapılar: 1×1–3×3–1×1 residual hatları, projection shortcut, channel MLP
gating, stage döngüsü, global pooling; callbacks/fit sorumlulukları da üç yerde
ayrı. **Benzer görünen stochastic depth aynı algoritma değil:**

- ELO: her block aynı p_drop; batch içindeki her örnek için feature cihazında
  maske, tutulan main branch 1/(1-p) ile ölçeklenir; eval'da drop yok.
- CEMO: block_id ile 0'dan p_drop'a artan olasılık; block çağrısında tek CPU
  `torch.rand(1).item()` kararı bütün batch main branch'ini düşürür; keep rescale
  yok. Conv/BN/SE bu karardan önce çalışır; branch drop hesaplamayı atlamaz.

Bunları tek helper'a taşımak RNG tüketimini, numeriği ve resume davranışını
değiştirir. Bu gözlem refactor sınırıdır; M1-A training bug düzeltmesi önermez.
Default mutable list argümanları ve farklı cihaz politikaları da mevcut API'nin
parçası; yeni sınıflarda tuple/explicit device yönetimi seçilebilir, eskiler değiştirilmez.

## 5. State_dict, API ve checkpoint uyumluluk riskleri

| İşlem | Somut kırılma |
|---|---|
| EBO.model'ı kaldırmak/yeniden adlandırmak | model.stem.0.weight → stem.0.weight; wrapper/child strict load uyuşmaz |
| Flat blocks'ı nested stages'a çevirmek | blocks.2.conv1.weight → stages.1.0.conv1.weight |
| Legacy Conv/BN'yi yeni ConvNormAct içine almak | stem.0.weight veya blocks.0.conv1.weight yeni alt prefix alır |
| BN↔GN, group/width/bottleneck değişimi | Running buffers eklenir/silinir; shape/işlev değişir, affine shape aynı olsa bile davranış farklı |
| Standard↔depthwise varsayılanını değiştirmek | conv2.weight yerine dw_seq.0.weight/dw_seq.3.weight; shapes farklı |
| SE/CBAM veya head'i değiştirmek | attention.fc1.* / se.fc1.* / conv_spatial.* / head.0,head.3 / fc.* uyuşmaz |
| Stateless Identity/activation ekleyip sırayı değiştirmek | Key'ler bazen aynı kalabilir; forward/numerik, Sequential indeksleri veya repr planı değişebilir |
| Sınıfı başka modülde yeniden tanımlayıp eski adla re-export etmek | Class __module__/qualname/type metadata değişebilir; alias tek başına full-state uyumluluğu sağlamaz |
| Optimizer parametre sırasını değiştirmek | Parametre-name/group ordering ve moment state eşleşmesi bozulur |

Örnek legacy anahtarlar: ELO `blocks.0.attention.fc1.weight`,
`blocks.0.attention.conv_spatial.weight`, `fc_norm.weight`, `head.3.weight`;
EBO aynılarına `model.` prefix ekler. CEMO `stages.0.0.dwconv.weight`,
`stages.0.0.se.fc1.weight`, `fc.weight`. Mevcut frozen manifest key sırası,
shape, dtype ve parameter/buffer ayrımını denetler; tarihsel ağırlık değerleri içermez.

Checkpoint format v1 component type adı, tensor şeması ve optimizer parametre
sırasını denetler. EBO `_resume_plan` ayrıca **type(model) is EBO** ve
**type(model.model) is ELO** ister; plan `repr(model.model)` ve config içerir.
Dolayısıyla key'ler aynı kalsa bile sınıf/container/stateless modül değişikliği
eski planı geçersiz kılabilir. strict=False kullanmak veya plan kontrolünü
gevşetmek çözüm değildir. Eski sınıfları **yerinde ve aynı child registration
sırasında** korumak en küçük güvenli yol.

Yeni modüler model EBO child'ına atanmayacak, EBO subclass'ıyla bu gate aşılmaya
çalışılmayacak. Generic `save_training_checkpoint/load_training_checkpoint` caller
oluşturduğu yeni nn.Module/optimizer için ayrı round-trip test edilebilir;
bu, yeni modelin EBO epoch-resume desteği olduğu anlamına gelmez. Yeni trainer
veya generic epoch-resume tasarımı M1'in bu planında kapsam dışı.

## 6. Önerilen küçük modüler mimari

Yalnız yeni modeller için, geçici doğrudan submodule API'leriyle ilerle:

```text
hyso/models/blocks.py      ConvNormAct + tek sade ResidualBlock
hyso/models/attention.py   ECA + CoordinateAttention (M1-C'de)
hyso/models/modular.py     ModularCNN (M1-D'de)
```

Bu adlar **öneridir; dosyalar/sınıflar henüz mevcut değil**.
bcam.py, WithSE.py ve fit.py aynı kalır. `hyso.models.__all__` ve mevcut import
identity'leri değiştirilmez; yeni kullanıcı API'si yayımlamak ayrı karar.

| Parça | Şimdi gerekli mi? | En küçük tasarım |
|---|---|---|
| Reusable convolution | Evet, M1-B | Yeni ConvNormAct; explicit in/out/stride/groups ve sınırlı norm seçimi; shape/config validation |
| Residual block | Evet, bir tip | Projection + main path + attention + toplama; main output kanalıyla attention C eşleşsin. Legacy iki davranışı tekleştirme |
| Attention interface | Evet, sınıf hiyerarşisi gereksiz | `nn.Module.forward(N,C,H,W) -> aynı shape`; default nn.Identity; module constructor'da bir kez register edilir |
| Ayrı backbone sınıfı | Başlangıçta hayır | ModularCNN.features ve forward_features yeterli; classifier dışı tüketici çıkarsa ayır |
| Ayrı head framework'ü | Hayır | Pool/flatten/Linear doğrudan modelde; yeni model için tek basit logits head |
| Configuration framework | Hayır | Explicit keyword argümanları, immutable tuple widths/repeats/strides; gerektikçe küçük local factory |
| Registry/plugin/dynamic import | Hayır | Gerekirse none/se/cbam/eca/coordinate için sınırlı if/elif; YAML, runtime class-path import, global registry yok |

Yeni residual attention'ı projeksiyon/norm sonrası **out_channels** feature'ına,
shortcut toplamından önce uygular; blok dışına post-add bağlamak farklı bir mimari
olur ve açıkça ayrı deney olarak ele alınır. Yeni block'un stochastic depth'i ilk
incrementte eklenmeyebilir; eklenirse örnek bazlı maske/ölçekleme/RNG davranışı
ayrı sözleşmeyle test edilmeli. Legacy'nin davranışı değiştirilmez.

Legacy SEBlock ve CBAMBlock yeni modellerde doğrudan instance olarak kullanılabilir;
onları başka dosyaya taşımak veya aynısını yeni isimle kopyalamak gerekmiyor.
Yeni attention module instance'ları forward içinde üretilmez; train/eval/device ve
optimizer registration düzgün çalışmalı. Factory kullanılırsa her block için
ayrı instance yaratmalı; istemsiz ağırlık paylaşımı olmamalı. Config gerekiyorsa
yalnız plain scalar/list/string tutar; callable veya module checkpoint config'ine yazılmaz.

Yeni modelin ilk amacı forward/backward ve state_dict'tir; kendi fit, optimizer,
EMA, autocast veya cihaz seçme algoritması yazmayacak. EBO training core'unu
genelleştirmek bu küçük mimari çalışmasını ayrı bir trainer refactor'una dönüştürür.

## 7. ECA / Coordinate Attention entegrasyon noktaları

ECA, channel descriptor üzerinde boyut azaltmadan yerel cross-channel ilişkiyi
1D convolution ile kurar; kernel seçimi constructor'da sabit/izlenebilir olmalı.
Kaynak: [ECA-Net, özgün makale](https://arxiv.org/abs/1910.03151).
Yeni ResidualBlock'ta C_out kanallı feature için kanal kapısı üretip aynı NCHW
tensor shape'ini döndüren nn.Module olarak takılır. Pratik sözleşme
`attention(C_out): (B,C_out,H,W) -> (B,C_out,H,W)`.

Coordinate Attention iki spatial eksende ayrı 1D aggregation ile yön/konum
bilgisini koruyan iki gate üretir. Kaynak:
[Coordinate Attention, özgün makale](https://arxiv.org/abs/2103.02907).
Aynı yeni attention slot'una, spatial feature map pool edilmeden önce takılmalı;
classifier'ın global pool sonrasına koymak amaçlanan konum bilgisini kaybettirir.
H ve W sabitlenmemeli; rectangular/odd shape ve H=1/W=1 testleri gerekli.
BatchNorm kullanılacaksa batch=1/small spatial train davranışı açıkça test edilmeli.

İki öneri de **gelecek tasarımıdır**; depoda ECA/Coordinate Attention uygulanmadı.
Özgün makalelerin accuracy/efficiency sonuçları HYSo için test sonucu sayılmaz;
bu görev FLOPs, hız veya doğruluk üstünlüğü iddiası yapmaz.

## 8. M1-B/C/D/E uygulama planı

### M1-B — bağımsız temel block'lar

- Dosyalar: yeni `hyso/models/blocks.py`, `tests/test_modular_blocks.py`;
  packaging testinin MODULES listesine yalnız gerçekten eklenen dosyayı ekle.
  Legacy model dosyaları/export/manifest değiştirilmez.
- Testler: ConvNormAct standard/groups, stride=1/2, C_in≠C_out projection,
  identity shortcut, attention=Identity ve legacy SE/CBAM instance'ıyla shape/
  gradient, invalid channel/group/stride/norm girdileri, train/eval, dtype/device,
  strict round-trip. Küçük B=2, odd/rectangular input kullan.
- Risk: yeni container adlarının eski modellerde kullanılması; modülün forward
  içinde yaratılması; yanlış norm/dropout sırası; packaging'in sabit MODULES seti.
- Kabul: yeni block finite forward/backward; attention/output ve shortcut shape
  eşliği; frozen eski API/model-state testleri aynen PASS. Block içinde fit yok.

### M1-C — yeni attention modülleri

- Dosyalar: yeni `hyso/models/attention.py`, `tests/test_attention_modules.py`;
  packaging MODULES'e attention.py eklenir. Eski SE/CBAM taşınmaz/değiştirilmez.
- Sıra: ECA önce, ardından Coordinate Attention; ikisi bağımsız doğrulanır.
- Testler: C=1/C<reduction, farklı kanal/odd rectangular H/W, H=1 veya W=1,
  input/parameter finite gradients, same shape/device/dtype, eval deterministik
  davranış, yeni state strict round-trip, B block slot'uyla entegrasyon. Coordinate
  Attention BN varsa batch=1 train/eval kenarı ve norm state'i ayrıca kapsanır.
- Risk: yanlış squeeze/transpose ekseni, ECA even-kernel uzunluk kayması,
  coordinate gate broadcasting, küçük batch norm'u, forward'da yeni parametre yaratma.
- Kabul: makale mekanizmasıyla tutarlı, bağımlılıksız plain nn.Module; eski
  SE/CBAM state ve public API aynı. Accuracy/latency iyileşmesi kanıtsız duyurulmaz.

### M1-D — model composition ve küçük config

- Dosyalar: yeni `hyso/models/modular.py`, `tests/test_modular_model.py`;
  packaging MODULES güncellemesi. Kullanım gerekirse tek kısa tasarım/örnek belge.
- Model: explicit widths/repeats/strides, sınırlı attention seçimi; features,
  global pool ve linear logits head. Constructor başında config validation;
  ayrı registry/backbone/head/config framework'ü yok.
- Testler: attention=none/se/cbam/eca/coordinate ile tiny model, farklı input
  channel/class count, stage shape, parametre registration/instance bağımsızlığı,
  state key/shape sırası, optimizer update, eval round-trip, invalid config.
- Risk: EBO.model yerine yeni CNN atamak; yeni modeli eski checkpoint target'ı
  gibi sunmak; factory/module nesnesini config'e yazmak; star export'u değiştirmek.
- Kabul: yeni model standalone nn.Module ve kendi açık weight şemasına sahip;
  eski EBO/ELO/CEMO/CNNS config/outputs/state keys değişmez. EBO resume desteği
  vaat edilmez; training core ve checkpoint v1 yeniden yazılmaz.

### M1-E — mimari ve regression kapısı

- Dosyalar: yeni/ilgili modüler testler, gerekirse bağımsız
  `tests/cuda_modular_validation.py` ve mimari doğrulama raporu.
  Mevcut manifest/snapshot'lar eski modeller için aynen korunur; yeni şema
  gerekiyorsa ayrı fixture kullanılır.
- Testler: yeni model forward/backward/optimizer; generic component checkpoint
  v1 model/optimizer/scheduler/EMA round-trip ve weights_only=True; negative
  shape/config; wheel import ve modül içerikleri. Bu, yeni epoch-resume desteği değildir.
- Tam CPU suite: mevcut 364 PASS/1 SKIP test kimliği ve sonuçları korunur,
  yeni testler buna eklenir. API/packaging/model-state alt kümeleri ayrıca kimlikle
  denetlenir; yeni skip/xfail/tolerans gevşetmesi yok.
- GPU: ayrı CUDA-visible süreçlerde mevcut 7 temel + 4 pozitif + 6 negatif
  regression ve yeni block/model FP16/gradient/EMA/device testleri. Erişim yoksa
  GPU kısmı PARTIAL; eski sonucu yeni PASS diye sunma.
- Risk: yeni attention'ın dtype/norm/RNG davranışı ve eski block'lara yanlışlıkla
  bağlanması. Kabul: eski full-state/epoch-resume korunur, yeni model strict
  state round-trip ve final arşivler PASS; genel kernel determinism iddiası yok.

Her aşamada yalnız etkilenene yönelik testlerle başla; tam CPU/GPU regression'ı
M1-E'de topla. Public export, legacy class relocation ve generic trainer kararı
bu plana varsayılan olarak dahil değildir.

## 9. Bu görevdeki doğrulama sonuçları

Gerçek runtime: Windows/Python 3.10.18/Torch 2.5.1+cu121, CPU,
CUDA_VISIBLE_DEVICES=-1; envanterde gerçek checkout import provenance doğrulandı.

- 4 CNN × default/tiny = **8 gerçek forward**, finite logits ve hook shape trace PASS.
- SE/CBAM × C=1/8 = **4 bağımsız forward/backward**, finite input/parameter
  gradients ve shape/device korunumu PASS; spatial giriş 11×13.
- Hedefli pytest **66 PASS**, 5.50 s, exit=0; 41 public API + 1 published
  export + 18 model-state + 6 model forward/gradient testleri.
- Strict fresh-model load, tam eval output eşliği, wrapper `model.` prefix'i
  ve yanlış wrapper/child target reddi mevcut testlerle doğrulandı.
- CUDA initialized=False. GPU, tam 364 testlik suite, benchmark, dataset/weights
  indirme veya packaging build bu görevde yapılmadı. P0.6-E'nin 364 PASS/1 SKIP,
  46 API/packaging ve 17 CUDA senaryosu **geçmiş baseline**, yeni M1-A koşusu değil.

Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m1a-1cbfd981`:
`inventory.py`, `inventory.json`, `inventory.log`, `targeted.log`,
`targeted-junit.xml`. JSON tam constructor imzaları, parametre sayıları, state
key örnekleri ve her block/stage tensor boyutlarını içerir. Geçici script yeni
production bileşeni tanımlamaz; yalnız gerçek modelleri ölçer.

## 10. Değiştirilen dosyalar

Yalnız yeni `docs/audit/m1-a-model-architecture-inventory.md`.
Production/test/metadata/manifest/public API/checkpoint/training core değişmedi.
Version bump, dependency ekleme, commit/push/merge/rebase/reset/checkout,
tag/Release/upload veya kullanıcı dosyası silme yapılmadı.

## 11. Nihai öneri

**M1-A hedefli envanter/tasarım: PASS.** İlk implementasyon incrementi yalnız
iki küçük block ve onların testleri olsun. Sonra bağımsız ECA/Coordinate Attention,
ardından standalone composition ekle. Legacy EBO/ELO/CEMO sınıflarını yerinde
koru; benzer görünen training/stochastic-depth yollarını birleştirme. Yeni
modüler CNN için EBO epoch-resume desteğini varsaymadan ilerlemek mevcut
checkpoint/API sözleşmelerini koruyan en küçük güvenli yoldur.
