# Sampling, prediction head ve tensor yardımcıları

Aşama 13.5.4 ile ticari Registry 57, geliştirme Registry 85 girdiye ulaşır.
Önceki 48 ticari kayıt korunur. Yeni HYSo modülü `hyso/models/sampling.py`
public PyTorch operatörlerini birleştirir. Yeni bağımlılık eklenmez.

## Boyut değiştirme

Strided convolution mevcut `conv2d` girdisinin stride parametresiyle kullanılır.
Örneğin `(3,15,17)` girişte kernel=3, padding=1 ve stride=2 kullanan 3→8
convolution `[N,8,8,9]` üretir. Parametre sayısı stride ile değişmez.

`resize` nearest, bilinear, bicubic ve area modlarını tek kapalı sözleşmede sunar.
Giriş ve çıkış floating NCHW olur. Kanal ve batch sayısı korunur. Mode varsayılanı
nearest, output_size ve scale_factor varsayılanları None olur. Her çağrıda bu
iki hedef parametresinden tam biri verilmelidir.

```python
b.add('resize', output_size=[12,20], mode='bilinear', align_corners=False)
b.add('resize', scale_factor=[1.5,.5], mode='nearest')
```

İlk satır H/W'yi doğrudan 12/20 yapar. İkinci satırda `[N,C,5,7]` giriş
`[N,C,7,3]` çıkar. Her eksende `floor(input*scale_factor)` uygulanır.
Scale factor pozitif finite sayı veya iki sayıdan oluşan çift olmalı, her değer
8192'yi aşmamalıdır. Output size pozitif tam sayı veya iki tam sayıdır.
Her hesaplanan çıktı boyutu 1..8192 aralığında olmalıdır.

Nearest en yakın komşu seçimini, bilinear dört komşudan doğrusal ağırlıklı
örneklemeyi, bicubic kübik örneklemeyi kullanır. Bicubic girdinin değer aralığını
aşabilir. Area spatial hücre ortalamalarıyla yeniden örnekler. Bu operatörler
öğrenilen parametre taşımaz ve train/eval aynı matematiği uygular.

align_corners yalnız bilinear/bicubic için None, False veya True kabul eder.
None native False davranışını seçer. True köşe merkezlerini eşler. Nearest ve
area için parametre None kalmalıdır. Recompute-scale ve antialias seçenekleri
Registry'de sunulmaz. Native interpolate varsayılanları kullanılır.
[PyTorch interpolate](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html)

Yaygın hatalar iki hedefin birlikte verilmesi, hedefin eksik olması, sıfır/negatif
scale, bool veya non-finite değer, geçersiz mode ve nearest/area ile align_corners
seçilmesidir. Hedeflerden birini kaldırın, pozitif finite değer ve desteklenen
mode kullanın. Çok küçük ölçek sıfır çıktı üretirse scale veya input büyütülmelidir.
Non-finite JSON/config değerleri mevcut plain-data doğrulamasında belge düzeyinde
reddedilebilir. Böyle bir tanılamada layer_index None olabilir.

## PixelShuffle ve PixelUnshuffle

`pixel_shuffle` upscale_factor=r ile `[N,C,H,W]` girdisini
`[N,C/r²,H*r,W*r]` biçiminde yeniden düzenler. C, r²'ye bölünmelidir.
`pixel_unshuffle` downscale_factor=r ile bunun tersini yapar ve
`[N,C*r²,H/r,W/r]` üretir. H ve W, r'ye bölünmelidir.

```python
b = ModelBuilder((16,3,5),3).add('pixel_shuffle', upscale_factor=2)
# Feature shape [N,4,6,10]
b.add('pixel_unshuffle', downscale_factor=2)
# Feature shape [N,16,3,5]
```

Her faktör gerekli pozitif tam sayıdır ve en fazla 8192 olabilir. Çıktı C/H/W
boyutları ayrıca sınırlandırılır. İşlemler örnek başına eleman sayısını korur,
trainable parametre taşımaz ve train/eval davranışı aynıdır. Input'un PyTorch
yeniden düzenlemesiyle değer ve gradient parity testleri bulunur. Kanal veya
spatial bölünebilirlik hatalarında faktörü gerçek bölenlerden seçin.
[PixelShuffle](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html), [PixelUnshuffle](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html)

## Head seçimi ve sürümleme

`ModelBuilder((C,H,W), num_classes)` mevcut schema1 sequential_classifier yoludur.
Otomatik AdaptiveAvgPool2d(1), flatten ve `classifier` Linear katmanı aynı sırada
kalır. Schema1 JSON schema1 olarak saklanır. Eski state_dict anahtarları ve logits
sayısal davranışı regresyon testinde doğrudan native operatörlerle karşılaştırılır.

Yeni head seçimi opt-in `ModelBuilder.sequential` metoduyla yapılır.

```python
from hyso import ModelBuilder

classification = ModelBuilder.sequential((3,16,16), head='classification', num_classes=5, dropout=.2)
regression = ModelBuilder.sequential((3,16,16), head='regression', output_features=2)
embedding = ModelBuilder.sequential((3,16,16), head='embedding', embedding_dim=32, normalize=True)
```

Bu modeller schema_version=2, kind='sequential' kullanır. Input shape ve blocks
yanında ayrı `head={name,params}` kaydı bulunur. Head isimleri classification_head,
regression_head ve embedding_head olur. Params varsayılanları JSON'a yazılır.
Head türü sessizce num_classes alanına yüklenmez. Eski dosyalar otomatik migrate
edilmez. JSON yalnız kapalı, finite, plain mimari değerlerini taşır.

Üç head için pooling varsayılan 'avg' olur. 'max' ve 'none' da seçilebilir.
Avg/max global pooling NCHW'den kanal vektörü çıkarır. None, CHW'yi örnek başına
flatten eder veya terminal flatten'dan gelen NF vektörünü kullanır. Linear giriş
genişliği Builder tarafından feature shape'ten hesaplanır. Builder head params
içinde in_features kabul etmez. Registry üzerinden bağımsız head oluştururken
in_features gerekli ve explicit olur.

| Head | Gerekli çıktı parametresi | Çıktı ve seçenekler |
|---|---|---|
| Classification | num_classes | `[N,num_classes]` linear logits. Dropout varsayılan 0, aralık [0,1). |
| Regression | output_features | `[N,output_features]` sınırsız linear çıktı. Otomatik sigmoid uygulanmaz. |
| Embedding | embedding_dim | `[N,embedding_dim]` linear projection. Normalize=False ve eps=1e-12 varsayılanları. |

Tüm feature/çıktı genişlikleri 1..8192 aralığında tutulur. Exact trainable parametre
sayısı `(in_features+1)*output_features` olur. Bias dahildir. Pooling ve dropout
trainable parametre eklemez. Embedding normalization `v/max(norm(v),eps)` uygular.
Dtype altında eps'nin sıfıra yuvarlanmasını engellemek için etkili epsilon en az
torch.finfo(dtype).tiny olur. Sıfır projection güvenli sıfır embedding üretir.
Çok küçük explicit epsilon büyük gradient üretebilir. Varsayılan eps=1e-12
çoğu CPU float32/float64 kullanımında sayısal pay sağlar.

Final head blocks listesine yerleştirilemez. INVALID_BLOCK_ORDER tanılaması bu
yanlış yerleşimi bildirir. Geçersiz head türü, boyutu veya feature/pooling uyumu
INVALID_OUTPUT_HEAD üretir. Schema2'de tüm head parametreleri ayrı terminal kayıtta
saklanır. Schema2 state_dict içinde `head.linear.weight/bias` kullanılır.

## Tensor yardımcılarının desteklenen kapsamı

`permute` N ve C eksenlerini sabit tutar. Dims yalnız `[0,1,2,3]` veya
`[0,1,3,2]` olabilir. İkinci seçenek H/W'yi değiştirir. `[N,4,3,5]` girdisi
`[N,4,5,3]` çıkar. NHWC dönüşümü veya batch/kanal taşıma kabul edilmez.

`reshape` shape=[C,H,W] ile her örneğin elemanlarını explicit yeni CHW biçiminde
yorumlar. Örneğin `[N,4,3,5]` girdisi shape=[2,5,6] ile `[N,2,5,6]` olur.
Çarpımlar eşit olmalıdır. Batch ekseni shape parametresine yazılmaz. -1 inference
ve rank değiştirme bu girdide sunulmaz. Gerekirse noncontiguous tensor için
native reshape kopya oluşturabilir. Sonraki convolution yeni C değerini bekler.

```python
b = (ModelBuilder((4,3,5),3)
     .add('permute', dims=[0,1,3,2])
     .add('reshape', shape=[2,5,6]))
flat = (ModelBuilder.sequential((4,3,5), head='regression', output_features=2, pooling='none')
        .add('flatten'))
```

`flatten` NCHW'den `[N,C*H*W]` üretir. Builder içinde yalnız schema2'nin son
feature bloğu olarak kullanılabilir ve head pooling='none' olmalıdır. Arkasına
identity dahil başka feature bloğu eklenmesi reddedilir. Vektör genişliği de
8192 sınırına tabidir. Bağımsız Registry flatten operatörü aynı batch koruyan
rank-4 giriş sözleşmesini kullanır.

Bu sınırlamalar input/output anlamını açık tutar. Genel rank-aware routing,
NHWC layout geçişleri ve vektörden yeniden NCHW'ye geçiş gelecekteki Graph Builder
tasarımına aktarılır. Bu aşamada bu yetenekler uygulanmış olarak sunulmaz.

## Preflight, kaynak ve eğitim

Preflight yeni sampling ve utility shape'lerini ağırlık oluşturmadan hesaplar.
Sampling ve utility parameter_count değeri exact 0 olur. Schema2 raporunda ayrı
head alanı feature/çıktı boyutlarını, exact parametre sayısını, aktivasyon eleman
sayısını ve head uyarılarını içerir. Toplam conservative 10M parametre bütçesine
head parametreleri de dahildir.

Sampling çıktıları mevcut advisory aktivasyon eşiğine dahildir. Schema2 head'in
feature ve çıktı sınırları da sayılır. Bu conservative tensor eleman hesabı
interpolation workspace, autograd ve gerçek peak RAM/VRAM için garanti vermez.
Attention'ın önceki zorunlu token/matris sınırları yeni sampling sonrasındaki
geometriye de uygulanır.

Classification head mevcut TrainingSession multiclass/binary sözleşmesiyle
kullanılabilir. Regression/embedding için TrainingSession'ın mevcut loss/metric
varsayımları uygun değildir. Bilinen schema2 regression/embedding modelleri
TrainingSession.prepare sırasında açıkça reddedilir. Bu modeller explicit
PyTorch optimizer ve differentiable loss döngüsüyle test edilir.

```python
import torch
from torch.nn import functional as F
model = regression.build()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
inputs = torch.randn(2,3,16,16)
targets = torch.randn(2,2)
optimizer.zero_grad()
loss = F.mse_loss(model(inputs), targets)
loss.backward()
optimizer.step()
```

Embedding için aynı döngüde hedef embedding'e MSE gibi uygun differentiable loss
kullanılabilir. Sequential automatic full resume kapsamı mevcut ModularCNN
koşullarıyla sınırlıdır. Normal state_dict strict yükleme iki schema için de
test edilir. CLI models show/validate/summary/inspect yeni head kaydını raporlar.

## Kaynak geçmişi ve Graph Builder girdileri

Yeni sampling.py HYSo için public torch sözleşmelerinden yazıldı ve root MIT
lisansı kapsamındadır. Mevcut lisans bildirimleri korunur. Hazır model kodu veya
pretrained ağırlık alınmadı. Önceki patent, marka ve dependency binary haklarına
ilişkin hukuki belirsizlikler sürer.

Graph Builder tasarımı her tensor kenarında rank, layout, sembolik batch ve
feature anlamını taşımalı, external add/concat kurallarını ve branch yaşam süresi
maliyetini tanımlamalıdır. Arbitrary permute/reshape ve flatten sonrası katman
bağlantıları bu genel sözleşmeye bağlıdır. Aşama 13.5.4 lineer zincir ve yukarıda
belirtilen terminal feature/head sınırını sağlar.
