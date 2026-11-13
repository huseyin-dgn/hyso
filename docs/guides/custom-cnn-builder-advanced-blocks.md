# İleri CNN yapı taşları

Aşama 13.5.3 ile ticari Registry 34 girdiden 48 girdiye çıkar. Önceki girdiler ve
schema1 mimari biçimi korunur. Yeni yapı taşları HYSo'nun `advanced.py` modülünde
public PyTorch operatörleriyle oluşturulur. Geliştirme profiline ait hazır model
modüllerinden kod veya constructor alınmaz.

## Residual bağlantıyı kullanma

Residual blok iki tensoru toplar. Ana dal filtreleme yapar, shortcut ham girdiyi
taşır. Kanal sayısı veya stride değiştiğinde shortcut üzerinde bias içermeyen 1×1
projection uygulanır. Diğer durumda shortcut identity olur. Resize/crop yapılmaz.
Çıktı `shortcut(x) + gate * DropPath(Dropout(attention(main(x))))` şeklindedir.
Gated blok dışındaki gate sabit 1 olur. Toplama sonrasında ek aktivasyon yoktur.

Yeni residual blokların ortak parametreleri aşağıdadır.

| Parametre | Sözleşme |
|---|---|
| in_channels, out_channels | Gerekli pozitif tam sayı, 1..8192. |
| intermediate_channels | Inverted dışında gerekli pozitif ara kanal genişliği, 1..8192. |
| expansion_ratio | Yalnız inverted için pozitif tam sayı, varsayılan 2. Ara genişlik in_channels ile çarpılır. |
| kernel_size | Pozitif tek tam sayı, varsayılan 3. |
| stride | Pozitif tam sayı veya H/W çifti, varsayılan 1. |
| dilation | Pozitif tam sayı veya H/W çifti, varsayılan 1. |
| normalization | 'batch', 'group' veya None, varsayılan 'batch'. GroupNorm grup sayısı gcd(8,C) olur. |
| activation | 'relu', 'silu' veya 'gelu', varsayılan 'silu'. |
| attention | 'none', 'eca', 'spatial_attention' veya 'channel_attention', varsayılan 'none'. Ana dal çıkışına uygulanır. |
| dropout, drop_path | [0,1) aralığında olasılık, varsayılan 0. Yalnız ana dalı etkiler. |
| groups | Yalnız grouped bottleneck, varsayılan 1. Ara genişliği bölmelidir. |
| gate_init, learnable_gate | Yalnız gated residual, varsayılan 1 ve True. Başlangıç değeri [0,1] aralığındadır. |

Convolution'lar bias içermez. Normalization affine ağırlıkları kullanır.
Spatial padding her eksende `dilation*(kernel_size-1)/2` seçilir. Böylece çıktı
yüksekliği ve genişliği `ceil(input/stride)` olur. Tek kernel kısıtı bu padding'i
tam sayı yapar. Her ara tensorun C/H/W boyutu 1..8192 aralığında doğrulanır.

### Bottleneck residual

`bottleneck_residual` önce 1×1 ile ara genişliğe geçer, K×K spatial filtre uygular,
son 1×1 ile çıkış kanallarını üretir. İlk iki convolution sonrasında norm ve
aktivasyon, son convolution sonrasında norm vardır. Ara genişliği azaltmak
spatial filtrenin parametre maliyetini düşürür.

```python
b.add('bottleneck_residual', in_channels=32, out_channels=64,
      intermediate_channels=16, stride=2)
```

Giriş `[N,32,15,17]`, çıktı `[N,64,8,9]` olur. Shortcut projection kullanır.
Ara genişliğin pozitif olmaması hata üretir. Önce daha küçük bir ara genişlik
seçerek 10M model parametre bütçesini sağlayabilirsiniz.

### PreActivation residual

`preactivation_residual` iki K×K convolution kullanır. Her convolution öncesinde
norm ve aktivasyon vardır. İlk convolution spatial stride uygular, ikinci stride
1 kullanır. Shortcut her zaman ham girdiden hesaplanır. Projection preactivation
görmüş girdiyi kullanmaz. Bu sıra, özellikleri convolution öncesinde normalize
etmek isteyen kullanıcı için açık bir seçenektir.

```python
b.add('preactivation_residual', in_channels=32, out_channels=32,
      intermediate_channels=16, normalization='group')
```

Giriş ve çıkış `[N,32,15,17]` olur, shortcut identity kalır. BatchNorm seçilirse
train istatistik koşulları geçerlidir. Küçük domain hatalarında batch/spatial
boyutunu veya normalization seçimini kontrol edin.

### Wide residual

`wide_residual_block` iki K×K convolution kullanır. Birinci işlem explicit
intermediate_channels genişliğine geçer, norm ve aktivasyon uygular. İkinci işlem
out_channels üretip norm uygular. Ara genişliği büyütmek filtre kapasitesini ve
parametre/aktivasyon maliyetini birlikte artırır.

```python
b.add('wide_residual_block', in_channels=32, out_channels=32,
      intermediate_channels=64)
```

`[N,32,15,17]` girdisi aynı boyutta çıkar. Ara tensor `[N,64,15,17]` olur.
Bütçe aşımında intermediate_channels veya kernel_size değerini azaltın.

### Grouped bottleneck

`grouped_bottleneck` bottleneck sırasını korur, ortadaki K×K convolution'ı gruplara
ayırır. Groups hem orta convolution'ın giriş hem çıkış genişliğini bölmelidir.
1×1 giriş ve çıkış işlemleri dense kalır.

```python
b.add('grouped_bottleneck', in_channels=32, out_channels=64,
      intermediate_channels=24, groups=3)
```

`[N,32,15,17]` girdisi `[N,64,15,17]` çıkar. Projection uygulanır. groups=5
geçersizdir. Hata önerisi 24'ün gerçek bölenlerini listeler.

### Inverted bottleneck

`inverted_bottleneck` 1×1 expansion, depthwise K×K ve 1×1 projection sırasını
kullanır. Ara kanal `in_channels*expansion_ratio`, depthwise grup sayısı bu ara
kanal sayısıdır. İlk iki işlem norm/aktivasyon, son işlem norm içerir.

```python
b.add('inverted_bottleneck', in_channels=64, out_channels=64,
      expansion_ratio=2)
```

`[N,64,15,17]` girdisi aynı boyutta çıkar, ara genişlik 128 olur. Kanal/stride
uygunsa identity shortcut üzerinden toplama yapılır. Ara çarpım 8192'yi aşarsa
expansion_ratio azaltılmalıdır. Projection maliyeti rapora dahildir.

### Gated residual

`gated_residual` wide residual ana dalını kullanır. gate_init başlangıç katkısını
belirler. learnable_gate=True iken gate doğrudan öğrenilen tek skalerdir ve
eğitim sırasında başlangıç aralığından çıkabilir. Sigmoid ile sınırlandırılmaz.
Sabit sıfır gate, ana dalın çıktıya katkısını kapatır. Ana dal hesaplaması ve
BatchNorm güncellemeleri yine gerçekleşir. Öğrenilen sıfır gate ilk adımda ana dal
parametrelerine sıfır gradient verebilir, gate gradient'i üzerinden açılabilir.

```python
b.add('gated_residual', in_channels=32, out_channels=32,
      intermediate_channels=16, gate_init=.1, learnable_gate=True)
```

Giriş ve çıkış `[N,32,15,17]` olur. Sabit gate için learnable_gate=False seçin.
Başlangıçta negatif veya 1 üstü gate_init hatası alınırsa [0,1] aralığını kullanın.

## Attention seçimi

Önceki SE, CBAM, ECA ve Coordinate Attention girdileri korunur. Yeni attention
girdileri input channels değerini doğrular ve aynı `[N,C,H,W]` boyutunu döndürür.
Kanal değiştirmek için attention öncesinde veya sonrasında convolution ekleyin.

### Spatial attention

`spatial_attention` kanal eksenindeki mean ve max özetlerini birleştirir. İki
kanallı descriptor üzerinde bias içermeyen K×K convolution ve sigmoid uygulanır.
Son `[N,1,H,W]` kapısı girdiyi çarpar. Konumsal vurgulama için kullanılabilir.

```python
b.add('spatial_attention', channels=32, kernel_size=7)
```

Giriş ve çıkış `[N,32,15,17]` olur. Parametre sayısı `2*K*K` olur. kernel_size
pozitif ve tek olmalıdır. Çift kernel hatasında 3, 5 veya 7 gibi tek değer seçin.
Train/eval davranışı aynıdır.

### Channel attention

`channel_attention` spatial mean ve max descriptor'larına aynı iki katmanlı 1×1
MLP'yi uygular. MLP arada ReLU kullanır. Sonuçlar toplanır ve sigmoid ile kanal
kapısı elde edilir. Gizli genişlik `m=max(1,C//reduction)` olur. Bu işlem farklı
kanalların katkısını öğrenmek için kullanılabilir.

```python
b.add('channel_attention', channels=32, reduction=8)
```

`[N,32,15,17]` input/output boyutu korunur. Öğrenilen parametre sayısı
`2*C*m+m+C` olur. reduction varsayılan 8 ve pozitif tam sayıdır. Descriptor'lar
`[N,C,1,1]` boyutundadır. Train/eval davranışı aynıdır.

### Self-attention ve çok başlıklı spatial attention

`spatial_self_attention` tek head, `multihead_spatial_attention` varsayılan dört
head kullanır. Her spatial konum bir token olur, `L=H*W` alınır. 1×1 QKV
projection C kanalı E embedding boyutuna taşır. Her head `D=E/heads` genişlik
kullanır. `softmax(Q*K^T/sqrt(D))*V` hesabından sonra head'ler birleştirilir ve
1×1 output projection C kanalına döner. Maske seçeneği sunulmaz.

```python
b.add('multihead_spatial_attention', channels=32, embedding_dim=32,
      num_heads=4, dropout=.1)
```

`[N,32,15,17]` input/output boyutu korunur. Parametre sayısı `4*C*E+3*E+C` olur.
embedding_dim None veya eksikse C kullanılır. E en fazla 2730 olur, böylece QKV
ara kanalı 8192 sınırında kalır. E head sayısına bölünmelidir. Örneğin channels=60,
num_heads=8 ve varsayılan embedding geçersizdir. E=64 seçmek veya 60'ın gerçek
bölenlerinden bir head sayısı kullanmak geçerlidir. Dropout varsayılan 0 ve
[0,1) aralığındadır. Yalnız attention ağırlıklarına train modunda uygulanır.

Her iki giriş için **L en fazla 1024**, **N*heads*L² en fazla 4.194.304** olur.
Bu sınırlar zorunludur. Preflight açık batch_size varsayımıyla, runtime gerçek N
ile QKV işleminden önce kontrol eder. `[N,C,64,64]` 4096 token üretir ve reddedilir.
Önce pooling ile çözünürlüğü azaltabilirsiniz. Spatial/channel attention veya
ECA quadratic token matrisi üretmeden kullanılabilir. Hata üretilen token ve
matris eleman sayısını verir. Sınırlar peak bellek garantisi sağlamaz.

## Regularization

`dropout` elemanları, `dropout2d` her örnek için tüm spatial kanalları sıfırlar.
Varsayılan p=.5, aralık [0,1] olur. Train modunda hayatta kalan katkılar 1/(1-p)
ile ölçeklenir. p=1 native operatörün sıfır çıktısını verir. Eval identity olur.
Mevcut `dropblock2d` implementasyonu ve varsayılanları korunur.

`drop_path` `[N,1,1,1]` Bernoulli maskesiyle örnek başına verilen tensorun
tamamını maskeler. `x*Bernoulli(1-p)/(1-p)` uygulanır. Varsayılan p=0,
aralık [0,1) olur. Residual içindeki drop_path parametresi yalnız ana dalı
maskeler. Standalone `b.add('drop_path', p=.1)` kendisine verilen bütün feature
tensorunu maskeler. Kullanıcı bu iki kullanımın etkisini katman yerleşimiyle seçer.
Eval identity olur. p=1 hatasında 1 altında olasılık seçin.

`gaussian_noise` train modunda her elemana bağımsız `Normal(0,std²)` gürültüsü
ekler. Varsayılan std=.1, desteklenen aralık [0,1.000.000] olur. Eval ve std=0
identity döndürür. Input yerinde değiştirilmez. Doğrudan modül forward çağrısı
CPU `generator=` kabul eder. Generator JSON'a yazılmaz. Aynı CPU generator seed'i
tekrarlanabilir gürültü üretir.

```python
b.add('dropout', p=.2)
b.add('dropout2d', p=.1)
b.add('drop_path', p=.1)
b.add('gaussian_noise', std=.05)
```

Bu dört girdide input/output `[N,32,15,17]` kalır, trainable parametre yoktur.
Negatif std veya olasılık sınırı hatalarında ilgili aralığı kullanın. Çıktı
tensorları ve random maskeler aktivasyon maliyeti oluşturur. Global CPU seed'i
veya GaussianNoise generator'ı deterministik testler için kullanılabilir.
[PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html), [Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html), [softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)

## Preflight ve çalışan zincir

```python
from hyso import ModelBuilder

b = (ModelBuilder((3,64,64),3)
     .add('conv2d', in_channels=3, out_channels=32)
     .add('batch_norm', num_features=32).add('silu')
     .add('bottleneck_residual', in_channels=32, out_channels=64, intermediate_channels=16)
     .add('eca', channels=64).add('max_pool2d', kernel_size=2)
     .add('inverted_bottleneck', in_channels=64, out_channels=64, expansion_ratio=2)
     .add('drop_path', p=.1).add('channel_attention', channels=64)
     .add('global_avg_pool2d'))
report = ModelBuilder.preflight(b.to_config(), batch_size=2)
assert report['valid']
assert report['output_shape'] == ['N',3]
```

Yeni katmanların parameter_count alanı exact trainable sayıdır. Projection ve
affine normalization parametreleri bu sayıya dahildir. İç operator planı hem
construction hem geometry/maliyet hesabında kullanılır. Ara norm/aktivasyon
sınırları conservative eleman toplamına dahil edilir. Rapor projection,
residual_active ve attention_cost alanlarını ekler. Attention matris maliyeti
ayrı gösterilir. Başlangıçtaki gate katkısı residual_active ile belirtilir.

Mevcut advisory aktivasyon eşiği korunur. ATTENTION_RESOURCE_LIMIT zorunlu
attention sınır hatasıdır. Normalization istatistik koşulları warnings içinde
görünür ve gerçek girdide kontrol edilir. Schema1 JSON aynı name/params biçimini
kullanır. Eski state_dict anahtarları yeniden adlandırılmaz. Yeni bloklar strict
state_dict ve TrainingSession CPU smoke ile doğrulanır. Sequential automatic
resume kapsamı mevcut ModularCNN koşullarıyla sınırlıdır.

## Kaynak ve ilerideki graph gereksinimleri

Yeni modül HYSo için public torch operatör sözleşmelerinden yazıldı ve root MIT
lisansı kapsamındadır. Üçüncü taraf hazır model implementasyonu alınmadı. Mevcut
THIRD_PARTY_NOTICES.md bildirimleri korunur. Patent, marka, veri/ağırlık hakları
ve dependency binary transitive koşullarına ilişkin önceki belirsizlikler sürer.

İlerideki Graph Builder tasarımı external add/concat girişlerinin ayrı shape
sözleşmelerini, branch activation lifetime hesabını, katman kimliklerini ve
attention kaynak bütçesinin tüm dallarda uygulanmasını tanımlamalıdır. Bu aşama
lineer Builder içinde blokların internal shortcut bağlantılarını sağlar.
