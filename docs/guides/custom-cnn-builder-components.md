# Kendi CNN modelinizi temel bileşenlerle kurma

Bu rehber Aşama 13.5.2 bileşenlerini açıklar. Bir katman görüntünün kanal sayısını,
yüksekliğini veya genişliğini değiştirir. Registry kullanılabilir yapı taşlarını ve
parametrelerini listeler. Builder bunları sırayla bağlar ve sonuna sınıflandırma
başlığını ekler. Ticari profilde 34 Registry girdisi bulunur. Önceki 16 isim korunur.

## İlk çalışan zincir

```python
from hyso import ModelBuilder

builder = (ModelBuilder((3, 64, 64), 5)
    .add('conv2d', in_channels=3, out_channels=32)
    .add('batch_norm', num_features=32)
    .add('gelu')
    .add('depthwise_conv2d', in_channels=32)
    .add('pointwise_conv2d', in_channels=32, out_channels=64)
    .add('group_norm', num_groups=8, num_channels=64)
    .add('max_pool2d', kernel_size=2)
    .add('se', channels=64)
    .add('adaptive_avg_pool2d', output_size=1))

report = ModelBuilder.preflight(builder.to_config(), batch_size=2)
assert report['feature_shape'] == ['N', 64, 1, 1]
assert report['output_shape'] == ['N', 5]
model = builder.build()
```

Gerçek giriş `[N,3,64,64]`, çıktı `[N,5]` olur. N pozitif batch sayısıdır.
Builder'a yalnız `(C,H,W)` verilir. Ön doğrulamadaki `N` semboliktir.
Tensor floating NCHW düzeninde olmalı, modelin dtype/device bilgisiyle eşleşmelidir.
`build()` yeni CPU ağırlıkları oluşturur. `preflight()` ağırlık oluşturmaz.

## Ortak convolution parametreleri

| Parametre | Sözleşme ve varsayılan |
|---|---|
| in_channels, out_channels | Gerekli pozitif tam sayılar, en fazla 8192. Depthwise çıkışı ayrıca hesaplanır. |
| kernel_size | Pozitif tam sayı veya `[kH,kW]`, varsayılan 3. Pointwise sabit 1 kullanır. |
| stride | Pozitif tam sayı veya `[sH,sW]`, varsayılan 1. |
| padding | Negatif olmayan tam sayı veya `[pH,pW]`, varsayılan 1. Pointwise sabit 0 kullanır. |
| dilation | Pozitif tam sayı veya `[dH,dW]`, varsayılan 1. Conv2d ve yeni spatial operatörlerde bulunur. |
| groups | Conv2d ve ConvTranspose2d için varsayılan 1. Giriş ve çıkış kanallarının ortak böleni olmalıdır. |
| bias | Bool, varsayılan True. Çok aşamalı bloklarda her convolution'a uygulanır. |
| depth_multiplier | Depthwise ve depthwise-separable için pozitif tam sayı, varsayılan 1. Ara kanal `in_channels * depth_multiplier` olur. |
| intermediate_channels | Spatial-separable ve factorized için gerekli pozitif tam sayı. Factorized değeri iki sınır kanal sayısını da aşamaz. |
| output_padding | Yalnız ConvTranspose2d, varsayılan 0. Her eksende stride veya dilation değerinden küçük olmalıdır. |

Her spatial çiftin elemanları en fazla 8192 olur. `padding_mode` Registry
parametresi olarak sunulmaz. Operatörler zeros padding kullanır. `conv_norm_act`
önceki parametre kümesini korur, dilation seçeneği `conv2d` üzerinden kullanılır.

Bir convolution ekseninde çıktı
`floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1` olur.
Transposed convolution için çıktı
`(input-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1` olur.
Her ara ve son C/H/W boyutu 1..8192 aralığında doğrulanır. [PyTorch Conv2d](https://docs.pytorch.org/docs/2.5/generated/torch.nn.Conv2d.html), [ConvTranspose2d](https://docs.pytorch.org/docs/2.5/generated/torch.nn.ConvTranspose2d.html)

## On convolution seçeneği

Aşağıdaki örneklerde `b = ModelBuilder((4,9,11),3)` kullanılır.
Her satırdaki çağrı yeni ve boş bir `b` üzerine uygulanır. Çıktı sütunu feature CHW
boyutudur. Builder başlığından sonraki sonuç her örnekte `[N,3]` olur.

| Seçenek | Davranış ve örnek çağrı | Feature CHW | Yaygın hata ve düzeltme |
|---|---|---|---|
| Standard | Her filtre tüm giriş kanallarını kullanır. `b.add('conv2d', in_channels=4, out_channels=8)` | `[8,9,11]` | in_channels önceki katmanın çıkışıyla eşleşmeli. |
| Pointwise | Her konumda 1×1 kanal karışımı. `b.add('pointwise_conv2d', in_channels=4, out_channels=8)` | `[8,9,11]` | Spatial kernel seçmek için conv2d kullanın. |
| Grouped | Kanallar bağımsız gruplara ayrılır. `b.add('conv2d', in_channels=4, out_channels=8, groups=2)` | `[8,9,11]` | groups=3 bu kanalları bölmez. Geçerli ortak bölenler 1,2,4 olur. |
| Depthwise | Her giriş kanalına multiplier kadar filtre. `b.add('depthwise_conv2d', in_channels=4, depth_multiplier=2)` | `[8,9,11]` | out_channels/groups elle verilmez. Çıkışı multiplier belirler. conv2d ile eşdeğer ayar groups=4, out_channels=8 olur. |
| Depthwise separable | Depthwise ardından dense 1×1. `b.add('depthwise_separable_conv2d', in_channels=4, out_channels=8, depth_multiplier=2)` | `[8,9,11]` | Ara kanal çarpımı 8192'yi aşmamalı. Multiplier azaltılabilir. |
| Dilated | Kernel örnekleri arasında boşluk. `b.add('conv2d', in_channels=4, out_channels=8, dilation=2, padding=2)` | `[8,9,11]` | Etkin kernel `d*(k-1)+1` olur. Padding veya giriş boyutunu buna göre seçin. |
| Spatial separable | Önce kH×1, sonra 1×kW. `b.add('spatial_separable_conv2d', in_channels=4, out_channels=8, intermediate_channels=3, kernel_size=[3,5], padding=[1,2])` | `[8,9,11]` | H stride/padding/dilation ilk, W ayarları ikinci operatöre gider. İki ara shape de geçerli olmalı. |
| Asymmetric | H/W ayarları bağımsızdır. `b.add('conv2d', in_channels=4, out_channels=8, kernel_size=[2,5], stride=[2,3], padding=[0,2], dilation=[1,2])` | `[8,4,3]` | Her eksenin etkin kernel ve padding hesabını ayrı kontrol edin. |
| Transposed | Explicit spatial büyütme/filtreleme. `b.add('conv_transpose2d', in_channels=4, out_channels=8, stride=2, output_padding=1)` | `[8,18,22]` | output_padding=2, stride=2 ve dilation=1 ile geçersizdir. output_padding=1 geçerlidir. |
| Factorized | Dense 1×1 kanal bottleneck ardından dense K×K. `b.add('factorized_conv2d', in_channels=4, out_channels=8, intermediate_channels=2)` | `[8,9,11]` | intermediate_channels bu örnekte en fazla 4 olabilir. Gizli activation veya normalization bulunmaz. |

Standard, grouped, dilated ve asymmetric seçenekleri ortak `conv2d` primitive'ini
kullanır. Pointwise ve depthwise girdileri aynı operatörü kısıtlı parametrelerle
sunar. Çok aşamalı üç blok HYSo modül bileşimidir. Ağırlıkları öğrenilebilir
bağımsız operatörlerdir. Eğitilmiş tek bir dense kernel'in ayrıştırılması yapılmaz.

Parametre hesabında `a=in_channels`, `b=out_channels`, `m=ara kanal`,
`q=int(bias)` ve `K=kH*kW` alınır.

| Bileşen | Gerçek trainable parametre sayısı |
|---|---|
| Conv2d / transposed | `a*b*K/groups + q*b` |
| Depthwise | `m*K + q*m`, `m=a*depth_multiplier` |
| Depthwise separable | `m*K + q*m + m*b + q*b` |
| Spatial separable | `a*m*kH + q*m + m*b*kW + q*b` |
| Factorized | `a*m + q*m + m*b*K + q*b` |

## Normalization

Normalization ölçekleri dengeler. Kanal sayısını veya spatial boyutları değiştirmez.
Her örnekte giriş ve çıkış `[N,4,9,11]` olur.

| Seçenek ve örnek | Matematiksel davranış | Parametreler | Yaygın hata ve düzeltme |
|---|---|---|---|
| BatchNorm2D `b.add('batch_norm', num_features=4)` | Kanal başına N/H/W üzerinde ortalama ve varyans. Eval modunda kayıtlı istatistikler kullanılabilir. | num_features gerekli, eps=1e-5, momentum=.1 veya None, affine=True, track_running_stats=True | Giriş kanalı eşleşmeli. Input statistics kullanılırken N*H*W>1 olmalı. |
| GroupNorm `b.add('group_norm', num_groups=2, num_channels=4)` | Örnek başına kanal grupları ve spatial eksenlerde normalization. | num_groups ve num_channels gerekli, eps=1e-5, affine=True | num_groups kanalları bölmeli. Native istatistik kontrolü `N*(C/groups)*H*W>1` gerektirir. |
| Kanal LayerNorm `b.add('channel_layer_norm', channels=4)` | Her N/H/W konumunda C üzerinde `(x-mean)/sqrt(var+eps)` ve opsiyonel affine. | channels gerekli, eps=1e-5, affine=True | CHW'nin tamamını normalize etmez. Dış tensor düzeni NCHW kalır. Kanalları NHWC olarak vermeyin. |
| InstanceNorm2D `b.add('instance_norm', num_features=4)` | Her örnek ve kanal için H/W üzerinde normalization. | num_features gerekli, eps=1e-5, momentum=.1, affine=False, track_running_stats=False | Input statistics kullanıldığında H*W>1 gerekir. Running stats yoksa eval da bu koşulu taşır. |
| Identity / NoNorm `b.add('identity')` | Girdiyi aynen döndürür. | Parametresiz. | Normalization parametresi eklemeyin. |

eps pozitif finite sayı olmalı. Momentum desteklenen türlerde `[0,1]` aralığındadır.
BatchNorm için None kümülatif ortalamayı seçer. InstanceNorm bu Registry profilinde
None kabul etmez. Affine ve track_running_stats yalnız bool kabul eder.
Affine açıkken kanal başına iki parametre vardır. Kanal LayerNorm da `2*C`
parametre kullanır. Affine kapalıysa bu sayı sıfırdır.

Batch sembolik olduğunda runtime istatistik koşulları `warnings` içinde raporlanır.
InstanceNorm H*W=1 ve track_running_stats=False için her iki modda da geçersiz
olduğundan hata üretilir. True seçilirse train koşulu uyarı olarak kalır, eval
kayıtlı istatistiklerle kullanılabilir. [BatchNorm2d](https://docs.pytorch.org/docs/2.5/generated/torch.nn.BatchNorm2d.html), [GroupNorm](https://docs.pytorch.org/docs/2.5/generated/torch.nn.GroupNorm.html), [LayerNorm](https://docs.pytorch.org/docs/2.5/generated/torch.nn.LayerNorm.html), [InstanceNorm2d](https://docs.pytorch.org/docs/2.5/generated/torch.nn.InstanceNorm2d.html)

## Activation

Aktivasyonlar her tensor elemanını dönüştürür. Tüm örneklerde giriş ve çıkış
`[N,4,9,11]` olur. Kullanım `b.add(ad, **parametreler)` şeklindedir.

| İsim ve örnek | Matematiksel davranış | Parametreler ve yaygın hatalar |
|---|---|---|
| `b.add('relu')` | `max(0,x)` | Parametresiz. inplace kabul edilmez. |
| `b.add('leaky_relu', negative_slope=.1)` | Negatif x için slope*x, diğer değerlerde x. | negative_slope=.01, finite ve negatif olmayan sayı. |
| `b.add('prelu', num_parameters=4, init=.25)` | Kanal bazlı veya ortak öğrenilen negatif eğim. | num_parameters=1, init=.25. Eğim sayısı 1 veya C olmalı. init varsayılan CPU float32 ağırlıklarında temsil edilebilmeli. |
| `b.add('silu')` | `x*sigmoid(x)` | Parametresiz. |
| `b.add('gelu', approximate='none')` | `x*Phi(x)` veya tanh yaklaşımı. | approximate yalnız 'none' veya 'tanh', varsayılan 'none'. |
| `b.add('mish')` | `x*tanh(softplus(x))` | Parametresiz. |
| `b.add('hardswish')` | `x*clip(x+3,0,6)/6` | Parametresiz. |
| `b.add('hardsigmoid')` | `clip(x/6+.5,0,1)` | Parametresiz. |

PReLU'nun `num_parameters` kadar trainable parametresi vardır. Diğer aktivasyonlar
parametresizdir. Callable veya import yolu kabul edilmez. Yanlış parametre adını
`models show` çıktısındaki kapalı listeye göre düzeltin.

## Pooling

Pooling yakın spatial değerleri tek değere toplar. Kanal sayısı korunur.
Aşağıdaki örneklerde giriş `[N,4,9,11]` olur.

| İsim ve örnek | Davranış ve çıktı | Parametreler | Yaygın hata ve düzeltme |
|---|---|---|---|
| `b.add('max_pool2d', kernel_size=2)` | Penceredeki maksimum, `[N,4,4,5]`. | kernel_size gerekli, stride=None ile kernel kullanılır, padding=0, dilation=1, ceil_mode=False | Padding en fazla floor(kernel/2) olmalı. Etkin kernel girişten büyükse çıktı hesabını kontrol edin. |
| `b.add('avg_pool2d', kernel_size=3, stride=2, padding=1)` | Pencere ortalaması, `[N,4,5,6]`. | kernel_size gerekli, stride=None, padding=0, ceil_mode=False, count_include_pad=True | count_include_pad=False seçilirse padding bölene katılmaz. Dilation parametresi kabul edilmez. |
| `b.add('adaptive_avg_pool2d', output_size=[2,3])` | Hedef hücrelerin ortalaması, `[N,4,2,3]`. | output_size gerekli pozitif tam sayı veya H/W çifti. | None, sıfır, negatif ve 8192 üstü boyutlar kabul edilmez. |
| `b.add('adaptive_max_pool2d', output_size=[2,3])` | Hedef hücrelerin maksimumu, `[N,4,2,3]`. | output_size gerekli pozitif tam sayı veya H/W çifti. | return_indices kabul edilmez. Çıktı her zaman tensordur. |
| `b.add('global_avg_pool2d')` | Tüm spatial konumların ortalaması, `[N,4,1,1]`. | Parametresiz. | Otomatik flatten beklemeyin. NCHW çıkış sınıflandırma başlığına uyumludur. |

Sabit pencerelerde önce convolution benzeri floor formülü uygulanır. Ceil mode
için bölme öncesi paya `stride-1` eklenir. Son pencerenin başlangıcı
`(out-1)*stride >= input+padding` olursa çıktı bir azaltılır. MaxPool padding
bölgelerini negatif sonsuz gibi ele alır. AvgPool count_include_pad ve sağ sınır
davranışı doğrudan PyTorch operatöründen gelir. [MaxPool2d](https://docs.pytorch.org/docs/2.5/generated/torch.nn.MaxPool2d.html), [AvgPool2d](https://docs.pytorch.org/docs/2.5/generated/torch.nn.AvgPool2d.html)

## Hatalar, uyarılar ve maliyet

```python
report = ModelBuilder.preflight(
    builder.to_config(), batch_size=2, activation_limit=16_777_216)
for layer in report['layers']:
    print(layer['name'], layer['input_shape'], layer['output_shape'])
    print(layer['parameter_count'], layer['parameter_estimate'], layer['warnings'])
```

`diagnostics` doğrulamayı başarısız kılan hataları taşır. `warnings` yapılandırılmış
istatistik koşullarını ve advisory aktivasyon eşiği aşımlarını taşır. Her tanılama
code, message_tr, message_en, layer_index, layer_id, expected, actual ve suggestion
alanlarını korur. Group hatalarında gerçek ortak bölenler listelenir.

Yeni kodlar `INVALID_OUTPUT_PADDING`, `INVALID_POOLING_CONFIGURATION`,
`INTERMEDIATE_SHAPE_ERROR`, `STATISTICS_REQUIREMENT` ve `RESOURCE_LIMIT_WARNING`
olur. Önceki dokuz kod korunur. Dilation, activation ve normalization scalar
hataları mevcut `INVALID_PARAMETER` kodunu kullanır.

Katman `parameter_count` alanı yeni operatörler ve doğrudan hesaplanabilen eski
primitive'ler için gerçek trainable parametre sayısıdır. Diğer eski bloklarda None
kalabilir. `parameter_estimate` conservative üst sınırdır. Eski convolution
hesabındaki güvenlik payı korunur. Toplam 10M sınırı bu üst sınıra uygulanır.
Running mean/variance buffer'ları trainable parametre sayısına dahil değildir.

`activation_estimate` açık batch varsayımıyla input, sequential katman sınırları ve
yeni çok aşamalı convolution ara tensorlarının eleman sayılarını verir.
`peak_tensor_elements` bu kapsamdaki en büyük tek tensor, `summed_tensor_elements`
ise kapsanan tensorların toplamıdır. Varsayılan 16.777.216 eleman eşiği advisory
uyarı üretir. Yapılandırmayı veya build davranışını otomatik değiştirmez.
Eşiği aşan modeller build öncesi preflight ile görülebilir. Eski geçerli schema1
modellerine yeni zorunlu aktivasyon kotası uygulanmaz.

Bu hesap dtype byte sayısı, operator workspace, autograd kayıtları, optimizer,
eski attention/residual içleri, classifier head ve allocator davranışını kapsamaz.
Peak RAM/VRAM garantisi sağlamaz. Bellek ölçümü veya benchmark çalıştırılmaz.

## JSON ve eğitim uyumluluğu

Schema1 aynı kalır. Yeni bileşenler `blocks=[{name,params}, ...]` listesinde
saklanır. Eksik eski Conv2d dilation alanı 1 olur. Eski BatchNorm ve GroupNorm
parametreleri aynı varsayılanları ve state_dict anahtarlarını kullanır.
Yeni bileşenler deterministic JSON roundtrip ve strict state_dict yüklemesiyle
doğrulanır. JSON ağırlık, callable, import yolu veya executable kod taşımaz.

`TrainingConfig(...).prepare(model, loader).fit()` ile kısa CPU eğitimi kullanılabilir.
Sequential classifier için otomatik tam epoch resume kapsamı genişletilmedi.
Bu yol mevcut ModularCNN koşullarıyla sınırlıdır. Eğitimli sequential modelin
ağırlıkları normal `state_dict` ile ayrıca saklanabilir.

## Kaynak geçmişi ve dağıtım

`hyso/models/components.py` bu aşamada public PyTorch operatör sözleşmelerinden
HYSo için yazıldı. Üçüncü taraf hazır model implementasyonu kopyalanmadı.
Geometri ve maliyet Python tamsayı hesaplarıdır. Modül root MIT lisansı kapsamındadır.
PyTorch bağımlılığının lisans bildirimleri THIRD_PARTY_NOTICES.md içinde korunur.
Ticari pozitif izin listesine yalnız bu yeni modül eklendi.

Hazır M3/M4 ailelerinin önceki dışlamaları sürer. Patent, marka, veri/ağırlık hakları
ve bağımlılık binary'lerinin tüm transitive koşulları için evrensel hukuki onay
verilmez. Bu aşamanın kaynak incelemesi sıfır risk iddiası taşımaz.
