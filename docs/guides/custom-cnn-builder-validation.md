# Custom CNN Builder giriş, çıkış ve hata doğrulaması

ModelBuilder'ın mevcut schema1 mimari JSON'u, add/build/modular yolları ve
ModelConfigError'ın ValueError tabanı korunur. `preflight()` ağırlık oluşturmadan
doğrulama ve shape özeti verir.

```python
from hyso import ModelBuilder

builder = (ModelBuilder((4, 9, 11), 3)
           .add('conv2d', in_channels=4, out_channels=8, groups=4, stride=2)
           .add('eca', channels=8))
report = ModelBuilder.preflight(builder.to_config())
assert report['valid']
assert report['input_shape'] == ['N', 4, 9, 11]
assert report['feature_shape'] == ['N', 8, 5, 6]
assert report['output_shape'] == ['N', 3]
```

## Input/output sözleşmesi

Builder input_shape parametresi `(C,H,W)` şeklindedir. JSON içinde üç elemanlı
liste olarak kalır. Gerçek model girdisi floating NCHW tensörüdür. N, çalışma
anında pozitif batch boyutudur. Mimari JSON'una sabit batch eklenmez.
Ön doğrulama çıktısındaki `'N'` semboliktir.

C/H/W ve num_classes 1..8192 aralığında Python tam sayılarıdır. Bool, kesirli,
negatif, sıfır, eksik/fazla boyutlar reddedilir. Tensorun device/dtype bilgisi
model ile eşleşmelidir. Sıfır batch ve farklı CHW runtime'da reddedilir.
Sequential classifier bloklardan sonra adaptive global average pool ve Linear
başlık kullanarak `(N,num_classes)` üretir.

## Shape inference

| Bileşen | Kanal/spatial kontrolü |
|---|---|
| Conv2d, ConvNormAct | Giriş kanalı önceki çıkışla eşleşir. Kernel/stride/padding her H/W ekseninde hesaplanır. |
| Grouped/depthwise convolution | groups pozitif olur ve giriş/çıkış kanallarını böler. Spatial formül aynı kalır. |
| ResidualBlock | Çıkış kanalı out_channels, spatial boyut ceil(input/stride). Kanal/stride değişiminde projection ile uyum sağlanır. |
| SE/CBAM/ECA/Coordinate | channels eşleşir, NCHW shape korunur. Residual içindeki attention kanalı out_channels ile eşleşir. |
| BatchNorm/GroupNorm | num_features/num_channels eşleşir. GroupNorm grup sayısı kanalları böler. |
| Activation, Identity, ChannelShuffle, DropBlock2D | Shape korunur. Shuffle groups kanalları böler. DropBlock olasılığı ve block_size doğrulanır. |

Genel convolution formülü her eksende
`floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1` şeklindedir.
Mevcut arayüz dilation=1 kullanır. JSON'a dilation eklenmesi bilinmeyen parametre
hatasıdır. Asimetrik kernel/stride/padding için H/W çiftleri desteklenir.
Çıktının pozitif ve en fazla 8192 olması gerekir.

Sequential ön doğrulama layers listesinde sıfır tabanlı indeks, bileşen adı,
giriş ve çıkış shape'lerini döndürür. parameter_estimate mevcut conservative
10M sınırını kullanır. Boyut ve parametre sınırları activation RAM garantisi
sağlamaz.

ModularCNN ve geliştirme profilindeki named modeller eski config validation
yollarını kullanır. Bu profillerde preflight shape_scope='configuration_only'
döndürür. Spatial input belirtilmediği için H/W ve feature_shape bilinmiyor
olarak bırakılır. Ticari Registry 16 girdidir ve hazır aileleri açmaz.

## Yapılandırılmış hatalar

| Kod | Türkçe anlam / düzeltme |
|---|---|
| INVALID_INPUT_SHAPE | Geçerli üç CHW boyutu ve runtime'da pozitif N kullanın. |
| INVALID_PARAMETER | Bileşenin desteklediği parametre/tür/aralıkları kontrol edin. |
| CHANNEL_MISMATCH | Giriş kanalını önceki çıkış kanalına eşitleyin. |
| GROUP_DIVISIBILITY_ERROR | İlgili kanal sayılarını bölen groups/num_groups seçin. |
| SPATIAL_DIMENSION_ERROR | Kernel, padding ve stride ile pozitif spatial çıktı sağlayın. |
| INVALID_BLOCK_ORDER | Lineer zincirde tam model yerine NCHW blokları kullanın. |
| PARAMETER_BUDGET_EXCEEDED | Kanal, kernel veya blok sayısını azaltın. |
| UNSUPPORTED_COMPONENT | Kurulu profilin Registry listesinden seçim yapın. |
| INVALID_OUTPUT_HEAD | num_classes için 1..8192 aralığında bir tam sayı kullanın. |

```python
from hyso import ModelBuilder, ModelConfigError

config = ModelBuilder((4, 9, 11), 3).to_config()
config['blocks'] = [{'name': 'conv2d', 'params': {'in_channels': 3, 'out_channels': 8}}]
report = ModelBuilder.preflight(config)
error = report['diagnostics'][0]
assert error['code'] == 'CHANNEL_MISMATCH'
assert error['layer_index'] == 0
assert error['expected'] == 3 and error['actual'] == 4
# Türkçe mesaj: Katmanların kanal sayıları uyuşmuyor.
# English detail: block 0: conv2d: incoming channels 4, expected 3
# Düzeltme: in_channels=4 kullanın.

try:
    ModelBuilder.from_config(config)
except ModelConfigError as exc:
    assert isinstance(exc, ValueError)
    diagnostic = exc.to_dict()
    assert diagnostic['message_en'] == str(exc)
```

Tanılama code, message_tr, message_en, layer_index, layer_id, expected, actual
ve suggestion alanlarını taşır. Bilinmeyen bağlam None olabilir. Schema1 layer
ID alanı kabul etmez. layer_id gelecekteki tanılama bağlamı için opsiyoneldir.
to_dict() JSON'a uygun bağımsız veri döndürür. Teknik exception ve neden zinciri
geliştirici için erişilebilirdir.

Ön doğrulama ilk hatayı ve sonraki blokların bağımsız Registry/parametre
hatalarını toplar. Önceki shape belirsiz olduğunda sonraki kanal/spatial hataları
tahmin edilmez. Hatalı zincirin output_shape ve layers sonuçları boş bırakılır.
Başlık/bağlantı hataları giderildikten sonra bütün zincir yeniden doğrulanmalıdır.

## CLI ve güvenlik

```console
hyso models validate architecture.json --json
hyso models summary architecture.json --json
```

Başarılı sonuçlarda mevcut alanlara `validation` ön doğrulama verisi eklenir.
ModelConfigError durumunda eski error alanına ek olarak diagnostics listesi
bulunur. Hatalı girdi exit code 2 üretir. stderr Türkçe açıklama ve öneri sunar.
`--debug` teknik traceback'i gösterir. Varsayılan hata çıktısında traceback yoktur.

JSON mevcut kapalı whitelist, boyut/derinlik sınırları, duplicate-key ve finite
sayı kontrollerini kullanır. Import yolu, callable veya eval çalıştırmaz.
Validation/summary/preflight ağırlık, pretrained indirme veya CUDA context
oluşturmaz. Build ve inspect --execute mevcut açık CPU oluşturma yollarıdır.

## Sınırlar ve sonraki çalışma girdileri

BatchNorm/GroupNorm train istatistik koşulları N ve runtime değerlerine bağlıdır.
Geçerli sembolik shape bunların her batch için sağlanacağını garanti etmez.
Tensor bellek kapasitesi, precision politikası ve çalışma zamanı maliyeti ayrı
konulardır. Yeni dtype/device veya dilation seçenekleri eklenmedi.

Graph bağlantıları, DAG/concat/add, kullanıcı katman ID'leri, dinamik H/W ve
kullanıcı blok kaydı sonraki tasarım girdileridir. Bu teslim yalnız 13.5.1
doğrulamasını kapsar.
