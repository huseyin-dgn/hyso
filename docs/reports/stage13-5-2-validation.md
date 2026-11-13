# Aşama 13.5.2 doğrulama raporu

13 Kasım 2026 tarihinde `feature/hyso-models-next` üzerinde tamamlandı.
Başlangıç ve teslim HEAD değeri `71b84b3a3fe164f5bbe8fb0a6621bf378a30ea40`.

## Mekanizmalar ve Registry

Ticari Registry önceki 16 girdiyi koruyarak **34 girdiye** çıktı.
Geliştirme checkout Registry'si **62 girdi** içerir. Ticari Python izin listesi
152 modülden **153 modüle** çıktı. Yeni bağımlılık eklenmedi.

| Alan | Önceden bulunan mekanizmalar | Bu aşamadaki eklemeler |
|---|---|---|
| Convolution | conv2d üzerinden standard, pointwise, grouped, depthwise ve asymmetric ayarlar. conv_norm_act ve residual korunur. | conv2d dilation desteği, pointwise_conv2d ve depthwise_conv2d açık kayıtları, depthwise_separable_conv2d, spatial_separable_conv2d, conv_transpose2d, factorized_conv2d |
| Normalization | batch_norm, group_norm, identity ile NoNorm | channel_layer_norm, instance_norm. Eski normalization girdilerine açık eps/affine ve uygun momentum/running-stat parametreleri |
| Activation | relu, silu, hardswish | leaky_relu, prelu, gelu, mish, hardsigmoid |
| Pooling | Sequential classifier'ın implicit adaptive-average/flatten/linear başlığı | max_pool2d, avg_pool2d, adaptive_avg_pool2d, adaptive_max_pool2d, global_avg_pool2d |
| Diğer korunan girdiler | se, cbam, eca, coordinate, modular_cnn, channel_shuffle, dropblock2d | Mevcut isim ve operatör davranışları korunur |

Factorized sözleşmesi dense 1×1 kanal bottleneck ardından dense K×K convolution
olarak seçildi. intermediate_channels iki sınır kanal sayısını da aşamaz.
Spatial-separable sözleşmesi kH×1 ardından 1×kW operatörüdür. Her iki yeni bileşimde
ara şekiller ayrıca doğrulanır. Depthwise-separable depthwise ardından dense 1×1
kullanır. Gizli activation, normalization veya bağlantı eklenmez.

LayerNorm açık `channel_layer_norm` adıyla her spatial konumda C eksenini normalize
eder. Public input/output NCHW kalır. Transposed output_padding kuralı her eksende
`output_padding < stride or output_padding < dilation` olarak test edildi.

Parametre varsayılanları, tüm mekanizmaların matematiği, kullanım örnekleri,
giriş/çıkış boyutları ve hata düzeltmeleri bileşen rehberinde bulunur.
[Bileşen rehberi](../guides/custom-cnn-builder-components.md)

## Preflight ve kaynak hesabı

Önceki dokuz hata kodu korunur. Beş kod eklendi.

| Kod | Kullanım |
|---|---|
| INVALID_OUTPUT_PADDING | Transposed convolution çapraz alan kısıtı |
| INVALID_POOLING_CONFIGURATION | Pool padding/kernel kısıtı |
| INTERMEDIATE_SHAPE_ERROR | Geçersiz convolution ara kanalı veya spatial shape |
| STATISTICS_REQUIREMENT | Kesin geçersiz istatistik domain'inde hata, sembolik N veya train/eval koşullarında uyarı |
| RESOURCE_LIMIT_WARNING | Advisory aktivasyon eleman eşiği aşımı |

Dilation ve activation/normalization scalar hataları `INVALID_PARAMETER` kullanır.
Group hatalarında gerçek ortak bölenler hesaplanır ve öneriye eklenir.
Tanılamalar mevcut sekiz alanını korur. Hatalar diagnostics, uyarılar warnings
listesinde yer alır. Geçersiz zincirden downstream shape üretilmez.

Katman raporları isim, indeks, input/output, yeni convolution ara shape'leri,
parameter_count, parameter_estimate, estimate türü, aktivasyon elemanları ve
uyarıları içerir. Yeni bileşenlerde gerçek trainable parametre sayısı operatörün
parametreleriyle karşılaştırıldı. Eski conservative hesaplara ait güvenlik payı
ve model başlığını içeren 10M parametre sınırı korunur. Eski karmaşık bloklarda
parameter_count None olabilir, estimate açıkça conservative üst sınır olarak
etiketlenir.

`preflight(config, batch_size=1, activation_limit=16_777_216)` için batch_size
yalnız aktivasyon hesabının açık varsayımıdır. İstatistik koşullarında N sembolik
kalmaya devam eder. Aktivasyon kapsamı input, sequential sınırlar ve yeni
convolution iç aşamalarıdır. En büyük tek tensor ve kapsanan tensorların toplam
eleman sayıları ayrı verilir.

16.777.216 elemanlık varsayılan eşik advisory olarak seçildi. Eski geçerli
schema1 modellerinin kabulünü korumak için eşik aşımı uyarı üretir. Kullanıcı
build öncesinde bu raporu inceleyebilir. Hesap gerçek peak RAM/VRAM, operator
workspace, autograd, optimizer veya eski blokların tüm iç aktivasyonlarını
temsil etmez. Gerçek bellek ölçümü gerçekleştirilmedi.

## Test sonuçları

**413 benzersiz test başarılı, 0 başarısız, 0 atlanan.** Ön koşular bu sayıya
tekrar eklenmedi. Son birleşik koşu 412 test, ayrı dışlama regresyonu 1 test içerir.

| Test modülü | Son başarılı test sayısı | Kapsam |
|---|---|---|
| test_builder_components | 227 | A–K bileşen, geometry, hata/uyarı, JSON, eğitim ve state_dict kontrolleri |
| test_cli | 76 | Mevcut stdout/stderr, exit code, izin ve metadata sözleşmeleri |
| test_model_builder | 42 | Mevcut block API, eski yapılandırmalar ve ModularCNN/state_dict davranışı |
| test_builder_validation | 61 | Preflight, allocation fence, önceki dokuz kod ve kurulu wheel/sdist preflight |
| test_commercial_distribution | 4 | Wheel/sdist gerçek içerik ve kurulu ticari bileşen testleri |
| test_packaging | 2 | Pozitif izin listesi, paket dosyaları, lisans ve metadata |
| test_m4_architectures seçili dışlama testi | 1 | Geliştirme Registry sayısı ve FasterNet/PartialConv dışlamaları |

Yeni testler A Registry, B convolution, C normalization, D activation, E pooling,
F hata/uyarı, G JSON, H CPU gradient, I strict state_dict, J TrainingSession ve
K CLI kategorilerini kapsar. G/H/I aynı parametrik vakalarda birlikte doğrulanır.
L ticari wheel/sdist testleri ayrı dağıtım modülündedir.

| Parity grubu | Sonuç |
|---|---|
| Convolution | Tek/çift kernel, asimetrik H/W, stride 1/2/3, dilation 1/2/3, groups 1/2/C, depthwise multiplier, transposed çıktı ve çok aşamalı ara shape eşleşti. Bias kapalı gerçek parametre sayıları da eşleşti. |
| Normalization | BatchNorm, GroupNorm, InstanceNorm train/eval sonuçları native PyTorch ile birebir eşleşti. Kanal LayerNorm açık eksen dönüşümüyle native LayerNorm'a eşleşti. |
| Activation | LeakyReLU, PReLU, GELU exact/tanh, Mish ve HardSigmoid native operatör sonuçlarına eşleşti. Önceki üç activation regresyonları geçti. |
| Pooling | Floor/ceil sağ sınır düzeltmesi, küçük spatial giriş, odd/even kernel, dilation, adaptive büyütme/küçültme ve average divisor davranışı eşleşti. Global average NCHW 1×1 çıkışı korudu. |

Tüm 18 yeni kayıt üzerinden JSON/API roundtrip, train/eval, CPU forward/backward,
finite input/parameter gradient, optimizer step ve strict state_dict doğrulandı.
Ağırlıksız preflight testlerinde Module oluşturma, torch.empty ve CUDA sorguları
engellendi. Testler deterministik seed ve tek CPU thread ile çalıştırıldı.

Birleşik koşuda değiştirilmeyen Trainer'ın mevcut `torch.cuda.amp.GradScaler`
kullanımından üç deprecation uyarısı görüldü. Test başarısızlığı oluşmadı.
Native benchmark, CUDA workload, uzun eğitim veya P01–P35 kabul koşusu başlatılmadı.

### TrainingSession

İstenen `[3,64,64]` zinciri Conv2d 3→32, BatchNorm, GELU, Depthwise 32→32,
Pointwise 32→64, GroupNorm, MaxPool, SE ve AdaptiveAvgPool üzerinden çalıştı.
Feature `[N,64,1,1]`, classification çıktı `[N,5]` oldu. İki sentetik CPU örneğiyle
bir epoch ve bir optimizer update tamamlandı.

Kurulu ticari wheel ve sdist içinde de yeni bileşenleri içeren zincirlerle birer
update tamamlandı. Sequential automatic resume sınırını koruyan save/resume
isteklerinin reddi test edildi. ModularCNN otomatik epoch-resume sözleşmesi
genişletilmedi.

## JSON/API ve CLI uyumluluğu

ModelBuilder add/build/modular/named, preflight, config/JSON roundtrip ve
Registry list/get/validate/create girişleri korunur. Yeni kayıtlar schema1
sequential_classifier içinde kullanılır. Conv2d'nin eski kayıtları dilation=1
ile açılır. BatchNorm/GroupNorm eski default davranışları korunur. Eski Conv2d
state_dict anahtarları ayrıca sabit listeyle doğrulandı.

CLI models show metadata'sına ek alanlar gelir. Validate/summary başarılı
çıktılarında validation mevcut yerinde kalır. Geçersiz giriş code 2, başarılı
metadata code 0 ve diğer mevcut exit-code sözleşmeleri regresyon testlerinden geçti.
Duplicate JSON keys, plain/finite veri sınırları ve kapalı whitelist korunur.

## Ticari wheel ve sdist

Her iki arşiv açıldı ve Python dosyaları pozitif izin listesiyle birebir eşleşti.
Wheel 153 HYSo Python modülü, sdist aynı 153 modül ve setup.py içerir.
Her modül byte düzeyinde çalışma ağacındaki kaynakla karşılaştırıldı.
LICENSE ve THIRD_PARTY_NOTICES.md varlığı ve içeriği test edildi.

İki paket ayrı hedeflere `--no-index --no-deps --no-build-isolation` ile kuruldu.
Her kurulumda tüm 33 bağımsız Registry girdisi float32/float64 forward/backward
ve strict state_dict testinden geçti. ModularCNN ve eğitim/CLI yolları da
doğrulandı. Yeni 18 kayıt buna dahildir. Ticari Registry tam olarak 34 girdidir.

M3/M4 hazır mimariler, dedicated blokları ve geliştirme marker'ı arşivlere girmedi.
Dışlanan ailelerin Registry/Builder erişimleri ve Python import'ları başarısız
oldu. Ortam değişkeni ticari profili geliştirme profiline çevirmedi.

| Arşiv | SHA-256 |
|---|---|
| hyso-0.2.0-py3-none-any.whl | `f0c605cdab15a97d5d1182d6d16b92c0995e26ec1162c8efca0824c1ec85ab19` |
| hyso-0.2.0.tar.gz | `42cfd895f5cc9b3eba623d365534cbc734195fafa470f53b7f620529d7c7c26b` |

Yerel kanıt dizini `%LOCALAPPDATA%/hyso-stage13-5-2-20261113` altında final.xml,
exclusions.xml, validation-summary.json, protection.json ve dist arşivlerini içerir.

## Değiştirilen dosyalar

- `hyso/models/components.py` yeni operatör bileşimleri, scalar ilişkileri, geometry ve maliyet hesabı.
- `hyso/models/registry.py` yeni kayıtlar, metadata ve tanılamalar.
- `hyso/models/builder.py` geometri entegrasyonu ve genişletilmiş preflight raporu.
- `commercial-profile.json` yeni modül için tek pozitif izin listesi girdisi.
- `tests/test_builder_components.py` yeni kategorili CPU testleri.
- `tests/test_builder_validation.py` geçerli dilation desteğine ve artan Registry'ye uyarlanmış beklentiler.
- `tests/test_model_builder.py` yeni Registry toplamı.
- `tests/test_m4_architectures.py` yeni Registry toplamı, mevcut dışlama testi korunur.
- `tests/test_commercial_distribution.py` yeni kayıtlar dahil kurulu paket kontrolleri.
- `tests/test_packaging.py` modül envanteri ve Registry toplamları.
- `docs/guides/custom-cnn-builder-components.md` yeni bileşen rehberi.
- `docs/guides/custom-cnn-builder-validation.md` doğrulama genişletmeleri ve eski dilation açıklamasının güncellenmesi.
- `docs/guides/model-registry-builder.md` güncel katalog ve bileşen bağlantısı.
- `docs/reports/stage13-5-2-validation.md` bu rapor.

## Kaynak ve lisans kapsamı

Yeni components.py HYSo için bu aşamada yazıldı. Public PyTorch operator API'leri
ve matematiksel geometri sözleşmeleri kullanıldı. Hazır üçüncü taraf model
implementasyonu, yeni pretrained ağırlık veya veri kopyalanmadı. HYSo kodu root
MIT lisansı kapsamındadır. PyTorch bildirimleri mevcut THIRD_PARTY_NOTICES.md
içinde korunur. Yeni bir upstream model lisansı eklenmesini gerektiren kod alınmadı.
[Operator sözleşmeleri ve kaynak bağlantıları](../guides/custom-cnn-builder-components.md)

Önceki ticari risk değerlendirmesindeki patent, marka, ağırlık/veri hakları ve
dependency binary transitive koşullarına ilişkin değerlendirilmemiş alanlar sürer.
Bu aşama bütün kullanım senaryoları için hukuki güvence veya sıfır risk sonucu
üretmez. Ticari M3/M4 dışlama politikasında değişiklik yapılmadı.

## Koruma ve teslim durumu

Başlangıçtaki 634 tracked dosyanın SHA-256 kayıtları alındı. Yalnız yukarıdaki
görev kapsamındaki mevcut dosyalar değişti. Önceki 47 forensic/debug dosyası
**47/47**, native performance/benchmark/Trainer/resume/checkpoint kaynakları
**71/71**, üç checkpoint ve üç dış kabul/index kaydı **6/6** eşleşti.
P01–P35 kabul raporları ve kayıtlı 70/70 sonuç kanıtları korundu.

Git HEAD başlangıçla aynı, staged alan boş. Commit/push yapılmadı.
Çalışma ağacındaki görev değişiklikleri kullanıcı incelemesine hazırdır.

## Kalan sınırlar ve sonraki arayüz girdileri

Planlanan mekanizmalarda başarısız veya eksik kalan seçenek bulunmuyor.
Aktivasyon hesabı advisory ve belirtilen tensor kapsamıyla sınırlıdır.
Sembolik batch istatistik koşullarını her runtime girdisi için garanti etmez.
Sequential otomatik tam resume mevcut kapsamın dışındadır.

Aşama 13.5.3 tasarımı için Registry metadata'sındaki constraints,
parameter_dependencies ve common_errors alanları, preflight'ın ayrı hata/uyarı
listeleri ve katman geometry/maliyet alanları kullanılabilir. Bileşen eklemeleri
aynı kapalı specifications/validate/stages/parameter_count sözleşmesini izleyebilir.
Bu rapor sonraki aşama için arayüz girdilerini kaydeder. Aşama 13.5.3 veya kullanıcı
Graph/DAG sözleşmesi geliştirilmedi.
