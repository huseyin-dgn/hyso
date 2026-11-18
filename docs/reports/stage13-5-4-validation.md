# Aşama 13.5.4 doğrulama raporu

13 Kasım 2026 tarihinde `feature/hyso-models-next` üzerinde tamamlandı.
Başlangıç ve teslim HEAD değeri `03fc5b3cf2e7ba743df7832e2836c2b279b3f4af`.

## Mekanizmalar ve public API

Ticari Registry **48 → 57**, geliştirme Registry **76 → 85**, ticari Python
izin listesi **154 → 155** oldu. Önceki 48 ticari kayıt korundu.

| Yeni kayıt | Sözleşme |
|---|---|
| resize | Nearest, bilinear, bicubic veya area. Output size ve scale factor hedeflerinden tam biri gerekir. |
| pixel_shuffle | C/r², H*r, W*r. C, r²'ye bölünür. |
| pixel_unshuffle | C*r², H/r, W/r. H/W, r'ye bölünür. |
| flatten | NCHW → NF, yalnız schema2 terminal feature konumu. |
| permute | N/C eksenleri sabit, identity veya H/W değişimi. |
| reshape | Explicit pozitif CHW, örnek başına eleman sayısı ve N korunur. |
| classification_head | Global avg/max veya flat features, optional dropout ve linear logits. |
| regression_head | Global avg/max veya flat features, sınırsız linear çıktı. |
| embedding_head | Global avg/max veya flat features, linear projection ve optional güvenli L2 normalization. |

Strided convolution mevcut conv2d stride parametresini kullanır. Duplicate
convolution implementasyonu eklenmedi. Trainable parametre sayısı sampling ve
utility işlemlerinde sıfır, head'lerde `(in_features+1)*output_features` olur.
Ara/çıktı boyutları 1..8192 ve toplam conservative model parametre bütçesi 10M
olarak korunur. Sampling ve explicit head aktivasyonları advisory hesaba dahildir.

## Schema1 ve schema2

`ModelBuilder((C,H,W),num_classes)` schema1 sequential_classifier üretmeye devam
eder. Eski JSON dosyası schema1 olarak yeniden kaydedilir. Default Conv2d,
AdaptiveAvgPool2d(1), flatten ve Linear sırası native referansla aynı seed altında
sayısal olarak eşleşti. `features.*` ve `classifier.weight/bias` anahtarları korundu.

Yeni `ModelBuilder.sequential(input_shape, head='classification'|'regression'|'embedding',
**head_params)` opt-in schema2, kind='sequential' üretir. Head ayrı name/params
kaydındadır. Builder in_features değerini feature chain'den hesaplar. Bağımsız
Registry head constructor'ında in_features açıkça verilir. Otomatik migration
yapılmaz. Schema2 state_dict `head.linear.weight/bias` anahtarlarını kullanır.

Head yalnız terminal kayıtta bulunur. Blocks içine head ekleme INVALID_BLOCK_ORDER
üretir. Yanlış head türü, boyutu, explicit Builder in_features veya pooling/rank
uyuşmazlığı INVALID_OUTPUT_HEAD üretir. Preflight ayrı head raporunda shape,
parametre sayısı ve aktivasyon uyarılarını gösterir. Yeni hata kodu eklenmedi.
Mevcut 15 hata/uyarı kodu korunur.

Flatten yalnız son schema2 feature bloğunda ve pooling='none' head ile çalışır.
Arkasına feature bloğu eklenemez. Permute N/C eksenlerini taşımaz. Reshape yalnız
explicit üç pozitif CHW boyutu kabul eder. -1 inference, genel rank geçişleri ve
NHWC routing sonraki Graph Builder sözleşmesine aktarılır. Bu sınırlamalar test
ve metadata'da açıkça belirtilir.

## Sampling parity ve input validation

Interpolation float32/float64, küçük/asimetrik H/W, explicit size ve fractional
scale ile native F.interpolate değerleri ve input gradient'leriyle birebir eşleşti.
Scale output shape her eksende floor(input*scale) olur. Nearest/area align_corners
None gerektirir. Bilinear/bicubic None/False/True varyantları test edildi.
Recompute-scale ve antialias Registry'ye eklenmedi.

PixelShuffle/Unshuffle faktör 1/2/3 roundtrip, değer, gradient ve statik shape
testlerini geçti. Geçersiz kanal/spatial bölünmesi, boş/oversized output, yanlış
mode, yanlış target kombinasyonu ve bool/non-finite değerler reddedildi.
Non-finite plain-data hatalarında mevcut belge düzeyindeki layer_index=None
sözleşmesi geçerlidir. Geçerli katman verisi üzerindeki shape hataları indekslidir.

Permute/reshape gerçek tensor değerleriyle ve eleman sayısıyla karşılaştırıldı.
Batch taşıyan veya kanalı NHWC konumuna götüren permutation, yinelenen eksen,
eksik/fazla elemanlı reshape ve yanlış flatten konumları reddedildi.

Embedding sıfır norm testi finite sıfır çıktı ve finite gradient verdi. Etkili eps
dtype.tiny ile birlikte korunur. Çok küçük explicit eps büyük gradient üretebilir.
Regression çıktısında negatif sabit değer testi otomatik sigmoid uygulanmadığını
doğruladı.

## Test sonuçları

**638 benzersiz test başarılı, 0 çözülmemiş başarısız, 0 atlanan.** Son CPU koşusu
630 başarılı ve iki installed-preflight testi deselected olarak tamamlandı.
Sampling/packaging koşusu 107 başarılı vaka içerir. İki installed-preflight vakası
birleşik koşuda geçti. Tekrarlar unique sayıya ikinci kez eklenmedi.

| Modül | Benzersiz başarılı test |
|---|---|
| test_builder_sampling_heads | 101 |
| test_builder_advanced | 124 |
| test_builder_components | 227 |
| test_model_builder | 42 |
| test_builder_validation | 61 |
| test_cli | 76 |
| test_commercial_distribution | 4 |
| test_packaging seçili canonical/content testleri | 2 |
| test_m4_architectures seçili dışlama testi | 1 |

Ön koşularda yeni test dosyasındaki decorator yazımı, non-finite girdinin layer
indeksi beklentisi ve eski 76 Registry toplamını bekleyen bir regresyon testi
düzeltildi. Sonraki koşularda bu vakalar geçti. Mevcut Trainer GradScaler API'sinden
deprecation uyarıları görüldü. Native motor kaynakları değiştirilmedi.

Preflight allocation fence testlerinde Module constructor ve torch.empty
engellendi. CUDA initialization ve sorguları CPU testlerinde engellendi.
Yeni head'ler float32/float64 forward, backward, finite gradient, optimizer step,
train/eval, JSON roundtrip ve strict state_dict kontrollerini geçti.
Önceki residual/projection, attention limitleri ve regularization testleri geçti.

## TrainingSession ve CLI

Schema2 classification bilinen classifier olarak TrainingSession ile tek epoch,
iki sentetik örnek ve bir optimizer update tamamladı. Schema1 classification
regresyonları da geçti. Regression ve embedding açık MSE loss/SGD döngüleriyle
optimizer update testinden geçti. TrainingSession'ın mevcut sınıflandırma
loss/metric varsayımları nedeniyle bilinen schema2 regression/embedding modelleri
prepare sırasında açıklayıcı TrainingConfigError ile reddedilir.

Sequential automatic full resume kapsamı mevcut ModularCNN koşullarıyla sınırlı
kaldı. CLI models show/validate/summary/inspect, schema2 head bilgisi ve mevcut
exit code değerleriyle doğrulandı. Schema1/schema2 alan karışımı ve duplicate
JSON keys code 2 üretti. JSON executable code veya weight taşımaz.

## Ticari dağıtım ve kaynak geçmişi

Wheel ve sdist açılarak 155 HYSo Python modülünün kaynak byte'larıyla eşleşmesi
doğrulandı. Sdist ek olarak setup.py içerir. Pozitif izin listesine yalnız yeni
HYSo sampling.py eklendi. Her iki arşiv ayrı hedeflere dependencies indirilmeden
kuruldu. Yeni sampling/utility katmanları ile üç head türü kurulu paketlerde
float32/float64, gradient, optimizer ve strict state_dict kontrollerini geçti.
Önceki 47 bağımsız blok ve ModularCNN/training/CLI kontrolleri korundu.

M3/M4 hazır aileleri, development marker ve dedicated yardımcı modüller ticari
arşivlerde ve kurulu import/Registry yollarında dışlanmaya devam eder. LICENSE ve
THIRD_PARTY_NOTICES.md korundu. Yeni kod root MIT kapsamında, public PyTorch
operatör sözleşmelerinden HYSo için yazıldı. Üçüncü taraf hazır model kodu,
pretrained ağırlık veya yeni bağımlılık alınmadı.

Yerel kanıt dizini `%LOCALAPPDATA%/hyso-stage13-5-4-20261113` altında XML koşu
kayıtları, test-summary-final.json, distribution-final.json, protection.json,
integrity-final.json ve dist arşivleri bulunur. Arşiv hash'leri distribution
receipt içinde kaydedilir.

## Değiştirilen dosyalar

- `hyso/models/sampling.py`
- `hyso/models/registry.py`
- `hyso/models/builder.py`
- `hyso/cli/models.py`
- `hyso/training/session.py`
- `commercial-profile.json`
- `tests/test_builder_sampling_heads.py`
- `tests/test_builder_advanced.py`
- `tests/test_builder_components.py`
- `tests/test_builder_validation.py`
- `tests/test_model_builder.py`
- `tests/test_m4_architectures.py`
- `tests/test_commercial_distribution.py`
- `tests/test_packaging.py`
- `docs/guides/custom-cnn-builder-sampling-heads.md`
- `docs/guides/custom-cnn-builder-validation.md`
- `docs/guides/custom-cnn-builder-components.md`
- `docs/guides/model-registry-builder.md`
- `docs/reports/stage13-5-4-validation.md`

## Koruma ve kalan sınırlar

Başlangıçtaki 642 tracked dosya kaydedildi. Önceki **47/47** forensic/debug,
**71/71** native kaynak, üç checkpoint ve üç dış kabul/index kaydı **6/6**
SHA-256 ile eşleşti. Kapsam dışındaki tracked dosyalar başlangıç byte'larıyla
karşılaştırıldı. P01–P35 kabul kanıtları korundu. HEAD başlangıç değeriyle aynı,
staged alan boş. Commit/push yapılmadı.

İstenen sampling ve head seçeneklerinde eksik kalan mekanizma bulunmuyor.
Tensor utilities yukarıdaki güvenli rank/layout kapsamıyla çalışır. Genel rank
graph, external add/concat/skip, detection ve segmentation head geliştirilmedi.
Sampling workspace ve gerçek peak bellek için garanti verilmez. Regression ve
embedding TrainingSession loss/metric desteği bu aşamanın dışında kalır.
Patent, marka, ağırlık/veri hakları ve dependency binary transitive koşullarına
ilişkin önceki hukuki belirsizlikler devam eder.
