# Aşama 13.5.3 doğrulama raporu

13 Kasım 2026 tarihinde `feature/hyso-models-next` üzerinde gerçekleştirildi.
Başlangıç ve teslim HEAD değeri `f2cfc5e007d66b07251369b6d8664f9e3bd9f075`.

## Eklenen mekanizmalar

Ticari Registry **34 → 48**, geliştirme Registry **62 → 76** oldu. Eski 34 ticari
girdi korundu. Pozitif ticari kaynak listesine yalnız `hyso/models/advanced.py`
eklendi. Toplam Python modülü **154** oldu.

| Alan | Yeni Registry isimleri |
|---|---|
| Residual | bottleneck_residual, preactivation_residual, wide_residual_block, grouped_bottleneck, inverted_bottleneck, gated_residual |
| Attention | spatial_attention, channel_attention, spatial_self_attention, multihead_spatial_attention |
| Regularization | dropout, dropout2d, drop_path, gaussian_noise |

Mevcut residual, SE, CBAM, ECA, Coordinate Attention ve DropBlock2D implementasyonları
korundu. Yeni generic bloklar kendi `advanced.py` modülünde public torch
convolution, normalization, matmul, softmax ve rastgele sayı operatörleriyle yazıldı.
Hazır M3/M4 aileleri veya dedicated yardımcı modülleri import edilmedi.

Residual ortak operatör planı construction, geometri ve exact trainable parametre
hesabını besler. Kanal/stride değişiminde ham girdiye bias içermeyen 1×1 projection
uygulanır. Diğer durumda identity shortcut kullanılır. İç dal ve shortcut şekilleri
statik hesapta ve runtime toplamasından önce karşılaştırılır.

PreActivation iki K×K convolution öncesinde norm/activation kullanır, projection
ham girdiden hesaplanır. Wide ve gated iki spatial convolution kullanır.
Grouped bottleneck ortadaki convolution'ı explicit groups ile böler.
Inverted expansion, depthwise filtre ve projection kullanır. Gated residual
başlangıç gate değerini [0,1] aralığında kabul eder, öğrenilen skaler eğitim
sırasında bu aralıktan çıkabilir. Sabit sıfır gate dal katkısını kapatır.

Spatial attention mean/max kanal descriptor'larından spatial sigmoid kapısı,
channel attention mean/max spatial descriptor'larından paylaşılan MLP ile kanal
kapısı üretir. Self-attention spatial konumları token olarak işler. Explicit
embedding projection sonrasında output projection aynı C kanalına döner.

DropPath residual içinde yalnız ana dala, standalone kullanımda kendisine verilen
tensorun tamamına örnek başına survival scaling uygular. GaussianNoise train
modunda bağımsız sıfır ortalamalı Gaussian gürültüsü ekler. İkisi de eval modunda
identity olur. Gürültü input'u yerinde değiştirmez ve CPU generator kabul eder.

Her bileşenin matematiği, parametreleri, varsayılanları, örnek input/output şekli,
hata düzeltmeleri ve kaynak sınırları ileri bloklar rehberinde açıklanır.
[İleri bloklar rehberi](../guides/custom-cnn-builder-advanced-blocks.md)

## Preflight ve attention güvenliği

Önceki **14 hata/uyarı kodu** korunur. **ATTENTION_RESOURCE_LIMIT** eklendi.
Scalar ve normalization hatalarında mevcut INVALID_PARAMETER, ara şekillerde
INTERMEDIATE_SHAPE_ERROR, grup/head bölünebilirliğinde GROUP_DIVISIBILITY_ERROR
kullanılır. Öneriler gerçek bölenlerden hesaplanır. Her diagnostic mevcut code,
message_tr, message_en, layer_index, layer_id, expected, actual ve suggestion
alanlarını taşır.

Self-attention için **1024 token** ve **4.194.304 attention matris elemanı**
zorunlu sınırları seçildi. Token sayısı L=H*W, matris hesabı N*heads*L² olur.
Preflight açık batch_size varsayımıyla hesaplar. Runtime gerçek batch ile QKV
projection başlamadan önce aynı sınırı uygular. Sınırın batch yüzünden aşılması
da reddedilir. Softmax/dropout kopyaları veya çalışma alanı nedeniyle gerçek peak
bellek bu eleman sayısından büyük olabilir. Sınırlar peak bellek garantisi vermez.

Varsayılan embedding_dim giriş C değeridir. Explicit embedding seçilebilir.
Embedding head sayısına bölünmelidir. E en fazla 2730 olur, QKV ara C=3E böylece
8192 sınırını aşmaz. channels=60, num_heads=8 ve varsayılan embedding reddedilir.
Hata gerçek geçerli head bölenlerini verir.

Yeni katman raporları exact parameter_count, conservative parameter_estimate,
ara operator shape'leri, aktivasyon elemanları, projection, residual_active ve
attention_cost alanlarını içerir. Normalization ve projection parametreleri
hesaba dahildir. Bilinmeyen parametre maliyetleri sıfır gibi gösterilmez.
Eski karmaşık bloklarda mevcut None/existing conservative hesap sözleşmesi sürer.

Mevcut 10M model parametre bütçesi ve CHW 1..8192 sınırı korunur. Eski advisory
aktivasyon eşiği geçerli önceki modellerin kabulünü korur. İleri blokların norm
ve aktivasyon sınırları conservative toplamda yer alır. Norm istatistik koşulları
warning olarak raporlanır ve gerçek runtime girdisinde ayrıca denetlenir.

## CPU test sonuçları

**537 benzersiz test başarılı, 0 başarısız, 0 atlanan.** Birleşik koşu 532 test
içerir. Beş yeni sınır/rastgelelik vakası eklendikten sonra ileri blokların 124
testi tekrar geçti. Tekrarlanan vakalar toplam sayıya ikinci kez eklenmedi.
Son metadata düzeltmesinden sonra altı commercial/packaging testi yeniden geçti.

| Modül | Benzersiz başarılı vaka |
|---|---|
| test_builder_advanced | 124 |
| test_builder_components | 227 |
| test_model_builder | 42 |
| test_builder_validation | 61 |
| test_cli | 76 |
| test_commercial_distribution | 4 |
| test_packaging seçili canonical/content testleri | 2 |
| test_m4_architectures seçili dışlama testi | 1 |

Tüm 14 yeni bileşen için başarılı ve hatalı yapılandırma testleri bulunur.
Float32 ve float64 CPU forward/backward, finite input/parameter gradient,
optimizer step, train/eval, JSON roundtrip ve strict state_dict doğrulandı.
Her yeni blokta gerçek parameter sayısı preflight hesabıyla eşleşti.

Residual testleri altı tür için identity/projection shortcut, stride 1 ve asimetrik
2/3, dilation 2/3 ve batch/group/None normalization seçeneklerini kapsar.
İç ECA/spatial/channel attention seçenekleri ayrıca shape ve parametre hesabıyla
karşılaştırıldı. PreActivation operator sırası ve sabit sıfır gate identity
sonucu doğrulandı.

Self-attention tek head explicit token cebiriyle eşleşti. Head bölünebilirliği,
token sınırının tam değeri, yüksek çözünürlük, batch kaynaklı matris aşımı,
QKV'den önce runtime reddi ve dropout train/eval tekrarlanabilirliği doğrulandı.
Allocation fence testlerinde nn.Module constructor ve torch.empty engellendi.
CUDA sorguları ve initialization testlerde engellendi.

Regularization testleri global CPU seed, Gaussian CPU generator, eval identity,
input'un korunması, DropPath survival scaling, Dropout2d kanal maskesi ve native
dropout p=0/1 uçlarını kapsar. İç normalization istatistik hataları layer_index
ve açıklayıcı STATISTICS_REQUIREMENT koduyla iletildi.

İstenen `[3,64,64]` zinciri Bottleneck 32→64, ECA, pooling, Inverted 64→64,
DropPath, ChannelAttention ve GlobalAveragePool üzerinden çalıştı. İki sentetik
örnekle tek epoch/tek optimizer update TrainingSession smoke tamamlandı.
Sequential automatic epoch resume kapsamı genişletilmedi.

Mevcut Trainer'ın GradScaler API kullanımından deprecation uyarıları görüldü.
Native motor kaynakları bu görevde değiştirilmedi. CUDA workload, native
benchmark veya uzun eğitim başlatılmadı.

## JSON, CLI ve ticari dağıtım

Schema1 sequential_classifier aynı name/params düzenini kullanır. ModelBuilder
add/build/modular/named, preflight, config/JSON ve Registry list/get/validate/create
girişleri korunur. Eski state_dict anahtarları yeniden adlandırılmadı.
Mimari JSON callable/import path/class/eval kabul etmez.

CLI models validate/summary/inspect testleri ve mevcut 76 CLI regresyonu geçti.
Yeni models show metadata'sı Türkçe açıklama, train/eval bilgisi, bağımlılıklar,
maliyet sınırları ve düzeltme önerileri içerir. Exit-code sözleşmesi korunur.

Wheel ve sdist gerçek içerikleri açılarak 154 modüllük izin listesiyle ve kaynak
byte'larıyla karşılaştırılır. Her iki paket ayrı hedeflere dependencies indirilmeden
kurulur. Kurulu paketlerde 47 bağımsız bileşen float32/float64 forward/backward,
strict state_dict ve preflight shape testinden geçer. ModularCNN ve TrainingSession
smoke da doğrulanır. Yeni residual ve multihead attention zinciri kurulu paket
TrainingSession testine dahildir.

Önceki M3/M4 ailelerinin Registry/Builder erişimleri, dedicated modül import'ları
ve public aile export'ları ticari kurulumlarda dışlanmaya devam eder. LICENSE ve
THIRD_PARTY_NOTICES.md korunur. Yeni bağımlılık veya üçüncü taraf model kodu alınmadı.

Yerel kanıt dizini `%LOCALAPPDATA%/hyso-stage13-5-3-20261113` olur. XML test
kayıtları, source protection kaydı, distribution receipt ve dist arşivleri burada
saklanır. Arşiv SHA-256 değerleri distribution-final.json içinde bulunur.

## Değiştirilen dosyalar

- `hyso/models/advanced.py`
- `hyso/models/registry.py`
- `hyso/models/builder.py`
- `commercial-profile.json`
- `tests/test_builder_advanced.py`
- `tests/test_builder_components.py`
- `tests/test_builder_validation.py`
- `tests/test_model_builder.py`
- `tests/test_m4_architectures.py`
- `tests/test_commercial_distribution.py`
- `tests/test_packaging.py`
- `docs/guides/custom-cnn-builder-advanced-blocks.md`
- `docs/guides/custom-cnn-builder-components.md`
- `docs/guides/custom-cnn-builder-validation.md`
- `docs/guides/model-registry-builder.md`
- `docs/reports/stage13-5-3-validation.md`

Mevcut testlerde katalog sayıları, metadata sözleşmesi ve ticari smoke örnekleri
güncellendi. Eski mekanizma testleri korundu. Yeni testler ayrı advanced modülündedir.

## Bütünlük ve sınırlar

Başlangıçtaki 638 tracked dosyanın SHA-256 kaydı alındı. Önceki **47/47** forensic
dosyası, **71/71** native performance/benchmark/Trainer/resume/checkpoint kaynağı,
üç checkpoint ve üç dış kabul/index kaydı **6/6** eşleşti. Kapsam dışındaki tracked
dosyalar başlangıç byte'larıyla karşılaştırılır. P01–P35 kabul kanıtları korunur.
HEAD başlangıç değeriyle aynı ve staged alan boş kalır. Commit/push yapılmadı.

Planlanan mekanizmalarda başarısız veya uygulanmamış seçenek bulunmuyor.
Attention sınırları tek attention işlemi içindir. Modelin toplam peak belleği,
autograd kayıtları, operator workspace ve optimizer durumu için garanti verilmez.
Öğrenilmiş gate aralığı eğitim sırasında sınırlanmaz. Normalization'ın runtime
istatistik koşulları geçerli kalır.

Yeni kod root MIT kapsamındaki HYSo implementasyonudur. Mevcut dependency
lisans bildirimleri korunur. Patent, marka, veri/ağırlık hakları ve dependency
binary transitive koşullarına ilişkin önceki hukuki belirsizlikler devam eder.
Sıfır risk veya evrensel ticari kullanım garantisi verilmez.

Graph Builder tasarımına external add/concat shape kuralları, branch yaşam süresi
ve toplam kaynak hesabı, kararlı katman kimlikleri, kaynak sınırlarının her dalda
uygulanması ve serialization sürümleme gereksinimleri aktarılabilir. Bu görevde
genel DAG veya dış skip bağlantısı arayüzü geliştirilmedi.
