# Aşama 13.5 için mevcut Builder girdileri

Bu belge Aşama 13-R sonunda kullanılabilen arayüzleri ve sonraki çalışma için
eksikleri kaydeder. Yeni bir Builder API'si tanımlanmadı.

## Ticari profil

Varsayılan wheel ve sdist, `commercial-profile.json` dosyasındaki 152 Python
modülünü içerir. Kaynak checkout geliştirme marker'ı sayesinde eski M3/M4
API'lerini ve testlerini korur. Kurulu ticari paket marker'ı içermez.
Profil ortam değişkeni veya mimari JSON alanıyla değiştirilemez.
Editable kurulum geliştirme kaynaklarını kullanır. Ticari teslim normal wheel
ve sdist kurulumu üzerinden doğrulanır.

| Registry girdisi | Mevcut girdiler | Çıktı ve sınırlar |
|---|---|---|
| conv2d | in_channels, out_channels, kernel_size, stride, padding, groups, bias | NCHW, grouped/depthwise groups üzerinden |
| conv_norm_act | Aynı spatial girdiler, norm, activation | BN/GroupNorm/None, SiLU/ReLU/None |
| residual | in_channels, out_channels, stride, norm, attention | İki 3×3, otomatik projection, SiLU sonrası toplama |
| se | channels, reduction | SiLU descriptor, sigmoid channel gate |
| cbam | channels, reduction, kernel_size | Channel ve spatial gate, tek spatial boyut |
| eca | channels, kernel_size | Odd Conv1d kernel, channel gate |
| coordinate | channels, reduction | H/W descriptor ve gate, internal BN |
| batch_norm | num_features | Native BatchNorm2d istatistik kuralları |
| group_norm | num_groups, num_channels | Bölünebilir channel/group sayıları |
| relu / silu / hardswish | Parametresiz | NCHW shape korunur |
| identity | Parametresiz | Aynı tensor |
| channel_shuffle | channels, groups | groups kanalları böler, permutation |
| dropblock2d | p, block_size | Eğitimde maskeleme, eval identity |
| modular_cnn | input_channels, num_classes, widths, repeats, strides, attention, norm | Mevcut configurable ModularCNN |

Toplam 16 girdinin 15'i bağımsız blok veya primitive, biri configurable modeldir.
SE/CBAM/ECA/Coordinate ve normalization sınıflarının mevcut parametre/state_dict
düzeni korunur. EMA ve loader-plan yardımcıları native Trainer'ın önceki import
yollarında bulunur. Kullanıcının kendi nn.Module modeli TrainingSession'a
doğrudan verilebilir.

## Korunan arayüzler

`ModelRegistry.list/get/validate/create`, `ModelBuilder.add/build/modular`,
`to_config/from_config`, `to_json/from_json` kullanılabilir. `named` metodu
ticari profilde kapsam dışı hazır aileler için açıklayıcı ModelConfigError üretir.
Schema1 sequential_classifier ve modular_cnn girişleri korunur.

Builder kapalı Registry üzerinden lineer NCHW zincir kurar. JSON plain ve finite
değerlerle sınırlıdır. Her boyut 1..8192, zincir en fazla 64 blok ve conservative
10M parametre hesabıyla doğrulanır. Residual içindeki skip/projection kendi blok
tarafından yönetilir. JSON weights, optimizer veya çalıştırılabilir Python taşımaz.

CLI `models list/show/validate/summary/inspect --execute` kurulu profilin
Registry'sini kullanır. Validation ve summary allocation başlatmaz. Açık inspect
CPU modeli oluşturur. Eğitim, reporting, Advisor, Manager ve P01–P35 arayüzleri
aynı giriş noktalarını kullanır.

## Sonraki aşama için eksikler

| Alan | Mevcut sınır | Tasarım girdisi |
|---|---|---|
| Graph bağlantıları | Lineer zincir ve blok içi residual | Kullanıcı tanımlı DAG, concat/add ve shape kuralları |
| Kullanıcı blokları | JSON kapalı whitelist, nn.Module doğrudan eğitimde kullanılabilir | Güvenli kayıt, schema ve provenance sözleşmesi |
| Katman seçenekleri | Sınırlı norm/activation/kernel parametreleri | Dilation, pooling, flatten, dropout ve daha geniş activation seçimi |
| Attention | Ticari profilde dört mevcut attention | SimAM/Triplet/SK kaynak/lisans değerlendirmesi ve ayrı modül kararları |
| Residual çeşitleri | HYSo ResidualBlock | Bağımsız bottleneck/preactivation için açık provenance ve blok sözleşmeleri |
| Mobile blokları | ChannelShuffle ve DropBlock | MBConv/Ghost/fusion gibi dışlanan operatorlerin ayrı değerlendirmesi |
| Serialization | Mimari JSON + strict state_dict | Yeni kullanıcı graph'ı için versioning ve uyumluluk |
| Automatic resume | Mevcut ModularCNN kapsamı | Kullanıcı graph'ının epoch/RNG/loader resume sözleşmesi |
| Bellek ve maliyet | Parametre tahmini | Aktivasyon belleği ve kaynak ölçümü için ayrı protokol |
| Dağıtım girdileri | Pozitif kaynak izin listesi | İzin listesi ve lisans kanıtı güncelleme prosedürü |

Bu maddeler geliştirme taahhüdü veya uygulanmış özellik değildir. Patent, marka,
ağırlık/veri hakları ve binary bağımlılık dağıtımı için genel onay verilmez.
Hukuki kapsam ve kaynak sınıfları
[risk denetiminde](../reports/stage13-commercial-risk-audit.md) açıklanır.
