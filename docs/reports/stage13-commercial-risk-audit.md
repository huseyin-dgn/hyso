# Aşama 13-R ticari kaynak ve risk denetimi

Tarih 2026-11-12. Başlangıç HEAD
`9ef0205210bb926b51441f81a8f5e969b36f158a`, branch `feature/hyso-models-next`.
Bu inceleme kullanıcının hazır üçüncü taraf CNN'leri ticari kapsamdan çıkarma
tercihini uygular. Kaynak lisansı, patent, marka, pretrained ağırlık, veri ve
binary bağımlılık dağıtımı ayrı değerlendirilir.

## Sınıflandırma ve kanıt sınırı

| Sınıf | Bu rapordaki anlam |
|---|---|
| OWNED_VERIFIED | İncelenen implementasyonun HYSo Git geçmişindeki ilk eklenişi, yazar kaydı, değişiklikleri ve MIT kapsamı izlenebilir. Doğrulama repository geçmişiyle sınırlıdır. Dışarıdan kopyalanmadığına veya bütün hakların münhasıran devredildiğine dair bağımsız hukuki tasdik değildir. |
| PERMISSIVE_VERIFIED | Belirlenen kaynak veya referans için izin verici lisans metni ve atıf kaydı mevcut. Diğer hukuki haklar ayrıca değerlendirilir. |
| UNRESOLVED | Kaynak eşlemesi, lisans kapsamı, karma binary bileşenler veya ürün dağıtım yükümlülükleri tam doğrulanmadı. |
| EXCLUDED | Ticari HYSo wheel/sdist kaynak kapsamı dışında. Çıkarma kararı hukuka aykırılık hükmü içermez. |

Hiçbir kategori sıfır patent riski veya ticari kullanım onayı sağlamaz.
OWNED_VERIFIED kaydı araştırma fikrinin sahipliğini HYSo'ya atfetmez.
İnceleme kaynak implementasyonunun eldeki repository kayıtlarına dayanır.

## Ticari paket ve geliştirme kaynakları

`commercial-profile.json` pozitif izin listesi 152 Python dosyası içerir.
`setup.py` içindeki build_py ve sdist komutları bu listeyi uygular.
Wheel doğrudan tam checkout'tan üretildiğinde de yalnız bu modüller kopyalanır.
Sdist aynı listeyle süzülür ve tekrar wheel üretmek için gerekli build dosyalarını
taşır. Eski M3/M4 kaynakları ve testleri checkout'ta geliştirme kapsamındadır.
Editable kurulum da checkout kaynaklarını kullanır ve geliştirme kapsamındadır.
Ticari teslim doğrulaması normal wheel ve sdist kurulumları için geçerlidir.

Geliştirme marker'ı `hyso/models/_development.py` ticari paketlerden çıkarılır.
`_profile.py` dosyası kurulu kaynak kümesini tanır. Ticari profil JSON veya ortam
değişkeniyle geliştirme profiline geçirilemez. Runtime'da eksik modeller için
sessiz fallback bulunmaz. Python kurulumuna yeni güvenilir kaynak dosyaları
eklenmesi kurulumun kodunu değiştirir ve bu denetimin kapsamını aşar.

Ticari sınır HYSo'nun wheel/sdist kaynak payload'ıdır. Ayrı kurulan torchvision
bağımlılığının kendi model kataloğu vardır. Bu işlem bağımlılıkların global Python
namespace'lerini yeniden yazmaz. HYSo'nun dışlanan wrapper/import yolları ticari
kurulumda bulunmaz. Bağımlılık paketleri HYSo arşivlerine vendored edilmez.

## Hazır aileler ve ayrılan modüller

| Aile / bileşen | Kaynak ve lisans kanıtı | Sınıf ve ticari durum | Gerekçe / belirsizlik | Builder |
|---|---|---|---|---|
| ResNet, ResNeXt, DenseNet | torchvision BSD-3-Clause adaptörleri | EXCLUDED | Kullanıcının ticari profil tercihi | Kapalı |
| PreActResNet, WideResNet | HYSo M3 kaynakları, makale topolojileri, MIT repository geçmişi | EXCLUDED | Hazır aile politikası | Kapalı |
| MobileNetV2/V3, EfficientNetV2, ShuffleNetV2 | torchvision BSD-3-Clause wrapper'ları | EXCLUDED | Hazır aile politikası | Kapalı |
| GhostNetV1 | Huawei Apache-2.0 ve directory-specific notice | EXCLUDED | Hazır aile politikası | Kapalı |
| RepVGG | DingXiaoH MIT | EXCLUDED | Hazır aile politikası ve kendine özgü fusion modülü | Kapalı |
| MobileOne | Apple özel lisansı ve RepVGG MIT acknowledgement | EXCLUDED | Hazır aile politikası | Kapalı |
| FasterNet, PartialConv | Önceki incelemede kapsamı tam belirlenemeyen MIT referansı | EXCLUDED | Önceki dışlama kararı korunur | Kapalı |
| M3 residual adaptörleri ve preactivation/wide blokları | M3 kaynakları ve torchvision referansları | EXCLUDED | Bu teslimde hazır aile modülleriyle aynı geliştirme kapsamı | Kapalı |
| InvertedResidual, MBConv, FusedMBConv, ShuffleUnit | torchvision adaptörleri | EXCLUDED | M4 ailesi modülü ticari profilden ayrıldı | Kapalı |
| GhostConv, GhostBottleneck | GhostNet kaynakları | EXCLUDED | Aileye özgü modül | Kapalı |
| RepVGGBlock, MobileOneBlock ve conversion API | M4 reparameterization kaynakları | EXCLUDED | Aileye özgü fusion ve phase yardımcıları | Kapalı |
| SimAM | M3 kodu ve SimAM makale/repository referansı | EXCLUDED | Referans implementasyonun lisans kapsamı UNRESOLVED | Kapalı |
| TripletAttention | landskape-ai MIT referansı, atıf bildirimi korundu | EXCLUDED | Karışık M3 attention dosyasının bu profil dışındaki kapsamı | Kapalı |
| SelectiveKernelAttention | M3 implementation ve makale referansı | EXCLUDED | Tam dış kaynak/değişiklik zinciri UNRESOLVED | Kapalı |

Ticari kapsamdan ayrılan dosyalar `architectures.py`, `m3_residual.py`,
`m3_attention.py`, `m4_blocks.py`, `m4_config.py`, `m4_registry.py`,
`m4_reparam.py`, `mobile_architectures.py`, `reparam_architectures.py` ve marker
`_development.py` dosyalarıdır. Hepsi `hyso/models/` altındadır.

Bu dosyalar kaynak checkout'ta bulunur. Eski örnekler ve testler geliştirme
profilini doğrular. Ticari arşivlerin içinde örnek/test kaynakları bulunmaz.
Önceki kaynak/lisans revizyonları
[Aşama 13 denetiminde](stage13-license-audit.md) kaydedilmiştir.

## Korunan ortak kod ve mevcut kullanıcı modelleri

| Bileşen | Kaynak geçmişi / lisans | Sınıf | Tutma gerekçesi ve kalan belirsizlik | Ticari / Builder |
|---|---|---|---|---|
| ConvNormAct, ResidualBlock | HYSo `b86cc76`, sonraki açık Git değişiklikleri, MIT | OWNED_VERIFIED | Configurable convolution/projection bileşimi. Repository dışı hak devri bağımsız tasdik edilmedi | Dahil / açık |
| SEBlock, BottleneckSEBlock, EMA | HYSo `0010bb2`, sonraki düzeltmeler, MIT | OWNED_VERIFIED | Paylaşılan SE ve native Trainer EMA bağımlılığı, önceki state düzeni | Dahil / SE açık, diğerleri mevcut standalone API |
| CBAMBlock, BottleneckResidualBlock | HYSo `baed0ec`, sonraki düzeltmeler, MIT | OWNED_VERIFIED | Eski configurable kullanıcı modelleri ve ortak attention | Dahil / CBAM açık, bottleneck standalone |
| ECAAttention, CoordinateAttention | HYSo `b5b669d`, makale referansları, upstream MIT bildirimleri | PERMISSIVE_VERIFIED | Kaynak motiflerinin atıfları ihtiyatlı şekilde korundu. Patent/marka hakları ayrı | Dahil / açık |
| ModularCNN | HYSo `d1521f3`, MIT | OWNED_VERIFIED | Kullanıcı parametreleriyle kurulan mevcut modüler sistem | Dahil / açık |
| ELO/EBO, CEMO, CNNSENormalWithFit | Yukarıdaki HYSo model kaynakları ve fit geçmişi, MIT | OWNED_VERIFIED | HYSo'ya özgü configurable kullanıcı modelleri. Adı verilen üçüncü taraf canonical ailelerle aynı implementasyon değiller. Git kaydı bağımsız sahiplik tasdiki içermez | Dahil / eski import API, Registry'ye yeni ekleme yok |
| ChannelShuffle, DropBlock2D | HYSo `9ef0205`, regularization.py içine aynı operator gövdeleri çıkarıldı | OWNED_VERIFIED | Permutation ve explicit masking, aile modülünden bağımsız | Dahil / açık |
| Conv2d, BatchNorm2d, GroupNorm, ReLU, SiLU, Hardswish, Identity | Mevcut PyTorch operatörleri ve upstream lisansları | PERMISSIVE_VERIFIED | Operator kaynak izinleri, kurulu binary bileşenler için ek inceleme sınırı | Dış bağımlılık / açık |
| Registry, Builder, BuiltClassifier | HYSo Aşama 7 ve sonraki Git kayıtları, MIT | OWNED_VERIFIED | Kapalı metadata ve schema1 zincir | Dahil / mevcut metotlar |
| TrainingConfig/Session, Manager, CLI, reporting, Advisor, P01–P35 | Mevcut HYSo Git kaynakları, MIT | OWNED_VERIFIED | Ortak ürün altyapısı | Dahil / mevcut API |

ELO/CEMO kaynaklarını hazır üçüncü taraf aileler kategorisine otomatik taşımak,
kullanıcının mevcut configurable modellerini ve native Trainer import zincirini
bozardı. İncelenen geçmişte bu kod HYSo yazarı tarafından MIT repository'ye
eklenmiş ve izlenebilir değişikliklerle sürdürülmüştür. Bu kanıt sınırı açıkça
kaydedilir. Dış kaynak sahiplik iddiası ortaya çıkarsa statü tekrar değerlendirilir.

EMA, loader-plan, SE ve CBAM import yolları aynı kaynak dosyalarındadır.
Native 71 kaynak dosyası üzerinde değişiklik yapılmadan ayrıştırma sağlandı.

## Bağımlılık ve bildirim kapsamı

Sekiz doğrudan runtime bağımlılığının Windows/Python 3.10 ortamındaki transitif
kapanışı 32 distribution içerir. Deployment extra için ek dört distribution
ayrıca kaydedildi. Envanter sürümleri, dependency edge'lerini, kurulu lisans
dosyalarını ve SHA256 değerlerini içerir. Kaynaklar ve tam metadata dış kanıt
dizininde `%LOCALAPPDATA%/hyso-stage13r-20261112/` altında saklanır.

PERMISSIVE_VERIFIED dependency satırları incelenen birincil lisans metinleriyle
sınırlıdır. Tek başına dosya varlığı onay ölçütü değildir. Binary paketlerin
bundle ettiği farklı lisanslı bileşenler, özel hükümler ve yeniden dağıtım
yükümlülükleri ayrıca ele alınır. PyTorch, NumPy, SciPy, Pillow, tqdm ve
torchsummary için ürün dağıtımı açısından açık sınırlar aşağıda belirtilir.
Bu bağımlılıkların dosyaları HYSo wheel/sdist içinde dağıtılmaz.

- PyTorch/NumPy/SciPy/Pillow kurulu binary paketleri ek lisans ve subcomponent
  metinleri taşır. Bu metinlerin varlığı uygunsuzluk kanıtı değildir. Tüm binary
  bileşenler için ürün düzeyinde kapsam incelemesi tamamlanmadığından toplam
  dağıtım statüsü UNRESOLVED olarak kaydedildi.
- tqdm için MPL-2.0 ve MIT hükümleri birlikte bulunur. MPL ticari kullanım yasağı
  değildir. Yeniden dağıtım/değişiklik yükümlülüklerinin ürün bağlamında
  değerlendirilmesi gerekir. Bu satır PERMISSIVE_VERIFIED sayılmadı.
- torchsummary upstream MIT metni bulundu ve korundu. Kurulu 1.5.1 distribution
  kendi lisans dosyasını sunmuyor. Kurulu binary/source eşlemesi bu oturumda tam
  kesinleştirilmedi. Statü UNRESOLVED olarak bırakıldı.
- torchvision ayrı bağımlılık olarak data transforms ve eski API'ler için kalır.
  Kendi model implementation'ları HYSo'nun izin listesiyle vendored edilmez.
  Üçüncü taraf distribution içeriğinin tamamını kaldırma iddiası yoktur.

Tüm ilk M4 lisans metinleri korunur ve geliştirme kapsamı işaretlenir. ECA,
Coordinate Attention, torchsummary ve Triplet kaynak bildirimleri eklenmiştir.
tqdm LICENCE metni ve karma lisans durumu ayrıca kaydedilmiştir.
[Tam bildirimler](../../THIRD_PARTY_NOTICES.md),
[ECA lisansı](https://github.com/BangguWu/ECANet/blob/b332f6b3e6e2afe8a3287dc8ee8440a0fbec74c4/LICENSE),
[Coordinate Attention lisansı](https://github.com/houqb/CoordAttention/blob/7619bea9acbe260b3793833cc78cef3f124c8112/LICENSE),
[tqdm lisansı](https://github.com/tqdm/tqdm/blob/0ed5d7f18fa3153834cbac0aa57e8092b217cc16/LICENCE).

## Kalan hukuki sınırlar

Bu teslim, HYSo kaynak payload'ı için teknik dağıtım profilini doğrular.
Binary bağımlılıkların beraber paketleneceği nihai ticari ürün için otomatik
onay verilmez. Patent araştırması, marka izni, sahiplik uyuşmazlığı incelemesi,
pretrained ağırlık lisansı ve veri kümesi hakları NOT_EVALUATED.
Git geçmişi veya izin verici kaynak lisansı bu alanlarda evrensel garanti sağlamaz.

## Symbol-level source inventory

Private helpers inherit the recorded source-file status and have no separate Registry entry. Repository provenance is the evidence boundary stated above.

| Symbol | Source / origin | License | Classification | Distribution / Builder | Remaining uncertainty |
|---|---|---|---|---|---|
| `validate_family` | `hyso/models/architectures.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `ResNet` | `hyso/models/architectures.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `ResNeXt` | `hyso/models/architectures.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `DenseNet` | `hyso/models/architectures.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `PreActResNet` | `hyso/models/architectures.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `WideResNet` | `hyso/models/architectures.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_positive_int` | `hyso/models/attention.py`, a5a3548 huseyin-dgn feat: add ECA and Coordinate Attention modules | HYSo MIT + conservative upstream MIT notices | PERMISSIVE_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_check_input` | `hyso/models/attention.py`, a5a3548 huseyin-dgn feat: add ECA and Coordinate Attention modules | HYSo MIT + conservative upstream MIT notices | PERMISSIVE_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ECAAttention` | `hyso/models/attention.py`, a5a3548 huseyin-dgn feat: add ECA and Coordinate Attention modules | HYSo MIT + conservative upstream MIT notices | PERMISSIVE_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `CoordinateAttention` | `hyso/models/attention.py`, a5a3548 huseyin-dgn feat: add ECA and Coordinate Attention modules | HYSo MIT + conservative upstream MIT notices | PERMISSIVE_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_resume_type` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_resume_transform_plan` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_resume_dataset_plan` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_resume_loader_plan` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_resume_plan` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_validate_resume_progress` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `CBAMBlock` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_find_group_count` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `stochastic_depth` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `BottleneckResidualBlock` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ELO` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `EBO` | `hyso/models/bcam.py`, baed0ec HÃ¼seyin DoÄŸan Implements CBAM and BottleneckResidualBlock layers, along with CNN models (ELO and EBO) and their building blocks. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_positive_int` | `hyso/models/blocks.py`, 7883601 huseyin-dgn feat: add reusable CNN and residual blocks | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_pair` | `hyso/models/blocks.py`, 7883601 huseyin-dgn feat: add reusable CNN and residual blocks | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ConvNormAct` | `hyso/models/blocks.py`, 7883601 huseyin-dgn feat: add reusable CNN and residual blocks | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ResidualBlock` | `hyso/models/blocks.py`, 7883601 huseyin-dgn feat: add reusable CNN and residual blocks | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_cost` | `hyso/models/builder.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_shape` | `hyso/models/builder.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_pairs` | `hyso/models/builder.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `BuiltClassifier` | `hyso/models/builder.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ModelBuilder` | `hyso/models/builder.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `model_summary` | `hyso/models/fit.py`, 23a9301 HÃ¼seyin DoÄŸan Provides training, validation, and reporting utilities for models like CNNSENormalWithFit. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `CNNSENormalWithFit` | `hyso/models/fit.py`, 23a9301 HÃ¼seyin DoÄŸan Provides training, validation, and reporting utilities for models like CNNSENormalWithFit. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `SimAM` | `hyso/models/m3_attention.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_TripletGate` | `hyso/models/m3_attention.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `TripletAttention` | `hyso/models/m3_attention.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `SelectiveKernelAttention` | `hyso/models/m3_attention.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `normalization` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_settings` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `BasicResidualBlock` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `Bottleneck` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `ResNeXtBottleneck` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `PreActivationBlock` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `WideResidualBlock` | `hyso/models/m3_residual.py`, 3c0bd5e huseyin-dgn feat(models): add M3 residual dense and attention architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `block_channels` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `expansion_ratio` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `odd_kernel` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `InvertedResidual` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `MBConv` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `FusedMBConv` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `GhostConv` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `GhostBottleneck` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `ShuffleUnit` | `hyso/models/m4_blocks.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `positive_scale` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `efficient_settings` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `rep_settings` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_conv` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_se` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_ghost` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_rep_block` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `validate_mobile` | `hyso/models/m4_config.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `specifications` | `hyso/models/m4_registry.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `validate_block` | `hyso/models/m4_registry.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `block_cost` | `hyso/models/m4_registry.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `block_shape` | `hyso/models/m4_registry.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_conv_bn` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_check_bn` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_ReparamBlock` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `RepVGGBlock` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `MobileOneBlock` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `reparameterize_model` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `ReparameterizableModel` | `hyso/models/m4_reparam.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `MobileArchitectureConfig` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_replace_input_conv` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `MobileNetV2` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `MobileNetV3` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `EfficientNetV2` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `ShuffleNetV2` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `GhostNet` | `hyso/models/mobile_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `_positive_int` | `hyso/models/modular.py`, adacdfb huseyin-dgn feat: add configurable ModularCNN architecture | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_stage_values` | `hyso/models/modular.py`, adacdfb huseyin-dgn feat: add configurable ModularCNN architecture | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_attention` | `hyso/models/modular.py`, adacdfb huseyin-dgn feat: add configurable ModularCNN architecture | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ModularCNN` | `hyso/models/modular.py`, adacdfb huseyin-dgn feat: add configurable ModularCNN architecture | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ModelConfigError` | `hyso/models/registry.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_plain` | `hyso/models/registry.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ParameterSpec` | `hyso/models/registry.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ComponentSpec` | `hyso/models/registry.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `_p` | `hyso/models/registry.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ModelRegistry` | `hyso/models/registry.py`, 36956c5 huseyin-dgn feat(api): integrate HYSo stages 2-6 workflows | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `probability` | `hyso/models/regularization.py`, Stage 13-R extraction / profile glue | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `ChannelShuffle` | `hyso/models/regularization.py`, Stage 13-R extraction / profile glue | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `DropBlock2D` | `hyso/models/regularization.py`, Stage 13-R extraction / profile glue | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `RepVGG` | `hyso/models/reparam_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `MobileOne` | `hyso/models/reparam_architectures.py`, 9ef0205 huseyin-dgn feat(models): add licensed M4 mobile and edge architectures | Development-source notices | EXCLUDED | development only / Closed | Patent/trademark and independent ownership attestation unverified |
| `EMA` | `hyso/models/WithSE.py`, 0010bb2 HÃ¼seyin DoÄŸan Includes CNNSENormal (CEMO) and EMA wrapper; manages SE (Squeeze-and-Excitation) blocks and model weight updates. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `SEBlock` | `hyso/models/WithSE.py`, 0010bb2 HÃ¼seyin DoÄŸan Includes CNNSENormal (CEMO) and EMA wrapper; manages SE (Squeeze-and-Excitation) blocks and model weight updates. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `BottleneckSEBlock` | `hyso/models/WithSE.py`, 0010bb2 HÃ¼seyin DoÄŸan Includes CNNSENormal (CEMO) and EMA wrapper; manages SE (Squeeze-and-Excitation) blocks and model weight updates. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |
| `CEMO` | `hyso/models/WithSE.py`, 0010bb2 HÃ¼seyin DoÄŸan Includes CNNSENormal (CEMO) and EMA wrapper; manages SE (Squeeze-and-Excitation) blocks and model weight updates. | HYSo MIT; Torch operators retain upstream terms | OWNED_VERIFIED | commercial and development / Existing Registry/API as documented | Patent/trademark and independent ownership attestation unverified |

## Installed runtime dependency inventory

Versions describe the audited local environment. Dependency binaries and their subcomponents are separate from the commercial HYSo source payload. SHA256 evidence for every discovered license file is stored in dependency-inventory.json outside the checkout.

| Distribution | Version | Source / license evidence | Classification | Remaining scope |
|---|---|---|---|---|
| torch | 2.5.1+cu121 | [source](https://pytorch.org/) / BSD-3-Clause | UNRESOLVED | Primary upstream license identified; mixed/bundled binary components or additional redistribution obligations remain outside a complete product-specific clearance. |
| torchvision | 0.20.1+cu121 | [source](https://github.com/pytorch/vision) / BSD | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| numpy | 2.2.2 | [source](https://github.com/numpy/numpy) / Copyright (c) 2005-2024, NumPy Developers. | UNRESOLVED | Primary upstream license identified; mixed/bundled binary components or additional redistribution obligations remain outside a complete product-specific clearance. |
| matplotlib | 3.10.8 | [source](https://github.com/matplotlib/matplotlib) / License agreement for matplotlib versions 1.3.0 and later | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| seaborn | 0.13.2 | [source](https://github.com/mwaskom/seaborn) / License :: OSI Approved :: BSD License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| scikit-learn | 1.7.2 | [source](https://github.com/scikit-learn/scikit-learn) / BSD-3-Clause | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| tqdm | 4.67.1 | [source](https://github.com/tqdm/tqdm) / MPL-2.0 AND MIT | UNRESOLVED | Primary upstream license identified; mixed/bundled binary components or additional redistribution obligations remain outside a complete product-specific clearance. |
| torchsummary | 1.5.1 | [source](https://github.com/sksq96/pytorch-summary) / UNKNOWN | UNRESOLVED | Upstream MIT reference retained. Exact installed 1.5.1 source mapping and absent installed license metadata remain unresolved. |
| filelock | 3.20.0 | [source](https://github.com/tox-dev/py-filelock) / Unlicense | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| typing-extensions | 4.15.0 | [source](https://github.com/python/typing_extensions) / PSF-2.0 | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| networkx | 3.4.2 | [source](https://github.com/networkx/networkx) / License :: OSI Approved :: BSD License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| jinja2 | 3.1.6 | [source](https://github.com/pallets/jinja/) / License :: OSI Approved :: BSD License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| fsspec | 2025.9.0 | Installed distribution metadata / BSD-3-Clause | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| sympy | 1.13.1 | [source](https://github.com/sympy/sympy) / BSD | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| pillow | 12.0.0 | [source](https://github.com/python-pillow/Pillow) / MIT-CMU | UNRESOLVED | Primary upstream license identified; mixed/bundled binary components or additional redistribution obligations remain outside a complete product-specific clearance. |
| contourpy | 1.3.2 | [source](https://github.com/contourpy/contourpy) / BSD 3-Clause License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| cycler | 0.12.1 | [source](https://github.com/matplotlib/cycler) / Copyright (c) 2015, matplotlib project | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| fonttools | 4.60.1 | [source](http://github.com/fonttools/fonttools) / MIT | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| kiwisolver | 1.4.9 | [source](https://github.com/nucleic/kiwi) / ========================= | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| packaging | 25.0 | [source](https://github.com/pypa/packaging) / License :: OSI Approved :: Apache Software License; License :: OSI Approved :: BSD License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| pyparsing | 3.2.5 | Installed distribution metadata / MIT | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| python-dateutil | 2.9.0.post0 | [source](https://github.com/dateutil/dateutil) / Dual License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| pandas | 2.3.3 | [source](https://github.com/pandas-dev/pandas) / BSD 3-Clause License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| scipy | 1.15.3 | [source](https://github.com/scipy/scipy) / Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers. | UNRESOLVED | Primary upstream license identified; mixed/bundled binary components or additional redistribution obligations remain outside a complete product-specific clearance. |
| joblib | 1.5.2 | [source](https://github.com/joblib/joblib) / BSD 3-Clause | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| threadpoolctl | 3.6.0 | [source](https://github.com/joblib/threadpoolctl) / BSD-3-Clause | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| colorama | 0.4.6 | Installed distribution metadata / License :: OSI Approved :: BSD License | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| markupsafe | 3.0.3 | [source](https://github.com/pallets/markupsafe/) / BSD-3-Clause | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| mpmath | 1.3.0 | [source](https://github.com/fredrik-johansson/mpmath) / BSD | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| six | 1.17.0 | [source](https://github.com/benjaminp/six) / MIT | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| pytz | 2025.2 | [source](http://pythonhosted.org/pytz) / MIT | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |
| tzdata | 2025.2 | [source](https://github.com/python/tzdata) / Apache-2.0 | PERMISSIVE_VERIFIED | Primary source grant reviewed; bundled binaries, patents and product distribution require separate review. |

## Optional deployment dependency inventory

These optional binaries were inventoried through installed metadata and were not used for a workload. Their product-level distribution status remains UNRESOLVED.

| Distribution | Version | Declared license | Status |
|---|---|---|---|
| onnx | 1.20.0 | Apache-2.0 | UNRESOLVED |
| onnxruntime | not installed | unknown | UNRESOLVED |
| protobuf | 6.32.1 | 3-Clause BSD License | UNRESOLVED |
| ml-dtypes | 0.5.4 | Apache-2.0 | UNRESOLVED |
