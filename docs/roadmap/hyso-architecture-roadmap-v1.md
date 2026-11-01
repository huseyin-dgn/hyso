# HYSo Mimari Roadmap v1.0 — Sabit Kapsam

Tarih: 2026-10-19. Kaynak HEAD: `757efc985822cc4b254d528ef6c173e28faafbbc`,
branch `feature/hyso-models-next`; başlangıç çalışma ağacı temiz.
Bu belge kullanıcı talimatıyla **sabitlenmiş geliştirme kapsamıdır**.
Roadmap v1.0, paket sürümü değildir; HYSo metadata'sı 0.2.0 olarak kalır.

97 başlık, 97 bağımsız sınıf/model anlamına gelmez. İç içe kavramlar ve ortak
PyTorch işlemleri tekrar implement edilmez. Her satırın sabit kimliği, durumu,
fazı ve kod kanıtı/eksik kapsamı aşağıdadır. `Implemented` yalnız belirtilen
mevcut kapsamın kodda bulunduğunu gösterir; tüm varyant/platformların doğrulandığı
anlamına gelmez. `Planned` başlığın tamamlanmadığını gösterir; mevcut alt parçalar
not edilir. Bir primitive'in PyTorch'ta bulunması, HYSo'da kullanıldığının kanıtı değildir.

## Sabit fazlar ve sıralama

| Faz | Kapsam | Durum |
|---|---|---|
| P0 | Training Core | COMPLETE |
| M1 | Modular CNN, temel blocks, ECA, Coordinate Attention | GO |
| M2 | Generic Trainer, Benchmark, Experiment Management | Planned |
| M3 | Residual CNN aileleri, DenseNet, SimAM, Triplet, SK Attention | Planned |
| M4 | Mobile/Edge CNN, MBConv, GhostConv, RepVGG, MobileOne, FasterNet, DropBlock | Planned |
| M5 | ConvNeXt V2, DynamicCNN, RepViT, FastViT, GC, LKA | Planned |
| M6 | CSPDarknet, C2f, C3k2, FPN, BiFPN, ASPP, U-Net, DeepLabV3+, detection/segmentation interfaces | Planned |
| M7 | ONNX, quantization, pruning, distillation, edge deployment | Planned |

M2 sırası: önce M2-A tasarım ve M2-B1–B5 Trainer doğrulaması; Benchmark ve
Experiment Manager ayrı sonraki işlerdir. Normalization/activation/augmentation
gibi ortak alt parçalar aşağıdaki eşlemelerde ilk ihtiyaç duyulan faza bağlanır;
faz tablosu dışında yeni bir geliştirme fazı yaratılmaz.

P0/M1 tamamlanma durumu son M1-E raporuna dayanır: 720 PASS/1 SKIP, 17 legacy
CUDA, 18 M1 CUDA, gerçek Torch 2.2.2 installed-wheel beş attention. Bunlar bu
belge yazılırken yeniden çalıştırılmış testler değildir; exact profiller ve
determinism sınırları [M1-E raporunda](../audit/m1-e-final-architecture-quality-gate.md).

## A. CNN model mimarileri — 23

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| A01 | ELO (CBAM CNN) | Implemented | P0, M1 | hyso/models/bcam.py: ELO |
| A02 | CEMO (SE CNN) | Implemented | P0, M1 | hyso/models/WithSE.py: CEMO |
| A03 | ModularCNN | Implemented | M1 | hyso/models/modular.py: ModularCNN |
| A04 | ResNet | Planned | M3 | Ayrı ResNet ailesi yok |
| A05 | PreAct ResNet | Planned | M3 | Ayrı pre-activation aile yok |
| A06 | WideResNet | Planned | M3 | Kanal ayarı, WideResNet implementasyonu sayılmaz |
| A07 | ResNeXt | Planned | M3 | Ayrı grouped-bottleneck ailesi yok |
| A08 | DenseNet | Planned | M3 | Dense connectivity ailesi yok |
| A09 | MobileNetV2 | Planned | M4 | Ayrı model yok |
| A10 | MobileNetV3 | Planned | M4 | Ayrı model yok |
| A11 | EfficientNetV2 | Planned | M4 | Ayrı model yok |
| A12 | GhostNet | Planned | M4 | Ayrı model yok |
| A13 | ShuffleNetV2 | Planned | M4 | Ayrı model yok |
| A14 | FasterNet | Planned | M4 | Ayrı model yok |
| A15 | RepVGG | Planned | M4 | Model ve deploy dönüşümü yok |
| A16 | MobileOne | Planned | M4 | Model ve deploy dönüşümü yok |
| A17 | ConvNeXt V2 | Planned | M5 | Ayrı model yok |
| A18 | DynamicCNN | Planned | M5 | Ayrı model yok; yeni dinamik config framework'ü varsayılmaz |
| A19 | RepViT | Planned | M5 | Ayrı model yok |
| A20 | FastViT | Planned | M5 | Ayrı model yok |
| A21 | CSPNet / CSPDarknet | Planned | M6 | Tek kapsam satırı; iki bağımsız sınıf zorunluluğu yok |
| A22 | U-Net | Planned | M6 | Segmentation model/interface yok |
| A23 | DeepLabV3+ | Planned | M6 | Segmentation model/interface yok |

EBO, ELO'nun training wrapper'ıdır; CNNSENormalWithFit, CEMO'nun eğitim subclass'ıdır.
EMA da CNN değildir. Korunurlar fakat bu 23 model satırına ek model olarak sayılmazlar.

## B. Attention mekanizmaları — 9

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| B01 | SE | Implemented | P0, M1 | hyso/models/WithSE.py: SEBlock |
| B02 | CBAM | Implemented | P0, M1 | hyso/models/bcam.py: CBAMBlock |
| B03 | ECA | Implemented | M1 | hyso/models/attention.py: ECAAttention |
| B04 | Coordinate Attention | Implemented | M1 | hyso/models/attention.py: CoordinateAttention |
| B05 | SimAM | Planned | M3 | Modül yok |
| B06 | Triplet Attention | Planned | M3 | Modül yok |
| B07 | SK Attention | Planned | M3 | Modül yok |
| B08 | Global Context (GC) Block | Planned | M5 | Modül yok |
| B09 | Large Kernel Attention (LKA) | Planned | M5 | Modül yok |

## C. Residual / Bottleneck blokları — 12

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| C01 | Basic Residual Block | Implemented | M1, M3 | hyso/models/blocks.py: ResidualBlock; iki conv/SiLU, özgün ResNet ailesi iddiası yok |
| C02 | ResNet Bottleneck | Planned | M3 | Mevcut BottleneckResidualBlock/BottleneckSEBlock attention'lı farklı tasarımlar; kanonik ResNet bottleneck yok |
| C03 | Pre-activation Residual Block | Planned | M3 | Ayrı pre-activation block yok |
| C04 | Wide Residual Block | Planned | M3 | WideResNet block sözleşmesi yok |
| C05 | ResNeXt Grouped Bottleneck | Planned | M3 | Grouped conv primitive'i var; bu bottleneck yok |
| C06 | Inverted Residual / MBConv | Planned | M4 | Mevcut daraltıcı bottleneck, expansion MBConv değildir |
| C07 | Ghost Bottleneck | Planned | M4 | Block yok |
| C08 | RepVGG Block | Planned | M4 | Block/fusion yok |
| C09 | MobileOne Block | Planned | M4 | Block/fusion yok |
| C10 | CSP Bottleneck | Planned | M6 | Block yok |
| C11 | C2f Block | Planned | M6 | Block yok |
| C12 | C3k2 Block | Planned | M6 | Block yok |

## D. Convolution / Feature-Mixing — 14

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| D01 | Standard Convolution | Implemented | P0, M1, M3 | ConvNormAct ve legacy nn.Conv2d kullanımı |
| D02 | Pointwise Convolution | Implemented | P0, M1, M4 | BottleneckSEBlock.conv1/conv_pw, projection 1×1 conv'lar |
| D03 | Grouped Convolution | Implemented | M1, M3 | ConvNormAct(groups=...), bölünebilirlik doğrulaması |
| D04 | Depthwise Convolution | Implemented | P0, M4 | BottleneckSEBlock.dwconv, groups=mid_channels |
| D05 | Depthwise Separable Convolution | Implemented | P0, M4 | bcam.BottleneckResidualBlock(use_depthwise=True).dw_seq; WithSE depthwise/pointwise hattı |
| D06 | Dilated / Atrous Convolution | Planned | M3, M6 | Mevcut block API'lerinde dilation seçimi yok; PyTorch'un desteklemesi HYSo kullanımı sayılmaz |
| D07 | Ghost Convolution | Planned | M4 | İşlem yok |
| D08 | Partial Convolution (PConv) | Planned | M4 | İşlem yok |
| D09 | Dynamic Convolution | Planned | M5 | İşlem yok |
| D10 | Coordinate Convolution (CoordConv) | Planned | M5 | Coordinate Attention, koordinat-kanalı convolution değildir |
| D11 | RepConv / Structural Reparameterization | Planned | M4, M7 | Training/deploy dönüşümü yok |
| D12 | Large-Kernel Depthwise Convolution | Planned | M5 | Mevcut depthwise k3; large-kernel block yok |
| D13 | Channel Shuffle | Planned | M4 | İşlem yok; DataLoader shuffle bu kavram değildir |
| D14 | RepMixer | Planned | M5 | İşlem yok |

## E. Normalization — 6

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| E01 | BatchNorm2d | Implemented | P0, M1, M3 | Legacy modeller, ConvNormAct, CoordinateAttention; PyTorch primitive'i kullanılır |
| E02 | GroupNorm | Implemented | P0, M1, M3 | bcam norm factory ve ConvNormAct(norm='group') |
| E03 | LayerNorm2d | Planned | M5 | ELO.fc_norm nn.LayerNorm'ı flatten sonrası vektöre uygular; LayerNorm2d yok |
| E04 | InstanceNorm2d | Planned | M3, M6 | Kullanım/API yok |
| E05 | Global Response Normalization (GRN) | Planned | M5 | İşlem yok |
| E06 | Weight Standardization | Planned | M3 | İşlem yok |

## F. Activation — 6

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| F01 | ReLU | Implemented | M1, M3 | ConvNormAct(activation='relu') nn.ReLU kullanır |
| F02 | LeakyReLU | Planned | M3 | Kullanım/API yok |
| F03 | PReLU | Planned | M3 | Kullanım/API yok |
| F04 | SiLU / Swish | Implemented | P0, M1, M4 | Legacy F.silu/nn.SiLU ve ConvNormAct; tek kapsam satırı |
| F05 | GELU | Planned | M5 | Kullanım/API yok |
| F06 | HardSwish | Implemented | M1, M4 | CoordinateAttention.activation nn.Hardswish; genel block seçicisi henüz sunmaz |

## G. Regularization — 4

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| G01 | Dropout | Implemented | P0, M3 | ELO.head ve CEMO.dropout_fc nn.Dropout |
| G02 | SpatialDropout2D | Planned | M3, M6 | nn.Dropout2d kullanımı/API yok |
| G03 | DropBlock2D | Planned | M4 | İşlem yok |
| G04 | DropPath / Stochastic Depth | Implemented | P0, M5 | bcam.stochastic_depth ve BottleneckSEBlock.stochastic_depth; farklı RNG/ölçekleme semantiği korunur, ortak modül oldukları iddia edilmez |

## H. Multi-Scale — 4

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| H01 | FPN | Planned | M6 | Multi-scale head yok |
| H02 | BiFPN | Planned | M6 | Multi-scale head yok |
| H03 | ASPP | Planned | M6 | İşlem/head yok |
| H04 | SPPF | Planned | M6 | İşlem yok |

## I. Data Augmentation — 7

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| I01 | Random Crop / Resize | Implemented | P0, M3 | augmentation.build_transforms: Resize, advanced RandomResizedCrop; her crop varyantı iddiası yok |
| I02 | Horizontal Flip | Implemented | P0, M3 | build_transforms basic RandomHorizontalFlip |
| I03 | ColorJitter | Implemented | P0, M3 | build_transforms basic ColorJitter |
| I04 | Random Erasing | Implemented | P0, M3 | build_transforms advanced RandomErasing; existing EBO safe-resume transform whitelist'inde değil |
| I05 | Cutout | Planned | M3 | İşlem yok; Random Erasing ile aynı başlık sayılmaz |
| I06 | Mixup | Planned | M3 | Batch mixing/soft-target training sözleşmesi yok |
| I07 | CutMix | Planned | M3 | Batch mixing/soft-target training sözleşmesi yok |

## J. Destekleyici sistemler — 12

| Kimlik | Başlık | Durum | Faz | Kod kanıtı / tamamlanacak kapsam |
|---|---|---|---|---|
| J01 | Generic Trainer | Planned | M2, M6 | EBO.fit generic değildir; M2-A tasarımı var, implementasyon M2-B |
| J02 | Checkpoint / Resume | Implemented | P0, M1, M2, M6 | checkpoint.save/load_training_checkpoint v1, EBO epoch-resume; ModularCNN component round-trip mevcut, generic Trainer epoch-resume henüz yok |
| J03 | Dataset Manager | Planned | M2, M6 | get_loaders/build_transforms mevcut; ayrı dataset manager/task lifecycle yok |
| J04 | Experiment Manager | Planned | M2 | Ortak experiment persistence/management yok |
| J05 | Metrics System | Planned | M2, M6 | EBO/CEMO loss/accuracy ve sklearn raporları var; ortak task metrics sistemi yok |
| J06 | Benchmark Engine | Planned | M2, M7 | Smoke/process süresi ve peak bellek, benchmark engine değildir |
| J07 | Model Builder | Planned | M3, M4, M5, M6 | ModularCNN constructor'ı composition yapar; aileler arası model builder henüz yok; registry zorunlu değildir |
| J08 | Visualization | Implemented | P0, M2 | EBO/CEMO/CNNS validation confusion matrix ve matplotlib/seaborn çıktıları; bağımsız dashboard/engine yok |
| J09 | ONNX Export | Planned | M7 | Export/validation pipeline yok |
| J10 | Quantization / Pruning / Distillation | Planned | M7 | Tek kapsam satırı; üç alt iş ayrı doğrulanacak |
| J11 | CPU / GPU Tests and CI | Planned | P0, M1, M2, M3, M4, M5, M6, M7 | CPU/GPU testleri mevcut ve M1-E GO; .github/workflows tabanlı CI yok, birleşik başlık henüz tamamlanmış değil |
| J12 | Documentation / Model Cards | Planned | P0, M1, M2, M3, M4, M5, M6, M7 | README/audit/release notes mevcut; model cards ve aile bazlı dokümantasyon eksik |

## Sayım ve çapraz faz kuralları

| Kategori | Sayı |
|---|---:|
| A / CNN | 23 |
| B / Attention | 9 |
| C / Residual | 12 |
| D / Convolution | 14 |
| E / Normalization | 6 |
| F / Activation | 6 |
| G / Regularization | 4 |
| H / Multi-Scale | 4 |
| I / Augmentation | 7 |
| J / Sistemler | 12 |
| Toplam | **97** |

Her ID tek kategoriye ve en az bir faza bağlıdır. Çoklu faz eşlemesi aynı işi
yeniden yazmak anlamına gelmez: checkpoint, data, metrics, test/CI ve documentation
önce mevcut kapsamı koruyup sonraki task/device/aile ihtiyaçlarıyla genişler.
J11/J12'nin Planned olması mevcut testleri/belgeleri yok saymaz; CI/model-card
alt kapsamları eksik olduğu için birleşik başlık tamamlanmış sayılmamıştır.

Mevcut EBO/ELO/CEMO/CNNS/EMA API'leri, public export sırası, legacy state_dict,
checkpoint v1, EBO resume, M1 bileşenleri ve metadata/dependency sözleşmesi
koruma sınırıdır. Yeni aileler aynı işlev için mevcut primitive'leri yeniden kullanır.

## Freeze ve değişiklik yönetimi

Bu v1.0 listesi sabittir. Yeni öneri, framework veya başka model ailesi otomatik
eklenmez. Kapsam ekleme/çıkarma veya kapsamı değiştiren yeniden adlandırma için
**açık v1.1 değişikliği** gerekir: gerekçe, etkilenen ID'ler/fazlar, sayım ve
compatibility/test etkileri kaydedilir, kullanıcı kararı alınır.
Implemented durumuna geçiş için gerçek kod ve test kanıtı güncellenir;
yalnız plan yazılması implementasyon sayılmaz.

Bu dokümantasyon görevinde runtime regression çalıştırılmadı. Sayım/ID/status/
faz kontrolleri statik olarak yürütüldü; yeni production PASS iddiası yok.
Generic Trainer tasarımı: [M2-A](../audit/m2-a-generic-trainer-design.md).
