# Aşama 13 M4 doğrulama raporu

Çalışma tarihi 2026-11-12. Başlangıç branch'i `feature/hyso-models-next`,
HEAD `3c0bd5eca5953163d4e00760add41e8ffdc0592e`.

İlk sekiz aile hedefinin yedisi uygulanıp doğrulandı. Kullanıcının öncelikli
hukuki uygunluk talimatı üzerine FasterNet T0/T1/T2 ve PartialConv dağıtım
kapsamından çıkarıldı. Bu iki bileşenin lisans kapsamı kesinleştirilemedi.
Başlangıçtaki sekiz ailelik hedefin tamamı için tamamlanma iddiası yoktur.
İzin kapsamı belgelenen bileşenlerin teknik teslimi aşağıda kaydedilmiştir.
Ticari kullanıma hukuki onay ve patent serbestisi doğrulanmadı.
[Lisans denetimi](stage13-license-audit.md),
[tam üçüncü taraf bildirimleri](../../THIRD_PARTY_NOTICES.md).

## Başlangıç envanteri ve yeniden kullanım

M3 raporu, `hyso/models`, convolution/residual/attention kaynakları,
Registry/Builder schema1, TrainingSession/Trainer sınırları, CLI model komutları,
paketleme testleri ve örnekler incelendi. Başlangıç Registry'sinde 27 bileşen
bulunuyordu. Başlangıçta staged alan boştu ve önceki 47 dosyanın SHA256 değerleri
mevcut kayıtlarla eşleşti.

| Mevcut bileşen | Kullanım |
|---|---|
| torchvision MobileNetV2/V3, EfficientNetV2, ShuffleNetV2 | Canonical gövdeler kontrollü wrapper ile kullanıldı |
| torchvision InvertedResidual, MBConv, FusedMBConv, ShuffleNetV2 unit | Bağımsız kanal/stride doğrulamalı adaptörler |
| torchvision Conv2dNormActivation, SqueezeExcitation | GhostNetV1 ve Ghost bloklarının Conv/BN/aktivasyon ve hard-gated SE işlemleri |
| PyTorch Hardswish, Hardsigmoid, BN, Conv2d, pooling | Mevcut operatörler üzerinden bileşim |
| PyTorch fuse_conv_bn_eval | RepVGG/MobileOne Conv–BN fusion hesabı |
| HYSo SEBlock | Opsiyonel MobileOneBlock sigmoid SE, reduction 16 |
| Registry, Builder ve CLI | Mevcut kapalı metadata/JSON/CPU oluşturma yolları |
| TrainingSession | Standart dış nn.Module için mevcut kısa CPU eğitim yolu |

M3 mimari, attention ve residual kaynakları, ELO/CEMO, ModularCNN ve mevcut
custom CNN blokları byte düzeyinde korundu. Eski public wildcard export listesi
beş isim içerir. Yeni sınıflar explicit import ile kullanılır.

## Aileler ve varyantlar

| Aile | Doğrulanan varyantlar / mimari |
|---|---|
| MobileNetV2 | Canonical width 1, özgün yedi stage dizilimi, ReLU6, inverted residual ve linear bottleneck |
| MobileNetV3 | Small ve Large, farklı stage dizilimleri, SE/HardSigmoid/HardSwish/ReLU seçimleri |
| EfficientNetV2 | S/M/L, özgün FusedMBConv/MBConv repeat dizileri, SE ve stochastic depth, BN eps .001 |
| GhostNet | V1, 16 Ghost bottleneck, depthwise downsample, hard-gated SE ve 1280 kanal post-pool head |
| ShuffleNetV2 | x0.5/x1.0/x1.5/x2.0, `[4,8,4]` repeat, channel split/concat/shuffle |
| RepVGG | A0/A1, `[2,4,14,1]` stage repeat ve açık train/deploy topolojileri |
| MobileOne | s0/s1/s2, `[2,8,10,1]` depthwise/pointwise çiftleri, s0 için dört stage dalı, tek dallı stem |
| FasterNet | T0/T1/T2 ve PartialConv çıkarıldı, kaynak/config lisans kapsamı belirsiz |

Desteklenen 16 canonical varyant için gerçek CPU oluşturma ve 32×32 eval forward
çalıştırıldı. Bu kontroller varsayılan stage/width/depth topolojilerini bir
giriş kanalı ve üç sınıflı başlıkla kullanır. Bütün ailelerde farklı giriş
kanalı ve sınıf sayısı, küçük width
konfigürasyonları, float32/float64 forward/backward ve finite gradient doğrulandı.
EfficientNetV2 M/L ve büyük canonical modeller için tam backward çalıştırılmadı.
Genişlik/derinlik azaltılmış modeller açık türetilmiş konfigürasyonlardır.
Parametreler, yuvarlama kuralları ve sınırlamalar
[mimari rehberde](../guides/mobile-edge-architectures.md) açıklanmıştır.

## Bağımsız bloklar

`InvertedResidual`, `MBConv`, `FusedMBConv`, `GhostConv`, `GhostBottleneck`,
`ChannelShuffle`, `ShuffleUnit`, `RepVGGBlock`, `MobileOneBlock`, `DropBlock2D`
eklendi. Her blok için giriş/çıkış sözleşmesi, Builder shape/maliyet hesabı,
float32/float64 gradient ve strict state_dict roundtrip test edildi.

Stride ve kanal uyuşmazlığı, groups bölünebilirliği, kernel sınırları, expansion,
SE oranı ve olasılıklar doğrulanır. ChannelShuffle sırası doğrudan referans
permutasyonla karşılaştırıldı. ShuffleUnit stride-one dal bileşimi kontrol edildi.
Ghost primary/cheap concat ve kırpma referans çıktıyla karşılaştırıldı.
InvertedResidual expansion oranı bire çok yakınken oluşan ek projection öncesi
katmanın parametre hesabı için özel regresyon eklendi.

DropBlock eğitimde bitişik kare maske uygular. Küçük ve çift spatial boyutlar,
block clipping, p=0/p=1, eval identity, tüm elemanların maskelenmesi ve finite
normalization/gradient test edildi. Float16/bfloat16 mask uygulamasındaki
istatistikler float32 üzerinden hesaplanır. DropBlock modellerde açıkça seçilir.

## Registry, Builder, CLI ve eğitim

Registry toplamı **44 bileşen**. Önceki 27 bileşene yedi aile ve on blok eklendi.
FasterNet/PartialConv için Registry, export ve Builder reddi ayrıca test edildi.

Schema1 `registered_model` whitelist genişletildi. Eski sequential/modular
sözleşmeleri regresyonlarla doğrulandı. Metadata validation ve summary sırasında
constructor çağrısı yasaklanarak allocation yapılmadığı kontrol edildi. Unknown
alan, mimari adı, weights parametresi ve aşırı parametre bütçeleri reddedildi.
Named modellerdeki 200M ve sequential zincirdeki 10M parametre sınırları
aktivasyon belleği ve RAM kapasitesinden ayrı değerlendirilir.

Yedi aile için CLI list/show/validate/summary ve açık inspect --execute
kontrolleri geçti. Inspect CPU modeli oluşturur. Forward ve conversion metadata
komutlarında çalıştırılmaz. CLI kaynakları ve exit-code/JSON sözleşmeleri korundu.

Her aile TrainingConfig/TrainingSession ile iki sentetik örnek üzerinde bir SGD
optimizer adımı tamamladı. Float32/float64 küçük modellerde finite gradient ve
değişen ağırlıklar doğrulandı. Automatic full-resume kapsamı ModularCNN ile
sınırlıdır. Yeni ailelerde save_last_to isteğinin reddi ayrıca test edildi.
TrainingConfig, TrainingSession, Trainer, ExperimentManager ve reporting
kaynaklarının SHA256 değerleri başlangıçla eşleşti.

## Fusion ve serialization

RepVGG ve MobileOne farklı dal topolojilerini ortak fusion yardımcılarıyla işler.
Eval BN running mean/variance, gamma/beta, epsilon, Conv bias, grouped identity
kernel ve 1×1 padding hesaba katılır. Referans testlerde BN değerleri ve epsilon
değiştirilerek varsayılan identity BN durumunun ötesi kontrol edildi.

Kopya conversion kaynak modeli korur. In-place conversion bütün blok planları
başarılı olduktan sonra grafiği değiştirir. İkinci conversion idempotenttir.
Train modu, running statistics eksikliği, negatif variance, uyumsuz geometri ve
float16 conversion reddedilir. Geçerli fusion dtype kapsamı float32/float64'tür.
Device/dtype, aktivasyon ve varsa SE konumu korunur. Donanımda GPU fusion
doğrulaması yapılmadı.

Train state_dict dalları ile deploy `reparam.weight/bias` anahtarları farklıdır.
Yanlış aşamanın ağırlıkları strict=False durumunda da reddedilir. Aile loader'ı
ağırlık kopyalamadan önce bütün blokların aşamasını kontrol eder. Deploy JSON
güncel deploy=True durumunu taşır. Aynı davranış sequential Builder içindeki
convert edilmiş bloklar için de doğrulandı.

Model mimari JSON ve weights-only strict state_dict ayrı kaydedilip yüklenir.
Bu yöntem optimizer, RNG, loader veya epoch-resume durumu taşımaz.
GhostNet/RepVGG/MobileOne anahtar düzeni HYSo'ya özgüdür. Canonical torchvision
wrapper anahtar/shape ve eval eşdeğerliği dört temsilci ailede karşılaştırıldı.
Pretrained uyarlaması, tam model pickle, ONNX/TorchScript ve mobil runtime
deployment doğrulaması bu teslimin sınırları arasındadır.

## Örnekler ve belgeler

`examples/models/m4_workflows.py` on iki kısa CPU senaryosu sağlar. MobileNetV2,
MobileNetV3 Small/Large, EfficientNetV2, GhostNetV1, ShuffleNetV2, RepVGG deploy,
MobileOne deploy, MBConv/FusedMBConv, DropBlock, Builder ve JSON/state_dict
roundtrip senaryolarının tamamı çalıştırıldı. Amaç, parametre, komut, beklenen
çıktı ve teknik sınırlar examples/models/README.md içinde verilmiştir.

M4 rehberi convolution türlerini, mimari farkları, kaynakları, deploy phase
kurallarını ve serialization sınırlarını açıklar. M3 rehberine yönlendirme eklendi.
THIRD_PARTY_NOTICES.md tam lisans ve atıf metinlerini korur. Lisans denetimi her
aile/blok için kaynak repository, immutable revizyon, dosya, lisans kapsamı ve
HYSo kullanım konumlarını kaydeder.

## Test sonuçları

**1.376 benzersiz pytest testi ve 146 alt test başarılı.** Sonuçlar
`(classname, name)` kimliğiyle birleştirildi. Tekrar çalıştırılan Builder ve
kurulu paket testleri toplamda bir kez sayıldı. Çözümlenmemiş failure/error
sayısı **0**, son doğrulama durumunda skipped sayısı **0**.

| Kanıt dosyası | Sonuç |
|---|---|
| `m4-final.xml` | 210 passed, 168 yeni M4 testi ve 42 Builder regresyonu |
| `regression-final.xml` | 1.201 passed, 1 optional saved-P34 skip, 7 deselected, 146 alt test |
| `p34-saved.xml` | Önceki optional test mevcut JSON/CSV ile ayrıca çalıştırıldı, 1 passed |
| `packaging-final.xml` | 6 passed, 2 ağır installed native smoke deselected |
| `installed-m4.xml` | İki mevcut installed test M4 kontrolleriyle genişletildi, 2 passed |

Yeni test dağılımı 91 blok, 64 aile/entegrasyon ve 13 örnek/typing testidir.
Regresyonlar M3, Builder, CLI, TrainingConfig/Session, eski attention/modular
modeller, ELO/CEMO, state compatibility, native execution, reporting, Advisor,
comparison ve ExperimentManager workflow yollarını kapsar. P01–P35 için mevcut
saved native kanıtlarının 39 offline okuma testi ayrıca geçti. Bu okumalar
donanım kabulünü yeniden yürütmedi.

İlk örnek turunda iki senaryoda hatalı `expansion` anahtar adı görüldü.
Örnekler public `expand_ratio` parametresiyle düzeltildi ve son turda geçti.
İlk sekiz aile için alınmış eski test çıktıları nihai benzersiz sayıya katılmadı.
FasterNet/PartialConv eski testleri dağıtılan özellik kanıtı olarak kullanılmadı.

Son geniş seçimin dışında tutulan dokuz test vakası, gerçek native CPU ölçümleri,
isolated worker, genel README workload örneği ve iki kapsamlı installed native
smoke yoludur. Kapsam dışı testler için seçim ifadesi
`not real_four_cpu_adapters and not insufficient_p01_budget and not failure_does_not_hide
and not uninitialized_binding and not common_isolated_real_owned_cpu_worker
and not readme_examples` ve packaging için
`not installed_distribution_import_and_cpu_forward` kullanıldı. Kurulu paket
kontrolünün son tekrarındaki dört deselection daha önce başarıyla çalıştırılmış
testlerin seçim dışı kalmasıdır.

CPU test ortamı Python 3.10.18, torch 2.5.1+cu121 ve torchvision 0.20.1+cu121.
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`CUDA_VISIBLE_DEVICES=-1`, `MPLBACKEND=Agg` kullanıldı. Pytest
`-c tests/pytest.ini -q -p no:cacheprovider --tb=short` seçenekleriyle çalıştı.
Model testleri deterministik sentetik veri ve tek CPU thread kullanır.
Mevcut Trainer'daki GradScaler deprecation uyarısı devam eder.

## Sayısal fusion sonuçları

| Kapsam | Dtype | En büyük mutlak fark | En büyük norm-relative fark | atol / rtol |
|---|---|---|---|---|
| 16 blok vakası, grouped/stride/kernel çeşitleri | float32 | 1.1920928955078125e-6 | 3.858383139915159e-7 | 2e-6 / 2e-5 |
| Aynı 16 blok vakası | float64 | 2.6645352591003757e-15 | 5.875581249082976e-16 | 3e-12 / 3e-12 |
| Küçük RepVGG tam model | float32 | 7.450580596923828e-9 | Blok ölçümleri yukarıda | 2e-6 / 2e-5 |
| Küçük RepVGG tam model | float64 | 1.3877787807814457e-17 | Blok ölçümleri yukarıda | 3e-12 / 3e-12 |
| Küçük MobileOne tam model | float32 | 4.470348358154297e-8 | Blok ölçümleri yukarıda | 2e-6 / 2e-5 |
| Küçük MobileOne tam model | float64 | 3.122502256758253e-17 | Blok ölçümleri yukarıda | 3e-12 / 3e-12 |

Norm-relative değer `max(abs(actual-reference))/max(abs(reference))` hesabıdır.
Elementwise kabul `torch.testing.assert_close` ile yukarıdaki atol/rtol üzerinden
yapıldı. Tam model ölçümlerinde max_abs ve uygulanan atol/rtol ayrıca kaydedildi.
`fusion-blocks-final.json` 32 blok kaydını, aile/dtype JSON dosyaları dört tam
model kaydını içerir. BN affine=False ve Conv bias içeren ek referans testi geçti.

## Paketleme ve kurulu dağıtım

Wheel ve sdist checkout dışında üretildi. Her iki arşivdeki **159 Python
modülünün tamamı** güncel kaynak byte'larıyla eşleşti. Python 3.10 AST denetimi
159 modülü kapsadı. Public M4 typing ifadeleri çözümlendi. Yeni bağımlılık
gereksinimi eklenmedi.

İki paket ayrı target dizinlerine offline ve no-deps ile kuruldu. Checkout
dışında isolated Python ile HYSo kaynağının kurulu target'tan geldiği kontrol
edildi. Kurulu CLI yedi yeni aileyi listeledi. Her aile küçük CPU forward ve
JSON/state_dict roundtrip gerçekleştirdi. RepVGG/MobileOne deploy conversion ve
strict reload kontrolleri kurulu paketlerde de geçti. Gerçek console entry point
ve `python -m hyso` kontrolleri ayrıca başarılıydı. CUDA context başlatılmadığı
kurulu smoke sonunda doğrulandı.

THIRD_PARTY_NOTICES.md hem wheel license metadata'sında hem sdist içinde bulunur.
Paketleme testi bütün bildirim byte'larını karşılaştırır. Ağırlık, checkpoint,
veri kümesi ve forensic dosyaları dağıtım arşivine alınmadı.

Kalıcı teslim kopyaları `%LOCALAPPDATA%/hyso-stage13-20261112/dist/` içindedir.

| Dağıtım | Boyut | SHA256 |
|---|---|---|
| `hyso-0.2.0-py3-none-any.whl` | 701266 byte | `dda31853ddeb53783086f893ee4827dc3b2ea449a3911d0936f3cd91ecce76b8` |
| `hyso-0.2.0.tar.gz` | 611859 byte | `0892f99a86b1e63cebd45d016f4e5705f1a97e8176001e57100e21028616e7d9` |

## Bütünlük, Git ve sınırlar

Başlangıç envanterindeki 654 dosya SHA256 ile karşılaştırıldı. Planlanan dokuz
mevcut dosya dışında başlangıç dosyalarının bütünlüğü korundu. Önceki 47
forensic/debug/ilgisiz dosyanın tamamı eski SHA256 değerleriyle eşleşti. Native
performance/benchmark ve Trainer/resume/checkpoint kaynaklarından oluşan 71
dosya başlangıçla eşleşti. Üç checkpoint ve üç harici kabul/index dosyası için
**6/6 SHA256 eşleşmesi** sağlandı. Mevcut 70/70 kabul kanıtı korundu.

Branch `feature/hyso-models-next`, son HEAD
`3c0bd5eca5953163d4e00760add41e8ffdc0592e`. Staged alan boş.
Çalışma ağacında önceki 47 dosya ile bu görevin 23 dosyası bulunur.
Değişiklikler kullanıcı incelemesi için çalışma ağacında bırakıldı.

Donanım hızı, FPS, enerji, GPU inference/fusion ve yeni donanım kabul durumu
**NOT_EVALUATED**. Gerçek veri başarımı, pretrained kalite, export backend ve
ticari dağıtımın patent/marka değerlendirmesi ayrıca kanıt gerektirir.
FasterNet/PartialConv lisans kapsamının çözülmesi veya uygun bağımsız kaynak
sürecinin kurulması açık teknik/hukuki sınırdır. Automatic epoch-resume yeni
aileler için kapsam dışıdır.

Kanıt dizini `%LOCALAPPDATA%/hyso-stage13-20261112/`. `protection.json` başlangıç
hash'lerini, `integrity-final.json` son karşılaştırmayı, `test-summary-final.json`
benzersiz sayımı ve `distributions-final.json` arşiv denetimini içerir.
Lisans kaynakları ve inceleme dışına çıkarılan eklemelerin kopyaları aynı dış
kanıt dizinindedir. Çalışma Aşama 13 sınırında durduruldu.

## Dosya kapsam?

Bu g?revin 23 dosyas? a?a??dad?r.

| Dosya | Durum |
|---|---|
| [MANIFEST.in](../../MANIFEST.in) | Güncellendi |
| [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) | Yeni |
| [docs/guides/mobile-edge-architectures.md](../../docs/guides/mobile-edge-architectures.md) | Yeni |
| [docs/guides/model-architectures.md](../../docs/guides/model-architectures.md) | Güncellendi |
| [docs/reports/stage13-license-audit.md](../../docs/reports/stage13-license-audit.md) | Yeni |
| [docs/reports/stage13-m4-validation.md](../../docs/reports/stage13-m4-validation.md) | Yeni |
| [examples/models/README.md](../../examples/models/README.md) | Güncellendi |
| [examples/models/m4_workflows.py](../../examples/models/m4_workflows.py) | Yeni |
| [hyso/models/__init__.py](../../hyso/models/__init__.py) | Güncellendi |
| [hyso/models/builder.py](../../hyso/models/builder.py) | Güncellendi |
| [hyso/models/m4_blocks.py](../../hyso/models/m4_blocks.py) | Yeni |
| [hyso/models/m4_config.py](../../hyso/models/m4_config.py) | Yeni |
| [hyso/models/m4_registry.py](../../hyso/models/m4_registry.py) | Yeni |
| [hyso/models/m4_reparam.py](../../hyso/models/m4_reparam.py) | Yeni |
| [hyso/models/mobile_architectures.py](../../hyso/models/mobile_architectures.py) | Yeni |
| [hyso/models/registry.py](../../hyso/models/registry.py) | Güncellendi |
| [hyso/models/reparam_architectures.py](../../hyso/models/reparam_architectures.py) | Yeni |
| [setup.cfg](../../setup.cfg) | Güncellendi |
| [tests/test_m4_architectures.py](../../tests/test_m4_architectures.py) | Yeni |
| [tests/test_m4_blocks.py](../../tests/test_m4_blocks.py) | Yeni |
| [tests/test_m4_examples.py](../../tests/test_m4_examples.py) | Yeni |
| [tests/test_model_builder.py](../../tests/test_model_builder.py) | Güncellendi |
| [tests/test_packaging.py](../../tests/test_packaging.py) | Güncellendi |

## Korunan 47 dosyanın SHA256 kaydı

Her satırın son SHA256 değeri başlangıç ve önceki koruma kaydıyla eşleşti.

| Dosya | SHA256 |
|---|---|
| `.vscode/settings.json` | `fb8d616967cb453d8e0e455e922b6350c7bc2475afc429d03c3c6afc8f63cd6a` |
| `HYSo_G1_feature_planning.patch` | `ddcf3de148385f21b9011b6e36c3c127b61dc2912024e8b42a2d479dda4003c4` |
| `HYSo_G2_cpu_bridge_v1.patch` | `39fd6c156123e460f4f7b7b712f91dc9c39c776d6a6b83769a0abedcfabc4e7c` |
| `HYSo_G3_isolated_cpu_v1.patch` | `df7599b2b6e04de9553a850a5c1edacddf3fe77096f4c41db813bf402c1f94f4` |
| `HYSo_P13_capture_scopes_cpu_only.patch` | `3391212272c8f3ba88acb61031bc23881d88ef770d9990cb7a7040a00e66e69c` |
| `HYSo_P33_P31_verified_duration_gate_fix.patch` | `98704a5be9f0b54e070fb7f02aae24b7463b813b848ba8931c1ccff339d10843` |
| `HYSo_Packaging_120_strict_inventory.patch` | `5c4c8d1ce340cc3284ce57b1ddc67b75efa7a0dc2df718699ffb87cb0ec07797` |
| `apply_model_b_probe.py` | `fb936e6e5a5a0afb3c122e414cb5f554ae7125f2101988f74cd62797f760bc7c` |
| `apply_p13_native_v1.py` | `97dd850137cd16379c9b410c575d99dfd9bd8c0df4ec50dd4c4ce40fa79d9911` |
| `apply_single_probe.py` | `d6c59d7a9689f96d162f97edde29d21dafa7b76ae837afd2ddc0d48c252afaa2` |
| `apply_step4b.py` | `07d0e0f8db59c93c50d87c1b316017c57b425f30c896001a706b946a977ed778` |
| `apply_step4d.py` | `47259f2f4cf5f1de833e079a8d7d4b2df8f8e2bcb5bd714aefcc9642361227ab` |
| `apply_step4e.py` | `243c92d2baf5cd99d54dfdb5dd7b91e990210047aa7f8bba83d33228f8e08e0e` |
| `diagnose_cuda_constructor.py` | `53ddc9e5810370732f1d67c426f82c326e539bba85e0aafc37a00df7775c646c` |
| `diagnose_cuda_tensors.py` | `43dc2991dbb8cfad3ca646b91858625a08ed9e9d54e86a790b67ba2de6d0763c` |
| `diagnose_cuda_throughput.py` | `aa3623a3a1850caba0ca22d0d33f06b0603cc858ab55dbc253924e713b839b26` |
| `docs/guides/p23-forensic-plan.md` | `86bea6b98f7a0229826f58b0713a10e84b8d8bd26dc747bf101abfc4bb3f0a56` |
| `docs/guides/p23-fresh-admission.md` | `5af842a28990ce604ec9ba23995707d3e07a653911a9b735c0ffc1c855f45246` |
| `docs/reports/p23-basic-cuda-20261111.md` | `9bfb30cbab1931ceebd96f8dab9241008dd44d0bd44a5fdf90d9b4250ccd2a1e` |
| `docs/reports/p23-basic-cuda-pass-20261111.md` | `c0b24bb2b9de6aa26827e1e8cbd944ca91046a7d80ccdeaf47bf9e92ed76880c` |
| `docs/reports/p23-basic-memory-20261111.md` | `136a0922032e3642b2c5f74fb41167d365c76c34001e20de823f8a71447dbb7a` |
| `hyso_06_layer_core_budget_update.py` | `3859302f6967e287fda86919ead2fd77101f84f79be8adfc4694f61ffa31918a` |
| `hyso_06_layer_result_budget_v4.py` | `40f8ca69ada7a8faa0949de06acce8b6a8a66cac9158750fd2a1df6e83e7edee` |
| `hyso_06_layer_safe_apply_v3.py` | `2a47900166f4df17a5a895e3104eb155f823874551cfaa14cadad3e8731bee46` |
| `hyso_06_layer_witness_fix.patch` | `95286017eb0f4d4baad102cd44da5fcdec99e69dbca67600f8248e4cf0171a1a` |
| `hyso_06_layer_witness_fix_v2.patch` | `789d079eaf42dcd4051d541857f576adc237d7e13a01dc603fbfad01841e848f` |
| `hyso_heartbeat_onedrive_v2.py` | `77b43fa02f37ad3893bf80059dbd088e50831d63346ad087a568d9412e70ad5a` |
| `hyso_independent_acceptance_v1.py` | `36c9a136dfe1f065408f824f332aae5d914ec347ecb0a84a6b78e36ad2145d4e` |
| `hyso_p18_single_capture.ps1` | `61389123f0b1005faf5b2eba691e726baa7a8af998c5f5084ffc55ce1120192a` |
| `hyso_p18_single_capture_decimal_comma.ps1` | `d933b8becbdd2c3945816987d5b2cd94b1e82b9b560ce232314b05c8f997abb7` |
| `hyso_p18_target_locale_fix_v1.py` | `fb9ceb0d6e941904133ecb1b31e0726cf774972146c197915f9526352aed7f5a` |
| `hyso_p31_p32_overnight.ps1` | `ef551c3d811dd1d9cc129ecc412cd82a001eeddbf29e0ab02bb77f77217fe9e0` |
| `hyso_prepare_final_manifest.ps1` | `b9437f5b1ac7f75b4050413d31724b9d7c4a0f8f2e60ca4a9eefee9f574a1dc8` |
| `hyso_provider_cpu_status_fix_v1.py` | `6e4ff43f8063982c1a39747c373098555056a4cf0390cb9de94327e42cb7eba2` |
| `hyso_reconcile_existing_runs.py` | `9054c35e4340211c65bbef2f396937fe31b817ecb4bcaa0c61690f4cdec39b61` |
| `hyso_reconcile_existing_runs_v2.py` | `92b462ee47ff9b9f6a1f0ce2d488eed209fb500f29ba5d8c9c06affee6102b75` |
| `hyso_remaining_audit.py` | `e8c14daa44cdf7679e9f13249af9705e5cdba1b08769bd284aa2683d5859dc28` |
| `p13_capture_finalize.py` | `7bf57867cc3af63b655a0dce3ead8cac6a90a547d4a3e6807c9d177f8ad69f05` |
| `p13_capture_native.py` | `16535176c076788b68050e1a53c5d07ae3a7d0196e9a4eb99dd8d4c90c3b8ef0` |
| `p13_capture_worker.py` | `7c58437b6ad25520da8c8c835e9922626833d6d0457e8b76abdb4fced628b616` |
| `test_p13_capture_native.py` | `1acc14f0a27b0174a877bf0087942e1f06189f8fe209e7350e25620418c62216` |
| `tests/forensics/test_p23_forensic_runner.py` | `5e1c7fd668cc31fc450adc071851aa2d18016dd19e11c7303472a7ca7c88aba5` |
| `tests/forensics/test_p23_fresh_admission.py` | `09bb046f17e22c35ebf29714397b2e4992634220c5dae3d204864b7472b68358` |
| `tools/forensics/native_admission.py` | `00bc31b070c2d48a713507dddf1d95044e49536d3b1523504ce4fd3762ec199d` |
| `tools/forensics/p23_forensic_runner.py` | `3275a9021e3382d38686ee49c08dded3dfb54dfb5dc87abbcc9e87770cee9ab9` |
| `tools/forensics/p23_forensic_windows.py` | `dbd9e0d41f5c3baa37ec919d70ac9bf661e9719b445816d28f455efb24f2ec56` |
| `tools/forensics/p23_forensic_worker.py` | `3c188fe3bec7944c1a395397da81da62c6cb8bcf575d1a0137a2562f5e2ec822` |

## Harici kabul ve checkpoint kayıtları

Altı dosyanın son SHA256 değeri başlangıç kaydıyla eşleşti.

| Konum | SHA256 |
|---|---|
| `%USERPROFILE%/AppData/Local/hyso-full-resume-20261111-v2/final-results.json` | `86a014ef432ed96a552bfc35a9c65c3bb78714c117e72708ce7b3c8051795bde` |
| `%USERPROFILE%/hyso-cnn-user-acceptance-20261027/training/last-training.pth` | `87f54bc66618c341f5f4004fec07b7b2f54823f54cba486425ff6c4168ebf4c5` |
| `%USERPROFILE%/AppData/Local/hyso-p30-assets/mobilenet_v3_small-047dcff4.pth` | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` |
| `%USERPROFILE%/OneDrive/Belgeler/GitHub/hyso/outputs/real-two-model-comparison-20261029/model-b/last-training.pth` | `8279ea41b6b26d9d5e0454327d4548e78fa89ae70a7cd6b7948e76ed6446aaef` |
| `%USERPROFILE%/AppData/Local/hyso-full-resume-20261111-v1/Model-A/P01-P35-final-acceptance-matrix.json` | `ffdbc8ebf26a14fe2f3e49901725dbffb283cbb3e906fb50cf4a104345c86ee1` |
| `%USERPROFILE%/AppData/Local/hyso-full-models-20261111-v1/Model-A/P01-P35-final-acceptance-matrix.json` | `0c97e4a083bc450729a0a0e798057eadbedc93625383a16630647e161abd3b0f` |
