# Aşama 12 — M3 mimari doğrulaması

Aşama 12 tamamlandı. Branch `feature/hyso-models-next`, başlangıç ve son HEAD
`a0b16d72217343b3db48a39ddbec062d2feca90f`. Commit/push veya staging yapılmadı;
Aşama 13'e geçilmedi. Çalışma ortamı Windows, Python 3.10.18; paket kaynakları
Python 3.10 AST denetiminden geçti. Yeni bağımlılık eklenmedi.

## Başlangıç envanteri ve uygulama kararı

İlk incelemede `hyso/models`, Registry/Builder, bloklar, attention, ModularCNN,
ELO/CEMO ve mevcut testler okundu. Başlangıç Registry'si 14 bileşendi.

| Mevcut kaynak | Tespit / karar |
| --- | --- |
| `blocks.py` | ConvNormAct ve post-add SiLU kullanan iki-3x3 ResidualBlock; kanonik ReLU ResNet bloğu olarak yeniden adlandırılmadı |
| `WithSE.py` | SEBlock, depthwise BottleneckSEBlock, CEMO ve EMA; kaynakları değiştirilmedi |
| `bcam.py` | CBAMBlock, opsiyonel depthwise CBAM bottleneck, ELO/EBO; kaynakları değiştirilmedi |
| `attention.py` | ECA ve Coordinate Attention; kaynakları değiştirilmedi |
| `modular.py` | Configurable residual stages, SiLU, mevcut attention string seçimleri; kaynak/state layout korunur |
| `registry.py`, `builder.py` | Kapalı whitelist, pure JSON/shape validation, CPU construction; genişletilerek kullanıldı |
| torchvision | Zaten bağımlılık; ResNet/ResNeXt/DenseNet ve BasicBlock/Bottleneck algoritmaları yeniden yazılmadan kullanıldı |

SE/CBAM/ECA/Coordinate yeniden uygulanmadı. Eksik full-preactivation/Wide blokları
ve SimAM/Triplet/SK ayrı modüllerde tamamlandı. Standalone eski ResidualBlock'a
yeni attention tipleri için uyum eklendi; önceki parametreler ve state anahtarları
korundu. Top-level legacy `hyso.models.__all__` beş isim olarak kaldı.

## Eklenen aileler ve gerçek yapı

| Aile | Varyant / yapı |
| --- | --- |
| ResNet | 18,34 BasicBlock; 50,101,152 expansion-4 bottleneck; torchvision ImageNet stem; v1.5 3x3 stride yerleşimi |
| PreActResNet | 18,34,50,101,152 stage sayıları; pre-norm/ReLU, raw identity veya activated projection, final norm/ReLU; ImageNet stem ve açık base_channels |
| WideResNet | CIFAR-style 6n+4, varsayılan 28-10; 3x3 stem, 16k/32k/64k; torchvision wide_resnet50_2 değildir |
| ResNeXt | 50/101 grouped bottleneck; varsayılan 32x4d; 101-32x8d açık width_per_group=8 ile |
| DenseNet | 121/169/201 torchvision DenseNet-BC; growth=32, bn_size=4, compression=0.5; özel growth açık width varyantıdır |

Tüm ailelerde input_channels ve num_classes açık. Default ResNet/DenseNet
topolojileri torchvision'dan gelir; değiştirilmiş kanal/head/normalization/width
aynı checkpoint veya aynı mimari konfigürasyonu gibi sunulmaz. PreAct/Wide için
stem, projection ve stride tercihlerinin tam kapsamı [rehberde](../guides/model-architectures.md).

Bloklar: `BasicResidualBlock`, `Bottleneck`, `PreActivationBlock`,
`WideResidualBlock`, `ResNeXtBottleneck`. Projection stride/kanal uyuşmazlığında
otomatik oluşur. out_channels genişletilmiş çıkış kanalıdır; bottleneck expansion
4, basic expansion 1. Normalization batch/group/None; preactivation bloğunda
toplamadan sonra aktivasyon yoktur.

Attention: `SimAM`, `TripletAttention`, `SelectiveKernelAttention`. SimAM singleton
variance ve FP16/BF16 istatistikleri için açık stabilite işlemlerine sahiptir.
Triplet convolutional Rotate-to-Attend mekanizmasıdır. SK dalları softmax ile
birleştirir; squeeze kısmındaki linear/ReLU, descriptor BN içermeyen açık bir
varyanttır. Bu modül bir tam SKNet veya Caffe referansının birebir checkpoint'i
olarak sunulmaz. Kanıtsız kalite/hız artışı iddiası yoktur.

## Registry / Builder / CLI / training

Registry **27 bileşen**: önceki 14 + 5 aile + 5 residual blok + 3 attention.
Yeni sınıflar `hyso.models` üzerinden explicit import edilir.

`ModelBuilder.named(name, **params)` kapalı schema1 `registered_model` kind'ını
ekler. Önceki sequential_classifier/modular_cnn JSON sözleşmeleri değişmedi.
Validation model/weight oluşturmaz; unknown fields/import/callable reddedilir.
Sequential chain 10M, named family ayrı conservative 200M parametre tahmini
ile allocation öncesi sınırlandırılır. Shape ve maliyet hesapları yeni blokları
ve attention'ı kapsar. Bu sınırlar aktivasyon/RAM/runtime garantisi değildir.

CLI `models list/show/validate/summary` entegrasyonu test edildi. Named model
summary, family ve normalized options içerir; validate/summary sırasında
constructor çağrısı yasaklanarak sıfır model-allocation doğrulandı.

TrainingConfig, TrainingSession, ExperimentManager ve reporting formatları
değiştirilmedi. Yeni modeller normal nn.Module olarak kullanılır. Küçük
PreAct/Wide/Dense modellerde gerçek birer CPU optimizer adımı doğrulandı;
ResNet/ResNeXt için prepared session ayrıca doğrulandı. Büyük modellerde tam
backward yapılmadı; residual temsilcileri ve küçük width varyantları kullanıldı.

## Test kanıtları

**771 benzersiz pytest testi başarılı; 88 yeni M3 testi + 683 mevcut regresyon.**
Son test sonuçlarında çözümlenmemiş failure/error **0**, skip **0**. Dört ağır/
gereksiz mevcut örnek-installed-smoke vakası bilinçli deselect edildi.

Kanıt dizini: `%LOCALAPPDATA%/hyso-stage12-20261112/`.

| Kanıt | Sonuç / ilişki |
| --- | --- |
| `models-and-api.xml` | İlk geniş tur: 693 passed, 4 test-koruması hatası, 2 deselected; toplam 697 vaka |
| `m3-final.xml` | 88 passed; eski 85 M3 vakasının tamamı ve üç ek default/canonical konfigürasyon |
| `packaging.xml` | 6 passed, 2 ağır installed smoke deselected |
| `legacy.xml` | 65 passed: ELO/CEMO/EBO forward-backward, frozen state schema/strict load ve eski public API |
| `installed-m3-final.json` | Son wheel ve sdist target'larında yeni aile/attention import, Registry=27 ve gerçek küçük CPU forward başarılı |
| `final-verification.json` | Test kimlikleri bazında son sonuçlar ve koruma denetimi |

İlk geniş turdaki dört hata, CPU AdamW'nin graph-capture health metadata sorusu
olan `cuda.is_available()` çağrısını yanlışlıkla GPU işi sayan yeni test
korumasından kaynaklandı. Üretim Trainer/PyTorch algoritmaları değiştirilmedi.
Test CPU availability'yi false tutacak şekilde düzeltildi; gerçek CUDA context
başlatma ve cihaz işlemleri yasak kaldı. Son M3 turunda dört vaka dahil tümü geçti.
Tekrarlanan vakalar toplam test sayısına yeniden eklenmedi. Eski başarısız JUnit
kanıtı değiştirilmedi; son vaka sonucu M3 rerun ile güncellenerek değerlendirilir.

Doğrulanan başlıca kontroller:

- İstenen tüm ResNet/DenseNet derinlikleri, beş küçük-width PreAct derinliği,
  default-width PreAct18, Wide10-1/28-2/**28-10**, ResNeXt50/101-32x4d ve
  **101-32x8d** gerçek CPU forward; output `[N,num_classes]` ve finite değerler.
- Input channels 1 ve 2; sınıf sayıları 3 ve 5; ayrıca canonical RGB/1000 sınıflı
  ResNet18/DenseNet121 state-key/shape eşleşmesi ve strict torchvision load.
- Stride 1/2, channel mismatch, expansion ve batch/group/None normalizasyonu;
  float64 backward ve finite input/parametre gradient'leri.
- SimAM formül karşılaştırması ve autograd gradcheck; singleton spatial alan,
  FP16/BF16 ve büyük sabit değerlerde finite gradient. Triplet için bağımsız
  rotation/gate formülü; SK için channel yerine branch softmax kontrolü.
- Beş aile için gerçek torch.save / weights_only torch.load state_dict
  roundtrip ve aynı input'ta birebir eval output. Attention strict state roundtrip.
- Unknown JSON/parametreler, yanlış depth/groups/stride/expansion ve büyük
  allocation bütçeleri constructor öncesinde reddedilir.
- Mevcut attention/modular, Builder, CLI, TrainingConfig, reporting, comparison,
  Manager ve public API regresyonları. GradScaler deprecation uyarıları mevcut
  Trainer'dan gelir; bu görevde değiştirilmedi.

Gerçek dataset, uzun eğitim, CUDA benchmark veya ağır performans kabul testleri
çalıştırılmadı. Kısa CPU eğitim/forward/backward işlemleri sentetik veri kullandı.

## Paketleme, örnekler ve serialization sınırı

Strict wheel/sdist envanteri **153 Python modülü** (150 + üç yeni model modülü).
Kaynaklar arşivlerle byte düzeyinde eşleştirildi. İki format da checkout dışında
geçici target'a kuruldu; console-script/python -m girişleri ve ayrıca kurulu
M3 modeli doğrulandı. Aktif Python ortamına paket kurulmadı, yeni bağımlılık
indirilmedi. Arşivler `hyso-stage12-20261112/dist-final/` altında tutulur.

[examples/models/m3_workflows.py](../../examples/models/m3_workflows.py) sekiz
çalışan senaryo sunar: ResNet, varyant seçimi, küçük DenseNet, residual blok,
attention ekleme, custom Builder CNN, TrainingSession ve state_dict. Her senaryo
CPU testinde çalıştırıldı; varsayılan Builder senaryosudur. Training açık senaryo
seçiminde bir optimizer adımıdır ve kullanıcı modeli kalitesi gibi sunulmaz.

Mimari JSON **ağırlık değildir**. state_dict optimizer/loader/RNG/full-resume
durumu değildir. Yeni ailelerin automatic Trainer save_last_to/resume_from
profili **açılmadı**; ModularCNN sınırı ve reddetme davranışı test edildi.
Pretrained weights, ONNX/TorchScript export, büyük model full backward, Linux/
GPU çalışması veya model doğruluğu/hız kazancı bu aşamada sertifikalandırılmadı.

## Değişen dosyalar ve koruma

Mevcut dokuz dosya:

```text
README.md
examples/README.md
hyso/cli/models.py
hyso/models/__init__.py
hyso/models/blocks.py
hyso/models/builder.py
hyso/models/registry.py
tests/test_model_builder.py
tests/test_packaging.py
```

Yeni sekiz dosya:

```text
hyso/models/architectures.py
hyso/models/m3_residual.py
hyso/models/m3_attention.py
tests/test_m3_architectures.py
examples/models/m3_workflows.py
examples/models/README.md
docs/guides/model-architectures.md
docs/reports/stage12-m3-validation.md
```

Toplam **17 Aşama 12 dosyası**. Önceki **47/47** forensic/debug/ilgisiz dosya
başlangıçta ve sonda SHA256 ile eşleşti; önceki 2 tracked değişiklik ve 45
untracked dosya korundu. P01–P35 performance/benchmark motorları,
Trainer/resume/checkpoint dahil **71 dosya** değişmedi. TrainingConfig,
ExperimentManager ve reporting kaynakları da hash denetiminde aynı kaldı.

70/70 kabul raporu ve dışarıdaki üç kabul index/manifest dosyası ile üç benzersiz
checkpoint hash'i korundu. Dış kabul checkpoint'leri deserialize edilmedi;
state_dict testleri yalnızca bu görevde üretilen geçici/sentetik ağırlıkları okudu.
Kabul `final-results.json` SHA256:
`86a014ef432ed96a552bfc35a9c65c3bb78714c117e72708ce7b3c8051795bde`.

Staged alan boş; branch/HEAD aynı. Çalışma ağacında 17 yeni görev dosyası ile
önceki 47 dosya bulunur. Commit/push, reset/clean/restore, toplu staging veya
yıkıcı çalışma ağacı temizliği yapılmadı. Aşama 12 sonunda duruldu.
