# M1-D — ModularCNN Model Composition

Tarih: 2026-10-19 (Europe/Istanbul).

## 1. Git durumu

Branch `feature/hyso-models-next`, HEAD
`a5a35481a46de2cda95e6f2828d07c5f00b60fb2` (`feat: add ECA and Coordinate Attention modules`).
Başlangıç çalışma ağacı temiz; M1-C'nin altı dosyalık değişikliği bu commit'te
mevcut. Branch/HEAD korunuyor. Yeni üretim kodu yalnız `hyso/models/modular.py`.

## 2. ModularCNN mimarisi

Standalone `nn.Module`: `ConvNormAct stem → residual stages → AdaptiveAvgPool2d(1)
→ flatten → tek Linear classifier → ham logits`.

Stem, mevcut ConvNormAct'in 3×3/padding=1/stride=1 varsayılanını kullanır.
Her stage mevcut ResidualBlock'lardan bir Sequential, stage listesi ModuleList'tir.
Kanal/spatial değişiminde projection mevcut ResidualBlock tarafından yönetilir.
Conv/residual/attention işlemleri yeniden implement edilmedi; ayrı backbone/head,
fit, optimizer, callback, checkpoint veya AMP sistemi eklenmedi.

`forward_features(x)` son stage'in NCHW feature map'ini döndürür.
`forward(x)` bunu pool/classifier'dan geçirip `(B,num_classes)` logits üretir.
Cihaz veya dtype otomatik seçilmez; normal `.to(device/dtype)` ve PyTorch eğitim
döngüsü kullanılır. Yeni model EBO subclass'ı veya EBO.model replacement değildir.

## 3. Constructor sözleşmesi

```python
from hyso.models.modular import ModularCNN

model = ModularCNN(
    input_channels=3, num_classes=10,
    widths=(32, 64, 128), repeats=(1, 1, 1), strides=(1, 2, 2),
    attention="none", norm="batch",
)
```

Bu değerler gerçek varsayılanlardır. Input kanalı, sınıf sayısı ve bütün stage
değerleri pozitif Python integer olmalı; bool/float/string integer'a çevrilmez.
widths/repeats/strides nonempty tuple veya list, uzunlukları eşit. Listeler
değerleri dönüştürmeden tuple'a kopyalanır; caller'ın sonradan listeleri
değiştirmesi kayıtlı config veya model yapısını değiştirmez.

Stage stride'ları pozitif scalar integer'dır; per-axis pair, generator/set ve
otomatik uzunluk doldurma desteklenmez. Attention seçenekleri exact lowercase
`none/se/cbam/eca/coordinate`; norm `batch/group/None`. Bütün config validation
ilk convolution yaratılmadan önce tamamlanır. Giriş NCHW tensor, doğru C ve
nonempty boyutlar olmalı; yanlış input channel/rank/empty shape açıkça reddedilir.

Config readback için input_channels/num_classes/widths/repeats/strides ve
attention_name/norm_name saklanır; config framework veya runtime rebuild yoktur.

## 4. Stage ve tensor shape yapısı

Stage'in ilk block'u ilgili stride'ı, kalan repeat'ler stride=1 kullanır.
Stem spatial boyutu korur. Stage çıkışı her eksende ceil(input/stride)'dır;
widths[stage] output kanalını belirler. Stage sayısı en az bir, yapılandırılabilir.

Gerçek test örneği: input_channels=3, num_classes=2, widths=(4,6,8),
repeats=(2,1,2), strides=(1,2,2), giriş `(2,3,11,15)`:

| Nokta | Boyut |
|---|---|
| Stem | (2,4,11,15) |
| Stage 0, iki block | (2,4,11,15) |
| Stage 1, bir block | (2,6,6,8) |
| Stage 2, iki block / forward_features | (2,8,3,4) |
| Global pool / flatten | (2,8,1,1) / (2,8) |
| Classifier | (2,2) |

Ek shape matrisi bir ve çok stage, aynı genişliğin tekrarı, ilk stage stride=2,
input C=1/3/5, class count=1/2/7 ve B=1/2 içerir. Hook'larla stage output'ları
ve ilk/sonraki block stride'ları doğrulandı. Final classifier bias'ı kontrollü
[-2,3] verilince aynı ham değerlerin çıkması softmax uygulanmadığını doğrular.

## 5. Attention entegrasyonu

Küçük private factory yalnız existing nn.Identity/SEBlock/CBAMBlock/ECAAttention/
CoordinateAttention instance'ı yaratır; registry/dynamic import yoktur. Her
ResidualBlock için ayrı constructor çağrısı ve out_channels ile eşleşen C verilir.
Tüm parametreler optimizer kurulmadan register edilmiş olur; forward'da yeni
module/parameter yaratılmaz. Attention'ın block içindeki konumu M1-B/C ile aynıdır.

Existing attention constructor varsayılanları korunur: SE reduction=8,
CBAM reduction=16/kernel=7, ECA kernel=3, Coordinate reduction=32.
Model norm seçimi convolution block'larını yönetir; **CoordinateAttention kendi
BatchNorm'unu korur**, norm=None veya group bu attention iç norm'unu değiştirmez.
Yeni attention mekanizması veya istemsiz parameter sharing yok.

## 6. Forward/backward/optimizer sonuçları

Beş attention × üç norm × CPU float32/float64 = **30 gerçek öğrenme senaryosu**.
Her model widths=(4,6), repeats=(2,1), strides=(1,2), B=2 sentetik girişle
CrossEntropy backward ve Adam update çalıştırdı. Finite logits, finite/nonzero
input gradient, bütün parametrelerde finite gradient ve gerçek weight değişimi PASS.
Attention olan her block'ta en az bir attention parametresi değişti.

Attention instance/parameter identity'leri bloklar arasında farklı ve model
parametre listesinde kayıtlı. Forward/update boyunca registration identity'si
korunuyor; classifier dahil standalone normal PyTorch eğitimi çalışıyor.

## 7. Strict state_dict round-trip

Beş attention × float32/float64 = **10 serialized round-trip**. Train forward/
backward ve SGD update sonrası state diske yazıldı, weights_only=True ile
fresh aynı-config modele strict=True yüklendi. Key sırası, shape ve bütün
tensor değerleri; güncellenmiş BatchNorm buffer/counter'ları dahil birebir eşit.
Missing/unexpected key yok. Eval source/target çıktıları ve eval tekrarları
rtol=0/atol=0 PASS; eval sırasında running buffer'lar değişmiyor.

Yanlış widths, num_classes, input_channels, repeats, attention ve norm ile
uyuşmayan weight şemaları strict load'da reddedildi. Bu normal PyTorch weight
loading sözleşmesidir: örneğin yalnız stride değiştirmek weight shape'ini
değiştirmeyebilir ve state_dict tek başına bütün mimari config'ini denetlemez.
Yüklerken özgün config yeniden kurulmalı; model-config/checkpoint framework'ü
eklenmedi. Bu dosyalar model ağırlıklarıdır, EBO full-state epoch-resume desteği değildir.

## 8. Legacy uyumluluk

M1-C attention/block testleri **254 PASS** (119 attention + 119 eski block +
16 integration). Legacy hedefli testler **66 PASS**: 41 public API,
1 published export, 18 model-state, 6 model smoke/gradient. Önceki M1-C JUnit
kimlikleriyle eşleşme kontrol edildi; yeni model testlerinden ayrı sayıldı.

bcam.py, WithSE.py, fit.py, checkpoint.py, blocks.py, attention.py, package
__init__ dosyaları, frozen model manifest ve metadata HEAD ile aynı.
EBO/ELO/CEMO/CNNS/EMA davranışı, constructor/public export listesi, model-only/
format v1 ve resume planları değiştirilmedi. Yeni sınıf public models.__all__
listesine eklenmedi; yalnız `hyso.models.modular` doğrudan import'u var.
ModularCNN için EBO epoch-boundary resume desteği iddia edilmiyor.

## 9. Packaging

Packaging MODULES envanterine modular.py eklendi. Gerçek build arşivlerinde
doğru hyso namespace'inde **13 Python modülü**; wheel/sdist kaynak byte'ları
güncel checkout ile eşit. README/LICENSE, metadata/dependency tutarlılığı ve
tests/docs/cache/model weights/sentinel dışlama assertion'ları korunuyor.

Wheel ve sdist ayrı target'lara gerçek kuruldu; ModularCNN dosya kökü target
altında assert edildi. Her kurulu target'ta Coordinate Attention'lı tiny model:
input `(2,1,15,19)` → features `(2,6,8,10)` → sonlu logits `(2,2)` PASS.
Legacy üç model forward'u ve tüm modüllerin import provenance kontrolleri de PASS.
Smoke JSON'ları yeni modülün kaynağını ve output shape'ini kaydeder.

Metadata hâlâ 0.2.0, Python>=3.10, torch>=2.2.2, vision>=0.17.2; yeni dependency
yok. Build/target kurulumlar repo dışında, mevcut bağımlılıklar `--no-deps`
ile kullanıldı. Bu yeni bağımsız clean dependency install veya yayın değildir.
Installed smoke süreçlerinde CUDA initialized=False.

## 10. Test sonuçları ve kanıtlar

Windows/Python 3.10.18/Torch 2.5.1+cu121/vision 0.20.1+cu121, CPU,
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
Mevcut test interpreter'ı:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.

| Sıra | Kapsam | Sonuç | Süre |
|---|---|---|---:|
| 1 | tests/test_modular_model.py | **102 yeni PASS** | 6.73 s |
| 2 | test_attention_modules.py + test_modular_blocks.py | **254 PASS** | 6.29 s |
| 3 | Public/published API + model-state + models | **66 PASS** | 5.88 s |
| 4 | tests/test_packaging.py | **4 PASS** | 28.76 s |

Toplam **426 PASS**, **0 FAIL/ERROR/SKIP/XFAIL/XPASS**. M1-C baseline'ın
324 test kimliği/sonucu korunuyor; yeni modelin 102 testi ayrıca eklendi.
Dört aşama belirtilen sırayla birer kez çalıştırıldı; başarısız koşu, saklanan
hata, assertion/tolerans gevşetmesi veya yeni skip/xfail yok.

Yeni 102 testin dağılımı: 18 architecture/shape, 30 attention/learning,
10 trained state round-trip, 1 raw-logit contract, 1 list snapshot,
32 invalid config, 4 invalid input, 6 incompatible state schema.
Tam CPU/GPU regression ve benchmark çalıştırılmadı; M1-E'de yapılacak.
Diğer runtime/GPU/FP16 profilleri için yeni PASS veya accuracy/latency kazancı
iddiası yok. Büyük dataset/pretrained weights/dependency indirilmedi.

Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m1d-c4d83ea8`.
Dört aşamanın log/JUnit/run JSON'ları, `verification-summary.json` ve
`pytest-packaging/packaging0` altında build logu, arşivler, target kurulumlar,
wheel/sdist smoke JSON'ları mevcut. Son modular.py byte'ları test edilen wheel
ile SHA-256 üzerinden de karşılaştırıldı.

## 11. Değiştirilen dosyalar

1. Yeni `hyso/models/modular.py`: ModularCNN ve küçük validation/attention factory.
2. Yeni `tests/test_modular_model.py`: 102 gerçek CPU testi.
3. `tests/test_packaging.py`: modül/import envanteri, kurulu yeni model forward
   assertion'ları ve source/output smoke JSON alanları.
4. Yeni `docs/audit/m1-d-modular-cnn.md`: bu rapor.

Legacy modeller, modüler bloklar/attention'lar, eğitim/checkpoint çekirdeği,
sürüm/dependency/public export listesi korunuyor. Kapsam dışı model aileleri,
trainer/registry/plugin/config framework eklenmedi. Commit/push/pull/merge/
rebase/reset/checkout, branch/version değişikliği, tag/Release/upload veya
kullanıcı dosyası silme yapılmadı.

## 12. Karar

**M1-D: PASS.** ModularCNN beş attention seçeneğiyle gerçek öğrenme ve strict
state round-trip kontrollerinden geçti. M1-C ve legacy baseline testleri,
wheel/sdist ve installed model import/forward başarıları korundu. Sonraki
M1-E aşamasında tam CPU/GPU regression ve mimari quality gate tamamlanmalı.
