# M1-B — Reusable CNN Blocks Implementation

Tarih: 2026-10-19 (Europe/Istanbul).

## 1. Git durumu

Branch `feature/hyso-models-next`, HEAD
`36f08fc41c64dfda365414fa2d9c2f2b3aee668c` (`docs: inventory HYSo model architectures`).
Başlangıç çalışma ağacı temizdi; M1-A raporu bu commit'te mevcut. Branch/HEAD
değiştirilmedi. Üretim değişikliği yalnız yeni `hyso/models/blocks.py` dosyasıdır.

## 2. ConvNormAct tasarımı

```python
ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1,
            padding=1, groups=1, norm="batch", activation="silu")
```

Akış `Conv2d → norm → activation`. Kernel/stride pozitif integer veya iki
integer'lık tuple/list; padding negatif olmayan integer/pair. Pozitif kanal
ve groups değerleri ile groups'un hem input hem output kanalı bölmesi
constructor'da doğrulanır. Bool, float, bozuk pair ve geçersiz seçimler reddedilir.
Padding explicit'tir; kernel değişince otomatik same-padding hesaplanmaz.

Norm seçenekleri `"batch"`, `"group"`, `None`. GroupNorm, out_channels'ı bölen
en büyük <=8 grup sayısını kullanır: örneğin 10 kanal→5 grup, 12→6, 16→8.
Norm varsa convolution bias=False, norm=None ise bias=True. Affine norm
parametreleri PyTorch varsayılanıyla açık. Activation `"silu"`, `"relu"` veya
None; None bir Identity'dir. In-place aktivasyon kullanılmaz. Factory/registry,
dropout veya stochastic depth eklenmedi; device/dtype normal `.to()` ile yönetilir.

Yeni modülün kayıt isimleri `conv`, `norm`, `activation`; bu kayıt düzeni eski
CNN'lerde kullanılmıyor ve legacy ağırlık şemasını değiştirmiyor.

## 3. ResidualBlock tasarımı

```python
ResidualBlock(in_channels, out_channels, stride=1, norm="batch", attention=None)
```

Main: `3×3 ConvNormAct(SiLU,stride) → 3×3 ConvNormAct(activation=None,stride=1)
→ attention`. İki convolution padding=1. Shortcut kanal ve stride=(1,1)
korunuyorsa Identity, aksi halde `1×1 ConvNormAct(norm, activation=None,
padding=0, stride)` projection. Sonuç `SiLU(attention(main) + shortcut)`.
Norm seçimi iki main convolution ve projection'a aynı politikayla uygulanır.

Spatial sonuç her iki path'te ceil(H/stride_h), ceil(W/stride_w); yalnız bir
spatial eksende downsample da desteklenir. Yeni CNN/head/fit/trainer oluşturulmadı.
Legacy stochastic-depth yollarına dokunulmadı; yeni block'ta stochastic depth yok.

## 4. Attention entegrasyonu

Şimdiki kabul edilen exact instance tipleri: None, nn.Identity, mevcut
`WithSE.SEBlock`, mevcut `bcam.CBAMBlock`. Başka türler TypeError verir;
ECA/Coordinate Attention bu aşamada yoktur. SE/CBAM fc1 input ve fc2 output
kanalları out_channels ile constructor'da karşılaştırılır. Shape'i büyüten
çift CBAM spatial kernel'i constructor'da reddedilir.

Verilen instance **aynı nesne olarak** `self.attention` altında constructor'da
register edilir; deepcopy veya forward sırasında yeniden yaratma yok.
Main path sonucu attention'a gider; shortcut toplamı bundan sonra yapılır.
Attention'ın döndürdüğü tensor main shape'iyle eşit değilse broadcasting'e izin
verilmeden ValueError üretilir. Caller'ın sonradan module/hook değiştirmesi için
genel yapı doğrulama framework'ü yoktur; bu guard shape ihlalini kapsar.

Aynı attention instance'ını iki block'a vermek açıkça weight sharing yapar;
bu caller sorumluluğudur ve docstring'de belirtilmiştir. Ayrı instance'lar bağımsız
kalır. Varsayılan Identity her block için ayrı oluşturulur. `.eval()` ve `.double()`
kayıtlı attention'a yayılır; paylaşılmış instance'ın diğer kullanımını da etkiler.

Kullanım yalnız doğrudan submodule üzerinden:

```python
from hyso.models.blocks import ConvNormAct, ResidualBlock
from hyso.models.WithSE import SEBlock

conv = ConvNormAct(4, 6, stride=2, groups=2, norm="group")
residual = ResidualBlock(4, 6, stride=2, attention=SEBlock(6, reduction=2))
```

Yeni sınıflar `hyso.models.__all__` içine eklenmedi; eski beş public model export'u aynı.

## 5. Forward tensor shape kontrolleri

Gerçek küçük CPU tensorlarıyla doğrulanan örnekler:

| Yapı | Giriş | Çıkış |
|---|---|---|
| ConvNormAct 4→6, k3/p1, stride=1 | (2,4,11,13) | (2,6,11,13) |
| Aynı conv, stride=2, groups=2 | (2,4,11,13) | (2,6,6,7) |
| Conv 4→8, k=(3,5), stride=(2,1), p=(0,2), groups=4 | (2,4,11,13) | (2,8,5,13) |
| Residual 4→4, Identity shortcut | (2,4,11,13) | (2,4,11,13) |
| Residual 4→6, projection, stride=1 | (2,4,11,13) | (2,6,11,13) |
| Residual 4→4, projection, stride=2 | (2,4,11,13) | (2,4,6,7) |
| Residual 4→6, projection, stride=(2,1) | (2,4,11,13) | (2,6,6,13) |

Residual matrisi norm=batch/group/None ve attention=None/Identity/SE/CBAM ile
çalıştırıldı. Attention kanal uyuşmazlığı, even spatial kernel, unsupported type
ve hook ile tek kanala düşürülerek broadcasting denemesi reddedildi.

## 6. Gradient, optimizer ve train/eval sonuçları

Conv testlerinde üç norm, üç activation ve standard/grouped convolution ile
finite/nonzero input gradient, bütün trainable parametrelerde finite gradient
ve en az bir nonzero parameter gradient doğrulandı. Invalid kanal/grup/
kernel/stride/padding/norm/activation constructor testleri PASS.

Residual 48 shape/norm/attention kombinasyonunda finite forward/backward ve
SGD step sonrası en az bir parametrede gerçek değişim PASS. Parametre identity'si
forward/update boyunca aynı; attention sonradan optimizer'a eklenmek zorunda
kalmıyor. CPU float32 ve float64 destekleri gerçek backward ile doğrulandı.

BatchNorm train'de running mean değişiyor; eval'da tracked counter sabit ve
çıktı tekrarlanabilir. GroupNorm/None train/eval aynı girdide aynı sonucu
veriyor. Disabled activation ve projection/ikinci conv activation=None kontrolleri PASS.

## 7. Fresh strict state_dict round-trip

None/Identity/SE/CBAM × float32/float64 = sekiz gerçek senaryo. Kaynak block
train forward/backward ve Adam update sonrasında, bağımsız fresh target'a
strict=True yüklenir. Missing/unexpected key yok; key sırası ve bütün state
tensorları tam eşit. BN'nin güncellenmiş buffer'ları da taşınır. Eval çıktıları
`rtol=0, atol=0` ile birebir eşit; attention-before-add ve final SiLU formülü
ayrıca gerçek çıktıyla karşılaştırılır. Bu, yeni block'un kendi weight şemasıdır;
eski modele ağırlık migration'ı veya yeni epoch-resume API'si değildir.

## 8. Legacy API / model-state uyumluluğu

Mevcut 66 testin kimlikleri M1-A JUnit'iyle eşleşiyor ve tamamı PASS:
41 public API + 1 published export + 18 model-state + 6 model smoke/gradient.
Constructor/default/fit signature'ları, canonical import identity ve export
sırası; frozen legacy key/shape/order/dtype/buffer ayrımı, wrapper `model.`
prefix'i, strict fresh-load output eşliği ve yanlış wrapper/child target reddi korunuyor.

`bcam.py`, `WithSE.py`, `fit.py`, `checkpoint.py`, package __init__ dosyaları,
frozen manifest ve metadata için HEAD farkı yok. Model-only/full-state v1 ve
EBO epoch-resume üretim yolu değiştirilmedi. Tam checkpoint/resume suite'i ve
GPU senaryoları bu aşamada yeniden çalıştırılmadı; M1-E regression kapısında kalıyor.

## 9. Packaging sonuçları

`tests/test_packaging.py` MODULES beklentisine yalnız `hyso/models/blocks.py`
eklendi. Installed distribution smoke'u yeni sınıfları ilgili submodule'den
import edip import dosyasının kurulan target altında olduğunu da doğruluyor.
Mevcut packaging assertion'ları korunuyor; yeni skip/xfail yok.

Dört packaging testi PASS: canonical metadata, gerçek wheel/sdist içerik ve
source byte eşliği, iki ayrı installed distribution import/legacy CPU forward.
İki arşivde de **11 Python modülü** var; blocks.py ve checkpoint.py doğru hyso
namespace'inde. tests/docs/cache/weights/sentinel dışlanıyor. README/LICENSE ve
sekiz dependency denetimleri geçiyor; metadata hâlâ 0.2.0, Python>=3.10,
torch>=2.2.2, vision>=0.17.2. Yeni dependency veya global NumPy kısıtı yok.

Build ve target install repo dışındaki pytest geçici dizininde; mevcut
bağımlılıklar `--no-deps` ile kullanıldı. Bu yeni clean dependency install veya
yayımlanacak yeni release artefaktı değildir. PyPI/TestPyPI upload yapılmadı.

## 10. Hedefli test sonuçları ve sıra

Windows / Python 3.10.18 / Torch 2.5.1+cu121 / vision 0.20.1+cu121,
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
Mevcut interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.

| Sıra | Testler | Sonuç | Süre |
|---|---|---|---:|
| 1 | tests/test_modular_blocks.py | **119 PASS** | 5.55 s |
| 2 | public API + published API + model-state + models | **66 PASS** | 5.90 s |
| 3 | tests/test_packaging.py | **4 PASS** | 25.37 s |

Toplam **189 PASS, 0 FAIL/ERROR/SKIP/XFAIL/XPASS**; üç süreç de exit=0.
Kontroller sırayla ve birer kez çalıştırıldı; başarısız koşu veya gizlenen
failure yok. Tam 364 testlik CPU suite, CUDA script'leri, pretrained weight,
dataset veya dependency indirme yapılmadı. Önceki 364 PASS/1 SKIP ve 17 CUDA
başarısı bu görevde yeni sonuç olarak sunulmuyor.

Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m1b-0b5ec265`.
`blocks.log`, `legacy.log`, `packaging.log`, üç JUnit XML'i,
`verification-summary.json` ve `pytest-packaging/packaging0` altında build
logu, arşivler, installed target'lar ve import/forward smoke JSON'ları mevcut.

## 11. Değişen dosyalar

1. Yeni `hyso/models/blocks.py`: yalnız ConvNormAct ve ResidualBlock,
   küçük private integer/pair validation yardımcıları.
2. Yeni `tests/test_modular_blocks.py`: 119 küçük gerçek CPU testi.
3. `tests/test_packaging.py`: MODULES ve installed-import listesine blocks ekleme.
4. Yeni `docs/audit/m1-b-modular-blocks.md`: bu rapor.

Legacy üretim dosyaları/export, training/checkpoint, version ve bağımlılıklar
korundu. ECA/Coordinate Attention, CNN, generic trainer/config/registry/plugin
framework eklenmedi. Commit/push/pull/merge/rebase/reset/checkout veya yayın
yapılmadı; kullanıcı dosyaları silinmedi.

## 12. Karar

**M1-B: PASS.** İki yeni bağımsız block gerçek forward/backward/update ve
strict state round-trip ile çalışıyor; legacy API/model-state ve packaging
kontrolleri başarılı. Üretimde yalnız blocks.py eklendi. Full CPU/GPU regression
M1-E'de yapılmalı; bu sonuç daha geniş dtype/device/runtime matrisi için garanti değildir.
