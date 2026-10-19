# M1-C — ECA & Coordinate Attention Implementation

Tarih: 2026-10-19 (Europe/Istanbul).

## 1. Git durumu

Branch `feature/hyso-models-next`, HEAD
`7883601833396134353292854dbda43e4cc3fad5` (`feat: add reusable CNN and residual blocks`).
Başlangıç çalışma ağacı temiz; M1-B'nin block/test/packaging/rapor değişiklikleri
bu commit'te mevcut. Branch ve HEAD değiştirilmedi.

Bu increment yalnız yeni iki attention modülü, mevcut yeni ResidualBlock'un
açık kabul listesine ekleme ve bunların test/paketleme doğrulamasıdır.

## 2. ECAAttention implementasyonu

Doğrudan kullanım:

```python
from hyso.models.attention import ECAAttention, CoordinateAttention

eca = ECAAttention(channels=8, kernel_size=3)
coordinate = CoordinateAttention(channels=8, reduction=32)
```

`ECAAttention(channels, kernel_size=3)` kanalları ve kernel'i pozitif integer
olarak doğrular; bool/float/string kabul edilmez. Kernel tek olmalı. Seçim açık
ve deterministiktir: default=3, caller pozitif odd kernel seçebilir; otomatik
kanal-bağımlı adaptive kernel hesabı eklenmedi.

Akış: `(B,C,H,W) → AdaptiveAvgPool2d(1) → (B,C,1,1)
→ squeeze/transpose (B,1,C) → Conv1d(1,1,k,padding=k//2,bias=False)
→ sigmoid → (B,C,1,1) → x * gate`.

Convolution spatial eksende değil channel descriptor uzunluğu üzerinde çalışır;
channel reduction/MLP yok. C<k ve C=1 durumlarında da zero padding kanal
uzunluğunu korur. Öğrenilebilir parametre sayısı k, tek state key `conv.weight`.
NCHW rank, yapılandırılmış C ve pozitif/nonempty boyutlar forward girişinde
kontrol edilir. Bütün parametre/module'ler constructor'da yaratılır.

## 3. CoordinateAttention implementasyonu

`CoordinateAttention(channels, reduction=32)` pozitif integer kanal ve reduction
ister. Intermediate genişlik **max(8, channels//reduction)**; küçük C/reduction
durumunda sıfır kanal oluşmaz. Shared 1×1 Conv, BatchNorm2d, nn.Hardswish ve
iki ayrı 1×1 gate Conv constructor'da register edilir. Convolution bias'ları
yazarların referansındaki gibi açıktır.

| İşlem | Boyut |
|---|---|
| Giriş | B,C,H,W |
| W üzerinden pooling, H korunur | B,C,H,1 |
| H üzerinden pooling, W korunur | B,C,1,W |
| Width descriptor transpose | B,C,W,1 |
| H ve W descriptor concat | B,C,H+W,1 |
| Shared conv/BN/h-swish | B,M,H+W,1 |
| H/W split ve width transpose geri | B,M,H,1 ve B,M,1,W |
| Ayrı conv/sigmoid gate'ler | B,C,H,1 ve B,C,1,W |
| Orijinal giriş × width gate × height gate | B,C,H,W |

H/W forward girdisinden alınır; sabit görüntü boyutu veya square varsayımı yok.
İki spatial eksen ortak global scalar descriptor'a indirgenmez. Pool katmanları
spatial eksenlerden birini koruyan adaptive average pooling'dir.

**Batch=1 küçük spatial davranışı:** BN, giriş H×W yerine concat descriptor'un
H+W konumu üzerinde çalışır. H=W=1 için bile descriptor uzunluğu 2'dir; normal
BatchNorm training için tek değer hatası oluşmaz. Fallback/frozen BN/GN ile
mekanizma değiştirilmedi. İki descriptor değeri eşitse variance sıfır olabilir;
eps sayesinde sayılar sonlu kalır. Train running mean/var/counter güncellenir,
eval running statistics kullanır; her parametrede her girdide nonzero gradient
garantisi verilmez.

## 4. Özgün mekanizmalarla teknik uyum

Kaynaklar bu görevde yeniden okundu:

- [ECA-Net özgün makale](https://arxiv.org/abs/1910.03151) ve
  [yazarların ECA modülü](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py):
  spatial average descriptor, reduction olmadan channel-axis Conv1d ve sigmoid
  gating korunuyor. Explicit k seçimi referans modülün sabit k kullanımına uygundur.
- [Coordinate Attention özgün makale](https://arxiv.org/abs/2103.02907) ve
  [yazarların CoordAtt modülü](https://github.com/houqb/CoordAttention/blob/main/coordatt.py):
  directional pooling, H+W ortak dönüşüm, minimum 8 hidden kanal, BN,
  h-swish ve ayrı height/width kapıları korunuyor. Buradaki channels API'si
  shape koruyan input=output kanal durumunu sunuyor. nn.Hardswish,
  x·ReLU6(x+3)/6 mekanizmasını sağlar; referans formülle test edildi.

Yeni validation, isimler ve sınırlı constructor API'si HYSo'ya aittir;
pretrained state dosyaları için upstream class/key uyumluluğu iddiası yok.
Makale sonuçları HYSo accuracy/latency/benchmark kanıtı sayılmadı; böyle kazanım
iddiası veya dataset/model indirme yapılmadı.

## 5. ResidualBlock entegrasyonu

`blocks.py` yalnız attention import'u, docstring, explicit kabul listesi ve yeni
tipler için `attention.channels == out_channels` constructor kontrolüyle genişledi.
None/Identity/SE/CBAM davranışı aynı; arbitrary nn.Module veya subclass kabulüne
açılmadı. ECA/Coordinate subclass'ları da exact type sınırında reddedilir.

Verilen nesne doğrudan aynı instance olarak constructor'da `self.attention`
altına register edilir; forward'da yaratılmaz/kopyalanmaz. Main path çıktısına
uygulanır ve shortcut toplamından önce çalışır. Mevcut tensor/shape guard aynen
korundu; hook ile output'u tek kanala indirip broadcasting denemesi reddedilir.
Aynı instance'ı iki block'a vermek açık sharing'dir ve caller sorumluluğudur;
ayrı attention constructor'ları parametre paylaşmaz.

Import yönü `blocks → attention → torch`; attention.py legacy block veya model
modülü import etmez. Circular import yok. ConvNormAct'in tamamı ve
ResidualBlock.forward AST'si HEAD ile aynı. Eski block state_dict registration
isimleri/sırası değiştirilmedi. Yeni tiplerde ek state yalnız attention altında.

## 6. Tensor shape ve gradient sonuçları

Yeni standalone matriste ECA C=1/3/8/16; Coordinate C/reduction çiftleri
(1,32), (3,1), (8,2), (16,1) ile test edildi. B/H/W profilleri:
(1,5,7), (2,3,5), (1,1,7), ayrıca width=1 ve (1,1,1) sınırları. İki modül
CPU float32/float64 ile gerçek forward/backward verdi; aynı shape/device/dtype,
finite input/parameter gradients ve en az bir nonzero gradient doğrulandı.

ECA'nın k=1/3/5/9 channel convolution sonucu, production reshape/Conv1d hattını
kullanmayan scalar channel-neighborhood hesabıyla karşılaştırıldı. Coordinate
pool değerleri input mean'leriyle, concat bölümleri H/W descriptor'larıyla,
split sonrası h-swish değerleri özgün formülle karşılaştırıldı. Gate'ler
örnek (B=2,C=3,H=3,W=5) için (2,3,3,1)/(2,3,1,5), son output (2,3,3,5).
Bu asimetrik input testi eksenlerin yanlışlıkla yer değiştirmesini denetler.

Residual entegrasyonda 12 pozitif kombinasyon: iki attention ×
(4→4/stride1, 4→6/stride1, 4→6/stride2) × float32/float64. Giriş (2,4,9,11);
stride2 projection sonucu (2,6,5,6). Attention/main ve shortcut shape eşliği,
finite gradients, SGD update sonrası gerçek attention parametre değişimi ve
constructor/forward boyunca parametre identity korunumu PASS.

Ek dört integration case constructor kanal mismatch, subclass reddi, shape
guard ve açık sharing'i doğrular. Invalid standalone C/kernel/reduction ve
rank/channel/empty-input vakaları da reddedildi. Test toleransları eski
testlerden alınarak genişletilmedi; mevcut assertion'lar değiştirilmedi.

## 7. Strict state_dict round-trip ve buffer'lar

Her attention için float32/float64 ile gerçek Adam update sonrası state_dict
diske yazıldı, `torch.load(weights_only=True)` ile fresh instance'a strict=True
yüklendi. Missing/unexpected key yok; key sırası ve bütün state tensorları,
BN buffer/counter dahil birebir eşit. Eval tekrarları ve source/target output
eşliği rtol=0/atol=0 PASS; eval sırasında running buffer'lar sabit.

Residual 12 pozitif entegrasyon case'inde bütün block state'i fresh target'a
strict load edildi; tüm tensorlar ve eval output'ları tam eşit. Bunlar yeni
component'lerin model-weight round-trip'leridir; yeni attention'lar için EBO
epoch-resume veya format v1 training integration desteği iddiası değildir.

## 8. Legacy API/model-state ve M1-B uyumluluğu

M1-B'nin mevcut **119 blok testi** aynı kaynak gövdeleriyle korundu ve PASS;
bu dosyaya yalnız yeni testler eklendi. Baseline test kimlikleri önceki JUnit
ile birebir eşleştirildi. Eski **66 test** de aynı kimliklerle PASS:
41 public API + 1 published export contract + 18 model-state + 6 model testi.

Ek geçici probe, HEAD'deki M1-B blocks.py kaynağını ayrı namespace'te çalıştırıp
yeni block'la doğrudan karşılaştırdı: batch/group/None norm ×
None/Identity/SE/CBAM × identity/projection = **24 konfigürasyon**;
**48 train/eval output karşılaştırması** rtol=0/atol=0 PASS. State key sırası,
strict load ve train sonrası BN dahil tensor değerleri aynı.

`bcam.py`, `WithSE.py`, `fit.py`, `checkpoint.py`, package __init__ dosyaları,
frozen manifest ve metadata değişmedi. EBO/ELO/CEMO/CNNS/EMA, model-only,
format v1 ve EBO resume production yolu aynen korundu. Public model export
listesi hâlâ EBO/ELO/CEMO/EMA/CNNSENormalWithFit; yeni kullanım yalnız doğrudan
attention submodule import'udur.

## 9. Packaging sonuçları

MODULES beklentisine attention.py ve installed-import smoke listesine iki yeni
sınıf eklendi. Dört test PASS: canonical config, wheel/sdist içerik/metadata,
iki gerçek target kurulumundan import ve mevcut model CPU forward'ları.
Her arşiv doğru hyso namespace'inde **12 Python modülü** içerir; attention.py
ve blocks.py byte'ları güncel checkout ile aynı. Her iki sınıfın import dosyası
gerçek kurulu target altında; checkout'tan import edilmedi.

README/LICENSE/dependency ve istenmeyen tests/docs/cache/weights dışlama
kontrolleri korundu. Metadata 0.2.0, Python>=3.10, Torch>=2.2.2,
vision>=0.17.2; yeni dependency veya NumPy kısıtı yok. Build/kurulum geçici
repo dışı dizinde; mevcut dependency'lerle target install yeni clean install
veya yayımlanmış artefakt değildir.

## 10. Hedefli test sonuçları ve sıra

Windows / Python 3.10.18 / Torch 2.5.1+cu121 / vision 0.20.1+cu121,
CPU, CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
Mevcut test interpreter'ı:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.

| Sıra | Seçim | Yeni sonuç | Süre |
|---|---|---|---:|
| 1 | tests/test_attention_modules.py | **119 PASS** | 5.84 s |
| 2 | test_modular_blocks.py -k 'not new_attention' | **119 PASS**, 16 deselected | 5.99 s |
| 3 | test_modular_blocks.py -k new_attention | **16 PASS**, 119 deselected | 5.96 s |
| 4 | Legacy API/published API/model-state/models | **66 PASS** | 6.15 s |
| 5 | tests/test_packaging.py | **4 PASS** | 26.62 s |

Toplam **324 PASS**, **0 FAIL/ERROR/SKIP/XFAIL/XPASS**; beş süreç exit=0.
Deselected testler diğer block aşamasında çalıştırıldı; skip değildir. Yeni
kapsam 119 standalone + 16 integration = **135 yeni PASS**; baseline
119 block + 66 legacy + 4 packaging = **189 PASS** aynen korundu. Her aşama bir
kez çalıştı, başarısız koşu veya tolerans gevşetmesi yok.

Tam CPU suite ve mevcut CUDA regression çalıştırılmadı; M1-E'de kalır.
GPU/daha geniş runtime/FP16 matrisi için yeni PASS iddiası yok. Uyumluluk
probe'unda CUDA initialized=False. Önceki 364 PASS/1 SKIP ve 17 gerçek CUDA
senaryosu bu increment'in yeni test sonucu olarak kullanılmadı.

Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m1c-08d76258`.
Beş stage log/JUnit/run JSON'u, `verification-summary.json`,
`m1b-compatibility.py/json` ve `pytest-packaging/packaging0` altında build
logu, wheel/sdist, installed target'lar ve smoke JSON'ları saklandı.

## 11. Değiştirilen dosyalar

1. Yeni `hyso/models/attention.py`: ECAAttention, CoordinateAttention ve küçük
   private validation yardımcıları.
2. `hyso/models/blocks.py`: yalnız yeni attention tiplerinin açık kabulü ve
   constructor channel kontrolü, import/docstring.
3. Yeni `tests/test_attention_modules.py`: 119 standalone test.
4. `tests/test_modular_blocks.py`: mevcut 119 test korunarak 16 integration case eklendi.
5. `tests/test_packaging.py`: modül ve installed-import envanterine attention ekleme.
6. Yeni `docs/audit/m1-c-attention-modules.md`: bu rapor.

Legacy model/training/checkpoint/API/metadata/sürüm değişikliği yok. Yeni CNN,
trainer/config/registry, benchmark, dependency veya pretrained weights eklenmedi.
Commit/push/pull/merge/rebase/reset/checkout, tag/Release/upload veya kullanıcı
dosyası silme yapılmadı.

## 12. Karar

**M1-C: PASS.** İki mekanizma gerçek CPU forward/backward, strict serialized
state ve residual entegrasyon kontrollerinden geçti; baseline block/API/
model-state/packaging başarıları korundu. Sonraki adım bağımsız model composition
M1-D; tam CPU/GPU kapısı M1-E'de. Bu karar HYSo accuracy/latency kazancı veya
yeni modeller için EBO epoch-resume desteği anlamına gelmez.
