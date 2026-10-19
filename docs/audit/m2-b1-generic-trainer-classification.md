# M2-B1 — Minimal Generic Trainer Classification

Tarih: 2026-10-19 (Europe/Istanbul). **Karar: PASS — M2-B1.**

Yeni, modelden bağımsız CPU sınıflandırma döngüsü uygulandı. Nihai kaynaklarda
hedefli **385 farklı pytest testi PASS**, 0 FAIL/ERROR/SKIP/XFAIL/XPASS.
Ek olarak mevcut minimum Torch 2.2.2 CPU ortamında installed-wheel üzerinden
4 gerçek eğitim senaryosu PASS. Bu karar B2/B3/B4 veya tam B5 regression onayı değildir.

## 1. Git durumu ve tasarım dayanağı

- Branch: `feature/hyso-models-next`.
- HEAD: `7e95c492728fbc92cd21a7ab1b0ae911d3af788e`
  (`docs: freeze architecture roadmap and design generic trainer`).
- Başlangıç çalışma ağacı temizdi. M2-A ve roadmap bu HEAD'de commit edilmişti.
- İncelenen referanslar: [Roadmap v1.0](../roadmap/hyso-architecture-roadmap-v1.md),
  [M2-A DESIGN GO](m2-a-generic-trainer-design.md),
  [M1-E kalite kapısı](m1-e-final-architecture-quality-gate.md).
- Branch ve HEAD korundu. Commit, push, checkout, merge, rebase veya reset yapılmadı.

## 2. Eklenen public API

```python
from hyso.training import Trainer

trainer = Trainer(
    model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device="cpu",
    task="multiclass",
)
history = trainer.fit(train_loader, val_loader=None, epochs=5)
```

Constructor'da model dışındaki argümanlar keyword-only; fit'te epochs keyword-only.
Standart `nn.Module`, tek tensor giriş ve tensor logits beklenir. Modelin normal
`forward()` çağrısı kullanılır; model.fit çağrılmaz. Yeni namespace yalnız Trainer
export eder; root paket ve public model export listesi değişmedi.

CPU float32/float64 desteklenir. Caller modeli optimizer yaratmadan önce doğru
cihaza yerleştirir. Trainer model.to çağırmaz; parametre identity'leri korunur.
Tüm model parametre/buffer cihazları doğrulanır; yanlış cihaz veya CPU half/complex
model açıkça reddedilir. Batch tensorları explicit CPU cihazına taşınır; giriş
dtype'ı sessizce değiştirilmez. CUDA yoksa CUDA isteği açık ValueError;
CPU dışındaki erişilebilir cihazlar B1 kapsamında NotImplementedError alır.

## 3. Eğitim lifecycle

1. Fit preflight: validation isteği, pozitif Python int epochs (bool hariç),
   loss/model/optimizer bağlantısı ve finite/sized/nonempty iterable kontrolü.
2. Her epoch `model.train()`; autograd caller'ın no_grad bağlamında dahi açılır.
3. Her batch tensor/shape/label kontrolü, optimizer.zero_grad, model forward,
   mean loss ve update öncesi prediction accuracy hesabı.
4. Finite logits/loss denetimi; gerçek backward; mevcut her gradient'in finite
   denetimi; ardından tam bir optimizer.step.
5. Gerçek örnek sayısıyla epoch metrikleri; yalnız tamamlanan epoch history'ye eklenir.

Grad=None kullanılmayan parametreler için geçerlidir; bütün parametrelerde
zorunlu nonzero gradient aranmaz. Sparse COO gradient varsa coalesced values
denetlenir. Closure gerektiren LBFGS bu lifecycle'da açıkça unsupported'dır.
SGD/Adam/AdamW gerçek numerik reference testleriyle doğrulanan optimizer profilidir;
arbitrary custom optimizer/harici state davranışı için genel garanti verilmez.

Loader bir DataLoader veya aynı sözleşmeyi sağlayan sized iterable olabilir.
Batch tam `(inputs, targets)` tensor çifti olmalıdır. len() batch sayısıyla gerçek
iteration sayısı eşleştirilir; fazla/eksik batch açık hatadır. Sized olmayan
iterator ve sıfır batch (drop_last dahil) reddedilir. Tüm veri önceden tüketilmez.

Bir Trainer tek fit lifecycle'ına sahiptir. Tamamlanmış aynı total epochs bütçesi
tekrar çağrıldığında update yapmadan history kopyasını verir. Farklı bütçeyle
yeniden başlatma veya başarısız run'ı retry fresh Trainer gerektirir.
Hatalar bastırılmaz. Önceki başarılı batch update'leri/forward buffer değişimleri
hata halinde geri alınmaz; B1 bir training transaction veya resume sistemi değildir.

## 4. Loss, accuracy ve history matematiği

Multiclass: exact standart `nn.CrossEntropyLoss`, reduction='mean', weight=None,
label_smoothing=0; logits `(B,C)`, C>=2; targets `(B,)` torch.long hard indices.
Class range doğrulanır. Configured ignore_index herhangi bir gerçek hedefte
bulunursa reddedilir; kullanılmayan ignore_index ayarı mean denominator'ı etkilemez.

Binary: exact standart `nn.BCEWithLogitsLoss`, reduction='mean', weight=None,
pos_weight=None; logits/targets `(B,)` veya `(B,1)`. Yalnız bu iki şekil kontrollü
olarak `(B,)` yapılır; broadcasting veya genel squeeze yoktur. Floating targets
logits dtype'ına çevrilir; finite hard 0/1 gereklidir. Soft labels desteklenmez.
Binary prediction eşiği logits>=0; multiclass prediction argmax(dim=1).

Custom loss subclass/forward da farklı denominator getirebileceği için reddedilir.
Sum/none reduction, weighted loss, smoothing ve masked targets sessizce kabul edilmez.
Finite logits'e rağmen loss overflow'u ve backward'ta nonfinite gradient,
optimizer.step öncesinde FloatingPointError üretir.

```text
epoch_loss = sum(batch_mean_loss * actual_batch_size) / total_samples
epoch_acc  = 100 * total_correct / total_samples
```

Unequal batch ve son eksik batch aynı sample-weighted kurala uyar. Her batch
kendi forward/update öncesi prediction'larıyla ölçülür; training sonrasında
yeniden eval yapılmış model metriği değildir.

History düz dict: `train_loss`, `train_acc`, `val_loss`, `val_acc`.
Her completed epoch her alana bir giriş ekler; acc yüzde [0,100]. Validation
uygulanmadığından val_loss ve val_acc her epoch `None`, sahte sıfır değildir.

## 5. Optimizer ownership

Optimizer gerçek `torch.optim.Optimizer` olmalı. Modelin bütün trainable
parametreleri identity ile kapsanır. Optimizer'daki her parametre modelin kendi
parametresi olmalı ve tam bir kez bulunmalıdır. Yabancı, duplicate ve eksik
trainable parametreler reddedilir. Frozen model parametrelerinin optimizer'da
bulunması veya dışarıda bırakılması geçerlidir. Hiç trainable parametre yoksa hata.
Kontroller constructor'da ve fit öncesinde tekrar yapılır; arada değiştirilmiş
model parametresi/eski optimizer bağlantısı da reddedilir.

## 6. Test matrisi ve gerçek sonuçlar

Windows / Python 3.10.18 / Torch 2.5.1+cu121 / torchvision 0.20.1+cu121.
CPU test interpreter'ı:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
Gerçek mevcut dependency'ler kullanıldı; yeni dependency kurulmadı.

| Sıra | Test grubu | Sonuç | Pytest süresi |
|---|---|---:|---:|
| 1 | tests/test_trainer_classification.py (nihai) | 116 PASS | 6.11 s |
| 2 | tests/test_modular_model.py | 102 PASS | 7.67 s |
| 3 | Legacy API/model-state/EMA/checkpoint/EBO resume, aşağıdaki 7 dosya | 163 PASS | 18.10 s |
| 4 | tests/test_packaging.py | 4 PASS | 27.39 s |
| Toplam | Birbirinden farklı nihai JUnit testcase kimlikleri | **385 PASS** | |

İlk Trainer koşusu 111 PASS idi. Sonrasında 5 ek sınır testi yazıldı ve Trainer
dosyasının tamamı yeniden çalıştırılarak 116 PASS alındı. Toplamda aynı testler
iki kez sayılmadı. Ek 5 test production değişikliği gerektirmedi; diğer üç grubun
doğruladığı production/packaging kaynakları sonraki aşamada değişmedi.

Legacy dosyaları: `test_public_api.py`, `test_published_api_compatibility.py`,
`test_model_state_compatibility.py`, `test_models.py`, `test_ema_regression.py`,
`test_training_checkpoint.py`, `test_ebo_training_resume.py` (tests/ altında).
Legacy koşusunda 112 warning: mevcut sklearn undefined metric ve GradScaler
deprecation uyarıları; yeni skip veya assertion gevşetmesi yoktur.

Trainer test kapsamı: ModularCNN/Sequential/Linear × FP32/FP64 × SGD/Adam/AdamW
gerçek üç epoch update ve bağımsız native loop parametre eşliği; Linear binary
dört logits/target şekil birleşimi × FP32/FP64, gerçek iki epoch Adam eşliği;
bilinen logits ile %80 accuracy ve üç unequal/tail batch bölünmesinde doğru loss;
history uzunlukları/None val alanları; model.fit çağrılmaması; train mode,
zero_grad, unused/frozen parametreler ve modelin taşınmaması. Ayrıca invalid
input/output/label/task/loss/epoch/loader/device/ownership kontrolleri,
finite logits'ten overflow loss, nonfinite logits/gradient ve unsupported argümanlar.

Ana komut biçimi:

```text
<test-python> -B -m pytest -c tests/pytest.ini <grup dosyaları> -q --junitxml <kanıt>/<grup>-junit.xml
```

M1-E'nin **720 PASS / 1 SKIP** sonucu geçmiş baseline olarak korunur;
bu görevde tam suite veya CUDA regression yeniden koşulmadı. Yeni tam-suite
PASS iddiası yoktur. Bütün baseline testcase kimliklerinin final eşliği ve
tam CPU/GPU gate M2-B5'te yapılacaktır.

## 7. Gerçek packaging ve minimum runtime

Packaging MODULES envanterine iki training dosyası eklendi; eski assertion'lar
korundu. Gerçek `python -m build --no-isolation --sdist --wheel` çalıştı.
15 Python modülünün tamamı wheel/sdist ile checkout arasında byte seviyesinde
karşılaştırıldı. Metadata **0.2.0**, Python>=3.10 ve mevcut dependency listesi aynı.
Sentinel docs/tests/weights/cache dosyaları arşivlere girmedi.

Wheel ve sdist ayrı target dizinlerine `--no-index --no-deps --no-build-isolation`
ile kuruldu. Repo dışındaki cwd'den isolated Python ile import provenance,
legacy forward ve kurulu `from hyso.training import Trainer` doğrulandı.
Installed ModularCNN'de gerçek bir epoch SGD forward/backward/step ve classifier
weight değişimi gözlendi. İki kurulumda CUDA initialized=False.

Ek minimum profil: mevcut
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`,
Python 3.10.18 / Torch **2.2.2+cpu** / torchvision **0.17.2+cpu**.
Üretilmiş aynı wheel fresh target'a `--no-index --no-deps` ile kuruldu.
Installed Trainer ile multiclass/binary × FP32/FP64, gerçek AdamW, beş örnek,
batch_size=2, iki epoch: **4/4 PASS**; parametre update/history/finite loss ve
paket import yolu doğrulandı. Bunlar pytest toplamına eklenmiş testcase değildir.
Mevcut runtime dependency'leri veya eski installed hedefler değiştirilmedi.

Build dosyaları kanıt kökünün `pytest-packaging/packaging0/dist/` dizininde:

| Dosya | SHA-256 |
|---|---|
| hyso-0.2.0-py3-none-any.whl | 3018560484439037071718d5fff12400863b540a209e51a3f4cf73b80cbe71d9 |
| hyso-0.2.0.tar.gz | 1e323a562e976a7d7d5b32f864b6cefbb82caf68078597a81652ffe59b9a3945 |

## 8. Legacy compatibility

Önceden var olan 13 production Python modülünün ve setup.cfg/pyproject.toml/
MANIFEST.in dosyalarının Git çalışma ağacı filtreleriyle hesaplanan object ID'leri
HEAD ile eşleştirildi. Değişiklik yok. EBO/ELO/CEMO/EMA/CNNSENormalWithFit,
ModularCNN, dört mevcut attention, exports, model-state, checkpoint format v1
ve EBO resume implementasyonları aynı kaldı. Hedefli gerçek compatibility
testleri yukarıda listelendi. Eski engine refactor edilmedi veya kopyalanmadı.

## 9. Bilerek desteklenmeyen özellikler

AMP/GradScaler, accumulation, clipping, scheduler, EMA, EarlyStopping,
observer callbacks, validation, checkpoint/resume, generic dataset manager,
benchmark ve experiment manager uygulanmadı. Gelecek constructor/fit argümanları
Python'ın açık unexpected-keyword TypeError hatasını üretir; sessiz no-op yoktur.
val_loader None dışında verilirse update başlamadan M2-B3 açıklamalı
NotImplementedError gelir. Closure optimizer ve CPU dışı eğitim de B1 dışında.
Loss profili dışındaki weighted/ignored/soft-label/Mixup/CutMix destek iddiası yok.

## 10. Değişen dosyalar ve kanıtlar

Yalnız beş repository dosyası:

- `hyso/training/__init__.py` — yeni public subpackage.
- `hyso/training/trainer.py` — yeni bağımsız minimal lifecycle.
- `tests/test_trainer_classification.py` — 116 yeni gerçek CPU testi.
- `tests/test_packaging.py` — modül envanteri ve installed Trainer smoke eklemeleri.
- `docs/audit/m2-b1-generic-trainer-classification.md` — bu rapor.

Yerel kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b1`.
`trainer-junit.xml`, `modular-junit.xml`, `legacy-junit.xml`,
`packaging-junit.xml`, `verification-summary.json`, `installed-minimum.json`,
`installed-minimum.py`, `summarize.py`; packaging alt dizininde build.log,
wheel-smoke.json, sdist-smoke.json, arşivler ve kurulu target'lar mevcut.
`git diff --check` başarılı. Yeni dependency, büyük veri/model indirmesi,
version bump, upload veya GitHub Release yapılmadı.

## 11. Karar

**PASS — M2-B1.** Bağımsız forward/backward lifecycle, açık CPU/device/loss ve
optimizer ownership sınırları, sample-weighted metrikler, gerçek update kanıtı,
hedefli compatibility ve wheel/sdist installed import kapıları başarılı.
Bu sınırlı gate'in sonucu, henüz uygulanmamış B2/B3/B4 özelliklerini veya
tam CPU/GPU regression sonucunu temsil etmez.

## 12. M2-B2'ye geçiş önerisi

M2-B2'ye geçilebilir: mevcut küçük döngü üzerine sample-weighted accumulation,
gerçek tail flush, unscale→normalize→clip sırası, CUDA FP16 AMP overflow/skip ve
tek-owner scheduler cadence eklenmeli. Validation/EMA/callback orchestration
B3'te, coherent checkpoint ve epoch-resume B4'te kalmalı. B1 unsupported
kontrolleri yalnız karşılık gelen özellik gerçekten uygulandığında kaldırılmalı.
Legacy dosyalar ve checkpoint format v1 aynen korunmalı.
