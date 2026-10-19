# M2-B4-A — Generic Trainer Safe Checkpoint Save

Tarih: 2026-10-19 (Europe/Istanbul). **Karar: PASS — yalnız M2-B4-A save.**

Gerçek hedefli CPU koşuları **815 farklı testcase PASS**, 0 FAIL/ERROR/SKIP/XFAIL.
Ek native CUDA save kontrolü **3/3 PASS**; mevcut minimum Torch 2.2.2 CPU
ortamında güncel installed-wheel save profili **24/24 PASS**. Güvenli epoch
kaydı eklendi; Trainer epoch-resume/restore transaction uygulanmadı.

## 1. Git / HEAD

Branch `feature/hyso-models-next`, HEAD
`6fb103ab25798b41b179e37f9bf73662f3b93700`
(`feat: add trainer validation EMA and early stopping`). M2-B3 production/test/
raporu commit edilmişti. Başlangıçta yalnız `tests/test_gradient_accumulation.py`
içindeki iki yorum satırının silinmesi kullanıcı değişikliğiydi; bu dosyaya
dokunulmadı. Başlangıç ve son SHA-256 aynı:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Branch/HEAD korunuyor; commit/push/checkout/merge/rebase/reset yapılmadı.

İncelenen referanslar: [Roadmap v1.0](../roadmap/hyso-architecture-roadmap-v1.md),
[M2-A](m2-a-generic-trainer-design.md),
[M2-B3](m2-b3-trainer-validation-ema-lifecycle.md),
[M1-E](m1-e-final-architecture-quality-gate.md). Ayrıca gerçek checkpoint.py,
ModularCNN, native EMA/EarlyStopping ve bounded legacy loader planı okundu.

## 2. Mevcut checkpoint API'sinin gerçek sözleşmesi

Kullanılan mevcut public API değişmedi:

```python
save_training_checkpoint(
    path, model, *, optimizer=None, scheduler=None, scaler=None, ema=None,
    completed_epochs=0, history=None, metrics=None, include_rng=False,
)
```

Mevcut loader API'si `load_training_checkpoint(path, model, *, optimizer=None,
scheduler=None, scaler=None, ema=None, map_location='cpu', restore_rng=False,
validate_metadata=None)` şeklindedir. Bu görev yeni Trainer restore davranışı
eklemez; mevcut component loader'ı da değiştirmez.

FORMAT_VERSION=1 ve exact top-level alanlar aynı:
format, format_version, model, optimizer, scheduler, scaler, ema,
completed_epochs, history, metrics, rng_state. Native serializer component
type/state_dict, optimizer param_names, scheduler canonical state ve EMA
shadow+decay bilgilerini taşır. include_rng=True Python, NumPy, Torch CPU ve
zaten initialize edilmiş CUDA RNG'lerini yakalar. In-flight gradient kayıt
formatının parçası değildir; kayıt tamamlanmış epoch/update sınırındadır.

Serializer'ın safe plain/tensor kopyası, strict component/schema doğrulaması,
aynı dizinde temporary file, flush/fsync ve os.replace yolu aynen kullanılır.
Yeni paralel torch.save formatı veya top-level Trainer alanı oluşturulmadı.

## 3. Eklenen save arayüzü

```python
cfg = dict(input_channels=1, num_classes=2, widths=[3], repeats=[1],
           strides=[1], attention="eca", norm="batch")
model = ModularCNN(**cfg).to("cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
trainer = Trainer(model, optimizer=optimizer,
                  loss_fn=torch.nn.CrossEntropyLoss(), model_config=cfg)
history = trainer.fit(train_loader, val_loader, epochs=2,
                      save_last_to="last.pth")
```

Yeni optional argümanlar constructor'da model_config=None, fit'te
save_last_to=None. Model config'in yedi constructor alanının tamamı caller
tarafından sağlanır; otomatik sahte config oluşturulmaz. Config plain finite
tiplerle bağımsız kopyaya normalize edilir. Kayıt kapalıysa normal generic
model/loader kullanımına save profilinin kısıtları uygulanmaz.

İlk save profili exact standart ModularCNN, non-fused SGD/Adam/AdamW, mevcut
bare scheduler'lar, hard-label classification loss, standard CPU TensorDataset
ve DataLoader(num_workers=0), callbacks=() gerektirir. CUDA model/AMP için de
veri kaynağı CPU TensorDataset'tir. Unsupported kombinasyonlar iterator açılmadan
ve optimizer update başlamadan açıkça reddedilir. Caller modelini Trainer taşımaz.
Dosyanın parent dizini mevcut olmalı; target bir dosya yolu olmalıdır.

Tek lifecycle korunur. Tamamlanmış bir fit'e sonradan save_last_to eklemek,
best/finalized state'i last sanıp yazmaz veya isteği sessizce yok saymaz: açık
hata verir. Kayıtsız completed-budget no-op ve diğer B1/B2/B3 davranışları korunur.
resume_from halen unexpected-keyword TypeError üretir.

## 4. Metadata namespace

Mevcut metrics altında yalnız bu namespace kullanılır:

```text
metrics.generic_trainer = {
  schema_version: 1,
  plan: {... finite canonical configuration ...},
  progress: {
    completed_epochs, microbatches, attempts, updates, ema_updates,
    early_stop, model_modes, ema_modes
  },
  early_stopping: None veya typed config/state + best tensor snapshot
}
```

Global v1 ile metadata namespace sürümü ayrı kavramlardır. Plan string key'li
plain dict/list/scalar/None içerir; tuple'lar list olur, scalar subclass,
tensor/callable/object ve NaN/inf config değerleri reddedilir. Best snapshot
tensorları yalnız typed state bölümündedir. Native state_dict OrderedDict
metadata'sı mevcut safe serializer ile korunur; callback/model/function
nesneleri serialize edilmez. weights_only=True ile gerçek dosyalar açıldı.

Progress/history uzunluğu, completed budget, attempt/microbatch formülü,
applied<=attempts, EMA/applied eşliği, scheduler counter cadence'i ve bool
submodule mode haritaları kayıt öncesinde doğrulanır. Model/EMA ve optimizer
state'inde nonfinite tensor/numeric durumlar reddedilir.

## 5. Model / config / data planı

Model planı class adı, tam ModularCNN stage/attention/norm config'i,
named_modules type+structural extra_repr listesi, convolution kernel/stride/
padding/dilation/groups/bias/padding_mode bilgisi, named parameter/buffer
shape/dtype/device/requires_grad ve ilgili HYSo model modüllerinin source
SHA-256'larını içerir. Torch runtime ve Python major/minor da kaydedilir.

Caller config model attributes ile type ayrımını koruyan canonical JSON olarak
karşılaştırılır; True ile 1 eşdeğer sayılmaz. RNG'yi koruyan CPU fork bağlamında
kurulan standart structural reference aynı stage yapısını doğrular. Canlı
model taşınmaz veya yeniden initialize edilmez. Same-weight-shape stride ve
attention farklarının planları farklıdır; config'ten sapmış convolution/activation
yapısı, instance forward override, model/gradient hook'ları reddedilir.
Bu statik bounded descriptor, arbitrary Python davranışı veya harici state'in
eksiksiz semantik fingerprint'i olduğu iddiasını taşımaz.

Optimizer type, parametre isim/sırası/grupları ve başlangıç hyperparameter'ları;
scheduler type/config/OneCycle budget; loss type/reduction/ignore_index/smoothing,
task, device, AMP, native scaler initial policy/state, accumulation/clipping,
toplam epochs, EMA cadence/decay ve EarlyStopping config/target plan içindedir.
Scheduler'ın sahibi olduğu LR ve OneCycle momentum/beta değişimleri native
state'te saklanır. Diğer optimizer ayarlarının değişmesi reddedilir; scheduler
yokken LR değişimi de static-plan uyuşmazlığıdır. Bilinmeyen scheduler state
alanları veya arbitrary component pickle state'i preflight'ta reddedilir.

Data planı mevcut `_resume_loader_plan` yardımcısına dar delegation kullanır;
öncesinde exact TensorDataset/DataLoader profiliyle sınırlar. İki standard dense
CPU no-grad tensor, NCHW/target/dtype/hard-label kontrolleri; tensor shape/dtype/
tam içerik SHA-256'ları; batch size, drop_last, sampler type/sample count/batches,
worker/pin-memory config'i ve validation loader'ın varlığı kaydedilir.
Random/SequentialSampler ve default_collate desteklenir. Explicit generator,
replacement/custom sampler, custom collate, multi-worker, Subset/custom dataset,
instance dataset/loader metotları ve worker_init_fn destek dışıdır.

Epoch kaydında model/data ve scheduler/ES planı yeniden kontrol edilir; bu
profilde her epoch tensor hash hesaplaması bilinçli bounded maliyettir. Plan
hazırlamanın CPU Torch/Python/NumPy RNG'sini tüketmediği gerçek testle doğrulandı.
Kayıt gelecekteki arbitrary dış mutation/harici RNG replay garantisi vermez.

## 6. EarlyStopping snapshot serialization

Legacy EarlyStopping sınıfı aynı kaldı. Save helper'ında exact native tip,
pozitif integer patience, finite nonnegative delta, bool restore/verbose,
nonnegative counter ve tutarlı terminal flag denetlenir. Kaydedilen typed state:
patience, delta, restore_best_weights, verbose, target ('live'/'ema'), counter,
early_stop, best_loss ve best_state; ayrıca native type adı.

Başlangıç best_loss=inf, yalnız geçerli başlangıç halinde None'a normalize
edilir; best_state None'dır. Improvement sonrası finite best_loss ve exact
evaluation-target state keys/order/shape/dtype ile uyumlu finite tensor snapshot
zorunludur. Best state existing safe tensor-copy yolundan geçirilir. EMA açıksa
snapshot'ın shadow'a ait olduğu target alanında açıkça belirtilir.
Deserialize, callback state restore veya yeni ES algoritması eklenmedi.

## 7. Coherent last checkpoint sırası

```text
training → validation → epoch scheduler → ES evaluation/best snapshot
→ history/epoch_end observer → coherent last save → early-stop kararı
→ final best-target restore → train_end
```

Save optional dalı B3'te ayrılan coherent boundary'ye yerleştirildi. Son dosya
best restore öncesindeki model/optimizer/scheduler/scaler/EMA anını taşır.
Terminal early_stop flag'i ve özgün total epoch bütçesi de kaydedilir.
callbacks=() save profilinde observer replay veya I/O transaction yoktur.

Gerçek kötüleşen validation loss senaryolarında sekiz epoch bütçesi iki epoch'ta
sonlandı. Dosyadaki live/shadow BN counter=2, ES best counter=1; restore açıkken
son inference target counter=1 kaldı. Saved optimizer/scheduler son continuing
state ile eşleşti. EMA varsa canlı model last halinde kaldı, best shadow canlı
modele yüklenmedi. Restore açık/kapalı ve EMA None/0/.8 durumları test edildi.

## 8. Atomic save ve hata davranışı

Atomik dosya operasyonları mevcut checkpoint.py içindedir; değişmedi.
Native torch.save sırasında partial write, os.replace hatası ve ikinci epoch'ta
başarısız kayıt enjekte edildi. Önceki geçerli dosyanın byte'ları değişmedi,
geçici .tmp dosyası kalmadı ve eski dosya weights_only=True ile açılabildi.
İkinci epoch hatasında dosya completed_epochs=1/history=1 olarak kaldı;
Trainer gerçek iki epoch/four update anındaydı. Sahte rollback veya başarılı
completion raporu üretilmedi.

Exception dışarı çıkar; train_end ve final best restore çalışmaz. Gerçekleşmiş
update/history/BN değişimleri geri yüklenmez. Failed lifecycle sessizce yeniden
başlatılmaz. Atomik save başarısı, epoch-resume doğrulanmış anlamına gelmez.

## 9. CPU ve ek runtime test sonuçları

Windows / Python3.10.18 / Torch2.5.1+cu121 / torchvision0.20.1+cu121.
CPU interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.

| Grup | Nihai sonuç | Pytest süresi |
|---|---:|---:|
| Yeni test_trainer_checkpoint_save.py | 135 PASS | 15.96 s |
| B1/B2/B3 dört Trainer dosyası | 342 PASS | 8.34 s |
| Hedefli legacy/API/M1-D/checkpoint/data/numerics | 334 PASS | 36.06 s |
| Packaging | 4 PASS | 27.91 s |
| JUnit kimlikleri ayrık toplam | **815 PASS / 0 SKIP** | |

Yeni 135 testin 80'i beş attention × SGD momentum/AdamW × scheduler kapalı/
StepLR/Cosine/OneCycle × EMA kapalı/açık gerçek iki epoch state matrisi.
Diğerleri ES last-before-best/terminal, FP32/FP64 binary/multiclass, birden çok
optimizer group/name order, BN buffers, v1 exact fields/weights_only, RNG/plan
determinism, invalid config/loader/component metadata, static LR planı, atomic
failure ve kayıt kapalı normal generic fit davranışıdır. Native optimizer
state'leri ve gerçek update'ler kullanıldı; mock step öğrenme kanıtı sayılmadı.

B1 classification **116**, B2 updates **141**, B3 validation **32**, B3 lifecycle
**53 PASS**. Bu dört dosyanın assertion'ları değiştirilmedi.
Legacy ayrı sayımlar: public API **41**, published export **1**, model-state
**18**, models **6**, ModularCNN **102**, training checkpoint **34**, EBO resume
**45**, OneCycle compatibility **20**, gradient accumulation **28**, CV dataset
resume **39 PASS**. Kullanıcının yorum satırı değişikliği korunarak test çalıştı.

Ara 118/129/131/133/134 save koşuları nihai 135'e tekrar eklenmedi; JUnit
kimlikleri ile sayıldı. Son save-only guard eklemeleri kayıt kapalı regression
yolunu değiştirmedi. Source'a bağlı packaging/minimum/CUDA save kontrolleri son
save kaynaklarında tekrar yapıldı. Warning'ler minimum API için kullanılan
GradScaler deprecation uyarılarıdır. Tam CPU suite veya eski tüm GPU gate'leri
koşulmadı; geçmiş M1/B1/B2/B3 PASS'leri yeni tam regression sonucu diye sunulmaz.

### Ek gerçek CUDA save kanıtı

Mevcut RTX3060 Laptop GPU / Torch2.5.1+cu121 / CUDA12.1 üzerinde ayrı süreçte
küçük ECA ModularCNN, AdamW, OneCycle, accumulation ve EMA ile **3/3 PASS**:
FP32, enabled native FP16 AMP, native AMP'de büyük finite initial scale ile
gerçek overflow/all-skipped. İlk iki koşulda 4 attempt/4 applied/4 EMA update;
all-skipped koşulda 4 attempt/0 applied/0 EMA update kaydedildi. Enabled scaler'ın
gerçek scale/growth/backoff/tracker state'i dosyadan mevcut native state ile
eşleştirildi; CUDA RNG listesi ve live/shadow tensorları doğrulandı. All-skipped
scale yaklaşık 1e35→6.25e33 oldu. Gradient/scaler simülasyonu kullanılmadı.
Bu üç senaryo B5 tam GPU gate'i veya CUDA epoch-resume testi değildir.

### Minimum installed-wheel kanıtı

Mevcut `hyso-p06c-oi2koqex\clean\Scripts\python.exe` ortamı, Python3.10.18 /
Torch **2.2.2+cpu** / torchvision **0.17.2+cpu**. Son wheel fresh isolated
target'a --no-index/--no-deps ile kuruldu. Repo dışındaki cwd ve -I -B süreçte
installed source/version doğrulandı: **24/24 gerçek save senaryosu PASS**.
Beş attention × dört scheduler seçeneği 20; Plateau+ES live/EMA iki terminal
kayıt; iki ek FP64 SGD/Adam kaydı. OneCycle 2.2.2 native callable canonicalization,
disabled scaler, RNG, metadata ve last-before-best doğrulandı. Dosyalar
weights_only=True ile okundu; bu minimum CPU save testidir, restore testi değildir.

## 10. Legacy uyumluluk

Trainer dışındaki 14 mevcut production Python dosyası, paket config dosyaları
ve test_packaging.py/kullanıcının yorum değişikliği dışındaki mevcut testler
Git çalışma ağacı filtreleriyle HEAD object ID'sine eşit doğrulandı.
EBO/ELO/CEMO/CNNS/EMA API/export/state_dict, native callbacks, M1 model/attention,
checkpoint.py ve EBO resume planı aynı kaldı. Roadmap v1.0 değiştirilmedi.
Private `_resume_loader_plan`, `_scheduler_state`, `_component` ve `_safe`
coupling'i yalnız yeni save helper'ındadır; legacy helper'lar kopyalanmadı/refactor edilmedi.

## 11. Packaging

Yeni training/resume.py ile MODULES envanteri **16** oldu. Gerçek
`python -m build --no-isolation --sdist --wheel`, arşiv source-byte/exclusion/
metadata kontrolleri, wheel ve sdist'in ayrı --no-index/--no-deps hedeflerine
kurulması başarılı. Installed import listesi yeni helper modülünü de kapsar.
B1/B2/B3 smoke'ları korunarak installed ECA ModularCNN+AdamW+StepLR+EMA save
smoke'u eklendi: iki epoch, dört applied update, v1 weights-only dosya ve RNG.

Metadata **0.2.0**, Python>=3.10, Torch>=2.2.2, torchvision>=0.17.2 ve dependency
listesi aynı. Yeni dependency kurulmadı; yalnız üretilen HYSo paketi yeni target
dizinlerine kuruldu. Son arşivler kanıt kökünün
`pytest-packaging-verified/packaging0/dist/` dizininde:

| Arşiv | SHA-256 |
|---|---|
| hyso-0.2.0-py3-none-any.whl | e3bece3ca112bf08e448b946df69cf2137dce703898d21ff73e4bc91a950e927 |
| hyso-0.2.0.tar.gz | 495231d60a88096c730a62f3faf95c0bd95871ac15f0a51c23d91921b4796c01 |

## 12. Eksik restore özellikleri

resume_from, metadata deserialize/apply, typed ES restore, Trainer counters/
history/mode restore, outer restore transaction/rollback ve exact epoch replay
bu görevde yoktur. Mid-epoch, multi-worker, arbitrary custom dataset/callback,
ImageFolder/Subset save profili, external state/RNG replay, benchmark/experiment
manager veya yeni model/attention eklenmedi. Kayıt dosyasının güvenle açılması
tam Trainer resume PASS iddiası değildir.

## 13. Değiştirilen dosyalar ve kanıtlar

Agent değişiklikleri yalnız beş dosya:

- `hyso/training/trainer.py` — optional model_config/save_last_to ve coherent boundary.
- `hyso/training/resume.py` — yeni bounded plan, typed metadata ve save-only helper'ları.
- `tests/test_trainer_checkpoint_save.py` — 135 yeni gerçek CPU testi.
- `tests/test_packaging.py` — yeni modül envanteri ve installed save smoke.
- `docs/audit/m2-b4a-trainer-checkpoint-save.md` — bu rapor.

Git'te ayrıca görünen tests/test_gradient_accumulation.py farkı başlangıçtaki
kullanıcı değişikliğidir; aynı hash ile korundu.

Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b4a`.
`save-junit.xml`, `regression-junit.xml`, `legacy-junit.xml`, `packaging-junit.xml`,
`verification-summary.json`, `installed-minimum.json`, `installed-minimum.py`,
`cuda-save.json`, `cuda-save.py`, `summarize.py`. Son packaging alt dizininde
build.log, wheel/sdist-smoke.json, arşivler ve installed target'lar; minimum
senaryo dosyaları ve üç CUDA dosyası da kanıt kökünde bulunur. Ara build'ler
nihai arşivlerle karıştırılmaz. `git diff --check` başarılı. Commit/push/version
bump/upload/release, dependency veya dataset/pretrained model indirmesi yapılmadı.

## 14. Karar

**PASS — M2-B4-A safe checkpoint save.** Desteklenen bounded profilde tam epoch
eğitim state'i mevcut atomik v1 formatında, verified plan/typed metadata ve
last-before-best sırasıyla kaydediliyor. CPU, supplemental native CUDA,
packaging ve minimum installed-runtime save kapıları başarılı. Trainer restore
ve exact epoch-resume halen uygulanmamıştır.

## 15. B4-B için gerekli restore sınırları

Restore, caller-created model/optimizer/scheduler/scaler/EMA/loader planını
checkpoint apply öncesinde doğrulamalı. Schema/budget/model structure/code,
dataset hash/order, optimizer names/groups, scheduler cadence, precision ve
ES target uyumu; history/progress/counter/mode şemaları ayrı staged validator
ile denetlenmeli. İlk yeni iterator RNG restore'dan sonra açılmalı.
Existing v1 load/rollback üzerine Trainer metadata/mode/callback için outer
transaction eklenmeli ve geç metadata-commit hatasında tam rollback test edilmeli.
Terminal early_stop ve completed budget no-op davranışları açıkça doğrulanmalı;
best restore continuing optimizer state'iyle karıştırılmamalı. İlk exact-resume
profilinde callbacks=() korunmalı; güvenli load/read testleri epoch-resume eşliği
yerine geçmemeli. Tam CPU/GPU final kalite kapısı B5'te ayrıca yapılmalıdır.
