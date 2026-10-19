# M2-B4-B — Safe Trainer Checkpoint Restore

Tarih: 2026-10-19. Karar: **PASS — B4-B bounded safe restore**.
Bu rapor B4-C kapsamındaki tam replay matrisi veya B5 final regression kararı değildir.

## 1. Git/HEAD ve mevcut kullanıcı değişiklikleri

Branch `feature/hyso-models-next`; başlangıç/son HEAD
`5becb6569718b6fed3984530a851dab2d870bb0f`:
`feat: add safe generic trainer checkpoint saving`. B4-A commit edilmiş durumda.
Başlangıçtaki tek working-tree değişikliği `tests/test_gradient_accumulation.py`
içindeki kullanıcıya ait iki yorum silmesiydi. Dosyaya yazılmadı, stage edilmedi;
başlangıç ve son SHA-256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Commit, push, branch değişikliği, checkout/reset/merge/rebase yapılmadı.

## 2. Restore API

```python
trainer.fit(
    train_loader, val_loader, epochs=5,
    save_last_to="last.pth", resume_from="previous-last.pth",
)
```

Caller önce model, optimizer, scheduler, scaler, EMA, fresh EarlyStopping ve
loader'ları oluşturur. `model_config` bütün yedi ModularCNN constructor alanını
tanımlamalıdır. `epochs` orijinal toplam bütçedir: completed=3/total=5 dosyası
yalnız iki epoch çalıştırır. Total=3 dosyasını epochs=5 ile genişletmek reddedilir.
`save_last_to` bağımsız olarak isteğe bağlıdır; devamda coherent last-before-best
kaydı aynı B4-A yoluyla sürer. Yeni public class veya production modülü eklenmedi.

## 3. Metadata staging ve doğrulama

Dosya bir kez immutable byte dizisine okunur; `weights_only=True`, CPU mapping
ve mevcut native güvenli tip validator'ü kullanılır. Unsafe pickle fallback yoktur.
Staging ve gerçek native loader aynı byte dizisini okur; arada dosya yolunun
değiştirilmesi uygulanacak payload'ı değiştiremez. FORMAT_VERSION=1 ve mevcut
top-level alanlar korunur. Metadata yalnız `metrics.generic_trainer` altında,
aynı schema_version=1 / plan / progress / early_stopping alanlarındadır.

Apply öncesinde native component/schema/RNG validator'ü, tam metadata alanları,
planın canonical JSON eşliği, typed epoch/counter/mode alanları, finite history
ve uzunlukları, optimizer state, model/EMA finite tensor'ları doğrulanır.
History doğrulaması validation yokken None, accuracy için 0..100 aralığını kapsar.
Bool/int karışıklığı typed counter ve canonical plan denetimleriyle reddedilir.

## 4. Model/data/optimizer/scheduler eşliği

B4-A `prepare_save`, model/data descriptors ve scheduler plan üretimi tekrar
kullanılır. Model class/config, widths/repeats/strides, attention/norm,
structural descriptor, convolution ayrıntıları, source fingerprint,
parameter/buffer name/order/shape/dtype/device/requires_grad eşleşir. Aynı weight
shape'ine sahip farklı stride yeterli sayılmaz. Dataset tensor hash'i,
batch/sampler/drop_last/collate ve validation loader planı da eşleşir.

Optimizer type, parameter name/order/groups ve statik seçenekler korunur;
scheduler'ın yönettiği LR ve OneCycle momentum/beta değişimleri ayrılır.
Scheduler'ın config, budget, last_epoch, step counter ve optimizer LR eşliği
denetlenir. OneCycle applied update başına; Step/Cosine completed epoch başına;
Plateau validation loss başına mevcut Trainer akışını kullanır. Restore çift
step yapmaz. AMP overflow update/EMA/OneCycle sayacını artırmaz.
AMP/scaler policy, accumulation, clipping, loss/task, EMA decay/cadence,
runtime ve device planı da aynı olmalıdır.

## 5. Transaction snapshot ve rollback

Full staging sonrası outer transaction native `load_training_checkpoint`'i
çağırır; legacy loader değiştirilmedi. Snapshot model, optimizer, scheduler,
scaler, EMA shadow/decay, history, bütün Trainer lifecycle/progress alanları,
EarlyStopping, her alt modülün training bayrağı, mevcut gradient'ler ve
Python/NumPy/Torch CPU/başlatılmış CUDA RNG state'lerini kapsar.

GradScaler `state_dict` devam eden unscale/step bookkeeping'ini içermez;
transaction tüm native scaler runtime alanlarını da snapshot eder. Başarılı
epoch-boundary restore eski pending bookkeeping ve gradient'leri temizler;
başarısız restore bunları aynen geri getirir. Torch 2.2.2 fresh Plateau'da
olmayan `_last_lr` staging'de yalnız kopya üzerinde hazırlanır; apply sırasında
transaction içinde eklenir ve rollback gerekirse önceki alan kümesi korunur.

Native component apply, ES/history/progress/mode commit, terminal finalization
ve son RNG restore aynı outer transaction içindedir. Hata durumunda bütün
snapshot geri yüklenir. Rollback da başarısızsa açık fatal
`runtime may be partially restored` hatası üretilir. Transaction içinde user
callback veya dış I/O çalıştırılmaz. Fault injection yalnız testlerdedir.

## 6. EarlyStopping restore

Native EarlyStopping değiştirilmedi. B4-A typed serializer doğrulaması tekrar
kullanılır; config, patience/delta/verbose/restore_best_weights, live/EMA target,
counter/terminal/best_loss/best_state schema ve tensor buffer'ları denetlenir.
Validation history native EarlyStopping ile staged replay edilerek counter,
best_loss ve terminal eşliği doğrulanır; terminal sonrası history reddedilir.
Best snapshot doğru evaluation target device'ına taşınır. EMA açıkken best
snapshot yalnız EMA shadow'a uygulanır; live model continuing state'i korunur.

## 7. RNG ve loader continuation

Desteklenen profil standart CPU TensorDataset, standart DataLoader,
num_workers=0, default collate ve standart SequentialSampler/RandomSampler'dır.
Plan üretimi iterator açmaz; structural reference model RNG fork içinde kurulur.
Başarılı restore'un son adımı checkpoint RNG restore'dur. İlk yeni epoch
iterator'ı ancak bundan sonra açılır. Kalan epoch testleri baştan epochs=5
hedefiyle çalışır; üçüncü gerçek completed epoch save'inden sonra kontrollü
kesilir ve aynı epochs=5 ile iki epoch devam eder. epochs=3→5 testi kullanılmadı.

## 8. Completed/terminal davranışlar

Completed total budget ve terminal EarlyStopping dosyaları history/components/
modes/RNG restore eder; yeni iterator, optimizer update veya scheduler step
yapmaz. Gerekli best-target finalization transaction içinde uygulanır.
Continuing checkpoint ise best inference weights ile değiştirilmeden eğitime
devam eder. Completed testlerde DataLoader class iterator'ı hata verecek şekilde
korunarak yeni iterator açılmadığı doğrulandı.

## 9. CPU test sonuçları

Python 3.10.18, Torch 2.5.1+cu121, torchvision 0.20.1+cu121;
CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
`python -B -m pytest -c tests/pytest.ini ... -q --junitxml ...` ile gerçek
CPU testleri çalıştırıldı. Sonuç: **1007 farklı test identity PASS, 0 skip**.
JUnit classname/name çiftleri birleştirildi; tekrar koşular toplama eklenmedi.

| Test dosyası (tests/) | PASS |
|---|---:|
| test_trainer_checkpoint_restore.py | 179 |
| test_trainer_restore_transaction.py | 13 |
| test_trainer_checkpoint_save.py | 135 |
| test_trainer_classification.py | 116 |
| test_trainer_updates.py | 141 |
| test_trainer_validation.py | 32 |
| test_trainer_lifecycle.py | 53 |
| test_public_api.py | 41 |
| test_published_api_compatibility.py | 1 |
| test_model_state_compatibility.py | 18 |
| test_models.py | 6 |
| test_modular_model.py | 102 |
| test_training_checkpoint.py | 34 |
| test_ebo_training_resume.py | 45 |
| test_onecycle_checkpoint_compatibility.py | 20 |
| test_gradient_accumulation.py | 28 |
| test_cv_dataset_resume.py | 39 |
| test_packaging.py | 4 |
| **Toplam** | **1007** |

Son Trainer+packaging koşusu: 673 PASS / 75.41 s; ayrı legacy koşusu:
334 PASS / 29.47 s. Native GradScaler deprecation ve mevcut legacy warning'leri
failure değildir; skip, xfail veya test kimliği tekrarı yoktur.

Yeni completed-state matrisi 5 attention × 3 optimizer × 4 scheduler ×
EMA açık/kapalı = 120 case içerir. Sekiz gerçek 5-budget 3+2 smoke, altı
terminal ES case, iki partial ES continuation, bozuk schema/plan/unsafe dosya
kontrolleri ve 13 transaction case ayrıca çalıştı. Gerçek optimizer/model
işleyişi mock ile değiştirilmedi; apply sonrası hata üreten test wrapper'ları
gerçek native işlemleri önce çalıştırır. Mevcut testler silinmedi/zayıflatılmadı.

## 10. Gerçek CUDA testleri

`tests/cuda_trainer_checkpoint_restore.py --output <evidence>/cuda-restore.json`:
RTX 3060 Laptop GPU, Torch 2.5.1+cu121, CUDA 12.1; **5/5 PASS**.
CPU pytest sayısına eklenmedi; script CUDA yoksa hata verir.

- FP32 ve native FP16 AMP: gerçek 5-budget 3+2 devam; model/optimizer/scheduler/
  scaler/EMA/history/mode/GPU RNG eşliği; 10 attempt, 10 applied update.
- Native AMP overflow: 10 attempt, 0 applied update; EMA ve OneCycle ilerlemedi.
- FP32 ve AMP geç metadata hatası: bütün runtime/RNG rollback; başarılı retry
  10 update'e ulaştı. AMP case gerçek pending unscale bookkeeping'ini önce geri
  getirdi, başarılı restore'da temizledi.

Bunlar sınırlı ECA/AdamW/OneCycle supplemental kontrolleridir; final CUDA gate
ve bütün kombinasyonların bitwise replay iddiası değildir.

## 11. Torch 2.2.2 minimum runtime

Mevcut Python 3.10.18 / Torch 2.2.2+cpu / torchvision 0.17.2+cpu ortamında,
checkout dışındaki izole target'a oluşturulan wheel kuruldu (`--no-index
--no-deps`). `-I -B` ile source path'in installed target altında olduğu denetlendi.
**10/10 PASS**: dört scheduler × EMA açık/kapalı gerçek 5-budget 3+2 replay,
terminal ES+EMA+Plateau ve önceki B4-A çalışmasının gerçek
`hyso-m2b4a/minimum-save-0.pth` dosyasının doğrudan restore'u (4 update).
B4-A dosyası yeniden üretilmedi veya migrate edilmedi. Yeni dependency kurulmadı.

## 12. Packaging

Wheel ve sdist gerçek build edilip checkout dışına kurulmuş paketlerde eski
B1/B2/B3/B4-A smoke'ları ve yeni public fit restore smoke'u çalıştırıldı.
Yeni smoke completed ECA+AdamW+StepLR+EMA dosyasını restore eder; history,
components, dört update ve scheduler epoch=2 doğrulanır. MODULES **16** kalır;
wheel production source byte'ları working tree ile aynıdır. Paket metadata'sı
HYSo 0.2.0, Python>=3.10, Torch>=2.2.2, torchvision>=0.17.2 ve dependency
listesi değişmedi. Son arşivler kanıt kökünde `pytest-final/packaging0/dist/`
altındadır; byte boyutları ve SHA-256 `verification-summary.json` içindedir.

| Arşiv | Byte | SHA-256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl | 55675 | 0da0c018c2de1958cec5ecc9b3622dc9e1d15cafaf65a2460fc3afaa806bd762 |
| hyso-0.2.0.tar.gz | 54342 | 88e37e58561c1dfeee9f5f19bcdea10eb1752291a24b0d0da81cdb5342c67b12 |

## 13. Legacy compatibility

`hyso/checkpoint.py`, bcam/WithSE/fit/modular/attention/blocks ve gerçek callback
paketi (`hyso/callbacks/__init__.py`, `hyso/callbacks/callbacks.py`) Git HEAD blob
eşliğiyle değişmemiştir. Talepteki `hyso/callbacks.py` repository'de bir dosya
değildir; native callback kodu mevcut pakettedir. Legacy checkpoint/EBO resume,
public API/model-state, M1 ModularCNN ve mevcut accumulation/CV testleri geçti.
EBO/ELO/CEMO/CNNS/EMA API/state_dict, roadmap ve v1 format değiştirilmedi.

## 14. Desteklenmeyen kapsam

Restore callbacks=() gerektirir; custom observer state/replay veya dış I/O
transaction yoktur. Normal callback fit desteği korunur. Custom dataset,
multi-worker, custom sampler/collate, external random state, mid-epoch restore,
budget expansion, cross-runtime/device plan değişimi desteklenmez. Normal fit'te
desteklenmeleri exact-resume garantisi oluşturmaz. B4-A bounded model/loss/
optimizer/scaler/hook sınırlamaları restore için de geçerlidir.

## 15. Değiştirilen dosyalar

- `hyso/training/trainer.py`: resume_from ve kalan epoch/finalization akışı.
- `hyso/training/resume.py`: ortak plan kontrolü, staging ve outer transaction.
- `tests/test_trainer_checkpoint_restore.py`: 179 gerçek CPU testi.
- `tests/test_trainer_restore_transaction.py`: 13 transaction testi.
- `tests/cuda_trainer_checkpoint_restore.py`: bağımsız beş gerçek CUDA case.
- `tests/test_packaging.py`: mevcut smokes korunarak installed restore eklenmesi.
- `docs/audit/m2-b4b-trainer-checkpoint-restore.md`: bu rapor.

Git'teki ek gradient_accumulation farkı kullanıcıya aittir ve korundu.
Kanıt kökü `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b4b`:
`final-trainer-packaging-junit.xml`, `legacy-junit.xml`, `cuda-restore.json`,
`installed-minimum.json`, `installed-minimum.py`, `summarize.py`,
`verification-summary.json`; packaging altında build log, installed smoke
JSON'ları ve arşivler. Önceki ara JUnit sonuçları toplamda tekrar sayılmadı.

## 16. PASS/PARTIAL/FAIL kararı

**PASS — M2-B4-B.** Güvenli v1 restore, tam metadata/plan staging, gerçek kalan
epoch continuation ve geç apply/metadata/RNG hatalarında bütün runtime için
rollback doğrulandı. B4-A gerçek dosya uyumluluğu, installed packaging ve minimum
runtime kapıları geçti. Hiçbir test skip/xfail ile gizlenmedi. Version bump,
release/upload, dependency kurulumu veya dataset/pretrained indirmesi yapılmadı.
`git diff --check` başarılıdır.

## 17. B4-C'ye kalan conformance görevleri

Bütün attention/optimizer/scheduler/EMA/ES kombinasyonlarında kesintisiz ve
kesintili eğitim için geniş sayısal/RNG eşdeğerliği; farklı kesinti sınırları,
shuffled loader/validation planları ve uygulanabilir precision/overflow
varyasyonlarının kapsamlı conformance matrisi B4-C'ye kalır. Buradaki completed
restore matrisi tam continuation matrisi olarak sunulmaz. Bütün CPU/GPU final
regression ve release kalite kapısı B5'te ayrıca çalıştırılmalıdır.
