# M2-B4-C — Deterministic Epoch-Resume Conformance

Tarih: 2026-10-19. Karar: **PASS — desteklenen bounded replay profilleri**.
Sonuçlar: **1044 farklı CPU testi**, **10/10 gerçek CUDA**, **2/2 minimum
installed-wheel replay** PASS. B5 final CPU/GPU kalite kapısı ayrıdır.

## 1. Git/HEAD ve çalışma ağacı

Branch `feature/hyso-models-next`; başlangıç ve son HEAD
`29e1a43e7eaac1c636cda6717afde2f049087e08`,
`feat: add safe generic trainer checkpoint restore`. B4-B commit edilmişti.
Başlangıçta yalnız `tests/test_gradient_accumulation.py` kullanıcı değişikliği
vardı. Dosya değiştirilmedi/stage edilmedi; başlangıç/son SHA-256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Commit/push/branch değişikliği/checkout/reset/merge/rebase yapılmadı.

## 2. Replay test yöntemi

Ortak test harness'i `tests/trainer_replay_support.py`; production API veya
engine eklenmedi. İki koşu da baştan **epochs=5** ile başlar. Aynı seed'ler,
initial model/optimizer/scheduler/scaler/EMA/ES state ve RNG eşliği önce
doğrulanır. Baseline beş epoch çalışır. İkinci koşuda native coherent save
başarıyla tamamlandıktan hemen sonra test wrapper'ı kontrollü exception üretir.
Yeni nesneler farklı seed ile oluşturulur; aynı plan ve epochs=5 ile devam eder.
Epoch bütçesi değiştirilmez. Temel kesinti üçüncü epoch'tur, kalan iki epoch'un
gerçek update'leri sayılır. Diğer kesinti konumları ayrıca test edilir.

Native save/update/DataLoader iterator'ı test tarafında gözlenir; forward,
backward, optimizer veya scaler işleyişi simüle edilmez. Baseline da aynı
save/instrumentation yolunu çalıştırır. Her epoch'taki **tam güvenli checkpoint
payload'ı**, final runtime state, LR trace ve örnek sırası karşılaştırılır.
Source prefix + resumed suffix, bütün baseline trace'ine eşit olmalıdır.
Yeni production pause/callback/hook veya custom sampler/dataset kullanılmaz.

## 3. Kapsam matrisi

Yedi CPU örneği, 1×5×5 girdi, widths=[3], repeats=[1], strides=[1], batch norm,
default collate, num_workers=0. Batch 2/3/4 daima son eksik batch üretir.
Accumulation 1/2/5; 5, epoch batch sayılarının hepsinden büyüktür.
25 temel case bütün **25 attention×scheduler** ve **15 attention×optimizer**
çiftini kapsar; kalan eksenler dönüşümlü atanır. FP32/FP64 × sequential/shuffle
dört çifti de vardır. Bu, bütün eksenlerin Cartesian product'ı değildir.
Tabloda 1=açık; ES continue=patience20 ile terminal olmayan lifecycle.

| Attention | Optimizer | Scheduler | EMA | ES | Accum | Batch | Shuffle | Val | FP |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| none | sgd | none | 0 | continue | 1 | 2 | 0 | 1 | 32 |
| none | adam | step | 1 | - | 2 | 4 | 1 | 1 | 64 |
| none | adamw | cosine | 0 | - | 5 | 3 | 0 | 0 | 32 |
| none | sgd | onecycle | 1 | continue | 1 | 2 | 1 | 1 | 32 |
| none | adam | plateau | 0 | - | 2 | 4 | 0 | 1 | 64 |
| se | adam | none | 1 | - | 5 | 3 | 1 | 1 | 32 |
| se | adamw | step | 0 | continue | 1 | 2 | 0 | 1 | 32 |
| se | sgd | cosine | 1 | - | 2 | 4 | 1 | 1 | 64 |
| se | adam | onecycle | 0 | - | 5 | 3 | 0 | 0 | 32 |
| se | adamw | plateau | 1 | continue | 1 | 2 | 1 | 1 | 32 |
| cbam | adamw | none | 0 | - | 2 | 4 | 0 | 0 | 64 |
| cbam | sgd | step | 1 | - | 5 | 3 | 1 | 1 | 32 |
| cbam | adam | cosine | 0 | continue | 1 | 2 | 0 | 1 | 32 |
| cbam | adamw | onecycle | 1 | - | 2 | 4 | 1 | 1 | 64 |
| cbam | sgd | plateau | 0 | - | 5 | 3 | 0 | 1 | 32 |
| eca | sgd | none | 1 | continue | 1 | 2 | 1 | 1 | 32 |
| eca | adam | step | 0 | - | 2 | 4 | 0 | 0 | 64 |
| eca | adamw | cosine | 1 | - | 5 | 3 | 1 | 1 | 32 |
| eca | sgd | onecycle | 0 | continue | 1 | 2 | 0 | 1 | 32 |
| eca | adam | plateau | 1 | - | 2 | 4 | 1 | 1 | 64 |
| coordinate | adam | none | 0 | - | 5 | 3 | 0 | 0 | 32 |
| coordinate | adamw | step | 1 | continue | 1 | 2 | 1 | 1 | 32 |
| coordinate | sgd | cosine | 0 | - | 2 | 4 | 0 | 0 | 64 |
| coordinate | adam | onecycle | 1 | - | 5 | 3 | 1 | 1 | 32 |
| coordinate | adamw | plateau | 0 | continue | 1 | 2 | 0 | 1 | 32 |

Dört ek ECA/AdamW/OneCycle/EMA/ES/shuffle/accum2/batch3 case sınır 1,2,4,5'i
kapsar. Sekiz terminal case: EMA açık/kapalı × best restore açık/kapalı ×
sınır1/sınır2. Böylece yeni CPU test identity sayısı **25+4+8=37**.

## 4. Kesintisiz vs kesintili model eşliği

CPU FP32/FP64 ve aşağıdaki CUDA replay'lerinde model state_dict, ayrı named
parameter'lar, bütün BatchNorm running mean/variance/num_batches_tracked
buffer'ları, gradient'ler ve mixed mode haritaları eşittir. Tensor'lar CPU'ya
alınıp dtype/shape ve contiguous uint8 byte görünümüyle karşılaştırılır;
signed zero dahil tolerans kullanılmaz. Epoch checkpoint'lerinde de eşlik
aranır; yalnız son ağırlıkların yakın olması yeterli sayılmaz. History ve
Python scalar'lar exact equality ile karşılaştırılır. Production bug bulunmadı.

## 5. Optimizer/scheduler/scaler/EMA karşılaştırmaları

Optimizer state_dict ve param groups, canonical native scheduler state_dict,
scaler state_dict, EMA shadow/decay, ES state, history ve bütün Trainer
progress/lifecycle alanları karşılaştırılır. Microbatch/attempt/applied update/
EMA update sayıları ayrıca formülle doğrulanır. Her gerçek optimizer attempt
öncesi/sonrası LR ve attempt/update/EMA counter trace'i saklanır; epoch scheduler
sonrası LR, tam epoch payload karşılaştırmasıyla da denetlenir. OneCycle'ın
last_epoch değeri applied update sayısına eşit olmalıdır.

Ek native scaler `vars()` karşılaştırmasında başlangıçta `_init_growth_tracker`
farkı görüldü. Yerel Torch 2.5.1 `GradScaler.load_state_dict` ve
`_lazy_init_scale_growth_tracker` kaynakları incelendi: native load,
`_init_scale`/`_init_growth_tracker` lazy initialization değerlerini yüklenen
checkpoint değerlerine ayarlar; aktif tensor'lar oluştuktan sonra bu başlangıç
alanları farklı tarihsel değerler taşıyabilir. Bu bir Trainer replay bug'ı
değildir. Replay eşliğinde yalnız bu iki inactive initialization alanı dışarıda
tutulur; **state_dict, aktif _scale/_growth_tracker ve diğer runtime alanları
exact kalır**. Rollback testleri ise bütün `vars(scaler)` alanlarını, bu ikisi
dahil, aynen karşılaştırmaya devam eder. Genel tolerans gevşetilmedi.

## 6. EarlyStopping ve terminal davranış

Nonterminal ES ve EMA+ES continuation, tamamlanmış bütçe ve best inference
finalization denetlenir. Terminal örnekler native ES patience=1/delta=1e6 ile
ikinci epoch'ta durur; restore_best_weights açık/kapalı ve live/EMA target
ayrımları kapsanır. İlk epoch checkpoint'inden devam terminal epoch'a ulaşır;
terminal ikinci epoch checkpoint'inin restore'u hiç training work üretmez.
Counter/best_loss/best_state ve doğru final evaluation target bit eşlidir.

Completed total=5 ve terminal=2 suffix trace'leri boş olmalıdır: training veya
validation iterator yok, optimizer attempt/update yok, scheduler step yok,
history tekrarı yok. Final runtime state bütünüyle baseline ile eşittir.

## 7. Shuffled loader/RNG/örnek sırası

Local generator sentetik dataset'i üretir; global seed'ler model/optimizer
kurulumundan önce kurulur. Resume nesneleri farklı seed ile yeniden oluşturulur.
Test iterator wrapper'ı her native batch'in ilk pixel'ine kodlanmış örnek
indekslerini kaydeder. Her train/validation iterator'ının tamamındaki gerçek
örnek sırası baseline prefix+suffix ile exact karşılaştırılır; yeni sampler
veya loader generator'ı eklenmez. Restore dönmeden iterator açılması hata verir.

Wrapper her batch'te Python, NumPy, Torch CPU ve CUDA kullanılıyorsa GPU RNG'den
test amaçlı çekim yapar; native RNG snapshot'ları bu tüketimleri de kapsar.
Final RNG yanında bütün epoch payload RNG'leri eşittir. Restore öncesi farklı
RNG tüketimiyle oluşturulan yeni modelin etkisi giderilir. Custom dataset,
multi-worker veya external RNG desteği iddia edilmez.

## 8. Gerçek CUDA FP32/FP16/overflow

`tests/cuda_trainer_replay_conformance.py --output <evidence>/cuda-replay.json`
gerçek RTX 3060 Laptop GPU, Torch 2.5.1+cu121, CUDA12.1 üzerinde **10/10 PASS**.
Script CUDA yoksa FAIL verir. Deterministic algorithms, cuDNN deterministic,
CUBLAS_WORKSPACE_CONFIG=:4096:8 kullanılır; benchmark ve TF32 kapalıdır.

| Grup | Case | Sonuç |
|---|---:|---|
| FP32 × EMA kapalı/açık | 2 | 10 attempt, 10 update; 3+2 exact replay |
| FP16 native autocast/GradScaler × EMA kapalı/açık | 2 | scale128, tracker10; 10 update |
| Native overflow→recovery × EMA kapalı/açık | 2 | 10 attempt, 5 update; final scale32768/tracker5 |
| Dataset-plan mismatch, FP32/AMP | 2 | Iterator açılmadan tam runtime korunur |
| Geç metadata commit rollback, FP32/AMP | 2 | Tam rollback; başarılı retry10 update |

Overflow gerçek forward/backward, native scaler init scale=2^20 ve native
unscale/step/update ile oluşur. Resume sonrasında applied update sayısının
checkpoint değerinden arttığı ayrıca denetlenir; kalan iki epoch dört gerçek
attempt üretir. EMA/OneCycle yalnız toplam
beş update ile ilerler. Model/optimizer/scaler/EMA, GPU RNG, LR, örnek sırası
ve epoch payload'ları exact'tir; kullanılan tolerans **0**, tensor farkı **0**.
Rollback case'leri B4-B'nin gerçek warm optimizer ve pending AMP unscale
bookkeeping testlerini tekrar kullanır; başarılı retry temizliği de doğrulanır.

## 9. Torch 2.2.2 installed-wheel

Mevcut temiz Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu ortamına son
wheel yalnız izole target olarak kuruldu (`--no-index --no-deps --no-compile`).
Repository dışındaki `installed-replay.py`, `-I -B` ile çalıştırıldı. Test-only
harness açık dosya yoluyla okunur; **bütün hyso import'larının installed target
altında olduğu ayrıca doğrulanır**. Runtime'a yeni dependency kurulmadı.

**2/2 PASS**: ECA+AdamW+OneCycle+EMA+ES+shuffle FP32 accum2 (10 update) ve
ECA+AdamW+StepLR+EMA+ES+shuffle FP64 accum5 (5 update). İkisi de gerçek baştan
5-budget / epoch3 interruption / iki epoch continuation; aynı strict state,
epoch payload, LR, örnek sırası ve RNG kontrollerini kullanır. Torch2.2 scheduler
canonical state ve disabled native GradScaler yolları gerçek paketle çalıştı.

## 10. Negatif/rollback testleri

B4-B testleri tekrar kullanıldı; mevcut assertion'lar korunarak schema/plan/
unsafe/ES corruption testlerine model, optimizer, scheduler, scaler, EMA, ES,
history/progress/modes/gradient/RNG için full snapshot equality eklendi.
Same-shape stride, attention/norm, epochs/accumulation, parameter groups,
scheduler config/budget, dataset hash, sampler/batch/drop_last/validation,
callback profili, corrupt metadata, unsafe pickle ve ES target reddi kapsanır.

13 mevcut transaction case component apply, scheduler/EMA apply sonrası,
ES/progress/modes commit ve RNG restore hatalarını, warm runtime rollback'i,
fatal rollback failure'ı ve immutable-payload TOCTOU korumasını denetler.
Yeni kopya fault-injection matrisi yazılmadı. GPU'da da gerçek component yükleme
sonrası hata ve tam runtime karşılaştırması vardır.

## 11. CPU test sonuçları

Python3.10.18, Torch2.5.1+cu121, torchvision0.20.1+cu121; CPU görünürlüğü,
Agg, plugin autoload kapalı, tek Torch CPU thread. Önce yeni conformance,
sonra ilgili regression çalıştırıldı. Son conformance: **37 PASS / 31.49s**;
regression: **1007 PASS / 103.94s**. JUnit classname/name identity kümeleri
ayrık: **1044 unique PASS, 0 failure/error/skip**. Tekrarlanan test koşuları
yeniden sayılmadı. Mevcut GradScaler deprecation warning'leri failure değildir.

| Test identity dosyası (tests/) | PASS |
|---|---:|
| test_trainer_replay_conformance.py | 37 |
| test_trainer_checkpoint_restore.py | 179 |
| test_trainer_restore_transaction.py | 13 |
| test_trainer_checkpoint_save.py | 135 |
| test_trainer_classification.py | 116 |
| test_trainer_updates.py | 141 |
| test_trainer_validation.py | 32 |
| test_trainer_lifecycle.py | 53 |
| test_training_checkpoint.py | 34 |
| test_ebo_training_resume.py | 45 |
| test_onecycle_checkpoint_compatibility.py | 20 |
| test_public_api.py | 41 |
| test_published_api_compatibility.py | 1 |
| test_model_state_compatibility.py | 18 |
| test_models.py | 6 |
| test_modular_model.py | 102 |
| test_gradient_accumulation.py | 28 |
| test_cv_dataset_resume.py | 39 |
| test_packaging.py | 4 |
| **Toplam** | **1044** |

Komut biçimi: `python -B -m pytest -c tests/pytest.ini <yukarıdaki dosyalar>
-q -x --junitxml <evidence>/...xml`; packaging için ayrıca izole `--basetemp`.
CUDA ve minimum installed replay sayıları bu CPU toplamına eklenmedi.

## 12. Packaging

Son checkout'tan wheel ve sdist üretildi; ikisi de checkout dışındaki izole
target'lara kurulup eski B1/B2/B3/B4-A/B4-B smokes ve yeni gerçek replay ile
doğrulandı. Yeni smoke ECA/AdamW/OneCycle/EMA/ES/shuffle/accum2 için aynı harness
ile 5-budget/3+2, dört resumed attempt, exact state/payload/LR/RNG/örnek sırası
eşliğini denetler. Harness test kaynağıdır; distribution'a production modülü
olarak eklenmez. MODULES=16; iki arşivin bütün production kaynak byte'ları son
checkout ile eşittir. Metadata: HYSo0.2.0, Python>=3.10, Torch>=2.2.2,
torchvision>=0.17.2 ve dependency listesi aynı.

| Arşiv | Byte | SHA-256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl | 55675 | c66868556f59fc08c4d905ab0878f0cd4db7455d9642fc0a84023ab528f6ff38 |
| hyso-0.2.0.tar.gz | 54342 | 6dc155e49b09006aa5d67e48db9fa9e4af19063348d642a9fd2520a8a06d73be |

Arşivler `<evidence>/pytest-packaging/packaging0/dist/` altındadır. Archive
timestamp'ları nedeniyle aynı production byte'ları arşiv hash'inin önceki
milestone ile aynı olmasını gerektirmez; bu çalışmanın gerçek hash'leri yukarıdadır.

## 13. Legacy uyumluluk

`git diff --name-only -- hyso` boştur: **bütün production dosyaları HEAD ile
aynıdır**. Native checkpoint v1, EBO/ELO/CEMO/CNNS/EMA, ModularCNN, attention,
blocks, callbacks, model exports ve roadmap değiştirilmedi. Legacy checkpoint,
EBO epoch-resume, OneCycle compatibility, API/state_dict/M1 kapıları geçti.
Model/data/loss/optimizer/runtime/device bounded plan sözleşmesi genişletilmedi.

## 14. Sınırlamalar

Bu sonuçlar küçük multiclass ModularCNN, standard TensorDataset/DataLoader,
num_workers=0/default collate, aynı runtime/device planı ve callbacks=()
profilindedir. Tam Cartesian ürün, farklı GPU mimarileri, driver/runtime
değişimleri, arbitrary custom state/RNG, multi-worker, mid-epoch veya budget
expansion garantisi verilmez. Ölçülen CUDA profili exact geçti; bütün CUDA
kernel/konfigürasyonları için evrensel bitwise garanti çıkarılmaz.
Yeni özellik veya production bug fix gerekmedi.

## 15. Değiştirilen dosyalar

- `tests/trainer_replay_support.py`: ortak strict replay/instrumentation harness.
- `tests/test_trainer_replay_conformance.py`: 37 CPU replay identity.
- `tests/cuda_trainer_replay_conformance.py`: bağımsız native CUDA gate.
- `tests/test_trainer_checkpoint_restore.py`: mevcut negatif testlerde ek full
  runtime equality; mevcut assertion'lar korunur.
- `tests/test_packaging.py`: mevcut smokes üzerine installed gerçek replay.
- `docs/audit/m2-b4c-trainer-resume-conformance.md`: bu rapor.

Git'teki gradient_accumulation farkı başlangıçtaki kullanıcı değişikliğidir.
Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b4c`.
`replay-junit.xml`, `regression-junit.xml`, `cuda-replay.json`,
`installed-minimum.json`, `installed-replay.py`, `coverage.json`, `coverage.md`,
`summarize.py`, `verification-summary.json` (bütün 1044 identity dahil).
Packaging alt dizininde build log, installed wheel/sdist smoke JSON'ları ve
arşivler bulunur. Arşivlerin kaynak byte/hash eşliği ayrıca
denetlendi; `git diff --check` başarılı.

## 16. PASS/PARTIAL/FAIL kararı

**PASS — M2-B4-C.** Dengeli CPU profilleri, kesinti sınırları, shuffled örnek
sırası/RNG, native CUDA FP32/FP16/overflow-recovery, negatif/rollback, installed
wheel/sdist ve minimum2.2.2 replay kapıları geçti. Tolerans eklenmedi; native
scaler initialization cache farkı gerekçesiyle yukarıda açıklandı. Production
değişmedi. Test silme/skip/xfail, dependency veya veri/model indirmesi,
version bump, commit/push/upload/release yapılmadı.

## 17. M2-B5 önerileri

Final checkout üzerinde tam CPU suite ve bütün mevcut CUDA validation
script'lerini ayrı ayrı yeniden çalıştır; bu yeni replay script'ini de final
gate'e dahil et. Test identity toplamlarını ayrıştır, minimum runtime ve son
wheel/sdist provenance/hash denetimini koru. B4-C'nin hedefli sonuçları B5 tam
regression veya release onayı olarak kullanılmamalıdır.
