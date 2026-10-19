# M2-D-B3 — Trainer & Benchmark Integration

Tarih: 2026-10-19 (Europe/Istanbul). Kanıt dizini:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2db3-evidence` (E).
Bu rapordaki execution sonuçları bu görevde çalıştırılmıştır; eski gate
sayıları yeni test sonucu olarak kullanılmaz.

## 1. Branch/HEAD ve changed files

Branch `feature/hyso-models-next`; başlangıç/son HEAD
`def610885990d12c58bbfc0efbec50a0e35f0d2c` —
`feat: add experiment artifact management and recovery`. B2 HEAD'de commit edilmişti.
Başlangıç index'i boş; yalnız `tests/test_gradient_accumulation.py` kullanıcı
farkı vardı. Bu dosyaya yazılmadı/stage edilmedi/geri alınmadı. Hash baseline:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

Değişiklikler: [manuel kullanım rehberi](../guides/experiment-manager.md),
`tests/test_experiment_integration.py`,
`tests/cuda_experiment_integration_validation.py`,
`tests/test_packaging.py` ve bu audit. Production değişikliği veya yeni public
helper API yoktur. Commit/push/branch/version/dependency değişikliği yapılmadı.

## 2. Trainer integration

M2-D-A/B1/B2 sözleşmesi ve ilgili Trainer/resume/Benchmark/Manager/packaging
API'leri incelendi. Rehberdeki uygulama fonksiyonları public Trainer.fit,
history ve ExperimentRun API'lerini kullanır. Aynı fenced code blokları testlerde
compile/execute edilir; ayrıca rehberin tüm akışı çalıştırılır. Synthetic küçük
TensorDataset, gerçek ModularCNN/SGD/StepLR/EMA ve CPU FP32/FP64 kullanılır.
Store schema1 yaratılıp kapatılır, explicit migrate_store sonrası schema2 açılır.
Tracking fit sonrasındadır; hidden callback/background writer veya engine'e yeni
dependency eklenmez. Model/optimizer/scheduler/EMA/EarlyStopping caller/Trainer'a aittir.

## 3. Metric mapping / accuracy / None

train_loss→train/loss, val_loss→validation/loss: mean_loss;
train_acc→train/accuracy, val_acc→validation/accuracy: percent.
Gerçek history ile event değer/unit/axis/source eşliği assert edilir. Sabit
prediction'lı gerçek eğitim75% üretir ve75 olarak kaydedilir. Validation None
event üretmez. Epoch1-based; history'de olmayan LR/duration yazılmaz.
65 gerçek epoch sonunda260 metric,256+4 atomic batch ile yazılır; exact replay
duplicate üretmez. Batch'ler arası bütün-history atomicity iddiası yoktur.

## 4. Checkpoint ve resume attribution

Gerçek save_last_to checkpoint1 dosyalarının reference/copy SHA, size, producer
ve reopen ilişkileri kontrol edilir. Manager deserialize etmez; runtime guard
testinde torch.load yasaklanarak artifact işlemleri çalıştırılır.
DataLoader public iterator sınırında gerçek fit bir coherent save sonrası
KeyboardInterrupt ile kesilir. Failed/interrupted parent immutable kalır.
Public load_training_checkpoint, disposable caller bileşenlerinde doğrulanmış
completed_epochs/history metadata'sı sağlar; gerçek strict resume fresh Trainer.fit
ile, original total epochs ve callbacks=() korunarak yapılır. Prefix epoch1 yalnız
parent'a, yeni epoch2/3 yalnız retry'a kaydedilir; input producer değişmez.
Completed-budget ve EarlyStopping terminal resume yeni metric/checkpoint/update
üretmiş gibi gösterilmez; reason=resume_no_new_epochs'tur. Güvenilir public
metadata yoksa epoch sınırını tahmin etmemek rehberde açıkça belirtilir.

## 5. Benchmark/artifact entegrasyonu

Gerçek CPU FP32/FP64 Result döndükten sonra JSON/summary CSV/samples CSV export
edilir; üç artifact copy olarak bağlanır. Schema1, SHA/size, row count ve
close/reopen ilişkileri doğrulanır. Metadata benchmark UUID ile Experiment Run
UUID'yi ayrı kaydeder. Raw latency samples metric_events'e çoğaltılmaz.
Manager tensor/model serialize etmez; GPU timing veya memory probe içine I/O
eklenmez. Hayalî from_json/load_json API'si kullanılmaz.

## 6. Comparison entegrasyonu

Gerçek iki CPU Result ile compatible/incompatible; gerçek Result metadata'sının
explicit redaction'ı ile insufficient_metadata doğrulanır. Architecture farkı
tek başına incompatibility değildir. Plain JSON report allow_nan=False ile
no-clobber yazılır, input artifact ID'leri ve manager input link'leriyle
ilişkilendirilir. CPU memory not_applicable'dır. Native GPU'da measured pair'in
timing'i compatible iken isolation bilinmediğinden memory insufficient_metadata;
disabled probe karşısında timing compatible, memory not_comparable kalır.
Winner/speedup/significance/Pareto engine eklenmez; allocator baseline/cache ve
fresh-process isolation sınırları korunur.

## 7. Hata senaryoları

Gerçek fit sırasında RuntimeError/KeyboardInterrupt; bunlarla eşzamanlı logging
ve terminal kayıt hatası; başarılı eğitim sonrası metric/artifact hatası;
gerçek SQLite TEMP trigger transaction abort ve kapalı storage test edilir.
Asıl workload exception nesnesi korunur. Önceden committed metric silinmez.
Başarılı workload + başarısız tracking için workload_outcome=succeeded ve
failure_domain=tracking/artifact kullanılır. Storage kapalıysa running kalır;
reopen completion yapmaz, caller explicit recovery uygular. Checkpoint kaydı
coherent workload başarısı garantisi olarak sunulmaz.

## 8. Yeni CPU testleri ve hedefli regression

Yeni dosyanın ilk koşusu **29 PASS** (`E/new.xml`). Model/gradient/buffer/mode ve
Python/NumPy/Torch CPU RNG, optimizer/scheduler/EMA state korunumu gerçek işlemlerle
kontrol edilir. Manager sırasında forward/step/zero_grad/scheduler step,
CUDA synchronize/Event/reset ve torch.load fail-fast guard altındadır.
Benchmark before/after state assertions ve comparison input-link kontrolleri
güçlendirildikten sonra ilgili üç testcase ayrıca **3 PASS**
(`E/strengthened.xml`); bunlar toplamda tekrar sayılmaz.

Final hedefli koşu bütün B1/B2 Experiment testleri ile yeni entegrasyon dosyasını,
sekiz Trainer dosyasını, altı Benchmark CPU/protocol/export/comparison/result/state
dosyasını, checkpoint/OneCycle, model-state/public API/published compatibility ve
packaging testlerini kapsar. JUnit/log'lar E/targeted.*; exact identity envanteri
E/verification.json'dadır. Native GPU sonuçları CPU sayımına eklenmez.

**28 dosyada1389 farklı CPU PASS**,0 fail/error/skip/xfail/xpass.
Pytest167.16s, command wall172.33s;687 mevcut dependency/deprecation warning.
Dağılım: yeni29 + B1/B2 230 + Trainer706 + Benchmark306 + checkpoint/OneCycle54 +
model-state18 + public/published API42 + packaging4 =1389. Eski230 Experiment
identity kümesiyle eşlik kontrolü yapıldı. Üç güçlendirilmiş tekrar yeni testcase
olarak eklenmedi. Bütün source/test değişiklikleri ve gerçek sonuçlar E'de kayıtlıdır.

## 9. Gerçek CUDA entegrasyon doğrulaması

**PASS — iki native profile scenario**: FP32 ve AMP FP16. RTX3060 Laptop GPU,
Torch2.5.1+cu121/CUDA12.1; ayrı süreçte yeni native script çalıştırıldı.
Her profile gerçek ölçülmüş memory probe, üç export/artifact, SHA/reopen,
registered model/gradient/reference/mode, Python/NumPy/CPU ve selected CUDA0 RNG
korunumu içerir. Measurement sonrası Manager/export/comparison sırasında ek
forward/synchronize/Event/allocator reset/read/optimizer step/torch.load yasaktır.
Ayrıca gerçek disabled GPU probe ile timing/memory ayrımı kontrol edildi.
Kanıt: `E/native-cuda/report.json`, result/CSV dosyaları ve SQLite/artifact store.
CPU fallback veya fixture metadata ile native başarı üretilmedi. Eski160 CUDA
senaryosu bu görevde yeniden çalıştırılmadı.

## 10. Minimum Torch2.2.2 installed smoke

Mevcut Python3.10 minimum ortamı kullanılır:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Son wheel E/installed-minimum isolated target'a no-index/no-deps/no-build-isolation/
no-compile ile kurulur. Repo dışı cwd ve -I -B ile mevcut legacy/Trainer/checkpoint/
replay/CPU FP32/FP64 Benchmark smoke'una rehberin gerçek manuel entegrasyon akışı
eklenmiştir. Bütün HYSo import'ları installed target altında olmalıdır.
Clean distributions/recorded non-pyc source/metadata/binary hash'leri önce/sonra
karşılaştırılır. Global dependency kurulumu yapılmaz. Bu smoke CUDA kanıtı değildir.

**Installed smoke PASS**: Python3.10.18, Torch2.2.2+cpu,
torchvision0.17.2+cpu, SQLite3.50.2; manual_integration=true ve
cuda_initialized=false. Kaynak `E/installed-minimum/hyso/experiments/manager.py`.
Kanıt: E/installed-minimum.json, minimum-install.log, minimum.stdout.log ve
minimum.stderr.log. Installed smoke pytest/native scenario toplamına eklenmez.
Clean before/after kayıtları **eşittir:35 distribution,21432 non-pyc dosya hash'i**;
Python/prefix/version bilgileri de aynıdır. E/clean-before.json,
E/clean-after.json ve E/clean-comparison.json sonucu doğrular.

## 11. Wheel/sdist

Production inventory filesystem'den21 modüldür; yeni production modülü yoktur.
Wheel/sdist tüm21 modülün source byte'larını, legacy exports/metadata ve
tests/docs/DB/artifacts/cache/weights exclusion'ını doğrular. İki archive ayrı
isolated target'ta, belgelenmiş gerçek manuel entegrasyon dahil smoke edilir.
Son build `E/targeted-temp/packaging0/dist`; hash/size E/verification.json'dadır.
HYSo0.2.0/Python>=3.10/Torch>=2.2.2/torchvision>=0.17.2 metadata korunur.

**4 packaging testi PASS**;21-module source-byte equality ve iki installed smoke
geçti. Wheel88666 byte, SHA256
`a8b938057b33196df484c3991335860ed60e96ea37168e85c69e34ab58a27b7f`;
sdist85652 byte, SHA256
`397ef05bb77ceaa22b06d9b0e17cc1dfdc1b3d865613378b7cf6cc909eac6a1a`.

## 12. Legacy freeze

Başlangıç tracked dosyalarının hash envanteri `E/initial-hashes.json` içindedir.
Trainer/resume/checkpoint/Benchmark/M1/root dahil21 production modülü byte-identical
kalır. Roadmap/backlog/config metadata ve kullanıcı gradient dosyası korunur.
Index boş; git diff --check kontrolü yapılır. Root legacy API, checkpoint1 ve
BenchmarkResult1 değişmez. Yeni dependency/dataset/weights/release yoktur.
Başlangıç140 tracked dosyadan yalnız tests/test_packaging.py bu görevde değişti;
diğer139'un byte hash'i eşittir. Yeni dosyalar bölüm1'de listelendi.

## 13. Bilinen sınırlamalar

Manuel history logging fit sonrasındadır; her batch atomic'tir, bütün iş akışı
değildir. Workload exception anındaki history yalnız observed completed epoch
kanıtıdır; en son checkpoint save başarısını tek başına kanıtlamaz. Resume sınırı
yalnız doğrulanmış checkpoint metadata'sıyla belirlenir. Reference locator
varsayılan redacted; explicit path olmadan verification unavailable olabilir.
SQLite/filesystem ortak atomic transaction değildir; disk hatası terminal kaydı
engelleyebilir. Recovery process iptali değildir. Memory process/allocator scope,
caller isolation beyanı ve fiziksel power-loss/hostile FS sınırları B2'deki gibidir.
Bu görev full CPU/native CUDA final gate veya performans doğruluğu sıralaması değildir.

## 14. PASS/PARTIAL/FAIL

**M2-D-B3: PASS. M2-D-C'ye geçilebilir.** Trainer/history/accuracy/None,
checkpoint1/resume attribution, Benchmark schema1 export/artifact, comparison
semantiği, failure ayrımı, state/RNG/no-runtime-call sınırı doğrulandı.
29 yeni ve1389 farklı hedefli CPU testcase PASS; native CUDA FP32/AMP FP16
iki profile scenario PASS. Torch2.2.2 installed manual integration, clean ortam
hash eşliği ve21-module wheel/sdist kontrolleri PASS. Yeni skip/xfail veya
test gevşetme yoktur. İlk yeni test koşusunda başarısızlık yaşanmadı.
Son Git/index/hash kontrolleri branch/HEAD/index ve kullanıcı değişikliğinin
korunduğunu doğrular; git diff --check başarılıdır. Full CPU/eski native CUDA
final gate burada yapılmış sayılmaz.

## 15. M2-D-C için gereklilikler

Bütün CPU suite, ilgili eski/yeni native CUDA senaryoları, storage/recovery/
migration, minimum installed runtime ve final packaging/freeze birlikte yeniden
doğrulanmalıdır. Exact identity/skip durumları tekil sayılmalıdır. B3 public manual
integration kalır; otomatik engine tracking, background writer, dashboard veya
P01–P35 implementation'a geçiş onayı değildir.
