# P24/P27/P28/P29 entegre kabulü — 23 Ekim 2026

**Son kabul: 697 PASS / 0 FAIL / 0 SKIP.** P24, P27, P28 ve P29 aşağıda
tanımlanan finite CPU classification scope'unda ayrı ayrı **implemented**.
CUDA ve belirtilen diğer unsupported kapsamlar bu statüye dahil değildir.
Branch `feature/hyso-models-next`, başlangıç HEAD `24ce7d5`.
Commit/push yapılmadı. P21, 15 scratch P21 dosyası, P26 kaynakları, mevcut
benchmark/performance schema v1 ve diğer implemented feature kaynakları korundu.

## Uygulama ve kanıt

| Feature | Durum | Gerçek measured/supported kapsam | Açık sınır |
|---|---|---|---|
| P24 | implemented | P19 training_s + P26 checkpoint evaluation; gerçek tekrarlı seed; ilk gözlenen target geçişi, interval, right censor, reached-only mean/sample variance | CPU fresh Trainer/sequential TensorDataset; censor-adjusted population estimator ve checkpoint arası kesin an yok |
| P27 | implemented | Aynı ID/label/order/split/class/preprocessing için gerçek per-sample corruption ve iki inference; accuracy/macro-F1 delta; seed/severity/callable provenance | Finite CPU classification; corruption semantiği caller'a ait; detection/segmentation yok |
| P28 | implemented | Aynı master model/cohort ile P08 FP32/FP64/BF16/dynamic INT8 Linear recipes; gerçek native dtype/operator guards; P26 kalite; pp/relative-percent ayrımı | CPU; genel INT8 graph/static calibration/CUDA AMP FP16 yok |
| P29 | implemented | Mevcut compare_results korunarak matched FP32/64 CPU resident latency + kalite; Pareto, quality constraints, regression; uygun paired accuracy için exact McNemar | Timing ve macro-F1 significance unsupported; throughput/memory Pareto ve mixed precision yok |

P24 factory her distinct seed için gerçek fresh model/optimizer oluşturur.
Bağımsız storage, boş optimizer state, aynı eğitim verisi ve model/loss/
optimizer/scheduler/batching protokolü doğrulanır. Evaluation epoch0, cadence
ve son epoch'ta çalışır. P19 `train_end_ns` sınırından sonra yapılan eval,
training_s toplamına dahil edilmez. Checkpoint model hash'i P26 ile bağlanır.
Eksik training/eval/checkpoint/seed ve overlap geçerli sonuç üretmez.

Son test turundaki kaynak bütçesi: P24 reached örneği 3 seed×2 epoch;
censored örneği 2×2; initial-target örneği 2×1; reuse reddi öncesi 1×1;
parity örneği 2×1 ve 1 bağımsız kontrol fit. Toplam 11 tamamlanan gerçek fit,
16 tamamlanan epoch ve ayrıca 1 kasıtlı başarısız training başlangıcı.
Her fit'te 4 train sample/tek batch, eval'de4 sample kullanıldı. Factory failure
ve geçersiz seed budget senaryoları fit üretmedi. Önceki geliştirme turları bu
son kabul bütçesine dahil değildir; gerçek tekrarlar loglarda ayrı korunur.
API limiti 2..8 seed, seed başına en çok10 epoch/256 train/eval sample ve256
batch/epoch; duvar saati timeout'u veya kullanıcı factory sandbox'ı yoktur.

P27 örneği 12 sample'da sign-flip işlemini gerçekten12 kez çağırdı. Accuracy
1→0 (−100 pp); absent class dahil macro-F1 .5→0 (−50 pp). Perturbed input,
IDs, labels ve metadata içerikleri JSON'da saklanır. Şekil/count/metadata
uyuşmazlığı ve perturbation/inference failure reddedilir. Seed'li gerçek
Gaussian callable örneğinin yeniden üretilebilirliği de doğrulandı.

P28 dört CPU recipe'si gerçek inference ile geçti: FP32, FP64, BF16 ve
dynamic qint8 Linear. Kabul örneğinde accuracy1→1, delta0. Her batch'in native
output dtype'ı kontrol edilir; INT8'de her batch profiler'ı gerçek
`quantized::linear_dynamic` görmelidir. Skipped quantized path ve dtype fallback
negatif testleri geçti. BF16 P26 raw capture için float32'ye genişletilir;
tüm operator'ların BF16 olduğu iddia edilmez. Sıfır baseline relative change
null; .5→.6 örneğinde +10 pp ve +20 relative-percent bağımsız doğrulandı.

P29 örneğinde iki farklı gerçek model aynı12 sample/split/precision/protocol
ile evaluate ve benchmark edildi. Bir warmup, 2 repeat×3 measured iteration
vardır. Accuracy0→1, +100 pp; 12 discordant pair için exact two-sided
p=2/4096=0.00048828125. Independence açık caller attestation'ıdır; veri hash'i
bu varsayımı ispatlamaz. En az10 discordant pair ve accuracy gerekir.
Macro-F1, yetersiz pair veya attestation yokluğunda p_value/significant null.
İki timing repeat bağımsız deney sayılmadı; timing significance null.
Pareto ve latency regression gerçek pooled mean latency'den türetildi;
test belirli bir timing kazananını zorlamadı. Noise/load/thermal control veya
genel speedup iddiası yoktur.

Benchmark input hash/dtype/shape/batch hesabı kalite cohort'una bağlanır.
Model tensor byte witness'ı P26 ve benchmark'ın farklı hash biçimlerini yeniden
üretir; structure/policy witness'ları model hash'ine katılır. Başka ağırlık,
architecture, runtime, host, device, data, protocol veya precision ile kaynak
karıştırma negatif testleri geçti. Host adı fiziksel unit UUID'si değildir.
En çok16 MiB model tensor byte witness kabul edilir. Aynı API'de FP64,
registered float/integer buffer, mixed training modes ve state/RNG preservation
gerçek CPU modeliyle doğrulandı.

## Sonuç sözleşmesi ve dosyalar

- `hyso/quality.py`: immutable EvaluationCohort, QualityExperimentResult ve
  dört opt-in API; native kaynak validation ve bağımsız analysis derivation.
- `hyso/__init__.py`: additive public export'lar.
- `tests/test_quality_experiments.py`: 55 kalıcı kabul testi/parametrizasyonu.
- `docs/design/p24-p27-p28-p29-quality-experiments.md`: API, kaynak bütçeleri,
  istatistik kuralları ve unsupported kapsamlar.
- `docs/design/hyso-analysis-optimization-contract.md`: yalnız dört feature
  satırı, bunların sayımı22/6/7 ve ilgili geliştirme sırası notu.
- Bu rapor ve `p24-p27-p28-p29-evidence/`: 8 gerçek JSON/CSV result çifti,
  test logları ve byte/SHA256 manifest.

Yeni `hyso.quality_experiment` schema_version1 mevcut formatlardan ayrıdır.
UUID/source identity/provenance/class metadata ve kaynak SHA256 tutulur;
import raw native sonuçları doğrular ve analysis'i yeniden hesaplar. Immutable
snapshot, JSON round-trip, source/analysis tamper ve no-clobber geçti. CSV
provenance/analysis özetidir; eksiksiz raw round-trip JSON içindir. Hash signed
authenticity değildir; tüm kaynak ve hash'leri yeniden üretebilen saldırgana
karşı çalıştırma kanıtı/attestation sunulmaz.

Model/optimizer/history parity gerçek kontrol fit'iyle eşit bulundu. Eğitim
dışındaki API'lerde özgün model parameter/buffer/mode/device ve RNG korundu;
failure cleanup RNG finally yolu da test edildi. Arbitrary callable dış yan
etkileri veya hidden Python state rollback sandbox'ı yoktur. Held-out üyeliği
ve preprocessing ID'nin semantiği caller sorumluluğundadır; cohort gerçek
input bytes/label/ID sırasını doğrular.

## Çalıştırılan komutlar

Windows, Python3.10.18, PyTorch2.5.1+cu121, Conda `torch_gpu`.
`tests/conftest.py` CUDA görünürlüğünü kapatır ve CPU thread sayısını1 yapar.

```powershell
$env:HYSO_QUALITY_EVIDENCE='docs/audit/p24-p27-p28-p29-evidence-final-55'
& C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe -m pytest tests/test_quality_experiments.py -q --tb=short
& C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe -m pytest tests/test_evaluation.py tests/test_performance_sweeps.py tests/test_benchmark_cpu.py tests/test_benchmark_comparison.py tests/test_benchmark_export.py tests/test_benchmark_result.py tests/test_benchmark_state_safety.py tests/test_public_api.py tests/test_published_api_compatibility.py -q --tb=short
& C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe -m pytest tests/test_dataloader_protocol.py tests/test_performance_dataloader.py tests/test_performance_training.py tests/test_training_details.py tests/test_training_throughput_protocol.py -q --tb=short
git diff --check
git diff --stat
git status --short
```

İlk komutun export klasörü kabul sonrası `p24-p27-p28-p29-evidence` olarak
yeniden adlandırıldı. Tekrar evidence üretmek için yeni/boş bir klasör seçilmeli;
mevcut artifact'lar varsayılan olarak üzerine yazılmaz. Geliştirme turlarının
ara artifact'ları silinmeden `%TEMP%/hyso-quality-exploratory-20261023` içine
taşındı. Son test stdout/stderr logları aşağıdadır.

| Koşu | PASS | FAIL | ERROR | SKIP | Süre |
|---|---:|---:|---:|---:|---:|
| Son P24/P27/P28/P29 | 55 | 0 | 0 | 0 | 9.41 s |
| P26/P08/benchmark/API regresyonu | 420 | 0 | 0 | 0 | 25.01 s |
| P19/P20/P21/P22 CPU regresyonu | 222 | 0 | 0 | 0 | 58.80 s |
| **Son kabul toplamı** | **697** | **0** | **0** | **0** | |

420 içinde mevcut46 P26 testi korunur. Warning'ler mevcut GradScaler ve
benchmark testlerinin deprecation kullanımlarıdır; bu kaynaklara dokunulmadı.
Son yeni paket5, training regression199 ve benchmark regression2 warning verdi.

Geliştirme geçmişi saklandı; tekrarlar kabul toplamına eklenmedi:
ilk tur22 PASS/3 FAIL/8 ERROR; düzeltme sonrası33 PASS; genişletilmiş tur50
PASS/1 FAIL; sonraki51,52,54 ve son55 PASS. İlk hatalar benchmark mean'in
yanlış dict yolu, absent class macro-F1 test beklentisi ve model construction
öncesi RNG snapshot'ıydı. Genişletilmiş turdaki fixture standart olmayan loss
ile Trainer admission'a takıldı; gerçek train-mode forward exception fixture'ı
ile değiştirildi. Son55 test, default collate admission ve yalnız CPU
generator'ını seed etme düzeltmelerinden sonra tekrar55 PASS verdi. Bu ek
doğrulamalar kabul toplamına ikinci kez eklenmedi. Evidence üreten55 testlik
tur `artifact-tests.txt`, son kaynak turu `final-tests.txt` olarak korunur.
Ürün kodunda başarı uydurulmadı.

Loglar: [son kabul](p24-p27-p28-p29-evidence/final-tests.txt),
[evaluation/benchmark](p24-p27-p28-p29-evidence/regression-evaluation-benchmark.txt),
[training](p24-p27-p28-p29-evidence/regression-training.txt),
[ilk hatalar](p24-p27-p28-p29-evidence/initial-tests.txt),
[genişletilmiş ilk tur](p24-p27-p28-p29-evidence/expanded-initial-tests.txt).
Gerçek sonuçlar: [P24 reached](p24-p27-p28-p29-evidence/p24-reached.json),
[P24 censored](p24-p27-p28-p29-evidence/p24-censored.json),
[P27](p24-p27-p28-p29-evidence/p27-paired.json),
[P28 INT8](p24-p27-p28-p29-evidence/p28-int8_dynamic.json),
[P29](p24-p27-p28-p29-evidence/p29-comparison.json),
[manifest](p24-p27-p28-p29-evidence/manifest.json).

## NOT_RUN ve kapsam dışı

CUDA evaluation/precision/benchmark/training: **NOT_RUN / bu yeni adapter'larda
unsupported**. Native P21 spawn/crash/hang paketleri bu görevde **NOT_RUN**;
P21 kaynakları değişmedi, ilgili mevcut CPU protocol regresyonları çalıştı.
Detection/segmentation, distributed/multiworker/custom sampling, genel quant
graphs, static calibration, censor-adjusted survival/CI, timing significance,
macro-F1 significance ve throughput/memory Pareto unsupported. Bu alanlara
tahmini değer yazılmaz. Desteklenen scope için açık kabul engeli kalmadı.

## Git kontrolleri

Son `git diff --check`, `git diff --stat`, `git status --short` çıktıları
aynı evidence klasöründeki `git-checks.txt` dosyasındadır. Diff stat tracked
dosyaları gösterir; yeni kaynak/test/doc/artifact'lar status'ta untracked'dir.
`git diff --name-only -- hyso/evaluation.py hyso/performance hyso/benchmark`
boştur. HEAD değişmedi; commit/push yapılmadı.
