# P26 classification evaluation kabulü — 23 Ekim 2026

**P26: implemented — finite CPU binary/multiclass classification kapsamı.**
Detection/segmentation ve aşağıdaki diğer unsupported kapsamlar bu statüye
dahil değildir. Başlangıç HEAD `8b14fb6`, branch `feature/hyso-models-next`.
P21 milestone, mevcut 17 implemented özelliğin kaynakları ve 15 scratch P21
dosyası korundu. Commit/push yapılmadı.

## Değişen ve eklenen dosyalar

- `hyso/evaluation.py`: P26 public API, independent metrics, model evaluation,
  immutable/versioned EvaluationResult ve JSON/CSV export.
- `hyso/__init__.py`: P26 root public export'ları; `hyso.evaluation` da public.
- `tests/test_evaluation.py`: 46 kalıcı kabul testi/parametrizasyonu.
- `docs/design/p26-classification-evaluation.md`: kullanım ve kesin kapsam.
- `docs/design/hyso-analysis-optimization-contract.md`: yalnız P26 capability
  satırı ve P26'nın eklenmesine bağlı 18/7/10 sayımı.
- Bu rapor ve `p26-evaluation-evidence/` altındaki log/örnek artifact'lar.

## Uygulanan davranış

Binary/multiclass, rows=true/columns=predicted confusion matrix; accuracy,
per-class precision/recall/F1/support; macro, micro, support-weighted ortalamalar.
Class isimlerinin açık sırası indeksleri belirler. Known confusion ve dengesiz
örnekler bağımsız integer/Fraction referanslarıyla doğrulandı.

Binary `logits >= 0` kuralı mevcut `Trainer._loss_and_correct` ile aynıdır;
sıfır pozitif sınıftır. Multiclass argmax eşitliğinde ilk indeks seçilir.
Accuracy dahil oranlar 0..1'dir. Nonempty sıfır payda açık `zero_division=0|1`
ile çözülür; absent sınıflar macro'ya katılır. Empty değerlendirme `empty`,
support 0 ve null oranlar üretir; complete başarı olarak gösterilmez.

NaN/Inf, soft/geçersiz label, shape, class-map boyutu/sırası sözleşmesi, unknown
task ve yanlış veri türleri fail-closed işlenir. Detection/segmentation için
classification metrikleri üretilmez; capability unsupported'dır.

Model evaluation eğitim başlatmaz; verilen modelin bağımsız deepcopy kopyasını
eval/inference_mode içinde çalıştırır. Özgün model taşınmaz ve mode/parametre/
buffer/gradient/version state değiştirilmez. Object/storage alias copy ve
inference sırasında kopyanın tensor state veya mod değişimi reddedilir.
Python/NumPy/Torch RNG ve DataLoader generator state, length/deepcopy/forward/
dataset/collate hataları dahil finally içinde geri yüklenir. Standart eager
model kapsamı doğrulandı; arbitrary kullanıcı kodu için sandbox iddiası yoktur.

Finite sequential map-style standart DataLoader, workers=0 ve drop_last=False
ile beklenen sample/batch sayıları ve her batch/tail boyutu doğrulanır. Eksik
sample, fazladan/yanlış batch veya model exception geçerli sonuç üretmez.
Split, ordered class metadata, değerlendirme koşulları, raw labels/logits,
model tensor-state hash/type ve provenance saklanır. Eğitim kaynağına otomatik
bağ kurulmaz (`training_source=null`).

EvaluationResult yeni ve ayrı `hyso.evaluation.classification` schema v1'dir;
benchmark/performance v1 formatları değişmedi. JSON import ham kanıttan metrikleri
yeniden türetir ve kaynak hash'ini doğrular. Immutable snapshot, type-sensitive
metric tamper (True/1 ayrımı dahil), raw/source/class/accounting tamper ve
no-clobber kontrolleri geçti. CSV class/average özetleri, confusion hücreleri,
source identity ve provenance içerir; formula hücreleri escape edilir.

## Gerçek test komutları ve sonuçlar

Ortam: Windows, Conda `torch_gpu`, Python 3.10.18, PyTorch 2.5.1+cu121.
Testlerde CUDA görünürlüğü kapalıdır; inference CPU'da çalıştırıldı.

```powershell
& 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe' -m pytest tests/test_evaluation.py -q --tb=short
& 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe' -m pytest tests/test_dataloader_protocol.py tests/test_performance_dataloader.py tests/test_performance_training.py tests/test_training_details.py tests/test_training_throughput_protocol.py -q --tb=short
```

| Koşu | PASS | FAIL | SKIP | Süre |
|---|---:|---:|---:|---:|
| İlk P26 turu | 34 | 11 | 0 | 18,45 s |
| Provenance düzeltmesi sonrası | 45 | 0 | 0 | 12,42 s |
| Son P26 kabulü | 46 | 0 | 0 | 13,11 s |
| P19/P20/P21/P22 CPU regresyonu | 222 | 0 | 0 | 46,45 s |

**Son kabul toplamı: 268 PASS / 0 FAIL / 0 SKIP.** Önceki turlar tekrar bu
toplama eklenmedi. Regresyonda mevcut GradScaler kullanımından 199 deprecation
warning alındı; ilgili Trainer kodu değiştirilmedi.

İlk 11 başarısızlık gizlenmedi: `torch.__version__` bir str alt sınıfı olan
TorchVersion nesnesiydi; strict plain JSON validator reddetti. P26 provenance
alanında native `str` olarak kaydetme düzeltmesi yapıldı. Sonraki bütün kabul
testleri geçti. CSV confusion/provenance kapsamı genişletildi; length exception
ve alias-copy kabulü eklendi. Bunlar son 46 testlik turda çalıştırıldı.

Loglar: [ilk tur](p26-evaluation-evidence/initial-tests.txt),
[ara tur](p26-evaluation-evidence/intermediate-tests.txt),
[son P26](p26-evaluation-evidence/final-tests.txt),
[regresyon](p26-evaluation-evidence/regression.txt).
Gerçek API export örnekleri:
[known JSON](p26-evaluation-evidence/known-evaluation.json),
[known CSV](p26-evaluation-evidence/known-evaluation.csv),
[CPU model JSON](p26-evaluation-evidence/cpu-evaluation.json),
[CPU model CSV](p26-evaluation-evidence/cpu-evaluation.csv).

## Kalan kapsamlar

- Detection/segmentation, multilabel, probability encoding, soft labels:
  unsupported; classification sonucu bu görevlere uygulanmaz.
- Model inference: CPU, dense real tensors ve finite float32/64 giriş/çıktı.
  CUDA evaluation **NOT_RUN / unsupported**; otomatik device/cast fallback yok.
- Multiworker/persistent, shuffle/custom/replacement sampling, IterableDataset,
  pin_memory ve custom batching: bu finite coverage/RNG adapter'ında unsupported.
  P21 native spawn/crash testleri bu görevde **NOT_RUN**; P21 kaynakları aynı,
  ilgili CPU protocol/replay regresyonları 222 test içinde çalıştırıldı.
- Dataset mutable Python state veya external side effect rollback, keyfi
  deepcopy/forward hook davranışı ve callback hang timeout'u sandbox edilmez.
  Source model hash registered tensors içindir; arbitrary Python durumunun veya
  dataset/preprocessing içeriğinin tam kimliği otomatik kanıtlanmaz. Caller'ın
  source_id'si veri sürümünü bildirebilir; hash signed authenticity değildir.
- En fazla 1.000.000 sample, 100.000 batch, 1000 sınıf; raw logits capture
  bellek maliyeti O(N*C). Bu kapsam dışında sonuç uydurulmaz.

Bu sınırlamalar açık capability/admission sınırlarıdır; P26'nın doğrulanan
finite CPU classification kapsamı tamamlandı. Sonuç başka eğitim çalışmasına
otomatik bağlanmadı ve mevcut 17 implemented özelliğe geçilmedi.

## git diff --check

Çıkış kodu 0; stdout boş, whitespace hatası yok. Git'in CRLF satırsonu uyarısı
kontrolü başarısız yapmadı. `git diff --name-only -- hyso/performance` boş;
P21/performance kaynakları değiştirilmedi.

## git diff --stat

Stat yalnız tracked dosyaları gösterir; yeni dosyalar dosya listesinde ve
aşağıdaki untracked status satırlarında bulunur.

```text
 docs/design/hyso-analysis-optimization-contract.md | 6 +++---
 hyso/__init__.py                                   | 1 +
 2 files changed, 4 insertions(+), 3 deletions(-)
```

## git status --short

```text
 M docs/design/hyso-analysis-optimization-contract.md
 M hyso/__init__.py
?? docs/audit/p26-classification-acceptance.md
?? docs/audit/p26-evaluation-evidence/
?? docs/design/p26-classification-evaluation.md
?? hyso/evaluation.py
?? scratch_p21_dataloader.py
?? scratch_p21_delivery_audit.py
?? scratch_p21_failure.py
?? scratch_p21_integrity.py
?? scratch_p21_ipc.py
?? scratch_p21_ipc_v2.py
?? scratch_p21_live_delivery.py
?? scratch_p21_multistage.py
?? scratch_p21_multistage_spawn.py
?? scratch_p21_result_integration.py
?? scratch_p21_stages.py
?? scratch_p21_trace_integration.py
?? scratch_p21_variable_stages.py
?? scratch_p21_worker_core.py
?? scratch_p21_worker_shutdown.py
?? tests/test_evaluation.py
```
