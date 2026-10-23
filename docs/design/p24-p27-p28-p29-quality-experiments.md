# P24/P27/P28/P29 — finite CPU quality experiments

Public API `hyso` veya `hyso.quality` üzerinden sunulur. P26 ve benchmark
schema v1 değişmez; yeni immutable `QualityExperimentResult` formatı
`hyso.quality_experiment`, schema_version=1, feature_id=P24/P27/P28/P29'dur.

## Ortak veri ve sonuç sözleşmesi

```python
from hyso import (EvaluationCohort, evaluate_robustness,
                  evaluate_precision_quality, compare_performance_quality,
                  measure_time_to_quality, QualityExperimentResult)

cohort = EvaluationCohort(
    inputs, targets, sample_ids=sample_ids, class_names=["negative", "positive"],
    split="holdout-v1", preprocessing_id="normalize-v2", task="multiclass",
)
result = evaluate_precision_quality(model, cohort, candidate="int8_dynamic")
result.to_json("quality.json")
result.to_csv("quality.csv")
# Re-import validates sources, hashes, native results and derived analysis.
restored = QualityExperimentResult(result.to_dict())
```

Cohort içerik snapshot'ıdır: CPU float32/64 input, hard labels, unique ordered
string sample IDs, ordered class map, split ve preprocessing ID. En çok 4096
sample/1.000.000 input element; boş veya nonfinite veri reddedilir. Binary
logits>=0 pozitif; absent classes/zero division=0 ve macro-F1 P26 ile aynıdır.
Accuracy/F1 0..1, delta percentage point=100×(candidate−baseline), relative
percent=100×(candidate−baseline)/baseline; baseline0 için relative null.
Detection/segmentation/multilabel ve CUDA bu adapter'larda unsupported'dır.

Model inference P26'nın isolated deepcopy/eval/inference_mode adapter'ını
kullanır. Kullanıcının model ağırlığı, buffer, gradient, mode ve cihazı
değiştirilmez; dış Python/NumPy/Torch RNG finally ile korunur. P24 factory'nin
oluşturduğu model/optimizer'lar gerçek eğitim için değişir. Factory, forward
ve corruption callable güvenilir kullanıcı kodudur; process sandbox veya
keyfi callable hang timeout'u sağlanmaz. Hidden Python state/external side
effect rollback ve model kaynak kodunun tam semantic kimliği iddia edilmez.

JSON; input content, IDs, labels, raw P26 outputs, native source results,
provenance, UUID ve source SHA256 içerir. Analysis import sırasında yeniden
türetilir; tutarsız hash/analysis/embedded source reddedilir. Bu bütünlük
kontrolüdür, signed authenticity veya çalıştırma attestation'ı değildir:
tüm ham veriyi ve hash'leri yeniden üretebilen kişi sahte artifact yapabilir.
CSV; source identities/hashes, split/classes/preprocessing/protocol ve analysis
özetidir. Raw input/model witnesses ve eksiksiz round-trip JSON'dadır; CSV
import yoktur. Export varsayılan no-clobber; overwrite ancak açık True ile.

## P24 — time to target

```python
# factory(seed) -> (fresh standard CPU Trainer, DataLoader(TensorDataset(...)))
result = measure_time_to_quality(
    factory, cohort, seeds=[11, 23, 37], target=0.9,
    metric="accuracy", epochs=3, cadence=1,
)
```

2..8 distinct seed; seed başına 1..10 epoch, 1..256 train ve eval sample,
en çok 256 train batch/epoch. Loader sequential, workers=0, drop_last=False,
default collate ve pin_memory=False olmalıdır.
Model storage'ları bağımsız, optimizer state boş, Trainer fresh olmalı;
callbacks/EMA/early stopping kabul edilmez. Seed her factory çağrısından önce
Python/NumPy/Torch CPU generator'ına uygulanır. Aynı train content, model structure, loss
ayarları, optimizer/scheduler planı, grad_clip ve batching/training protocol
zorunludur. Başlangıç ağırlıklarının seed ile değişmesi beklenebilir.

Epoch0 evaluation fit'ten önce; sonraki evaluation her cadence ve son epoch'ta
gerçekleşir. P19 training_end, epoch_end callback'ten önce kaydedilir: evaluation
clock bu sınırdan sonra ve epoch end'den önce olmalıdır. Checkpoint'in model
hash'i P26 kaynağına bağlanır. Süre yalnız P19 training_s toplamıdır: evaluation,
model copy, factory, export ve fit finalization dahil değildir. Instrumentation
overhead çıkarılmaz; isolated eval'in cache etkisi düzeltilmez.

İlk gözlenen >=target checkpoint raporlanır. Gerçek geçişin checkpoint'ler
arasındaki tam anı bilinmez; interval_low/high saklanır. Başlangıç hedefi
sağlıyorsa training time0; hiçbir checkpoint sağlamıyorsa right_censored ve
gerçek son training horizon kaydedilir. Tam epoch bütçesi hedef sonrasında da
çalışır; bu yalnız ilk crossing analizi içindir. Mean/sample variance yalnız
reached runs içindir; censor-adjusted population mean, Kaplan-Meier veya CI
üretilmez. Split adı açık ve train/training olamaz; train dataset'inde sample
IDs olmadığından held-out üyeliği/leakage semantiği caller sorumluluğundadır.
Factory/training/eval failure veya eksik run/checkpoint sonuç üretmez.

## P27 — robustness

```python
def gaussian(sample_batch, generator, severity):
    import torch
    return sample_batch + severity * torch.randn(sample_batch.shape, generator=generator)

result = evaluate_robustness(model, cohort, gaussian,
    corruption_id="gaussian-v1", severity=0.1, seed=17)
```

Callable her sample için tam bir kez `(one_sample_batch, generator, severity)`
ile çalışır. Batch size1 input shape/dtype/CPU device korunmalıdır; sınırlı
finite çıktı zorunludur. Ortak generator seed'i sabittir. ID/label/order/class
map/split/preprocessing kopyalanır, içerik hash'leri ayrı saklanır. Corruption
ID, severity[0,1], seed, gerçek call count, callable qualified name ve mümkünse
inspect source SHA saklanır; source erişilemiyorsa null açık kalır.
İki inference aynı seed'den başlatılır ve aynı model tensor hash'i zorunludur.
Accuracy ve macro-F1 paired delta hesaplanır. Callable'ın etiket koruyan veya
adına uygun fiziksel corruption olması otomatik ispatlanmaz; kimlikli gerçek
input/output snapshot korunur. Hazır corruption kataloğu yoktur.

## P28 — precision quality

FP32 baseline ve FP32/FP64/BF16/dynamic INT8 candidate, aynı master modelin
P08 `_prepare` recipe'siyle oluşturulan bağımsız kopyalarıdır. Veri, label,
split, class sırası ve preprocessing aynıdır; dtype cast policy'nin parçasıdır.
Her inference batch'te native output dtype doğrulanır; BF16 output P26 raw
capture için float32'ye genişletilir. BF16 karışık operator graph'ının tamamı
BF16 diye sunulmaz. INT8 yalnız Sequential içindeki Flatten/Linear/ReLU/Identity
recipe'sidir; qint8 weight snapshot ve **her gerçek batch'te** profiler'da
`quantized::linear_dynamic` zorunludur. Fallback reddedilir. Packed quantized
weights digest'i registered buffer witness ile P26 model hash'ine katılır.
Native dtype/operator ve verified batch count sonuçta saklanır. Bu profiler
scope latency ölçümü değildir. Dynamic activation quantization calibration
dataset gerektirmez; static calibration/graph quantization desteklenmez.
P08 conversion/load maliyeti ve hız sonucu kalite yerine geçmez.

## P29 — performance/quality comparison

```python
result = compare_performance_quality(
    baseline_model, candidate_model, cohort, precision="fp32", iterations=5,
    metric="accuracy", min_quality=0.85, max_quality_drop=0.01,
    max_latency_regression=0.05, independent_samples=False,
)
```

Resident CPU NCHW FP32/FP64 scope; iki model aynı precision kullanır. Her model
için gerçek P26 inference ve BenchmarkRunner çalışır: 1 warmup, 2 repeat,
repeat başına 2..50 measured iteration. Cohort tek batch olarak iki scope'ta
da aynıdır. Model kopyaları P08 ile hazırlanır; legacy compare_results API'si
değişmez. Uyumlu runtime/device/input/protocol/precision ve aynı host_id gerekir.
Host adı fiziksel cihaz UUID'si veya exclusive-load attestation'ı değildir.
Quality/benchmark input hash ve batch muhasebesi eşleştirilir. En çok16 MiB
model tensor bytes witness, legacy P26/benchmark farklı hash formatlarını
bağımsız yeniden üretir; structure/policy witness'ları da P26 hash'ine katılır.
Farklı ağırlık veya kayıtlı architecture descriptor ile kaynak karıştırılamaz.

Pareto: pooled mean latency azalır ve kalite artar/eşitse, en az biri strict
iyiyse dominance vardır. Aksi halde baseline veya neither; tolerance dominance'a
uygulanmaz. Constraints absolute min quality, fractional quality drop ve
fractional latency regression sınırlarıdır; regression için strict > kullanılır.
Exact eşitlik regression değildir. Gürültülü timing'de dominance yalnız
gözlenen sample özetine ilişkindir; genel speedup veya significance iddiası yoktur.

Accuracy için caller örneklerin bağımsızlığını açıkça bildirirse ve en az10
discordant pair varsa two-sided exact paired McNemar p-value hesaplanır:
`min(1, 2*sum(C(n,k), k=0..min(wins,losses))/2**n)`, alpha=.05.
Bu attestation veri içeriğinden çıkarılmaz; duplicate ID reddi istatistiksel
bağımsızlık kanıtı değildir. Macro-F1, yetersiz n, bildirilmeyen independence
ve bağımsız timing experiment yokluğu için p_value/significant null kalır ve
neden saklanır. Timing significance, throughput/memory Pareto, CUDA ve mixed
precision comparisons bu scope'ta unsupported'dır.
