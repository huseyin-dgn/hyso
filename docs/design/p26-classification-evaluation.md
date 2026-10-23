# P26 classification evaluation

Public API, `hyso` veya `hyso.evaluation` üzerinden:
`evaluate_predictions`, `evaluate_classification`, `EvaluationResult`,
`evaluation_capabilities`.

```python
from hyso import evaluate_predictions, evaluate_classification

known = evaluate_predictions(
    [0, 1, 1], [0, 0, 1], class_names=["negative", "positive"],
    split="holdout", task="binary", source_id="labels-v1",
)
# Model zaten CPU'da olmalı; loader standart, sequential, workers=0,
# drop_last=False olmalı. Her batch (inputs, targets) tensor çiftidir.
observed = evaluate_classification(
    model, labeled_loader, class_names=["negative", "positive"],
    split="holdout", task="binary", source_id="holdout-v1",
)
observed.to_json("evaluation.json")
observed.to_csv("evaluation.csv")
```

## Metrik sözleşmesi

Class names sırası indeksleri `0..C-1` tanımlar; isimler benzersizdir. Binary
tam iki sınıf gerektirir, pozitif sınıf indeks 1'dir. Model binary targets,
Trainer ile aynı floating hard-label `(B,)` veya `(B,1)` kuralını izler.
Raw prediction API'sinde target vektörü integer class ID; binary floating 0/1
de kabul edilir. Multiclass model targets `torch.long (B,)` olmalıdır.
Raw evidence girdileri CPU Torch tensor veya native list/tuple'dır; arbitrary
NumPy/container dönüşümü otomatik yapılmaz.

`prediction_type="labels"` hard label vektörü; `"logits"` binary `(N,)`/`(N,1)`
ve multiclass `(N,C)` logits alır. Binary tahmin `logit >= 0` (sıfır pozitiftir),
multiclass argmax ve eşitlikte ilk indeks. Probability encoding veya soft labels
otomatik algılanmaz. Tensor logits floating olmalıdır; NaN/Inf reddedilir.

Confusion matrix satırları gerçek, sütunları tahmin sınıfıdır. Accuracy ve tüm
oranlar **0..1** aralığındadır; Trainer'ın yüzde gösteriminden farklı olarak
burada oran saklanır. Her sınıf için TP/predicted precision, TP/support recall,
`2TP/(support+predicted)` F1 ve gerçek support hesaplanır. Macro bütün belirtilen
sınıfları, weighted gerçek support ağırlıklarını içerir; micro global TP/FP/FN
karşılığıdır ve single-label classification'da accuracy'ye eşittir.

Nonempty değerlendirmede sıfır payda açık `zero_division=0` veya `1` ile çözülür;
absent class da macro'ya katılır. Empty değerlendirme `status="empty"`, support
0, confusion 0 ve accuracy/precision/recall/F1 null üretir; başarı veya yüzde100
olarak sunulmaz. Detection/segmentation capability'si unsupported'dır.

## Model ve veri bütünlüğü

Opt-in model API'si bağımsız `deepcopy` kopyasını eval/inference_mode içinde
çalıştırır. Özgün modelin modları, parametre/buffer cihazı ve ağırlıkları
değiştirilmez; model taşınmaz, eğitim/optimizer/loss çağrısı yapılmaz. Alias
object/storage üreten copy reddedilir. Kopyanın registered tensor state'inin
veya eval modlarının inference sırasında değişmesi hata üretir.

Python, NumPy, Torch CPU ve daha önce initialize edilmiş CUDA RNG snapshot'ı
ile loader generator state başarı/exception yollarında geri yüklenir.
Kullanıcı kodunun keyfi external side effect'leri veya registered tensor state
dışındaki mutable Python durumları sandbox edilmez. Custom deepcopy, hooks ve
model/dataset kodu normal eager inference sözleşmesiyle uyumlu olmalıdır.

Yalnız standart DataLoader, map-style finite dataset, standart sequential
sampler/batching, `num_workers=0`, `drop_last=False`, `pin_memory=False` desteklenir.
Beklenen dataset sample sayısı, batch sayısı, her batch'in gerçek büyüklüğü ve
son tail doğrulanır. Shuffle/replacement/custom batching/IterableDataset ve
multiworker akışlar eksiksiz coverage/RNG kanıtı için ayrı adapter gerektirir;
bu sürümde reddedilir. Tensor girdiler finite CPU float32/64'tür. CUDA evaluation
desteklenmez; otomatik fallback yoktur. Arbitrary Python callback'lerinin hang
ve side effect davranışları için process sandbox veya timeout garantisi yoktur.

Capture en fazla 1.000.000 sample ve 100.000 batch'tir; dense confusion boyutu
en fazla 1000 sınıftır. Ham logits JSON için saklandığından bellek maliyeti
`O(N*C)` olabilir; caller max_samples ile daha küçük bütçe seçebilir.

## Artifact

`hyso.evaluation.classification`, schema v1, feature P26. Bu yeni format mevcut
benchmark/performance v1 formatlarından ayrıdır. Immutable snapshot; JSON yeniden
importunda ham etiket/logits üzerinden confusion ve metrikler türetilir, hash
ve batch muhasebesi doğrulanır. CSV per-class/average özetlerini ve confusion
hücrelerini kaynak/provenance metadata'sıyla verir. JSON/CSV overwrite varsayılan
kapalıdır, formula hücreleri escape edilir.

Evaluation UUID, source SHA256, explicit split, ordered class map, caller source
ID ve model inference'da model tensor state hash/type saklanır. Dataset type ve
sampling koşulları kaydedilir; dataset içerik kimliği veya preprocessing'in
yeniden üretilebilirliği otomatik kanıtlanmaz. Caller source_id bunun için kendi
veri sürümünü bildirebilir. Hash bütünlük kontrolüdür; imzalı authenticity kanıtı
değildir. `training_source=null`: başka eğitim çalışmasına otomatik bağ yoktur.
