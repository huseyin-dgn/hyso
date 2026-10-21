# P23 — Training memory observations

`measure_training_memory` bir fresh standard `Trainer` üzerinde **bir gerçek fit**
çalıştırır. Model, optimizer, history ve RNG normal eğitimdeki gibi ilerler.
İkinci forward/probe/replay veya state restoration yapmaz. P23 **PARTIAL**:
saved-activation lifetime ve izole edilmiş gerçek window peak adapter'ı yoktur.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import measure_training_memory, TrainingMemoryProtocol

model = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
trainer = Trainer(model, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
                  accum_steps=2)
train = DataLoader(TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])),
                   batch_size=2)
result = measure_training_memory(trainer, train, epochs=1,
    protocol=TrainingMemoryProtocol(device="cpu", max_samples=10000,
                                    max_boundaries=10000, max_tensor_records=100000))
data = result.to_dict()
assert data["observations"][0]["logical"]["summary"]["optimizer"]["logical_bytes"] == 0
assert data["observations"][-1]["logical"]["summary"]["optimizer"]["logical_bytes"] > 0
assert data["activation"]["value"] is None
```

CUDA için model/Trainer önceden uygun cihaza yerleştirilir; protocol device açık
index ile örneğin `cuda:0` olmalıdır. Native allocator desteklenir; alternatif
allocator, custom optimizer veya unsupported tensor temsilleri doğrulanmış gibi
sunulmaz. CUDA availability için CPU fallback yoktur. Mevcut P19/P20/P21/P22
çağrıları, Trainer constructor/fit imzaları ve checkpoint v1 değişmez.

## Üç bağımsız domain

| Alan | Ölçülen | Sınır |
|---|---|---|
| logical | Parameter, frozen subset, registered buffer, mevcut gradient, SGD/Adam/AdamW state tensor metadata | `numel * element_size`; fiziksel RAM/VRAM değildir |
| RSS | P10 native current-process RSS/HWM adapter, baseline/end ve timestamped sampled peak | Runtime, loader main process, observer dahil; worker RSS unsupported |
| CUDA | Gerçek allocated/reserved, önceden var olan lifetime peaks/counters, boundary sampled maxima | Process/device Torch allocator; physical/device-wide VRAM değildir |

Her component içinde aynı tensor objesi bir kez sayılır; tied isimler `names`
listesindedir. Ayrı view objeleri logical olarak ayrı sayılır. Snapshot-local
`tensor_id`/`storage_id`, storage offset/stride/dtype/device/shape ve deduplicated
**full storage extent** ayrıca sunulur. Storage bytes da physical resident bytes
değildir. Frozen bytes parameter total'ının alt kümesidir. Component toplamları
veya farklı boundary'ler birbirine eklenerek RAM/VRAM hesaplanmaz. Optimizer state
gerçekten mevcutsa sayılır; ilk applied step öncesi Adam momentleri varsayılmaz.
Non-tensor optimizer state değerleri plain metadata'dır; Python heap byte maliyeti
null'dır. EMA/scaler/scheduler tensorları logical kapsamda değildir.

Boundary'ler ready, epoch_ready, pre_microbatch, post_forward, post_backward,
pre_attempt, post_scaler_step, post_zero_grad, post_attempt, epoch_end ve
segment_end'dir. Epoch/microbatch/window/attempt ID'leri embedded, hash-associated
P19 ledger ile aynıdır. Post_backward accumulation gradientlerini; post_scaler_step
zero-grad öncesi optimizer state'i; post_attempt applied/skipped kararını gösterir.
Post_forward snapshot saved activation bytes ölçmez: `activation=null/unsupported`.
Activation peak VRAM hakkında çıkarım yapılmaz; autograd saved-tensor hook eklenmez.

## Scope ve güvenlik

Varsayılan ve tek desteklenen `peak_reset_policy="never"`; `scope="in_process_fit"`.
`reset_peak_memory_stats`, `empty_cache`, global GC kontrolü veya context reset
çağrılmaz. Gerçek window/incremental peak **null/unsupported**; lifetime peak ile
baseline farkı window peak gibi sunulmaz. P09 single-forward reset sözleşmesi
değişmez. Ortak benchmark lock aynı process'te çakışan ölçümleri reddeder. Caller
process/device kullanımını exclusive tutmalıdır; dış allocator reset'i tutarsız
peak ledger ile reddedilir. Keyfi başka thread/process attribution yapılamaz.

Bu API workload için hard RAM/VRAM budget veya timeout sağlamaz. Protocol sınırları
**retained observation** bütçeleridir; trusted, sonlanan workload gerekir. Production
factory/isolated worker API'si yoktur; canlı Trainer/tensor subprocess'e aktarılmaz.
Test supervisor'ının izolasyonu production özelliği olarak gösterilmez. Uzun veya
OOM riski taşıyan training için ayrıca güvenilir recipe, OS resource guard ve
hard-timeout tasarımı gerekir. Reset iznini değiştirmek desteklenmeyen protokol
hatasıdır; otomatik reset'e dönüştürülmez.

RSS sampled peak kısa tahsisleri kaçırabilir; OS HWM ve CUDA lifetime peaks
ölçüm öncesini içerir. RSS query timestamps/gaps, query süreleri ve actual sample
sayısı JSON'dadır. Backend ölçüm sırasında kaybolursa önceki raw örnekler korunur,
summary null/unsupported olur. Sampler bütçesi dolarsa başarılı incomplete result
üretilmez. Failure sonrası eğitim ilerlemiş olabilir; otomatik retry yoktur.

Ready snapshot fit/resume restore öncesindedir; resume suffix epoch_ready snapshot'ı
restore sonrasıdır. RSS full-fit scope validation/callback/explicit checkpoint I/O
ve best-weight finalization'ı içerir. Embedded P19 epoch timing daha dar scope'unu
korur; checkpoint I/O P19 time'a eklenmez. CUDA boundary query'leri host zamanındadır;
her boundary'de ek sync yapılmaz, P19 selected-stream drain'leri korunur. Query
maliyeti raporlanır; toplam instrumentation overhead bilinmez/null'dır. Cache/RSS
artışı tek başına leak kanıtı değildir.

## Export ve tracking

`result.to_json(path)` canonical tam kayıt; `to_csv(path)` boundary/component ve
allocator snapshot tablosudur. RSS raw samples JSON'da kalır. Export atomic ve
no-clobber'dır; overwrite yalnız explicit `overwrite=True` ile olur. Immutable
payload/defensive copy ve schema/source/evidence SHA256 doğrulaması vardır; checksum
bir güvenlik imzası değildir. Ölçüm içinde export veya SQLite write yapılmaz.

Ölçüm bittikten sonra JSON/CSV'yi mevcut Experiment Manager `training` run'ına
`add_artifact(..., ownership="copy")` ile ekleyip hash verify edin; yalnız supported
scalar değerleri generic metric olarak loglayın. Unsupported/null metrikler sıfır
değildir ve loglanmamalıdır. Tracking failure başarılı training'i tekrar çalıştırma
gerekçesi değildir. Yeni run kind, tablo veya migration yoktur.
