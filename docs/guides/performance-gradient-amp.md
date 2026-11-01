# P25 — Gradient / AMP diagnostics

`measure_gradient_diagnostics` bir fresh standard Trainer üzerinde **bir gerçek
fit** çalıştırır. Parametre, optimizer/scaler, EMA, scheduler, history ve RNG normal
eğitimdeki gibi ilerler. İkinci backward/step/update veya otomatik rollback yoktur.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import measure_gradient_diagnostics, GradientDiagnosticsProtocol

model = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
trainer = Trainer(model, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
                  accum_steps=2, grad_clip=.1)
train = DataLoader(TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])),
                   batch_size=2)
result = measure_gradient_diagnostics(trainer, train, epochs=1,
    protocol=GradientDiagnosticsProtocol(max_attempts=10000, max_tensor_records=100000,
                                         chunk_elements=65536, observe_postclip=True))
data = result.to_dict()
assert [r["samples"] for r in data["observations"]] == [4, 1]
assert all(r["preclip"]["summary"]["finite"] for r in data["observations"])
assert all(r["optimizer_executed"] for r in data["observations"])
```

CUDA için model/optimizer/Trainer önceden aynı CUDA cihaza yerleştirilir; Trainer'ın
`device="cuda:0", amp=True` seçeneği native FP16 GradScaler yolunu etkinleştirir.
CPU fallback veya alternatif AMP backend desteği eklenmez. API yalnız standard
Trainer/native GradScaler kabul eder; Trainer'ın native SGD/Adam/AdamW admission
kuralları değişmez. Validation, checkpoint ve resume argümanları mevcut fit'e iletilir.

## Sınır ve norm sözleşmesi

Her gerçek accumulation window sonunda şu sıra korunur:

1. Native scaler unscale.
2. Trainer finite check ve gerçek window sample sayısıyla normalization.
3. P25 **preclip** snapshot.
4. Varsa Trainer native clipping; optional ayrı **postclip** snapshot.
5. Native scaler step, native execution hooks, scaler update.
6. Applied ownership'e göre EMA/OneCycle; native finally zero-grad.

`preclip.summary.global_l2.value`, `sqrt(sum(abs(g)^2))` normudur; `g` unscaled ve
sample-normalized gradienttir. `accum_steps` ile tekrar bölünmez. Gradient değerleri
diagnostics tarafından değiştirilmez. Her tensor bounded parçalarla bulunduğu
cihazda FP64'e çevrilir; max ile ölçeklenmiş kare toplamı ve host `math.hypot`
birleştirmesi kullanılır. Tam gradient CPU'ya taşınmaz. Scratch chunk limiti workload
RAM/VRAM güvenlik bütçesi değildir; element sayısı toplam hesap süresini sınırlandırmaz.

Native optimizer parametre objeleri bir kez sayılır; tied objenin isimleri aynı
kayıttadır. Ayrı parameter objeleri aynı storage'ı kullansa da logical normda ayrı
sayılır. `None` gradientler ayrı without_gradient sayısındadır; boş/zero gradient
normu gerçek0 olabilir. Frozen/unused optimizer parametreleri de bu sayıya dahildir.
Tensor dtype/device/layout/shape/elements, finite/has_nan/has_inf ve norm evidence
ayrıdır. NaN/Inf normu JSON numeric NaN/Inf değildir: value=null, status=nonfinite.
Finite norm FP64 aralığını aşarsa null/unrepresentable ve result partial olur.

## Coverage

Doğrulanmış adapter ordinary real FP16/BF16/FP32/FP64 dense packed strided CPU/CUDA
gradientleridir. Transpose/channels-last packed strides contiguous bir view sırasına
alınabilir; full copy gerekmez. Sparse COO (coalesced/uncoalesced), CSR/diğer sparse,
gapped/overlapping/expanded storage, complex/meta ve custom Tensor adapter'ları
unsupported'dır. P25 sparse gradienti coalesce/dense yapmaz. Trainer'ın mevcut
finite-check sparse coalesce davranışı değişmez.

Bir unsupported gradient tüm global norm coverage'ını incomplete yapar: global
value=null/status=unsupported, result partial. Known per-tensor evidence korunur;
bilinmeyen tensorlar sıfır sayılmaz. Sparse duplicate index normu hakkında iddia yoktur.
Bu adapter kapsamı dışında kalan gerçek training operasyonu native Trainer veya
optimizer tarafından ayrıca reddedilebilir; P25 bunları bypass etmez.

## AMP ownership ve recovery

`scale_before/scale_after/scale_change` gerçek native `get_scale()` readout'larıdır.
Scale azalması execution kararı değildir. `optimizer_executed` P20 native pre/post
hooks; `applied/skipped/applied_update_id` Trainer counter ledger ile doğrulanır.
P19 segment/epoch/window/attempt ve microbatch ID'leri JSON'a bağlanır; embedded P19
ve P20 aynı tek fit'in hash-associated source evidence'ıdır. Serializer'ları değişmez.

Native skip ile nonfinite gradient birlikte gözlenirse
`overflow_evidence="native_skip_with_nonfinite_gradient"`; root cause yine unknown'dur.
Ölçülmeyen found_inf internalleri veya yanlış scale-drop attribution kullanılmaz.
`summary.recovered_after_skip_attempt_ids`, **aynı segment** içinde skip sonrası ilk
applied attempt'leri listeler. Resume prefix'inden recovery/timing uydurulmaz.

CPU/non-AMP nonfinite gradient, Trainer'ın mevcut FloatingPointError davranışını
korur. Başarısız fit başarılı partial result döndürmez; model/state ilerlemiş olabilir.
Completed checkpoint `observations=[]`, `status="no_new_attempts"` üretir. EarlyStopping
best-weight restore mevcut hedefi ve sırasıyla kalır; diagnostics checkpoint'e girmez.

## Overhead ve sınırlar

P25 opt-in'dir; default Trainer/P19/P20/P21/P22/P23 yoluna yeni traversal, timer,
profiler, sync veya RNG tüketimi eklenmez. P25 aktifken FP64 reductions ve scalar
`.item()` readout'ları CUDA host synchronization yapar. `scalar_reads`, query ns ve
chunk counts evidence'dır; scaler-before read ve toplam observer maliyeti ayrı
izole edilmediği için total overhead null/unknown'dur. Per-stage P20 timer'ları query
aralıklarını içermez; source P19 attempt/window timing ve allocator kullanımı
diagnostics maliyetinden etkilenir. Bu zamanlar bare training performansı değildir.

Selected current stream korunur; hidden streams/concurrent mutation attribution ve
CUPTI kernel profiling desteklenmez. Native optimizer hooks yalnız gerçek step
attempt süresince kurulur, finally'da kaldırılır; optimizer.step monkeypatch edilmez.
GC, allocator peak reset veya empty_cache çağrısı yoktur. Exception sonrası retry
yoktur. Observation record budget dolarsa explicit failure olur; silent truncation yoktur.

## Export / Experiment Manager

`result.to_dict()` defensive copy; `to_json(path)` canonical tam evidence ve
`to_csv(path)` attempt tablosu üretir. Schema/source/evidence checksum doğrulaması,
atomic/no-clobber export ve explicit `overwrite=True` desteklenir. Checksum bir
authentication imzası değildir. CSV ortak formula-injection guard'ını kullanır.

Tracking yalnız ölçüm **sonrasında** yapılır. JSON/CSV copied artifact'larını SHA
verify edin, yalnız non-null normları generic metric olarak loglayın. Mevcut Manager
attempt axis kullanımı `step=attempt_id, step_kind="optimizer_attempt"` şeklindedir.
Unsupported/null norm sıfır değildir; loglanmamalıdır. Finish/close/reopen/read mevcut
store ile yapılır; yeni tablo/migration/run kind yoktur. Tracking failure training'i
tekrar çalıştırma gerekçesi değildir.

P25'in dense/native doğrulaması P21 worker trace veya P23 activation/isolated peak
eksiklerini tamamlamaz. B10 full GO ve stable release onayı çıkarılmaz; B4 riski açıktır.
