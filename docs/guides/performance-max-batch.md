# Maximum feasible batch — P17

`find_max_batch_size(ColdStartRecipe(...), protocol=BatchSearchProtocol(...))`
çalıştırılan adaylar için inference kapasitesi ölçer. İlk kapsam Windows,
ordinary dense eager CPU FP32/FP64 ve CUDA FP32/AMP FP16'dır. Her aday fresh
interpreter/context kullanır. Canlı model/Tensor/stream aktarılmaz; caller model,
registered state, gradients, modes ve RNG'ye erişilmez. Factory trusted,
importable `module:qualified_name` olmalı, CPU FP32 master üretmeli ve external
side effect/subprocess başlatmamalıdır. Recipe kwargs plain JSON'dır; `seed`
her worker'da aynıdır. İstenirse mevcut ColdStartRecipe artifact_path/SHA256 ile
weights_only CPU load kullanılır. Input recipe exact contiguous NCHW `ones`'dır;
gerçek veriye ilişkin kapasite veya kalite garantisi değildir.

CPU örneği (Torch importundan önce Job Object private-commit sınırı; Windows):

```python
from pathlib import Path
import tempfile
from hyso.performance import ColdStartRecipe, BatchSearchProtocol, find_max_batch_size

output_dir = Path(tempfile.mkdtemp(prefix='hyso-capacity-guide-'))
recipe = ColdStartRecipe('torch.nn:Conv2d',
    model_kwargs={'in_channels': 1, 'out_channels': 2, 'kernel_size': 1},
    input_shape=(1, 1, 5, 7), device='cpu', precision='fp32', seed=3)
protocol = BatchSearchProtocol(candidates=(1, 2, 3), maximum_batch=3,
    cpu_memory_limit_bytes=3 * 1024**3, input_limit_bytes=8 * 1024**2,
    timeout_seconds=45, warmup_iterations=1, measure_iterations=3, repeats=1,
    stop_policy='exhaust_candidates')
capacity = find_max_batch_size(recipe, protocol=protocol)
data = capacity.to_dict()
assert data['measurements']['status'] == 'complete', data
assert data['measurements']['maximum_proven_within_range'] == 3
capacity.to_json(output_dir / 'capacity.json')
capacity.to_csv(output_dir / 'capacity.csv')
```

CUDA örneğini CPU örneğinden bağımsız, gerçek CUDA interpreter'ında çalıştırın:

```python-cuda
from hyso.performance import ColdStartRecipe, BatchSearchProtocol, find_max_batch_size
recipe = ColdStartRecipe('torch.nn:Conv2d',
    model_kwargs={'in_channels': 1, 'out_channels': 2, 'kernel_size': 1},
    input_shape=(1, 1, 7, 9), device='cuda:0', precision='amp_fp16')
protocol = BatchSearchProtocol(candidates=(1, 2, 4), maximum_batch=4,
    cpu_memory_limit_bytes=3 * 1024**3, input_limit_bytes=8 * 1024**2,
    timeout_seconds=45, warmup_iterations=1, measure_iterations=3,
    cuda_memory_fraction=.1, stop_policy='first_infeasible')
capacity = find_max_batch_size(recipe, protocol=protocol)
assert capacity.to_dict()['measurements']['maximum_observed_feasible_batch'] == 4
assert capacity.to_dict()['measurements']['maximum_proven_within_range'] is None
```

Explicit candidate list (en fazla512) verilen sırayla yürür; `maximum_batch`
aşılamaz. `exhaust_candidates`, yalnız verified feasible ve typed OOM sonrasında
yeni worker açar. `first_infeasible`, ilk OOM'da durur. Exception, unsupported,
resource rejection, timeout veya unexpected exit her iki politikada admission'ı
durdurur; kalanlar `not_run/unknown` olur. Adaptive/binary/exponential otomatik
aday üretimi bu sürümde uygulanmaz; powers-of-two listesi kullanıcı verebilir.
Monotonluk varsayılmaz. Timeout process startup/import dâhil toplam wall süresidir.

`maximum_observed_feasible_batch` yalnız tamamlanmış forward'lardan türetilir.
`maximum_proven_within_range`, ancak 1..maximum_batch bütün integer noktaları
feasible/infeasible_oom olarak gözlendiğinde değer alır; sonuç yalnız bu recipe,
seed, çalışma anı ve bütçe içindir. Sparse liste veya erken durma aradaki noktaları
kanıtlamaz. Cap'e ulaşmak fiziksel/global maksimum kanıtı değildir. Aborted arama
başarılı maximum yayımlamaz; önceki kanıt `prior_observed_feasible_batch` alanında
korunur. Başarısız listede feasible değer yoksa maximum null'dır.

Her satırda candidate UUID, requested/actual batch, shape, backend/precision/device,
worker/launcher PID, exit code, process termination, execution wall seconds,
latency, model/input hashes ve runtime provenance bulunur. `sweep` alanı B4
`_point` çekirdeğinden üretilen embedded **PrecisionResult**'tır. B4 BenchmarkRunner
warmup/repeats/sample ve P09 separate single-forward memory probe'u aynen kullanılır;
yeni timing framework yoktur. Memory probe warmup/sample sayısından ayrıdır.
Failed forward completion/actual batch/latency alanları null veya false kalır;
hazırlık aşamasına bağlı memory/provenance da null olabilir.

Windows Job Object tüm worker private commit'ini sınırlarken active-process limit1
factory child process açılmasını engeller. Guard kurulamazsa çalışmaya devam edilmez.
`input_limit_bytes` bir dense input'un allocation öncesi sınırıdır; bütün model
working set'i değildir. CPU private-commit sınırı resident RSS veya fiziksel RAM
değildir. Mevcut P10 RSS/HWM sorguları ölçümün öncesinde/sonrasında alınır; exact
inference-window peak olarak sunulmaz. Linux/diğer OS resource guard unsupported.

`cuda_memory_fraction` yalnız child PyTorch allocator bütçesidir; context/driver,
diğer process'ler ve fiziksel VRAM sınırı değildir. P09 allocator peak ayrı probe
sonucudur, RSS ile toplanmaz. OOM worker'ı yeniden kullanılmaz; terminal process
context'i bırakır. Timeout yalnız bu aramanın hâlâ çalışan launcher ağacını sonlandırır.
Global reset/cache flush/GC/retry yoktur. Native crash OOM sayılmaz; exact nonzero
exit code korunur ve arama aborted olur. B4 risk kaydı **OPEN / UNRESOLVED** kalır.
Uzun vadeli kararlılık, training capacity ve stable release onayı bu API'den çıkmaz.

Ölçüm bittikten sonra mevcut schema2/generic metrics ve artifacts ile ilişkilendirme:

```python
from hyso.experiments import ExperimentManager
root = output_dir / 'runs'
manager = ExperimentManager(root)
experiment = manager.create_experiment('capacity')
manager.close()
ExperimentManager.migrate_store(root)
manager = ExperimentManager(root)
run = manager.create_run(experiment['id'], 'benchmark', config={'feature': 'P17'},
    provenance={'capacity_result_id': data['result_id']})
run.start()
artifact_ids = []
for filename in ['capacity.json', 'capacity.csv']:
    artifact = run.add_artifact(output_dir / filename, kind='performance_result', ownership='copy')
    artifact_ids.append(artifact['id'])
for index, row in enumerate(data['raw_evidence']['candidates']):
    if row['status'] != 'feasible':
        continue
    for name, value, unit in [('latency', row['latency_mean_ms'], 'ms/batch'),
        ('rss_after', row['memory']['rss_after']['rss_bytes'], 'byte')]:
        run.log_metric(name='capacity.' + name, value=value, unit=unit, step=index,
            step_kind='observation', source='performance.capacity_v1')
run.finish()
run_id = run.id
manager.close()
manager = ExperimentManager(root)
assert manager.get_run(run_id).to_dict()['status'] == 'completed'
assert len(manager.get_run(run_id).read_metrics()) == 6
assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in artifact_ids)
manager.close()
```

Result schema1 independent/frozen/plain/finite'dır; JSON/CSV atomic ve default
no-clobber export'tur. Tracking/disk/SQLite/network ölçüm penceresine girmez.
Existing Benchmark schema1, checkpointv1 ve B1–B4 sonuç formatları değişmez.
