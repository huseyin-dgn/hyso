# FLOPs/MACs, layers ve operators — P12/P13/P14

Profiling, BenchmarkRunner latency ölçümünden ayrı bir gözlem penceresidir.
API yalnız trusted importable **ColdStartRecipe** alır; canlı caller modelini,
Tensor'ını, hook/stream/context handle'ını worker'a taşımaz. Windows fresh worker
Torch importundan önce mevcut Job Object private-commit sınırını uygular. CPU
FP32/FP64, CUDA FP32/AMP FP16 eager inference profilleri desteklenir. Factory CPU
model kurar; worker kendi modelini seçilen dtype/device'a taşır ve eval/inference
mode'da yürütür. Trained weights için SHA-verified weights_only state_dict recipe
verilebilir. Caller modelinin mevcut mode/state/hook'ları böylece değiştirilmez.
Factory external side effect, child process veya hidden stream/device işi yapmamalı.
Linux resource guard bu ilk scope'ta unsupported'dır.

Bir capture'dan üç farklı frozen/schema1 sonuç türetin:

```python
from pathlib import Path
import tempfile
from hyso.performance import (ColdStartRecipe, ProfilingProtocol,
    profile_operators, profile_layers, analyze_flops)

output_dir = Path(tempfile.mkdtemp(prefix='hyso-profiling-guide-'))
recipe = ColdStartRecipe('hyso.models.modular:ModularCNN',
    model_kwargs={'input_channels': 1, 'num_classes': 2, 'widths': [3],
                  'repeats': [1], 'strides': [1], 'attention': 'eca'},
    input_shape=(2, 1, 5, 7), device='cpu', precision='fp32')
protocol = ProfilingProtocol(warmup_iterations=1, measure_iterations=2,
    timeout_seconds=45, cpu_memory_limit_bytes=3 * 1024**3,
    input_limit_bytes=32 * 1024**2, max_events=20000)
operators = profile_operators(recipe, protocol=protocol,
    trace_path=output_dir / 'model-trace.json')
layers = profile_layers(operators)
counts = analyze_flops(operators)
operator_data = operators.to_dict()
layer_data = layers.to_dict()
count_data = counts.to_dict()
assert operator_data['actual']['measured_forwards'] == 2
assert count_data['measurements']['status'] == 'partial'
assert count_data['measurements']['total_flops_per_batch'] is None
top_layers = [row for row in layer_data['measurements']['modules']
              if row['cpu_exclusive_us'] is not None]
top_layers.sort(key=lambda row: row['cpu_exclusive_us'], reverse=True)
print('Instrumented CPU layers:', [(row['path'] or '<root>', row['cpu_exclusive_us'])
                                   for row in top_layers[:3]])
print('Instrumented CPU operators:', [(row['name'], row['self_cpu_us'])
                                     for row in operator_data['measurements']['operators'][:3]])
```

`profile_layers(recipe, protocol=...)` ve `analyze_flops(recipe, protocol=...)`
kısayolları yeni capture açar; yukarıdaki result girdileri tekrar forward çalıştırmaz.
P14 OperatorProfileResult, P13 LayerProfileResult ve P12 OperationCountResult
bağımsız format/UUID taşır; derived sonuçlar source UUID/SHA ve trace checksum
ilişkisini korur. Input recipe exact contiguous NCHW ones'dır. Dynamic control
flow kanıtı yalnız bu input/seed ve gerçekleşen yollar içindir.

Analitik doğruluğu bilinen grouped convolution örneği:

```python
tiny_recipe = ColdStartRecipe('torch.nn:Conv2d',
    model_kwargs={'in_channels': 2, 'out_channels': 4, 'kernel_size': 3,
                  'padding': 1, 'groups': 2, 'bias': True},
    input_shape=(2, 2, 5, 7), precision='fp64')
tiny_counts = analyze_flops(tiny_recipe, protocol=protocol)
tiny = tiny_counts.to_dict()['measurements']
assert tiny['total_macs_per_batch'] == 2520
assert tiny['total_flops_per_batch'] == 5320
assert tiny['total_flops_per_sample'] == 2660
assert tiny['coverage']['unknown_calls'] == 0
assert tiny_counts.to_dict()['provenance']['input']['dtype'] == 'torch.float64'
```

Politika: logical dense Conv MAC = output.numel × weight.Cin_per_group × kernel
elements; Conv1d/2d, grouped/depthwise, Linear/mm/matmul/bmm ve default addmm
desteklenir. **1 MAC = 2 FLOPs**; bias output başına ayrı1 addition'dır. Add/sub/
mul/div output broadcast cardinality ile sayılır. ReLU karşılaştırmaları ayrı
`comparisons` alanıdır, FLOPs değildir. Memory/layout/copy/cast aritmetik maliyeti
explicit0'dır; trafik/latency/enerji maliyetinin0 olduğu anlamına gelmez.
Adaptive average pool gerçek integer bin sınırlarıyla; fixed average/max pool
yalnız nonpadded/non-ceil scope'ta modellenir. FMA sayımı fiziksel GPU instruction
veya counter ölçümü değildir; compiler optimizasyonu/fusion maliyeti değildir.

SiLU/sigmoid/softmax, normalization, fused attention, transposed convolution,
genel einsum/custom namespace ve henüz modellenmemiş işlemlerin maliyeti **null**
ve explicit reason'dır. ECA/CoordinateAttention içindeki desteklenen Conv/mul
işlemleri sayılabilir; bu bütün attention graph'ının total FLOPs'u değildir.
ModularCNN örneğinde subtotal raporlanır, model total null kalır. Coverage fraction
**gözlenen ATen çağrı sayısı oranıdır**, FLOPs yüzdesi veya hardware coverage değildir.
`per_forward` satırları her ölçülen çağrının batch toplamını ayrı taşır;
`total_*_per_batch` bunların mean'idir, per-sample batch cardinality'ye bölünür.

Katman kayıtları canonical module identity/path, tüm kayıtlı alias paths ve
invocation-parent ağacını taşır. Shared module tekrarları kaybolmaz. CPU inclusive
süre doğrudan record_function aralığıdır; module exclusive = inclusive − doğrudan
child-module inclusive toplamı. Profiler'ın self-CPU operator zamanıyla aynı
tanım değildir. Inclusive parent/child sürelerini toplayıp latency üretmeyin.
Sıralama instrumented CPU exclusive veya operator self CPU alanına göredir.
Çağrılmayan module maliyeti null/not_executed'dır. Existing worker-local module
hook'ları korunur; yalnız bu capture'ın eklediği handle'lar success/exception'da
kaldırılır. BenchmarkRunner'ın hook rejection kuralı değiştirilmez/bypass edilmez.

CUDA örneğini gerçek GPU interpreter'ında ayrı çalıştırın:

```python-cuda
from hyso.performance import ColdStartRecipe, ProfilingProtocol, profile_operators, analyze_flops
recipe = ColdStartRecipe('torch.nn:Conv2d',
    model_kwargs={'in_channels': 1, 'out_channels': 2, 'kernel_size': 1, 'bias': False},
    input_shape=(1, 1, 7, 9), device='cuda:0', precision='fp32')
gpu_operators = profile_operators(recipe, protocol=ProfilingProtocol(
    warmup_iterations=1, measure_iterations=2, timeout_seconds=45, cuda_memory_fraction=.1))
gpu_data = gpu_operators.to_dict()
assert gpu_data['raw_evidence']['output']['device'] == 'cuda:0'
assert analyze_flops(gpu_operators).to_dict()['measurements']['total_flops_per_batch'] == 252
print(gpu_data['measurements']['cuda_activity'])
```

Yerel Windows/Torch2.5.1+cu121 build'i GPU forward çalıştırmasına rağmen yalnız
CPU tracing activity bildirir. Gerçek Chrome trace `cat=kernel/ph=X` activity
olmadan kernel name/time uydurulmaz: `cuda_activity.status=unsupported`, duration
ve kernel_count null'dır. Legacy FunctionEvent.kernels CPU operator etiketleri
taşıyabildiğinden kullanılmaz. Bununla ilgili fallback sınırları
[PyTorch profiler belgesinde](https://docs.pytorch.org/docs/stable/profiler.html)
açıklanır. Gerçek activity veren bir build'de parser yalnız gözlenen trace
kernel adlarını/duration_us değerlerini taşır; bu yol burada native olarak
doğrulanmış sayılmaz. Raw correlation/External id varsa saklanır, fakat güvenilir
operator→kernel veya kernel→layer attribution iddiası yoktur. CUDA layer süreleri
null/unsupported kalır. Kernel-duration toplamı da overlap nedeniyle latency değildir.

Worker yalnız warmup sonrası ve capture sonunda seçilen stream'i drain eder;
per-layer synchronization eklenmez. Shape capture, module hooks ve versiyonları
test edilen internal TorchDispatchMode kayıt mekanizması allocation/scheduling'i
değiştirir. `overhead` collector lifecycle wall ms ve instrumented host-forward
windows taşır. Bu windows GPU kernel süresi değildir; gerçek uninstrumented
perturbation delta **not_measured/null**'dır. Saf latency için ayrı BenchmarkRunner
kullanın. Export/trace flush/SQLite işleri forward penceresine eklenmez.

Her capture bounded worker timeout, private-commit/input/event/trace bütçesine
sahiptir. `.1` CUDA fraction yalnız child allocator bütçesidir, physical VRAM
değildir. Worker reset_peak_memory_stats/empty_cache/global GC kullanmaz; kendi
context'i exit ile bırakılır. Parent device/stream/RNG/allocator/hook'ları
değişmez. ProfilingError diagnostics process exit, worker phase ve cleanup
bilgisini taşır. Native unexpected exit OOM diye sınıflandırılmaz; yeni capture
ile kör retry yapılmaz. B4 risk OPEN/UNRESOLVED ve ayrı stable release gate sürer.

Ölçüm bittikten sonra mevcut schema2/generic metrics ve copied artifacts:

```python
for name, result in [('operators', operators), ('layers', layers), ('counts', counts)]:
    result.to_json(output_dir / (name + '.json'))
    result.to_csv(output_dir / (name + '.csv'))
from hyso.experiments import ExperimentManager
root = output_dir / 'runs'
manager = ExperimentManager(root)
experiment = manager.create_experiment('profiling')
manager.close()
ExperimentManager.migrate_store(root)
manager = ExperimentManager(root)
run = manager.create_run(experiment['id'], 'benchmark', config={'features': ['P12', 'P13', 'P14']},
    provenance={'operator_result_id': operator_data['result_id']})
run.start()
artifact_ids = []
for filename in ['operators.json', 'layers.json', 'counts.json', 'model-trace.json']:
    record = run.add_artifact(output_dir / filename, kind='performance_result', ownership='copy')
    artifact_ids.append(record['id'])
metric_count = 0
for index, row in enumerate(operator_data['measurements']['operators']):
    if row['self_cpu_us'] is not None:
        run.log_metric(name='profile.cpu_self', value=row['self_cpu_us'], unit='us',
            step=index, step_kind='observation', source='performance.operator_profile_v1')
        metric_count += 1
run.log_metric(name='profile.collector_wall',
    value=operator_data['measurements']['overhead']['collector_lifecycle_wall_ms'], unit='ms',
    step=0, step_kind='observation', source='performance.operator_profile_v1')
metric_count += 1
run.finish()
run_id = run.id
manager.close()
manager = ExperimentManager(root)
assert manager.get_run(run_id).to_dict()['status'] == 'completed'
assert len(manager.get_run(run_id).read_metrics()) == metric_count
assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in artifact_ids)
manager.close()
```

Sonuçlar plain/finite/frozen schema1; JSON/CSV ve trace yayını atomic/no-clobber'dır.
Trace byte/SHA özgün haliyle korunur. Benchmark schema1, checkpointv1 ve Experiment
Manager schemas değişmez. Yeni table, migration veya run kind eklenmez.
