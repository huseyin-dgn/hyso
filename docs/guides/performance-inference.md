# Resident inference: P99 latency ve sustained throughput

`hyso.performance`, mevcut `hyso.benchmark` schema1'ini değiştirmez. Latency resident
batch forward süresidir; mevcut derived serial service rate sustained throughput
değildir. Yeni throughput, host completion penceresinde gerçekten tamamlanan image
sayısının wall seconds'a oranıdır; kamera/display/E2E FPS değildir.

## CPU ve P99

Bu örnekler sırayla aynı Python oturumunda çalışır. Dosyalar yeni bir geçici dizine
yazılır. Yerel model/input kullanılır, hiçbir şey indirilmez.

```python
from pathlib import Path
import tempfile
import torch
from hyso.benchmark import BenchmarkRunner
from hyso.performance import analyze_latency, measure_throughput, ThroughputProtocol

torch.set_num_threads(1)
model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1),
                            torch.nn.AdaptiveAvgPool2d(1)).eval()
inputs = torch.ones(2, 1, 5, 7)
benchmark = BenchmarkRunner(model, example_input=inputs, warmup_iterations=2,
                            measure_iterations=500, repeats=2).run()
latency = analyze_latency(benchmark, quantiles=(.50, .95, .99))
assert latency.to_dict()['actual']['samples'] == 1000
assert latency.to_dict()['measurements'][-1]['status'] == 'derived'

throughput = measure_throughput(model, inputs, protocol=ThroughputProtocol(
    device='cpu', precision='fp32', warmup_iterations=2, batch_count=20,
    repeats=2, queue_depth=8))
assert throughput.to_dict()['actual']['completed_images'] == 80
output_dir = Path(tempfile.mkdtemp(prefix='hyso-performance-'))
latency.to_json(output_dir / 'latency.json')
latency.to_csv(output_dir / 'latency.csv')
throughput.to_json(output_dir / 'throughput.json')
throughput.to_csv(output_dir / 'throughput.csv')
```

P99 linear interpolation `(n-1)*q` kullanır. Varsayılan TailPolicy: n≥1000 ve
(1−q)n≥10; n=999 için P99 null/insufficient_samples olur. `TailPolicy` iki pozitif
integer eşiği explicit değiştirebilir; bu yalnız descriptive exploration içindir,
confidence interval veya bağımsız örnek kanıtı değildir. R repeat aynı koşunun
correlated örneklerini bağımsız trial yapmaz. Batch latency, ms/image diye bölünmez.
Analysis source result UUID, SHA256 ve raw R×N sample'ları korur; modeli çalıştırmaz.

Süre penceresi için `ThroughputProtocol(batch_count=None, duration_seconds=1.0)`
kullanın; batch_count ile duration birlikte verilemez. Süre admission sınırıdır:
admitted işler tamamlanır, terminal tail toplam süreye dahildir. Çok kısa pencerede
hiç iş tamamlanmazsa successful result yerine hata alınır. Queue depth≥2 ve
concurrency=1 desteklidir; CPU synchronous loop'un actual queue depth'i 1'dir.

## Native CUDA

CUDA ayrı explicit çağrıdır; cihaz yoksa hata verir, CPU fallback yapmaz. Input ve
model caller tarafından aynı CUDA device üzerinde önceden hazırlanır. FP32 master
weights/input korunur; `amp_fp16` autocast kullanır, her op FP16 olmak zorunda değildir.
CPU fp32/fp64, CUDA fp32/amp_fp16 desteklenir; diğer backend/precision desteklenmez.

```python
def cuda_example():
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA is required; no CPU fallback')
    gpu_model = torch.nn.Conv2d(1, 2, 3, padding=1).to('cuda:0').eval()
    gpu_input = torch.ones(2, 1, 5, 7, device='cuda:0')
    stream = torch.cuda.Stream(device=0)
    # Producer setup is complete before another stream consumes these tensors.
    torch.cuda.synchronize(0)
    with torch.cuda.stream(stream):
        return measure_throughput(gpu_model, gpu_input, protocol=ThroughputProtocol(
            device='cuda:0', precision='amp_fp16', warmup_iterations=2,
            batch_count=33, queue_depth=8, repeats=2))
```

CUDA events/output ring bounded'dır. Tam queue grubunda completion gerekiyorsa
bir backpressure event wait yapılır; her forward'a synchronize eklenmez. Terminal
event admission sonunda kaydedilir, tail tamamlanmadan completion sayılmaz.
Per-window completed ID aralığı/counter/groups, timestamps, pending ve max_pending
JSON evidence içindedir. Her batch aynı cloned resident input'u kullanır; input
iterator, camera veya dataset pipeline bu API'nin kapsamı değildir.

## Mevcut Experiment Manager ile manuel kayıt

Ölçüm tamamlandıktan sonra export, hash/artifact ve bounded metric işlemleri yapılır.
Yeni store schema1 kalır; artifact için migration açıkça ve kapalı store üzerinde
yapılır. Yeni SQLite schema/run kind/tracking adapter yoktur.

```python
from hyso.experiments import ExperimentManager

root = output_dir / 'experiment-store'
manager = ExperimentManager(root)
experiment = manager.create_experiment('P01-P02 resident inference')
manager.close()
ExperimentManager.migrate_store(root)
manager = ExperimentManager(root)
run = manager.create_run(experiment['id'], 'benchmark',
    config={'protocol': throughput.to_dict()['protocol']},
    provenance={'throughput_result_id': throughput.to_dict()['result_id'],
                'source_benchmark_id': benchmark.to_dict()['identity']['run_id'],
                'model_input_runtime': throughput.to_dict()['provenance']})
run.start()
artifact = run.add_artifact(output_dir / 'throughput.json', kind='performance_result',
                            ownership='copy', role='output')
run.add_artifact(output_dir / 'latency.json', kind='performance_result', ownership='copy')
run.log_metric(name='perf.throughput.images', value=throughput.to_dict()['measurements']['value'],
               unit='images/s', source='performance.throughput_v1', step=0, step_kind='observation')
p99 = latency.to_dict()['measurements'][-1]
if p99['value'] is not None:
    run.log_metric(name='perf.latency.p99', value=p99['value'], unit='ms',
                   source='performance.latency_v1', step=0, step_kind='observation')
assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
run.finish()
run_id, artifact_id = run.id, artifact['id']
manager.close()
manager = ExperimentManager(root)
assert manager.get_run(run_id).to_dict()['status'] == 'completed'
assert manager.verify_artifact(artifact_id)['status'] == 'verified'
manager.close()
```

JSON tam canonical kayıttır; CSV latency için quantile başına, throughput için
repeat window başına satır verir. Null/empty değer status/reason ile açıklanır.
Export immutable sonucu değiştirmez; overwrite varsayılan false, publication
atomic'tir; var olan hedefe yazmak için explicit overwrite=True gerekir.
Run ID, benchmark ID ve yeni result UUID farklı kimliklerdir. Raw samples artifact
içinde kalır, metric_events'e örnek başına çoğaltılmaz. Tracking-only işlemler
forward/sync/allocator reset yapmaz; başarısız tracking başarılı ölçümü tekrarlatmaz.

## Sınırlar

Measurement scope `resident_forward_completion_wall`: forward, device finite
reductions, host loop/queue bookkeeping, bounded backpressure ve terminal tail
dahildir. Clone/transfer, descriptor/hash, preflight/warmup, state inspection,
scalar validity readback ve dosya/DB işlemleri dışarıdadır. Sonuç bu overhead'i
içerir, bare latency ile aynı scope değildir. Her repeat ayrı warmup ve drained
window'dur; aggregate rate toplam completed images / toplam window seconds'tır.
Output shape/dtype/device sabit ve finite olmalıdır.

Caller registered state/gradient/mode ve Python/NumPy/Torch CPU/selected CUDA RNG
guard edilir; unsupported mutation successful result üretmez. Structural rollback,
arbitrary Python state/external I/O ve poisoned CUDA context için transaction
garantisi yoktur. Hidden cross-stream/device çalışma veya aynı model/device üzerinde
concurrent caller desteklenmez. Aynı process/device allocator peak reset edilmez;
P09 davranışı değişmez. Clock/thermal/power kontrol edilmez, driver/source revision
otomatik toplanmıyorsa explicit null/status metadata bulunur. Windows RTX3060
kanıtı farklı OS/device kombinasyonunun native validation'ı değildir.
