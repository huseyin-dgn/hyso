# Batch, resolution ve precision sweeps

`run_sweep` ordinary dense **CPU FP32 master model** alır; her noktada bağımsız
copy/device/dtype hazırlığı yapar. Caller model taşınmaz. Noktalar explicit finite
listededir; otomatik arama, interpolation veya comparison compatibility yoktur.
Her noktada aynı başlangıç RNG ve bağımsız warmup vardır. Factory exact contiguous
NCHW/device/dtype döndürmelidir; silent resize/cast yapılmaz.

```python
from pathlib import Path
import tempfile
import torch
from hyso.performance import SweepProtocol, SweepResult, PrecisionResult, run_sweep

torch.set_num_threads(1)
output_dir = Path(tempfile.mkdtemp(prefix='hyso-sweeps-'))
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(35, 8),
                           torch.nn.ReLU(), torch.nn.Linear(8, 2))
def inputs(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)

plan = SweepProtocol(warmup_iterations=1, measure_iterations=3, repeats=1,
                     sustained_batches=4)
batches = run_sweep(model, inputs, axis='batch', points=[1, 2, 4, 8], protocol=plan)
spatial_model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1),
                                   torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
resolutions = run_sweep(spatial_model, inputs, axis='resolution',
                       points=[(5, 7), (8, 8), (9, 13)], protocol=plan)
precisions = run_sweep(model, inputs, axis='precision',
                      points=['fp32', 'fp64', 'bf16', 'int8_dynamic'], protocol=plan)
assert batches.to_dict()['actual']['measured'] == 4
assert resolutions.to_dict()['actual']['measured'] == 3
for name, result in [('batch', batches), ('resolution', resolutions), ('precision', precisions)]:
    result.to_json(output_dir / (name + '.json'))
    result.to_csv(output_dir / (name + '.csv'))
    assert isinstance(SweepResult(result.to_dict()), SweepResult)
    for point in result.to_dict()['raw_evidence']['points']:
        if point['status'] == 'measured':
            measurement = PrecisionResult(point['result'])
            measurement.to_json(output_dir / (point['point_id'] + '.json'))
            measurement.to_csv(output_dir / (point['point_id'] + '.csv'))
```

Latency samples ms/batch, actual cardinality ve sustained completion images/s ayrı
scope'lardır. FP32/FP64/CUDA AMP FP16 mevcut Benchmark/P02 protokollerini kullanır.
BF16 FP32 master/input ve autocast kullanır; output BF16 değilse unsupported olur.
CUDA capability≥8 ve runtime availability gerekir. CPU actual BF16 output execution
kanıtıdır; native BF16 instruction/acceleration garantisi verilmez, backend emüle
edebilir. Host/device event saatleri ayrıdır; selected stream completion beklenir.

INT8 yalnız CPU `Sequential(Flatten/Linear/ReLU/Identity)` dynamic qint8 Linear
recipe'sidir. Active engine x86/fbgemm/qnnpack/onednn olmalıdır. Packed qint8
weights ve preflight profiler'da gerçek `quantized::linear_dynamic` execution
kanıtı zorunludur; profiling timed window dışındadır. Diğer allowed ops floating
point kalır. Dynamic activation quantization calibration dataset gerektirmez;
calibration not_applicable/reason olarak kaydedilir. Unsupported graph/device
explicit sonuçlanır. Global quantization engine değiştirilmez. Conversion/copy
ms ayrı scope; artifact/load cost not_measured, P28 accuracy not_measured'dır.

```python
def cuda_example():
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA required')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        return run_sweep(model, inputs, axis='precision',
            points=['fp32', 'amp_fp16', 'bf16', 'int8_dynamic'],
            protocol=SweepProtocol(device='cuda:0', measure_iterations=5))
```

Source/input hashes, precision/master/input/output dtype, backend/recipe,
runtime/device, point UUID/observation index, independent result UUID ve actual
measurement refs raw evidence içindedir. Her başarı embedded validated
PrecisionResult taşır; eski BenchmarkResult schema1 aynen saklanır.

Ordinary ValueError/RuntimeError continuation `continue_on_error` flag'ini izler.
Unsupported aynı flag'i izler. OOM her zaman admission'ı durdurur; kalanlar
not_run/null olur. OOM recovery, cache flush veya maximum feasible batch araması
yoktur. State/stream mutation veya CUDA drain failure fatal exception'dır. Caller
registered state/gradient/mode ve Python/NumPy/Torch/selected CUDA RNG restore edilir;
external Python side effects caller sorumluluğudur. Concurrent callers unsupported.
Output/event lifetime bounded; global GC/allocator reset eklenmez.

`measure_cuda_memory=True` legacy precision noktalarında existing P09 separate
probe'u explicit opt-in eder ve peak reset geri alınamaz; BF16/INT8 probe unsupported
ve null'dır. RAM/model-only/physical footprint gibi başka semantiklere çevrilmez.

## Ölçüm sonrası Experiment Manager

```python
from hyso.experiments import ExperimentManager
root = output_dir / 'runs'
manager = ExperimentManager(root)
experiment = manager.create_experiment('sweeps')
manager.close()
ExperimentManager.migrate_store(root)
manager = ExperimentManager(root)
run = manager.create_run(experiment['id'], 'benchmark', config={'axis': 'batch'},
                        provenance={'sweep_result_id': batches.to_dict()['result_id']})
run.start()
artifact_ids = []
for filename in ['batch.json', 'batch.csv']:
    record = run.add_artifact(output_dir / filename, kind='performance_result', ownership='copy')
    artifact_ids.append(record['id'])
for point in batches.to_dict()['raw_evidence']['points']:
    if point['status'] != 'measured':
        continue
    record = run.add_artifact(output_dir / (point['point_id'] + '.json'),
                             kind='performance_result', ownership='copy')
    artifact_ids.append(record['id'])
    value = point['result']['measurements']['latency_mean_ms']['value']
    if value is not None:
        run.log_metric(name='perf.point.latency', value=value, unit='ms/batch',
            step=point['observation_index'], step_kind='observation', source='performance.sweep_v1')
run.finish()
run_id = run.id
manager.close()
manager = ExperimentManager(root)
assert manager.get_run(run_id).to_dict()['status'] == 'completed'
assert len(manager.get_run(run_id).read_metrics()) == 4
assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in artifact_ids)
manager.close()
```

Manifest point→observation/result refs eşlemesini taşır. Unsupported/null numeric
zero metric yapılmaz. Measurement windows içine SQLite/disk/network tracking
eklenmez; existing schema2/generic metrics/artifacts kullanılır. Backend/artifact
alanları future adapters için ayrıdır; ONNX/LibTorch/TensorRT/OpenVINO/TFLite/
Core ML/YOLO conversion burada uygulanmaz.
