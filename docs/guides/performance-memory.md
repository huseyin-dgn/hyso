# Process RAM ve model footprint

`hyso.performance` P10/P11 sonuçları versioned `MemoryResult` ve `FootprintResult`
türleridir; BenchmarkResult schema1 ve inference P01/P02 ölçümleri değişmez.
RAM byte'ı, logical tensor byte'ı ve serialized file byte'ı farklı scope'lardır.

## Native current-process RSS

Aşağıdaki Python blokları sırayla aynı oturumda çalışır. Yerel CPU tensor/model
kullanılır; dosyalar yeni geçici dizinde üretilir, indirme veya dependency kurulmaz.

```python
from pathlib import Path
import tempfile
import time
import torch
from hyso.performance import MemoryProtocol, measure_memory, measure_footprint

torch.set_num_threads(1)
output_dir = Path(tempfile.mkdtemp(prefix='hyso-memory-'))
current = measure_memory()  # tek native OS observation; workload yok

def workload():
    tensor = torch.empty(2 * 1024 * 1024, dtype=torch.float32)
    tensor.fill_(1.25)  # native CPU storage'u gerçekten touch et
    time.sleep(.03)    # bu örnekte sampler'a gözlem fırsatı verir
    del tensor

memory = measure_memory(workload, protocol=MemoryProtocol(interval_seconds=.005, max_samples=1000))
memory_data = memory.to_dict()
assert memory_data['measurements']['current_rss_bytes']['status'] == 'measured'
assert memory_data['actual']['observations'] >= 2
assert memory_data['protocol']['lifetime_peak_reset'] is False
memory.to_json(output_dir / 'memory.json')
memory.to_csv(output_dir / 'memory.csv')
```

Windows `GetProcessMemoryInfo`: WorkingSetSize ve PeakWorkingSetSize, native byte.
Linux adapter `/proc/self/status`: VmRSS ve VmHWM, kB×1024 byte. Bu sürümün **fully
verified native platform kapsamı Windows**; Linux parser/protocol testleri native
Linux kanıtı değildir. Diğer OS/backend unavailable ise metric value=null,
status=unsupported, reason açıklanır ve workload çalıştırılmaz. Backend ölçüm
ortasında kaybolursa exception gelir, fake partial peak üretilmez.

Baseline/current RSS, timestamped observations, sampled peak, başlangıç/son
lifetime HWM, actual gaps ve query start/end timestamp'leri JSON'dadır. Sampled
peak mutlak peak değildir; kısa allocation'lar kaçabilir. OS HWM tüm process
ömrünü içerir ve bu API tarafından sıfırlanmaz. Thread/GIL/native query overhead'i
ölçüm scope'una dahildir; max_samples bütçesi aşılırsa hata verilir, sample'lar
sessizce atılmaz. Interval bir scheduling talebidir, actual gaps garanti değildir.

API yalnız current process'i gözler; model-only/process-isolated peak değildir.
Callable senkron tamamlanmalı ve kendi tensor/resource temizliğini yapmalıdır;
return value son observation'dan önce bırakılır. Thread success/failure'da join
edilir, Python/NumPy/Torch CPU RNG restore edilir. Aynı process'te concurrent RNG/
model kullanımı desteklenmez; workload'un model mutasyonları geri alınmaz. GPU
sync/reset/RNG çağrısı yoktur. Fresh process gerekiyorsa dış harness spawn/importable
factory ile yönetir; bu API hidden worker veya hard timeout başlatmaz. Ayrı worker
HWM'si bile worker'ın initialization ve sampler maliyetini içerir.

## Logical model ve gerçek artifact

```python
model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
logical_only = measure_footprint(model)
model_path = output_dir / 'model-state.pt'
footprint = measure_footprint(model, serialize_to=model_path, verify_roundtrip=True)
footprint_data = footprint.to_dict()
artifact_info = footprint_data['raw_evidence']['artifact']
assert artifact_info['size_bytes'] == model_path.stat().st_size
assert artifact_info['format_validation'] == 'roundtrip_verified'
assert footprint_data['measurements']['physical_memory_bytes']['value'] is None
footprint.to_json(output_dir / 'footprint.json')
footprint.to_csv(output_dir / 'footprint.csv')
```

Unique parameter **elements**, trainable/frozen elements, parameter/buffer tensor
objects ve buffer elements ayrı count alanlarıdır. Her kategori tensor object
identity ile deduplicate edilir. Parameter/buffer union byte'ı cross-category
aynı object'i bir kez sayar. Distinct view objects aynı storage'u paylaşsa bile
logical olarak ayrı sayılır; shared storage ve tied/cross-category alias diagnostics
descriptor'da korunur. Bu convention mevcut Benchmark logical-byte hesabıyla aynıdır.
Mixed CPU dtype, integer/bool buffer desteklenir; meta/sparse/quantized/GPU tensor,
custom state_dict/extra-state/hooks profilleri reddedilir. Model forward çalışmaz,
parameter/buffer/gradient/mode ve RNG değişmez; caller exclusive kullanım gerekir.
Nonpersistent buffer logical byte'a dahildir, normal state_dict dosyasına dahil değildir.

Explicit serialization sibling temporary file → torch.save(state_dict) → flush/
fsync → streamed stat/SHA → optional trusted round-trip → atomic no-clobber publish
yapar. Round-trip yalnız API'nin az önce oluşturduğu temp dosyada, CPU map_location
ve **weights_only=True** ile yapılır. Unsafe load fallback yoktur. Var olan dosya
overwrite=False iken korunur; explicit overwrite=True atomic replacement ister.
Temporary dosyalar failure'da temizlenir. Üretilen kalıcı artifact caller'a aittir;
function onu success sonrasında silmez. Physical memory ölçülmediyse unsupported
kalır; `numel×element_size` gerçek disk byte diye sunulmaz.

Caller'ın mevcut artifact'ı yalnız byte olarak da ilişkilendirilebilir:

```python
existing = measure_footprint(model, artifact_path=model_path,
    artifact_format='pytorch.state_dict', expected_sha256=artifact_info['sha256'])
assert existing.to_dict()['raw_evidence']['artifact']['model_match'] == 'not_verified'
```

Mevcut artifact asla deserialize edilmez. SHA/size gerçek bytes'tan; format
caller-declared'dır, model eşleşmesi veya graph geçerliliği kanıtı değildir.
Boş/missing/directory/symlink/değişen dosya veya expected SHA mismatch reddedilir.
Gelecekte ONNX, LibTorch, TensorRT, OpenVINO, TFLite, Core ML artifact'ları kendi
format identity/validation adapter'larıyla gelebilir; bu sürüm converter/exporter
veya deployment doğrulaması içermez, arbitrary format adını PyTorch'a eşitlemez.

## Mevcut Experiment Manager ile association

```python
from hyso.experiments import ExperimentManager

root = output_dir / 'runs'
manager = ExperimentManager(root)
experiment = manager.create_experiment('P10-P11 CPU memory')
manager.close()
ExperimentManager.migrate_store(root)  # explicit offline schema2 artifact capability
manager = ExperimentManager(root)
run = manager.create_run(experiment['id'], 'benchmark',
    config={'memory_protocol': memory_data['protocol']},
    provenance={'memory_result_id': memory_data['result_id'],
                'footprint_result_id': footprint_data['result_id'],
                'model': footprint_data['provenance']['model']})
run.start()
artifact_ids = []
for path, kind in [(output_dir / 'memory.json', 'performance_result'),
                   (output_dir / 'footprint.json', 'performance_result'),
                   (model_path, 'model_state_dict')]:
    artifact = run.add_artifact(path, kind=kind, ownership='copy', role='output')
    artifact_ids.append(artifact['id'])
    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
for name, metric in [('perf.rss.sampled_peak', memory_data['measurements']['sampled_peak_rss_bytes']),
                     ('perf.model.logical_bytes', footprint_data['measurements']['parameter_buffer_bytes']),
                     ('perf.model.serialized_bytes', footprint_data['measurements']['serialized_artifact_bytes'])]:
    if metric['value'] is not None:
        run.log_metric(name=name, value=metric['value'], unit=metric['unit'],
                       source='performance.memory_v1', step=0, step_kind='observation')
run.finish()
run_id = run.id
manager.close()
manager = ExperimentManager(root)
assert manager.get_run(run_id).to_dict()['status'] == 'completed'
assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in artifact_ids)
manager.close()
```

File/SQLite/network writes, artifact hashing ve export sampling window'dan sonra
yapılır; native OS query ölçümün explicit instrumentation'ıdır. Unsupported/null
metric numeric event yapılmaz. Büyük raw samples metric_events'e çoğaltılmaz,
JSON artifact'ta kalır. Yeni run kind/table/implicit migration yoktur. Sonuç UUID
ile run UUID farklıdır; mevcut generic metric/artifact API'leri kullanılır.
