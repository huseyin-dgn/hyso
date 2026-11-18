# Transfer ve utilization ölçümü — P15/P16

B7 eager dense contiguous, tek `cuda:0` ve Windows worker kapsamındadır.
B4 native crash riski OPEN/UNRESOLVED; B6 CUDA kernel profiling UNSUPPORTED.
Bu API'ler BenchmarkRunner latency'sine instrumentation eklemez.

## CPU sayaçları: mevcut process'in kullanımı

```python
import json
from pathlib import Path
import tempfile
from hyso.performance import UtilizationProtocol, UtilizationResult, measure_utilization

utilization = measure_utilization(protocol=UtilizationProtocol(
    duration_seconds=.12, interval_seconds=.02, gpu=False))
utilization_data = utilization.to_dict()
assert utilization_data['actual']['cpu_samples'] >= 2
assert utilization_data['measurements']['gpu']['tensor_core_utilization']['value'] is None
assert utilization_data['actual']['gpu_queries'] == 0

export_root = Path(tempfile.mkdtemp(prefix='hyso-transfer-utilization-guide-'))
utilization.to_json(export_root / 'utilization.json')
utilization.to_csv(export_root / 'utilization.csv')
restored_utilization = UtilizationResult(json.loads(
    (export_root / 'utilization.json').read_text(encoding='utf-8')))
assert restored_utilization.to_dict() == utilization_data
```

Windows `GetProcessTimes` kernel/user sayaç farkı process'in bütün thread'lerini
içerir. `single_core_percent = 100*(delta_kernel+delta_user)/wall_ns` birden
fazla çekirdekte 100'ü aşabilir. `machine_capacity_percent` bunu OS logical
core sayısına böler: 16 core sistemde bir core tam kullanılırsa 100 ve 6.25.
Sampler'ın kendi CPU maliyeti dahildir; child process CPU zamanı dahil değildir.
Bu bir system-wide CPU metriği değildir. Ayrı raw interval
`system_busy_percent`, `GetSystemTimes` idle dahil kernel değerinden idle
çıkarılarak hesaplanır; 64 logical core üzerindeki system scope unsupported.
Windows dışındaki native CPU adapter unsupported döner; Linux doğrulanmadı.

Çağrı workload çalıştırmaz. Başka thread/process'te devam eden işi örneklemek
için pencereyi onunla eşzamanlayın. `workload_label` yalnız caller beyanıdır;
exclusive attribution veya otomatik workload kontrolü değildir. Counter
reset/negatif delta için wrap tahmini yapılmaz; interval missing/invalid kalır.
Raw midpoint/query timestamps, gaps ve query overhead saklanır. İki kat
interval'i aşan boşluk gap'tir; aradaki kullanım interpolate edilmez.

## Gerçek H2D / D2H kopyaları

CUDA örneği CPU-only ortamda çalıştırılacak rehber bloklarından ayrıdır:

```python-cuda
from hyso.performance import TransferProtocol, TransferError, measure_transfers

copies = measure_transfers(protocol=TransferProtocol(
    shape=(262144,), dtype='float32', directions=('H2D', 'D2H'),
    pinned=True, non_blocking=True,
    warmup_iterations=1, measure_iterations=3, repeats=2,
    timeout_seconds=45, input_limit_bytes=8*1024**2,
    cpu_memory_limit_bytes=3*1024**3, cuda_memory_fraction=.1))
copy_data = copies.to_dict()
assert copy_data['raw_evidence']['tensor']['transfer_bytes'] == 1048576
assert all(row['equal'] and row['terminal_complete']
           for row in copy_data['raw_evidence']['copies'])
assert copy_data['measurements']['overlap']['value'] is None
```

Her çağrı kendi fresh worker'ını, Job Object private-commit limitini,
önceden ayrılmış host/device buffer'larını, stream ve event'lerini kullanır.
Parent Tensor/model kabul edilmez; yalnız shape/dtype/policy reçetesi geçer.
`shape` rank 1–8, pozitif boyutlar; dtype float16/32/64, int32/64 veya uint8.
Layout, sparse, quantized, noncontiguous ve farklı-device profilleri unsupported
olarak `TransferError.diagnostics` üzerinden bildirilir. Silent CPU fallback yok.
Typed CUDA OOM, pinned allocation failure, normal failure, unsupported, timeout
ve unexpected process/native crash ayrı sınıflardır. Crash olursa tekrar deneyip
gizlemeyin; B4 stop kuralını uygulayın. Bu sınıflar başarısız ölçümü başarı
result'ına dönüştürmez.

`pinned` H2D source veya D2H destination host buffer'ına aittir; runtime
`is_pinned()` kaydı vardır. Pageable/pinned ve blocking/nonblocking dört ayrı
explicit profildir; biri diğerinin yerine seçilmez. `non_blocking=True`
asenkronluk talebidir; kanıtlanmış host async veya compute overlap değildir.

Ölçülen copy öncesi hedef farklı değerlerle doldurulur. Terminal event
tamamlandıktan sonra round-trip equality doğrulanır. Validation/reverse copy,
allocation, initialization ve export ölçülen event interval'ine dahil değildir.
Warmup her direction/repeat için ayrıdır. `transfer_bytes=numel*element_size`;
warmup/validation kopyaları bandwidth numerator'ına eklenmez.

- `host_enqueue_ms`: yalnız `copy_` çağrısının host wall aralığı; blocking veya
  pageable staging nedeniyle beklemeyi içerebilir.
- `device_stream_ms`: copy çevresindeki CUDA event aralığı ve terminal completion.
  Özellikle pageable/blocking durumda host staging/stream idle etkileri olabilir;
  CUPTI memcpy activity veya saf PCIe hardware süresi iddiası değildir.
- `completed_bytes_per_second`: toplam ölçülen logical byte / toplam CUDA event
  interval saniyesi. Birim decimal byte/s; GiB/s istenirse 2**30'a bölünür.
  Host enqueue süresi denominator olarak kullanılmaz. Sıfır event süresinde
  bandwidth null olur.

Host monotonic ile CUDA event clock toplanmaz. Compute/copy overlap için ortak
timeline ve gerçek correlation yoktur: `overlap=null/unsupported`, P15 PARTIAL.

## İsteğe bağlı cihaz genelindeki GPU sensörü

```python-cuda
from hyso.performance import UtilizationProtocol, measure_utilization
sensor = measure_utilization(protocol=UtilizationProtocol(
    duration_seconds=2, interval_seconds=.2, gpu=True, gpu_selector='0',
    sensor_timeout_seconds=1, workload_label='independently-running-workload'))
sensor_data = sensor.to_dict()
```

Kurulu `nvidia-smi` read-only NVML tabanlı CSV query backend'i kullanılır;
Python NVML paketi veya driver/toolkit kurulmaz. Her query ayrı, timeout'lu
subprocess'tir. UUID/PCI/name/driver ve native tool path saklanır. Selector,
NVIDIA physical enumeration/UUID/PCI kimliğidir; CUDA_VISIBLE_DEVICES index'i
olarak yorumlanmaz. Tek query'de birden fazla GPU satırı reddedilir.
CLI format değişirse parse hatası unavailable olur; sıfır değer uydurulmaz.
Total memory mevcutsa used/free alanlarının her biri onu aşamaz; tutarsız sorgu
unavailable/null olur. Used+free=total eşitliği varsayılmaz. Aynı kontrol result
validation'da da uygulanır. Pozitif olmayan host counter aralığından process veya
system CPU yüzdesi yayımlanmaz; eksik/geçersiz interval sıfır kullanım değildir.

GPU busy: sensörün penceresinde en az bir kernel çalıştığı süre yüzdesi.
Memory-controller busy: device memory read/write aktif zaman yüzdesi.
Memory used/free/total: MiB çıktısından byte'a çevrilmiş device occupancy.
Bunlar birbirlerinin yerine geçmez ve yalnız HYSo worker'ına ait değildir.
Process-specific GPU usage ve Tensor Core utilization bu backend'de null'dır.

GPU sensörünün iç sample penceresi query interval'i değildir; ürüne göre yaklaşık
1/6–1 saniyedir ve CLI kesin pencere sınırını vermez. Kısa işler görünmeyebilir
veya sensör penceresi collection dışına taşabilir. Sensor wall timestamp ile
host monotonic query start/end ayrı clock domain'lerde tutulur. Alanların
`sample_mean` değeri aritmetik sample ortalamasıdır; time integral veya
workload average değildir. Eksik/N/A alanlar null/status/reason olarak kalır.
Minimum valid interval/sample ve max_samples/duration sınırları vardır;
query overhead collector maliyetidir, baseline'dan çıkarılmaz.

## Experiment Manager

Pencere bittikten sonra standart benchmark run/generic metric/copied artifact
akışı kullanılır; migration veya schema değişikliği yoktur:

```python
from hyso.experiments import ExperimentManager
store = export_root / 'runs'
manager = ExperimentManager(store)
experiment = manager.create_experiment('transfer-utilization')
manager.close()
ExperimentManager.migrate_store(store)
manager = ExperimentManager(store)
run = manager.create_run(experiment['id'], 'benchmark', config={'feature': 'P16'})
run.start()
artifact = run.add_artifact(export_root / 'utilization.json',
    kind='performance_result', ownership='copy')
run.log_metric(name='cpu.samples', value=utilization_data['actual']['cpu_samples'],
    unit='count', step=0, step_kind='observation', source='performance.utilization')
run_id = run.id
run.finish()
manager.close()
manager = ExperimentManager(store)
assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
assert manager.get_run(run_id).read_metrics()
manager.close()
```

TransferResult da aynı JSON/CSV/artifact API'sini kullanır; bağımsız schema1 ve
P15 feature identity taşır. UtilizationResult P16'dır. Result'lar immutable plain
data, export atomic/no-clobber'dır. JSON tam kayıt, CSV observation payload
satırlarıdır. Önceki Benchmark/checkpoint/Experiment Manager schema'ları değişmez.

Resmi kaynaklar: [Torch CUDA stream/event ve pinned memory semantiği](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst),
[NVIDIA sensör tanımları](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
[Windows process CPU times](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes),
[Windows system CPU times](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemtimes).
