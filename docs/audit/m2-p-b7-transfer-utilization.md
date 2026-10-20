# M2-P-B7 — CPU–GPU transfer ve CPU/GPU utilization

2026-10-20, Europe/Istanbul. Branch `feature/hyso-models-next`, başlangıç HEAD
`787065d353638e9d687d2e7c9a482fec00f1d2d3` (beklenen `787065d` ile eşleşir).
Başlangıç index'i boş; tek working-tree farkı kullanıcının gradient testiydi.
Kanıt dizini **E**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb7-20261020`.

**P15 PARTIAL; P16 PASS within verified Windows scope. B7 PARTIAL / full-scope
NO-GO.** Gerçek completed-copy ve sensör ölçümleri var; compute/copy overlap
ve CUPTI copy/kernel correlation yok. Yeni conditional acceptance ilan edilmedi.
B8 uygulanmadı ve bu kapıyla B8'e geçiş önerilmez.

## 1. Ön kontrol, mimari ve korunan sınırlar

M2-P-A feature contracts/DAG ve B7 gate; B4/B5/B6 audit'leri, B4/B6 koşullu
kabul kayıtları, P09/P10, B3 ownership, B5 guard ve B6 profiler incelendi.
[B4 kabulü](m2-p-b4-conditional-acceptance.md) ve
[B6 kabulü](m2-p-b6-conditional-acceptance.md) yalnız development continuation
içindir; stable/public/PyPI release onayı değildir.

Torch **2.5.1+cu121**, CUDA build12.1, RTX3060 Laptop6GB, driver591.74;
runtime `supported_activities` yalnız CPU, build `LIBKINETO_NOCUPTI` içerir.
B6 kernel profiling tamamlanmış sayılmadı. B4 orijinal native fail-fast
**OPEN / root cause UNRESOLVED** olarak korunur; vendor/HYSo kusuru veya fix
kanıtı üretilmedi. Yeni native crash olursa test runner durur; crash OOM
veya sensor-unavailable şeklinde gizlenmez.

P15'in P14 hard dependency'si, overlap/common timeline/correlation alanında
engeldir. Event tabanlı completed-copy alt kapsamı bu engeli aşmış gibi
gösterilmedi. P16'nın OS/vendor telemetry yolu CUPTI gerektirmez.

## 2. Additive public API ve schema

```python
TransferProtocol(shape=(262144,), dtype='float32', directions=('H2D','D2H'),
    pinned=False, non_blocking=False, device='cuda:0', layout='contiguous_dense',
    warmup_iterations=2, measure_iterations=5, repeats=2,
    timeout_seconds=45, cpu_memory_limit_bytes=3*1024**3,
    input_limit_bytes=8*1024**2, cuda_memory_fraction=.1)
measure_transfers(*, protocol=TransferProtocol()) -> TransferResult
# Başarısız admission/ölçüm: TransferError.diagnostics

UtilizationProtocol(duration_seconds=1, interval_seconds=.1, max_samples=128,
    min_valid_intervals=2, gpu=False, gpu_selector='0',
    sensor_timeout_seconds=1, workload_label=None)
measure_utilization(*, protocol=UtilizationProtocol()) -> UtilizationResult
```

Result'lar independent schema_version1, P15/P16 identity, plain immutable data,
validated derived metrics, raw evidence, runtime/device/sensor provenance ve
atomic no-clobber JSON/CSV taşır. JSON tam kayıt; CSV observation payload
satırlarıdır. BenchmarkResult schema1, checkpointv1 ve Experiment Manager
schema/storage/run kind değiştirilmedi. Önceki result yolları değiştirilmedi;
dispatcher/export'a yalnız yeni kind branch'leri eklendi.

Yeni production modülleri: `transfers.py`, `_transfer_worker.py`,
`utilization.py`, `_telemetry.py`. API/result dispatch ve explicit packaging
inventory additive genişledi. Yeni genel benchmark framework veya dependency yok.

## 3. P15 ölçüm ve resource protokolü

API caller Tensor/model kabul etmez; yalnız explicit recipe fresh child'a gider.
Windows Job Object private-commit3GiB ve active-process1 guard'ı B5'ten,
owned-process-tree timeout termination B3'ten yeniden kullanılır. Guard Torch
importundan önce, input logical byte admission allocation'dan önce uygulanır.
En fazla256 primary copy (warmup dahil), input cap64MiB ve timeout cap600s.
Doğrulama profilleri FP32 1MiB ve uint8 128KiB ile çok daha küçüktür.
CUDA allocator fraction0.1 yalnız worker'dadır. Hidden CPU fallback, reset,
unlimited retry/allocation, global GC veya parent allocator reset yoktur.
Pinned allocation failure ayrı sınıflanır; bu görevde allocation başarısızlığını
zorlamak için bellek baskısı üretilmedi. Başarılı pinned allocation native
olarak doğrulandı; failure branch'i native pressure testiyle kanıtlandı denmez.

Dense contiguous tek cuda:0 kapsamı; sparse/quantized/noncontiguous/cross-device
istekleri explicit unsupported reason ile reddedilir. Supported dtype'lar
float16/32/64,int32/64,uint8; gerçek native byte kanıtı bu görevde FP32/uint8
içindir. CPU input byte accounting deterministic testlerle ayrıca doğrulandı.

Preallocated host/device buffers, owned stream/event'ler kullanılır. Her copy
öncesi hedef poison edilir. H2D için GPU→CPU reverse validation, D2H için
CPU→GPU initialization ve completed D2H equality vardır. CPU output yalnız
terminal event tamamlandıktan sonra okunur. Allocation, poison, initialization,
warmup, validation/reverse copy ve persistence ölçülen copy aralığı dışındadır.
`validation_copies`, ölçülen copy'lerin validation pass sayısıdır; toplam PCIe
transaction sayısı değildir. Warmup validation'ları ayrıca başarı denetiminden
geçer, bandwidth numerator'ına dahil edilmez.

Host enqueue `copy_` host çağrı aralığıdır ve bloklanabilir. CUDA event süresi
stream interval'idir; pageable staging ve host/event scheduling boşluklarını
içerebilir. **CUPTI memcpy activity veya saf fiziksel PCIe süresi değildir.**
Completed bandwidth = toplam ölçülen `numel*element_size` byte / toplam CUDA
event saniyesi. Host enqueue payda değildir; host ve event clock çıkarılmaz/
toplanmaz. Sıfır event süresinde bandwidth null olur.

Pinned H2D source / D2H destination gerçek `is_pinned()` ile doğrulandı.
`non_blocking` yalnız request'tir; doğrulanmış async/compute overlap değildir.
Overlap `null/unsupported`, reason `no_verified_common_copy_compute_timeline_or_kernel_correlation`.

## 4. Yeni installed native copy sonuçları

Her FP32 profil/direction için **6 ölçülen copy** (N3×R2), direction/repeat
başına W1; copy başına **1048576 byte**. Aşağıdaki GB/s decimal10^9 byte/s.
Host/event kolonları copy başına aritmetik ortalama; bandwidth total-byte /
total-event-time ile hesaplanır. Küçük instrumented örneklerdir; kontrollü
karşılaştırma veya cihazın garanti peak'i değildir. CPU regresyonları da ayrı
process'te yürüdüğünden bu sayılardan saf pinning hızlanması sonucu çıkarılmaz.

| Host | non_blocking | Yön | Enqueue ms | Event interval ms | Completed GB/s |
| --- | --- | --- | ---: | ---: | ---: |
| pageable | False | H2D | 0.348750 | 0.363584 | 2.884 |
| pageable | False | D2H | 0.403367 | 0.402331 | 2.606 |
| pageable | True | H2D | 0.075517 | 0.233909 | 4.483 |
| pageable | True | D2H | 0.381033 | 0.364965 | 2.873 |
| pinned | False | H2D | 0.187550 | 0.199541 | 5.255 |
| pinned | False | D2H | 0.182617 | 0.191792 | 5.467 |
| pinned | True | H2D | 0.022950 | 0.181269 | 5.785 |
| pinned | True | D2H | 0.016133 | 0.173131 | 6.057 |

uint8 pinned/nonblocking ek profili: shape131072, **131072 byte/copy**,
direction başına N2×R1; H2D2.624 / D2H2.840GB/s. Toplam **52 ölçülen primary
copy +18 warmup primary copy** tamamlandı ve doğrulandı. Reverse/initialization
trafikleri bu sayının measured-byte numerator'ına eklenmedi.

E/installed-cuda-b7.json tüm row timestamp/event/equality/worker receipts'lerini,
E/b7-measurement-summary.json ise bu tabloyu türeten değerleri taşır.

## 5. P16 CPU ve GPU sensör yolu

Windows `GetProcessTimes` kernel/user FILETIME değerleri100ns'den ns'ye
çevrilir; iki native counter snapshot farkı host monotonic interval'e bölünür.
Single-core yüzde100'ü aşabilir; machine-capacity yüzde OS logical CPU sayısına
bölünür. System-wide `GetSystemTimes` ayrı interval alanıdır; kernel idle içerir,
busy hesabında idle çıkarılır. >64 logical core için system-wide scope
unsupported; Linux native CPU adapter bu görevde unsupported/doğrulanmadı.
Sampler/aynı process thread CPU maliyeti process metriğine dahildir, child CPU
zamanı dahil değildir. Sentetik16-core referansı100% /6.25%'i doğrular.

Native16-core CPU yük penceresi: 12 snapshot, 11 valid interval,
valid wall369392350ns; **101.518074% single-core, 6.344880% machine-capacity**.
Bu ölçüm workload thread + sampler içindir; workload'ın çıplak CPU yüzdesi
olduğu iddia edilmez. Raw kernel/user deltas ve ayrı system yüzdeleri saklıdır.

`pynvml` binding'i yok, System32 NVML DLL'i var. Yeni paket kurulmadan mevcut
read-only **nvidia-smi/NVML CLI** query backend'i kullanıldı. Query'ler explicit
physical selector ve bounded timeout ile shell'siz subprocess'tir. UUID/PCI/
name/driver/tool path kaydedilir; tek GPU CSV satırı ve numeric ranges validate
edilir. Format/version değişikliği unavailable olur. Process-specific GPU
usage veya Tensor Core counter bu backend'de sağlanmaz: null/unsupported.

Native GPU: `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`,
PCI `00000000:01:00.0`, driver591.74. Selector CUDA_VISIBLE_DEVICES index'i
değil, fiziksel NVIDIA enumeration/UUID/PCI domain'idir.

| Pencere | GPU sorgusu / valid busy sample | GPU busy sample mean | Memory-controller busy mean | VRAM used sample mean |
| --- | --- | ---: | ---: | ---: |
| Owned GPU işinin olmadığı idle, 1.2s request | 8 /8 | 12.75% | 1.00% | 847642624 byte |
| Ayrı bounded matmul child ile 2s request | 14 /13 | 17.692308% | 1.307692% | 1058658461.538462 byte |

Idle cihaz boş garantisi değildir; desktop/diğer process'ler sensöre dahildir.
Load child küçük256×256 matmul ve3s limit/Job guard kullanır; gerçek iteration
ve host start/end receipt'i ölçümle çakışır. Belirli bir busy yüzdesine erişmek
assert edilmedi; load-idle farkı HYSo'ya exclusive attribution değildir.
VRAM used/free/total vendor MiB çözünürlüğünden byte'a çevrildi; Torch allocator
P09, memory-controller busy veya compute kullanımının yerine geçmez.

Load penceresinde son query kalan deadline'a sığmadı: **1 timed_out query**
null values/reason ile saklandı; 13 geçerli sample'dan mean alındı. Idle/load
CPU snapshot sayıları8/14, gap sayısı0/0. Sensor query dahil toplam wall overhead
565356100ns /955605200ns; yüksek CLI maliyeti saklanmaz veya baseline'dan
çıkarılmaz. CPU-load penceresinde query overhead274476100ns, scheduling/GIL
beklemesini de içerebilir; native API'nin yalın maliyeti değildir.

Sensor internal sample period ürüne göre yaklaşık1/6–1s; CLI kesin pencere
başlangıç/bitişini vermez. Query interval150ms ile aynı şey değildir. Raw
sensor wall timestamp ve host monotonic query bounds ayrı clock'lardır.
`sample_mean` yalnız aritmetik sample ortalaması; continuous time integral veya
exact workload busy değildir. Kısa iş sensör penceresinden küçükse kaçırılabilir.
Counter reset/negatif delta için wrap tahmini yok; invalid/missing/gap alanları
vardır. N/A/unavailable/permission/timeout null+status+reason döner; CPU verisi
GPU yokken de korunur. Native sensor process crash ayrı stop exception'ıdır.

## 6. CPU, native ve package doğrulaması

Yeni B7 CPU testleri **58 PASS**: byte/duration/unit hesapları; deterministic
CPU normalizasyonu, kernel/user/system idle farkı; reset/invalid/gap/missing;
sensor parsing/N/A/range/multiple-device/timeout/permission/native-crash stop;
budget/unsupported/worker timeout; parent RNG; immutable JSON/CSV/no-clobber;
tamper rejection ve Experiment Manager reopen/SHA.

İlk test invocation'ındaki junit argüman hatasında test çalışmadı. Sonraki
23PASS/1FAIL selector validation eksiğini yakaladı; production regex düzeltildi.
34PASS/1FAIL koşusundaki tuple/plain-data hatası test fixture'ın worker wire
formatını taklit etmemesiydi; JSON list marshaling ile düzeltildi, schema
gevşetilmedi. Sonra50PASS; ek kontrollerle final58PASS. Alt kümeler tekrar
sayılmadı, eski failure log/XML'leri E'de korundu. Skip/xfail/tolerans gevşemesi yok.

Wheel/sdist packaging **4 PASS**, 147.54s; tüm38 production modülü içerir.
Minimum Python3.10.18 / Torch2.2.2+cpu / torchvision0.17.2+cpu installed smoke
PASS: B7 CPU rehberi/OS counter, CUDA-unavailable admission, önceki API/Trainer/
EManager/rehberler çalıştı, parent CUDA initialized=False kaldı. Paket yalnız
HYSo local wheel'den --no-index/--no-deps ile TEMP target'a kuruldu; mevcut
dependency METADATA/RECORD before/after eşit. Wheel, sdist, installed target
ve source production byte eşitliği denetlendi.

Local wheel163322 byte, SHA256
`8c59394fdbdf5dc6a74c5319f2ac16459b7bf5a5cfe82cc0a1ff13b5fbeb861d`;
sdist146710 byte, SHA256
`1a042664257997c6652751e9ccf3d80bdb7b0ab18edc03b7aa42875af6c7e38a`.
E/archive-integrity.json tüm38 modülü archive byte'larıyla karşılaştırır;
E/source-freeze.json production+test+rehber42 kaynak snapshot'ıdır.

Yeni B7 installed native **9 senaryo PASS**: dört FP32 pin/nonblocking profili,
uint8, idle sensor, controlled-load sensor, native CPU load, E artifact/reopen.
Bu9'un tamamı CUDA kernel profiling testi değildir. Parent canlı model/grad/
mode/hook, Python/NumPy/CPU+CUDA RNG, current device/non-default stream ve
allocated/reserved/peak değerleri korunur. Successful worker PID'leri exit
kontrolüyle ölü; timeout path owned-tree cleanup CPU testinde doğrulandı.

Experiment Manager native7 gerçek Transfer/Utilization result artifact'i
copied ownership ile aldı; generic observation metrics, finish/close/reopen,
read_metrics ve bütün SHA verification'ları PASS. Yazılar ölçüm pencerelerinin
dışında. CPU testindeki synthetic transfer fixture gerçek GPU sonucu diye
sunulmadı. Minimum smoke da bağımsız artifact/SHA/reopen akışını çalıştırdı.

İlgili **1629 CPU regresyon kontrolü PASS**, 601.79s, 570 mevcut warning,
0 fail/error/skip. Kapsam: B1/B2/B3/B4/B5/B6 performance; Benchmark/P09
protocol/state/export; Trainer/checkpoint/save/restore/transaction/updates/
lifecycle/CPU replay; Experiment Manager; public/model-state API uyumluluğu.
Bu1629, ayrı packaging4 ve yeni B7 CPU58 ile **1691 distinct CPU kontrolü**
bu görevde geçti. Tek1691-test pytest invocation'ı değildir. Tarihsel sayılar
bu toplama eklenmedi; tekrar edilen50/58 alt kümeler iki kez sayılmadı.

Final wheel üzerinde yeni çalıştırılmış eski native kapsam:

| Native kontrol grubu | Senaryo |
| --- | ---: |
| Windows P10 RAM | 3 |
| B6 profiling | 6 |
| B5 resource/capacity | 5 |
| Benchmark/P09 | 45 |
| Experiment Manager CUDA | 2 |
| B3 pipeline/cold | 8 |
| B1 timing | 15 |
| B4 sweep/precision | 11 |

Eski CUDA grupları toplam92; B7'nin9 karma native senaryosu ve Windows3 ile
**104 distinct native kontrol**. B6 CUDA rehberi de ayrıca çalıştı, formal
senaryo sayısına eklenmedi. B6 validation PASS kernel profiling PASS değildir;
activity hâlâ unsupported. Beklenen negative-fixture exception/OOM/timeout'ları
ayrı tutuldu; unexpected native crash görülmedi. Yeni native Trainer replay196
veya full CPU2313 gate tekrarlanmadı.

E/verification-summary.json final counts/source/protection kaydıdır.
42/42 freeze kaynağı aynı, tüm38 production dosyası installed target/wheel/
sdist ile byte-equal. Başlangıçtaki mevcut dosyalardan yalnız API namespace,
result dispatch/export, packaging ve backlog değişti; diğer mevcut kaynaklar,
testler ve B4/B5/B6 audit/kabul belgeleri byte-identical. Index boş; branch ve
HEAD aynı. Yeni8 dosya ile toplam13 working-tree farkı vardır; bunun1'i
başlangıçtaki kullanıcı gradient farkıdır. Bu görev12 dosya ekledi/değiştirdi.

## 7. Son kapsam ve kapı

- **P15 PARTIAL:** completed-copy/timing/byte/policy/equality ve izolasyon alt
  kapsamı PASS. P14 tabanlı gerçek copy/compute overlap timeline eksik.
- **P16 PASS within verified scope:** Windows process/system CPU ve bu RTX3060
  cihazındaki desteklenen device-wide NVIDIA sensörleri. Diğer OS/GPU/build
  veya opsiyonel process GPU/Tensor Core counters için PASS iddiası yok.
- Backlog: P15 partial, P16 implemented within scope; **13 implemented,
  5 partial, 17 planned**. P12/P13/P14 partial statüleri korunur.
- M2-P-A B7 full gate'in copy/overlap trace kapsamı kapanmadı: **B7 PARTIAL,
  full-scope NO-GO**. Sensor erişimi mevcut olsa da eksik overlap saklanmaz.
  Yeni conditional acceptance yok; B8'e geçiş önerilmez/başlanmadı.
- Sonraki en küçük eksik kanıt P14/CUPTI etkin destekli backend'de gerçek copy
  ve compute timeline/correlation'dır. Mevcut LIBKINETO_NOCUPTI ortamına kör
  sync/GC veya tekrar gate eklemek bunu sağlamaz. Bu görev ortam güncellemedi.

B4 native crash risk **OPEN**, root cause **UNRESOLVED**; B6 koşullu kabul ve
CUDA profiler unsupported sınırı değişmedi. Tarihsel CPU2313/native196 ve
önceki B6 test sayıları yeni koşuya taşınmadı. Yeni başarılı koşular eski
failure'ı geçersiz veya çözülmüş yapmaz. Release onayı verilmez.

Kullanıcının gradient SHA256 başlangıç/final:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Stage/commit/push/reset/checkout/rebase/merge yok. Driver/Torch/CUDA/dependency,
model/weight/dataset indirme veya güncelleme yok. Dump açılmadı/yüklenmedi.

Resmi referanslar ve API semantiği:
[Torch2.5.1 CUDA stream/event/pinning](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst),
[NVIDIA query ve sensor tanımları](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
[GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes),
[GetSystemTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemtimes).
