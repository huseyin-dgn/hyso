# M2-P-B2 — CPU Peak RAM & Model Footprint

2026-10-20, Europe/Istanbul. **PASS — P10 native Windows current-process RSS/HWM;
P11 dense CPU logical footprint ve gerçek artifact bytes/SHA kapsamı.** Native
Linux doğrulaması yoktur; fully verified platform kapsamı Windows ile sınırlıdır.
Bu görev B3, memory-leak diagnosis, conversion engine veya deployment implementasyonu
değildir. Full CPU/162 CUDA birleşik gate çalıştırılmadı.

## 1. Branch/HEAD, başlangıç ve dosyalar

Branch başlangıç/son `feature/hyso-models-next`; HEAD
`2b1825c3ed4c7eba4ce0e680d854b89a7a379b48`, `feat: add latency analytics and sustained throughput`.
B1 raporu/kodu bu commit'te doğrulandı. Index boş; başlangıçtaki tek kullanıcı farkı
`tests/test_gradient_accumulation.py` korundu, yazılmadı/stage edilmedi/geri alınmadı.
Başlangıç/son SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

Görevin dokuz dosyası:

- Yeni: `hyso/performance/memory.py`.
- Güncellenen: `hyso/performance/__init__.py`, `hyso/performance/result.py`.
- Yeni: `tests/test_performance_memory.py`, `tests/native_memory_validation.py`.
- Güncellenen: `tests/test_packaging.py`.
- Yeni: `docs/guides/performance-memory.md`, bu audit.
- Güncellenen: yalnız P10/P11 ve tarihsel/current count açıklamaları için backlog v1.1.

BenchmarkRunner/Result schema1, P01/P02 inference.py, P09 allocator protocol,
Trainer/checkpoint format1, Experiment Manager, root/legacy exports, metadata/
dependency/version, sabit roadmap v1.0 ve M2-P-A tasarımı byte-identical korunur.
Commit/push/branch/checkout/reset/rebase işlemi, dependency/weights/dataset indirme
veya kurulumu yoktur. Yalnız kendi local HYSo wheel'i explicit isolated target'lara
kuruldu; mevcut dependency prefix'i değiştirilmedi.

## 2. API ve versioned sonuçlar

Yeni additive public exports:

```python
MemoryProtocol(interval_seconds=.01, max_samples=10000)
measure_memory(workload=None, *, protocol=MemoryProtocol())
measure_footprint(model, *, artifact_path=None, serialize_to=None,
                  artifact_format=None, verify_roundtrip=False,
                  expected_sha256=None, overwrite=False)
MemoryResult
FootprintResult
```

Mevcut frozen/plain-data `_Result`/atomic export primitive'leri yeniden kullanılır;
schema dispatch ve CSV'ye yalnız yeni türler için branch eklenmiştir. Yeni bir
registry/framework/tracking adapter veya production worker orchestration yoktur.
Independent format'lar `hyso.performance.memory` / `hyso.performance.footprint`,
schema_version=1; BenchmarkResult schema1 değildir. UUID result_id, P10/P11 feature
ID, versioned protocol/scope, provenance, requested/actual gözlem bilgileri,
unit/status/reason, raw evidence ve diagnostics bulunur. Memory'de interval/max
observations bütçesi ve actual count; footprint'te requested artifact/round-trip,
actual tensor entry/artifact count kaydedilir. Model identity descriptor/weights
hash; artifact identity path/format/SHA'dır; Manager run ID sonuç UUID'sine yazılmaz.

to_dict defensive-copy, JSON finite/allow_nan=False, frozen/slots sonuç içinde
tensor/model/thread/filehandle bulunmaz. CSV memory'de observation, unsupported
durumda metric; footprint'te metric satırlarıdır. JSON canonical complete record'dur.
No-clobber, atomic publication/overwrite, UTF-8, formula-safe CSV ve temp cleanup
korunur. Eksik/nonfinite/forged count/unit/gap/HWM/hash/physical-memory payload'ları
reddedilir; unsupported alanlar null+reason'dır, sahte0 değildir.

## 3. P10 native OS sözleşmesi

Windows stdlib ctypes `GetProcessMemoryInfo` ile `WorkingSetSize` ve
`PeakWorkingSetSize` okur. Bu native byte working set/RSS'tir; private commit,
Python heap, model-only RAM veya GPU memory değildir. Linux adapter
`/proc/self/status` VmRSS/VmHWM okur; kB×1024 byte ve alan/unit/duplicate/inconsistency
validation vardır. tracemalloc/psutil mandatory dependency veya fallback yoktur.

`workload=None` tek gerçek snapshot verir. Senkron callable varsa baseline query →
bounded sampler thread → caller workload → stop/join → final query akışı uygulanır.
Workload return value final observation'dan önce bırakılır. Query start/end ve
midpoint monotonic ns, RSS/HWM, backend, pid, actual adjacent gaps ve window duration
raw evidence'tır. Sampled peak yalnız max(observed RSS); absolute window peak değildir.
Lifetime HWM process ömrünü içerir, baseline/current ayrı ve reset=False zorunludur.
RSS release sonrası baseline'a dönmek zorunda değildir; Torch/OS cache retain edebilir.

Sampling interval scheduling talebidir; OS query, GIL/thread/scheduling ve sample
buffer allocation overhead'i explicit scope'tadır. Actual gaps değişebilir; kısa
transient kaçabilir. Max_samples bütçesi aşılırsa hata verilir, sample atıp başarılı
peak üretilmez. Workload exception, mid-window backend loss veya thread-start failure
thread/lock cleanup ile propagate edilir. Initial unsupported backend workload'u
çalıştırmaz ve bütün ölçümler null/unsupported/reason olur.

Window'da API file/SQLite/network write veya artifact hash yapmaz; Linux proc read
native query instrumentation'ıdır. Export/config/platform-detail/Manager işlemleri
window sonrasındadır. GPU sync/allocator reset/empty_cache/CUDA RNG çağrısı yoktur.

## 4. Windows adapter cache düzeltmesi

İlk işlevsel gate sonrasında incelemede her sorguda yeni ctypes Structure/POINTER
oluşturmanın global pointer-type cache'i büyüttüğü bulundu. Gerçek eski installed
source üzerinde 20 query **22→42** (`E/ctypes-cache-before.json`); ilk checkout
diagnostic'i23→43 gösterdi. Bu instrumentation'ın kendi RSS etkisini gereksiz artırır.

Native API/Counters type tanımı artık bounded `lru_cache(maxsize=1)` ile process
başına bir kez hazırlanır; yalnız Counters instance her query'de geçicidir.
Fresh native worker'da 20 query **23→23**, installed final worker'da da cache count
eşitliği assert edilir. Bu P32 leak analysis özelliği değildir; P10 adapter'ının
resource-safety düzeltmesidir. Düzeltme sonrası gerekli targeted gate, Windows
native ve final wheel/minimum/installed testleri yenilendi; önceki build final
kanıt diye kullanılmadı.

## 5. P11 logical ve disk accounting

İteratif, cycle-checked module traversal; mevcut Benchmark `_descriptor` ve `_counts`
primitive'lerini kullanır. Ordinary initialized dense CPU Tensor/Parameter,
mixed floating dtype, integer/bool buffer desteklenir. Meta/sparse/quantized/GPU,
custom state_dict/extra-state/hooks profilleri açıkça reddedilir; forward veya
caller model movement yoktur.

Parameter elements/trainable/frozen elements, unique parameter/buffer tensor objects,
buffer elements ve logical byte alanları ayrıdır. Tensor object identity kategori
bazında dedup; parameter/buffer union cross-category aynı object'i bir kez sayar.
Distinct view objects ortak storage paylaşsa da logical olarak ayrı sayılır;
object/storage aliases ve strides descriptor'da korunur. Bu mevcut Benchmark
convention'ıdır, değiştirilmedi. Native/CPU testleri tied parameter, cross-category
aynı object, view aliases, FP32/FP64/FP16/int64/bool, nonpersistent buffers ve shared
child modules içerir. Nonpersistent buffer logical'da bulunur, normal serialized
state_dict'te bulunmaz.

`physical_memory_bytes` null/unsupported/physical_memory_not_measured'dır.
Logical byte toplamı physical storage, RSS veya serialized file size diye sunulmaz.

## 6. Artifact format, publication ve safe verification

Mevcut `artifact_path` opaque byte observation'dır: explicit caller-declared format,
regular/non-symlink file, streamed SHA256 ve fstat/stat/byte-count consistency.
Empty/missing/directory, concurrent change veya supplied expected SHA mismatch
reddedilir. Caller checkpoint **deserialize edilmez**, format/model match
`caller_declared_not_deserialized` / `not_verified` kalır. Model hash ile artifact
hash farklı kimliklerdir; bytes bilinmesi graph/model validity kanıtı değildir.

Explicit `serialize_to` yalnız `pytorch.state_dict` üretir: sibling temporary file,
torch.save, flush/fsync, gerçek stat+stream hash, optional round-trip, atomic link
no-clobber veya explicit replace. Round-trip yalnız API'nin az önce oluşturduğu
trusted temp dosyada `torch.load(map_location='cpu', weights_only=True)` ile;
key/type/dtype/shape/stride ve bitwise values karşılaştırılır. Unsafe fallback yoktur.
Default/verify format-validation durumları ayrı; verified payload caller artifact'a
uydurulamaz. Üretilen kalıcı dosya caller-owned'dır; temp dosyalar failure'da temizlenir.
Success/failure paths registered state, gradient, modes ve RNG invariant testlidir.

Format string PyTorch checkpoint'ine evrensel eşitlenmez: gelecekte ONNX/LibTorch/
TensorRT/OpenVINO/TFLite/Core ML kendi artifact validation adapter'ını sunabilir.
Bu sürüm opaque format label'ını execute/convert etmez; YOLO veya exporter geliştirilmedi.

## 7. RNG, lifetime ve process ownership

Python/NumPy/Torch CPU RNG finally restore edilir; ortak Benchmark/performance lock
concurrent/reentrant çağrıları reddeder. P11 caller modelini/parameter/buffer/grad/mode
değiştirmez; custom serialization side effects destek dışıdır. P10 arbitrary callable
own mutations/resources sahibidir; workload model mutasyonlarına rollback iddiası
yoktur. Caller-exclusive process/RNG kullanımı gerekir. GPU model/RNG restore
vaadi verilmez çünkü bu CPU API CUDA çağırmaz, GPU tensors P11'de reddedilir.

Yeni traversal recursive self-capturing closure kullanmaz. GC disabled model/tensor
weakref testleri ve sampler stop/join/failure lock cleanup geçer. C1'in mevcut dört
model/parameter/buffer closure-release regression'ı değişmeden PASS. P09 reset ve
CUDA event latency protokolleri aynı source byte'larında kalır.

Production API hidden subprocess başlatmaz: current-process scope ve lifetime HWM
ayrımı için worker zorunlu değildir. İzolasyon isteyen caller dış harness'in
spawn/importable entry-point ve termination ownership'ini üstlenir. Native Windows
harness bu sınırı gerçek fresh interpreter worker'larıyla doğrular; callable için
hard timeout/cancellation veya arbitrary transaction vaat edilmez.

## 8. Gerçek Windows ve Linux kanıtı

`tests/native_memory_validation.py` gerçek Windows zorunlu; unavailable OS skip
değil failure'dır. Üç fresh subprocess senaryosu:

1. 64MiB ordinary CPU tensor allocate/fill/touch; sampler held-phase observation;
   gerçek RSS artışı, actual sample peak ve lifetime HWM invariants; release sonrası
   weakref ölümü; ctypes cache stability; gerçek state_dict round-trip/stat/SHA.
2. Workload error: sampler thread join, CPU RNG/lock cleanup, beklenen exception.
3. Abrupt `os._exit(7)`: parent expected exit7, process poll/termination ve başarılı
   result dosyası olmaması. Bu case'in PASS'i workload'un başarılı olduğu demek değildir.

Cache-fix source native **3/3 PASS**, final installed native **3/3 PASS**.
`native-windows-cachefix*.json` ve `installed-windows-final2*.json` raw phase/RSS/HWM/
pid/timestamps/exit/cache/footprint kanıtını içerir. Örnek source worker baseline
588161024 byte, sampled peak655634432 byte; bunlar gerçek process observations'dır,
64MiB'nin model-only attribution'ı veya mutlak peak iddiası değildir.

Linux: parser/unit/order/units/invalid-field protocol testleri Windows CPU'da geçti.
Read-only WSL inventory'de docker-desktop Linux shell bulundu; `python3`/`python`
bulunmadı (probe exit127). Torch/dependency kurulmadı; native Linux adapter/workload
ve installed Linux doğrulaması **çalıştırılmadı**. `E/linux-runtime-probe.log` bu
sınırı kaydeder. Fully verified platform Windows; Linux destek kodu native-unverified.

## 9. Yeni CPU testleri ve targeted regression

Evidence kökü **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb2-20261020`.
Nihai `targeted-final2.xml/log`: **665 PASS, 0 FAIL/ERROR/SKIP/XFAIL/XPASS**,
pytest95.42s,26 mevcut warning (2 CPU-autocast DeprecationWarning,24 GradScaler
FutureWarning). Source test/assertion/tolerance gevşetilmedi; skip/xfail eklenmedi.

| Güncel hedefli kapsam | PASS |
|---|---:|
| Yeni memory/footprint tests |83|
| Performance B1 P01/P02 |122|
| Benchmark CPU85/state28/result24 |137|
| Experiment core60/metrics59/storage28/artifacts36/migration11/recovery17/retry19/integration29 |259|
| Public41/published1/model state18 |60|
| Packaging |4|
| Toplam |665|

Yeni83: native OS read; allocation/release, gaps/query overhead/HWM, unsupported
OS/backend, thread/budget/async rejection/failure, RNG/no-GPU/no-window-I/O,
alias/mixed-dtype/logical accounting, own/caller artifact/SHA/format, corrupt and
changing file, publication failure, frozen/plain/finite validation, CSV formula
safety, GC lifetime, actual Experiment reopen association ve executable guide.

İlk 78-test koşusunda test fixture buffer adı `half`, nn.Module.half metoduyla
çakıştı:77 PASS/1 FAIL; fixture adı düzeltildi, assertions gevşetilmedi. İlk
665-PASS gate sonrasında ctypes cache bug'ı bulunup production düzeltildi; final
665-PASS source gate ve gerekli package/native testleri bu nedenle tekrarlandı.
Önceki koşular nihai toplama eklenmez. Geçmiş B1 1153 veya full1995/1 SKIP güncel
koşu diye sunulmaz. Full Trainer/full CPU suite çalıştırılmadı.

## 10. CUDA/P09/C1 koruma

Gerçek RTX3060 Laptop/capability8.6, Python3.10.18/Torch2.5.1+cu121/CUDA12.1,
native Benchmark report driver591.74/6144MiB. Ayrı interpreter'larda sıralı:

- Mevcut `cuda_performance_validation.py`: **15 PASS**; P01 raw1000 sample P99,
  FP32/AMP sustained, non-default stream, async terminal drain/count/tail,
  bounded outputs/events, RNG/state ve controlled error paths.
- Mevcut `cuda_benchmark_validation.py`: **45 PASS**; P09 native allocator probe,
  disabled no-reset/previous peaks/release, event/stream/model guards dahil.
- CPU state28 içindeki mevcut dört C1 weakref regression'ı PASS.

`native-performance.json/log`, `native-benchmark.json/log` kaynak kanıtlarıdır.
Son ctypes-only cache fix'i CUDA timing/allocator path'ini değiştirmedi;
Benchmark dependency closure byte-identical'dır. Final wheel üzerinde ayrıca
P01/P02 **15 PASS installed tekrar**, `installed-cuda-performance-final2.json`:
current25-source byte equality ve target import provenance doğrulandı. Aynı15
ID'yi distinct toplama yeniden eklemiyoruz. Tüm eski162 script'i tekrar koşulmadı.
İki mevcut terminal fault senaryosu gerçek CUDA workload üstünde injection'dır,
gerçek cihaz arızası recovery kanıtı değildir.

## 11. Experiment Manager ve çalışır rehber

[performance-memory.md](../guides/performance-memory.md) tüm Python blokları source,
wheel/sdist installed ve minimum Torch smoke'unda execute edildi. Mevcut Manager
schema2 copy artifacts ve generic observation metrics kullanılır: JSON result'ları
ve model_state_dict artifact → SHA/size verify → üç bounded scalar → finish →
close/reopen/query/verify. Unsupported/null metric log edilmez. Raw samples metric
event başına çoğaltılmaz. Schema1 default store ve explicit offline schema2 migration
korunur; yeni table/migration/run kind/tracking adapter yoktur.

Measurement penceresinde file/SQLite/network write ve hashing yasağı testlidir;
native OS sampling explicit instrumentation'dır. Artifact/model/result/run kimlikleri
ayrıdır, ownership='copy' E'ye ilişkilendirmede seçilir; measurement artifact'ı
default caller-owned kalır. Existing E259 regression değişmeden PASS.

## 12. Minimum Torch2.2.2 ve package

Mevcut clean Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Final wheel local/no-index/no-deps/no-compile target **E/installed-final2**;
repo dışı cwd=E, `-I -B`, installed import provenance.

**PASS:** P10 current native RSS + sampled workload; P11 logical footprint,
gerçek state_dict stat/SHA/weights_only round-trip; typed JSON/CSV; actual E
association/reopen; P01/P02 ve legacy imports/checkpoint/replay smoke. CUDA initialized
false; Torch2.2.2 native CUDA iddiası yoktur. Minimum prefix dependency distribution
METADATA/RECORD hash'leri önce/sonra eşittir (`minimum-environment.json`); bütün
binary dosyaların full-content hash audit'i yapıldığı ileri sürülmez.

Explicit packaging inventory24→**25 production modules**; guard kaldırılmadı.
Final wheel/sdist source-byte equality, complete inventory, installed imports,
legacy/new exports, unchanged version/dependency bounds ve tests/docs/cache/DB/
weights exclusion PASS. Final artifact'lar E/pytest-final2/packaging0/dist:

| Artifact | Byte | SHA256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |107083|`f1ed96c30a240ec33b91709f85875861706ea113a7035428949d67404e8c682f`|
| hyso-0.2.0.tar.gz |101915|`c93407955fba6d54aeb3b464fe41d6bcc81977fd18d5ef1d3e876b6488bc5615`|

İlk build cache fix'ten önceydi; production değiştiği için final ikinci build
gerekti. Final source'ta gereksiz üçüncü build yapılmadı. Source inventory/SHA ve
installed25-byte equality `installed-integrity.json`, commands/stdout/stderr ve
minimum-smoke JSON E'dedir.

Final installed harness'in ilk Windows repeat'i önceki fixture output adını
yeniden kullandı; no-clobber **FileExistsError ile doğru reddetti**. Bu başarısız
harness kaydı `installed-commands.json` / installed-windows stderr'de korundu.
Üretim kodu veya overwrite policy değiştirilmeden fresh `installed-windows-final2`
output yollarıyla üç native case PASS oldu. Sonra final installed CUDA15 PASS.
Bu çevresel fixture çakışması test gevşetilerek veya failure gizlenerek kapatılmadı.

## 13. Backlog ve kapsam kararı

P10 **implemented**, yalnız native-verified Windows current-process RSS/HWM ve
açık sampled-peak limitleriyle. Linux adapter/native verification ayrı tutulur.
P11 **implemented**, dense CPU logical convention + explicit state_dict veya
opaque caller artifact byte/SHA kapsamıyla; physical memory unsupported'dır.
Güncel toplam **5 implemented, 2 partial, 28 planned**. P01/P02/P09 ve diğer bütün
satırların statüsü değişmedi. M2-P-A/sabit roadmap geçmiş durumları korunur.

## 14. Son karar, risk ve M2-P-B3

**M2-P-B2 PASS — doğrulanan Windows/CPU kapsamı için.** 83 yeni dahil665 targeted,
Windows native3, CUDA protection15+45, minimum installed ve25-module package
kapıları geçti; final installed Windows3/CUDA15 tekrarları ayrı kanıttır. Eksik Linux
native test başarıya çevrilmedi. Sonuç P10 model-only absolute RAM peak veya P11
physical storage ölçümü değildir.

Riskler: sample aliasing/OS scheduling, process-wide allocations/GC/cache etkisi,
uncontrolled concurrent caller, lifetime HWM'nin pre-window geçmişi, arbitrary
workload/custom Python side effects, concurrent/hostile file mutation. RSS query
overhead ve file consistency checks sınırlıdır; arbitrary model/filesystem
transaction veya worker hard timeout verilmez. Unsupported dense/format/backend
profilleri explicit hata/null status alır. Yeni conversion/training/other-P işi yoktur.

`E/initial-tracked-hashes.json` ile final karşılaştırmada yalnız izin verilen dört
eski dosya değişti: performance namespace/result, packaging test, backlog. Diğer
bütün eski dosyalar ve kullanıcı farkı korunur. Index boş, branch/HEAD aynı;
`git diff --check` başarılı, commit/push yok. `E/verification-summary.json` nihai
identity/count/source/archive/ownership/freeze envanterini taşır.

**M2-P-B3'e geçilebilir:** P03/P04 pipeline-stage timing ve P05 fresh-process cold
start için P01/P02 çekirdeği, actual artifact size/hash ve açık process ownership
sınırları hazırdır. B3 kendi importable worker factory, clock-domain/queue/phase
ve native acceptance kapılarını uygulamalıdır; bu görevde B3 kodu yazılmadı.
