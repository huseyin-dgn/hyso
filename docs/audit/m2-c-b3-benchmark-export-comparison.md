# M2-C-B3 — Benchmark Result Export & Fair Model Comparison

Tarih: 2026-10-19. **Karar: PASS — M2-C-B3.**
**140 yeni CPU testi; 1691 farklı hedefli CPU testcase PASS.** Gerçek GPU'da
B2'nin45 ve B3'ün8 senaryosu: **53 farklı CUDA testcase PASS**. Son installed
wheel'de B3'ün aynı8 senaryosu da geçti; toplamda tekrar sayılmadı.
Torch2.2.2 CPU installed smoke: iki precision, altı export ve üç comparison
kontrolü PASS. Son wheel/sdist **19 production modülüyle PASS**.
Bu hedefli kapıdır; full-suite veya M2-C-C final kalite kapısı değildir.

## 1. Branch / HEAD / working tree

Başlangıç/son branch `feature/hyso-models-next`, HEAD
`68b523c38fad5102f36343db0b701101df5f6ca6`,
`feat: add native CUDA inference benchmarking`. B2 commit edilmişti.
Başlangıçta yalnız `tests/test_gradient_accumulation.py` kullanıcı farkı vardı.
Dosyaya yazılmadı/stage edilmedi; başlangıç/son SHA-256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Commit/push/checkout/reset/branch/version/dependency değişikliği yapılmadı.
Index boş; `git diff --check` başarılı.

Kanıt kökü E=`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2cb3`.
`summarize.py` / `verification-summary.json`: JUnit tekil kimlikleri, eski1551
kimliğin altküme kontrolü, kaynak/metadata freeze, schema AST eşliği, modül
envanteri, final artifact bytes/hash, native ve installed/minimum kontrolleri.

## 2. B3 kapsamı

Mevcut result.py içinde explicit JSON, summary CSV, long-format sample CSV ve
pure compatibility report eklendi. Yeni production modülü yoktur; runner.py
**HEAD ile byte/blob eşliğinde**, CPU timing, CUDA event/stream, RNG/state ve
opt-in memory protokolü değişmedi. Yeni timing engine, automatic export,
leaderboard/ranking, speedup/significance, accuracy, regression alarm, profiler,
sweep veya training sistemi yoktur.

## 3. Public export/comparison API

```python
from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results

result = BenchmarkRunner(
    model, device="cpu", input_shape=(1, 3, 224, 224), precision="fp32",
    warmup_iterations=10, measure_iterations=50, repeats=3,
).run()
result.to_json("benchmark-result.json")
result.to_csv("benchmark-summary.csv")
result.to_samples_csv("benchmark-samples.csv")
report = compare_results(result, another_result)
```

Üç export method'u `(path, *, overwrite=False)` alır, None döndürür. Explicit
`overwrite=True` tamamlanmış dosyanın atomic replacement'ına izin verir.
Long format için tek isim **to_samples_csv**; alias/loader API eklenmedi.
Public namespace yalnız Runner/Result/compare_results. Root/model exports aynı.
Comparison yalnız exact BenchmarkResult nesnelerini kabul eder; dict/model/Tensor/
dosya yolu reddedilir. Sonuç plain mutable defensive report'tur; input'ları
değiştiremez. Run, forward, RNG, CUDA initialization/event/allocator veya I/O
comparison sırasında çalışmaz.

## 4. JSON schema integrity

UTF-8, ensure_ascii=False, allow_nan=False, recursively sorted keys, indent2 ve
son LF; aynı result tekrar export edildiğinde byte sırası deterministiktir.
JSON anlamsal olarak to_dict ile kayıpsız eşittir: schema_version=1, identity,
timestamp/provenance/environment, R×N nested samples, unsupported null/status/
reason, CUDA memory ve irreversible-reset warning'leri korunur.

Constructor'ın `_plain`, `_statistics`, `_throughput`, `_validate`, __init__ ve
to_dict gövdeleri HEAD AST'leriyle aynıdır. Frozen/slots/defensive yapı korunur.
Export boundary ayrıca plain/schema integrity kontrol eder; nonfinite/object/
Tensor reddedilir. Full JSON model weights, input array veya optimizer
checkpoint değildir; yalnız mevcut result metadata/schema/count/hash/sample
verisini taşır. Pickle/model/runtime referansı yoktur; export yeniden ölçmez.

## 5. CSV summary format

Standart csv.DictWriter, UTF-8, sabit48 kolon, header + tek run satırı; LF.
JSON full canonical kayıttır; summary nested metadata'nın yerine geçmez.
Kolon sırası aşağıdaki grupların birleşimidir:

| Sıra | Kolonlar |
|---|---|
| 1–8 | schema_version, run_id, started_at_utc, finished_at_utc, model_name, model_class, device, precision |
| 9–17 | input_shape, input_dtype, batch_size, warmup_iterations, measure_iterations, repeats, timing_method, timing_scope, synchronization_policy |
| 18–26 | sample_count, mean_ms, p50_ms, p90_ms, p95_ms, min_ms, max_ms, std_ms, std_status |
| 27–33 | throughput_images_per_second, throughput_method, throughput_source_timing_method, parameters_total, parameters_trainable, parameters_frozen, parameter_buffer_bytes |
| 34–37 | cuda_memory_status, cuda_memory_reason, cuda_memory_scope, cuda_counter_reset_performed |
| 38–48 | cuda_baseline_allocated_bytes, cuda_baseline_reserved_bytes, cuda_previous_peak_allocated_bytes, cuda_previous_peak_reserved_bytes, cuda_peak_allocated_bytes, cuda_peak_reserved_bytes, cuda_incremental_peak_bytes, cuda_current_allocated_after_probe_bytes, cuda_current_reserved_after_probe_bytes, cuda_current_allocated_after_output_release_bytes, cuda_current_reserved_after_output_release_bytes |

input_shape compact JSON list metnidir. None boş cell; CPU not_applicable ve
CUDA disabled status/reason ile ayrılır, bilinmeyen memory sahte0 değildir.
Bool True/False; float Python round-trip metniyle yazılır. Tek sample std boş
ve std_status=insufficient_samples kalır. Comma/quote/newline standart CSV
escaping ile korunur. İlk whitespace sonrası =,+,-,@ başlayan **text** cell'e
spreadsheet formula yorumunu önlemek için apostrophe prefix eklenir; JSON
orijinal metni kayıpsız korur. Sayısal değerler text sanitizer'dan etkilenmez.

## 6. Long-format samples

Sabit10 kolon: schema_version, run_id, repeat_index, iteration_index,
latency_ms, timing_method, timing_scope, device, precision, batch_size.
İndeksler **zero-based**, önce repeat sonra iteration sırası. Tam R*N veri
satırı; preflight/warmup/probe eklenmez. Float parse edildiğinde orijinal sample
ile exact eşlik ve sıra/indeks/satır sayısı doğrulandı. Summary'de raw samples
sessizce kaybolmaz; ayrı açık API ve full JSON vardır.

## 7. File I/O error policy

String/PathLike, nonempty destination ve bool overwrite doğrulanır. Parent
önceden var olmalı; otomatik mkdir veya append yoktur. Directory/special-file
ve symlink destination reddedilir; type/value/OSError'lar caller'a yayılır.
Existing regular file varsayılan FileExistsError ile korunur.

Önce tamamlanmış UTF-8 içerik hazırlanır; aynı parent içinde sibling temporary
file yazılır, flush+fsync ve close sonrası yayınlanır. Default no-clobber
publication `os.link(temp,destination)` kullanır; yarış sırasında yeni oluşan
destination da ezilmez. overwrite=True `os.replace` kullanır. Finally temp
temizlenir. Framework/fallback veya dependency eklenmedi; filesystem hard-link
ve atomic replace desteği gerekir, unsupported I/O açık hata verir.

Injected create/partial-write/fsync/link/replace PermissionError testleri,
missing parent/invalid path, symlink validation ve concurrent no-clobber race
testleri geçti. Publish öncesi hata caller dosyasını korur; result değişmez.
Directory crash durability/ACL veya eski inode metadata korunumu garantisi yoktur.
Publish sonrası cleanup hatası tam yazılmış bir destination bırakabilir; bu
arbitrary filesystem transaction veya başka writer'lara karşı global lock değildir.

## 8. Comparison compatibility key

Report `comparison_schema_version=1`; BenchmarkResult schema'sından ayrıdır.
Top status timing conditions eligibility'sidir. `mismatched_fields`,
`missing_required_fields`, `compared_fields` path/left/right veya sides/reason
taşır. Sıra fixed key listesidir. Missing/unknown/invalid required metadata
insufficient_metadata; **herhangi kesin mismatch öncelikle incompatible** olur,
missing listesi de korunur. False ve warmup0 geçerli değerlerdir; boş/unknown
hardware, invalid type/hash eşdeğer koşul kabul edilmez.

Kontrol edilen anahtarlar:

- schema/type/protocol; precision/autocast; W/N/R/preflight; forced eval,
  inference_mode ve fixed resident input policy.
- Timing method/scope/sync, throughput method/source, quantile method/ddof.
- Input shape, batch, dtype, layout, stride ve lowercase SHA-256 content hash.
- Full resolved device (type/index dahil), Python/Torch/HYSo version, OS/platform,
  CPU machine/processor/logical count ve intra/inter-op threads.
- Deterministic algorithms/warn-only ve float32 matmul precision. CPU'da mkldnn;
  CUDA'da GPU name/capability/total-device-memory, CUDA build/cuDNN version,
  cuDNN benchmark/deterministic/TF32, matmul TF32 ve allocator backend.

CPU wall/CUDA event, farklı batch/resolution/precision/runtime/backend koşulları
compatible olamaz. Device index equality tek başına hardware kanıtı değildir;
hardware/host profili ayrıca karşılaştırılır. UUID/driver/telemetry/clock/power
verisi çıkarılmaz; recorded-profile identity sınırı warning'de açıkça yer alır.

## 9. Compatible/incompatible/insufficient_metadata örnekleri

| Durum | Top status | Ayrıntı |
|---|---|---|
| Aynı kayıtlı CPU/CUDA koşulları; farklı model/weights | compatible | Model bir comparison değişkenidir |
| CPU perf_counter vs CUDA event | incompatible | Device, method, scope, sync ve rate source mismatch |
| CUDA FP32 vs amp_fp16; farklı batch/shape/W/N/R/Torch | incompatible | Alan bazında farklar listelenir |
| CPU processor veya GPU name/capability bilinmiyor | insufficient_metadata | Missing required fields ve taraflar |
| Processor bilinmiyor + warmup farklı | incompatible | Mismatch öncelikli; missing listesi de var |
| Source hash yok, temel koşullar tam | compatible | Informational provenance warning |

Report numerical speedup, p-value, confidence interval veya quality winner
üretmez. Matching recorded conditions significance/equal clocks/load/cache/
thermals/power garantisi değildir; bu sınır her raporda warning'dir.

## 10. CUDA memory comparison policy

`memory_comparability` top timing status'ından ayrıdır; observations iki
result'ın bütün memory alanlarını korur. İki CPU için not_applicable. CUDA'da
disabled/measurement yok veya CPU↔CUDA ise not_comparable; bir taraf measured
diğer taraf disabled olduğunda **timing status otomatik bozulmaz**.

İki measured probe'da allocator backend/scope/reset policy, timing/hardware/
runtime eligibility ve process isolation ayrı değerlendirilir. B1/B2 isolation
attestation toplamaz; normal native B2 results memory=insufficient_metadata
üretir, latency compatible kalabilir. Eksik bilgi uydurulmaz.

İsteğe bağlı mevcut extensible provenance alanında
`memory_isolation="exclusive_fresh_process"` verilirse yalnız **caller assertion**
olarak değerlendirilir; engine bunu doğrulamış sayılmaz. Başka/nonfresh label
insufficient veya explicit mismatch sonucu verir. Baseline allocated/reserved
farkları memory_baseline_differs warning'leri üretir; gözlemlenen peak ve
incremental alanları report'ta kalır. Absolute peak process/device allocator
ölçümüdür, model-only VRAM değildir: `absolute_peak_model_comparable=False`
her zaman korunur. CPU RSS ile GPU allocated kıyaslanmaz; peak ranking yoktur.
Geri alınamayan reset yan etkisi JSON/CSV/report observations içinde korunur.

## 11. Model identity vs benchmark conditions

Model name/class/config, output descriptor, parameter count ve weights hash
informational değişkenlerdir. Farklı nn.Identity ve gerçek Conv2d/Conv-BN
modelleri aynı input/runtime/protocol ile compatible çıktı. İsim veya weights
farkı tek başına mismatch değildir. Source hash/revision/path, input source/seed
aynı olmak zorunda değildir; input equality için content hash/shape/dtype/policy
esas alınır. Optional source hash eksikliği warning olur.

Aynı model regression'ı için config/weights/source eşliği, task/output semantics
ve quality ayrıca tasarlanmalıdır. Bu API bu analizi veya accuracy equivalence
iddiasını yapmaz. Report'taki informational veriler defensive plain snapshot'tır;
izin verilen derin model_config annotation'ı da kayıpsız taşınır.

## 12. Yeni CPU testleri

`tests/test_benchmark_export.py`: **66 PASS**. Gerçek CPU inference snapshots;
lossless/sorted UTF-8 JSON, schema/null/reset preservation, fixed summary/long
headers, exact float/order/index, formula/CSV escaping, overwrite/no-clobber,
injected permission/partial-write/publish errors, invalid metadata boundary,
immutability ve forward/CUDA/RNG side-effect yasağı.

`tests/test_benchmark_comparison.py`: **74 PASS**. Aynı/farklı CPU/GPU-shaped
profile, architecture/weights değişkenliği, her runtime/backend/hardware/input
farkı, unknown/invalid required fields, mismatch precedence, memory independence/
baseline/isolation, deep metadata ve defensive report/type checks.

CPU testlerindeki CUDA-shaped veriler **serialization/policy fixture'larıdır**;
CPU sample'a CUDA etiketi verilmesi native timing kanıtı diye raporlanmaz.
Gerçek CUDA result export/comparison aşağıdaki ayrı native scriptte doğrulandı.
Yeni140 test dahil `benchmark.xml` toplam306 PASS.

## 13. Gerçek CUDA doğrulaması

Python3.10.18 / Torch2.5.1+cu121 / CUDA build12.1;
NVIDIA GeForce RTX3060 Laptop GPU, 6144MiB, driver591.74.
Mevcut `cuda_benchmark_validation.py` değiştirilmedi: **45/45 PASS**.
Yeni `cuda_benchmark_export_validation.py`: **8/8 PASS**:
native FP32 measured export, AMP measured export, disabled export,
farklı native architecture compatibility, precision/batch mismatch, separate
memory eligibility ve serialization sırasında forward/runtime/RNG yasağı.

Full JSON eşliği, summary/sample CSV float/count/sıra ve reset diagnostics gerçek
GPU result'larında doğrulandı. Yeni script CUDA kapalıyken setup error+exit1
verdi; CPU fallback/skip yoktur. Bu beklenen negatif kontrol toplamı artırmaz.
Son wheel'in isolated target'ından aynı8 test tekrar geçti; source provenance
target içinde. **45+8=53 farklı senaryo**, tekrar8 toplam sayılmadı.

E/native-b2.json/log, native-b3.json/log, installed-native-b3.json/log ve
unavailable-b3.json/log kanıtlardır. B3 export klasörleri unique temp adlarıyla
rapordaki artifacts listesinde bulunur. Performans kazancı/latency eşiği iddiası
yoktur; bu gerçek GPU correctness smoke'udur.

## 14. B1/B2 CPU regression

CPU85 + Result24 + State24 + CUDA protocol33 = **166 PASS**; bütün eski testcase
kimlikleri korunur. CPU/state/protocol dosyaları Git blob eşliğinde aynı.
Result testindeki artık geçersiz “to_json yok” assertion'ı, istenen yeni üç
export API'sinin callable olduğunu doğrulayan assertion ile güncellendi;
aynı testin immutability/defensive/plain/no-model kontrolleri korunur.
Diğer assertion'lar silinmedi veya tolerans gevşetilmedi.

Runner ve constructor integrity gövdeleri unchanged; CPU FP32/FP64, native CUDA
event/memory/state işleyişi yeniden tasarlanmadı. Eski B2'nin1551 kimliği yeni
1691 toplamının altkümesidir;140 yeni kimlik tam olarak yeni iki dosyadadır.

## 15. Trainer/legacy/M1 regression

CUDA_VISIBLE_DEVICES=-1, MPLBACKEND=Agg, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
tests/pytest.ini ve tek Torch CPU thread ile gerçek hedefli pytest/JUnit:

| Grup | Unique PASS |
|---|---:|
| Sekiz Generic Trainer classification/update/validation/lifecycle/save/restore/transaction/replay dosyası |706|
| P0 checkpoint/EMA/scheduler/OneCycle |141|
| M1 blocks/attention/ModularCNN |356|
| Model state/models/public/published API |66|
| EBO resume/gradient accumulation/CV-dataset resume |112|
| B1/B2 + yeni export/comparison benchmark |306|
| Final packaging |4|
| **Toplam** |**1691**|

regression.xml1381, benchmark.xml306, packaging-final.xml4; (classname,name)
kimlikleri tekil, ara test/build tekrarları eklenmez. Failure/error/skip/xfail0.
Mevcut legacy deprecation/undefined-metric warning'ları değiştirilmedi.
Legacy16 production dosyası, runner dahil17 production dosyası, root/model
exports, checkpoint FORMAT_VERSION1, metadata ve sabit v1.0 HEAD ile aynı.
Bu full CPU suite veya bütün eski CUDA script'lerinin final gate'i değildir.

## 16. Torch 2.2.2 installed-wheel

Mevcut temiz Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu ortamı kullanıldı.
Son local wheel `pip --isolated --no-index --no-deps --no-compile --target`
ile E/installed-minimum-final'a kuruldu; repo dışından -I -B smoke çalıştı.
Runner/Result/compare_results provenance installed target içinde.

Gerçek **CPU FP32/FP64 iki profil PASS**; her biri JSON+summary+samples export:
**6 dosya**, iki compatible ve bir fp32↔fp64 incompatible kontrolü PASS.
State/RNG/sample/statistics/float round-trip ve legacy Trainer/replay smoke
assertion'ları da korundu. CUDA initialized=false; CPU installed sonucu native
CUDA2.2.2 kanıtı değildir. Dependency kurulmadı/değiştirilmedi; yalnız istenen
yerel HYSo artifact'ı isolated target'a kuruldu.

E/minimum-gate.py, installed-minimum.json, minimum stdout/stderr ve
clean-before/after.json:35 distribution sürüm/METADATA/eski HYSo source hash'i
önce/sonra eşit. Minimum sonuçları pytest1691 veya CUDA53 toplamına eklenmedi.

## 17. Wheel/sdist

Son kaynaklardan iki archive build edildi; gerçek inventory **19 modül**.
Wheel ve sdist'te bütün19 için source-byte equality, metadata, docs/tests/cache/
weights exclusion ve installed imports/smoke geçti. İki arşiv de checkout dışında
kurularak Runner/Result/new exports/compare_results ve legacy exports doğrulandı.
Final wheel'de B3 native GPU8/8 ayrıca geçti.

| Final artifact | Byte | SHA-256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl |73518|359d17324a665a94e006fa1f6cbdad75e396f3f70543288a2046cf613138b0fb|
| hyso-0.2.0.tar.gz |71073|d3ad22cb52a5067a1307f268ba489fefccdc4c7b6a5aa31202fb42cca57067a9|

Final konum `E/pytest-final-verified/packaging0/dist/`; build.log,
wheel-smoke.json, sdist-smoke.json ve installed target'lar aynı packaging0 altında.
E/pytest-final önceki ara build'dir. HYSo0.2.0, Python>=3.10, Torch>=2.2.2,
torchvision>=0.17.2 ve sekiz dependency değişmedi. Upload/release yapılmadı.

## 18. 35-feature backlog status

P01–P35 tam/tekil; yalnız **P29** satırı B3 kapsamı için partial oldu.
Diğer34 satır HEAD ile aynı. Durum: **1 implemented, 4 partial, 30 planned**.
P09 native allocator implemented; P01/P08/P11 partial kalır. P29 yalnız
export/compatibility temelidir; accuracy–speed–memory Pareto, quality evaluation,
statistical regression ve alarm sistemi uygulanmış sayılmaz. P29 implemented
değildir; sabit roadmap v1.0 değişmedi.

## 19. Değiştirilen dosyalar

- `hyso/benchmark/result.py`: explicit safe exports ve pure comparison.
- `hyso/benchmark/__init__.py`: compare_results public export.
- `tests/test_benchmark_export.py`:66 yeni test.
- `tests/test_benchmark_comparison.py`:74 yeni test.
- `tests/cuda_benchmark_export_validation.py`:8 yeni native senaryo.
- `tests/test_benchmark_result.py`: tek obsolete export-yok assertion'ının yeni API sözleşmesine uyarlanması.
- `tests/test_packaging.py`: additive installed export/comparison assertions.
- `docs/roadmap/hyso-performance-backlog-v1.1.md`: P29 partial ve B3 kanıtı.
- `docs/audit/m2-c-b3-benchmark-export-comparison.md`: bu rapor.

Git'teki gradient_accumulation farkı önceden mevcut kullanıcı değişikliğidir;
bu görev dosyayı değiştirmedi. Runner, legacy, schema validator ve dependency
freeze kanıtları E/verification-summary.json içinde bulunur.

## 20. Bilinen sınırlamalar

Compatibility recorded conditions eligibility'sidir, fiziksel host/GPU UUID,
driver/power/clock/thermal/load veya istatistiksel significance kanıtı değildir.
OS/hardware profili eşliği tek başına aynı fiziksel unit garantisi vermez;
report açık warning üretir. Bilinmeyen required hardware ise insufficient olur.
Device index'e karşı strict policy farklı ordinal mapping'i de incompatible
sayabilir. Source hash/driver gibi toplanmamış metadata uydurulmaz.

Memory isolation default bilinmez; fresh-process label caller attestation'dır.
Cached/baseline farklılıkları ve allocator dışı VRAM bu API ile model attribution'a
çevrilemez. Summary tam JSON değildir; null/empty ayrımı status/reason ve JSON'da
korunur. Formula text prefix CSV display'ini etkiler, JSON orijinali saklar.
Atomic publish filesystem desteğine bağlıdır; directory durability/ACL/global
writer isolation garantisi yoktur. New schema loader, append/multi-run CSV,
ranking/speedup/quality/regression/profiler/sweep yoktur.

## 21. PASS / PARTIAL / FAIL

**PASS — M2-C-B3.** JSON/CSV/sample export, schema1/immutability, safe explicit
I/O, fieldwise compatibility ve separate memory policy;140 yeni CPU testi,
1691 unique hedefli CPU testcase,53 unique native GPU testcase, minimum CPU
installed export/comparison ve final19-module wheel/sdist kapıları tamamlandı.
Backlog partial durumları B3'ten geniş ileri kapsamı gösterir. Commit/push/
branch/version/dependency/download/release değişikliği yoktur.

## 22. M2-C-C final kalite kapısı önerisi

**M2-C-C'ye geçilebilir.** Yeni full CPU suite, bütün gerekli eski native CUDA
script'leri ve yeni benchmark/export native smoke, minimum installed runtime,
son wheel/sdist/source equality ve legacy/state/metadata freeze birlikte tekrar
denetlenmeli. Unique identity sayımı, synthetic protocol vs native kanıt ayrımı,
hardware unknown ve memory reset/isolation sınırları korunmalı. Bu görevde
full-suite veya M2-C-C FINAL GO ilan edilmedi.
