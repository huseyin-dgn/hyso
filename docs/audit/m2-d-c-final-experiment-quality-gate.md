# M2-D-C — Final Experiment Manager Quality Gate

Tarih: 2026-10-20 (Europe/Istanbul). **Karar: PARTIAL GO.** CPU, B1/B2/B3,
minimum-runtime ve packaging kontrolleri başarılı. İlk native export sürecindeki
bir allocator hatasının kök nedeni kanıtlanamadığından **FINAL GO verilmez**.
Sonraki başarılı tekrarlar ilk hatayı silmez. Bu karar release onayı değildir.

## 1. Kaynak ve kanıt

Branch `feature/hyso-models-next`; başlangıç/son HEAD
`ee7a8ca365e399ccd54262063ca7bf1602b6f134` (B3 integration commit).
B1 `9db424d`, B2 `def6108`; başlangıç index boş. Başlangıçta yalnız
`tests/test_gradient_accumulation.py` kullanıcı farkı vardı.
144 başlangıç tracked dosyanın byte hash'i korundu; index boş kaldı.
Production/test/dependency/metadata/roadmap/backlog değişikliği yapılmadı.
Commit/push/branch değişimi, dataset/weights indirme veya dependency kurulumu yok.

Bütün yeni log/JUnit/JSON/helper/build/install kanıtları tek repo dışı kökte:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2dc-20261020-013213` (**E**). Önceki audit sonuçları yalnız identity/source baseline olarak
kullanıldı. D-A, D-B1/B2/B3 ve C-C audit'leri; ilgili production/test dosyaları,
manuel rehber ve packaging yapılandırması incelendi.

## 2. Tek full CPU koşusu

Repository'deki mevcut Python3.10.18 test interpreter'ı:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe`.
Torch2.5.1+cu121; CUDA build12.1 fakat bu süreçte visibility=-1,
`cuda_initialized=False`. `tests/pytest.ini` ve mevcut fixtures kullanıldı.

```text
<test-python> -B -m pytest -p gate_capture -c tests/pytest.ini tests -q
  --junitxml E/full-cpu-junit.xml --basetemp E/pytest-full
CUDA_VISIBLE_DEVICES=-1 MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=E PYTHONDONTWRITEBYTECODE=1
```

**42 dosya,1992 collected:1991 PASS,0 FAIL,0 ERROR,1 SKIP,0 XFAIL,0 XPASS;
exit0.** Pytest278.10s; process wall300.906s. Hook süresi
296.734s collection dahil farklı bir ölçümdür.
1043 warning:885 FutureWarning,156 UndefinedMetricWarning,2 DeprecationWarning;
mesaj/nodeid/category ayrıntıları `E/full-cpu-results.json` ve stdout'tadır.
Tek SKIP: `test_device_precision.py::test_r12_r13_real_cuda_device_smoke`,
`R12/R13: CUDA hidden/unavailable in the CPU-first test environment`.

Collection, phase inventory ve JUnit classname/name kümeleri1992 identity ile
birebir eşleştirildi. `full-cpu-command.json` argv/cwd/env/exit/süre;
`cpu-collection.json`, `cpu-inventory.json`, `full-cpu-junit.xml` ve
`full-cpu.stdout.log`/`stderr.log` asıl kanıttır. Full suite tekrar edilmedi.

## 3. Baseline ve B1/B2/B3

`E/baseline-comparison.json` eski kaynak path/SHA ve exact identity karşılaştırmasını
saklar. B5'in1427 identity'si (1426 PASS/1 SKIP) ve C-C'nin1733 identity'si
(1732 PASS/1 SKIP) korunur: **kayıp0,status değişimi0**. C-C'ye göre ek259 PASS.

| Experiment grubu | Dosya dağılımı | Exact eski identity / yeni PASS |
|---|---|---:|
| B1 | core60,metrics59,storage28 |147/147|
| B2 | artifacts36,migration11,recovery17,retry19 |83/83|
| B3 | integration29 |29/29|

B1 final-core-packaging.xml, B2 targeted.xml ve B3 targeted.xml erişilebilir
baseline'larıyla yalnız ilgili grupların exact JUnit kimlikleri karşılaştırıldı.
259 Experiment testi full1991 toplamının içindedir; hedefli tekrarlar eklenmez.

## 4. SQLite/migration/artifact/recovery

B1 UUID/FK/schema1, canonical UTF8/SHA256/redaction, lifecycle/owner/idempotency,
epoch/step/unit/source, atomic batch/duplicate/out-of-order, pagination,
transaction rollback, concurrent create ve corruption/unknown-schema testleri PASS.
B2 explicit migration öncesi SQLite backup ve tüm core row/hash korunumu;
DDL rollback/process-kill/ACK failure, reference/copy/streaming SHA,
immutable expected identity, no-clobber staging/publication ve link/state/check
observation/reconciliation kontrolleri PASS. Windows junction/path/permission/
fsync/link negatifleri ile writer recovery/fencing ve retry lineage PASS.

Yeni store schema1 kalır; normal open migration/recovery/completion yapmaz.
B2 explicit schema2 migration ister. Recovery running'i interrupted yapar,
completed yapmaz; workload'u durdurmaz. External artifact silinmez.
Checkpoint Manager için byte identity'dir; deserialize edilmez.

## 5. Trainer/resume ve Benchmark entegrasyonu

Gerçek CPU FP32/FP64 Trainer, history→metric eşliği, percent accuracy,
validation None omission,256-event atomic batches, gerçek FORMAT_VERSION1
checkpoint reference/copy ve persistence PASS. Gerçek strict retry/resume,
restored prefix'in parent'a ve yalnız yeni epoch'ların retry'a attribution'ı;
terminal/no-op resume ve workload/tracking/artifact failure ayrımı PASS.

Gerçek BenchmarkRunner Result→JSON/summary CSV/samples CSV→persistent artifacts;
benchmark UUID/manager UUID ayrımı, comparison üç status ve input links PASS.
Rehber fenced blokları testlerde compile/execute edildi; tam rehber akışı geçti.
Tracking sırasında forward/optimizer step/zero_grad/scheduler step/torch.load,
CUDA synchronize/Event/reset yasaklayan guards ve model/gradient/buffer/mode,
optimizer/scheduler/EMA,Python/NumPy/Torch RNG assertion'ları PASS.
Manager otomatik Trainer/Benchmark çalıştırmaz; timing/probe içine I/O eklenmedi.

Full CPU legacy regresyonu: Generic Trainer706, Benchmark306, M1 primitives/
attention/ModularCNN356, checkpoint34,OneCycle20,gradient accumulation28,
scheduler69,EMA18,EBO resume45 PASS; diğer eski testler de inventory'de PASS.
33 CUDA protocol testi CPU wiring testidir, native sayılmaz.

## 6. Gerçek native CUDA gate

Keşfedilen10 repository script'i ve byte-exact dış B4-A save helper'ı,
her biri yeni ayrı process'te sırayla çalıştı. CUDA_VISIBLE_DEVICES=0,
MPLBACKEND=Agg,CUBLAS_WORKSPACE_CONFIG=:4096:8; gerçek availability assertion.
Runtime Python3.10.18,Torch2.5.1+cu121,torchvision0.20.1+cu121,CUDA12.1,
RTX3060 Laptop GPU,capability8.6,driver591.74,API total6441926656 byte.
`E/gpu-runtime.json`, script path/SHA inventory, `cuda-runs.json` ve her script'in
stdout/stderr/result JSON'u runtime/argv/cwd/env/exit/senaryo kanıtıdır.

| İlk process script'i | PASS | ERROR | Exit | Süre s |
|---|---:|---:|---:|---:|
| `cuda_runtime_validation.py` | 7 | 0 | 0 | 29.625 |
| `cuda_imagefolder_validation.py` | 10 | 0 | 0 | 21.718 |
| `cuda_modular_validation.py` | 18 | 0 | 0 | 20.125 |
| `cuda_trainer_updates_validation.py` | 40 | 0 | 0 | 11.625 |
| `cuda_trainer_validation_lifecycle.py` | 16 | 0 | 0 | 10.813 |
| `cuda-save.py` | 3 | 0 | 0 | 10.891 |
| `cuda_trainer_checkpoint_restore.py` | 5 | 0 | 0 | 14.453 |
| `cuda_trainer_replay_conformance.py` | 10 | 0 | 0 | 21.937 |
| `cuda_benchmark_validation.py` | 45 | 0 | 0 | 12.766 |
| `cuda_benchmark_export_validation.py` | 7 | 1 | 1 | 10.875 |
| `cuda_experiment_integration_validation.py` | 2 | 0 | 0 | 11.516 |

B4-A original helper path `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2b4a\cuda-save.py`;
SHA256 `420690653ecf8bf4526e8575bff0ec6c5576c4c5b4a35e16c0e334b41924844b`
doğrulanıp E'ye byte-exact kopyalandı ve yeni process'te çalıştırıldı.

**İlk primary gate:164 execution=163 PASS+1 ERROR;162 farklı identity=
161 PASS+1 ERROR.** Replay'nin restore'dan import ettiği aynı rollback-False/True
iki execution'ı distinct toplama tekrar eklenmedi. Eski160 ve B3 fp32/amp_fp16
iki identity'nin tamamı mevcut. `E/cuda-baseline-comparison.json` exact kimlik ve
outcome reconciliation'ının authority dosyasıdır. Ham `cuda-unique-summary.json`
başarısız script'in7 başarılı case'ini saymadığından154 gösterir; bu eksik ham
sayı final distinct sonucu olarak kullanılmadı.

## 7. Açık blocker ve sınırlı tanılama

İlk `cuda_benchmark_export_validation.py` sürecinde7 PASS/1 ERROR:
`NativeExportComparison.test_readonly_exports_no_runtime_calls_or_rng_change`.
Hata export guard başlamadan `run(memory=True)` içinde, Runner→Result validation
sınırında `result.py:145 ValueError: invalid CUDA allocator bytes` olarak oluştu.
Bu assertion measured memory alanlarından en az birinin exact nonnegative int
olmamasını reddeder. İlk hata payload'ı hatalı memory değerlerini içermediği için
hangi alanın hangi değerle bozulduğu ve kök neden **kanıtlanamadı**.

İlgili tek senaryo yeni process'te memory-probe sonucunu kaydeden read-only
wrapper ile **1 PASS** (`export-failed-recheck.json/.log`). Önceki sıra etkisini
incelemek için yalnız aynı export/comparison grubunun8 testi ayrı diagnostic
process'te çalıştırıldı: **8 PASS** (`export-sequence-diagnostic.json/.log`).
Dört probe'un ham değerleri kaydedildi, hiçbir negatif değer tekrar üretilemedi.
Installed wheel'in aynı8 senaryosu da PASS. Tekrarlar distinct toplama eklenmedi.

Bu sonuç bir transient failure gösterir; GC/allocator/runtime veya başka bir
nedeni ispatlamaz. Nedeni bilinmeden production patch, tolerance gevşetme,
clamp,skip veya schema/probe sınırı değişikliği yapılmadı. Güvenilir regression
assertion'ı/düzeltme üretecek deterministik reproducer henüz yoktur. İlk hata
saklandı; sonraki PASS ile temiz bir primary CUDA gate iddiasına çevrilmedi.
**FINAL GO ve aşama kapanışı bu allocator hatasının açıklanmasına bağlıdır.**

## 8. State/RNG/event/memory sınırları

Native FP32/AMP FP16,event timing,non-default stream,selected-device restoration,
model/gradient/buffer/mode ve RNG,enabled/disabled probe,peak reset diagnostics,
export sonrası invariance ve Manager'ın GPU runtime çağırmaması assertion'ları
başarılı senaryolarda geçti. Hatalı case'in bağımsız tekrarında da guard'lar geçti.
Event latency/derived serial service rate sustained FPS değildir. Allocator peak
model-only fiziksel VRAM değildir. Memory reset geri alınamaz; isolation unknown
ise memory comparison insufficient_metadata kalır. Timing uygunluğu memory'den
ayrıdır. Timing ve memory probe sınırları bu görevde değiştirilmedi.
Bir Windows/GPU runtime'ı; fiziksel multi-GPU restore,diğer OS/GPU ve Torch2.2.2
CUDA doğrulanmış sayılmaz. Disk power-loss/hostile FS güvencesi verilmez.

## 9. Torch2.2.2 minimum installed-wheel

Mevcut clean Python3.10.18/Torch2.2.2+cpu/torchvision0.17.2+cpu/SQLite3.50.2
kullanıldı: `C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Final yeni wheel yalnız `E/installed-minimum` target'ına pip isolated/no-index/
no-deps/no-build-isolation/no-compile ile kuruldu; cwd=E,-I -B.
Gerçek create/run/metrics,schema1→2/backup,reference/copy,recovery/retry,
küçük Trainer/history/checkpoint,FP32/FP64 Benchmark,6 export/comparison,
close/reopen ve legacy Trainer/checkpoint/strict replay smoke **PASS**.
Rehberin gerçek manuel akışı çalıştı; tüm loaded HYSo kaynakları target altında.
`installed-minimum.json`,minimum stdout/stderr/install logları kanıttır.

Clean before/after tam JSON eşit: **35 distribution,21432 non-pyc recorded dosya**
(METADATA/RECORD/source/binary hash dahil). `clean-comparison.json` doğrular.
Bu minimum CPU kanıtıdır; Torch2.2.2 CUDA kanıtı değildir.

## 10. Wheel/sdist ve installed CUDA

Full CPU packaging fixture son repository source'lardan yeni wheel ve sdist
build etti.4 packaging testi PASS; iki archive21 production modülünün tamamı
source-byte equal,metadata/README/dependency eşliği ve tests/docs/DB/artifact/
cache/weights exclusions PASS. İki archive ayrı target'lara installed edildi;
legacy/root,Trainer,ExperimentManager ve Benchmark exports/smoke PASS.
HYSo0.2.0,Python>=3.10,Torch>=2.2.2,torchvision>=0.17.2 ve eski8 dependency aynı.

| Yeni artifact | Byte | SHA256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl | 88666 | `05e4e69c4729875e4e432ccf3a085d19cbfe31fab89e63dfa15676ecb9244080` |
| hyso-0.2.0.tar.gz | 85654 | `57dbb8b18e191fae9c0e4b816dd237d50e29250af9ec7c38ce4628adb2692c34` |

Build/archives/iki installed smoke `E/pytest-full/packaging0` altında;
`artifact-integrity.json` tam21-module inventory içerir.
Final wheel ayrıca `E/installed-cuda` target'ına kuruldu. Repo dışı cwd=E,-I -B,
CUDA0 ile benchmark45,export8,Experiment integration2: **55 tekrar PASS**.
21 installed source byte'ı checkout ile eşit; gerçek import target'ta.
Integration script'inin external adapter'ı yalnız installed-target argümanı/import
path ve provenance assertions ekler; senaryo gövdesi değişmez. Original/adapter
hash ve command `installed-experiment-command.json` içinde. Installed tekrarlar
native distinct162'ye eklenmez; ilk checkout hatasını ortadan kaldırmaz.

## 11. Freeze/backlog/değişen dosyalar

Generic Trainer,resume/checkpoint,M1 models/attention,BenchmarkRunner/Result,
ExperimentManager,root exports ve metadata HEAD blob'larıyla eşit. Eski C-C→HEAD
production farkı yalnız yeni iki experiments modülüdür. Checkpoint FORMAT_VERSION1,
BenchmarkResult schema1 ve eski API korunur. Roadmap v1.0 ve backlog byte'ları aynı.
P01–P35 tam/sıralı/tekil: **1 implemented,4 partial,30 planned**; bu gate yeni
performans özelliği veya GUI/dashboard geliştirmedi.

Kullanıcı gradient dosyası başlangıç/son SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Dosya değiştirilmedi/stage edilmedi/geri alınmadı. Repository'de bu görev yalnız
`docs/audit/m2-d-c-final-experiment-quality-gate.md` oluşturur. `git diff --check`
başarılı; Git'in kullanıcı dosyası için LF→CRLF uyarısı byte hash'ini değiştirmedi.

## 12. Geçiş kararı

**M2-D-C PARTIAL GO; FINAL GO değil.** CPU ve Experiment B1/B2/B3 entegrasyonu,
SQLite reliability,min runtime ve packaging kanıtı tamamdır. Açıklanamayan ilk
native allocator ERROR'u final kriterini karşılamaz. Önce o failure için ham
counter değerlerini yakalayan deterministik reproducer/kök neden ve gerekiyorsa
minimum fix+regression gerekir; bu kapanmadan gelişmiş performans aşamasına
**nihai geçiş onayı verilmez**. Backlog35 tamamlandı veya release onaylandı denmez.
