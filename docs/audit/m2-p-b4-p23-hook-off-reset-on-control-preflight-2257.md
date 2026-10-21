# HYSo — B4/P23 hook-off/reset-on: 22:57 GPU preflight STOP

2026-10-21, Europe/Istanbul. **NOT STARTED / PREFLIGHT REJECTED.**
Tek yeni preflight serisinde GPU utilization **%21/%32/%35** ölçüldü.
Audit üç örneğin her birinin **%35 altında** olmasını gerektirir; %35 eşitliği
başarısızdır. Başarısız kontrolden sonra tekrar ölçüm/admission yapılmadı.
Controller **0**, parent **0**, worker **0**, CUDA workload execution **0**.
Tek execution hakkı kullanılmadı; yeni native crash, timeout veya PASS yoktur.

Bu kayıt [önceki devam audit'inden](m2-p-b4-p23-hook-off-reset-on-control-continuation.md)
sonraki kullanıcı yönlendirmesine aittir. Önceki **28 dirty dosyanın tamamı
byte eşit** korundu. Sonuç yalnız bu yeni dosyaya yazıldı; final29 dirty.

## Korunan kimlik ve execution bütçesi

Branch `feature/hyso-models-next`, HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş.
Başlangıç path/status/SHA256 envanteri önceki evidence'ın
`final-repository-protection.json` envanteriyle eşleşti. Başlangıç28 dosyada
son byte/status farkı0. Production guard dosyası bu korumaya dahildir.
Korunan `tests/test_gradient_accumulation.py` SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

Snapshot **T**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-b4-blocker-20261021T173434/proposed-hook-off-control-r2`.
Yeni evidence **E**: `T/execution-evidence-20261021T225702`.
Run ID `2720e745-9324-43df-aa56-cc25cf7bb415`.
Mevcut manifest SHA256 başlangıç/son aynı:
`2be2ebe1562bfc0863b16dc6d1bb093f91f32f1dd9ab8b456ed94387cedbd78d`.
251 source/test/factory/bootstrap/config/doc ve26 runtime/interpreter/DLL
dosyası başlangıçta ve sonda rehash edildi: mismatch0, source read-only
attribute eksik0. Seal/run-ID/interpreter eşleşti. Snapshot kaynakları ve
manifest değiştirilmedi, yeniden mühürlenmedi. Loaded-source attestation
çalışmadı; eski C `unverified_source_identity` kalır.

## Tek güncel preflight

Kaynak kimliği ve koruma kontrollerinden sonra tek üç-örnek GPU serisi,
process/compute inventory, host RAM ve disk kontrolü yapıldı.
`live-preflight.json` zamanı **22:57:12.0078901 +03:00**, `passed=false`.

|Kontrol|Ölçüm|Audit koşulu / sonuç|
|---|---|---|
|Boş host RAM|6,030,532,608 B ≈5.616 GiB|>=4 GiB: PASS|
|GPU utilization|%21/%32/%35|Her örnek<%35: FAIL|
|Boş GPU memory|Üç örnekte minimum5498 MiB|>=2048 MiB: PASS|
|C: boş disk|47,777,071,104 B ≈44.496 GiB|>=12 GiB: PASS|
|Bilinen compute workload process filtresi|0|Inventory'de çakışan workload görülmedi|

GPU UUID `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`; isim, driver ve
toplam bellek ham GPU satırlarında saklandı. NVML sorguları CUDA training
veya smoke değildir. Process filtresi python/torchrun/ollama/ffmpeg/blender/
cdb/nvcc/llama/vllm adlarını kapsadı; bütün süreçlerin iş yükü analizi değildir.

Compute PID24364, CIM'de `WhatsApp.Root.exe` ve WindowsApps executable
yoluyla eşleşti. Bu desktop uygulaması önceki desktop ad listesinin dışında
kaldığından otomatik preflight kaydında `unresolved_compute` içine yazıldı;
PID kimliği gerçekte çözümlendi. WhatsApp'ın çakışan training yaptığı sonucu
çıkarılmadı. GPU utilization başarısızlığı tek başına STOP için yeterlidir;
liste genişletilip admission yeniden denenmedi. WDDM GPU process memory
alanları N/A olabilir; ham inventory E içinde korunur.

## Limitler, kanıt ve duruş

Yalnız önceden onaylanan AdamW FP32/default cuda:0 stream,
hooks-off/reset-on; seed71, epochs2, samples5, batch2, accumulation2,
Linear(3,4)–Dropout(.2)–Linear(4,2) recipe hazır kaldı; çalıştırılmadı.
Baseline/prefix/retry0. Worker60s, outer90s; process4 GiB,
child Job8 GiB/4 process, outer Job12 GiB/8 process, allocator fraction.15,
kill-on-close; stdout/stderr64 KiB/channel, result2 MiB,
metadata2000 row/1 MiB per process değişmedi.

`attempt-consumed.json`, `child-launch-consumed.json`,
`explicit-user-approval.json`, `fresh-preflight.json`, `outer-receipt.json`,
`child-os-receipt.json`, `diagnostic-result.json` başlangıçta/sonda yoktu.
Hazır worker/events dizinleri file_count0 kaldı. Kullanıcı koşullu onay
vermiştir; approval admission dosyası preflight geçmediğinden yazılmadı.
Job oluşturulmadı; kernel readback, owned PID/Job active0 cleanup receipt,
reset/hook execution ledger ve native exit sınıflaması **N/A**.

Mevcut dump inventory `dumps-before.json` içinde. Yeni worker/crash
olmadığından WER sorgusu veya crash sonrası bounded30s gözlem uygulanmadı.
Dump analizi/kopyalama/upload, policy/registry değişikliği yoktur.
Kullanıcı uygulaması kapatılmadı, process sonlandırılmadı; limitler,
production guard, dependency/Torch/driver değiştirilmedi.
Stage/commit/push/reset/restore/clean yapılmadı.

E: `start-protection.json`, `prelaunch-identity.json`, `live-preflight.json`,
`postflight-protection.json`, `final-repository-protection.json` ve
`evidence-manifest.json` ham ölçümleri ve koruma envanterlerini saklar.
**B4 OPEN/UNRESOLVED; P23 PARTIAL; B10 PARTIAL; R1 NO-GO** korunur.
Yeni native causality veya safety acceptance sonucu yoktur. Bu teslim durur;
başka senaryo veya otomatik retry yapılmaz.
