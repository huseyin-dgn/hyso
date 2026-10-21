# HYSo — B4/P23 hook-off/reset-on control: GPU preflight STOP

2026-10-21, Europe/Istanbul. **NOT STARTED / PREFLIGHT REJECTED.**
Son admission ölçümünde GPU kullanım örnekleri %31, %35, %71 oldu;
zorunlu koşul üç örneğin de **%35 altında** olmasıdır. Host RAM yeterliydi.
Controller launch **0**, parent launch **0**, worker launch **0**, CUDA workload
execution **0**. Tek execution hakkı kullanılmadı. Yeni native crash, timeout
veya training PASS sonucu yoktur. Otomatik retry yapılmadı; bu teslim durur.

Bu rapor, [önceki preflight audit'inin](m2-p-b4-p23-hook-off-reset-on-control.md)
devamıdır. Önceki rapor dahil başlangıçtaki **27 dirty dosya byte eşit**
korundu; sonuçlar bu ayrı dosyaya eklendi. Final dirty sayısı 28'dir.

## Yetki, kimlik ve koruma

Kullanıcı aynı mühürlü `proposed-hook-off-control-r2` üzerinde yalnız tek
AdamW FP32/default stream, hooks-off/reset-on CUDA kontrolüne koşullu onay verdi.
Seed71, epochs2, samples5, batch2, accumulation2 ve
Linear(3,4)–Dropout(.2)–Linear(4,2) recipe değişmedi. Baseline/prefix/retry0.

Branch `feature/hyso-models-next`, HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`; index boş korundu.
Önceki evidence `final-protection.json` içindeki dosya hash/status envanteriyle
başlangıç eşleşti; bu teslimin 27 dosyalık başlangıç envanterine karşı son
SHA256 kontrolünde değişen dosya0. Korunan gradient SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Production guard dosyası da bu byte eşit envanter içindedir.

Snapshot **T**:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-b4-blocker-20261021T173434/proposed-hook-off-control-r2`.

Yeni evidence **E**: `T/execution-evidence-20261021T224622`.
Run ID `2720e745-9324-43df-aa56-cc25cf7bb415`.
Manifest SHA256 başlangıç/son:
`2be2ebe1562bfc0863b16dc6d1bb093f91f32f1dd9ab8b456ed94387cedbd78d`.
251 source/test/factory/bootstrap/config/doc ve 26 runtime/interpreter/DLL
dosyası başlangıçta ve sonda rehash edildi: mismatch0, source read-only
attribute eksik0. Kaynaklar yeniden mühürlenmedi veya değiştirilmedi.
Bu disk kimliği doğrulamasıdır; parent/worker loaded-source attestation
çalışmadı. Eski C `unverified_source_identity` durumu değişmez.

## Güncel kaynak ölçümleri ve admission

|Kontrol|İlk güncel inventory|Son admission, 22:47:17 +03:00|Karar|
|---|---|---|---|
|Boş host RAM|7,835,189,248 B ≈7.297 GiB|7,764,926,464 B ≈7.232 GiB|PASS: en az4 GiB|
|C: boş disk|47,829,504,000 B|47,826,165,760 B ≈44.542 GiB|PASS: en az12 GiB|
|GPU kullanım, üç örnek|%17/%17/%33|%31/%35/%71|FAIL: her örnek<%35|
|Boş GPU bellek|5615 MiB|minimum5616 MiB|PASS: en az2048 MiB|
|Çakışan workload|0|0|Inventory'de görülmedi|
|Çözümlenemeyen compute PID|0|0|Inventory eşleştirmesi tamamlandı|

GPU RTX3060 Laptop,6144 MiB,driver591.74,
UUID `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`.
NVML sorguları eğitim/CUDA smoke değildir. Compute listesi desktop süreçleri
ve WDDM N/A bellek alanları içerdi. İzin yetersiz görünen PID1432, CIM
inventory'sinde `dwm.exe` olarak eşleşti; executable path/command line null.
Python/torchrun/ollama/ffmpeg/blender/cdb/nvcc/llama/vllm adlı workload
gözlenmedi. Bütün süreçlerin CPU/commit veya içerik profili çıkarıldığı
iddia edilmez. GPU yükselişinin nedeni bu inventory'den belirlenmedi.

İlk inventory sonrası controller'a <=30s güncel kayıtla giriş hazırlığı için
son taze preflight yapıldı. Bu son kontrol başarısız olunca admission durdu;
sonrasında eşik geçene kadar polling veya ikinci admission denemesi yoktur.
`live-preflight.json` ve `final-live-preflight.json` ham ölçümleri, zamanları,
process ve compute inventory'sini saklar; son `passed=false`.

## Execution, limitler ve sonuç sınırı

Controller çağrılmadı. Approval/fresh-preflight admission dosyaları T köküne
yazılmadı. `attempt-consumed.json`, `child-launch-consumed.json`,
`explicit-user-approval.json`, `fresh-preflight.json`, `outer-receipt.json`,
`child-os-receipt.json`, `diagnostic-result.json` başlangıçta ve sonda yoktu.
Hazır `worker`/`events` dizinleri file_count0 kaldı. Kullanıcının onayı vardır;
admission dosyasının yokluğu kaynak koşullarının geçilemediğini gösterir.

Mühürlü limitler aynen korundu: worker60s, outer90s; process4 GiB,
child Job8 GiB/4 process, outer Job12 GiB/8 process, allocator fraction.15,
kill-on-close; stdout/stderr64 KiB/channel, result2 MiB,
metadata2000 row/1 MiB per process. Job oluşturulmadığı için kernel readback,
Job active0 cleanup receipt, PID/exit, reset ve hook ledger kanıtı **N/A**.
Konfigürasyon değerleri runtime PASS veya sıfır callback gözlemi sayılmaz.

`dumps-before.json` yalnız mevcut dump inventory'sidir; yeni workload/PID
olmadığından native crash sonrası WER/dump30s gözlem penceresi uygulanmadı.
WER sorgusu, dump içerik analizi/kopyalama/upload veya global policy/registry
değişikliği yapılmadı. Uygulama kapatma, kullanıcı process sonlandırma,
limit düşürme/artırma, dependency/Torch/driver değişikliği yoktur.
Stage/commit/push/reset/restore/clean yapılmadı.

`start-protection.json`, `prelaunch-identity.json`,
`postflight-protection.json`, `final-repository-protection.json` ve
`evidence-manifest.json` koruma ve kanıt envanterlerini saklar.
**B4 OPEN/UNRESOLVED; P23 PARTIAL; B10 PARTIAL; R1 NO-GO** korunur.
Root cause, hooks gerekliliği veya peak-reset causality hakkında yeni native
sonuç yoktur. Yeni yönlendirme ve tam güncel preflight olmadan başka execution
yapılmayacaktır. Bu teslim burada durur.
