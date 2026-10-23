# P21 Worker Tracing kabul raporu — 23 Ekim 2026

Bu rapor 253 PASS v1 kabulünün tarihsel kaydıdır. Sonraki uygulama ve v2 kabulü:
[P21 feature completion](p21-feature-completion.md).

Nihai feature durumu **partial**. Windows CPU worker-local fetch/decode/transform
kanıtları ve güvenilirlik alt kapsamı kalıcı pytest testleriyle doğrulandı.
Collate/dispatch, host/worker clock mapping, queue occupancy, prefetch efficiency
ve bağımsız execution/epoch/batch ilişkilendirmesi tamamlanmış sayılmaz.

## Başlangıç ve kapsam

Branch: `feature/hyso-models-next`; HEAD: `2366607`.
Başlangıçta `_loader_dataset_trace.py`, `_loader_event_channel.py`,
`_loader_trace_groups.py` değiştirilmişti. `worker_trace_result.py` ve 15
`scratch_p21_*.py` dosyası untracked durumdaydı. Yerel değişiklikler incelendi ve
korundu; reset/clean/rebase, commit veya push yapılmadı. Diğer performans
özelliklerinin uygulamaları değiştirilmedi.

Ortam: Windows, `torch_gpu`, Python 3.10.18, PyTorch 2.5.1+cu121.
CUDA görünürlüğü testlerde `-1`; GPU stress veya CUDA kabulü yapılmadı.
Ortamda pytest yoktu; aynı interpreter ile `-m pip install pytest` çalıştırıldı
(pytest 9.1.1). Pip mevcut ortamın `-umpy` distribution uyarısını verdi.

## Değişiklikler ve giderilen hatalar

| Dosya | Sonuç |
|---|---|
| `hyso/performance/_loader_event_channel.py` | Queue feeder/yarım pipe frame riskini kaldıran bounded synchronous shared-memory aktarımı; timed lock; senkron JSON snapshot; saturation/oversize rejection; zorunlu sayaç ve event audit |
| `hyso/performance/_loader_dataset_trace.py` | Yerel per-fetch paket yaklaşımı korundu; drop sayısı trace başına delta; kanal diagnostic hataları dataset sonucunu veya asıl exception'ı gölgelemez |
| `hyso/performance/_loader_trace_groups.py` | Değişken stage sayısı, kimlik, süre, nesting ve trace sayımı; malformed hashable olmayan değerlerde fail-closed |
| `hyso/performance/_loader_trace_validation.py` | Ortak strict worker event doğrulayıcısı; saat, duration, PID/worker/sample/trace, outcome ve clock scope kontrolü |
| `hyso/performance/_loader_worker_lifecycle.py` (yeni) | Caller'ın açıkça kullanabileceği tek toplam deadline altında graceful/terminate/kill cleanup; forced veya nonzero kapanış hata |
| `hyso/performance/worker_trace_result.py` (başlangıçta untracked) | Immutable companion; kaynak ve ham evidence hash; yeniden türetilmiş analizle tamper kontrolü; unknown execution/epoch/batch; eski prototype v1 JSON okuma |
| `hyso/performance/__init__.py` | Companion result ve analiz fonksiyonunun additive export'u |
| `tests/test_worker_tracing.py` (yeni) | Kalıcı birim sözleşmeleri ve native kabul testleri |
| `tests/p21_worker_acceptance.py` (yeni) | Mevcut Windows Job supervisor altında gerçek spawn/failure senaryoları |
| `docs/design/p21-worker-tracing.md` (yeni) | Kullanım, sahiplik, sayaç kapsamı ve unsupported sınırları |
| `docs/design/hyso-analysis-optimization-contract.md` | Yalnız P21 satırında ölçülen ve kalan yeteneklerin güncel tanımı |

`_loader_worker_trace.py`, `_loader_stage_trace.py` ve `dataloader.py` incelendi;
bu dosyalar değiştirilmedi. Mevcut 17 implemented özelliğin uygulamalarına
geçilmedi. `scratch_p21_*.py` dosyaları referans olarak okundu, değiştirilmedi.

## Gerçek komutlar ve sonuçlar

PowerShell'de kullanılan interpreter:

```powershell
$p21Python = 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe'
& $p21Python -m pytest tests/test_worker_tracing.py -q --tb=short
& $p21Python -m pytest tests/test_dataloader_protocol.py tests/test_performance_dataloader.py tests/test_performance_training.py tests/test_training_details.py tests/test_training_throughput_protocol.py -q --tb=short
```

İlk P21 turu: **22 PASS / 7 FAIL / 0 SKIP**, 117,30 saniye.
Başarısızlıklar saklanmadı: 15 saniyelik cold worker startup sınırı baseline'da
aşıldı; test dosyasının BOM'u düzenleme sırasında bozuldu; başarısız senaryo
aynı admission dosyasıyla tekrar denenmişti. UTF-8 düzeltildi, output adları
benzersiz yapıldı, worker başlangıç sınırı 45 saniye ve dış Job deadline 180
saniye oldu. Worker cleanup bütçeleri bundan bağımsız olarak kısa kaldı.

İkinci P21 turu: **30 PASS / 0 FAIL / 0 SKIP**, 325,95 saniye.
Ardından gerçek hanging-worker senaryosu ve prototype v1 uyumluluk kontrolü
eklenerek son tur çalıştırıldı.

Nihai P21 turu: **31 PASS / 0 FAIL / 0 SKIP**, **339,13 saniye**.
Sekiz native senaryonun dış launcher exit code'u 0; hiçbir dış Job deadline
aşılmadı. Kontrollü worker crash exit 7 ve forced worker termination, bu
senaryolarda beklenen hata kanıtıdır; temiz eğitim başarısı olarak sayılmadı.

| Senaryo | Gerçek sonuç |
|---|---|
| CPU eğitim, iki epoch | 16/16 paket, 46 event; attempted=accepted=received=16, rejected=0; worker exit 0/0 |
| Saturation | attempted=16, accepted=received=1, rejected=15; incomplete |
| Recorder overflow | 30 drop; incomplete |
| Dataset exception | 8/8 paket gelse de failed event nedeniyle incomplete; worker alive false/false; cleanup 1,750 s |
| Ani worker çıkışı | Worker exit 7/0; 7/8 paket; incomplete; alive false/false; cleanup 1,500 s |
| Kilit sahibi ölümü | Gerçek process exit 7; üç kanal işlemi TimeoutError; close dahil 0,312 s |
| Shutdown thread timeout | Forced hata; 0,187 s; temiz receipt üretilmedi |
| Gerçek bekleyen worker | 2 s toplam bütçe altında forced termination; exit -15/-15; alive false/false; cleanup 1,265 s |

Kanıtlar: [ilk başarısız tur](p21-worker-tracing-evidence/initial-pytest.txt),
[ara tur](p21-worker-tracing-evidence/intermediate-pytest.txt),
[son P21 logu](p21-worker-tracing-evidence/final-pytest.txt),
[native receipt ve sonuçlar](p21-worker-tracing-evidence/native-summary.json),
[gerçek companion artifact](p21-worker-tracing-evidence/worker-trace.json).

P19/P20/P21/P22 regresyonu: **222 PASS / 0 FAIL / 0 SKIP**, **50,10 saniye**.
Mevcut `GradScaler` deprecation uyarısından 199 warning alındı; başarısız test
yok. Eski P19 schema1 fixture, source hash/tamper/no-clobber, native replay,
isteğe bağlı P20 ve P22 throughput sözleşmeleri geçti.
[Regresyon logu](p21-worker-tracing-evidence/regression-pytest.txt).

Nihai iki komut toplamı: **253 PASS / 0 FAIL / 0 SKIP**. İlk başarısız tur ve
ara tur bu toplamın içinde tekrar sayılmadı. `git diff --check` temizdir.

## Kanıt kapsamı ve uyumluluk

- Gerçek iki Windows spawn worker, persistent workers ve iki epoch; her sample
  ID iki kez, 16 farklı trace ID ve değişken 2/3/4 event sayısı.
- Normal IPC'de attempted/accepted/received 16, rejected 0; saturation'da
  attempted 16, accepted/received 1, rejected 15. İkinci durum incomplete.
- Recorder overflow'da 30 drop; drop sayıları her trace'in delta'sıdır.
- Eksik/duplicate/bozuk paket, schema/PID/sample/stage/clock/duration/outcome
  hataları ve eksik/tutarsız sayaçlar başarı üretemez.
- Dataset exception, `os._exit(7)`, kilit sahibi ölümü ve shutdown deadline
  subprocess içinde sınanır; Job handle kapanışı process ağacını temizler.
- Baseline ve observed ayrı karşılaştırma eğitimleridir; her çalışmada fit
  yalnız bir kez. Model, optimizer, gradient, RNG, loader generator state ve
  gözlenen input sırası bit düzeyinde eşittir.
- Aynı observed fit'ten P19 timing, P20 stages, P21 loader ve P22 throughput
  üretilir. Companion oluşturulması veya export eğitim çalıştırmaz.
- P19/P20 source hash ve DataLoaderAnalysisResult schema v1 korunur.
  Companion JSON round-trip, immutable snapshot, no-clobber, tamper rejection
  ve eski prototype v1 payload round-trip kontrol edilir.

## Kalan sınırlar

`analysis.status=complete` yalnız teslim edilen worker kanıtının tutarlılığını
gösterir. Kaynağa bağlanan hash ve doğrulanan PID/sayım, aynı execution'a
aitliğin bağımsız kanıtı değildir; execution/epoch/batch `unknown` kalır.
Mevcut v1 kaynak provenance, kanala bağlanmış bir session receipt taşımaz;
persistent PID ve sample ID tekrarları bağımsız segment kimliği sağlamaz.
Hash'ler bütünlüğü denetler, üreticiyi doğrulamaz. Legacy companion v1, ham
evidence hash alanı olmadan kendi eski payload'ını koruyarak okunur; yeni
artifact'lar hash ve explicit association içerir.

Validation-loader worker ilişkilendirmesi desteklenmez. Prefetch/drop-last veya
erken kesilme nedeniyle fetch sayısı tüketilen sample sayısıyla eşleşmezse
companion incomplete kalır. Kanal tek audit kapsamı içindir; farklı run'lar
ve legacy event akışı karıştırılmamalıdır. Parent tarafından alınan tüm paketler
final audit'e verilmelidir.

Kanal varsayılan 1024 × 65536 byte, en fazla 128 MiB; bütçe dışı paketler
reddedilir. Kilit sahibi ölürse kanal kurtarılmaz: bounded TimeoutError ve
kanalı bırakma gerekir. Cleanup helper PyTorch private iterator API'sine
bağımlıdır ve yalnız çağıranın açık kullanımı içindir. Standart ölçüm API'leri
worker sahipliğini değiştirmez.

CUDA kabulü **NOT_RUN**; diğer platformlar **NOT_RUN**. Bunlar bu Windows CPU
testlerinin PASS sayısına dahil değildir. P21'in `implemented` olarak
işaretlenmesi için kalan ölçüm ve ilişkilendirme sözleşmeleri tamamlanmalıdır.

## Git diff --stat

Bu çıktı yalnız tracked dosyaları kapsar; yeni/untracked dosyalar yukarıdaki
dosya tablosunda ve aşağıdaki status çıktısında listelenir.

```text
 docs/design/hyso-analysis-optimization-contract.md |   2 +-
 hyso/performance/__init__.py                       |   2 +
 hyso/performance/_loader_dataset_trace.py          |  50 +++-
 hyso/performance/_loader_event_channel.py          | 333 ++++++++++++++++++---
 hyso/performance/_loader_trace_groups.py           | 121 ++++++--
 hyso/performance/_loader_trace_validation.py       |  22 ++
 6 files changed, 452 insertions(+), 78 deletions(-)
```

## Git status --short

```text
 M docs/design/hyso-analysis-optimization-contract.md
 M hyso/performance/__init__.py
 M hyso/performance/_loader_dataset_trace.py
 M hyso/performance/_loader_event_channel.py
 M hyso/performance/_loader_trace_groups.py
 M hyso/performance/_loader_trace_validation.py
?? docs/audit/p21-worker-tracing-acceptance.md
?? docs/audit/p21-worker-tracing-evidence/
?? docs/design/p21-worker-tracing.md
?? hyso/performance/_loader_worker_lifecycle.py
?? hyso/performance/worker_trace_result.py
?? scratch_p21_dataloader.py
?? scratch_p21_delivery_audit.py
?? scratch_p21_failure.py
?? scratch_p21_integrity.py
?? scratch_p21_ipc.py
?? scratch_p21_ipc_v2.py
?? scratch_p21_live_delivery.py
?? scratch_p21_multistage.py
?? scratch_p21_multistage_spawn.py
?? scratch_p21_result_integration.py
?? scratch_p21_stages.py
?? scratch_p21_trace_integration.py
?? scratch_p21_variable_stages.py
?? scratch_p21_worker_core.py
?? scratch_p21_worker_shutdown.py
?? tests/p21_worker_acceptance.py
?? tests/test_worker_tracing.py
```
