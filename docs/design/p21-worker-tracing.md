# P21 worker tracing: sınırlar ve kullanım

Bu belge v1 companion kapsamını kaydeder. Additive v2 için
[worker tasks](p21-worker-tasks.md) belgesine bakın. V1 unknown/unsupported alanları
eski JSON uyumluluğu için korunur.

P21 durumu **partial**. Worker-local `dataset_fetch`, açıkça enstrümante edilen
`decode` ve `transform` ölçülebilir. Collate, dispatch, queue occupancy, prefetch
efficiency ve host/worker clock mapping desteklenmez. Worker süreleri, parent
`next()` beklemesinden çıkarılmaz veya bu beklemeye eklenmez.

## Tek eğitim ve companion artifact

`TracedDataset(dataset, channel=channel, trace_batches=True)` mevcut map-style
Dataset'i sarmalar. Dataset içindeki isteğe bağlı `measure_stage(None, ...,
sample_id=str(index))` çağrıları etkin recorder'a yazılır. Her fetch UUID alır;
aynı sample ID farklı epoch'larda tekrar edebilir. Stage sayıları sabit değildir.
Eğitim `measure_training_with_loader(...)` ile **bir kez** çalıştırılır. Kanal,
DataLoader ve worker yaşam döngüsü çağıranın sorumluluğundadır.

Tam iterasyonlardan sonra worker'ları durdurun; final paketleri ve
`channel.trace_counters()` snapshot'ını alın. Worker ID, PID, alive ve gerçek
exitcode kayıtlarını `analyze_worker_traces(loader_result, packets,
trace_counters=..., worker_states=...)` fonksiyonuna verin. Bu fonksiyon eğitim
çalıştırmaz. `WorkerTraceAnalysisResult` ayrı JSON artifact'tır; eski
`DataLoaderAnalysisResult` schema v1 ve P19/P20 source hash sözleşmesi değişmez.

`analysis.status=complete` yalnız verilen worker kanıtlarının sayım, kimlik,
süre, nesting ve process receipt denetimini geçtiğini ifade eder. Execution,
epoch ve batch ilişkilendirmesi `unknown` kalır: eşleşen sayım ve PID, aynı
training segmentine aitliğin bağımsız kanıtı değildir. Kaynak ve evidence SHA256
hash'leri bütünlük denetimidir; kötü niyetli üreticinin kimliğini doğrulamaz.

## IPC ve kapanış

Kanal feeder thread/pipe içeren `multiprocessing.Queue` yerine senkron JSON
snapshot'larını sabit boyutlu paylaşılan bellek slotlarına yazar. Varsayılan
1024 slot × 65536 byte; toplam kanal bütçesi en fazla 128 MiB. Büyük veya JSON'a
çevrilemeyen paketler ve kapasite aşımı reddedilir. Paketler sessizce kesilmez.
`attempted = accepted + rejected`; final audit için `attempted = expected_fetches`,
`accepted = received` ve `rejected = 0` zorunludur. `received` audit'in gerçekten
aldığı paketlerin sayısıdır; consumer'a gönderilen toplamı tahmin etmez.

Kanal tek audit kapsamına aittir: farklı training çalışmaları veya legacy
`send()` event'leri aynı audited trace kanalında karıştırılmamalıdır. Audit öncesi
alınan tüm paketler korunmalıdır. Drop-last, erken durdurma veya prefetch nedeniyle
fetch sayısı tüketilen sample sayısından farklıysa companion `incomplete` kalır;
eksik kanıt tamamlanmış gibi gösterilmez.

Kilit acquisition varsayılan 0.25 saniyeyle sınırlıdır. Lock sahibi ölürse
snapshot/drain/send `TimeoutError` üretir; kanal tekrar kullanılmaz. Wrapper bu
diagnostic hatayı veri çıktısını değiştirmeden sayar; audit başarı üretemez.
`close()` lock edinmez, thread join etmez ve worker öldürmez. Paylaşılan bellek
referanslar bırakıldığında geri kazanılır.

`shutdown_traced_workers(iterator, timeout=8)` açıkça çağrılabilen yardımcıdır.
PyTorch private shutdown daemon thread'de çalışır; kalan süre terminate/kill ve
join için aynı toplam bütçeden ayrılır. Forced kapanış veya nonzero exit code
hata üretir. Standart ölçüm API'leri bu yardımcıyı otomatik çağırmaz. Private
PyTorch API bağımlılığı nedeniyle sürüm uyumluluğu test edilmelidir.

## Kalıcı kabul testleri

`tests/test_worker_tracing.py` birim sözleşmelerini ve native senaryoları yürütür.
`tests/p21_worker_acceptance.py` gerçek Windows spawn, iki persistent worker,
iki epoch, variable stages, saturation, recorder overflow, exception, crash,
abandoned lock ve shutdown timeout senaryolarını içerir. Mevcut
`tests/p21_native_supervisor.py` Job nesnesi alt process ağacını sınırlar ve
senaryo sona erdiğinde veya dış deadline aşıldığında handle kapanışıyla temizler.
Gerçek eğitim baseline ve gözlemlenen ayrı karşılaştırma çalışmalarıdır; her
çalışmada fit yalnız bir kez çağrılır. Model, optimizer, gradient, RNG, generator
state ve gözlenen input sırası bit düzeyinde karşılaştırılır.

Kabul logları ve nihai sayılar `../audit/p21-worker-tracing-acceptance.md`
raporundadır. CUDA ve diğer platformlar bu CPU Windows kabulünün kapsamı dışındadır.
