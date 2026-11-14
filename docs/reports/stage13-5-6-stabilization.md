# 13.5.6 — Kritik stabilizasyon

## Durum

Güncel kaynaklarla başlayan production HTTP sunucusunda gerçek Chromium üzerinden model oluşturma/düzenleme/kaydetme, CPU test sonucu, classification evaluation sonucu, kesin parametre sayımı ve rapor üretimi tamamlandı. Kullanıcı 8765 sunucusunu yeniden başlattıktan sonra **canlı geçiş de doğrulandı**: workflow_contract=2, model_revision kayıt sorgusu HTTP 200, 3 kayıtlı model ve 7 mantıksal SQLite kaydı birebir korunmuş durumda. Canlı Chromium oturumunda ana sayfa, test/evaluation formları ve Bilgi Merkezi açıldı; page error/HTTP hata yok, teknik ayrıntılar kapalı. Kanıt: QA `live-after-restart.json` ve `live-after-restart.png`.

İlk otomatik yeniden başlatma denemesi, süreç durdurma adımında `blocked by policy` gerekçesiyle reddedilmişti. Bu engel alternatif bir araçla aşılmadı; yeniden başlatmayı kullanıcı kendi terminalinde yaptı.

Branch `feature/hyso-models-next`, HEAD `d325c613caa2718ea4010bffbe7560219707843d`. Commit/push veya 13.5.7 çalışması yoktur.

## Yeniden üretilen nedenler

1. **Eski backend / güncel statik arayüz karışımı.** 8765 süreci 13 Kasım 18:01:56'da başlamıştı. Python modülleri eski kalırken statik dosyalar her istekte diskten okunuyordu. Sürüm metni iki tarafta da 13.5.6 olduğundan eski sürüm kontrolü bunu yakalamıyordu.
2. **Test/evaluation sayfasındaki ilk başarısız adım kayıt sorgusuydu.** Yeni arayüz `model_revision` filtresi gönderiyor, eski service bu alanı kabul etmiyordu. Salt okunur gerçek `GET /api/records?model_revision=1` çağrısı HTTP 400 / `INPUT_ERROR` döndürdü. Kabul edilen alan listesinde model_revision yoktu. Bu hata iş başlamadan oluşuyordu.
3. **Eksik karşılaştırma/knowledge alanları.** Aynı çalışan backend'in preview yanıtında parameter_status/parameter_reason, catalog yanıtında cost/when_to_change/risk bulunmuyordu. Güncel frontend bu alanlarla başlık oluşturduğu için boş içerik ortaya çıkıyordu.
4. **Üç kayıtlı kullanıcı modeli ayrıca gerçekten geçersiz.** Orijinal config'ler salt okunur API ile alındı ve yalnız izole workspace'e kopyalandı. Çıktıya ulaşmayan düğümler, gelen/beklenen kanal uyuşmazlığı ve NC vektörün NCHW isteyen Reshape'e bağlanması Core tarafından raporlandı. Bu modeller için sayı uydurmak veya otomatik mimari değiştirmek doğru değildir.
5. **Teknik ayrıntı tercihi otomatik açılıyordu.** Önceki GLOBAL technical_details=true kaydı bütün panelleri açabiliyordu.
6. **Tamamlanan işte eski Queued bildirimi kalabiliyordu.** Sonuç yüklenmesine rağmen üst bildirim ilk submit durumunu taşıyordu.

Kanıtlar repository dışındaki `%LOCALAPPDATA%/hyso-stage13-5-6-stabilization` klasöründedir. Token/header kaydı yapılmadı. `live-reproduction.json`, `first-record-query-failure.json`, `copied-model-diagnostics.json` ve `before-restart.png` ilk durumun kanıtıdır.

## Düzeltmeler

- Sunucu beş statik dosyayı başlangıçta sabitler; çalışan eski Python modüllerine sonradan değişen frontend dosyaları karışmaz. Health yanıtında workflow_contract=2 vardır. Güncel UI bu sözleşme yoksa anlaşılır yeniden başlatma mesajı gösterir; yanlış sözleşmeyle iş göndermeye devam etmez.
- Kayıt veya rapor sorgusu başarısız olduğunda test/evaluation model seçimi ve formu yine çizilir. Hata görünürdür; taslak silinmez. 4xx yanıtı tamamlanan/başarısız job gibi sunulmaz.
- API hatasında gerçek backend nedeni genel Türkçe fallback'in önüne alınır. Evaluation eksik dosya mesajı gerekli alanları açıklar. Core classification metadata ve label doğrulaması setup aşamasında çalışır; targets/predictions alanı, geçerli sınıf indeksleri ve aynı uzunluk gereği belirtilir. Doğrulama kaldırılmadı.
- Teknik ayrıntılar her oluşturuluşta ve sayfa geçişinde kapalıdır. Eski global auto-open tercihi artık etkin UI kontrolü değildir; backend'deki eski tercih/veri silinmez. Açmak panel başına açık kullanıcı eylemidir.
- Knowledge boş içerikli başlık çizmez. Güncel backend'deki 61 mekanizmanın cost ve 225 parametre alanının açıklamaları korunur; eski backend alan eksikliği görünmez boş başlığa dönüşmez.
- Geçersiz karşılaştırma kartı gerçek node ve Core nedenlerini gösterir. Kesin sayı farkı yoksa iki kesin sayım gereği belirtilir; eğitim uygunluğu yanlış biçimde bilinmiyor sayılmaz. Kesin sayım ilerleme durumu görünürdür. Sıfır değerler sıfır kalır.
- Çalışan işin üst bildirimi son kuyruğa kabul durumunda takılmaz; ilgili modelin güncel iş durumu izlenir.

## Gerçek kullanıcı kabul akışı

`test_stabilization_complete_classification_workflow`, aynı production StudioServer + frontend + Core zincirinde:

1. Ana sayfadan CNN oluşturdu; stem kernel_size değerini değiştirdi, Core doğrulaması ve taslak kaydını yaptı.
2. Doğrulanmış sürüm oluşturdu; forward ve strict serialization seçeneklerini gerçek UI üzerinden işaretledi, setup ve ayrı başlatma eylemlerini kullandı. Gerçek test sonucu ekranda göründü.
3. Evaluation dosyası seçti: targets `[0,1,1]`, predictions `[0,1,0]`, iki sınıf adı ve test split. Setup ve ayrı değerlendirme işi tamamlandı. Ekranda **2/3 accuracy**, macro precision/recall **0,75**, macro F1 **2/3** gerçek evaluator çıktısından görüldü. Bu izole fixture sonucu kullanıcı workspace'ine yazılmadı; modelden inference yapıldığı iddia edilmedi.
4. Taslak ve sürümü karşılaştırdı. Görünür kesin CPU sayımı düğmesiyle iki gerçek modül oluşturuldu; her iki kart ve tablo **42 EXACT parametre**, fark **0** gösterdi.
5. Conv2D ve SE parametre açıklamalarını açtı; görünür açıklama paragraflarının boş olmadığını doğruladı.
6. Kalıcı technical_details=true kaydı bulunmasına rağmen başlangıç ve navigasyonda ayrıntıların kapalı kaldığını, paneli elle açıp başka sayfaya geçince kapandığını doğruladı.
7. TR geçişi sonrasında evaluation kaydı ve test sayfası korundu; gerçek rapor işi tamamlandı ve indirme etkinleşti.

Önceki gerçek Chromium eğitim/performance testleri de gerçek dosya seçimi, setup, kısa CPU training/benchmark işi ve ekrandaki retained sonucu doğrular. Native kabul suite'i, CUDA, uzun eğitim veya checkpoint/resume geliştirmesi çalıştırılmadı.

`stabilization-http.json` gerçek POST payload'larını headers/token olmadan kaydeder. Browser audit dosyaları page error ve HTTP yanıtlarını içerir. `stabilized-evaluation.png`, `stabilized-count.png` ve `copied-user-model-diagnostics.png` görüntüleri alındı ve görsel olarak incelendi.

## Test ve koruma

Son gruplar: Studio **175** (14 gerçek Chromium dahil), Core **676**, dağıtım **10** başarılı test. Son UI durum bildirimi değişikliğinden sonra birleşik Chromium kabulü ayrıca yeniden geçti; arşiv içerikleri ayrıca yeniden karşılaştırıldı. Tekrar koşular toplamda yeniden sayılmaz. Ayrıntılı XML sonuçları QA klasöründedir.

Wheel/sdist 167 ticari Python + 5 statik kaynak, toplam 172 kaynak içerir. Son arşiv/kaynak hash eşleşmeleri `final-proof.json` içindedir. Ticari M3/M4 dışlamaları, schema1/2/3 ve Core public API değiştirilmedi.

47/47 forensic/debug, 71/71 native ve 6/6 checkpoint/kabul dosyası önceki koruma manifestiyle eşleşir. P01–P35 kanıtlarına yazılmadı. Kullanıcı workspace'ine test modeli veya fixture sonucu eklenmedi. Yeniden başlatma öncesi aktif/kuyruk işi yoktu; SQLite online backup ile **7 mantıksal kayıt** tutarlı biçimde yedeklendi. Süreç durdurma reddedildiğinden eski sunucu çalışmaya devam etti.

Değişen kod: `hyso/studio/server.py`, `hyso/studio/jobs.py`, `hyso/studio/static/app.js`, `tests/test_studio_browser.py`. Rehber/ana rapor bu stabilizasyon raporuna yönlendirir. Yeni runtime bağımlılığı veya endpoint yoktur.

## Kalan sınırlar ve kullanıcı sunucusuna geçiş

Güncel kaynaklarla başlayan izole production sunucusunda test/evaluation çalışmaktadır. Kullanıcının yeniden başlatmasından sonra 8765 canlı oturumunda da test/evaluation formlarının ve kayıt sorgularının çalıştığı doğrulandı. Yeniden başlatma gerektiğinde mevcut terminalde Ctrl+C, ardından:

```powershell
& 'C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe' -m hyso.studio --port 8765
```

Kaydedilmemiş tarayıcı düzenlemesi varsa önce koruyun; sonra sayfayı yenileyin. Yeniden başlatma SQLite modellerini silmez. Üç mevcut geçersiz modelin bağlantı/kanal hataları ayrıca düzeltilmelidir; sıfırdan oluşturulan geçerli CNN bunlardan bağımsız çalışır.

Kısa CPU, <=1M parametre üst sınırı, classification-only evaluation, fresh initialization ve önceki diğer destek sınırları sürer. Otomatik onay denetiminin engellediği süreci alternatif bir yolla sonlandırma denenmedi. Commit/push yapılmadı; 13.5.7'ye geçilmedi.
