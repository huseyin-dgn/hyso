# 13.5.6 — HYSo Studio Reborn kabul raporu

## Sonuç ve kapsam

Studio'nun 12 ekranı işlevsel olarak denetlendi; kurulum, açık başlatma, gerçek iş geçmişi ve okunabilir sonuç düzeni yeniden kuruldu. Özellikle yapısal doğrulama ile forward/backward/state_dict çalıştırması artık ayrı akışlardır. Kabul, gerçek Chromium + production StudioServer + HTTP + Core zincirinde yapıldı. Kullanıcı SQLite'ına fixture yazılmadı.

Branch `feature/hyso-models-next`, HEAD `d325c613caa2718ea4010bffbe7560219707843d`. Commit, push, stage işlemi veya 13.5.7 çalışması yapılmadı. Uzun eğitim, CUDA/native kabul suite'i başlatılmadı.

8765 kullanıcı sunucusu otomatik durdurulmadı. Kullanıcı kendi terminalinde yeniden başlattıktan sonra **canlı Reborn geçişi de doğrulandı**: workflow_contract **3**, dört JavaScript/CSS kaynağı güncel dosyalarla birebir aynı, ana başlık HYSo Studio. Gerçek canlı Chromium'da ana sayfa, modeller, yeni test akışı, evaluation ve Bilgi Merkezi açıldı. Page/HTTP hatası yoktur. **3 model / 9 mantıksal kayıt** kontrol öncesi ve sonrasında birebir aynıdır. Canlıda yalnız allocation-free preview POST'u oluştu; test/eğitim işi veya ayar yazılmadı. Kanıt: `live-reborn-after-restart.json` ve `live-reborn-*.png`. A–H yürütme kabulü kullanıcı verisinden ayrı izole sunucularda yapıldı.

## Ön denetim ve yeni düzen

[On iki ekran matrisi](../guides/hyso-studio-reborn-audit.md) her sayfanın kullanıcı görevini, gerçek endpoint'ini, girdisini ve iş/sonuç ilişkisini listeler. Değişiklik öncesi ekranlar `before-*.png` ve `before-screen-audit.json` ile kaydedildi.

Temel, açıklama ve teknik bilgi seviyeleri ayrıldı. Her sayfada kısa görev açıklaması vardır. Teknik JSON/provenance paneli başlangıçta kapalıdır ve sayfalar arasında açık durumu taşınmaz. Emerald/jade, grafit, kırık beyaz ve bakır token sistemi korunarak 16px temel tipografi, 26px çalışma alanı aralığı ve 14px yüzey köşesi kullanıldı. Masaüstünde kurulum ve sonuçlar yan yana; 1200px altında alt alta gelir. Mobilde test kartları tek sütundur. Geniş tablolar kendi alanında kayar.

Ana sayfa HYSo Studio başlığı ve istenen ürün açıklamasıyla açılır. Oluştur/kayıtları aç/yükle/bileşenleri incele eylemleri gerçek sayfalara bağlıdır. Model, doğrulanmış sürüm, test kaydı, eğitim kaydı ve mimari karşılaştırma sayıları SQLite'tan gelir. Son test/eğitim/karşılaştırma özetleri tam sonuç JSON'unu ana sayfaya taşımaz. Açık mimari karşılaştırma eylemi ad, zaman ve referans kaydı tutar; bunu ölçüm sonucu saymaz.

## Çözülen gerçek sorunlar

- Kullanıcının tarif ettiği “test seçtim, yalnız doğrulama geçmişi görüyorum” durumu gerçek tarayıcıda önce structure-only çalıştırılarak yeniden üretildi. Bu `validate` işi runtime sonucuna dönüşmez. Ardından aynı modelde üç runtime seçeneği ayrı `test` işi oluşturdu ve üç gerçek sonuç kartı gösterdi. Başlangıçta örtük test seçilmez; setup kontrolü iş başlatmaz.
- Runtime işi seçilen testleri ve gerçekten gözlenen phase/case durumunu saklar. İş kartında kimlik, model, başlama zamanı ve durum görünür. Tamamlanan iş saklanan sonuca bağlanır. Başarısız iş gerçek backend nedenini gösterir; bekleyen/çalışmamış case PASS sayılmaz.
- İlgisiz iş tipleri her ekranın geçmişini doldurmaz. Test sayfasında yapısal geçmiş ayrı kapalı bölümde, runtime geçmişi ve retained test sonuçları ayrı bölümlerdedir.
- Core kanal/bağlantı nedeni temel görünümde açıklanır. Hatalı katmana git düğmesi canvas'ı gerçek node'a ortalar ve inspector'ı açar. Model veya koordinatlar bu odak eylemiyle değiştirilmez.
- Geçersiz karşılaştırmada açıklamasız parametre sayısı yerine gerçek Core tanısı ve düzeltme eylemi görünür. İki kesin değer olmadan kesin fark üretilmez. Tamamen boş maliyet satırları tablodan çıkarılır; mevcut olmayan maliyetler kompakt gerekçe alanındadır.
- Evaluation ham metrik yolları yerine Accuracy/Precision/Recall/F1 ve ortalama türlerini gösterir. Sayılar beş ondalığa kadar okunabilir biçimde sunulur; kaynak değeri değiştirilmez. Verilen tahminlerin puanlandığı ve model inference yapılmadığı açıkça yazılır.
- CPU profilinde boş CUDA kartları ana sonucu kaplamaz. Ölçülmeyen alanlar gerekçeli kapalı bölümdedir. Gerçek benchmark kaydındaki hız, parametre ve mantıksal depolama değerleri gösterilir. Aktivasyon maliyeti aynı config hash'ine bağlı Core tahminidir; batch/birim ve tahmini durumuyla sunulur.
- Rapor önizlemesi ham JSON ile başlamaz; model, kontroller, metrikler ve tavsiyeler okunabilir olarak çizilir. HTML/Markdown/JSON dışa aktarımı korunur. Başarısız rapor işi gerçek hata nedenini taşır.
- Workflow 3 ve API/Studio sürümleri doğrulanmadan frontend POST göndermez. Statik dosyalar süreç başlangıcında sabitlenir. Böylece eski Python ile yeni JS'in sessizce karışmasına karşı önceki koruma sürer.

## Gerçek kullanıcı kabulü — A–H

İzole başlangıç modeli **Reborn acceptance A**: schema3, `image=[1,2,2]`, `conv_norm_act` stem (8 kanal, kernel 1, padding 1, batch norm, SiLU), iki sınıflı classification head; adlandırılmış çıktı `predictions`. Kullanıcı ana sayfadaki oluştur eylemiyle modeli kurdu, kernel değerini düzenledi ve kaydetti. Fresh seed-0 CPU başlangıcı kullanıldı.

| Senaryo | Kullanıcı eylemi / gerçek HTTP | Backend / sonuç | Kullanıcının gördüğü sonuç ve kanıt | Kabul |
| --- | --- | --- | --- | --- |
| A | Oluştur, düzenle, `/api/models`; `/api/preview`; yapı seç → `/api/job-preflight` → `/api/jobs`; sonra forward/backward/state_dict seç ve ayrıca başlat | Önce `validate`: retained runtime sonucu yok. Sonra `test`: tamamlandı; gerçek phase complete ve üç case | Structure-only geçmişi ayrı; Forward **PASS**, Backward **PASS**, State Dict **PASS**. `reborn-A-structure-only.png`, `reborn-A-execution-results.png` | Başarılı |
| B | Aynı modelde iki gerçek JSON örneği ve `[0,1]` hedefi yükle, head/loss ve ayarları kontrol et; `/api/job-preflight` → `/api/jobs` | `train`, CPU, 1 epoch, batch 2, Adam, cross_entropy; tamamlandı | Gerçek epoch 1, train loss **0.7057975531**, train accuracy **%50**; validation verisi yok ve validation eğrisi uydurulmadı. `reborn-B-training.png` | Başarılı |
| B ek profil | Aynı model ve aynı features dosyasıyla kontrol et ve açık ölçüm başlat | `benchmark`, CPU fp32, 1 warmup, 3 ölçülen iterasyon, 1 tekrar | Latency, kaynak koşulları, gerçek parametre/depolama, türetilmiş hız ve Core aktivasyon tahmini. `reborn-B-performance.png` | Başarılı |
| C | `targets=[0,1,1]`, `predictions=[0,1,0]`, sınıflar a/b, qa-test split yükle; setup ve ayrı `/api/jobs` | `evaluate`, gerçek classification evaluator tamamlandı | Accuracy **2/3**, macro precision **0.75**, recall **0.75**, F1 **2/3**. Inference yapılmadığı görünür. `reborn-C-evaluation.png` | Başarılı |
| D | A'yı kopyala, head sınıf sayısını 3 yap, B olarak kaydet; `/api/architectures/compare`; açık kesin sayım düğmesi → iki `/api/jobs` | İki `build`, `parameter_count_only=true`; gerçek nn.Module numel, forward yok | İki farklı kayıtlı mimari: **42 Exact**, **51 Exact**, fark **+9**. Kartlar/tablo otomatik güncellendi. `reborn-D-exact-comparison.png` | Başarılı |
| E | Kopyada stem in_channels=9 yap; `/api/preview`; Review layer stem; 1 olarak düzelt | Core kanal sözleşmesi hatası, düzeltmeden sonra valid | Hata inspector'daki gerçek alana götürdü; 9 değeri görünürken görüntü alındı, 1 sonrası hata kalktı. `reborn-E-invalid-node.png` | Başarılı |
| F | Katalogda Conv2D ve SE seç, parametre açıklamalarını aç, teknik seviyeyi seç | `/api/catalog`, gerçek implementasyon kaynaklı içerik | Açılan başlıkların açıklama paragrafları boş değil; teknik JSON ayrı kapalı. `reborn-F-conv2d.png`, `reborn-F-se.png` | Başarılı |
| G | İngilizce gerçek işlem akışı; Türkçe geçiş, iki tema ve 1920×1080 / 390×844 gezinme | Gerçek ayar POST'ları; sayfa/dil/tema ve model bağlamı korunur | 12 ekran + teknik knowledge varyantının **52** ekran görüntüsü; sayfa yatay taşma kontrolü geçti. `v2-{route}-{theme}-{device}.png`, dört `contact-*.jpg` | Başarılı |
| H | İzole gerçek HTTP sunucusunun contract değerini 2 yap; UI'yi yükle; kayıt eylemini DOM üzerinden de zorla | Frontend **sıfır POST**, sıfır job; kaydedilmiş model aynen kaldı | Eski kod ve doğru restart talimatı görünür. `reborn-H-contract-mismatch.png` | Başarılı |

`reborn-journey.json` gerçek POST yolları ve payload'ları, sekiz tamamlanan işin UUID'leri/model bağlamları ve overview sonucunu içerir. Header/token kaydedilmez. `reborn-retained-results.json` aynı gerçek test workspace'inden salt okunur alınmış dört sonucu saklar. Son iş/ölçüm özeti ayrıca `acceptance-summary.json` içindedir. Bunlar kullanıcı performansına ilişkin üretim sonucu değildir.

Son QA benchmark'ı: ortalama **0.33010 ms**, P95 **0.38533 ms**, türetilmiş sıralı hız **6058.77007 örnek/s**, **42 parametre**, **240 byte mantıksal parametre+buffer depolaması**. Core statik batch-1 aktivasyon tahmini **138 eleman**; benchmark girdisi batch-2'dir ve bu iki kapsam UI'de ayrıdır. Bu kısa ölçüm makine/yük/protokole bağlıdır; genel performans iddiası değildir.

Bu kabulde yeni workspace: 2 model, 0 doğrulanmış sürüm, 1 runtime test kaydı, 1 eğitim kaydı, 1 mimari karşılaştırma. Sürüm oluşturma/yeniden açma, gerçek Python export çalıştırma, sürükleme/pan/zoom/undo/redo, beş model sınırı ve gerçek öneri kopyası diğer Chromium testlerinde doğrulandı. Tam E2E listesi `browser.xml` içindedir.

## Test, paket ve koruma

| Grup | Başarılı |
| --- | ---: |
| Studio backend, frontend mantığı, workflow ve UX | 162 |
| Gerçek Chromium | 16 |
| Schema1/2/3 ve Core builder/CLI/component regresyonu | 676 |
| Packaging ve commercial dağıtım | 10 |
| Tekrarlı koşular hariç toplam | **864** |

Son UI düzenlemesinden sonra Chromium grubu tekrar koşuldu. Son wheel/sdist içerik kontrolü ayrıca **3** testle tekrarlandı; bu üç test toplamda yeniden sayılmaz. Ruff ve üç frontend modülünün Node syntax kontrolü geçti. Mevcut Trainer GradScaler deprecation uyarıları vardır; Core bu görevde değiştirilmedi.

Wheel/sdist **167 ticari Python modülü + 5 statik kaynak = 172** kaynak içerir. Son arşivler bu 172 dosyanın güncel baytlarıyla karşılaştırıldı; hash'ler ve arşiv konumları `archive-proof.json` içindedir. Yeni runtime bağımlılığı yoktur. Yeni `/api/overview` ile 19 farklı API yolu vardır; kapalı yazma sözleşmeleri korunur.

**47/47 forensic/debug, 71/71 native, 6/6 dış checkpoint/kabul dosyası** koruma manifestiyle hash eşleşir. Başlangıçta mevcut yedi yetkili Studio entegrasyon değişikliği dikkate alınarak **651 tracked dosyada bu görev kaynaklı ek değişiklik yoktur**. Core, Registry, GraphModelBuilder ve ModelBuilder'a yazılmadı. P01–P35 kanıtları değiştirilmedi.

Kullanıcı workspace'inde **3 model / 9 mantıksal kayıt** vardır. Önceki yedekteki yedi kayıt birebir korunur. Ek iki `validate` işi 20:18 UTC'de, Reborn başlangıç manifestinin 20:24:42 UTC zamanından önce oluşturulmuştur; bu çalışmanın QA işleri değildir. `protection-check.json` bu ayrımı kaydeder. Kullanıcı verisine düzeltme, temizlik veya fixture ekleme yapılmadı; mevcut geçersiz modeller sessizce değiştirilmedi.

QA kanıtlarının tamamı repo/kullanıcı workspace'i dışında `%LOCALAPPDATA%/hyso-stage13-5-6-reborn` içindedir. Browser audit dosyalarında Chromium sürümü, izole origin, page errors ve HTTP errors bulunur. Son kabulde bunlar boştur. Görüntüler ve dört contact sheet görsel olarak incelendi; bu bir tam WCAG sertifikası değildir.

## Değişen dosyalar

- `hyso/studio/service.py`: gerçek overview ve açık karşılaştırma etkinlik kaydı.
- `hyso/studio/jobs.py`: seçilen runtime testler ve gerçek phase/case gözlemleri.
- `hyso/studio/server.py`: overview GET ve workflow 3.
- `hyso/studio/static/app.js`, `app.css`, `index.html`: sayfa düzeni, test çalışma alanı, tanı eylemleri, anlamlı durum/metrik sunumu ve sürüm engeli.
- `tests/test_studio.py`, `test_studio_ux.py`, `test_studio_browser.py`: HTTP overview/contract, gerçek overview/progress, structure/runtime ayrımı, A–H kabulü ve görsel matris.
- Beş rehber: `hyso-studio.md`, `hyso-studio-design-system.md`, `hyso-studio-local-api.md`, `hyso-studio-model-comparison.md`, `hyso-studio-test-selection.md`.
- Yeni `hyso-studio-reborn-audit.md` ve bu rapor.

Bu görev 14 mevcut Studio dosyasını değiştirdi, iki belge ekledi. Önceden mevcut untracked/dirty kullanıcı çalışmaları korunmuştur; git add/commit/push yapılmamıştır.

## Teknik sınırlar ve kullanıcı oturumuna geçiş

Kısa işlemler CPU/fp32/fresh initialization kullanır. Eğitim sonrası ağırlıklar sonraki sentetik test veya benchmark'ın eğitilmiş checkpoint'i değildir. Testler sentetik zero-input eval fixture'ıdır; genel doğruluk veya her parametreye gradient garantisi vermez. Evaluation verilen tahminleri puanlar. Multi-input/multi-loss eğitim ve regression/embedding evaluation mevcut profilde desteklenmez. Güvenli kısa CPU sayımı <=1M Core üst sınırını kullanır. Aktivasyon tahmini gerçek RSS/VRAM ölçümü değildir; benchmark türetilmiş sıralı hızdır, sürdürülebilir uygulama FPS'i değildir.

Canlı sunucunun yeni sürümü yüklemesi için önce kaydedilmemiş taslağı koruyun; sunucunun kendi terminalinde Ctrl+C, ardından:

```powershell
& 'C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe' -m hyso.studio --port 8765
```

Sayfayı yenileyin. Bu yeniden başlatma kayıtları silmez. Üretim sürecini otomatik sonlandırma denenmedi. Kullanıcının yeniden başlatması sonrası Reborn geçişi ve veri korunması doğrulanmıştır; açık yeniden başlatma işi kalmamıştır.
