# Aşama 13.5.6 — Studio UX V2 doğrulama raporu

> Güncel kritik stabilizasyon ve kullanıcının yeniden başlatmasından sonra doğrulanan canlı sunucu geçişi için [stabilizasyon raporuna](stage13-5-6-stabilization.md) bakın. 8765 artık workflow_contract=2 sunuyor; 3 model ve 7 SQLite kaydı korundu. Önceki test sonuçları tek başına bellekte eski Python koduyla çalışan oturumun güncel olduğunu kanıtlamıyordu.

Tarih: 13 Kasım 2026. Bu rapor önceki 385 testlik raporun yerine geçer; eski rapor QA klasöründeki `previous-report.md` dosyasında korunmuştur. Aşağıdaki toplamlar yalnız son test gruplarına aittir; ara koşular toplama eklenmemiştir.

## Sonuç ve güvenlik

Studio'ya açıklayıcı varsayılan genel bakış, jade/bakır tasarım sistemi, yeniden taşınabilir graph düğümleri, kesin CPU parametre sayımı, ayrıntılı mimari karşılaştırma, mekanizmaya özgü Bilgi Merkezi, seçilebilir CPU kontrolleri ve Core kanıtına bağlı tanılar eklendi. Kullanıcı verisi veya ölçülmemiş performans sonucu üretilmedi.

| Alan | Sonuç |
| --- | --- |
| Branch | `feature/hyso-models-next` |
| Başlangıç / son HEAD | `d325c613caa2718ea4010bffbe7560219707843d` |
| Staged alanı | Boş |
| Önceki görev manifesti | Başlangıçta 36/36 eşleşti |
| Başlangıçtaki tracked dosyalar | 651/651 bu oturumdaki başlangıç baytlarıyla aynı |
| Önceki Studio kapsamı + iki yeni rehber | 38 görev dosyası |
| Bu oturumda değişen / eklenen | 21 dosya |
| Commit / push / reset / restore / clean | Yapılmadı |
| 13.5.7 | Başlatılmadı |

Mevcut kirli dosyalar ve önceki Studio çalışmaları korundu. Core Builder/Graph/Registry kaynakları bu UX V2 oturumunda değiştirilmedi. Schema1/2/3, ModelBuilder ve GraphModelBuilder sözleşmeleri korundu. Testler yalnız ayrı geçici workspace'lerde çalıştı; kullanıcının varsayılan SQLite workspace'i test amacıyla açılmadı veya yazılmadı. Çalışan bir kullanıcı sürecinin kendi workspace hareketleri için bayt eşitliği iddia edilmez.

## Ekranlar ve kullanıcı akışları

| Ekran | Doğrulanan davranış |
| --- | --- |
| HYSo — Mühendislik Çalışma Alanı | Varsayılan `#overview`; HYSo'nun amacı, yerel işlem, tahmin/ölçüm ayrımı, dört gerçek eylem, gerçek sayımlar ve altı sayfa bağlantısı |
| Modellerim | Ayrı `#home`; kayıtlı taslaklar, değiştirilemez sürümler, gerçek son işler |
| Model Mimarı | Düğüm/port/branch/skip/fusion, yeniden taşıma, bağlantı düzenleme, duplicate/delete/rename, history, kaydet/sürüm ve gerçek Python |
| Bileşen Bilgi Merkezi | Arama/kategori, mekanizma kartları, seçili detay, okunabilir parametreler |
| Bileşen teknik görünümü | Ayrı matematik, giriş/çıkış, syntax-highlight Python/kopyalama ve kaynak kartları; Ham JSON kapalı |
| Eğitim ve Deneyler | Model/veri/çıktı/loss ve açıklamalı eğitim ayarları; kontrol et → ayrı başlat → izle → gerçek sonuç |
| Test ve Doğrulama | Yapı, forward, gözlenen gradient ve serialization seçimi; açık plan ve gerçek sonuçlar |
| Performans Profili | Amaç/koşul/yorum açıklaması, model ve profil işi seçimi, gerçek metrik kartları ve kayıtlar |
| Model Değerlendirme | Desteklenen classification prediction/target dosyası ve ayrı kontrol/başlat akışı |
| Model Karşılaştırma | 2–5 kayıtlı mimari; üst özet/kategori kartları, ayrıntılı tablo, ayrı gerçek sonuç modu |
| Öneriler | Modele özgü Core tanısı, somut sonraki adım, downstream etkiler, isteğe bağlı teknik kanıt |
| Raporlar | Model/sürüm ve gerçek kayıt bağlamında TR/EN JSON/HTML/Markdown |
| Proje Ayarları | Kalıcı tercihler, light/dark/dil ve salt okunur Core/ticari politika |

Ana akış: Model oluştur → Doğrula → Eğit/Test Et → Ölç → Karşılaştır → Raporla. Ana sayfadaki Model Yükle gerçek dosya seçiciyi açar ve yüklenen modeli Model Mimarı'na taşır. Hiç kayıt yoksa istenen ilk CNN modeli oluşturma mesajı ve gerçek sıfır sayımlar gösterilir; dekoratif performans grafiği yoktur.

Model Mimarı'ndaki eylemler model bağlamını taşır. Diğer ekranlarda aynı altı düğmeyi tekrar göstermek yerine seçili model çubuğu ve gerçek ana navigasyon kullanılır. Kayıtlı modeller model kimliği + draft revision, sürümler version ID, kaydedilmemiş işler ayrı `draft/kimlik` snapshot kaynağıyla filtrelenir. Aynı mimari hash'i farklı model veya snapshot koşusunu yeniden adlandırmaz. Snapshot çalıştırmak otomatik model kaydı yapmaz. Eski bağlamsız API kayıtları genel kayıt/sonuç karşılaştırma yolunda korunur.

## Tasarım sistemi ve görsel QA

Merkezî CSS token'ları koyu jade/zümrüt, kontrollü bakır, sıcak kırık beyaz ve grafit yüzeyler kullanır. Başarı/uyarı/hata token'ları ayrıdır. HY marka işareti korunur. Fontlar, formlar, kontrol boyutları, kart boşlukları ve büyük masaüstü alanı yeniden düzenlendi. Kapsam rozetleri 12 px'e çıkarıldı. Harici şablon/font/arayüz bağımlılığı eklenmedi.

13 görünümün 1920×1080 masaüstü ve 390×844 mobil genişlikte, dark/light temalarda **52 tam sayfa görüntüsü** alındı. Dört contact sheet ile bütün düzenler; ayrıca ana sayfa, graph, eğitim, test, teknik Bilgi Merkezi ve karşılaştırma tam çözünürlükte incelendi. Görsel inceleme sonucunda ovalleşen portlar, dar mobil akıştaki tekrarlı model eylemleri, çok uzun genel bakış başlığı ve küçük rozetler düzeltildi. Tüm matris adımlarında yatay sayfa taşması kontrolü geçti. Geniş tablo kendi alanında kayar; mobil navigasyon ve bileşen kartları yatay kaydırılabilir. Inspector mobilde alt paneldir.

Klavye focus/Enter/ok hareketleri ve reduced-motion korunur. Bu, hedefli erişilebilirlik kontrolüdür; tam WCAG, ekran okuyucu veya çoklu tarayıcı sertifikasyonu değildir.

İki temadaki 16 temel metin/yüzey ve ana düğme token çiftinin kontrastı ayrıca hesaplandı; en düşük oran 5,72:1 oldu. Tam değerler `token-contrast.json`, görsel inceleme notları `visual-review.md` içindedir. Native dosya seçici metinleri tarayıcı/işletim sistemi dilini izler.

## Graph yeniden taşıma

Gerçek Chromium testi aynı düğümü yerleştirdikten sonra birden çok kez taşır; zoom değiştirir, arka planı pan yapar ve yeniden sürükler. Beklenen model koordinatı ekran farkı / gerçek transform ölçeğiyle karşılaştırılır. Pan model konumunu değiştirmez. Pointer up/cancel/lost capture yolları sonlandırılır. Tekerlek zoom'u işaretçi konumuna bağlıdır; sürükleme sırasında tekerlek hareketi geometriyi değiştirmez.

Inspector sürükleme başında açılarak canvas'ı kaydırmaz; tamamlanan tıklamada açılır. Tek taşıma bir history snapshot'ıdır. Undo/redo, düğüm seçimi ve kayıt/yükleme doğrulandı. SQLite tam kayan nokta koordinatını saklar; DOM CSS gösterimi yuvarlayabildiği için tarayıcı karşılaştırmasında küçük tolerans kullanılır. Sabit sürümün düğüm konumu pointer veya ok tuşuyla değiştirilemez; viewport pan/zoom kullanılabilir.

Port bağlantıları ve sıralı inspector girişleri edge düzenleme yoludur. Dallar, harici skip ve fusion normal Core graph düğüm/bağlantılarıdır. Frontend shape engine eklenmedi.

## Parametre sayımı ve karşılaştırma

Kök neden: Graph Core, bazı eski mekanizmalarda kesin statik sayaç döndürmez; `_describe` sonucu `None` toplam kesin sayımı da bilinmeyen yapar. Buna karşılık `_cost` korumacı üst sınır sağlayabilir. Önceki Studio karşılaştırması sayı bulunmadığında dahi durumu ESTIMATED olarak etiketliyordu.

Artık `EXACT`, `ESTIMATED`, `UNAVAILABLE` ayrıdır. Kesin Core sayacı varsa kullanılır; yalnız üst sınır varsa etiketli tahmin ve eksik sayaç adları gösterilir; çözülmeyen ön doğrulamada gerçek neden korunur. Sayı uydurulmaz.

**Kesin CPU parametre sayımı** açık kullanıcı düğmesiyle seçili kayıtlar için tek worker kuyruğunda gerçek build başlatır. Yalnız Core parametre üst sınırı <=1.000.000 olan modeller kabul edilir. Gerçek CPU fp32 `nn.Module.parameters()` üzerinden `numel()` toplanır. İleri hesaplama, CUDA, optimizer veya eğitim çalıştırılmaz. Yaklaşık 4 MiB fp32 parametre depolaması sınırı buffer/oluşturma ek yükünü kapsamaz; süreç RSS veya OS zaman sınırı garantisi değildir. Başarısızlık nedeni ilgili kartta kalır. Sonuç job ID, config hash, reference kind/ID ve revision'a bağlanır; eski revision veya başka modelde kullanılmaz.

60 küçük graph mekanizma örneğinin üretilen Python kodu gerçekten çalıştırıldı. Gerçek modül toplamı varsa Core kesin sayacıyla eşit, Core tahmininden küçük/eşit bulundu. Legacy SE için açık CPU sayımı ve cache/revision izolasyonu ayrıca test edildi. Büyük parametre bütçesi build öncesinde reddedildi; CUDA başlatma yasağı test edildi.

Karşılaştırma giriş/çıkış, model/sürüm, node/branch, fusion/head türleri, kategori dağılımı, residual/attention metadata'sı, parametre durumu/dayanağı/farkı, activation/attention, build/eğitim uygunluğu ve bütçe yakınlığını içerir. Kategori meter'ları gerçek mimari sayılarını kullanır. Parametre farkı yalnız iki sayı da kesin olduğunda hesaplanır. Eşdeğer hash açıkça belirtilir; farklı giriş geometrisi maliyet uyarısı üretir. Sonuç modu 2–8 gerçek kaydı ve gerçek koşul denetimini korur. Eğitilmemiş mimariye accuracy/latency eklenmez.

## Bilgi Merkezi ve parametre eğitimi

61 ticari mekanizma, constructor/forward ve Core geometri kaynaklarına göre ayrı `AUDIT` notları taşır. Kategoriye göre aynı attention/residual/pooling/normalization sınırlaması kopyalanmaz. SE, CBAM, ECA, Coordinate, spatial/channel gating ve spatial self-attention ayrı açıklanır. SE global kanal özeti + daraltılmış MLP kullanır; `(H*W)²` QK matrisi oluşturmaz. Token/matris bütçeleri ilgili self-attention mekanizmalarına aittir.

Her mekanizma amaç/işlem, kullanım, sınır, hesaplama maliyeti, fark ve kaynak metadata'sı taşır. Detaylar ortak cümleleri yeniden yazdırmak yerine görev kartlarında gruplanır. Parametre varsayılanı, seçimler ve sınırlar Core'dan gelir; tamsayı/aralık/listeler okunabilir cümlelere çevrilir. Reduction Ratio — Daraltma Oranı açıklaması mekanizmanın gerçek `max(1 veya 8, C//reduction)` ara genişliğine dayanır. Yanlış değer riski ve yeniden doğrulama ihtiyacı açıklanır.

TEMEL görünüm matematik veya JSON gerektirmez. TEKNİK görünüm dört ayrı açıklama/kod/kaynak kartı ekler. Ham JSON ayrı ve kapalıdır; teknik seviye seçimi bunu açmaz. Python örnekleri syntax highlight ve kopyalama içerir. 60 graph örneğinin gerçek CPU forward/backward ve parametre kontrolleri geçti; complete modular_cnn gerçek Registry örneğiyle sunulur. Harici makale bağlantısı eklenmedi. `knowledge-source-audit.json` 61 mekanizmanın kaynak ve örnek hash'lerini kaydeder.

## Test seçimi, eğitim ve ölçüm

| Seçim | Gerçek kapsam |
| --- | --- |
| Yapı doğrulama | Tensor/modül oluşturmayan Core bağlantı, şekil, parametre ve kaynak kontrolü |
| Forward | Yeni CPU modelinde sentetik sıfır girdilerle sonlu çıktı kontrolü |
| Backward/gradient | Eval çıktı toplamının türevi; yalnız gözlenen gradientlerin sonluluğu; gradient yoksa SKIP |
| State dict | Gerçek ağırlıkları ikinci CPU modülüne strict yükleme ve aynı fixture üzerinde eşit eval çıktısı |
| Veri uyumluluğu | Gerçek dosyayla Eğitim/Performans kurulum kontrolüne bağlanır |
| Training readiness | Tek giriş/uyumlu classification head/loss ve TrainingSession ayarları için ayrı kurulum |
| Kayıt inceleme | Gerçek retained JSON/JUnit PASS/FAIL/XFAIL/SKIP ve kaynak mesajları |

Çoklu runtime seçim tek güvenli CPU işinde sıralanır. Forward, backward/serialization için açıkça anlatılan ön koşuldur. Yalnız seçili runtime kontrolleri case üretir; süreler ayrı ölçülür. Case toplamı tüm hazırlık maliyetini temsil ettiği iddiası taşımaz. Başarısız yürütme gerçek job hatasıdır, sahte PASS üretilmez. Seçim/ayar değişirse tekrar kontrol gerekir. Kontrol adımı iş oluşturmaz; başlatma ayrı kullanıcı eylemidir. P01–P35 kabul suite'i çağrılmaz.

Kısa CPU profili en çok 1M parametre üst sınırı, 1M aktivasyon tahmini, 65536 veri/fixture elemanı, 32 örnek/batch ve eğitimde 1–5 epoch ile sınırlıdır. Eğitim fresh initialization kullanır. Tek giriş ve tek seçili classification loss/head desteklenir; tüm graph çıktıları hesaplanır. Benchmark CPU fp32, 1 warmup + 3 timed iteration + 1 repeat kullanır. Sonuç kartları yalnız gerçek kaynaklardan gelir; MEASURED/ESTIMATED/UNAVAILABLE ve imported-unverified ayrımı korunur.

## Önerilerin kök neden kalitesi

Tanı açıklaması diagnostic code + orijinal Core mesajı + node ID + gerçek parametreler + expected/actual + Core giriş şekline dayanır. Reshape eleman çarpımı, PixelUnshuffle bölünebilirliği, vektör başlığın pooling=none gereksinimi ve in_features uyumsuzluğu ayrı açıklanır. Flatten terminal kısıtı yalnız gerçek schema2 tanısı varsa anlatılır; geçerli graph Flatten→NC destekleyen katman zinciri yanlış reddedilmez. Yanlış/geçerli örnekler test edildi.

Graph kenarlarından erişilebilen ve Core tarafından blocked işaretlenen downstream düğümler kök kartta listelenir. Bağımsız iki hata ayrı kartta kalır. Otomatik onarım doğrulanmamışsa bu açıkça belirtilir. Kanal düzeltmesi yalnız bağımsız kopya tüm Core preflight'tan geçtiğinde yeni aday taslağı açar. Katmana git ve teknik kanıt eylemleri gerçek bağlamı kullanır. Diğer tanılarda kanıtın taşıdığından daha özel neden uydurulmaz.

Mevcut kanıta bağlı training loss/learning rate/overfitting hipotezleri, gradient, test regression ve karşılaştırılabilir native latency kuralları korundu. Kural eşikleri ve kaynak kayıtlar teknik kanıttadır. Kayıt yoksa öneri üretmemek geçerli sonuçtur. Ölçülmüş nedensel kazanç veya genel model kazananı iddia edilmez.

## Son testler

| Grup | Sonuç |
| --- | --- |
| Studio backend/frontend/workflow/UX | 161 passed |
| Gerçek Chromium 143.0.7499.4 | 13 passed |
| Core graph/validation/builder/CLI/components/advanced/sampling-heads | 676 passed |
| Ticari dağıtım / wheel / sdist | 10 passed |
| **Toplam** | **860 passed; 0 failure/error/skipped** |

Chromium production HTTP sunucusu, gerçek DOM/pointer/file chooser/form ve isolated SQLite kullanır. 13 browser audit kaydında JavaScript page error ve HTTP hata sayısı sıfırdır. Testler DOM benzetimi değildir. Yeni kapsam: default overview/gerçek boş sayımlar/yükleme; tekrar drag/zoom/pan/history/konum roundtrip; çoklu test seçimi/no-tensor preflight; teknik kart/kapalı JSON; exact CPU count; bütün ekranların iki tema/iki boyut matrisi. Önceki dokuz gerçek browser akışı da korunur.

Core grubu önceki dört dosyaya ek olarak component, advanced ve sampling/head testlerini içerir. Bu nedenle eski 224 ve yeni 676 birleştirilmez. Trainer kaynaklı mevcut GradScaler deprecation uyarıları vardır; korunan Trainer değiştirilmedi. Ruff, üç Node syntax kontrolü ve staged/unstaged diff whitespace kontrolü geçti. Uzun eğitim, CUDA veya native kabul/benchmark çalıştırılmadı.

## TR/EN, API ve ticari paket

| Alan | Yeniden ölçülen |
| --- | --- |
| Ticari / geliştirme Registry | 57 / 85 |
| Ticari mekanizma | 61 |
| İki dilli parametre alanı | 225 (222 Registry + 3 fusion) |
| UI anahtarları | 443/443, yinelenen anahtar yok |
| Capability inventory | 161 |
| Ticari Python modülü | 167 |
| Statik web kaynağı | 5 |
| Paket kaynak eşleşmesi | 172/172 |
| API | 18 path; 10 GET + 13 POST; 13 request schema |

Yeni endpoint eklenmedi. Build işine sınırlı parameter_count_only, test işine selected tests, model_ref'e açık unsaved draft snapshot ve kayıtlara model_revision filtresi eklendi. Kapalı alanlar, loopback Host/Origin, token, body/queue sınırları ve tek worker korunur. Kamuya açık Core API değişmedi.

Wheel ve sdist ayrı kurulumlarda discovery/typing/CLI ve küçük gerçek CPU smoke kontrollerinden geçti. Her arşivdeki 167 Python + 5 statik kaynak checkout baytlarıyla eşleşti. Commercial allowlist ve M3/M4 dışlamaları korundu; test/docs/checkpoint/SQLite/kullanıcı verisi paketlenmedi. Arşivlerin yolları ve SHA-256 değerleri QA `archives.json` dosyasındadır.

## Korunan SHA-256 kümeleri

| Küme | Eşleşme | Aggregate SHA-256 |
| --- | --- | --- |
| Forensic/debug | 47/47 | `fb14a0783d295a32c15ff402d7e70abdb297e0fcb63dcb9daa7d590fde26ca7c` |
| Native kaynak | 71/71 | `3cc8c2de30a6f61e8778ff8eb23c0d0c0ec78aaee168d69432b4326426efc0ab` |
| Dış checkpoint/kabul/index | 6/6 | `578f0e51c7a902044f1e6a554d9139802b266b50474216e687149c9f250e0451` |

Aggregate sıralı compact UTF-8 JSON path→SHA-256 haritasının hash'idir. Başlangıç protection.json SHA-256: `0b0333571b9f92bfc4b669514de2869ed5a1a95029074e17b78ee650b6d973a5`. P01–P35 kanıtlarına yazılmadı. Eski manifest, başlangıç dosya hash'leri ve son doğrulama ayrı saklandı.

## Değişen dosyalar

- `hyso/studio/analytics.py`, `jobs.py`, `knowledge.py`, `service.py`
- `hyso/studio/static/app.css`, `app.js`, `index.html`, `presentation.js`
- `tests/test_studio_browser.py`, `test_studio_ux.py`, `test_studio_workflows.py`
- `docs/guides/hyso-studio.md`, `hyso-studio-model-editor.md`, `hyso-studio-component-knowledge.md`, `hyso-studio-model-comparison.md`, `hyso-studio-recommendations.md`, `hyso-studio-local-api.md`
- Yeni `docs/guides/hyso-studio-test-selection.md`, `hyso-studio-design-system.md`
- `docs/reports/stage13-5-6-capabilities.json`, `stage13-5-6-validation.md`

Tam 38 dosyalık görev hash manifesti ve bu oturumdaki 21 değişiklik `task-files.json` / `protection-verification.json` içindedir. Diğer önceki Studio dosyaları korunur.

## Açık sınırlar ve eklenmeyen yetenekler

- Kesin CPU sayımı yalnız küçük, Core bütçesiyle kabul edilen modellerde sunulur; daha büyük modeller tahmini/erişilemeyen durumunu ve nedenini korur. Otomatik veya her model için garanti edilen exact sayım yoktur.
- Statik activation/attention tahmini runtime RSS, allocator/workspace veya autograd belleği ölçümü değildir. Core maliyet sunmuyorsa bilinmiyor kalır.
- Backward kontrolü sentetik eval çıktı toplamı ve gözlenen gradientlerle sınırlıdır. Eğitim loss uyumunun ve her parametreye gradient ulaşmasının tam testi değildir.
- Full training/checkpoint resume, multi-input dataset, multi-loss, regression/embedding training/evaluation eklenmedi. Complete modular_cnn kısa runtime akışı somut girdi geometrisi olmadığından kapalıdır; Core build/JSON yolu korunur.
- Native/CUDA acceptance veya arbitrary external experiment-store çalıştırma/bağlama arayüzü yoktur. Aktif job iptali yok; queued cancel ve active drain vardır.
- Rapor biçimleri JSON/HTML/Markdown'dır; PDF/CSV eklenmedi. Her rapor düzeni için yayın kalitesinde ayrı çıktı denetimi yapılmadı.
- Bilinmeyen tanılara uydurma otomatik çözüm veya tüm olası training/performance sorunları için genel tavsiye üretilmez. Aday doğrulaması statiktir, ölçülmüş hız/kalite etkisi değildir.
- Görsel QA bir Chromium sürümüyle hedefli kontroldür; tam ekran okuyucu/WCAG/çoklu tarayıcı matrisi değildir.

## Kanıtlar ve bitiş

QA kökü: `%LOCALAPPDATA%/hyso-stage13-5-6-ux-v2` (repository dışında). Son XML: `studio-backend.xml`, `browser-final.xml`, `core.xml`, `packaging.xml`. Ek kanıtlar: 52 `v2-...-desktop/mobile.png`, dört contact sheet, 13 `browser-audit-...json`, `counts.json`, `knowledge-source-audit.json`, `source-snapshot.json`, `archives.json`, `initial-files.json`, `initial-protection.json`, `protection-verification.json`, `task-files.json` ve `verify-v2.py`.

Başlatma: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe -m hyso.studio --port 8765`. Varsayılan adres `http://127.0.0.1:8765/` olur. QA sunucuları test fixture'ları tarafından kapatıldı. Kullanıcının mevcut sunucusu kapatılmadı. Branch/HEAD ve staged alanı korunur. Commit/push yapılmadı; 13.5.7'ye geçilmedi.
