> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Reborn ön işlev denetimi

12 ekran production StudioServer ve Chromium ile izole audit-workspace üzerinde açıldı. Başlangıç görüntüleri ve görünen kontroller `%LOCALAPPDATA%/hyso-stage13-5-6-reborn/before-screen-audit.json` içindedir. Core kaynaklarına veya kullanıcı SQLite'ına yazılmadı.

| Ekran / kullanıcı görevi | Backend / gerekli girdi | İş ve sonuç ilişkisi | Başlangıç sorunu / yapılacak düzeltme |
| --- | --- | --- | --- |
| Ana sayfa / başla | models, versions, jobs, records GET | Gerçek kayıt sayıları | Test/eğitim/karşılaştırma özeti eksik; SQLite kaynaklı görev özeti |
| Modellerim / kayıt aç | models/versions; kimlik ve revision | Kaydedilmiş model/sabit sürüm | Eylemler mevcut; açıklama ve kart hiyerarşisi yenilenecek |
| Model Mimarı / kur ve düzelt | preview, models, versions, snippet, build job; config | Core preflight ve model_ref | Hatalı katmana doğrudan odak, temel/ikincil kontrol hiyerarşisi |
| Bilgi Merkezi / öğren | catalog; kategori/bileşen | 61 mekanizma, 225 alan | İçerik var; kısa temel görünüm ve teknik kartlar korunacak |
| Eğitim / öğrenme çalıştır | job-preflight → train job; dosya, loss/head, ayarlar | Gerçek epoch/run ve retained sonuç | Uzun form; aşamalar gruplandırılacak, yalnız ilgili iş geçmişi |
| Test / çalışabilirliği denetle | validate veya test job; seçilen testler | Validate forward kanıtı değildir | En kritik açık: aynı geçmiş, seçimin amacı ve case durumları belirsiz; ayrı seçim/başlatma/sonuç |
| Performans / CPU ölç | benchmark job; model, features, output | Gerçek CPU koşulları ve retained metrik | Gerekçe/birim/ölçüm durumu gruplandırılacak |
| Evaluation / verilen tahminleri puanla | evaluate job; targets, predictions, class_names, split | Inference üretmez; supplied_predictions | Dosya sözleşmesi ve gerçek metrik kartları belirginleştirilecek |
| Karşılaştırma / farkları incele | architectures/compare; 2–5 referans veya comparison; retained ID'ler | Gerçek metadata/koşul denetimi | Boş maliyet satırları ve geçersiz model eylemi eksik; gerekçe + katmana git |
| Öneriler / kanıtı düzelt | recommendations; config + kaynak filtreleri | Kopyada doğrulanan aday, otomatik değiştirme yok | Nedeni/sonraki adımı öne çıkar; teknik JSON kapalı |
| Rapor / aktar | report job; model, gerçek kayıt filtreleri | JSON/HTML/Markdown korunur | Önizleme/tercih düzeni ve kayıt yok açıklaması |
| Ayarlar / tercih seç | settings; scope/owner/key/value | Desteklenen tercihler kalıcı | Salt okunur politika/yerel tercihler ayrımı; per-panel teknik görünüm |

Ortak sözleşme: model ID/revision veya version ID / ayrı draft snapshot korunur. 4xx iş tamamlandı anlamına gelmez. Bilinmeyen sayı yerine destek/ölçüm/doğrulama nedeni gösterilir. Son kabul matrisi ve gerçek A–H akışları [Reborn doğrulama raporundadır](../reports/stage13-5-6-reborn-validation.md).

## Son ekran denetimi

| Ekran | Kabul edilen davranış / gerçek kanıt |
| --- | --- |
| Ana sayfa | HYSo Studio başlığı, dört başlangıç eylemi, altı gerçek görev bağlantısı; beş SQLite sayacı ve son test/eğitim/karşılaştırma özetleri. Boş workspace sıfır gösterir. |
| Modellerim | Kaydı aç ve yeni model eylemleri; taslak/sürüm ayrımı, seçilen modelin diğer sayfalara taşınması. Gerçek yeniden açma/sürüm Chromium testi geçti. |
| Model Mimarı | Gerçek node taşıma, yeniden taşıma, pan/zoom, undo/redo, kaydedilmiş koordinatlar; Core kanal hatasından inspector'a odaklanıp düzeltme. Python export gerçek forward/backward ve strict load ile doğrulandı. |
| Bilgi Merkezi | Conv2D ve SE açıklamaları ve alanları açıldı; boş açıklama paragrafı yok; teknik mod JSON'u ayrı ve kapalı. Katalog 61 mekanizma / 225 alan. |
| Eğitim | Gerçek dosya → setup → ayrı train işi → tamamlanan epoch, loss/accuracy ve saklanan sonuç. Sağ çalışma geçmişi yalnız eğitim işlerini gösterir. |
| Test | Başlangıçta örtük seçim yok. Structure-only validate geçmişi ayrıdır, runtime sonucu üretmez. Aynı modelde üç seçili gerçek test üç ayrı PASS kartı oluşturur. Gerçek phase/case gözlemi; kalan testler bekliyor/atlandı. |
| Performans | Gerçek dosyayla CPU benchmark; ortalama/P95 gecikme ve türetilmiş işlem hızı, koşullar ve açıklama. Ölçülmeyen CUDA belleği ana kartlara girmez; nedeni açılabilir. |
| Değerlendirme | Prediction/target dosyasıyla gerçek accuracy, macro precision/recall/F1; okunabilir metrik adları ve beş basamaklı sunum. Model inference yapılmadığı görünür. |
| Karşılaştırma | İki farklı model; açık sayım eylemi gerçek 42/51 parametre ve +9 fark üretir. 2–5 model sınırı, aynı mimari uyarısı ve koşullu gerçek sonuç modu korunur. Geçersiz model katmanına git eylemi vardır. |
| Öneriler | Kanıt yoksa ölçüm başlatma eylemi; gerçek Core sorununda kopyada doğrulanmış aday açma. Tekilleştirme ve provenance teknik bölümde. Otomatik model değişikliği yok. |
| Raporlar | Seçilen model ve gerçek kayıtlardan rapor işi, okunabilir önizleme ve indirme; eksik metrikler ana sonuç kartlarını doldurmaz. HTML/Markdown/JSON backend kontrolleri geçer. |
| Ayarlar | Dil/tema kalıcı; desteklenen ayarlar ve salt okunur politika açıklaması. Teknik ayrıntı başka sayfaya taşınmaz. |

52 görüntü (12 ekran ve Bilgi Merkezi teknik varyantı × 2 tema × 2 boyut) gerçek Chromium'da tekrar alındı; sayfa genişliği 1920px ve 390px için yatay taşma denetimi geçti. Gerçek işlem ekranları ayrıca kaydedildi. Görsel inceleme boyutlandırılmış ekranlar ve dört tema/cihaz contact sheet üzerinden yapıldı. Bu sonuç tam erişilebilirlik sertifikası değildir.
