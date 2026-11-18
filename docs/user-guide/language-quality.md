# UX-02 — Dil ve anlam kalite kontrolü

Kapsam: HYSo 0.2.0 için 70 konu, her konuda tr/en/de/es metni ve P01–P35’in
tamamı. Dört dil aynı HTML’de bulunur. Bu belge çeviri inceleme kaydıdır;
ayrı bir dilde kullanıcı raporu değildir.

## Yöntem ve sınır

Metinler her dilin kendi cümle yapısıyla yazıldı. Türkçe kaynak açıklamalar;
İngilizce, Almanca ve İspanyolca karşılıklarla konu konu anlam açısından
karşılaştırıldı. Ardından ölçüm kapsamları ve sınırlamalar gerçek kaynaklar,
`analysis/feature_plan.py`, `reporting/catalog.py`, model Registry’si ve ilgili
rehberlerle tekrar eşleştirildi. Her konunun kaynakları paketlenmiş topic
registry’sinde kayıtlıdır.

Otomatik kontroller dil/alan tamlığını, karakterleri, sayısal sınırları ve
bağlantıları denetler. Bunlar tek başına çeviri doğruluğu sayılmadı. Aşağıdaki
anlam kontrolleri ayrıca yazar tarafından yapıldı. Bağımsız ana dili konuşan
editör incelemesi veya profesyonel çeviri sertifikası iddia edilmiyor.

## Anlam kontrolü

| Konu | Dört dilde korunan anlam ve yapılan kontrol |
|---|---|
| Model profilleri | Geliştirme M3/M4 aileleri commercial wheel’de varmış gibi anlatılmadı; klasik ve mobile isimler çevrilmedi. |
| EBO / EMA | Kaynak incelemesi EBO’nun blok değil ELO eğitim wrapper’ı olduğunu gösterdi; dört metin düzeltildi. EBO’nun otomatik CUDA seçimi ve açık `device='cpu'` uyarısı dört dilde mevcut. EMA bir model ailesi olarak sunulmadı. |
| Kurulum | Python ortamının aktive edildiği varsayılmadı. Windows örnekleri doğrudan `.venv/Scripts` kullanır; POSIX karşılıkları dört dilde açıklanır. `--no-deps` eksik bağımlılıkları çözmüş sayılmaz. |
| Builder / Graph | Tanım, CPU model allocation ve training ayrıdır. Graph çıktı seçimi diğer düğümleri budamış sayılmaz. Keyfi çoklu görev desteği üretilmedi. |
| Checkpoint / Resume | V1 varsayılanı ile açık Graph v2 ayrıdır; eski reader, CPU profil sınırları, total epoch bütçesi ve completed no-op korunur. Hash imza veya eğitim geçmişi değildir. |
| P01–P08 | Model/pipeline süreleri, kamera FPS/images/s ve performans/kalite ayrımları dört dilde aynı. Sonlu sweep genel optimum değildir. |
| P09–P11 | Allocated/reserved toplam fiziksel VRAM değildir; RSS model-only bellek değildir; mantıksal/file/resident byte ayrıdır. |
| P12–P18 | Bilinmeyen maliyet sıfır değildir. MAC/FLOP kuralı varsayılmaz. Host olayı kernel sahipliği değildir. Roofline hedef-model sayaçları ister. |
| P19–P25 | Geçmiş/resume süreleri yeni ölçüm sayılmaz; microbatch/update/skip ayrılır. P23 CUDA hook gate ve B4 unresolved korunur. Ulaşılmayan kalite hedefi sıfır süreli başarıya dönüşmez. |
| P26–P29 | Classification accuracy, precision ve recall ayrı anlamdadır. Fraction/yüzde/yüzde puanı ayrımı korunur. Detection mAP/segmentation IoU veya genel kazanan desteği çıkarılmaz. |
| P30–P35 | Kaynak/model/uygulama FPS’i ayrıdır. P31 tek kesintisiz 7200 saniye ister. Bellek büyümesi nedensel leak değildir. TDP enerji ölçümü değildir. Provider listesi node yürütme kanıtı değildir. |
| Advisor / Reporting | Öneri modeli değiştirmez ve eğitim başlatmaz. Teknik rapor API’leri yalnız tr/en; dört dilli rehber de/es rapor desteği sağlamaz. |
| CPU / GPU | Tarihsel CPU beta kaydı, canlı cihaz durumu ve yeni yürütme yetkisi ayrıdır. B4 root cause unresolved ve GPU admission blocked metinleri dört dilde korunur. |
| Hatalar / durumlar | Eksik/null değere sıfır denmedi. Almanca ifade açıkça “unbekannt, nicht den Zahlenwert 0” olarak netleştirildi. `--debug`, kodlar ve izin koşulları çevrilmedi. |

## Dil ve terminoloji

| Dil | İnceleme ölçütü |
|---|---|
| Türkçe | Önce sade açıklama, sonra terim; örnek grubu/batch, epoch, loss ve gradient açıklamaları. Doğruluk ile sayısal precision aynı anlamda kullanılmadı. |
| English | Native workflow wording, explicit scope and denominators; no literal Turkish word order. Precision-setting changes are separated from classification precision. |
| Deutsch | Doğal fiil sırası, Artikel/Kasus ve bileşik terimler gözden geçirildi. `Zahlenformat`, `Präzision`, `Durchsatz`, `Warteschlange`, `Grafikspeicher` bağlama göre seçildi. |
| Español | Doğal “usted” yönergeleri, vurgu işaretleri ve cümle akışı gözden geçirildi. `precisión numérica`, `accuracy/exactitud`, `lote`, `rendimiento`, `proveedor de ejecución` bağlamları ayrıldı. |

API, method, sınıf, Registry girdisi ve topic ID’leri ortak bırakıldı. Kritik
sınırların bir dilde yumuşatılmaması veya düşmemesi konu bazında incelendi.
Almanca/İspanyolca kullanıcı açıklamaları çevirildi; mevcut teknik rapor
ürününün dil kapsamı değiştirilmedi.

## Konu kapsam kaydı

| İstenen alan | Kalıcı topic ID’leri |
|---|---|
| A–B | `about.overview`, `getting-started.installation`, `getting-started.first-run` |
| C–D | `models.support`, `models.legacy`, `models.development-families`, `models.model-builder`, `models.graph-builder` |
| E | `components.convolution`, `normalization`, `activation`, `pooling`, `residual`, `attention`, `regularization`, `sampling`, `heads`, `fusion`, `mobile` altında `components.` öneki |
| F–G | `training.configuration`, `training.trainer`, `training.checkpoints` |
| H–I | `analysis.overview`, `analysis.p01`–`analysis.p35`, `metrics.interpretation` |
| J–L | `analyzer.workflow`, `models.checks`, `comparison.models`, `comparison.architecture` |
| M–O | `advisor.recommendations`, `reporting.outputs`, `experiments.manager` |
| P–S | `api.python-cli`, `troubleshooting.common`, `support.cpu-gpu`, `compatibility.versions` |

Eksikler desteklenmiş gibi sunulmadı: genel detection/segmentation/multi-task,
keyfi Graph/checkpoint birleşimleri, commercial wheel dışındaki M3/M4,
FasterNet/PartialConv, genel ONNX deployment kabulü, GPU production kabulü ve
de/es teknik rapor API desteği kapsam dışıdır. Rehber bunları açıklayan konu ve
bağlantıları içerir; yeni algoritma veya destek eklemez.
