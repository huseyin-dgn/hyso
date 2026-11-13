# Performance Advisor: kanıttan manuel öneriye

`hyso.advisor.recommend(source, request=None, *, context=None)` mevcut Advisor
kurallarını açıklama, yan etki, uygulama ve yeniden doğrulama adımlarıyla sunar.
İkinci bir ölçüm veya optimizasyon motoru değildir. Model, optimizer, eğitim
akışı ve veri hattı otomatik değiştirilmez; hiçbir ölçüm başlatılmaz.

## Girdi ve çıktı

- `UnifiedAnalysisReport`: doğrulanmış kayıtların mevcut Advisor kararları.
- `EvidenceIndex`: açık `AnalysisContext` zorunludur.
- `ModelComparisonResult`: yalnız seçilmiş kayıtlar; mevcut comparability
  kapıları ve baseline/candidate ayrımı korunur. Global kazanan seçilmez.

`AdvisorReport` tek başına kaynak doğrulaması sağlamadığından bu API'ye girdi
değildir. Eski `advise`, `to_display_dict`, JSON ve Markdown API'leri korunur.
Eğitim için native P19/P20/P22–P25 kayıtlarını evidence raporuna ekleyin;
`TrainingResult` geçmişi tek başına bu testler yapılmış anlamına gelmez.

```python
from hyso.analysis import UnifiedAnalysisReport
from hyso.advisor import recommend, RecommendationReport
from hyso.reporting import create_report

source = UnifiedAnalysisReport.from_json("saved-analysis.json")
advice = recommend(source)
cards = advice.to_display_dict(language="tr", level="beginner")
advice.to_json("recommendations.json")
advice.to_markdown("advisor.md", language="en", level="technical")
advice.to_html("advisor.html", language="tr", level="expert")
restored = RecommendationReport.from_json("recommendations.json")
presentation = create_report(source, restored, language="tr", level="technical")
# presentation.to_json("combined-presentation.json")
```

Çıktı yolları mevcutsa varsayılan davranış üzerine yazmayı reddeder. Açıkça
istendiğinde `overwrite=True` kullanılabilir. `RecommendationReport.to_json()`
dilden bağımsız kararları ve kaynakları saklar; yerelleştirilmiş görünüm JSON'u
`create_report(...).to_json()` ile alınır. İçe aktarma kaynak snapshot'ından
kararları yeniden türetir. Hash'i güncellemek sahte kazanç veya öneriyi geçerli
yapmaz. Bu doğrulama dış artifact dosyalarının özgünlüğünü veya hâlâ mevcut
olduklarını doğrulamaz; kayıtlı native otorite sınırları aynen korunur.

## Altı alan ve karar sınırları

| Alan | Kullanılan kural veya inceleme |
|---|---|
| inference | Mevcut latency-stage, batch-throughput ve precision-quality planları |
| training | P19 zaman defteri, P20 forward/backward/optimizer aşamaları, P22 kapasite, P23 bellek, P24/P25 kapsam incelemesi |
| memory | Mevcut memory-domain planı; RSS, mantıksal baytlar ve VRAM ayrılır |
| model_architecture | accuracy-preservation planı ve P11–P14/P18/P26–P29 kanıt incelemesi |
| deployment | Bağ/protokol, hedef tutarlılığı ve kayıtlı deployment kanıtının incelemesi |
| data_pipeline | P03/P04/P15/P21 aşama, aktarım ve bekleme kapsamları |

Her kart yedi alanı içerir: tespit edilen durum, P01–P35 kanıtları, önerilen
değişiklik/inceleme, neden faydalı olabileceği, olası yan etkiler, manuel adımlar,
yeniden ölçme ve doğrulama. `original_finding`, `rule_id`, `rule_version`, `plan`,
`evidence`, `coverage`, `missing_evidence`, `blocked_by` alanları makinece okunur.
`coverage` içindeki bir kaydın varlığı optimizasyon uygunluğu anlamına gelmez.
Mevcut metric resolver yalnız P01/P02/P03/P08/P10/P11/P26'yı destekler. Diğer
alanlarda ölçüm tespit edilmiş gibi davranılmaz; kayıt incelemesi veya eksik
test isteği sunulur. P09 GPU belleği, P10 süreç RSS'i ve P11 mantıksal model
boyutu birbirine çevrilmez. P19 native süreleri UX observer süreleri değildir.

Mevcut planlayıcı yalnız yeterli bağlı CPU throughput kanıtında **batch_size**
için koşullu deney adayı üretebilir. Diğer incelemeler precision değişimi,
quantization, optimizer değişimi, pruning veya worker artırma talimatı değildir.
Koşullu aday da `readiness="blocked"`, `authorization="not_authorized"`,
`expected_effect="unknown"`, `verified_effect="not_measured"` olarak kalır.
Sonlu batch noktaları, bütçe, kalite ve bellek kabul koşulları kullanıcı
tarafından netleştirilmelidir. Deployment bütçesi deney bütçesi yerine geçmez.
Gerçek kazanım öngörülmez; `estimated_improvement=None`, `priority=None` ve
`priority_status="not_established"` korunur. Kullanıcı önceliği ayrı kaydedilir;
liste sırası kanıta dayalı kazanç sıralaması değildir.

Belirsiz çoklu capture seçimi veya uygun olmayan karşılaştırma, koşullu adayı
kanıt incelemesine indirger. Farklı capture'lar ortalanmaz, aralarındaki fark
nedensel çelişki sayılmaz. Çelişkili hard hedeflerde yalnız tutarlılık incelemesi
sunulur. Karşılanmış/unknown hedeflerden yeni optimizasyon amacı türetilmez.
Model karşılaştırması uyumsuzsa yerel kayıt incelemesi yapılabilir; kazanan,
ortak üstünlük veya model değiştirme önerisi üretilmez.

## Dil, seviye ve raporlama

`language="tr"` ve `"en"`; `level="beginner"`, `"technical"`, `"expert"`.
Başlangıç düzeyi sade açıklama ve manuel adımları; teknik düzey native sayı,
birim ve kayıt kapsamını; uzman düzeyi ham kanıt, orijinal kural, bağlar,
protokol ve sınırları sunar. Kaynak ham teknik metni çevrilmeden korunur.
Her seviyede kanıt ve doğrulama bağlantıları açılabilir. HTML dış kaynak
gerektirmez. Grafikler önceki raporlama sisteminde mevcut veri varsa oluşturulur;
öneri kartları kendileri yeni grafik veya ölçüm üretmez.

## Çalıştırılabilir örnekler

Depo kökünde, HYSo'nun kurulu olduğu ortamda:

Seçili ortamda HYSo kurulu değilse, checkout'tan çalıştırmak için önce aynı
PowerShell oturumunda depo kökünü import yoluna ekleyin:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

```powershell
python examples/advisor/retained_recommendations.py --output C:/Temp/hyso-advisor-empty --language tr --level beginner
python examples/advisor/retained_recommendations.py --input saved-analysis.json --output C:/Temp/hyso-advisor-analysis --language en --level expert
python examples/advisor/retained_recommendations.py --input saved-comparison.json --output C:/Temp/hyso-advisor-comparison --language tr --level technical
```

İlk komut boş evidence fixture'ı kullanır: ölçülmüş darboğaz veya sahte sayı
göstermez. Diğer komutlar sizin mevcut JSON dosyalarınızı okur. Yeni çıktı
klasörü kullanın. Dört çıktı: `recommendations.json`, `report.json`, `report.md`,
`report.html`. Örnekler, tüm dil/seviyeler, kanıt yokluğu, native bağ eksikliği,
çelişkili hedefler ve uygun olmayan karşılaştırmalar kısa CPU testleriyle
doğrulanır; eğitim, benchmark, CUDA veya uzun kabul testi çalıştırılmaz.
