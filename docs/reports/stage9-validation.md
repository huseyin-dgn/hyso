# Aşama 9 doğrulama raporu

Tarih: 2026-11-12. Branch: `feature/hyso-models-next`.
Başlangıç ve bitiş HEAD: `f92dc078d3bdd7733eb155c5ef2a5191720532a2`.
Commit, push veya Aşama 10 çalışması yapılmadı.

## API ve karar sözleşmesi

`hyso.advisor.recommend(source, request=None, *, context=None)` immutable
`RecommendationReport` döndürür. `UnifiedAnalysisReport`, `ModelComparisonResult`
ve açık `AnalysisContext` ile `EvidenceIndex` kabul edilir. Kaynak snapshot'ları,
orijinal `AdvisorReport`, kural/sürüm, plan, native EvidenceRef, birim, kayıt
kimliği ve karşılaştırma kapıları korunur. `advise`, resolver, planlayıcı,
Advisor hesapları ve P01–P35 ölçüm motorları değiştirilmedi.

Altı alan: inference, training, memory, model_architecture, deployment,
data_pipeline. Kartlar tespit, kaynak kanıtı, önerilen değişiklik/inceleme,
gerekçe, yan etki, manuel uygulama ve yeniden doğrulama adımlarını içerir.
`tr`/`en` ve `beginner`/`technical`/`expert` desteklenir. Yeni açıklamalar ve
sunum sınırları yerelleştirilir; ham native teknik metin çevrilmeden saklanır.

Yeterli P02 native bağlarında mevcut CPU batch kuralı şu tip bir karar üretir
(gerçek çıktıdan alan projeksiyonu; sayısal kazanç tahmini değildir):

```json
{
  "area": "inference",
  "action_kind": "conditional_experiment",
  "rule_id": "plan.batch-throughput",
  "expected_effect": "unknown",
  "estimated_improvement": null,
  "priority": null,
  "authorization": "not_authorized",
  "readiness": "blocked",
  "verified_effect": "not_measured"
}
```

Yalnız orijinal uygun kural batch_size değişkeni için aday üretir. Gerekli P06,
bütçe, kalite, bellek ve bağımsız doğrulama önkoşulları kaldırılmaz. Belirsiz
ve farklı değerler içeren çoklu capture'lar birleştirilmez. Uyumsuz veya eksik
karşılaştırma kanıtı adayı incelemeye indirger; kazanan veya kazanç üretilmez.
Hard hedef çelişkileri yalnız tutarlılık incelemesi oluşturur. Karşılanmış ve
unknown hedeflerden yeni optimizasyon amacı çıkarılmaz. Otomatik uygulama yoktur.

## Test sonuçları

Tekrarlanan node'lar aşağıdaki toplamda yalnız bir kez sayılmıştır. Yeni testler
ve düzeltmeler hedefli turlarla doğrulandı; gereksiz tam tekrar yapılmadı.

| Kapsam | Benzersiz geçen test |
|---|---:|
| Aşama 9 öneri, güven sınırı, export, örnek ve belge testleri | 41 |
| Mevcut Advisor, karşılaştırma, raporlama, evidence ve public API regresyonları | 166 |
| Paketleme, wheel/sdist ve kurulu paket API/typing kontrolleri | 4 |
| Toplam | **211** |

Regresyon komutunun ham sonucu **168 passed, 2 deselected, 67 subtests passed**;
168'in ikisi yeni Aşama 9 testleriyle örtüşür. Paketleme komutunun ham sonucu
**4 passed, 2 deselected**. Son hedefli kontrollerde kalan başarısızlık yoktur.

İlk geliştirme turlarında iki test beklentisi düzeltildi: UnifiedAnalysisReport
snapshot kontrolünde zorunlu export yolu yerine canonical sözlük kullanıldı;
native P19 fixture'ının mevcut `partial` kapsamı `validated` beklenmeden korundu.
Varsayılan request.priority=None durumu ilk smoke kontrolünde ele alındı.

Testler, hazır native JSON/ledger fixture'larını ve geçici yerel dosyaları
kullandı. Yeni testler model forward, Trainer/TrainingSession.fit, benchmark,
CUDA sorgusu, checkpoint yükleme ve canlı analiz girişlerini engeller. Kurulu
paket smoke testindeki mevcut küçük CPU forward kontrolleri korunmuştur.
Gerçek eğitim, benchmark, CUDA ölçümü veya uzun kabul süiti çalıştırılmadı.

Çalıştırılan regresyon kapsamı:

```text
tests/test_advisor_planning.py
tests/test_advisor_goals.py
tests/test_advisor_native.py
tests/test_advisor_presentation.py
tests/test_model_comparison.py
tests/test_reporting.py
tests/test_analysis_report.py
tests/test_api_discovery.py
```

`current_readme_examples_execute_cpu_only` ve
`python_api_training_example_executes_cpu_only` eğitim/benchmark çalıştırmamak
için seçilmedi. Yeni README ve rehber blokları ayrı fixture testlerinde çalıştı.
Paketlemede geniş `installed_distribution_import_and_cpu_forward` testinin iki
varyantı seçilmedi. P01–P35 kabul süiti yeniden çalıştırılmadı.

Ortam: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`,
`CUDA_VISIBLE_DEVICES=-1`, `MPLBACKEND=Agg`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1`.
Pytest yapılandırması: `-c tests/pytest.ini`; cache kapalıdır.

## Örnekler ve HTML

`examples/advisor/retained_recommendations.py`, boş evidence, kayıtlı analiz ve
iki model karşılaştırması girdileriyle workload engelleyicileri altında geçti.
Dört çıktı doğrulandı: `recommendations.json`, `report.json`, `report.md`,
`report.html`. README fonksiyonu ve rehber Python bloğu geçici dosyalarla çalıştı.

Doğrudan script kontrolünde seçili ortamda HYSo kurulu olmadığı görüldü.
Checkout kullanımı için rehbere `$env:PYTHONPATH = (Get-Location).Path` eklendi.
Bu adımla aşağıdaki gerçek CLI komutu başarıyla tamamlandı:

```powershell
python examples/advisor/retained_recommendations.py --output C:/WINDOWS/TEMP/hyso-stage9-report-aklfcmho/cli-empty --language tr --level beginner
```

Altı dil/seviye birleşimi ve ayrıca boş veri için **7 HTML** oluşturuldu.
Headless Edge, `--disable-gpu`, 390 ve 1280 piksel genişlikte **14 yerleşim
kontrolü**: sayfa taşması, taşan kart/metin, bozuk iç bağlantı ve JavaScript
hatası yok. Detaylar açılarak ham kanıt metni de kontrol edildi. Türkçe teknik
kart, İngilizce mobil başlangıç görünümü ve boş veri ekran görüntüleri görsel
olarak incelendi; metin, liste ve etiketler okunabilir. Mevcut Matplotlib
grafikleri yeniden kullanılmaktadır; yeni ölçüm veya tahmini grafik yoktur.

Yerel çıktılar:
`C:/WINDOWS/TEMP/hyso-stage9-report-aklfcmho/`.
Bu dizin geçicidir; paket veya Git içeriğine dahil edilmedi.

## Değişen 15 dosya

| Dosya | Değişiklik |
|---|---|
| `hyso/advisor/recommendations.py` | Yeni read-only API, yeniden türetme, uygunluk kapıları ve export |
| `hyso/advisor/_recommendation_text.py` | Altı alan için iki dilli manuel açıklamalar |
| `hyso/advisor/__init__.py` | Yeni public export'lar |
| `hyso/reporting/report.py` | Explicit RecommendationReport kaynağı ve metrik bağlantıları |
| `hyso/reporting/render.py` | JSON sözleşmesindeki önerilerin Markdown/HTML sunumu, güvenli iç bağlantılar |
| `tests/test_advisor_recommendations.py` | 41 deterministik fixture testi |
| `tests/test_packaging.py` | 138 modül envanteri ve kurulu Advisor API kontrolü |
| `tests/test_api_discovery.py` | README örnek sayısı: 11 |
| `examples/advisor/retained_recommendations.py` | Çalıştırılabilir kısa örnek |
| `examples/advisor/README.md` | Örnek komutları ve veri sınırları |
| `examples/README.md` | Advisor dizini |
| `docs/guides/performance-advisor.md` | API, kullanım ve güven sınırları |
| `docs/guides/python-api.md` | API haritası ve rehber bağlantısı |
| `README.md` | Manuel Advisor örneği |
| `docs/reports/stage9-validation.md` | Bu doğrulama raporu |

## Korunan durum ve sınırlamalar

Başlangıçtaki **47 forensic/debug/ilgisiz dosyanın SHA-256 değerleri aynı**.
Yeni yazılan dosyalar yukarıdaki listeyle sınırlıdır. Staged alan boş;
HEAD değişmedi. 70/70 kabul kanıtları, checkpoint'ler ve önceki aşamaların
ölçüm/veritabanı formatları değiştirilmedi.

Mevcut resolver yalnız P01/P02/P03/P08/P10/P11/P26 için scalar uygunluğu
sağlar. Diğer alanlarda kayıt/kapsam incelemesi sunulur; yeni optimizer,
quantization, pruning veya worker ayarı öneren hesaplama kuralları eklenmedi.
Kaynakta yeterli etki/maliyet kanıtı olmadığından sayısal iyileşme ve fayda
sıralaması bilinmiyor kalır. JSON import'u snapshot'tan türetmeyi doğrular;
dış artifact dosyalarını tekrar okumaz ve authenticity attestation sağlamaz.
P09/P10/P11 alanları ve P19/UX observer süreleri birbirine dönüştürülmez.
Gerçek performans kazanımı bu aşamada ölçülmedi veya iddia edilmedi.
