> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio model comparison / Model karşılaştırma

Model Comparison opens in architecture mode. Select two to five saved drafts or validated versions; two are selected initially. No training data or result record is needed. The API independently enforces the limit and rejects duplicate references. Identical configuration hashes are explicitly grouped as equivalent architectures. Input geometry differences carry a cost-comparison warning.

The table presents model/version, type, inputs/outputs, nodes, branches, mechanisms, Core parameter counts or tagged estimates, activation/attention cost, validation and supported training scope. Unknown values remain unknown. Technical details retain exact hashes and complete Core preflight. No automatic winner is declared.

Switch to training/performance results to choose two to eight distinct retained records. This mode uses real native metrics and retained epoch curves. Basic tables show readable metric labels and units; technical details retain architecture SHA-256, run identity, source payload hash, measured/imported state and complete conditions.

`CONDITIONS_MATCH` means the relevant recorded condition fields are present and equal. It does not authenticate caller declarations or establish statistical significance. Performance additionally requires matching hardware environment, benchmark configuration and input descriptor. Training checks requested and effective settings. Evaluation checks configuration, class ordering, target hash and prediction type. Graph head mappings must agree. Different conditions produce `CONDITIONS_DIFFER`. Missing required conditions produce `INSUFFICIENT_EVIDENCE`. The view preserves differences and shows no automatic global winner.

Native parameter counts retain their recorded source. If an embedded or saved configuration matches the exact record hash, allocation-free Core preflight supplies separately tagged parameter/activation estimates. Missing configurations produce unavailable estimates.

`POST /api/comparison` accepts `record_ids` and optional `language` (`tr`/`en`). IDs must refer to existing retained records. Duplicates, fewer than two, more than eight and unknown records fail. No comparison action starts training or measurement. Source data remains unchanged.

`POST /api/architectures/compare` accepts `models`, each with `kind` (`model` or `version`), `id` and expected `revision`. Stale draft revisions are rejected. Actual supplied-data digests supplement declared dataset/split matching for locally executed result comparisons.

## Türkçe kullanım

Parametre sayımı üç durum taşır: `EXACT` kesin Core sayacı veya gerçek CPU `nn.Module.parameters().numel()` toplamıdır; `ESTIMATED` Core'un korumacı üst sınırıdır; `UNAVAILABLE` ise çözülmeyen ön doğrulama veya eksik metadata nedeniyle sayı verilemediğini belirtir. Geçerli legacy SE/CBAM/Residual gibi mekanizmalarda statik kesin sayaç bulunmaması, sayının hiç hesaplanamayacağı anlamına gelmez. UI hangi sayacın eksik olduğunu açıklar.

**Kesin CPU parametre sayımı** ayrı açık kullanıcı eylemidir. Seçili 2–5 kayıt için tek CPU worker kuyruğuna sınırlı build işleri gönderir. Core üst sınırı en çok 1.000.000 parametre olmalıdır; fp32 parametre depolaması yaklaşık 4 MiB'dır, buffer ve oluşturma ek yükü bu rakama dahil değildir. İleri hesaplama, CUDA veya eğitim yapılmaz. Başarısızlık nedeni model kartında görünür. Tamamlanan iş model hash'i ve kimlik/revision ile bağlanır; başka model veya yeni revision'a kesin sayım aktarılmaz.

Üst kartlarda kesin/tahmini sayı ve gerçek kategori dağılımı; tabloda birleşim operatörleri, head türleri, giriş geometrisi, build/eğitim uygunluğu ve Core bütçe oranı bulunur. Parametre farkı yalnız iki sayı da kesin olduğunda ilk modele göre hesaplanır. Aktivasyon/attention maliyetleri statik tahmindir; farklı geometriler uyarılır. Eğitim yapılmamış mimarilere accuracy veya latency eklenmez.

Mimari modunda 2–5 kayıtlı taslak veya sürüm seçin. Başlangıç seçimi ikidir; beşten sonra diğer seçimler kapanır. Aynı mimariler açıkça belirtilir. Eğitim veya ölçüm gerekmez. Maliyetler Core sayımı/tahminidir; giriş geometrisi farklıysa uyarı gösterilir. Sonuç modunda 2–8 gerçek kayıt karşılaştırılır. Sayılar, birimler ve mevcut gerçek epoch eğrileri gösterilir; tam hash/provenance Teknik ayrıntılar altındadır.

Bilinen koşullar eşitse CONDITIONS_MATCH verilir. Bu durum kullanıcının veri/split beyanını bağımsız olarak doğrulamaz. Donanım, girdi, benchmark veya eğitim koşulları farklıysa CONDITIONS_DIFFER gösterilir. Gerekli bilgi eksikse INSUFFICIENT_EVIDENCE kalır. Eksik sayı sıfır kabul edilmez. İstatistiksel anlamlılık veya genel kazanan iddiası verilmez.

Karşılaştırma model ve sonuç verilerini değiştirmez. Açık mimari karşılaştırma eylemi, son karşılaştırmalar için ad/referans/hash ve zaman kaydı tutar; bu bir ölçüm veya eğitim sonucu değildir. Ölçüm veya eğitim başlatmak ayrı açık kullanıcı eylemidir. Sentetik test fixture'ları gerçek kullanıcı sonucu olarak varsayılan depoya yüklenmez.
