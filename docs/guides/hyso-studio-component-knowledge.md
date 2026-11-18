> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio component knowledge

`GET /api/catalog` derives forms and knowledge records from `ModelRegistry` and graph fusion metadata without constructing modules. The development checkout has 85 Registry entries and four fusion operators. The commercial package has 57 Registry entries and the same four fusion operators. Studio's editable palette applies the commercial component policy in both environments. Complete-model entries appear in knowledge metadata and use their supported configuration interfaces.

Every Registry constructor parameter retains its technical key, kind, required/default state, choices and available Core constraints. All 222 commercial Registry parameter-field occurrences have Turkish/English labels and explanations. The three fusion parameter fields have the same coverage. Technical keys keep their original spelling.

The knowledge panel exposes component definitions, input/output conditions, parameter dependencies, common errors and constructor/source references. TECHNICAL mode additionally exposes available compatibility, intermediate-shape, train/eval and correction metadata. JSON field names and API identifiers retain their original spelling.

All 61 commercial mechanisms carry `BILINGUAL_SOURCE_MAPPED` content. BASIC presents operation, intended use, benefits and limits. TECHNICAL also presents the mathematical contract, related components, source references and an executable example. Sixty graph-compatible examples pass Core preflight and CPU forward/backward tests. The complete `modular_cnn` entry retains its original constructor example. Implementation and Registry files supply the references. Raw source metadata is available in a separately expandable technical view.

The notes distinguish grouped/depthwise channel mixing, internal residual projection, spatial self-attention quadratic matrix costs, channel-only LayerNorm, pooling divisor rules and raw WeightedSum coefficients. They state no unmeasured hardware speed or quality gain. Parameter bounds are preserved from Core. Some cross-parameter conditions are enforced by full preflight rather than a single numeric widget.

BASIC shows purpose, usage, limits, readable parameter descriptions and plain input/output names. It omits raw tensor contracts and validation JSON. More information expands natural-language bounds derived from actual Core metadata. WeightedSum explains its two-to-eight raw finite coefficients and their actual numeric bounds. TECHNICAL and separately expandable Technical details retain formulas, tensor metadata, exact validation fields, source references and Python examples. Presentation does not replace Core validation.

Constructor forms use supported JSON types. Registry validation and full graph preflight enforce grouped convolution divisibility, transpose output padding, attention limits, interpolation rules, normalization statistics and fusion geometry. Constraint descriptions supplement the existing validators.

The capability inventory contains 161 entries covering 19 public builder methods, 89 component/fusion records, 35 P-features and eighteen integration records. It records Python, CLI, config, UI, endpoint, validation, test and source mappings. [Capability inventory](../reports/stage13-5-6-capabilities.json)

## Türkçe bilgi merkezi

V2 denetimi 61 mekanizmanın her biri için `AUDIT` içinde ayrı kullanım, sınırlama ve maliyet notu tutar. Kategori düzeyindeki ortak attention/residual/pooling/normalization sınırlamaları içerik üretiminde kullanılmaz. Referanslar gerçek constructor, Registry ve Core geometri kaynağıdır. Harici makale bağlantısı eklenmemiştir. Her mekanizmanın küçük Python örneği çalıştırılabilir; 60 graph örneğinin gerçek CPU çıktıları ve parametre sayımları test edilir.

SE global uzamsal ortalamayı daraltılmış kanal MLP'sine taşır; `(H*W)²` dikkat matrisi oluşturmaz. Spatial self-attention QK/V token eşleştirmesi yapar; token ve `N*heads*T*T` matris bütçeleri yalnız ilgili self-attention mekanizmalarına uygulanır. ECA, CBAM, Coordinate, Spatial/Channel Attention notları kendi gerçek kapılama işlemlerini açıklar.

TEMEL görünüm kartlar, kullanım/sınırlama/maliyet ve okunabilir parametre açıklamaları sunar. TEKNİK görünüm matematik, giriş/çıkış koşulları, syntax-highlight Python/kopyalama ve kaynak kartları ekler. **Ham JSON** her iki düzeyde ayrı, kapalı bir bölümdür; teknik görünüm bunu kendiliğinden açmaz. Daraltma oranı gibi mekanizmaya özgü etkiler ayrı açıklanır; varsayılanlar ve sınır cümleleri Core metadata'sından gelir.

Bilgi Merkezi'nde 57 ticari Registry mekanizması ve dört fusion işlemi bulunur. Mekanizmanın Türkçe/İngilizce adı, çalışma şekli, matematiği, kullanım amacı, avantajları, sınırlamaları, tensor sözleşmesi, parametreleri, örneği ve ilişkili mekanizmaları gösterilir. Arama her iki dilde çalışır. Kategori filtresi ve ilişkili bileşen düğmeleri kullanılabilir.

TEMEL düzey kullanım ve kaçınma koşullarını açıklar. TEKNİK düzey formül, Core kaynakları, örnek graph ve Python kodunu ekler. Örneği yeni taslakta açarak değiştirebilirsiniz. Örnekler sentetik test verisidir ve kullanıcı ölçümü olarak saklanmaz.

222 Registry alanı tekrarı ve üç fusion alanının tamamı iki dilli etiket/açıklama taşır. Varsayılanlar, seçimler ve mevcut kısıtlar Registry'den gelir. Teknik anahtarlar çevrilmez. Ek bağımlılıkları Core doğrular. Derinlik evrişimi kanallar arası karışım yapmaz. Gruplu evrişim kanal bölünebilirliği ister. Spatial self-attention 1024 token ve 4194304 matris elemanı sınırını korur. WeightedSum ham katsayıları normalize etmez.
