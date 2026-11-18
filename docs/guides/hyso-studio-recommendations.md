> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio recommendations

The page starts with the selected model and counts of problems, validated fixes and observations. BASIC cards explain the issue, affected layer/test and next action. Explanation expands evidence type, validation and risks. Technical details retain original items, hashes, native Advisor coverage and source evidence. These raw blocks are generated only when opened.

Repeated rule/scope/affected-entity findings share one card while retaining all technical sources. Missing coverage and unverified-source native findings are consolidated into one insufficient-measurement state. Missing P-features are not separate user errors. The supported measurement action appears only for models admitted by the short CPU profile. No advice is invented for empty data, and static validity is never presented as measured performance improvement.

TEMEL görünüm sorun, konum ve uygulanabilir sonraki adımı gösterir. Açıklama kanıt türünü, riski ve doğrulama sınırını açar. Teknik ayrıntılar tam kaynak/JSON/hash verisini korur. Tekrarlanan bulgular tek kartta toplanır. Eksik native kapsam kullanıcı hatası sayılmaz; tek yetersiz ölçüm özeti gösterilir. Doğrulanmış aday yalnız açık kullanıcı eylemiyle yeni taslakta açılır.

Studio combines current Core model diagnostics with the existing evidence-based native Advisor. Analysis uses the current configuration hash and retained records bound to that hash. It runs no benchmark, training operation or automatic model mutation.

Each graph item has a stable content-derived ID, affected node, diagnostic evidence, observed condition, cause code, proposed action, expected effect, cost/risk, validation method, validation result, confidence basis, priority and manual-intervention state. Identical evidence and action produce the same deduplication ID.

The implemented automatic candidate rule handles `CHANNEL_MISMATCH` when the node has an `in_channels` parameter and Core supplies an integer incoming channel count. Studio deep-copies the graph, edits that parameter and runs full Core preflight again. Only a fully valid candidate receives `STATICALLY_VALIDATED` and an Open candidate control. The original draft remains unchanged until that explicit action opens the candidate as a new draft. Other deterministic invalidity diagnostics are `PROVEN_INVALID` findings for manual review.

Static validation establishes geometry and declared resource compatibility. Runtime behavior and speed effects remain unmeasured. Studio produces no `MEASURED_IMPROVEMENT` claim from this rule. A valid graph with no matching retained evidence returns `INSUFFICIENT_EVIDENCE`.

Retained native records feed `EvidenceRecord`, `EvidenceIndex` and the existing public `recommend` API. The original Advisor result and its evidence limitations remain visible. Its v1 native resolver supports P01/P02/P03/P08/P10/P11/P26. Broader P-feature catalog coverage supplies evidence-review context.

The central item contract adds scope, category, affected model/run/node/test, evidence references, observed condition, proposed change, validation status, trade-offs, cost/risk, confidence basis, priority and deduplication key. Categories distinguish static corrections, measured improvements, hypotheses, insufficient evidence and non-applicable findings. No current rule claims a measured causal improvement.

Training rules require at least four finite recorded epochs. They identify train/validation divergence, a last-three-epoch relative loss span at most 1%, final loss above 1.2 times its starting value, and a possible underfitting hypothesis when loss does not decrease and accuracy varies by at most one percentage point. These are transparent review thresholds. They do not establish causes. Learning-rate changes retain actual epoch observations. Validated P25 records can identify nonfinite gradients or three measured zero-gradient windows. Recorded FloatingPointError failures are linked to their job evidence.

Test diagnosis links each failed test name and source case. A PASS-to-FAIL regression requires ordered observation timestamps, identical model hash and matching known dataset/split/device conditions. Expected/actual are source test statuses. Relations to code changes remain unavailable without additional evidence.

Native Advisor recipes are projected into the central contract using their existing TR/EN text and source references. Recipes with no retained evidence are omitted. Original native reports remain available. Legacy inference records with zero warmup receive a specific measurement-condition review. A mean-latency increase above 10% is identified only with matching known hardware, input, benchmark, dataset/split and device conditions. It remains an experiment hypothesis with no statistical-significance claim. Priority sorts explicit HIGH/MEDIUM/LOW rules, followed by stable IDs. No measured benefit/cost ranking is invented.

## Türkçe kanıta dayalı öneriler

V2 graph açıklamaları orijinal Core mesajı, diagnostic code, node ID, gerçek parametreler, beklenen/alınan değerler ve Core giriş geometrisiyle ilişkilidir. Reshape eleman sayısı, PixelUnshuffle bölünebilirliği, vektör başlıkta `pooling=none` ve `in_features` uyumsuzluğu yalnız ilgili Core mesajı gerçekten bulunduğunda açıklanır. Flatten terminal kısıtı yalnız schema2'nin gerçek `INVALID_BLOCK_ORDER` tanısında anlatılır; graph'ta NC destekleyen sonraki katmanlar için yanlış terminal hatası üretilmez.

Bir kök hatanın ulaşabildiği ve Core tarafından `blocked` işaretlenen downstream katmanlar kartta ayrı listelenir. Bunlar bağımsız hata gibi çoğaltılmaz. Bağımsız hatalar ayrı kartlarda kalır. Gerekli yapılandırma bilgisi yoksa belirli neden varsayılmaz; orijinal Core kanıtı incelenir. Katmana git ve yalnız tüm kopya Core ön doğrulamasından geçtiyse yeni aday taslağı aç eylemleri kullanılabilir.

Önerileri İncele düğmesi güncel model hash değerine bağlı Core tanılarını ve saklanan gerçek kayıtları inceler. Her öneri kalıcı kimlik, kapsam, etkilenen öğe, kaynak hash, gözlem, önerilen değişiklik, doğrulama durumu, ödünleşim, risk, maliyet, güven dayanağı ve öncelik taşır. Aynı kanıt ve kural tekrar öneri çoğaltmaz.

Graph kanal düzeltmesi bağımsız kopyada tam Core ön doğrulamasından geçer. Yalnız tüm graph geçerliyse statik olarak doğrulanır ve yeni taslak açma düğmesi görünür. Eğitim ve performans hipotezleri kullanıcı deneyi gerektirir. Model ve proje ayarları öneri incelemesiyle değişmez.

En az dört sonlu epoch olduğunda kayıp eğilimi, train/validation ayrışması, kayıp platosu ve olası yetersiz öğrenme incelenir. Eşikler açıkça kanıtta tutulur. Gradient yorumu yalnız doğrulanmış P25 kaydı veya kaydedilmiş sayısal hata varsa yapılır. Başarısız test kimliği ve kaynak mesajı gösterilir. PASS→FAIL regresyonu zaman, model ve bilinen koşul eşliğini gerektirir.

Gecikme karşılaştırması gerçek native ölçümleri ve donanım/benchmark/girdi koşullarını kullanır. Eksik veya farklı koşullarda doğrudan hız kazanımı önerisi verilmez. Ölçülmüş nedensel iyileştirme iddiası üretilmez. Yetersiz kayıt için öneri bulunmaması geçerli sonuçtur. Ham native Advisor raporu ve teknik kaynak ayrıntıları kanıt görünümünde korunur.
