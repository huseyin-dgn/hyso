# Kontrollü gece çalışması ve Aşama 13.5.5 doğrulaması

13 Kasım 2026 tarihinde `feature/hyso-models-next` üzerinde gerçekleştirildi.
Plan kullanım yüzdelerine doğrudan erişilmedi. Kullanıcının bildirdiği oranlar
sabit işlem bütçesi olarak kullanılmadı. Kanıt ve test çıktıları repository
dışındaki `%LOCALAPPDATA%/hyso-stage13-5-5-20261113` dizininde tutulur.

## Bölüm A — Aşama 13.5.4 commit

Commit `66b90b61041ee2c06c265df9b5c19ce73e2f7b79` oluşturuldu.
Parent `03fc5b3cf2e7ba743df7832e2836c2b279b3f4af` olarak doğrulandı.
Mesaj `feat(builder): add sampling tensor utilities and prediction heads` oldu.

105 kısa regresyon geçti. Registry 57 ticari / 85 geliştirme girdisi, 155 modüllük
izin listesi, schema1/schema2 ve mevcut wheel/sdist kaynak eşleşmeleri doğrulandı.
19 dosyanın hash'leri Aşama 13.5.4 doğrulama kayıtlarıyla birebir eşleşti.
Dosyalar ayrı açık yollarla stage edildi. Cached scope/content ve whitespace
kontrolü geçti. Commit sonrası staged alan boş ve koruma kontrolleri başarılıydı.

Commit şu 19 dosyayı içerir.

```text
hyso/models/sampling.py
hyso/models/registry.py
hyso/models/builder.py
hyso/cli/models.py
hyso/training/session.py
commercial-profile.json
tests/test_builder_sampling_heads.py
tests/test_builder_advanced.py
tests/test_builder_components.py
tests/test_builder_validation.py
tests/test_model_builder.py
tests/test_m4_architectures.py
tests/test_commercial_distribution.py
tests/test_packaging.py
docs/guides/custom-cnn-builder-sampling-heads.md
docs/guides/custom-cnn-builder-validation.md
docs/guides/custom-cnn-builder-components.md
docs/guides/model-registry-builder.md
docs/reports/stage13-5-4-validation.md
```

## Bölüm B — Graph Builder API ve schema

Yeni public `GraphModelBuilder` hem hyso hem hyso.models named import yolunda
bulunur. API input specs, add, outputs, to_config/from_config, to_json/from_json,
preflight, preview ve build girişlerini sağlar. Graph JSON ayrı schema_version=3,
kind='graph' kullanır. Mevcut ModelBuilder kaynağı Bölüm A commit'iyle byte düzeyinde
eşleşir. Schema1/schema2 davranışı ve single-Tensor output sözleşmesi korunur.

Graph node'u stable ID, Registry component adı, params ve ordered input references
taşır. Input/node ad alanları ortaktır ve benzersiz olmalıdır. Named output'lar
node veya input referansına bağlanabilir. Çıktı adları ve hazır düğümler alfabetik
olarak seçilir. Deterministik topolojik sıra JSON'daki fiziksel sıradan bağımsızdır.
State anahtarları `nodes.node_<ID>.*` biçimindedir.

Ticari Registry **57**, geliştirme Registry **85** girdide kaldı. Dört fusion
Graph Builder'a özgü kapalı operatör kümesidir. İkinci convolution/attention
implementasyonu oluşturulmadı. Mevcut Registry constructors ve shape/cost
sözleşmeleri tekrar kullanılır. Ticari pozitif izin listesine graph.py ve
graph_ops.py eklendi. Python modülü toplamı **157** oldu.

## Fusion ve branch sözleşmeleri

Add ve Multiply 2..8 aynı şekilli input alır ve broadcasting yapmaz. Concat seçili
non-batch axis dışında aynı boyutları gerektirir. NC için axis=1, NCHW için
axis=1/2/3 kabul edilir. WeightedSum ordered input başına ham skaler ağırlık
kullanır. Başlangıç ağırlıkları finite ve [-1e6,1e6] aralığında, sayıları fan-in
ile aynı olmalıdır. Learnable ağırlıklar parameter, sabit ağırlıklar buffer olur.
Her iki biçim de state_dict içinde saklanır. Otomatik normalizasyon uygulanmaz.

External skip normal tensor reference olarak kullanılır. Projection ve resize
ayrı explicit Registry düğümleridir. Fusion şekil hatalarında 1×1 projection,
stride/resize veya uygun concat önerisi verilir. İşlemler otomatik eklenmez.

## Shape, rank, hata ve preview

NC ve NCHW input metadata'sı symbolic N, rank, shape, layout, feature meaning,
floating dtype ve model device sözleşmesi taşır. Her edge bu tensor metadata'sını
raporlar. NCHW Registry bileşenleri, sampling ve explicit prediction head'leri
kullanılır. Flatten NCHW → NC dönüşümüdür. NC identity, uygun activation, dropout,
fusion ve pooling='none' prediction head'lerini destekler. NC → Conv2d ve NHWC
yanlış bağlantıları reddedilir.

Preflight/preview allocation-free çalışır. Duplicate ID, missing reference,
self-loop, cycle, disconnected node, yanlış output, arity, unsupported component,
rank/shape, fusion axis, head bağlantısı, attention ve parametre sınırları test
edildi. Cycle raporu bloke ID'leri içerir. Bağımsız dallar doğrulanır, geçersiz
downstream shape None ve status blocked olur.

Yeni GRAPH_STRUCTURE_ERROR, GRAPH_SHAPE_ERROR ve GRAPH_LIMIT_ERROR kodları önceki
15 tanılamayı korur. Node ID layer_id alanında taşınır. ModelConfigError'ın mevcut
sekiz diagnostic alanı ve ValueError tabanı korunur. Node raporlarında ports,
metadata, input/output shapes, parametre özetleri, status, diagnostics ve warnings
bulunur. Geçersiz node'ların maliyeti rastgele sıfır sayılmaz. Aggregate alanlar
None, validated_parameter_estimate doğrulanan alt kümenin toplamı olur.

CLI models validate/summary/preview graph JSON'u tanır. Explicit inspect de
desteklenir. Mevcut JSON envelope, stdout/stderr ve exit-code sözleşmeleri geçti.
Mimari schema3 ile CLI envelope schema1 birbirinden ayrı version sözleşmeleridir.

## Kaynak sınırları ve runtime

Graph 0..64 node, en fazla 256 edge, fan-in 8, fan-out 16, 1..8 input ve output,
48 karakterlik güvenli ID, 2 MiB JSON ve nesting 16 ile sınırlıdır. Tüm C/H/W/F
boyutları 1..8192 olmalıdır. Conservative parametre bütçesi 10M kalır. Aynı node
fan-out nedeniyle tekrar sayılmaz. Eski kompleks bloklarda exact parameter_count
None olabilir, mevcut conservative estimate güvenlik sınırını korur.

Her self-attention düğümünde 1024 token ve N*heads*tokens² ≤ 4.194.304 sınırı
korunur. Preflight explicit batch_size ile, runtime gerçek N ile kontrol eder.
Runtime input ve node output shape/device/dtype doğrulaması yapar. Tüm input'lar
aynı batch N ve model dtype/device değerini paylaşır.

Peak-live-element tahmini forward referans lifetime'larını, named output'ları,
declared internal boundaries, fusion için 2*fan-in geçici çıktı slotunu ve
attention için üç matris slotunu sayar. Shared activation bir kez tutulur, son
tüketicisi geçince graph referans haritasından bırakılır. Autograd'ın tuttuğu
tensorlar değiştirilmez. Allocator/workspace, autograd retention ve raporlanmayan
legacy içleri bu tahminin dışındadır. Gerçek peak RAM/VRAM garantisi verilmez.
Advisory eşik varsayılan 16.777.216 elemandır, hard limitlerden ayrı uyarıdır.

## Test ve eğitim sonuçları

**681 benzersiz test başarılı, 0 çözülmemiş başarısız, 0 atlanan.**
674 vakalık güvenli CPU regresyonu geçti. Kısmi maliyet raporu düzeltmesinden sonra
44 graph ve altı paketleme testi birlikte geçti. Ayrı tek-dal classification
CrossEntropy optimizer testi de geçti. Tekrarlar unique sayıya eklenmedi.
İki önceki installed-preflight vakası CPU koşusundan deselected edildi. Mevcut
preflight ve yeni graph preflight ticari kurulum smoke testlerinde doğrulandı.

| Modül | Benzersiz başarılı test |
|---|---|
| test_graph_builder | 45 |
| test_builder_sampling_heads | 101 |
| test_builder_advanced | 124 |
| test_builder_components | 227 |
| test_model_builder | 42 |
| test_builder_validation | 59 |
| test_cli | 76 |
| test_m4_architectures seçili dışlama testi | 1 |
| test_commercial_distribution | 4 |
| test_packaging seçili canonical/content testleri | 2 |

Graph testleri önce yazıldı. Tek/iki/üç dal, Add/Concat/Multiply/WeightedSum,
learnable weight gradient, shared node parametre sayımı, external skip, projection,
sampling sonrası fusion, multi-output heads, terminal flatten, NC activation,
yanlış rank/axis, duplicate/missing/self/cycle/disconnected/output hataları,
node/edge/fan-in/fan-out/parametre/attention limitleri ve JSON boyut sınırı kapsandı.
Node fiziksel sırası, allocation fence, state_dict strict roundtrip, CPU
float32/float64 train/eval, backward, optimizer ve tüm trainable branch gradient'leri
doğrulandı. Runtime'da hatalı node çıktısı da node ID ile reddedildi.

Ön koşularda testlerin node listesindeki fiziksel indeks varsayımı stable ID
seçimine çevrildi. Ham weighted sum'ın cebirsel olarak yeniden yazılmış sonucunu
floating tolerance ile karşılaştıracak test düzeltildi. Son koşular geçti.
Mevcut Trainer GradScaler deprecation uyarıları görüldü. CUDA context, uzun eğitim
ve native P01–P35 workload başlatılmadı.

Tek classification graph CrossEntropy, multi-output graph açık differentiable
loss toplamı ve SGD ile test edildi. Graph sözlük output'u için standart PyTorch
loop kullanılır. Mevcut Trainer/TrainingSession Tensor-output varsayımı graph
multi-output'a uygun değildir. Graph veya sequential automatic resume eklenmedi.

## Ticari paketler

Son wheel ve sdist gerçek arşivleri açılarak 157 modüllük izin listesi ve kaynak
byte'larıyla karşılaştırılır. Kurulu wheel/sdist smoke testleri graph preflight,
construction, Add/Concat/Multiply/WeightedSum, explicit projection, multi-output,
backward, optimizer, strict state_dict ve allocation-free CLI preview içerir.
M3/M4 hazır aileleri ve yardımcı modüller arşiv/import/Registry dışlamalarını
korur. LICENSE ve THIRD_PARTY_NOTICES.md içerikleri doğrulanır.

Yeni iki modül HYSo için public torch/stdlib sözleşmelerinden yazıldı ve root MIT
kapsamındadır. Yeni üçüncü taraf model implementasyonu, pretrained ağırlık veya
dependency alınmadı. Son arşivlerin SHA-256 değerleri yerel distribution-final.json
receipt'inde saklanır. Paketler source değişikliği sonrası güncel içerikle test edildi.

## Bölüm B dosya kapsamı

- `hyso/models/graph.py`
- `hyso/models/graph_ops.py`
- `hyso/models/registry.py`
- `hyso/models/__init__.py`
- `hyso/__init__.py`
- `hyso/cli/models.py`
- `commercial-profile.json`
- `tests/test_graph_builder.py`
- `tests/test_builder_components.py`
- `tests/test_commercial_distribution.py`
- `tests/test_packaging.py`
- `docs/guides/custom-cnn-graph-builder.md`
- `docs/guides/model-registry-builder.md`
- `docs/guides/custom-cnn-builder-components.md`
- `docs/guides/custom-cnn-builder-validation.md`
- `docs/reports/stage13-5-5-validation.md`

## Koruma ve teslim durumu

Bölüm B başlangıcındaki 646 tracked dosya SHA-256 kaydıyla karşılaştırılır.
Önceki **47/47** forensic/debug dosyası, **71/71** native kaynak, üç checkpoint
ve üç dış kabul/index kaydı **6/6** eşleşir. Kapsam dışındaki dosyalar ve
ModelBuilder kaynağı başlangıç byte'larıyla eşleşir. P01–P35 kabul kanıtları korunur.
HEAD Bölüm A commit'idir. Graph değişiklikleri çalışma ağacında, staged alan boş
olarak teslim edilir. Bölüm B için commit veya push yapılmadı.

## Kalan sınırlar

İstenen temel DAG/fusion/multi-output mekanizmalarında eksik kalan seçenek yoktur.
Desteklenen layout'lar NC ve NCHW, batch ortak ve sembolik, spatial boyutlar sabittir.
Arbitrary rank/NHWC, dynamic spatial shape, cross-node parameter tying, Tensor tuple
output ve automatic graph resume destek kapsamına alınmadı. Peak bellek tahmini
belgelenen forward kapsamıyla sınırlıdır. Bazı legacy blokların exact parametre
sayısı None kalır. Root düzeyindeki hatalarda node ID bulunmayabilir.

Cycle tanılaması topolojik sıralama sonunda bloke kalan düğüm kümesini verir.
Bu küme gerçek döngü üyeleriyle döngüden etkilenen downstream düğümleri birlikte
içerebilir. Kesin döngü üyeliği ayrımı yapılmaz. Diagnostic layer_id bu bloke
kümenin alfabetik ilk düğümüdür ve gerçek döngünün üyesi olmayabilir. Son commit
kontrolünde loop_a/loop_b döngüsünün downstream düğümüyle birlikte raporlandığı
ve bağımsız dalın doğrulanmaya devam ettiği ayrıca gözlendi.

Patent, marka, ağırlık/veri hakları ve dependency binary transitive koşullarına
ilişkin önceki hukuki belirsizlikler sürer. Evrensel ticari kullanım veya sıfır risk
garantisi verilmez. Aşama 13.5.6 veya sonraki aşamalar geliştirilmedi.
