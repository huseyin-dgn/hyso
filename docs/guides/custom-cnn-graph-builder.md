# Kullanıcı tanımlı CNN hesaplama grafiği

GraphModelBuilder, Registry bileşenlerini adlandırılmış düğümlerle bağlayan ayrı
bir public API'dir. ModelBuilder schema1/schema2 lineer modellerini kullanmaya
devam eder. Graph mimarisi schema_version=3 ve kind='graph' ile saklanır.

## Düğüm, bağlantı ve çıktı

Bir düğüm bir component, parametre sözlüğü ve sıralı input referansları içerir.
Referans bir graph input adı veya başka düğümün ID'sidir. Aynı referans birden
fazla yerde kullanılabilir. Bu fan-out aynı düğümün çıktısını paylaşır. Modül bir
forward içinde bir kez çalışır, parametreleri toplamda bir kez sayılır.

```python
from hyso import GraphModelBuilder

b = (GraphModelBuilder({'image':[4,5,7]})
     .add('left','conv2d',['image'],in_channels=4,out_channels=4)
     .add('right','pointwise_conv2d',['image'],in_channels=4,out_channels=4)
     .add('joined','add',['left','right'])
     .add('skip','add',['joined','image'])
     .add('classification','classification_head',['skip'],in_features=4,num_classes=3)
     .outputs(classification='classification',features='joined'))
report = b.preview(batch_size=2)
assert report['valid']
model = b.build()
```

`image` `[N,4,5,7]`, features `[N,4,5,7]`, classification `[N,3]` olur.
skip düğümünde ham graph girdisi sonraki çıktıya açık referansla bağlanır.
Graph modelinin çıktısı adları alfabetik sıralanmış Tensor sözlüğüdür.

Tek input için `model(tensor)` veya `model({'image':tensor})` kullanılabilir.
Birden fazla input için tam ad kümesini içeren sözlük gerekir. Input'lar aynı
pozitif batch N değerini, modelin device ve floating dtype değerini paylaşmalıdır.
Model CPU float32 ağırlıklarıyla oluşturulur. Normal `.to(device/dtype)` kullanımı
graph modüllerini ve dtype/device kontrol buffer'ını birlikte taşır.

## Tensor metadata ve rank

Constructor'daki `[C,H,W]` kısaltması NCHW, `[F]` kısaltması NC input spec üretir.
JSON'da spec açık shape, layout, dtype='floating' ve device='model' alanlarıyla
saklanır. Shape batch boyutunu içermez. N semboliktir.

NCHW Registry feature bileşenlerini, sampling işlemlerini ve prediction head'lerini
kullanabilir. Flatten NCHW'yi NC'ye dönüştürür. NC formatında identity, uygun
activation girdileri, dropout, fusion ve pooling='none' prediction head'leri
desteklenir. NC doğrudan Conv2d veya NCHW normalization'a bağlanamaz.

```python
vector = (GraphModelBuilder({'image':[4,3,5]})
    .add('flatten','flatten',['image'])
    .add('activation','prelu',['flatten'],num_parameters=60)
    .add('embedding','embedding_head',['activation'],in_features=60,
         embedding_dim=8,pooling='none',normalize=True)
    .outputs(embedding='embedding'))
```

Her edge raporu symbolic shape, rank, layout, feature anlamı, dtype ve device
metadata'sı taşır. Prediction head çıktısı NC ve classification/regression/embedding
anlamıyla etiketlenir. Genel NHWC veya arbitrary rank dönüşümleri kabul edilmez.
Mevcut permute N/C eksenlerini korur, reshape NCHW içinde explicit CHW gerektirir.
Bu kısıtlar Graph Builder'da da geçerlidir.

## Dört fusion operatörü

Fusion operatörleri Graph Builder'ın kapalı operatör kümesidir. Registry'nin
57 ticari girdisine eklenmez. Lineer ModelBuilder bu isimleri kabul etmez.
Graph preview her fusion'ın metadata ve parametre özetini sağlar.

| Operatör | Matematiksel sözleşme | Parametreler |
|---|---|---|
| add | `x1 + ... + xk`, tüm şekiller eşit. | Parametresiz. |
| multiply | `x1 * ... * xk`, tüm şekiller eşit. | Parametresiz. |
| concat | Explicit seçili eksende birleştirme. Diğer boyutlar aynı. | axis varsayılan 1. NC için 1, NCHW için 1/2/3. Batch ekseni yasak. |
| weighted_sum | `w1*x1 + ... + wk*xk`, sıralı ham skaler ağırlıklar. | weights gerekli, learnable=False varsayılan. |

Her fusion 2..8 ordered input alır. Add, multiply ve weighted_sum broadcasting
yapmaz. WeightedSum ağırlıkları normalize edilmez. Negatif ve sıfır ağırlıklar
kabul edilir. Ağırlık sayısı input sayısıyla eşleşmelidir. Başlangıç değerleri
finite ve [-1e6,1e6] aralığında olmalıdır. Learnable ağırlıklar nn.Parameter,
sabit ağırlıklar buffer olarak state_dict içinde saklanır. Öğrenilebilir değerler
eğitim sırasında başlangıç aralığından çıkabilir. Train/eval aritmetiği aynıdır.

```python
b.add('mix','weighted_sum',['left','right'],weights=[.25,.75],learnable=True)
b.add('wide','concat',['left','right'],axis=1)
```

Concat örneğinde iki `[N,4,5,7]` tensor `[N,8,5,7]` üretir. Aynı bağlantıları
add ile toplamak `[N,4,5,7]` üretir. Floating toplama sırası nedeniyle cebirsel
olarak eşdeğer yeniden yazımlar son bitlerde farklılaşabilir.

## Explicit projection ve resizing

Add'de kanallar uyuşmuyorsa 1×1 projection düğümü ekleyin. Spatial boyutlar
uyuşmuyorsa açık stride veya resize kullanın. Concat için seçili eksen dışındaki
boyutları eşleştirin. Planlayıcı projection, crop veya resize eklemez.

```python
projected = (GraphModelBuilder({'image':[4,5,7]})
    .add('main','conv2d',['image'],in_channels=4,out_channels=8,stride=2)
    .add('projection','pointwise_conv2d',['image'],in_channels=4,out_channels=8,stride=2)
    .add('sum','add',['main','projection'])
    .outputs(features='sum'))
```

Her iki dal `[N,8,3,4]` üretir. Resize sonrası fusion da aynı statik shape
denetiminden geçer. Eski Registry bileşenlerinin parametre, dilation, group ve
normalization kuralları graph düğümlerinde aynen uygulanır.

## Multi-output ve eğitim

Aynı feature çıktısını classification_head, regression_head ve embedding_head
düğümlerine bağlayabilirsiniz. Her head kendi explicit in_features ve çıktı
genişliğini taşır. NC giriş için pooling='none' seçilir. Named output'lar head
veya ara feature referansı olabilir.

```python
b.add('regression','regression_head',['skip'],in_features=4,output_features=2)
b.add('embedding','embedding_head',['skip'],in_features=4,embedding_dim=8,normalize=True)
b.outputs(classification='classification',regression='regression',embedding='embedding')
```

Graph çıktı sözlüğü için açık PyTorch training loop kullanılır. Örneğin
classification loss, regression MSE ve uygun embedding loss ağırlıklı toplam
halinde hesaplanıp bir kez backward uygulanabilir. Her trainable branch'e
gradient ulaşması test edilmiştir. Graph modelinin sözlük çıktısı mevcut Trainer
ve TrainingSession sınıflandırma Tensor-output sözleşmesine uygun değildir.
Graph automatic resume veya yeni TrainingSession loss/metric desteği eklenmedi.

## JSON, stable IDs ve construction

```json
{
  "format": "hyso.model_config",
  "schema_version": 3,
  "kind": "graph",
  "inputs": {
    "image": {"shape": [4, 5, 7], "layout": "NCHW", "dtype": "floating", "device": "model"}
  },
  "nodes": [
    {"id": "stem", "component": "pointwise_conv2d", "params": {"in_channels": 4, "out_channels": 8}, "inputs": ["image"]}
  ],
  "outputs": {"features": "stem"}
}
```

Node ve input adları harfle başlar, devamında harf, sayı veya underscore bulunur.
En fazla 48 karakter kabul edilir. Node ve input ad alanları çakışamaz. Nokta
kabul edilmez. ModuleDict anahtarları `node_<ID>`, state anahtarları örneğin
`nodes.node_stem.weight` olur. Sabit ID'ler ve deterministik topolojik construction
strict state_dict roundtrip sağlar. JSON'daki fiziksel node sırası execution
sırasını değiştirmez. Hazır modellerin category='model' girdileri graph node'u
olarak kabul edilmez.

`add` ID/parametreleri doğrular ve düğüm ekler. Forward reference kullanılabilir.
`outputs` named output haritasını belirler. Yapısal/shape denetimi preview,
preflight, from_config/from_json ve build sırasında tamamlanır. Build yalnız
geçerli graph oluşturur. Config ve JSON detached plain veri taşır.
Duplicate JSON key, non-finite değer, callable, class object ve import yolu
reddedilir. JSON weight veya executable kod taşımaz.

## Sınırlar ve kaynak raporu

| Sınır | Değer |
|---|---|
| Named input/output | Her biri 1..8 |
| Node | 0..64. Sıfır node input'u doğrudan output'a bağlayabilir. |
| Edge | En fazla 256, yinelenen input reference ayrı edge sayılır. |
| Fan-in / fan-out | En fazla 8 / 16. Fusion fan-in en az 2. |
| ID uzunluğu | En fazla 48 karakter |
| JSON byte / nesting | 2 MiB / 16, bounded plain list/dict sözleşmesi |
| Tensor boyutu | Her C/H/W/F boyutu 1..8192 |
| Parametre bütçesi | Conservative toplam en fazla 10M |
| Self-attention | En fazla 1024 token ve N*heads*tokens² ≤ 4.194.304 |
| Advisory aktivasyon eşiği | Varsayılan 16.777.216 eleman |

Parametre sayımı her node için bir kez yapılır. Tek node'un birden çok tüketicisi
parametreleri çoğaltmaz. Ayrı ID ile eklenen aynı component ayrı ağırlıklar üretir.
Yeni hesaplanabilir operatörler exact parameter_count verir. Bazı eski kompleks
bloklarda exact sayı None kalır, mevcut conservative estimate güvenlik bütçesini
uygular. Aggregate exact count bu durumda None olur.

Activation estimate, son tüketicisine kadar tutulan graph referanslarını ve named
output olarak tutulan tensorları sayar. İç operator sınırlarının tamamı, fusion
için 2*fan-in kadar geçici çıktı slotu ve attention için üç matris slotu birlikte
konservatif forward peak-live tahminine dahil edilir. Input/output elemanları
açık batch_size varsayımıyla sayılır. Autograd retention, allocator, workspace
ve eski blokların raporlanmayan iç aktivasyonları bu kapsamın dışındadır.
Gerçek peak RAM/VRAM garantisi verilmez.

Node/edge/parametre/attention sınırları hata, advisory aktivasyon eşiği uyarıdır.
Runtime attention gerçek N ile QKV projection öncesinde sınırı tekrar uygular.
Runtime graph input ve her node output şekli/dtype/device değerini denetler.
Graph'ın ara referans haritası son tüketimden sonra referansları bırakır.
Autograd'ın kendi tuttuğu tensörlere müdahale edilmez.

## Preview, diagnostics ve CLI

`GraphModelBuilder.preflight(config, batch_size=1)` veya `builder.preview()`
ağırlık oluşturmadan rapor üretir. Rapor valid, diagnostics, warnings,
topological_order, inputs, outputs, nodes, edges, parameter_count,
parameter_estimate, activation_estimate ve attention_cost alanlarını içerir.
Node raporunda ports, component metadata, input references/shapes, output shape,
rank/layout, maliyet, status ve hata/uyarı işaretleri bulunur.

Cycle, missing ref, self-loop, duplicate ID, disconnected node ve yanlış output
referansı yapısal hata üretir. Cycle raporu bloke olan ID'leri verir. Bağımsız
dalların scalar ve shape kontrolleri sürer. Geçersiz dalın downstream shape'i None
kalır ve blocked olarak işaretlenir. Yeni GRAPH_STRUCTURE_ERROR,
GRAPH_SHAPE_ERROR ve GRAPH_LIMIT_ERROR kodları önceki ModelConfigError alanlarını
korur. Node bağlamı layer_id içinde taşınır. Graph geneline ait hatalarda ID None
olabilir.

```console
hyso models validate graph.json --json
hyso models summary graph.json --json
hyso models preview graph.json --json
hyso models inspect graph.json --execute --json
```

JSON envelope schema_version=1 ve mevcut exit code sözleşmesi korunur. Mimari
JSON schema_version=3 ayrı bir sözleşmedir. Başarılı metadata code 0, geçersiz
girdi code 2, izin verilmemiş inspect code 3 üretir.

## Kaynak ve kalan sınırlar

Graph ve fusion modülleri HYSo için yazıldı ve root MIT lisansı kapsamındadır.
Registry dışındaki executable component'ler veya hazır model aileleri alınmaz.
Ticari M3/M4 dışlamaları ve dependency bildirimleri korunur. Patent, marka,
veri/ağırlık ve dependency binary haklarına ilişkin önceki belirsizlikler sürer.

Bu aşama tek, ortak sembolik batch ve sabit C/H/W/F boyutlarını destekler.
Genel NHWC, arbitrary rank graph, dinamik spatial shape, Tensor tuple output,
cross-node parameter tying ve automatic graph resume eklenmedi. Mevcut named
input/output Tensor sözlüğü CPU eğitim ve strict state_dict testleriyle doğrulanır.
