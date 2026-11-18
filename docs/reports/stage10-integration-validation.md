# Aşama 10 entegrasyon doğrulaması

Başlangıç branch'i `feature/hyso-models-next`, HEAD `c51be9237ae3bbf545ab8ab1ffb2fe681af7e2fb`.
Tarih 2026-11-12. Çalışma mevcut 10A planını temel alır.

## 10B

Yeni `native` yürütme modu operasyon başına `NativeBinding`, tipli kaynak ve
`OperationPolicy` kullanır. Mevcut CPU policy snapshot'ı korunur. Metadata
preflight, yeniden inceleme, tek kullanımlık yetki ve operation cache eklendi.
Native constructor üzerinden sonuç ailesi/özellik üyeliği kontrol edilir.
P19 retained source ve P22 türetme aynı receipt/source SHA ile kaydedilir.

İlk doğrulama `tests/test_native_execution.py` için **6 passed**.
İlk turdaki iki hata düzeltildi. Python 3.10 için dosya hash'i parça parça
okunur. Retained türetmeler native typed result constructor'ını kullanır.

## 10C

CPU pipeline, sweep, cold-start, profiler/derive, capacity, classification ve
paired kalite rotaları eklendi. P03/P04 tek source kullanır. P12/P13/P14 tek
operator capture'dan türetilir. P29 tek operasyon ve iki açık model etiketi taşır.
Hazır native CPU payload'ları, prediction arithmetic ve mock routing doğrulandı.

## 10D

P09/P15/P16/P18/P34 ve native CUDA varyantlarının dispatch'i eklendi. CUDA
izni, recipe/protocol cihazı, resident tensor cihazı, allocator process izni ve
physical device UUID kapıları metadata üzerinden incelenir. P18 counter/NSYS
okuma retained operasyondur. P15 native cuda:0 kapsamını korur.
Hedefli kümülatif sonuç **29 passed**. GPU kabulü çalıştırılmadı.

## 10E

P19–P25 dispatch'i, compatible tek-fit projeksiyonları ve TrainingSession'ın
mevcut precondition/finalization yolunu kullanan opt-in orchestration eklendi.
P22-only yeni fit başlatmaz. P21/P23/P25 uyumsuz recorder bileşimleri kaynak
eksikliği verir. P21 companion v1/v2 ve isolated P23 reader'ları kapalı listeye
eklendi. P23 Basic diagnostics ayrı tutulur. CUDA saved-activation guard'ı
pre-admission aşamasında korunur. Hedefli kümülatif sonuç **35 passed**.
Fit call count mock üretici üzerinde incelendi. Gerçek eğitim çalıştırılmadı.

## 10F

P30/P31 owned runtime dispatch'i ve P31 kaynaklı P32/P33 retained analizleri
eklendi. Kamera, 7200 saniyelik çalışma ve GPU P31'in native power sampling'i
ayrı izin gerektirir. Native runtime output dosyaları SHA/size receipt alır.
Interrupted fixture `FAILED` kalır ve kaynak payload'ı korunur.
Hedefli kümülatif sonuç **39 passed**. Kamera/uzun çalışma/sensör başlatılmadı.

## 10G ve bütünleşik teslim

P35 mevcut ORT launcher/options yoluna bağlandı. Graph/feed factory, runtime ve
provider seçenekleri source metadata'sında bulunur. Canonical Node/fallback
diagnostics native kayıtta kalır. TrainingSession orchestration, ModelBuilder
config snapshot, Manager schema2 artifact links/idempotency/reopen, unified
report, Advisor, iki model karşılaştırması ve JSON/Markdown/HTML akışı doğrulandı.
Yeni örnek komutları varsayılan olarak dry-run üretir.

Sonuç yazılım entegrasyon kapsamı içindir. Yeni native donanım kabulü
`NOT_EVALUATED` kalır. Aşağıdaki matris uygulanmış rotaları gösterir.

## Alt aşama matrisi

| Alt aşama | Uygulama | Doğrulama ve kaynak gereği |
|---|---|---|
| 10B | Tamam | Kapalı family dispatch, typed binding, one-shot/concurrency, metadata refresh, source/receipt ve authorization kontrolleri |
| 10C | 14 yeni ID tamam | Saved native CPU kaynakları, mock routing, tek-forward profiler ve iki-sample pipeline smoke |
| 10D | 5 yeni ID ve CUDA dispatch tamam | Saved CUDA/counter/device fixtures ve metadata admission. Yeni GPU çalışması yapılmadı |
| 10E | P19–P25 tamam | Saved CPU/CUDA training payload'ları, mock fit ve session success/failure. Yeni fit çalıştırılmadı |
| 10F | P30–P33 tamam | Saved runtime/video/energy, interrupted source, izin ve source reuse kontrolleri |
| 10G | P35 ve kullanıcı akışı tamam | Saved CPU/CUDA ORT, mock launcher, Manager/report/Advisor/comparison ve paketleme |

## P01–P35 ortak yürütücü kapsamı

Her satırda native kaynak ve explicit execution yetkisi gerekir. Retained
binding'ler CPU üzerinde okunur. Tek planın execution device seçimi binding ile
eşleşir. Karma CPU/CUDA kaynaklar ayrı planlarla ve mevcut rapor API'siyle
sunulabilir. Standalone P22/P32/P33 kendi başına fit/runtime başlatmaz.

| ID | Ortak yol | Kaynak ve ölçüm kapsamı |
|---|---|---|
| P01 | trusted_cpu / native | CPU/CUDA benchmark → LatencyAnalysis; retained benchmark analizi |
| P02 | trusted_cpu / isolated_cpu / native | CPU/CUDA ThroughputResult; native completed-work ledger |
| P03 | native | CPU/CUDA PipelineSource capture |
| P04 | native | P03 retained derive veya açık tek pipeline capture |
| P05 | native | CPU/CUDA ColdStartRecipe; fresh interpreter |
| P06 | native | CPU/CUDA batch sweep; CPU master ve explicit points |
| P07 | native | CPU/CUDA resolution sweep; explicit input factory |
| P08 | native | CPU/CUDA precision sweep; unsupported noktalar korunur |
| P09 | native | CUDA BenchmarkResult; current-process peak reset izni. CPU offline okuma |
| P10 | trusted_cpu / native | Process RSS snapshot veya explicit callable workload |
| P11 | trusted_cpu / isolated_cpu / native | CPU logical footprint; artifact I/O opt-in |
| P12 | native | P14 shared/retained operation-count derive |
| P13 | native | P14 shared/retained layer derive; pinned NSYS module reader |
| P14 | native | CPU/CUDA owned profiler; retained ve pinned NSYS operator reader |
| P15 | native | cuda:0 native TransferProtocol owned worker |
| P16 | native | Current-process CPU sampler; explicit physical GPU telemetry |
| P17 | native | CPU/CUDA bounded candidate launcher |
| P18 | native | CPU offline analytic/counter sessions; explicit profile capture |
| P19 | native | CPU/CUDA tek native training fit timing |
| P20 | native | Aynı fit'ten native stages ve timing |
| P21 | native | Aynı loader fit; ayrı SHA-bağlı v1/v2 worker companion |
| P22 | native | Standalone retained P19; aynı yetkili fit'in throughput üyesi |
| P23 | native | CPU/CUDA Basic; in-process ve isolated envelope |
| P24 | native | CPU bounded independent multi-seed quality experiment |
| P25 | native | CPU/CUDA gradient/AMP diagnostics; embedded P19/P20 |
| P26 | native | CPU/CUDA classification; producer/structure capture CPU |
| P27 | native | CPU paired clean/corrupted cohort |
| P28 | native | CPU/CUDA matched precision-quality cohort |
| P29 | native | CPU iki model için tek paired native operation |
| P30 | native | CPU trained processor + explicit video/camera; owned Windows runtime |
| P31 | native | CPU/physical GPU UUID owned Windows runtime; explicit duration/power |
| P32 | native | Standalone CPU retained P31 trend; yetkili P31 sonrası shared derive |
| P33 | native | Standalone CPU retained P31 power analysis; yetkili P31 sonrası derive |
| P34 | native | Aynı recipe ile distinct CPU/physical GPU targets; serial Windows workers |
| P35 | native | CPU/Azure CPU veya CUDA/TensorRT ORT launcher; retained native Node evidence |

Native API'ler yeniden yazılmadı. Kaynak constructor'ları, mevcut native
protocol'ler ve kaynak dosyaların hash'leri otoritedir. P01/P09 BenchmarkResult
istisnası, tekil feature_id kullanan quality/runtime aileleri ve liste kullanan
diğer aileler kapalı registry üzerinden doğrulanır.

## Native format uyumluluğu ve public API

- Inference BenchmarkResult schema1 ile latency/throughput/pipeline/stage/cold/
  sweep/capacity/operator/layer/operation-count/transfer/utilization/roofline
  schema1 aileleri desteklenir.
- P19/P20/P21/P22/P23/P25 native schema1 ve QualityExperimentResult schema1
  korunur. Classification schema1/2/3 mevcut EvaluationResult constructor'ını
  kullanır. Native payload alanları yeniden yazılmaz.
- IsolatedTrainingMemoryResult schema1 probe/segment/worker zarfıyla saklanır.
  Basic metrics ve diagnostics aux alandadır. Historical Extended payload
  kendi reader'ıyla açılır. Yeni common capture Basic kapsamındadır.
- WorkerTraceAnalysisResult v1 ve WorkerTaskAnalysisResult v2 ayrı auxiliary
  native kaynaklardır. Base P21 bağlantısı ve companion SHA'ları doğrulanır.
  Companion'lar standalone P21 EvidenceRecord olarak indekslenmez.
- NSYS module/operator formatları mevcut read-only native parser'a delege
  edilir. Bu reader'lar pinned yan dosyaların erişimini gerektirir.
- RuntimeResult schema1 ve mevcut provider graph/analysis sürümleri korunur.
  Default 0/1 ORT execution kendi kapsamını korur. CUDA 3+10 kabul kapısı ayrı
  canonical native assessment'a aittir.

Yeni public tipler `ArtifactSource`, `OperationPolicy`, `NativeBinding`,
`RetainedSource`, `InferenceSource`, `PipelineSource`, `RecipeSource`,
`EvaluationSource`, `QualitySource`, `TrainingSource`, `ProtocolSource`,
`RooflineSource`, `RuntimeSource`. `TrainingSource.from_session()` hazırlanmış
session'ın epoch/checkpoint seçeneklerini kullanır.

`AnalysisExecutionResult.from_dict()/from_json()` mevcut Manager reader'ına
delege eder. `artifact_sources()/verify_artifacts()` source dosyalarının
explicit registration ve current availability kontrolünü sağlar.
`plan_analysis(..., execution_mode='native')` yeni moddur. Eski mode'lar ve
CPUAnalysisPolicy/TrainingConfig schema1 sözleşmeleri korunur.
Native tracking config operasyon fingerprint'lerini taşır. Kaynak ayrıntıları,
izinler ve native limitler execution artifact'indeki dry-run'dadır.

## Kullanım örnekleri

[Çalıştırılabilir CLI](../../examples/analysis/native_workflow.py),
[örnek komutlar ve hata/veri kapsamı](../../examples/analysis/README.md),
[typed API ve Manager rehberi](../guides/native-analysis-executor.md).

CLI CPU analizleri, multi-selection, fixed prediction analizi, model + labeled
loader quality, training preview, CUDA metadata preflight, retained P31/P32,
ORT provider analizi ve iki snapshot karşılaştırmasını kapsar. CPU P01 örneği
1000 measured sample bütçesini açıkça gösterir. Training örneği bir epoch ve
iki sentetik örnekle sınırlıdır. Capture komutlarında `--execute`, training'de
ayrıca `--allow-training` gerekir. Kullanıcı kendi model, veri, factory ve native
artifact kaynaklarını aynı typed source girişlerine bağlayabilir.

## Test sonuçları

Tekrarlanan test node'ları toplamda bir kez sayıldı. Hedefli düzeltme turlarının
deselected sayıları daha önce geçen node'ları yeniden atlanmış saymaz.

| Kapsam | Benzersiz geçen test |
|---|---:|
| Yeni native routing, source, status, concurrency, lifecycle, Manager ve örnek testleri | 60 |
| Ayrı bounded CPU profiler/pipeline smoke | 2 |
| Explicit saved native P01–P35/ek CUDA ve ORT dosyaları | 39 |
| Mevcut Advisor/raporlama/comparison/Manager/API regresyonları | 225 |
| Mevcut CPU executor güvenli routing/metadata/authorization regresyonları | 39 |
| Wheel/sdist içerik, kurulu paket public API/typing | 4 |
| Toplam | **369** |

Son durumda başarısız test **0**, pytest skip **0**. Kapsam dışında bırakılan
10 benzersiz node eğitim/benchmark veya geniş native çalıştırma içerebilir.
Mevcut regresyonda **67 subtest** geçti. Native Trainer'ın mevcut GradScaler
deprecation uyarısı görüldü ve native AMP API'si değiştirilmedi.

Saved turunun 39 kontrolü 34 feature payload'ı, P14 için 64 MiB sınır reddi ve
dört ek CUDA/ORT varyantından oluşur. Aynı artifact'lerin mevcut source
constructor'ları kullanıldı. P14'ün pozitif capture/constructor yolu ayrı
küçük CPU profiler smoke içinde P12/P13/P14 birlikte doğrulandı.

CPU smoke bir owned worker, bir 1×2×3×3 CPU forward, warmup=0, 30 s native
timeout, 64 KiB input budget ve 1 MiB trace budget kullandı. Pipeline smoke
iki sample içerdi. Paketleme discovery testleri mevcut dört küçük CPU forward'ı
korur. Gerçek eğitim, GPU benchmark, kamera, uzun run ve power sampling
başlatılmadı.

Ortam `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`,
`CUDA_VISIBLE_DEVICES=-1`, `MPLBACKEND=Agg`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1`.
Bütün pytest komutlarında `-c tests/pytest.ini -p no:cacheprovider` kullanıldı.

Temel doğrulama komutları aşağıdadır.

```powershell
python -B -m pytest -c tests/pytest.ini tests/test_native_execution.py -q -p no:cacheprovider
python -B -m pytest -c tests/pytest.ini tests/test_stage10_cpu_smoke.py -q -p no:cacheprovider
$env:HYSO_STAGE10_SAVED_ROOT = "$env:LOCALAPPDATA/hyso-full-resume-20261111-v2/Model-B/stages"
python -B -m pytest -c tests/pytest.ini tests/test_stage10_saved_native.py -q -p no:cacheprovider
python -B -m pytest -c tests/pytest.ini tests/test_packaging.py -k "not installed_distribution_import_and_cpu_forward" -q -p no:cacheprovider
```

Saved test dosyası explicit source dizini verilmeden skip olur. Bu çalışmada
source dizini verildi. Regresyon kapsamı `test_advisor_planning/goals/native/
presentation/recommendations`, `test_model_comparison`, `test_reporting`,
`test_analysis_report`, `test_experiment_workflow`, `test_api_discovery` ve
seçili `test_analysis_execution` testleridir.

Geliştirme turlarında P18'in device descriptor nesnesinin scalar device alanına
atanması düzeltildi. P13 NSYS ayrı native format/reader olarak ele alındı.
Projection hatasının paylaşılan source'u kaybettirmesi, P22-only/P32-only
yetkinin capture'a dönüşmesi ve isolated unsupported source'un embedded timing
üyelerinde FAILED görünmesi düzeltildi. Test fixture'larındaki eksik native
constructor parametreleri, Manager context-manager varsayımı, duplicate model
ID ve prerequisite satırını selected satır sanan assertion düzeltildi.

## Teknik kararlar ve kalan sınırlar

- Native operasyonlar için ayrı `native` modu seçildi. Eski CPU JSON policy'sine
  yeni backend alanları eklenmedi. Yetki ve budget bilgileri typed source
  metadata'sında görünür.
- Tek-fit/capture paylaşımı yalnız native API'nin desteklediği bileşimleri
  kullanır. Uyuşmayan recorder talepleri ayrı fresh episode gerektirir.
- In-process trusted Python için hard timeout/RAM sandbox garantisi yoktur.
  Callback/factory closure içeriği metadata snapshot'ının kapsamını aşabilir.
  Native owned worker'lar kendi timeout/cleanup sözleşmelerini sürdürür.
- P09 current-process allocator reset'i explicit PID izni ister. Diğer allocator
  kullanıcılarıyla eşzamanlı kullanım caller sorumluluğundadır.
- 64 MiB native payload ve 128 MiB Manager/presentation sınırları sürer. Native
  source publication limiti aşılırsa açık failure üretilir. Büyük ownership
  kaynakları kesilerek yayınlanmaz. Saved P14 ownership dosyası 191 MB olduğu
  için bu kaydın doğrudan EvidenceRecord import'u BLOCKED durumundadır.
- P21 native companion reader'ları ve kaynak bağlantıları eklenmiştir. Yeni
  worker tracing/training koşusu çalıştırılmadı. Native worker adapter'ın kendi
  PyTorch/source sürümü ve lifecycle sınırları korunur.
- Canlı ORT resource admission CPU/Azure CPU ve CUDA/TensorRT profillerini
  kapsar. Diğer provider'ların yeni capture yolu unsupported kalır. Retained
  provider payload'ları mevcut native parser kurallarıyla okunur.
- Native GPU capture, CUDA training, transfer/telemetry, cihaz karşılaştırması,
  kamera, 7200-second run ve provider runtime capture bu çalışmada yeniden
  donanım kabulünden geçirilmedi. Kaynak/driver/runtime yokluğu ilgili native
  yolun açık failure/unsupported/prerequisite sonucunda görünür.
- Advisor scalar ve iki model comparison metrik kümeleri mevcut kapsamlarını
  korur. Yeni ID bağlantısı yeni optimizasyon hesabı veya global winner üretmez.

## Değişen dosyalar

Bu çalışma 14 mevcut dosya ve 10 yeni dosya içerir.

| Dosya | Amaç |
|---|---|
| `hyso/analysis/sources.py` | Typed source, operation policy ve artifact identity |
| `hyso/analysis/native_execution.py` | Native admission, one-shot scheduler, source/receipt validation |
| `hyso/analysis/native_adapters.py` | P01–P35 kapalı native dispatch ve shared projections |
| `hyso/analysis/execution.py` | Yeni mode bağlantısı, ticket lock, reopen/artifact API |
| `hyso/analysis/_native.py` | Ek kapalı reader aileleri ve device descriptor projection |
| `hyso/analysis/evidence.py` | Isolated probe/segment identity ve companion indeks sınırı |
| `hyso/analysis/catalog.py` | Native coverage ve device metadata |
| `hyso/analysis/__init__.py` | Yeni public source export'ları |
| `hyso/training/session.py` | Aynı session precondition/finalization ile native fit orchestration |
| `hyso/experiments/workflow.py` | Native source, receipt ve complete row matrix reopen kontrolü |
| `examples/analysis/native_workflow.py` | Güvenli varsayılan CLI workflow örnekleri |
| `examples/analysis/README.md` | Komut, parametre, çıktı, bağımlılık, hata ve veri kapsamı |
| `examples/README.md` | Yeni örnek dizini |
| `docs/guides/native-analysis-executor.md` | Native API, lifecycle ve Manager rehberi |
| `docs/guides/selective-analysis.md` | Güncel coverage bağlantısı |
| `docs/guides/python-api.md` | Native rehber bağlantısı |
| `README.md` | Yeni native modunun keşfi |
| `tests/test_native_execution.py` | Yeni kısa offline/mock ve integration testleri |
| `tests/test_stage10_cpu_smoke.py` | Ayrı bounded CPU smoke |
| `tests/test_stage10_saved_native.py` | Explicit saved source validation |
| `tests/test_analysis_execution.py` | Yeni catalog device coverage beklentisi |
| `tests/test_reporting.py` | Yeni native executor coverage beklentisi |
| `tests/test_packaging.py` | 141 modül ve kurulu native API/typing |
| `docs/reports/stage10-integration-validation.md` | Bu rapor ve başlangıç hash kaydı |

## Bütünlük ve Git

Başlangıçta 617 mevcut dosyanın SHA-256 kaydı alındı. Mevcut 47
forensic/debug/ilgisiz dosyanın tamamı aynı hash'i korur. Silinen başlangıç
dosyası yoktur. 10A planı, `hyso.performance`, `hyso.benchmark` ve native
`hyso/training/trainer.py` dosyaları başlangıç byte'larıyla aynıdır.

70/70 final acceptance index SHA-256 değeri mevcut kabul raporuyla eşleşir,
`86a014ef432ed96a552bfc35a9c65c3bb78714c117e72708ce7b3c8051795bde`.
Final/prior acceptance index'lerinde bulunan üç checkpoint de kayıtlı
SHA-256 değerleriyle eşleşir. Checkpoint deserialization yapılmadı.

Branch `feature/hyso-models-next`, HEAD
`c51be9237ae3bbf545ab8ab1ffb2fe681af7e2fb`. Staged alan başlangıçtaki gibi
boştur. Commit/push ve branch değişimi yapılmadı. Başlangıçtaki iki tracked
değişiklik ve 46 untracked dosya korundu. Bu çalışmanın 14 tracked değişikliği
ve 10 yeni dosyası çalışma ağacındadır. Aşama 11'e geçilmedi.

## Başlangıç koruma kaydı

Plan dışındaki mevcut 47 dosyanın başlangıç SHA-256 değerleri aşağıdadır.

| Dosya | SHA-256 |
|---|---|
| `docs/guides/p23-forensic-plan.md` | `86BEA6B98F7A0229826F58B0713A10E84B8D8BD26DC747BF101ABFC4BB3F0A56` |
| `tools/forensics/native_admission.py` | `00BC31B070C2D48A713507DDDF1D95044E49536D3B1523504CE4FD3762EC199D` |
| `.vscode/settings.json` | `FB8D616967CB453D8E0E455E922B6350C7BC2475AFC429D03C3C6AFC8F63CD6A` |
| `HYSo_G1_feature_planning.patch` | `DDCF3DE148385F21B9011B6E36C3C127B61DC2912024E8B42A2D479DDA4003C4` |
| `HYSo_G2_cpu_bridge_v1.patch` | `39FD6C156123E460F4F7B7B712F91DC9C39C776D6A6B83769A0ABEDCFABC4E7C` |
| `HYSo_G3_isolated_cpu_v1.patch` | `DF7599B2B6E04DE9553A850A5C1EDACDDF3FE77096F4C41DB813BF402C1F94F4` |
| `HYSo_P13_capture_scopes_cpu_only.patch` | `3391212272C8F3BA88ACB61031BC23881D88EF770D9990CB7A7040A00E66E69C` |
| `HYSo_P33_P31_verified_duration_gate_fix.patch` | `98704A5BE9F0B54E070FB7F02AAE24B7463B813B848BA8931C1CCFF339D10843` |
| `HYSo_Packaging_120_strict_inventory.patch` | `5C4C8D1CE340CC3284CE57B1DDC67B75EFA7A0DC2DF718699FFB87CB0EC07797` |
| `apply_model_b_probe.py` | `FB936E6E5A5A0AFB3C122E414CB5F554AE7125F2101988F74CD62797F760BC7C` |
| `apply_p13_native_v1.py` | `97DD850137CD16379C9B410C575D99DFD9BD8C0DF4EC50DD4C4CE40FA79D9911` |
| `apply_single_probe.py` | `D6C59D7A9689F96D162F97EDDE29D21DAFA7B76AE837AFD2DDC0D48C252AFAA2` |
| `apply_step4b.py` | `07D0E0F8DB59C93C50D87C1B316017C57B425F30C896001A706B946A977ED778` |
| `apply_step4d.py` | `47259F2F4CF5F1DE833E079A8D7D4B2DF8F8E2BCB5BD714AEFCC9642361227AB` |
| `apply_step4e.py` | `243C92D2BAF5CD99D54DFDB5DD7B91E990210047AA7F8BBA83D33228F8E08E0E` |
| `diagnose_cuda_constructor.py` | `53DDC9E5810370732F1D67C426F82C326E539BBA85E0AAFC37A00DF7775C646C` |
| `diagnose_cuda_tensors.py` | `43DC2991DBB8CFAD3CA646B91858625A08ED9E9D54E86A790B67BA2DE6D0763C` |
| `diagnose_cuda_throughput.py` | `AA3623A3A1850CABA0CA22D0D33F06B0603CC858AB55DBC253924E713B839B26` |
| `docs/guides/p23-fresh-admission.md` | `5AF842A28990CE604EC9BA23995707D3E07A653911A9B735C0FFC1C855F45246` |
| `docs/reports/p23-basic-cuda-20261111.md` | `9BFB30CBAB1931CEEBD96F8DAB9241008DD44D0BD44A5FDF90D9B4250CCD2A1E` |
| `docs/reports/p23-basic-cuda-pass-20261111.md` | `C0B24BB2B9DE6AA26827E1E8CBD944CA91046A7D80CCDEAF47BF9E92ED76880C` |
| `docs/reports/p23-basic-memory-20261111.md` | `136A0922032E3642B2C5F74FB41167D365C76C34001E20DE823F8A71447DBB7A` |
| `hyso_06_layer_core_budget_update.py` | `3859302F6967E287FDA86919EAD2FD77101F84F79BE8ADFC4694F61FFA31918A` |
| `hyso_06_layer_result_budget_v4.py` | `40F8CA69ADA7A8FAA0949DE06ACCE8B6A8A66CAC9158750FD2A1DF6E83E7EDEE` |
| `hyso_06_layer_safe_apply_v3.py` | `2A47900166F4DF17A5A895E3104EB155F823874551CFAA14CADAD3E8731BEE46` |
| `hyso_06_layer_witness_fix.patch` | `95286017EB0F4D4BAAD102CD44DA5FCDEC99E69DBCA67600F8248E4CF0171A1A` |
| `hyso_06_layer_witness_fix_v2.patch` | `789D079EAF42DCD4051D541857F576ADC237D7E13A01DC603FBFAD01841E848F` |
| `hyso_heartbeat_onedrive_v2.py` | `77B43FA02F37AD3893BF80059DBD088E50831D63346AD087A568D9412E70AD5A` |
| `hyso_independent_acceptance_v1.py` | `36C9A136DFE1F065408F824F332AAE5D914EC347ECB0A84A6B78E36AD2145D4E` |
| `hyso_p18_single_capture.ps1` | `61389123F0B1005FAF5B2EBA691E726BAA7A8AF998C5F5084FFC55CE1120192A` |
| `hyso_p18_single_capture_decimal_comma.ps1` | `D933B8BECBDD2C3945816987D5B2CD94B1E82B9B560CE232314B05C8F997ABB7` |
| `hyso_p18_target_locale_fix_v1.py` | `FB9CEB0D6E941904133ECB1B31E0726CF774972146C197915F9526352AED7F5A` |
| `hyso_p31_p32_overnight.ps1` | `EF551C3D811DD1D9CC129ECC412CD82A001EEDDBF29E0AB02BB77F77217FE9E0` |
| `hyso_prepare_final_manifest.ps1` | `B9437F5B1AC7F75B4050413D31724B9D7C4A0F8F2E60CA4A9EEFEE9F574A1DC8` |
| `hyso_provider_cpu_status_fix_v1.py` | `6E4FF43F8063982C1A39747C373098555056A4CF0390CB9DE94327E42CB7EBA2` |
| `hyso_reconcile_existing_runs.py` | `9054C35E4340211C65BBEF2F396937FE31B817ECB4BCAA0C61690F4CDEC39B61` |
| `hyso_reconcile_existing_runs_v2.py` | `92B462EE47FF9B9F6A1F0CE2D488EED209FB500F29BA5D8C9C06AFFEE6102B75` |
| `hyso_remaining_audit.py` | `E8C14DAA44CDF7679E9F13249AF9705E5CDBA1B08769BD284AA2683D5859DC28` |
| `p13_capture_finalize.py` | `7BF57867CC3AF63B655A0DCE3EAD8CAC6A90A547D4A3E6807C9D177F8AD69F05` |
| `p13_capture_native.py` | `16535176C076788B68050E1A53C5D07AE3A7D0196E9A4EB99DD8D4C90C3B8EF0` |
| `p13_capture_worker.py` | `7C58437B6AD25520DA8C8C835E9922626833D6D0457E8B76ABDB4FCED628B616` |
| `test_p13_capture_native.py` | `1ACC14F0A27B0174A877BF0087942E1F06189F8FE209E7350E25620418C62216` |
| `tests/forensics/test_p23_forensic_runner.py` | `5E1C7FD668CC31FC450ADC071851AA2D18016DD19E11C7303472A7CA7C88ABA5` |
| `tests/forensics/test_p23_fresh_admission.py` | `09BB046F17E22C35EBF29714397B2E4992634220C5DAE3D204864B7472B68358` |
| `tools/forensics/p23_forensic_runner.py` | `3275A9021E3382D38686EE49C08DDED3DFB54DFB5DC87ABBCC9E87770CEE9AB9` |
| `tools/forensics/p23_forensic_windows.py` | `DBD9E0D41F5C3BAA37EC919D70AC9BF661E9719B445816D28F455EFB24F2EC56` |
| `tools/forensics/p23_forensic_worker.py` | `3C188FE3BEC7944C1A395397DA81DA62C6CB8BCF575D1A0137A2562F5E2EC822` |
