# Stage 13.5.22 — CPU beta readiness, packaging ve kullanıcı kabulü

Tarih: 2026-11-18 (Europe/Istanbul). Yalnız yerel çalışma; yayın/release/push yok.

| Aşama | Sonuç | Doğrulanan kapsam |
| --- | --- | --- |
| A — Packaging / Distribution | PASS | Wheel/sdist, açık allowlist, metadata, kurulu public importlar ve CLI |
| B — Clean Environment User Acceptance | PASS | Ağsız izole venv; küçük CPU eğitim/resume/evaluation/store/comparison/advice/report akışı |
| C — CPU Beta Readiness | CPU_BETA_READY | Yalnız aşağıdaki bounded CPU profilleri; genel platform veya CUDA kabulü değil |

## Başlangıç ve koruma

Branch `feature/hyso-models-next`; başlangıç HEAD tam olarak
`18d1378adf24875047d23c5dbfdac6d352f43e1c`. Index boştu. Önceki sekiz
commitin varlığı/HEAD ancestry kontrol edildi; tarih yeniden yazılmadı.
Başlangıçta iki tracked değişiklik ve 45 untracked gerçek dosya vardı.

QA kökü: `C:/Users/hdgn5/AppData/Local/hyso-stage13-5-22-20261118-163626`.
Bu raporda `QA/` bu dış dizini ifade eder. Kalıcı yeni build, install, test,
SQLite, checkpoint, rapor exportu ve loglar bu dizindedir. Pip ve bazı mevcut
testlerin kısa ömürlü geçici dosyaları ayrıca sistem temp dizininde, repository
dışında oluşturuldu. Mevcut repository çıktıları temizlenmedi. Önceki harici QA
dizinlerine yazılmadı.

`QA/baseline.json`, 809 tracked/nonignored dosyanın başlangıç byte hash'lerini,
status/HEAD/ancestry bilgisini ve ayrıca mevcut 19 checkpoint/model/SQLite
artifactının hash'lerini kaydeder. Önceki 13.5.12 `protected47.json` içindeki
dirty 47 ve forensic 47 kayıt başlangıçta birebir eşleşti. Bu iki grup
örtüşür; 94 ayrı kullanıcı dosyası iddiası değildir. Final koruma kontrolü
`QA/preservation-final.json` dosyasındadır. Tarihsel tracked QA/performance/CUDA
kayıtları ve korunan dosyalar değişmedi; eski checkpointler eğitime yüklenmedi.

## A — Paketleme ve düzeltme

`pyproject.toml`, `setup.cfg`, `setup.py`, `MANIFEST.in`, commercial profil,
public importlar, CLI ve mevcut packaging testleri incelendi. Metadata korundu:
`hyso==0.2.0`, Python `>=3.10`, setuptools backend, MIT ve üçüncü taraf lisans
bildirimleri; console script `hyso = hyso.cli:main`; package-data `hyso/py.typed`.
Runtime bağımlılıkları Torch `>=2.2.2`, torchvision `>=0.17.2`, NumPy,
matplotlib, seaborn, scikit-learn, tqdm, torchsummary. Optional deployment extra
ONNX `1.20.0` / ONNX Runtime `1.23.2` olarak kaldı. Dependency metadata değiştirilmedi.

Gerçek hata: kabul edilmiş P12 karşılaştırmasının kullandığı
`hyso/analysis/_p12_comparison.py` ve `hyso/advisor/p12_observation.py`
allowlist dışında kalmıştı. İlk wheel'in izole kurulumunda public
`analysis.compare_models(..., features='P12')` çağrısı, eksik kanıtı güvenli
yorumlamak yerine `ModuleNotFoundError` veriyordu. Reprodüksiyon
`QA/pre-fix-installed-p12.log` içindedir.

Yalnız bu iki runtime modülü commercial allowlist'e eklendi. Development
exclusion sözleşmesi genişletilmedi. Envanter testine eksik kaynak yolları
eklendi; eski `164` sabiti gerçek son envanter olan `168` ile düzeltildi
(başlangıç allowlist'i zaten 166 modüldü). Kurulu wheel ve sdist regresyonu,
public P12 observation importunu ve boş kanıtla iki
`INSUFFICIENT_EVIDENCE` / `null` fark satırını kontrol eder.

Final arşivler `QA/packaging-final/packaging0/dist/` altındadır:

| Dosya | Byte | SHA-256 |
| --- | ---: | --- |
| `hyso-0.2.0-py3-none-any.whl` | 769730 | `ad1c7ff2b89413bbfc32400de757f0a6b551b20b5ae05660b60a2cfa88e4ee2d` |
| `hyso-0.2.0.tar.gz` | 677822 | `83cd08f89bae5ec90375e6062d3d52f70ceb83ca003585dbc55b47fa492ed2e6` |

Wheel 168 Python modülü / 176 dosya, sdist 168 package modülü artı `setup.py` /
183 dosya içerir. Package Python byte'ları kaynakla karşılaştırıldı; tümü eşleşti.
Support/metadata üyeleri ayrıca incelendi. On development modülü dışarıda;
`.git`, QA arşivleri, dataset, kullanıcı checkpoint/SQLite dosyaları, dump,
cache veya özel veri pakete girmedi. Testlerde istenmeyen içerik sentinel'ları
da dışlandı. Credential-pattern kontrolünde eşleşme yoktu; bu bağımsız bir
güvenlik sertifikasyonu değildir. README/rehber/örnekler mevcut commercial
profil gereği dağıtımda yoktur; sdist build config, profil ve lisans dosyaları
vardır. Metadata açıklaması mevcut kısa açıklama olarak korunmuştur.

## B — İzole kurulum ve kullanıcı akışı

Ortam: Windows x64 (`Windows-10-10.0.26300-SP0`), Python 3.10.18. Sistem Python'u
değiştirilmedi. `QA/venv/pyvenv.cfg` içinde
`include-system-site-packages = false`; HYSo ve runtime bağımlılıkları bu venv'e
kuruldu. Çalışma dizini repository dışındaydı, `python -I -B` kullanıldı;
`sys.path` içinde checkout yoktu. Base interpreter'ın stdlib/DLL yolları
venv'in normal Python tabanıdır; base site-packages paylaşılmaz.

Yerel pip HTTP/wheel cache'inden offline wheelhouse hazırlandı. Kurulum
`pip --isolated install --no-index --find-links ...` ile yapıldı; indirme veya
sistem genelinde kurulum yok. İlk offline çözümlemedeki eksik yerel wheel
adayları cache'in diğer uyumlu tag'lerinden sağlandı; ağ kullanılmadı.
`QA/constraints.txt`, `venv-install.log` ve `venv-freeze.txt` sürümleri kaydeder.
Torch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 2.2.2, matplotlib 3.10.8,
seaborn 0.13.2 ve scikit-learn 1.7.2 kullanıldı. CUDA-capable wheel kullanılması
GPU çalıştırıldığı anlamına gelmez; context oluşturma testlerde yasaklandı.

Gerçek import kaynağı:
`QA/venv/Lib/site-packages/hyso/__init__.py`.
ModelBuilder, GraphModelBuilder, TrainingConfig, TrainingSession, Trainer,
ExperimentManager, evaluation/analysis/comparison/advisor/reporting girişleri
kurulu paketten doğrulandı. Import ve help sırasında forward, training,
benchmark veya CUDA başlatılmaması ilgili fence testleriyle kontrol edildi;
installed harness ayrıca ağ çağrılarını reddeder.

Mevcut `graph_evidence_workflow.py` örneğinin byte-identical kopyası QA'ya
alındı; örnek modülünün checkout'tan import edilmesi gerekmedi. Public API ile:

- İki küçük tek-input Graph, model başına bir epoch / iki update eğitildi;
  output contract, strict checkpoint restore, gerçek sınıflandırma evaluation,
  declared dataset/split/preprocessing bağları kontrol edildi.
- İki named input / seçilmiş tek Tensor head akışı iki epoch / dört update
  bütçesinde ilk epoch checkpointinden devam etti.
- İki named input / iki ayrı weighted classification head için A modeli
  ilk epoch checkpointinden devam etti; B varyantı aynı iki epoch bütçesinde
  eğitildi. İki model de dört update yaptı. Global accuracy üretilmedi.
- ExperimentManager'da eğitim/evaluation ve checkpoint artifactları kaydedilip
  SHA kontrolüyle açıldı. Yanlış artifact byte'ları, yanlış checkpoint kimliği
  ve yeniden digestlenmiş tutarsız association negatif kontrollerle reddedildi.
- Kayıtlı veriden Analysis, iki-model/per-head Comparison, manuel Advisor,
  TR/EN JSON/Markdown/HTML üretildi. Yeniden okuma sırasında model construction,
  forward, fit, profiler ve checkpoint load çağrıları fence ile engellendi.
- Native state/model/checkpoint kimlikleri caller model etiketinden ayrı kaldı.
  Import edilmiş Graph otoritesi `unverified`; presentations
  `presentation_only=true`, `authoritative_evidence=false` kaldı. Yeni native
  P12 capture yapılmadı; eksik receipt satırları null farkla yetersiz kanıttır.

Clean-install sırasında ikinci gerçek hata bulundu: Windows legacy redirected
stdout, insan-okur rapordaki Unicode eksi işaretini kodlayamadığında CLI hata
veriyordu. `hyso/cli/main.py`, yalnız `UnicodeEncodeError` durumunda text stream'i
UTF-8'e geçirerek aynı metni yazar. Uyumlu stream davranışı ve ASCII-safe machine
JSON korunur. Success/error yollarına iki cp1254 regresyonu eklendi.
Final gerçek module/console subprocess kontrolleri başarılıdır; Unicode rapor
stdout'u newline normalizasyonu dışında UTF-8 dosya exportuyla aynıdır.
Unicode model adı içeren `python -I -m hyso analysis plan` da geçti.

Final CLI değişikliğinden sonra eğitim tekrarlanmadı: final wheel ile önceki
kurulu workflow wheel'i arasında yalnız `hyso/cli/main.py` byte'ları farklıdır.
Kayıtlı sonuçlar final kurulu wheel ile yeniden doğrulandı. Sdist de aynı izole
venv'de `--no-index --no-deps --no-build-isolation --no-cache-dir` ile yeniden
build/install edildi; public import/P12 negatif yolu ve `pip check` doğrulandı.

## Test sonuçları ve kanıtlar

| Son kontrol | PASS | FAIL | Kanıt |
| --- | ---: | ---: | --- |
| Wheel/sdist metadata/content, installed public discovery/P12, CLI, allowlist | 8 | 0 | `QA/packaging-final.xml`, `packaging-final.log` |
| Küçük Graph örnekleri, combined resume/evaluation/store/report ve artifact/CLI reader negatifleri | 33 | 0 | `QA/acceptance.xml`, `acceptance-tests.log` |
| CLI temel sözleşmeler + iki Unicode regresyonu + Graph evaluation/comparison reader | 103 | 0 | `QA/cli-unicode.xml`, `cli-unicode.log` |
| Installed import/isolation, workflow sonuçları, workload fence, TR/EN, CLI subprocess assertions | 50 | 0 | `QA/installed-acceptance.json`, `installed-complete.log` |

Bunlar ayrı run sayılarıdır; örtüşen CLI testleri benzersiz toplam olarak
toplanmamalıdır. Sdist install/import ve iki `pip check` ayrıca geçti.
İlk packaging run'ı 1 FAIL / 9 PASS idi; eksik inventory yolları yakalandı.
Ara düzeltmede yanlış 166 sayımı 1 FAIL / 7 PASS üretti; tam gerçek envanter
168 olarak doğrulanıp final run 8/8 geçti. Hiç assertion gevşetilmedi.
Kurulu acceptance harness'inin örnek konumu, desteklenmeyen ham multi-head CLI
reader ve sabit glyph varsayımları düzeltildi; bu ara başarısız loglar korundu.
Gerçek CLI Unicode hatası ayrıca kaynakta düzeltildi. Eğitimdeki sekiz mevcut
GradScaler deprecation uyarısı korunur; CPU scaler disabled, CUDA context yok.

P01–P35 tarihsel kabulü, P31 7200 saniyelik testleri, full repository suite,
uzun benchmark/profiler, CUDA diagnostic veya T09/T10 çalıştırılmadı.

## C — Beta kapsamı, sınırlar ve engeller

Doğrulanan CPU beta profilleri: sequential Builder'ın kurulu küçük sınıflandırma
modeli; bounded Graph/GraphTensorOutput; tek-input classification; iki açık named
input; iki açık weighted classification loss ve birleşik iki-input/iki-head
CPU fp32 akışı; belgelenmiş strict v2 resume; mevcut per-head evaluation;
ExperimentManager artifact roundtrip; retained Analysis/Comparison/Advisor;
TR/EN rapor ve mevcut CLI grupları. Tek/iki-head binary/multiclass, SGD/Adam
ve native P12'nin daha önce kabul edilmiş dar profilleri korunur; bu aşama
bunların bütün kombinasyonlarını yeniden kabul etmiş sayılmaz.

Native P12 MAC/FLOP karşılaştırması için mevcut receipt/coverage/identity
politikası gerekir. Bu aşama paket girişlerini ve eksik kanıt davranışını
doğrular; yeni P12 sayım veya cihaz performans iddiası üretmez.

Sınırlar: genel multi-task/distributed eğitim, belirlenmemiş Graph task/topoloji
birleşimleri ve rehber dışındaki resume kombinasyonları destek kapsamına
alınmadı. Ham multi-head JSON, Python
`MultiHeadGraphEvaluationResult.from_json` ve ExperimentManager ile açılır;
CLI genel reader'ı bu formatı desteklemez. CLI, kayıtlı tek-head Graph
evaluation/comparison ve PresentationReport için doğrulandı. Python ve CLI
tüm seçeneklerde eşdeğer değildir. Minimum Python/dependency metadata'sı tüm
OS/Python/Torch sürümlerinde test edilmiş uyumluluk anlamına gelmez.

CUDA B4 `ROOT_CAUSE_UNRESOLVED`, T08 admission `BLOCKED`, GPU execution
`NO-GO` olarak kalır. CPU beta kararı bunları kaldırmaz; eski CUDA başarıları
ve diagnostic kayıtları değiştirilmedi.

İncelenen blocker alanları: clean install, import, CLI entry point, bounded CPU
workflow, checkpoint/experiment integrity, evidence authority, sahte PASS,
paket içeriği ve API/dokümantasyon uyumu. Bulunan iki hata düzeltildi; yukarıdaki
test edilen CPU kapsamında kalan release blocker yok. Karar:
**CPU_BETA_READY**, yalnız bu rapordaki kapsam için.

README ve iki Python-first rehberi, wheel/checkout kurulum ayrımı, dağıtılmayan
örnekler, 168-modül commercial profil ve CLI reader sınırıyla güncellendi.

## Yerel commitler ve bitiş

| Mantıksal değişiklik | Commit | Dosyalar |
| --- | --- | --- |
| P12 packaging | `4a72261707da8004d848a78e3575d20c795aa676` | `commercial-profile.json`, `tests/test_packaging.py` |
| Windows Unicode CLI | `521222bf8a39dbe66e4892f8ccf12de425b98526` | `hyso/cli/main.py`, `tests/test_cli.py` |
| Kurulum/kapsam ve bu rapor | Bu raporu ekleyen documentation commit'i; tam SHA `QA/final-state.json` ve son kullanıcı yanıtında | `README.md`, iki Python-first rehberi, bu rapor |

Bitiş HEAD bu raporun documentation commit'idir; kendine referans veren SHA
rapor içine yazılarak yeni bir commit döngüsü oluşturulmaz. Final index boş;
status yalnız başlangıçtaki 47 kullanıcı dosyasıdır. Koruma ve commit/index
kontrolleri `QA/preservation-final.json` / `QA/final-state.json` içinde tutulur.
Push, yayın, GitHub release, sistem genelinde kurulum veya GPU execution yok.
Stage 13.5.22 sonunda duruldu; yeni stage başlatılmadı.
