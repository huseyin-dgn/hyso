> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# HYSo Studio

Studio brings the existing HYSo builders, Registry metadata, retained results, reporting and selected CPU operations into a local web workspace. Python's standard-library HTTP server serves original, buildless JavaScript and CSS. The installed HYSo runtime supplies PyTorch, analysis, training and report rendering. Studio adds no runtime dependency or frontend build step.

The Reborn acceptance and functional audit are documented in [validation](../reports/stage13-5-6-reborn-validation.md) and [the twelve-screen audit](hyso-studio-reborn-audit.md). Frontend/backend workflow contract **3** is required before sending any operation. Overview now includes real test, training and architecture-comparison activity; runtime testing has its own selection/launch/results workspace, separate from structural validation.

## Start and stop

After updating Studio source, restart the Python server as well as reloading the browser. The running process freezes its frontend assets to keep UI and backend contracts consistent. A restart-required message means the existing server is older; it does not mean your saved model data should be removed. Details panels now start closed on each page even when an older global auto-open preference exists.

Use the Python environment containing HYSo and its dependencies.

```powershell
python -m hyso.studio --port 8765
```

Open `http://127.0.0.1:8765`. Stop the application with Ctrl+C in its launching terminal. The process cancels queued jobs and drains the active CPU operation before closing its database. Closing the browser leaves the explicitly launched server process running. Jobs have no scheduling service outside that process.

The default workspace is `%LOCALAPPDATA%/HYSo/Studio` on Windows and `~/.local/share/HYSo/Studio` on other systems. A trusted launch command can select another directory.

```powershell
python -m hyso.studio --port 8877 --workspace C:\HYSoData\Studio
```

The existing ExperimentManager local-store policy applies to this directory. Choose a dedicated local path outside synchronized folders, network shares and reparse points. A port conflict exits with code 3 before creating the workspace. Invalid CLI arguments use argparse exit code 2. Existing HYSo CLI exit contracts remain in their existing command handlers.

## Workspaces

The default `#overview` screen explains HYSo, local processing, static estimates and actual measurements. Model/version/test/training/comparison totals come from the real workspace; an empty workspace shows zeros and an invitation to create the first CNN. The six workflow links open actual application pages. `#home` remains the separate My models screen. New model opens editable convolution/classification layers with fresh weights. Select a model once; Train, Test, Benchmark, Compare, Recommendations and Reports keep its context.

| Workspace | Implemented path |
| --- | --- |
| Model Architect | Schema3 graph editing, Core preflight, draft persistence, JSON and Python export |
| Component Knowledge | Registry and fusion definitions, constructor fields, constraints, source references |
| Training & Experiments | Validated training imports, epoch curves, explicit sequential or selected graph-head CPU training, Studio-owned ExperimentManager records |
| Testing & Validation | User-selected structure, forward, observed-gradient and strict state_dict checks; retained JSON/JUnit PASS/FAIL/XFAIL/SKIP |
| Performance Profiler | Validated records and explicit sequential or selected graph-output CPU BenchmarkRunner jobs |
| Model Evaluation | Existing classification results and evaluation of caller-supplied predictions |
| Model Comparison | Two to five saved architectures without training; separate two-to-eight real-result comparison with comparability gates |
| Recommendations | Core diagnostics, copied channel correction preflight and retained native Advisor output |
| Reports | Model-hash-bound JSON, HTML and Markdown reports with filters and provenance |
| Project Settings | Persistent GLOBAL, PROJECT, MODEL, NODE and RUN settings with immutable Core policy |

Each result view filters on the current configuration SHA-256 and, when selected, immutable version identity. Empty stores have empty result panels and unavailable measurements. Imports ask which model produced the result; an external hash remains available in technical details. Native source consistency and source authenticity have separate fields.

Training, testing, performance and evaluation start with an explanation and setup form. Select a supported local JSON data file, choose the task/output/loss and adjust bounded training settings. Check setup validates without allocating tensors or submitting a job. Start is a separate explicit action. Any setup edit requires another check. The short CPU profile initializes fresh weights and supports up to five epochs, 32 samples and 65536 input elements. It is not a checkpoint resume workflow.

Basic views show model names, plain constraints and meaningful status messages. Dimensions and full technical evidence are opt-in. A successful synthetic CPU test confirms its checked execution conditions; it says nothing about dataset accuracy. Evaluation uses supplied classification predictions and targets.

## Verification status

Backend integration, editor state operations and installed packaging have dedicated tests. Real Chromium tests use the production HTTP server and isolated workspaces. They cover pointer dragging, port connections, all 60 graphical component parameter forms, import/export, undo/redo, bilingual navigation, reports, comparison and short graph training/benchmark jobs. Desktop and narrow-screen screenshots are retained outside the repository. The stage report records the latest results.

The CPU graph adapter supports one NC/NCHW input and one explicitly selected classification loss/head during a training run. Every configured graph output is still computed. Multi-loss optimization and multi-input datasets are outside this adapter. Regression and embedding evaluation producers remain unavailable in the existing evaluation API. Arbitrary external experiment-store attachment is unavailable. P01–P35 metadata and supported closed native result readers are listed in the inventory.

## Türkçe kullanım

Açılış sayfası **HYSo — Mühendislik Çalışma Alanı** açıklayıcı genel bakışıdır. Modellerim ayrı sayfadır. Zümrüt/jade ve bakır tasarım token'ları light/dark temalarda ortaktır. Gerçek model/sürüm/iş/kayıt sayıları gösterilir; boş alanda örnek başarı veya performans grafiği üretilmez. [Tasarım sistemi](hyso-studio-design-system.md).

Kayıtlı model ekranlarında sonuçlar model kimliği ve taslak revision'ı veya değiştirilemez sürüm kimliğiyle sınırlandırılır. Aynı mimari hash'i farklı modellerin kaynak ilişkisini değiştirmez. Kaydedilmemiş çalışmanın sonucu ayrı `draft/kimlik` snapshot bağlamında kalır; otomatik model kaydı yapılmaz. Kısa CPU çalıştırma profili ayrıca en çok 1.000.000 Core parametre üst sınırı ve 1.000.000 aktivasyon elemanı ister; bu tahminler toplam süreç belleği garantisi değildir.

[Test seçimi](hyso-studio-test-selection.md) ekranında yapı kontrolü, ileri hesaplama, gözlenen gradient ve ağırlık roundtrip kontrollerini seçebilirsiniz. Veri uyumluluğu ve training readiness gerçek dosyayla Eğitim/Performans kurulumunda kontrol edilir. Tam loss/gradient kapsamı, native kabul çalışması veya CUDA testi sunulmaz.

HYSo bağımlılıklarının kurulu olduğu Python ortamında `python -m hyso.studio --port 8765` çalıştırın ve `http://127.0.0.1:8765/` adresini açın. Windows varsayılan verisi `%LOCALAPPDATA%/HYSo/Studio` içindedir. `--workspace` ile ayrı yerel dizin seçilebilir. Ctrl+C kuyruktaki işleri iptal eder ve aktif CPU işinin tamamlanmasını bekler.

Model Mimarı'nda düğüm ekleyin, portları bağlayın ve adlandırılmış çıkışları seçin. Yeni düğümün adlandırılmış çıktıya ulaşması gerekir. Parametre ve şekil hataları Core ön doğrulamasından gelir. Taslağı kaydetmek canvas konumlarını ve düğüm etkinliğini de korur. JSON dışa aktarımı Core sözleşmesini, taslak paketi Studio düzenleme durumunu taşır.

Bilgi Merkezi 57 ticari mekanizma ve dört fusion işleminin iki dilli açıklamalarını, matematiğini, parametrelerini, kısıtlarını ve kaynaklarını sunar. TEMEL düzey kullanım ve sınırlamaları, TEKNİK düzey matematik, örnek Python ve kaynak ayrıntılarını gösterir. Arama ve kategori filtresi kullanılabilir.

Eğitim, Test, Performans ve Değerlendirme ekranları güncel mimari hash değerine bağlı gerçek kayıtları gösterir. Dış kaydın model, veri/split ve cihaz ilişkileri açıkça beyan edilir. `IMPORTED_UNVERIFIED` kaynak özgünlüğünü, `MEASURED` yerel iş gözlemini belirtir. Eksik değerler `UNAVAILABLE` kalır. Test işinin sentetik sıfır girdisi kayıt içinde açıkça yazılıdır.

Graph eğitiminde `head_mapping.output` ve göreve uygun `head_mapping.loss` seçin. Benchmark yalnız çıkış adı ister. Profil CPU fp32, en çok beş epoch, batch 32 ve 65536 veri elemanıdır. Eğitimde tek classification başlığı seçilir. Core'un graph ve dikkat bütçeleri korunur.

Ana sayfadan model oluşturun, düzenleyin ve taslağı kaydedin. Doğrulanmış sürüm oluşturmak mimariyi değiştirilemez bir kayıt olarak saklar; eski sürüm korunur. Model eylemleri seçili model/sürümü ilgili ekrana taşır. Eğitim formunda yerel veri dosyasını seçip ayarları kontrol edin, ardından açıkça başlatın. Ham kaynaklar Teknik ayrıntılar altında bulunur.

Model Karşılaştırma'da eğitim gerektirmeden 2–5 kayıtlı taslak/sürüm seçebilirsiniz. Aynı mimari açıkça belirtilir. Sonuç karşılaştırması ayrı modda 2–8 gerçek kayıt kullanır. Koşullar eksik veya farklıysa doğrudan karşılaştırma uygunluğu verilmez. Öneriler kanıt, durum ve risk taşır. Eksik kanıt tek özet altında toplanır. Doğrulanan graph adayı açık kullanıcı eylemiyle yeni taslakta açılır. Önerilen deneyler otomatik başlamaz.

Raporlar JSON, HTML ve Markdown üretir. Model, kayıt türü, run/experiment, tarih, metrik, grafik, dil ve hassasiyet seçilebilir. PDF/CSV kullanılamaz. Güvenlik ve ticari dışlama politikaları salt okunurdur. Uygulama normal iş yönetiminde checkpoint/full-resume altyapısı kurmaz.

See the stage validation report for the exact tested scope and remaining work. [Stage validation](../reports/stage13-5-6-validation.md)
