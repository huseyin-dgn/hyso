# Native analiz örnekleri

Checkout kullanımı için repo kökünde `PYTHONPATH` ayarlayın. HYSo bağımlılıkları
ve Python 3.10+ gerekir. HTML export Matplotlib kullanır.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python examples/analysis/native_workflow.py --help
python examples/analysis/native_workflow.py cpu
python examples/analysis/native_workflow.py cpu --features P03 P04 P26
python examples/analysis/native_workflow.py quality
python examples/analysis/native_workflow.py quality --execute --output C:/Temp/hyso-quality-demo
python examples/analysis/native_workflow.py cpu --features P10 P11 P26 --execute --output C:/Temp/hyso-native-cpu
python examples/analysis/native_workflow.py artifact --input saved-pipeline.json --features P03 P04 --execute --output C:/Temp/hyso-retained
python examples/analysis/native_workflow.py artifact --input saved-P31.json --features P31 P32 --warmup-seconds 60 --execute --output C:/Temp/hyso-trend
python examples/analysis/native_workflow.py training
python examples/analysis/native_workflow.py training --allow-training --execute --output C:/Temp/hyso-training-demo
python examples/analysis/native_workflow.py cuda-preflight --factory your_package:make_cpu_model --device cuda:0
python examples/analysis/native_workflow.py providers --factory your_package:onnx_bytes_and_feed --directory C:/Temp/hyso-new-ORT-run
python examples/analysis/native_workflow.py artifact --input saved-P35.json --features P35 --execute --output C:/Temp/hyso-providers
python examples/analysis/native_workflow.py compare --baseline A-execution.json --candidate B-execution.json --baseline-model A --candidate-model B --features P01 P02 P26 --output C:/Temp/hyso-comparison
```

| Komut | Amaç ve parametreler | Beklenen çıktı | Veri, bağımlılık ve hata kapsamı |
|---|---|---|---|
| `cpu` | `--features` ile CPU analizleri ve çoklu seçim | Varsayılan dry-run JSON. `--execute` ile execution/evidence ve rapor dosyaları | Küçük sentetik CPU modeli. P01 tail örnek gereğini korur. P03/P04 tek pipeline source kullanır. Eksik RSS backend unsupported kalır. |
| `artifact` | `--input`, `--features`, isteğe bağlı `--sha256` | Native validation, aynı payload veya açık türetme ve rapor | Kullanıcı mevcut artifact sağlar. Eksik dosya, SHA uyuşmazlığı, yanlış format/özellik ve 64 MiB sınırı hata verir. |
| `cpu --features P26` | Classification kalite hesabı | EvaluationResult ve native evidence | Sabit örnek etiket/tahmin aritmetiği. Gerçek model kalitesi için kendi etiketli loader/cohort binding'inizi kullanın. |
| `quality` | Model + sequential labeled DataLoader ile classification | Dry-run veya açık execution ile producer evidence içeren EvaluationResult schema2 | Dört sentetik örnek ve untrained CPU demo modeli. Kendi model/veri kaynağınızı aynı EvaluationSource girişine bağlayabilirsiniz. |
| `training` | P19/P20/P22 tek episode | Dry-run veya açık iki bayrakla tek küçük fit | İki sentetik örnek, bir CPU epoch. `--allow-training` kaynak izni, `--execute` özellik iznidir. State ilerler. Hata sonrası fresh session gerekir. |
| `cuda-preflight` | Factory ve explicit device metadata incelemesi | Kaynak/izin blocker'ları içeren dry-run | Factory import edilmez. CUDA capture izni ayrıca API'den verilir. GPU/driver kullanılabilirliği execution sırasında native admission'a bağlıdır. |
| `artifact --features P31 P32` | Mevcut runtime ve `--warmup-seconds` ile trend | P31 snapshot ve P32 descriptive slope | Kaydedilmiş P31 gerekir. Kısa kaydın native analizi kendi süresini korur. 7200 saniye kabulü kendi kayıt kapsamında kalır. |
| `providers` | ONNX bytes/feed factory ve yeni output directory | Dry-run. `--execute` ile native CPU ORT profili | Windows owned worker, optional onnx/onnxruntime gerekir. Factory mevcut self-contained graph/feed döndürür. Missing runtime/provider native sonucu korunur. |
| `artifact --features P35` | Hazır ORT Node ledger analizi | Retained provider evidence ve rapor | ONNX graph validation için optional onnx gerekir. Mapping unknown ve expectation mismatch native anlamlarını korur. |
| `compare` | İki snapshot, model kimlikleri ve özellik seçimi | comparison.json ve JSON/Markdown/HTML | Kaynak association veya protokol eksikse insufficient/incompatible sonucu beklenir. Mevcut karşılaştırma metrik kümesi kullanılır. |

`--output` için yeni dizin veya yeni dosya adları kullanın. Export no-clobber
sözleşmesini korur. `execution.json`, `evidence.json`, `recommendations.json`,
`report.json`, `report.md`, `report.html` açık export çıktılarıdır.
Her execution satırının status/reasons ve native coverage alanlarını inceleyin.

Canlı veri ve donanım gerektiren API örnekleri ve Manager artifact bağlantıları
[entegrasyon rehberinde](../../docs/guides/native-analysis-executor.md) açıklanır.
