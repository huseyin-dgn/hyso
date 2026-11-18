# Çalışan CLI örnekleri

Repo kökünde `python -m pip install -e .` sonrasında aşağıdaki komutlar
çalışır. Varsayılan senaryolar model benchmark'ı, eğitim, CUDA, kamera veya
uzun kabul testi başlatmaz. `prepare_demo.py` yalnızca açıkça etiketlenmiş
sentetik retained fixture ve mevcut API'lerle offline derive dosyaları üretir.

```console
python examples/cli/prepare_demo.py --output CLI-DEMO
hyso models list
hyso models show conv2d --json
hyso models validate CLI-DEMO/architecture.json
hyso models summary CLI-DEMO/architecture.json
hyso train validate CLI-DEMO/training.json
hyso train options --json
hyso analysis list
hyso analysis show P31 --json
hyso analysis plan --features P01 P02 P11 --json
hyso analysis run --features P01 P02 --factory my_app.resources:bindings --json
```

Son komut kaynak factory'sini yüklemez, `blocked`, exit 3 döndürür; bu beklenen
güvenli dry-run sonucudur. `my_app.resources` gerçek uygulamaya ait bir modüldür.

Hazırlık çıktısındaki `p19_sha256` değerini HASH yerine koyun:

```console
hyso analysis run --features P22 --source CLI-DEMO/p19.json --sha256 HASH --json
hyso analysis run --features P22 --source CLI-DEMO/p19.json --sha256 HASH --execute --authorize P22 --output CLI-DEMO/offline-execution.json
hyso compare CLI-DEMO/A-execution.json CLI-DEMO/B-execution.json --baseline-model A --candidate-model B --features P01 P11 --output CLI-DEMO/comparison.json
hyso compare CLI-DEMO/A-execution.json CLI-DEMO/B-execution.json --baseline-model A --candidate-model B --features P01 P11 --format markdown --output CLI-DEMO/comparison.md
hyso advisor recommend CLI-DEMO/A-analysis.json --language tr --level beginner --output CLI-DEMO/advice.json
hyso reports render CLI-DEMO/advice.json --format html --language tr --level expert --output CLI-DEMO/advice.html
hyso reports render examples/reporting/fixtures/training-result-v1.json --format html --output CLI-DEMO/training.html
```

Comparison fixture'larında P01/P11 bulunmadığı için yeterli karşılaştırma kanıtı
olmadığı korunur; kazanan üretilmez. Exportlar mevcut dosyanın üzerine yazmaz.
Farklı sonuçlarla tekrar için yeni dizin kullanın veya açık `--overwrite` verin.

Önceden oluşturulmuş gerçek Manager store'u için:

```console
hyso experiments list --root STORE --json
hyso experiments history EXPERIMENT_ID --root STORE --json
hyso experiments run RUN_ID --root STORE --json
hyso experiments open ARTIFACT_ID --root STORE --json
hyso reports experiment EXPERIMENT_ID --root STORE --format html --output CLI-DEMO/experiment.html
```

STORE mevcut dedicated local dizindir; CLI store oluşturmaz veya migration
yapmaz. Bu komutlarda ID'leri `list`/`history` çıktılarından alın. Testler geçici
schema1/schema2 store'larla bu akışları ayrıca doğrular.

`bindings.py` güvenilen factory örneği mevcut dosyayı hash ile okur ve typed
NativeBinding döndürür. Uygulamanızın kurulu modülüne aynı fonksiyonu taşıyın;
factory dry-run sırasında import edilmez. Canlı factory entegrasyonu, ayrı
izinler, JSON şemaları ve exit code sözleşmesi için
[CLI rehberi](../../docs/guides/cli.md).
