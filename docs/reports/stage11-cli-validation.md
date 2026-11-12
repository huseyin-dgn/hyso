# Aşama 11 — Genel CLI doğrulama raporu

Sonuç: Aşama 11 tamamlandı. `hyso` console-script ve `python -m hyso` aynı
argparse girişini kullanır. Yeni bağımlılık eklenmedi. Python 3.10+ sözleşmesi,
ModelBuilder/TrainingSession/native executor/Manager/Comparison/Advisor/Reporting
API'leri korunur. Aşama 12'ye geçilmedi; commit/push veya staging yapılmadı.

Başlangıç ve son branch: `feature/hyso-models-next`.
Başlangıç ve son HEAD: `d469eabc554f16227f127f12c80c898d4346040b`.
Ortam: Windows, Python 3.10.18, pytest 9.1.1. Linux üzerinde ayrı kurulum
çalıştırılmadı; platformdan bağımsız path işlemleri `pathlib` kullanır.

## Komut ağacı ve çalışan akışlar

```text
hyso [--json] [--debug | --verbose] [--version]
├── models
│   ├── list [--category CATEGORY]
│   ├── show NAME
│   ├── validate ARCHITECTURE.json
│   ├── summary ARCHITECTURE.json
│   └── inspect ARCHITECTURE.json --execute
├── train
│   ├── options
│   ├── validate CONFIG.json
│   └── run CONFIG.json [--factory module:callable] [--execute --allow-training --trust-factory]
├── analysis
│   ├── list
│   ├── show Pxx
│   ├── plan (--features Pxx ... | --profile PROFILE)
│   └── run (--features Pxx ... | --profile PROFILE)
│          [--source NATIVE.json --sha256 EXPECTED | --factory module:callable]
│          [--execute --authorize Pxx ...] [ayrı izinler]
├── experiments
│   ├── list --root STORE
│   ├── history EXPERIMENT_ID --root STORE
│   ├── run RUN_ID --root STORE
│   ├── artifacts --root STORE
│   ├── verify ARTIFACT_ID --root STORE
│   └── open ARTIFACT_ID --root STORE
├── compare BASELINE.json CANDIDATE.json --baseline-model A --candidate-model B --features Pxx ...
├── advisor
│   └── recommend RESULT.json
└── reports
    ├── render RESULT.json
    └── experiment EXPERIMENT_ID --root STORE
```

22 yaprak komut ve 29 parser yardım yolu doğrulandı. Tüm yardım yollarında
model/forward/fit/benchmark/factory/CUDA çağrıları test korumalarıyla engellendi.
Gerçek console launcher ve `python -m hyso`, hem wheel hem sdist kurulumunda
checkout dışından çalıştırıldı.

| Akış | Doğrulanan davranış |
| --- | --- |
| Models | Registry metadata/parametre açıklamaları, güvenli JSON doğrulama, mimari özeti; yalnızca açık inspect CPU modeli oluşturur, forward yapmaz |
| Train | Tam TrainingConfig schema1 doğrulaması, optimizer/scheduler/precision/device seçenekleri, sıfır iş yüklü dry-run, prepared TrainingSession ayar eşleşmesi, tek fit delegasyonu ve retry yasağı |
| Analysis | 35 özellik kataloğu, saf seçim, native metadata preflight, açık yetkili alt küme, retained hash/receipt, P22 ve offline P32/P33 derive yolları |
| Experiments | Mevcut schema1/schema2 history, run/artifact detayları, doğrulanmış result reopen, mismatch durumlarının korunması |
| Compare | Mevcut comparability kapıları, eksik kanıtın korunması, global winner yok, JSON/Markdown; P29 canlı deneyinden ayrı |
| Advisor | recommend() üzerinden mevcut kanıt; tr/en, üç seviye, JSON/Markdown/HTML; numerical_improvement ve estimated_improvement kanıtsız doldurulmaz |
| Reports | TrainingResult, UnifiedAnalysisReport, execution/index + context, Comparison, Recommendation ve orijinal kaynakla doğrulanmış raw AdvisorReport; report_experiment |
| Dosya/CI | Atomic/no-clobber export, açık overwrite, input/native artifact/store çakışmalarının reddi, Unicode-safe JSON envelope, stderr tanılama, kararlı exit kodları |

## Execution ve yetki sınırları

- Varsayılan eğitim ve analiz davranışı dry-run'dır. Factory dry-run'da import
  edilmez; canlı binding hazır olmadığı için ilgili kaynak satırları blocked kalır.
- Eğitim için `--execute --allow-training --trust-factory` ve gerçek uygulama
  factory'si gerekir. Factory `config.prepare(...)` ile hazırlanmış aynı ayarlara
  sahip TrainingSession döndürür; kendi içinde fit başlatmamalıdır.
- Analiz Stage 10'un `execution_mode='native'` yolunu kullanır. `--execute` ile
  ayrı `--authorize` benzersiz seçilmiş alt kümesi gerekir. Dependencies ve
  yetkisiz feature'lar otomatik çalıştırılmaz.
- Capture, training, CUDA, camera, power sampling, long-run ve native file-write
  izinleri ayrıdır. Factory policy'leri CLI izinleriyle değiştirilir; factory
  kendine izin veremez. Native admission ve narrowed authorization tekrar uygulanır.
- `--allow-peak-reset` yalnızca CLI PID'sini native allocator policy'sine taşır.
  Varsayılan reset yoktur; alternatif `--peak-reset-pid` mevcut PID denetimini
  aşmaz. Gerçek GPU allocator reset'i bu doğrulamada yapılmadı.
- Retained kaynakta beklenen SHA256 zorunludur. P22/P32/P33 offline derive yeni
  eğitim, 7200 saniye run veya sensor işi başlatmaz. Eksik süre/sensor ve aborted
  kaynak durumları korunur. Evidence/receipt ve `NOT_EVALUATED` kabul sınırı korunur.
- CLI export'u mevcut input, planlanan native artifact hedefi veya native runtime
  diziniyle çakışamaz. Checkpoint hedefi mevcutsa eğitim başlamadan reddedilir;
  `--overwrite` native checkpoint overwrite izni değildir.
- Manager store oluşturma, otomatik migration, repair veya artifact silme yoktur.
  Açık `experiments verify`, mevcut API üzerinden verification observation ekler;
  expected identity ve run durumunu değiştirmez. History/open/report salt okur.

Exit kodları: 0 başarı/hazır dry-run, 1 beklenmeyen hata, 2 girdi/şema/conflict,
3 blocked/eksik kaynak veya izin, 4 unsupported, 5 başarısız execution,
6 kısmi başarı, 130 kesinti. Komut/factory içindeki `SystemExit(0)` sonuç
döndürmeden başarı kabul edilmez. Varsayılan hata traceback içermez; debug/verbose
teknik ayrıntıyı stderr'e açar. JSON stdout tek envelope'dur; raw export yeniden
okunabilir API formatıdır.

## Test sonuçları

**496 benzersiz ana pytest testi ve 67 alt test geçti.** Yinelenen odaklı
çalıştırmalar toplam sayıya tekrar eklenmedi. Çözümlenmemiş fail/error/skip yoktur.

| Grup | Benzersiz test |
| --- | ---: |
| Yeni CLI ve CLI installed-distribution testleri | 78 |
| İlgili mevcut API/native/Advisor/reporting/Manager/packaging regresyonları | 367 |
| Ek mevcut ModelBuilder/TrainingConfig doğrulayıcıları | 51 |
| Toplam | 496 |

Önce yeni CLI testleri çalıştırıldı. Ardından mevcut regresyonlar ve doğrulayıcılar
çalıştırıldı; son incelemede eklenen sınır testleri ayrıca geçti.

JUnit kanıtları `%LOCALAPPDATA%/hyso-stage11-20261112/` altında:

- `cli-tests.xml`: 74 passed.
- `cli-final-boundaries.xml`: 5 passed; önceki testlerle dört tekrar, bir yeni SystemExit testi.
- `cli-boundary-review.xml`: 7 passed; önceki testlerle dört tekrar, üç yeni sınır testi.
- `regressions.xml`: 367 passed + 67 subtests; JUnit testsuite toplamı alt testler dahil 434.
- `validators.xml`: 54 passed; üç CLI tekrarı ve 51 ek mevcut doğrulayıcı.
- `protection-and-tests.json`: testcase kimliklerinden deduplikasyon ve hash denetimi.

Ana regresyonda 10 gerçek training/benchmark/owned-worker/ağır installed-smoke
vakası bilinçli deselect edildi. CLI izinleri ve lifecycle testleri mock native
producer/factory kullanır; gerçek fit yoktur. Paketleme discovery regresyonunda
toplam dört küçük CPU forward vardır. Yeni gerçek eğitim, CUDA benchmark,
kamera, power sampling veya uzun kabul testi çalıştırılmadı. Mevcut native Trainer
kaynaklı yedi GradScaler deprecation uyarısı vardır; motor değiştirilmedi.

## Wheel/sdist ve kurulum

Her iki arşiv checkout dışında üretildi. Paket envanteri **150 Python modülü**:
mevcut 141 + yeni CLI/`__main__` için dokuz. Wheel ve sdist içindeki 150 modülün
tamamı son çalışma ağacıyla byte düzeyinde eşleştirildi. Python 3.10 AST kontrolü,
metadata, import, public API/typing, dışarıda isolated target kurulumu ve gerçek
console-script/modül girişleri doğrulandı. Aktif Python ortamına paket kurulmadı;
ağdan bağımlılık indirilmedi.

Son arşivler `%LOCALAPPDATA%/hyso-stage11-20261112/dist/` altında korunur:

| Dosya | Bytes | SHA256 |
| --- | ---: | --- |
| hyso-0.2.0-py3-none-any.whl | 663859 | `3669cae533b45fc92014a8485825c77d776b19037f449b242f8e8ddbe9dd46b3` |
| hyso-0.2.0.tar.gz | 579891 | `9fc2f8ab6acee0561945ec817a9cb5e91bc5c693c558cd490259d4a77455888c` |

`final-install-smoke.json`, bu son arşivlerin kurulu target'larında dört ek
başarılı CLI çağrısını kaydeder: her format için gerçek `hyso.exe analysis run
--help` ve `python -m hyso train options --json`. Yeni peak-reset seçeneği ve
precision seçenekleri de bu son kurulumlarda doğrulandı.

## Yeni ve etkilenen dosyalar

Mevcut dört dosya değişti:

```text
README.md
examples/README.md
setup.cfg
tests/test_packaging.py
```

Yeni 19 dosya:

```text
hyso/__main__.py
hyso/cli/__init__.py
hyso/cli/main.py
hyso/cli/common.py
hyso/cli/models.py
hyso/cli/train.py
hyso/cli/analysis.py
hyso/cli/experiments.py
hyso/cli/presentation.py
tests/test_cli.py
tests/test_cli_packaging.py
docs/guides/cli.md
docs/reports/stage11-cli-validation.md
examples/cli/README.md
examples/cli/architecture.json
examples/cli/training.json
examples/cli/training-timing.json
examples/cli/prepare_demo.py
examples/cli/bindings.py
```

Toplam Aşama 11 kapsamı 23 dosyadır. Kullanım/parametre/JSON/hata kodları için
[CLI rehberi](../guides/cli.md); çalışan offline senaryolar ve Python typed-binding
factory örneği için [examples/cli](../../examples/cli/README.md).

## Koruma denetimi ve teknik sınırlar

- Önceki **47/47 forensic/debug/ilgisiz dosya**, Stage 10 raporundaki SHA256
  değerleriyle başlangıçta ve sonda eşleşti. Önceki iki tracked değişiklik ve 45
  untracked dosya aynı kaldı. Başlangıçtaki 627 dosyadan 623'ü byte düzeyinde aynı;
  değişen dört mevcut dosya yukarıdaki açık kapsamdır.
- Native performans/benchmark motorları, Trainer/resume/checkpoint dahil **71
  dosya** hash denetiminde değişmedi. Diğer mevcut public API kaynakları da aynı.
  [10A planı](../roadmap/stage10a-executor-integration-plan.md) korunmuştur.
- [70/70 kabul raporu](p01-p35-final-acceptance-20261112.md) değişmedi. Dışarıdaki
  üç kabul index/manifest dosyası ve bu kayıtlardan bulunan üç benzersiz checkpoint
  stream SHA256 ile yeniden eşleşti; Torch checkpoint deserialization yapılmadı.
  `final-results.json` SHA256:
  `86a014ef432ed96a552bfc35a9c65c3bb78714c117e72708ce7b3c8051795bde`.
- Staged alan boş; branch/HEAD aynı. Commit/push ve yasaklanan Git temizleme/
  toplu-stage işlemleri yapılmadı. Önceki kirli çalışma ağacı korunmuştur.
- Güvenilen factory Python'u sandbox değildir. Live kaynak hazırlığı gerçek
  uygulamanın sorumluluğudur; factory kendi içinde workload başlatmamalıdır.
  Dry-run bu Python'u yüklemediği için gerçek factory kaynaklarını doğrulamaz.
- Factory import öncesi P19–P25 training, P30 camera ve P31 long-run için
  konservatif izin ister. Recorded-video P30 factory'si de camera bayrağı ister.
  Yerel/native Windows ownership, protokol bütçeleri, profiler/evidence boyutları,
  CUDA saved-activation guard'ı ve hardware acceptance sınırları değiştirilmedi.
- Native in-process kaynaklar keyfi Python için hard timeout/RAM sandbox'ı
  sağlamaz. Export yarışı sonradan başarısız olursa tamamlanmış native iş geri
  alınmaz; otomatik retry yoktur. Linux üzerinde ayrıca çalışma kanıtı üretilmedi.
