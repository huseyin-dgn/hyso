# M2-D-B1 — Core Experiment Tracking Implementation

Tarih: 2026-10-19, Europe/Istanbul. **Karar: PASS — test edilen B1 core kapsamı.**
Yeni sonuç: **147 yeni CPU PASS + 1486 hedefli regresyon PASS = 1633 farklı
testcase PASS**, 0 FAIL/ERROR/SKIP/XFAIL/XPASS. Minimum installed smoke ve
21-module wheel/sdist doğrulaması başarılı. Geçmiş C-C sonuçları bu sayıya eklenmedi.

Kanıt kökü **E**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2db1`.
Gerçek JUnit dosyaları `E\targeted.xml` ve `E\final-core-packaging.xml`;
identity/source/archive reconciliation `E\verification.json`.

## 1. Branch / HEAD / working tree

Branch `feature/hyso-models-next`, HEAD
`911beac4e73deadfed06ac7e87975e83a2641844`,
`docs: design HYSo experiment manager`. M2-D-A bu commit içinde kayıtlı ve
DESIGN GO kararı mevcut. Index başlangıçta boştu; başlangıç tek farkı
`tests/test_gradient_accumulation.py` kullanıcı değişikliğiydi.
Commit/push/branch değişimi/checkout/reset/revert/merge/rebase yapılmadı.

Başlangıçtaki129 tracked dosyadan yalnız izinli `tests/test_packaging.py`
değiştirildi; diğer128 dosya başlangıç byte hash'leriyle aynı. Kullanıcı
gradient dosyasının başlangıç/son SHA256 değeri:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.

## 2. M2-D-A design compliance

[M2-D-A](m2-d-a-experiment-manager-design.md) temel sözleşme olarak uygulandı.
Sabit roadmap v1.0,35-feature backlog v1.1, C-C ve B5 final audit'leri;
Trainer, resume, BenchmarkRunner/Result ve packaging kaynakları incelendi.
SQLite authority, açık lifecycle, immutable snapshots/metrics, kısa transactions,
manuel entegrasyon ve ayrı B1/B2 schema sınırı korunuyor.

Tracking core yalnız stdlib kullanır. `torch`, NumPy, Trainer veya Benchmark
import etmez; standalone isolated subprocess testi bunu doğrular. Normal
`hyso.experiments` import'u HYSo root'un mevcut legacy imports'undan geçer;
root package değiştirilmedi ve paket bütünü Torch'tan bağımsız ilan edilmez.

## 3. Implemented B1 scope

Experiment/create/read, Run/create/read/list, config/provenance canonical
snapshots, lifecycle, owner session, request/event idempotency, finite metric
append, atomic batches, bounded queries, storage validation ve safe errors
uygulandı. SQLite fiziksel DB'si `<root_dir>/experiments.sqlite3`.
Otomatik training, benchmark, runtime metadata scraping veya GPU işlemi yok.

## 4. Public API

`from hyso.experiments import ExperimentManager, ExperimentError` çalışır;
namespace `__all__` yalnız bu iki export'u içerir. Dönen ExperimentRun runtime
handle'ı ayrı public constructor gerektirmez. Manager API:
`create_experiment`, `get_experiment`, `create_run`, `get_run`, `list_runs`, `close`.
Run API: `start`, `finish`, `fail`, `interrupt`, `log_metric`, `log_metrics`,
`read_metrics`, `to_dict`; `run.id` read-only public property.

request_id/event_id lowercase32-character UUID hex'tir. Optional ID yoksa
OS entropy tabanlı UUID4 kullanılır. `redact_paths=("/token",)` config'e uygulanır;
iki snapshot için `{"config": [...], "provenance": [...]}` mapping'i de desteklenir.
`fail(code, failure_domain="tracking", workload_outcome="succeeded")` workload
başarısıyla tracking hatasını ayrı kaydeder. Lifecycle return değeri sanitized
reason dict'idir; exact replay aynı committed reason'u döndürür. Tüm record'lar
defensive plain dict/list'tir; caller mutation DB'yi değiştirmez.

## 5. SQLite schema1

`user_version=1`; tam altı core tablo: store_meta, experiments, runs, run_events,
metric_series, metric_events. SQLite'ın kendi sqlite_sequence tablosu da vardır.
Store UUID/canonical_json_version/metric_limit persist edilir. IDs UNIQUE;
Experiment FK, series composite FK ve delete RESTRICT enforced. Kind/status,
run timestamp/ownership consistency ve izinli transition pairs CHECK'lerle
korunur. `(run_id,revision)` transition event uniqueness,
`(run_id,name,axis_index)` coordinate uniqueness, global metric event_id
uniqueness ve scoped create request uniqueness uygulanır.

Config/provenance JSON ve hash'leri Run'da immutable API alanlarıdır. Schema
SQL/index inventory beklenen schema1 ile doğrulanır; unknown/newer schema,
eksik metadata/index ve FK bozukluğu fail-closed'dur. B2 tabloları oluşturulmaz.

## 6. Storage settings

SQLite **3.50.2**, hem ana hem minimum interpreter'da. foreign_keys=ON,
journal_mode=DELETE, synchronous=FULL, busy_timeout=5000ms doğrulandı.
Writes BEGIN IMMEDIATE ile kısa transaction; reads BEGIN ile coherent snapshot
alır. Config validation/hash lock öncesindedir. Başarılı mutation return'ü commit
sonrasıdır; arka plan queue/writer veya implicit flush yoktur.

İlk DB sibling exclusive tempfile'da tamamen oluşturulur, commit/close/fsync
sonrası no-clobber hard link ile yayınlanır; concurrent ilk açılış complete DB
görür. Yalnız bu işlemin kendi tempfile'ı kaldırılır. Root içinde bilinmeyen dosya,
orphan journal, non-regular file, symlink/junction/reparse component, UNC/network
drive ve bilinen OneDrive root'ları reddedilir. Arbitrary dosya silme/overwrite yok.

Caller trusted, local, dedicated ve senkronize edilmeyen root sağlamalıdır.
Diğer sync yazılımlarının/network mount'ların tamamını otomatik teşhis garantisi
yoktur. ACL/auth/encryption/sandbox sağlanmaz. Hostile same-user path races ve
donanım/OS fiziksel elektrik kesintisi dayanıklılığı bu testlerin kapsamı dışıdır.

## 7. Identity / FK / idempotency

Display name unique veya path değildir. training/evaluation/benchmark kind'ları
ve gerçek Experiment ilişkisi test edildi. UUID collision UNIQUE constraint ile
açık integrity_conflict üretir; önceki kayıt korunur. Invalid experiment_id
not_found domain error üretir, yarım Run bırakmaz; FK enforcement ayrıca gerçek
SQL ile test edilir.

Create operations sanitized semantic payload hash'i saklar. Lifecycle operation_id
ve request_hash run_events içinde saklanır. Exact request retry aynı sonucu döndürür;
başka payload/run/operation lifecycle ID conflict üretir. Generated timestamps,
seq ve server metadata semantic hash'e girmez. Create request namespace'i ilgili
create tablosudur; lifecycle operation namespace'i store genelindedir.

## 8. Config / provenance sanitization

Exact native dict/list/string/bool/int/finite float/null; subclasses, bytes,
tuple/set, callable, Tensor/module ve arbitrary object reddedilir. repr/default=str,
pickle/eval yok. Canonical UTF8 JSON sorted keys/compact separators/ensure_ascii=False/
allow_nan=False ile üretilir; SHA256 yalnız sanitized canonical bytes'a uygulanır.
Bu RFC8785 implementation iddiası değildir.

Snapshot başına1MiB, depth16 ve string64KiB sınırı var. Python interpreter'ın
integer-to-JSON digit güvenlik limiti aşılırsa safe snapshot_encoding_limit gelir;
global integer politikası değiştirilmez, float'a sessiz conversion yapılmaz.
1501-digit config integer exact native int olarak round-trip test edildi.
Name256 byte, unit/source/failure code128 byte ve reason256 byte label sınırları
control characters/surrogate rejection ile uygulanır.

Explicit JSON Pointer list/dict traversal ve ~0/~1 escaping test edildi. Sensitive
password/token/secret/api_key/auth keys, explicit redaction olmadan reddedilir.
Redaction `{"value":null,"status":"redacted","reason":"caller_redaction"}`
marker'ıdır. İki farklı raw secret aynı sanitized hash üretir; raw secret test DB
byte'larında ve error metninde yoktur. Full env/argv/hostname/username/local path
kendiliğinden toplanmaz. Serbest metinlerde bütün sırları tespit iddiası yoktur.

## 9. Lifecycle / state machine

created→running/failed/interrupted; running→completed/failed/interrupted;
terminal'den yeni transition yok. Bütün20 source/target kombinasyonu parametrik
test edildi. Status, revision, UTC timestamp ve transition event tek transaction.
Metric insert içeren batch revision'ı bir artırır; exact metric/request replay
revision/event count artırmaz. Event revision'larında metric batches nedeniyle
boşluk bulunabilir; server seq ve revision ordering esastır, wall clock tek kanıt değil.

close/reopen running'i değiştirmez veya completed/interrupted yapmaz.
writer_liveness=unknown advisory görünür; eski session otomatik devralınmaz.
Terminal'e yeni metric yok; daha önce committed exact metric/transition replay'i
yalnız mevcut observation/result'u döndürebilir.

## 10. Writer session

Manager UUID session_id üretir; start DB writer_session_id bağlar. Yeni metric/
transition öncesi persisted state, owner ve revision doğrulanır. Revision/status/
owner guarded UPDATE ve rowcount kontrolü kullanılır; in-memory lock tek otorite değil.
Foreign manager read yapar, running owner'ın yeni metric/status'unu değiştiremez.
Process/thread binding ve non-reentrant access enforce edilir. Thread transfer
gerçek thread testiyle; inherited PID guard Windows'ta explicit PID mismatch
injection ile doğrulandı. Windows'ta POSIX fork testi çalıştırılmış sayılmaz.

## 11. Metric semantics

Name≤128 ASCII, her slash component harfle başlar; alnum/underscore/dot/hyphen
devamı kabul edilir. Empty/dot-only segment/control/non-ASCII reddedilir.
unit/source/axis_kind series boyunca immutable. Value exact native int/float ve
finite olmalı; bool/Tensor/NumPy scalars reddedilir. |integer|>2^53 reddi silent
float64 precision loss'u önler. Unit percent/fraction ayrımı korunur; automatic
accuracy conversion veya None→0 yoktur.

Epoch1-based; step nonnegative signed64 + explicit applied_update/optimizer_attempt/
microbatch/observation. Epoch+step birlikteyse primary step, epoch yalnız context.
Optional observed_at_utc UTC normalize edilir ve semantic payload'a dahildir;
timestamp axis index değildir. Same coordinate same semantic payload eski record'u
döndürür; farklı value/unit/source/axis/context conflict. Explicit farklı event ID
aynı coordinate için yeni identity oluşturamaz. Aynı event ID başka run/payload'da
conflict'tir. Yeni out-of-order coordinate reddedilir, committed eski exact retry
max axis'ten küçük olsa da kabul edilir.

## 12. Atomic batches

log_metrics native list/tuple içinde1–256 record alır. Bütün input validation
transaction öncesi; series/coordinate/budget kontrolleri tek write transaction'dadır.
Invalid/conflicting/out-of-order/trigger-failing son record bütün batch'i rollback
eder; önceki committed metric'ler ve revision korunur. Exact duplicates ikinci kez
eklenmez. Default1,000,000/run budget store metadata'da persist edilir;
explicit max_metrics_per_run policy reopen'da aynı olmalı. Budget aşımı atomic
error'dur, silent drop yok. Boundary testleri explicit2-event store ile yapılır;
milyon event yazılarak stress test edildiği iddia edilmez.

## 13. Query / pagination

list_runs default100, üst sınır1000; read_metrics default/üst sınır1000.
seq order ve exclusive after_seq cursor; optional Experiment/kind/status/name
filters SQL parameters ile uygulanır. Default100/1000 sınırları gerçek101-run ve
1001-metric kayıtlarıyla test edildi. Counts committed SQLite observations'dan
türetilir; defensive snapshot'tır. Query model/benchmark çalıştırmaz, best/winner/
Pareto/automatic score üretmez. to_dict metrics listesi yerine metric_count verir.

## 14. Error / failure boundaries

ExperimentError code, operation, entity_id/request_id/event_id, retryable ve
commit_state taşır. Validation/conflict/busy/storage/closed/context ayrımı var.
Storage exception metni veya caller payload repr'i outward mesaja taşınmaz.
Commit sınırı exception'ında konservatif commit_state=unknown kullanılır;
commit öncesi ve gerçek commit sonrası ACK loss testlerinin ikisinde de explicit
ID ile retry/read-back tek committed kayıt verir.

Gerçek SQLite TEMP triggers ile create/transition/batch rollback; corrupt/empty
DB, newer schema, missing metadata/index, invalid FK, UUID collision ve closed
session test edildi. Ayrı süreç commit öncesi/sonrası sınırda sonlandırıldı;
reopen row count ve integrity_check doğrulandı. Fiziksel disk-full veya gerçek
power cut uygulanmadı; trigger/ACK/process-kill sonuçları bunların kanıtı değildir.

## 15. Concurrent writer tests

Sabit sleep olmadan stdin/stdout barriers kullanıldı. Dört bağımsız process aynı
root'u eşzamanlı ilk açıp dört unique Experiment/Run oluşturdu. Üç process aynı
create_run request'ini paylaştığında tek Run oluştu. Foreign writer writes reddedildi.
Ayrı process BEGIN IMMEDIATE lock tutarken5000ms busy timeout gerçek storage_busy
üretti; lock release sonrası yeni create başarılı, başarısız çağrı row bırakmadı.
Her process bağımsız SQLite connection/session kullanır; background writer yok.

## 16. New CPU test counts

| Final yeni dosya | Farklı PASS |
|---|---:|
| tests/test_experiment_core.py | 60 |
| tests/test_experiment_metrics.py | 59 |
| tests/test_experiment_storage.py | 28 |
| Toplam yeni | **147** |

Final source koşusu final-core-packaging.xml:151 PASS/60.13s;147 yeni +4 mevcut
packaging testcase. İlk geliştirme koşularındaki hatalar düzeltildi: Windows
fsync read-only fd yerine writable fd ister. Ara135/146 PASS koşuları final
toplamına eklenmedi. Yeni skip/xfail veya gevşetilmiş assertion yok.

## 17. Targeted Trainer / M1 / Benchmark regression

| Hedef grup | Farklı PASS |
|---|---:|
| Generic Trainer: sekiz test dosyası | 706 |
| TrainingCheckpoint | 34 |
| OneCycle checkpoint compatibility | 20 |
| M1 attention119 + blocks135 + ModularCNN102 | 356 |
| Benchmark CPU/protocol/export/comparison/result/state | 306 |
| Legacy model-state | 18 |
| Public API41 + published compatibility1 | 42 |
| Packaging | 4 |
| Regresyon toplamı | **1486** |

Trainer detayları: **restore179, save135**, classification116, lifecycle53,
replay37, restore_transaction13, updates141, validation32. Regresyon23 dosyada
1486 PASS/128.16s,663 mevcut dependency warnings; command wall133.39s.
Table'daki Trainer toplamı bu sekiz dosyanın toplamıdır.
JUnit identity'lerinin1486'sı geçmiş C-C'de de PASS olan aynı identity'lerdir;
şimdiki sonuçlar yeni execution'lardan gelir. final-core-packaging ile dört
packaging identity overlap çıkarıldı:1486+151−4=**1633**. Aynı1486 eski testcase
PASS durumunu korur; packaging additive/stronger checks içerir.

Benchmark306 içindeki33 CUDA protocol testi **CPU wiring/failure tests**tir;
native CUDA olarak sayılmadı. Full CPU suite ve native CUDA scripts bu task'ta
çalıştırılmadı; son birleşik gate M2-D-C'ye aittir.

## 18. Torch2.2.2 installed smoke

Mevcut temiz interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Python3.10.18, Torch2.2.2+cpu, torchvision0.17.2+cpu, SQLite3.50.2.
Son wheel `--no-index --no-deps --no-build-isolation --no-compile` ile yalnız
`E\installed-minimum` isolated target'a kuruldu; yeni dependency kurulmadı.
Repo dışındaki E cwd'den -I -B child interpreter'da namespace provenance ve
create→start→append→query→finish→close→reopen persistence smoke PASS.
Manager kaynağı `E\installed-minimum\hyso\experiments\manager.py`.

Mevcut installed smoke'un legacy/model/Trainer/checkpoint/replay ve CPU Benchmark
FP32/FP64/export/comparison kontrolleri de geçti; cuda_initialized=False.
Bunlar pytest distinct sayısına veya native CUDA toplamına eklenmedi.
`E\installed-minimum.json`, minimum stdout/stderr/install logları gerçek kanıttır.
Clean ortam35 distribution/21432 recorded non-pyc file hash'inin önce/sonra
eşliği doğrulandı: `E\clean-before.json` ve `E\clean-after.json` aynı Python,
prefix, versions ve file_hashes içerir. Existing RECORD/metadata/binary/source
dosyaları da bu envanterdedir; interpreter-global kurulum yapılmadı.

## 19. Wheel / sdist integrity

Son build klasörü `E\final-package-temp\packaging0\dist`:

| Artifact | Byte | SHA256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl | 82647 | 4c1b02a765448b0c587540008dffa5a6c653d68bbc08e254890f0e5df317628a |
| hyso-0.2.0.tar.gz | 79756 | 04e0133276d2b04363e7deb7ddbf43cf6d2f767b327dc88eea29bfbdde36b33c |

Gerçek filesystem inventory **21 Python production modülü**, iki archive'da aynı
21 modül ve source-byte eşliği. Wheel ve sdist ayrı isolated target'lara installed
edilip legacy, Trainer, Benchmark ve yeni SQLite core smoke çalıştırıldı.
Test/docs/database/artifact/weights/cache sentinels paketlere girmedi.
Metadata HYSo0.2.0/Python>=3.10/Torch>=2.2.2/torchvision>=0.17.2, eski8 dependency
aynı; sqlite3 yeni dependency değildir. setup.cfg/pyproject/MANIFEST değiştirilmedi.

## 20. Legacy freeze

Eski19 production modülünün tamamı byte-identical. Trainer/resume/checkpoint,
models/callbacks/benchmark, root exports, EBO/ELO/CEMO/EMA/CNNS API ve state_dict,
checkpoint FORMAT_VERSION1 ve BenchmarkResult schema1 aynı. Roadmap v1.0 ve
performance backlog v1.1 aynı byte'larda;35 satır1 implemented/4 partial/30 planned.

| Kaynak | SHA256 |
|---|---|
| hyso/experiments/__init__.py | 871dadf921c9d01969ce5adcbdab2c40b7c49f7ccea34cfb05956c2b1c2cef75 |
| hyso/experiments/manager.py | 75070e84db6f2ed0fe33fd13b022dd569e7d96f25e4c45e88b43e21c4038a934 |
| hyso/training/trainer.py | c6283922efe60cee7a832b6f822d030f3e42c4d1061eba5e0d427914e120288d |
| hyso/training/resume.py | 66943d34e2a3830dc96f8dde335da90b5bc9381ef5dbc5506b41375331d08df1 |
| hyso/checkpoint.py | 00f6010bbb22d1de3a42593229ee975c32df0b9665d8f3a1b8d1b4d5adceba4a |
| hyso/benchmark/runner.py | 6ac2c5bc45d73eceb7d610a4190b3707f2717494511f80068b63d57598f76b73 |
| hyso/benchmark/result.py | c15a49d72e05906c28d4e7b424573e2ac4b7a7918713c2e4abddbb452f3dbdc0 |

Tam21-module hash inventory `E\verification.json` içinde bulunur.

## 21. Deferred B2 / B3

Artifact reference/copy/ingest/verify/link/reconcile, storage schema2 migration,
retry lineage/resume artifact, recovery/fencing/takeover/heartbeat API yok.
Non-null retry_of_run_id/resume_artifact_id explicit unsupported_b2 üretir.
B1 store schema1'de kalır; normal open migration veya automatic recovery yapmaz.
Trainer/Benchmark automation, callbacks adaptation, GUI/cloud/sweep/profiler,
best/winner/Pareto ve35-feature backlog implementasyonu bu scope'a alınmadı.

## 22. Changed files

- Yeni: hyso/experiments/__init__.py, hyso/experiments/manager.py.
- Yeni: tests/test_experiment_core.py, test_experiment_metrics.py, test_experiment_storage.py.
- Additive update: tests/test_packaging.py; eski assertions korunarak21-module
  inventory, exclusions, installed ExperimentManager smoke eklendi.
- Yeni: docs/audit/m2-d-b1-experiment-core.md (bu rapor).

Kullanıcının önceden mevcut tests/test_gradient_accumulation.py farkı aynen
korundu; bu görevin değişikliği değildir. Runtime DB/build/JUnit/temporary helper
dosyaları repository dışında E veya pytest local temp root'larındadır.

## 23. PASS / PARTIAL / FAIL decision

**PASS.** Schema1, canonical/redacted snapshots, lifecycle/ownership, idempotency,
atomic metrics, bounded queries, gerçek SQLite failure/concurrency tests,
1633 distinct CPU PASS, minimum installed smoke ve21-module wheel/sdist gates
tamamlandı. Clean interpreter'ın35 distribution ve21432 recorded file hash'i
aynı kaldı; eski production ve kullanıcı değişikliği korundu. Bu karar B1 core
kapsamınadır; full final Experiment Manager/M2-D-C veya bütün roadmap için GO değildir.

## 24. M2-D-B2 prerequisites

B1 schema1 baseline'ı hazır. B2 başlamadan bu working tree kullanıcı tarafından
review/commit edilebilir; bu görev commit oluşturmadı. Sonraki scope explicit
offline schema1→2 migration/backup/failure tests, artifact two-resource protocol,
reconciliation, recovery expected_revision fencing ve run_attempts lineage'dır.
Mevcut core observations/config/hash/IDs ve schema1 record envelope korunmalı;
Trainer/checkpoint/Benchmark freeze sınırları sürmelidir. B3 manuel integration,
D-C birleşik CPU/native/minimum/packaging final gate'i ayrıca yapılacaktır.
