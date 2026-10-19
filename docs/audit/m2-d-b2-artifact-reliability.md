# M2-D-B2 — Artifact Management, SQLite Migration & Reliability

Tarih: 2026-10-19 (Europe/Istanbul). Bu raporun sayımları aşağıda belirtilen
yeni çalıştırmalardan gelir; önceki B1/Trainer/Benchmark sonuçları yeni kanıt
olarak kullanılmaz. Kanıt kökü: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2db2-evidence`
(aşağıda E).

## 1. Branch / HEAD / working tree

Branch `feature/hyso-models-next`, başlangıç/son HEAD
`9db424dcab6eb87ebe872b81ce6864a7a14bb41f` —
`feat: add SQLite experiment tracking core`. B1 commit edilmişti. Başlangıçta
yalnız `tests/test_gradient_accumulation.py` kullanıcı farkı vardı. Bu dosyaya
yazılmadı, stage edilmedi veya geri alınmadı. Başlangıç hash'i:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Commit/push/branch değiştirme/checkout/reset/revert/rebase/merge yapılmadı.

## 2. B1 baseline ve design compliance

M2-D-A mimari sözleşmesi, B1 raporu/kaynakları/testleri, M2-C-C ve M2-B5
gate'leri, sabit roadmap ve performance backlog incelendi. Tracking yalnız
stdlib kullanır; Trainer/Benchmark otomasyonu eklenmedi. SQLite kayıt zarfı
`hyso.experiment` schema1 olarak kalır; storage version ayrı schema2'dir.
Yeni filesystem sorumluluğu mevcut private helper'larla yönetildi; yeni modül,
ORM veya generic storage framework eklenmedi.

## 3. Schema2 ve migration

Altı core tablo/index tanımı korunur. `artifacts`, `run_artifacts`,
`artifact_checks`, `run_attempts` eklenir; foreign keys, role/state/ownership/hash
CHECK'leri, request uniqueness, link uniqueness, UNIQUE retry parent ve
UNIQUE(lineage_id,attempt_no) uygulanır.

**Yeni store'lar bilinçli olarak schema1 ile başlar.** Normal open hiçbir store'u
migrate etmez. `ExperimentManager.migrate_store(root)` explicit schema2 geçişidir.
Schema2 store için `already_migrated`, backup üretmeyen no-op'tur. Unknown schema
ve version2 etiketi altında eksik/yanlış layout fail-closed kalır.

Kullanım sırası:

```python
from hyso.experiments import ExperimentManager

manager = ExperimentManager(root)
manager.close()
ExperimentManager.migrate_store(root)
manager = ExperimentManager(root)
```

Offline sınır: caller writer süreçlerini koordine edip durdurmalıdır. Persisted
running kayıt varsa migration reddedilir; terk edilmiş kayıt önce explicit
recovery ister. Schema1'de de recovery bu amaçla kullanılabilir. SQLite reserved
lock, yeni write'ları dışlar; eski schema1 session sonraki erişimde version guard
ile reddedilir. Bu mekanizma process liveness tespiti veya eski bir binary'yi
zorla durdurma garantisi değildir.

## 4. Backup ve data preservation

Backup SQLite `Connection.backup()` ile `backups/<uuid>.sqlite3` altında exclusive
oluşturulur. Integrity check, version ve altı core tablonun tüm ordered row
değerleri backup üzerinde kontrol edilir. Backup kapatılıp fsync edilir.
Backup ile DDL arasında yeni writer girmesini önlemek için BEGIN IMMEDIATE,
backup alınmadan önce tutulur; backup ayrı read connection üzerinden alınır.

DDL, initial attempt backfill ve `user_version=2` tek transaction'dadır.
Config/provenance SHA'ları ve foreign keys doğrulanır; altı core tablonun
snapshot'ı değişmemelidir. Reopen schema/layout/FK/quick_check ve snapshot
doğrulaması yapılır. Eski IDs, canonical JSON bytes/hash, metric events/axes/unit,
status/revision/timestamp korunur. DDL failure/process termination rollback'i
eski store'u korur; backup kalır. Commit ACK belirsizliğinde başarı dönülmez;
caller version/read-back ile durumu belirler. Otomatik restore veya veri silerek
yeniden oluşturma yoktur. Backup yalnız SQLite'ı kapsar, external file backup'ı
değildir.

## 5. Artifact data model

Manager UUID üretir. Public dict hem `id` hem `artifact_id`, ownership/kind,
producer, optional locator, managed relative path, ingest state, immutable
expected size/hash, timestamps, request ID ve defensive metadata içerir.
Session/request hash internal kalır. `role="output"` producer bilgisini caller'ın
üretim beyanı olarak current run'a bağlar; input/report için producer uydurulmaz.
Sonraki link consumer'ı producer yapmaz.

## 6. Public Artifact API

`run.add_artifact(path, kind="checkpoint", ownership="reference", role="output")`,
`ownership="copy"`, `manager.link_artifact(run_id, artifact_id, role="input")`,
`verify_artifact(id, path=None)` ve `reconcile_storage(limit=1000)` uygulanır.
`get_artifact` ve paginated `list_artifacts` defensive plain dict döndürür.
Başarılı return committed metadata demektir; idempotent replay pending/failed
metadata döndürebilir ve bu state açıkça görünür. Bu replay ingest'i yeniden
başlatmaz. Commit belirsizliği `ExperimentError(commit_state="unknown")` üretir.

## 7. Reference ownership

Kaynak yalnız okunur; kopyalama/silme/üzerine yazma yoktur. Regular file,
streaming hash/size, open descriptor fstat ve path stat/lstat karşılaştırmaları
kullanılır. Dosya kimliği/size/mtime/ctime değişimi reddedilir; hostile mutation
karşısında atomic snapshot iddiası yoktur.

Locator varsayılan olarak saklanmaz. `verify_artifact(id, path=caller_path)`
explicit byte kaynağı ister; locator yoksa `unavailable` döner, filesystem taraması
yapılmaz. `retain_locator=True` absolute locator'ın persist edilmesini explicit
seçer. Caller-supplied verification path locator'ı veya expected identity'yi
değiştirmez. Copy kaydında original locator tutulmaz.

## 8. Managed copy protocol

Pending reservation ve run link'i commit edilir. Kaynak 1 MiB chunk'larla
`.staging/<uuid>.part` exclusive dosyasına kopyalanır; SHA ve byte count aynı
stream'de hesaplanır. Varsayılan ve bu sürümün üst sınırı 1 GiB'dır; caller daha
küçük limit seçebilir. Flush/fsync/close sonrasında staged identity commit edilir.
`os.link` final `artifacts/<uuid>/payload` path'ine atomic no-clobber yayınlar;
overwrite/replace fallback yoktur. Final dosya yeniden hash'lenir, ardından ready
commit edilir. Uzun ingest copy/hash sırasında DB transaction açık değildir.

Staging hard link'i tutulur; ilk sürümde otomatik cleanup/prune veya owned-file
hard delete API'si yoktur. Bunlar korunan dedicated root altında caller'ın
değiştirmemesi gereken dosyalardır; API dışı filesystem değişikliği engellenmez.

## 9. Hash verification

Explicit verification `verified`, `mismatch`, `missing`, `unreadable`,
`unavailable` observation üretir. Her sonuç append edilir; expected SHA/size
değiştirilmez ve eski check silinmez. SHA yalnız byte identity'sidir; güvenilir
model/optimizer/epoch/CUDA RNG veya loadability kanıtı değildir. Checkpoint
deserialize, pickle, `torch.load` veya reconstruction çağrısı yoktur.

## 10. DB/filesystem atomicity sınırları

SQLite ve filesystem tek atomic transaction değildir. Reservation/staging/final
publication/ready commit arasında process ölümü kalıntı bırakabilir. Her state
persist edilir; unknown commit başarı sayılmaz. Başarısız I/O, mümkün olduğunda
failed state ve ingest observation bırakır; DB hatası bu gözlemi de engelleyebilir.
Kaynak dosya değiştirilmez. Fiziksel power-cut/disk firmware dayanıklılığı
doğrulanmadı; directory durability veya hostile same-user sandbox garantisi yoktur.

## 11. Pending/staged/ready

Pending usable değildir. Staged kaydın expected identity'si vardır; ready yalnız
doğrulanmış publication sonrasında commit edilir. Failed kayıt otomatik yeniden
başlatılmaz veya başarılı sayılmaz. Bilinen pending/staged/failed output bulunan
run `finish()` ile completed olamaz. Terminal run'a yeni artifact/link/metric
eklenmez; önceden committed exact replay mevcut kaydı döndürebilir.

## 12. Reconciliation

Explicit offline işlem persisted running kayıtlarını reddeder. Reserved write
transaction, cooperating writer'ların bakım sırasında başlamasını dışlar; ingest
state/session CAS ile staged→ready yapılır. Bu işlem bounded hashing sırasında
SQLite writer lock tutar; ingest gibi kısa transaction vaadi yoktur. Bu tercih
maintenance yarışını güvenli tutar ve caller'a görünür bir maliyettir.

En çok 1000 artifact ve `3 * limit` immediate filesystem entry incelenir;
sınır aşılırsa `reconcile_limit`, sessiz truncation yoktur. Recursive root taraması
yoktur. Staged/final bytes yalnız expected hash/size eşitse ready yapılır. Pending
unknown bytes ve failed ingest quarantine raporu olarak kalır; ready missing,
hash mismatch, orphan staging/directory ve unexpected extra file raporlanır.
Mismatch ready yapılmaz. Hiçbir dosya otomatik silinmez veya orphan adopt edilmez;
running run completed yapılmaz. 1000'den büyük store için paginated maintenance
bu sürümün kapsamı dışındadır.

## 13. Writer recovery/fencing

`recover_interrupted(run_id, expected_revision, reason)` persisted running state
ve revision CAS ister. Status/reason/event/revision ile writer authority revoke
aynı transaction'dadır. Eski handle yeni metric/lifecycle/ingest yazamaz. Reopen
running'i kendiliğinden değiştirmez. Bu işlem training process'ini öldürmez veya
GPU işini iptal etmez; eski writer'ı durdurmak caller'ın sorumluluğudur.

## 14. Retry lineage

Failed/interrupted parent aynı experiment/kind içinde yeni UUID child üretir.
Parent immutable kalır; child config/provenance ayrı snapshot'tır. Initial
lineage=self, attempt=1; retry parent+1 olur. Parent UNIQUE ve lineage/attempt
UNIQUE concurrent duplicate child'ı önler. Self-parent CHECK ve yalnız mevcut
terminal parent'a yeni UUID ilişkilendirme cycle oluşmasını engeller. B1 request
idempotency korunur; changed retry payload request conflict üretir.

## 15. Resume artifact association

`create_run(..., retry_of_run_id=..., resume_artifact_id=...)` mevcut ready
checkpoint artifact'ini byte identity doğrulamasıyla input olarak bağlar.
Başka store/missing/wrong kind/unavailable/mismatched artifact reddedilir.
Producer değiştirilmez. Exact committed create replay dosya daha sonra kaybolsa
da read-only döner. Yeni run created kalır; Trainer/fit/load/strict resume
otomatik çağrılmaz. Checkpoint FORMAT_VERSION=1 değişmez.

## 16. Security/path handling

Managed path yalnız validated manager UUID'sinden türetilir. Caller destination
seçemez. Path traversal/invalid ID, symlink mode, native Windows junction/reparse,
root escape, non-regular/missing source, source replacement, no-clobber conflict,
permission/partial-write/fsync/link failure, checksum/metadata validation,
duplicate request ve foreign owner negatifleri test edilir. Locator privacy
explicit'tir. Domain errors underlying path/secret/storage exception metnini
taşımaz. SQLite malformed/version/layout/FK kontrolü korunur. Arbitrary checkpoint
bytes yalnız hash'lenir. Tam hostile same-user filesystem sandbox iddiası yoktur.

## 17. Windows filesystem behavior

Native Windows/NTFS copy/publication ve junction reddi test edilir. Symbolic-link
mode rejection, symlink oluşturma privilege'ı gerektirmeyen lstat injection ile
ayrıca test edilir; bu native symlink oluşturma testi olarak sayılmaz. Windows
backup fsync'i write-enabled handle kullanır. Sharing/antivirus/indexer kaynaklı
publication PermissionError domain hatasına dönüşür; replace fallback veya
sonsuz retry yoktur. Bu hatalar injection ile denenir; gerçek antivirus kaynaklı
arızanın üretildiği iddia edilmez.

## 18. New CPU tests

Yeni dört dosya **83 PASS**: artifacts36, migration11, recovery17, retry19.
Son hedefli koşunun JUnit'i `E/targeted.xml`, tekil identity envanteri
`E/verification.json` içindedir. Migration Unicode fixture'i sonrasında ayrıca
11/11 tekrar geçti (`E/migration-unicode.xml`); tekrarlar toplam sayıya eklenmedi.

Yeni testler dört dosyadadır: migration, artifacts, recovery, retry. Gerçek SQLite
ve filesystem assertions kullanılır. Child process'ler DB/ingest phase barrier'ında
bekletilip gerçekten sonlandırılır; stdin/stdout barriers ve process termination
ack kullanılır, fixed sleep race gizlemez. Reservation öncesi, pending, pending
staging bytes, staged, final publication ve ready commit öncesi kill kapsanır.
Migration partial DDL kill, dört DDL rollback noktası, commit öncesi/sonrası ACK
loss, gerçek SQLite writer lock ve concurrent retry parent collision da kapsanır.

## 19. Existing B1 regression

Son koşuda core60 + metrics59 + storage28 = **147/147 PASS**. Eski final B1
JUnit identity kümesiyle eşlik kontrol edildi; identity silinmedi veya yeniden
adlandırılmadı.

Üç B1 dosyasının 147 testcase identity'si korunur. Eski
`test_b2_is_not_implemented` identity'sindeki artık geçersiz “method yok” assertion'ı,
schema1'in B2 artifact/link/reconcile işlemlerini reddetmesi ve implicit migration
olmaması kontrolleriyle güncellendi. Diğer B1 assertions korunur. Gerçek schema1
fixture yeni B1 init API'siyle yaratılır; version etiketi değiştirilerek migration
kanıtı uydurulmaz.

## 20. Trainer/M1/Benchmark regression

Yeni hedefli koşu **30 dosyada 1716 PASS, 0 FAIL/ERROR/SKIP/XFAIL/XPASS**;
pytest süresi169.43s, command wall174.66s. 663 mevcut dependency/deprecation
warning raporlandı. Testler değiştirilerek veya yeni skip/xfail eklenerek
gizlenmedi. Parallel bağımsız ortam snapshot'ı test assertion'larını değiştirmedi.

| Hedef grup | Farklı PASS |
|---|---:|
| Yeni B2 | 83 |
| B1 core/metrics/storage | 147 |
| Generic Trainer, sekiz dosya | 706 |
| P0 checkpoint + OneCycle | 54 |
| M1 attention/blocks/ModularCNN | 356 |
| Benchmark CPU/protocol/export/comparison/result/state | 306 |
| Legacy model-state | 18 |
| Public API + published compatibility | 42 |
| Packaging | 4 |
| **Tekil toplam** | **1716** |

Hedefli kapsam: sekiz Generic Trainer dosyası, P0 checkpoint/OneCycle, M1
attention/blocks/ModularCNN, altı Benchmark CPU/protocol/result/export/comparison/
state dosyası, model-state/public API/published compatibility ve packaging.
CPU CUDA-protocol testleri native CUDA kanıtı değildir. Full CPU suite ve native
CUDA final gate bu görevde çalıştırılmaz; M2-D-C kapsamıdır.

## 21. Torch2.2.2 installed smoke

**Installed smoke PASS**: Python3.10.18, Torch2.2.2+cpu,
torchvision0.17.2+cpu, SQLite3.50.2. Kaynak:
`E/installed-minimum/hyso/experiments/manager.py`. `E/installed-minimum.json`,
`minimum-install.log`, `minimum.stdout.log`, `minimum.stderr.log` kanıttır.
Minimum smoke pytest sayımına veya native CUDA sayımına eklenmez.
`E/clean-before.json` ile `E/clean-after.json` bütün içerikleriyle eşittir:
**35 distribution ve 21432 non-pyc dosyanın hash'i aynı**. Python/prefix/version
bilgileri de eşittir; `E/clean-comparison.json` sonucu kaydeder. Global minimum
ortama paket veya dependency kurulmadı.

Mevcut minimum interpreter:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-p06c-oi2koqex\clean\Scripts\python.exe`.
Son wheel yalnız E altındaki isolated target'a `--no-index --no-deps
--no-build-isolation --no-compile` ile kurulur. Repo dışı cwd, `-I -B` ve bütün
HYSo import'larının installed target provenance kontrolü kullanılır. Smoke
migration/backup/old records, reference/copy/SHA/link, recovery, retry/resume,
reopen/metrics/lifecycle ile mevcut legacy/Trainer/checkpoint/Benchmark smoke'unu
birlikte çalıştırır. Clean distribution/source/metadata/binary file hash'leri önce
ve sonra karşılaştırılır. Bu CPU sonucudur, CUDA sonucu değildir.

## 22. Wheel/sdist integrity

Son build: `E/targeted-temp/packaging0/dist`. Wheel/sdist build, byte eşliği ve
iki installed smoke dahil dört packaging testcase **PASS**.

| Artifact | Byte | SHA-256 |
|---|---:|---|
| hyso-0.2.0-py3-none-any.whl | 88666 | `775168c91f2db73c28dbd38b285c4265075b856fbd312baf2abc93aec05cd621` |
| hyso-0.2.0.tar.gz | 85648 | `b63abafb3521aa4b67ac674986e4c604d03a7a10cfcdc7ecd734d4fc8fdf87bd` |

Gerçek filesystem production inventory **21 Python modülü**dür; `artifacts.py`
eklenmedi. Packaging testleri wheel ve sdist'in tüm source byte'larını checkout'la
karşılaştırır. İki distribution ayrı isolated target'larda smoke edilir. Yeni B2
assertions mevcut installed smoke'a additive eklenmiştir. Test/docs/DB/artifacts/
cache/weights sentinels paketlere girmez. HYSo0.2.0, Python>=3.10,
Torch>=2.2.2, torchvision>=0.17.2 ve mevcut dependency metadata değişmez.

## 23. Legacy freeze

Trainer/resume/checkpoint, models/callbacks/benchmark, root public exports,
checkpoint1/BenchmarkResult1, setup metadata, roadmap ve performance backlog
korunur. Yeni dependency kurulmaz, veri/weights indirilmez. Kullanıcı gradient
dosyası başlangıç SHA'sında kalır. Index boş kalır. Eski freeze hash'leri yalnız
karşılaştırma baseline'ıdır; yeni runtime test sonucu yerine kullanılmaz.

## 24. Changed files

- `hyso/experiments/manager.py`
- `tests/test_experiment_core.py` — yalnız milestone feature-boundary assertion'ı
- `tests/test_experiment_migration.py`
- `tests/test_experiment_artifacts.py`
- `tests/test_experiment_recovery.py`
- `tests/test_experiment_retry.py`
- `tests/test_packaging.py`
- `docs/audit/m2-d-b2-artifact-reliability.md`

Önceden değişmiş `tests/test_gradient_accumulation.py` bu görevin değişikliği değildir.

## 25. Known limitations

Trusted dedicated non-synced local root gerekir. Filesystem ve DB ortak atomic
commit değildir. Backup external dosyaları kapsamaz. Recovery process kontrolü
değildir. Hostile filesystem/physical power-cut/Linux native davranışı test
edilmedi. Reference locator privacy nedeniyle locator/path yoksa doğrulama
unavailable'dır. Staging link'leri tutulur; cleanup/delete/prune yoktur. Bakım
offline'dır ve hash sırasında bounded writer lock tutar; maliyetsiz değildir.
1000 artifact üstünde reconciliation pagination ayrıca tasarlanmalıdır. Hash
model güvenliği/strict resume/reproducibility garantisi vermez. Yeni store'lar
B2 özelliklerinden önce explicit migration gerektirir.

## 26. PASS/PARTIAL/FAIL

**M2-D-B2: PASS; M2-D-B3'e geçilebilir.** Final test/package sonucu PASS:
schema migration/backup/data preservation,
reference/copy/SHA, linking, pending/staged/ready/failure handling, explicit
reconciliation, process-kill/commit/permission/fsync/link fault injection,
recovery/fencing, retry/resume association,83 yeni CPU testcase,147 B1 identity,
1486 regression ve21-module wheel/sdist doğrulandı. Ara başarısız koşular final
PASS sayısına eklenmez.

İlk B1 koşusunun146/147 sonucu, eski milestone “B2 method yok” assertion'ından
geldi; schema1 sınırını koruyan güncellemeden sonra147/147 geçti. İlk migration
koşusu Windows read-only handle fsync hatasını ortaya çıkardı; backup fsync handle'ı
düzeltildi. Bir process-kill testinde stdin kapanmasının pause'ı serbest bırakma
yarışı görüldü; kill acknowledgement ve EOF'de ilerlemeyen barrier ile düzeltildi.
Final koşu bu düzeltmeler üzerindedir.

Clean minimum ortamın before/after karşılaştırması da PASS:35 distribution/
21432 file hash eşittir. Son Git/index/hash kontrollerinde HEAD/branch aynı,
index boş, kullanıcı dosyası aynı SHA'dadır; `git diff --check` başarılıdır.
Fiziksel power-cut, hostile FS, Linux/native CUDA veya full CPU gate için PASS
verilmez. Yukarıdaki açık maintenance/FS sınırları bu onayın kapsamıdır.

## 27. M2-D-B3 prerequisites

B3 yalnız public API ile manuel Trainer history/checkpoint ve Benchmark result
artifact örnekleri geliştirmelidir; üretim Trainer/Runner'a tracking eklememelidir.
Resume byte association ile Trainer strict plan/state doğrulaması ayrı tutulmalı,
accuracy percent/None ve restored-history attribution korunmalıdır. Native GPU
state/RNG/no-sync integration ve birleşik full/native kalite gate'leri sonraki
aşamalarda yeniden çalıştırılmalıdır. Otomatik background writer/integration,
cleanup veya performance backlog uygulaması bu değişikliğin parçası değildir.
