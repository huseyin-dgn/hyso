# M2-P-B6 — Conditional acceptance ve güvenli commit hazırlığı

2026-10-20, Europe/Istanbul.

**B6: CONDITIONAL GO — development continuation only.**

Kullanıcı, aşağıdaki doğrulanmış kapsam ve açık eksikliklerle geliştirmeye
devam edilmesini açıkça kabul etti. Bu kayıt bir önerinin tekrarı değil,
kullanıcının koşullu kabulünün kaydıdır. Full B6 GPU gate PASS veya
stable/public/PyPI release onayı verilmedi.

| Kapsam | Kabul edilen durum |
| --- | --- |
| P12 | PARTIAL; desteklenen analitik FLOPs/MACs hesapları doğrulandı; genel/hardware FLOP coverage iddiası yok |
| P13 | PARTIAL; CPU layer profiling doğrulandı; CUDA layer attribution eksik |
| P14 | CPU PASS; CUDA kernel profiling UNSUPPORTED; backlog genel statüsü PARTIAL |
| CUDA profiler engeli | Mevcut Torch 2.5.1+cu121 Windows/MSVC build'inde LIBKINETO_NOCUPTI; gerçek GPU forward kernel trace desteğini kanıtlamaz |
| B4 native crash | Reliability risk OPEN; kök neden UNRESOLVED |
| Release | Stable/public/PyPI onayı yok; ayrı release değerlendirmesi gerekir |

Önceki [B6 FAIL/PARTIAL ve gate kaydı](m2-p-b6-flops-layer-operator-profiling.md),
[R1 capability incelemesi](m2-p-b6-r1-cuda-profiler-capability.md),
[B4 koşullu kabul](m2-p-b4-conditional-acceptance.md) ve B4/R1/R2/R3 crash
kanıtları değiştirilmedi. P12/P13/P14 backlog statüleri `partial` olarak kaldı.
B5/P17'nin doğrulanmış sonuçları korundu. Koşullu kabul geçmiş failure'ları
silmez, CUDA kernel ad/süre/correlation veya layer attribution PASS ilan etmez.

## Kaynak geçerliliği ve test kanıtlarının korunması

Bu görevde test, profiler probe, replay veya CPU/native/package gate
çalıştırılmadı. Önceki B6 test kanıtları **tarihsel** olarak korunur:
38 yeni CPU kontrolü ve 1595 distinct regression kontrolü; toplam 1633 distinct
CPU kontrolü birden fazla invocation üzerinden elde edilmişti. İlk regression
koşusundaki 1593 PASS / 1 FAIL ve rehber düzeltmesi sonrası packaging yeniden
doğrulaması eski audit'te korunur; tek 1633-PASS pytest koşusu diye sunulmaz.
92 distinct CUDA validation senaryosu ve 3 Windows RAM senaryosu önceki B6
kanıtlarıdır; bunlar CUDA kernel profiling PASS anlamına gelmez. Minimum
Torch 2.2.2+cpu, wheel/sdist ve 34 production modülü byte-equality kanıtları
da korunur. Tarihsel B4 CPU2313/native196 yeni çalıştırma sayılmadı.

Salt okunur SHA256 karşılaştırması:

- `%TEMP%\hyso-m2pb6-20261020\gate-source-freeze.json`: **39/39 eşleşti**.
- `%TEMP%\hyso-m2pb6-r1-20261020\initial.json`: **182/182 eşleşti**.
- Sonuç mevcut kaynakların önceki kanıt kaynaklarıyla eşleştiğini gösterir;
  yeni test PASS sonucu değildir. Yeni kabul belgesi eski paket artifact'inde
  yer almıyordu; bu belge için yeni paket üretildiği iddia edilmez.

Bu hazırlığın başlangıç/final hash envanteri repository dışındaki
`%TEMP%\hyso-b6-commit-prep-20261020` dizinindedir. TEMP kanıtları, dump,
trace, log, wheel/sdist ve diagnostic script'leri commit listesine dahil değildir.

## Git envanteri ve faz ilişkileri

Branch `feature/hyso-models-next`; HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`. Başlangıç index'i boş.
`git status --short`, `git diff --name-status` ve
`git ls-files --others --exclude-standard` ile envanter çıkarıldı.

Başlangıç **30 dosya = 5 modified tracked + 25 untracked**.
Bu kabul dosyası eklendikten sonra **31 dosya = 5 modified + 26 untracked**.
Commit adayı **30 dosya = 4 modified + 26 untracked**; kullanıcıya ait
tek tracked değişiklik commit dışında kalır.

| Grup | Sayı | Dosya kapsamı |
| --- | --- | --- |
| B4 | 9 | sweeps.py; sweep CPU/native testleri; sweep rehberi; B4 final gate, R1/R2/R3 ve conditional audit'leri |
| B5 | 7 | capacity.py, _batch_worker.py; batch factory/CPU/native testleri; max-batch rehberi ve audit |
| B6 | 10 | profiling.py, _profile_worker.py, _operation_models.py; profiling factory/CPU/native testleri; rehber; B6, R1 ve bu conditional audit |
| Ortak B4–B6 | 4 | performance/__init__.py, performance/result.py, test_packaging.py, performance backlog |
| Kullanıcı; hariç | 1 | tests/test_gradient_accumulation.py |

Tam dosya adları aşağıdaki executable PowerShell listelerindedir.
Diğer kullanıcı dosyası mevcut Git envanterinde görünmedi; sonradan oluşan
herhangi bir dosya bu explicit listeye otomatik eklenmez.

Ortak farklar aynı dosyalarda iç içedir:

- `hyso/performance/__init__.py`: B4 sweep, B5 capacity ve B6 profiling public
  import/export'ları birlikte eklenmiş.
- `hyso/performance/result.py`: üç fazın feature identity/validator dispatch,
  CSV branch ve result sınıfları aynı diff hunk'larında yer alıyor.
- `tests/test_packaging.py`: module inventory 28→34; üç fazın module/API,
  rehber/installed smoke kontrolü ve argv'leri bir arada. Tek faza tüm dosyayı
  eklemek sonraki fazların henüz bulunmayan modüllerini bekletir.
- Backlog: B4 P06/P07/P08, B5 P17 ve B6 P12/P13/P14 satırları ile tarihsel gate
  metinleri birlikte. Eski başarısız gate metinleri korunmuştur.

Bu bir unresolved Git merge conflict değildir; **fazlar arasında paylaşılan
değişiklikler** vardır. B5 `_batch_worker.py`, B4 `sweeps._point` yolunu;
B6 `_profile_worker.py`, B5 `_batch_worker.py` içindeki Windows bellek guard'ını
kullanır. Final kaynakların birlikte doğrulama kanıtı var; ayrıştırılmış ara
ağaçların yeni çalışırlık kanıtı yok.

**Öneri: tek atomik B4–B6 commit.** Üç commit teknik olarak özel hunk/ara ağaç
hazırlığıyla yapılabilir; mevcut dosyaları faz başına bütün olarak `git add`
etmek bağımsız, çalışır commit garantisi sağlamaz. Ayrıştırma yeni ara durum
doğrulaması gerektirir ve bu görevin test-tekrarı-yok sınırına uymaz. Bu yüzden
doğrulanmamış üç ayrı commit komutu verilmedi; her fazın exact add listesi
aynı atomik commit için aşağıda sağlandı. Gruplar arasında commit çalıştırılmaz.

## Kullanıcının manuel uygulayacağı komutlar

Aşağıdaki bloklar **hazırlık planıdır; ajan tarafından çalıştırılmadı**.
PowerShell'de repository kökünde uygulanır. `git add .`, `git add -A`,
`git add -u` ve `git commit -a` kullanılmaz. Başlangıç index'i artık boş
değilse komut durur; mevcut staged dosyalar otomatik silinmez.

```powershell
Set-Location -LiteralPath 'C:\Users\hdgn5\OneDrive\Belgeler\GitHub\hyso'
if ((git branch --show-current) -ne 'feature/hyso-models-next') { throw 'Branch değişmiş.' }
if ((git rev-parse HEAD) -ne '724e6bc63802a2f7a85dba40fb38167c56a9415c') { throw 'HEAD değişmiş.' }
if (@(git diff --cached --name-only).Count -ne 0) { throw 'Index boş değil; önce inceleyin.' }
if ((Get-FileHash -LiteralPath 'tests/test_gradient_accumulation.py' -Algorithm SHA256).Hash.ToLowerInvariant() -ne '17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d') { throw 'Kullanıcı dosyası hash farkı.' }

$b4Files = @(
  'hyso/performance/sweeps.py'
  'tests/test_performance_sweeps.py'
  'tests/cuda_sweeps_validation.py'
  'docs/guides/performance-sweeps.md'
  'docs/audit/m2-p-b4-sweeps-precision-final-gate.md'
  'docs/audit/m2-p-b4-r1-native-replay-recovery.md'
  'docs/audit/m2-p-b4-r2-native-failfast-diagnostics.md'
  'docs/audit/m2-p-b4-r3-native-callpath-investigation.md'
  'docs/audit/m2-p-b4-conditional-acceptance.md'
)
$b5Files = @(
  'hyso/performance/capacity.py'
  'hyso/performance/_batch_worker.py'
  'tests/batch_factories.py'
  'tests/test_performance_max_batch.py'
  'tests/cuda_max_batch_validation.py'
  'docs/guides/performance-max-batch.md'
  'docs/audit/m2-p-b5-max-feasible-batch.md'
)
$b6Files = @(
  'hyso/performance/profiling.py'
  'hyso/performance/_profile_worker.py'
  'hyso/performance/_operation_models.py'
  'tests/profiling_factories.py'
  'tests/test_performance_profiling.py'
  'tests/cuda_profiling_validation.py'
  'docs/guides/performance-profiling.md'
  'docs/audit/m2-p-b6-flops-layer-operator-profiling.md'
  'docs/audit/m2-p-b6-r1-cuda-profiler-capability.md'
  'docs/audit/m2-p-b6-conditional-acceptance.md'
)
$sharedFiles = @(
  'hyso/performance/__init__.py'
  'hyso/performance/result.py'
  'tests/test_packaging.py'
  'docs/roadmap/hyso-performance-backlog-v1.1.md'
)
git add -- $b4Files
if ($LASTEXITCODE -ne 0) { throw 'B4 add başarısız; index inceleyin.' }
git add -- $b5Files
if ($LASTEXITCODE -ne 0) { throw 'B5 add başarısız; index inceleyin.' }
git add -- $b6Files
if ($LASTEXITCODE -ne 0) { throw 'B6 add başarısız; index inceleyin.' }
git add -- $sharedFiles
if ($LASTEXITCODE -ne 0) { throw 'Ortak add başarısız; index inceleyin.' }

$expectedFiles = @($b4Files + $b5Files + $b6Files + $sharedFiles | Sort-Object -Unique)
$stagedFiles = @(git diff --cached --name-only | Sort-Object -Unique)
if (@(Compare-Object $expectedFiles $stagedFiles).Count -ne 0) { throw 'Staged liste beklenen 30 dosyayla aynı değil.' }
git diff --cached --name-only
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace kontrolü başarısız.' }
git diff --cached --stat
git --no-pager diff --cached
git status --short
```

Burada staged diff'i manuel inceleyin: tam 30 aday dosya; kullanıcı dosyası,
TEMP/dump/artifact yok; crash ve UNSUPPORTED/PARTIAL kayıtları mevcut.
Git'in LF→CRLF bilgilendirmesi tek başına test sonucu değildir; staged diff
incelemesinin yerini tutmaz. İnceleme uygun bulununca ayrı komut:

```powershell
git commit -m "feat(performance): add B4-B6 tools with conditional development acceptance"
git status --short
```

Commit yalnız index'i alır; `-a` eklenmez. Başarılı commit sonrasında beklenen
tek kalan değişiklik `tests/test_gradient_accumulation.py` olur. Commit mesajı
full CUDA profiling, crash fix veya stable release başarısı iddia etmez.
Push komutu bu plana dahil değildir.

## Koruma ve B7 sınırı

Kullanıcının `tests/test_gradient_accumulation.py` SHA256 değeri doğrulandı:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Dosya hiçbir add listesinde yoktur. Bu görev yalnız bu belgeyi ekledi;
production/test/backlog/eski audit dosyaları değiştirilmedi. Stage, commit,
push, checkout, reset, rebase veya merge yapılmadı.

Bu açık kullanıcı kabulü, B6'nın **geliştirmeye devam** kapısını koşullu açar;
eski full-scope NO-GO kaydı ve eksik CUDA capability geçerliliğini korur.
B7 bu görevde başlatılmadı. Sonraki geliştirme açısından B6'nın tam GPU PASS
olmaması artık tek başına mutlak development engeli sayılmaz. Ancak B7'de
gerçek kernel/correlation verisine ihtiyaç duyan kapsam mevcut backend'de
doğrulanamaz; destekli backend ve ayrı kanıt gerektirir. Tekrarlayan native
crash, regresyon veya state corruption geliştirmeyi durdurur. B4 riski ve
stable release kapısı bu kabulle kapanmaz.
