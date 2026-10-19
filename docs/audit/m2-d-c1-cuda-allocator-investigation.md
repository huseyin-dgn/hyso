# M2-D-C1 — CUDA Allocator Root-Cause Investigation

2026-10-20, Europe/Istanbul. **ROOT CAUSE CONFIRMED / FIX VERIFIED** —
HYSo'nun profile traversal'ında bıraktığı closure döngüsü ve bunun kontrollü
GC zamanlaması altında ürettiği negatif incremental peak için.
İlk M2-D-C sürecinin kaydedilmemiş raw değerleri geriye dönük bilinemez;
aşağıdaki -33792 byte değeri **yeni kontrollü yeniden üretime** aittir.
M2-D-C'nin PARTIAL GO kararı bu görevle otomatik FINAL GO yapılmaz.

## 1. Branch/HEAD/working tree ve kanıt

Branch `feature/hyso-models-next`; başlangıç/son HEAD
`ee7a8ca365e399ccd54262063ca7bf1602b6f134`. Index boş.
Başlangıç: kullanıcıya ait `tests/test_gradient_accumulation.py` değişikliği ve
önceki görevin untracked `docs/audit/m2-d-c-final-experiment-quality-gate.md` raporu.
İkisi de başlangıç hash'inde korundu. Commit/push/checkout/reset/rebase/branch
ve dependency/metadata/version değişikliği yapılmadı.

Yeni kanıt kökü **E**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2dc1-20261020-020945`.
Eski kanıt kökü `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2dc-20261020-013213`;
onun111 doğrudan evidence/helper/log dosyasının hash'i değişmedi. `initial.json`
başlangıç kaynak/önceki kanıt hash'lerini, `verification-summary.json` final
assertion sonuçlarını saklar. D-C, C-B2/C-B3 audit'leri ve istenen Runner/Result/
native test kaynakları incelendi.

## 2. İlk hata ve validation koşulu

Eski `benchmark-export.stdout.log`: orijinal8 senaryoda7 PASS+1 ERROR;
`NativeExportComparison.test_readonly_exports_no_runtime_calls_or_rng_change`.
Export guard başlamadan Runner result construction → `result.py:145`:
`ValueError: invalid CUDA allocator bytes`. İlk memory payload kaydedilmemiştir.
Eski single recheck1 PASS, sequence diagnostic8 PASS ve installed8 PASS,
çözüm kanıtı olarak kullanılmadı.

`result.py:134–176`: measured CUDA için11 byte alanının her biri
`type(value) is int and value >= 0` olmalı; bool/int subclass/float/NumPy scalar
reddedilir. `value` ayrıca exact int olup peak'e eşit; reason=None,
scope=`separate_single_forward`,counter_reset_performed=True.
Allocated≤reserved,peak≥baseline/current,incremental=peak−baseline ve previous
peak/reset-warning tutarlılığı ayrıca denetlenir. Byte alanlarında measured None yok.
CUDA disabled:11 alan+value=None,reason=`measure_cuda_memory_false`,reset=False.
CPU: yalnız `status=not_applicable,value=None,reason=cpu_only_b1` memory zarfı;
CUDA byte alanları CPU zarfına eklenmez.

## 3. Alan→üretici→validation eşliği

Final `runner.py` satırları; Result validation bu görevde byte-identical kaldı.
Bütün doğrudan okumalar `read()` içinde `int(torch.cuda.<API>(device))` üzerinden.

| Alan | Üretici ve anı | Kaynak satırı | Validation |
|---|---|---:|---|
| baseline_allocated_bytes | memory_allocated,ilk sync sonrası/reset öncesi |279|exact int≥0;peak≥baseline|
| baseline_reserved_bytes | memory_reserved,aynı aşama |279|exact int≥0;allocated≤reserved|
| previous_peak_allocated_bytes | max_memory_allocated,reset öncesi |280|exact int≥0;≥baseline;warning eşliği|
| previous_peak_reserved_bytes | max_memory_reserved,reset öncesi |281|exact int≥0;≥baseline;warning eşliği|
| peak_allocated_bytes | max_memory_allocated,probe forward+sync sonrası |286|exact int≥0;≥baseline/current;value eşliği|
| peak_reserved_bytes | max_memory_reserved,aynı aşama |286|exact int≥0;≥baseline/current|
| current_allocated_after_probe_bytes | memory_allocated,output tutulurken |287|exact int≥0;≤peak/reserved|
| current_reserved_after_probe_bytes | memory_reserved,output tutulurken |288|exact int≥0;≤peak_reserved|
| current_allocated_after_output_release_bytes | memory_allocated,del output+sync sonrası |292|exact int≥0;≤peak/reserved|
| current_reserved_after_output_release_bytes | memory_reserved,del output+sync sonrası |293|exact int≥0;≤peak_reserved|
| incremental_peak_bytes | HYSo:peak_allocated−baseline_allocated |298|exact int≥0;fark eşliği|

`value` peak_allocated alias'ıdır (299); state/scope/reset flag API byte counter'ı
olmayıp HYSo metadata'sıdır. Doğrudan10 okumadan farklı olarak yalnız incremental
iki farklı zamandaki değerden türetilir. `raw-api-observation.json` gerçek API
return değerlerini **int dönüşümünden önce** gözlemler:10 okumanın tamamı exact
`builtins.int` ve nonnegative; reset return'ü None. Hiçbir değere clamp uygulanmadı.
Yerel Torch2.5.1 source `installed-torch-memory-source.json` içinde.
[PyTorch v2.5.1 memory.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py)
ve [peak reset sözleşmesi](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html)
ile API/reset kapsamı karşılaştırıldı; PyTorch bug'ı ileri sürülmez.

## 4. Sınırlı doğal tekrarlar ve tanılama etkisi

Sırayla5 temiz-process single case +15 temiz-process orijinal8-case sequence:
**20/20 process başarılı,125 testcase execution PASS**. Doğal hata tekrar etmedi;
20/20 PASS kendi başına kök neden kanıtı değildir. `reproduction-runs.json`
her argv/cwd/env/exit/süre/timestamp ve diagnostic path'i kaydeder.
Her process'in `report.json` dosyası gerçek Python/Torch/CUDA/driver/GPU,
pid,device/stream,field/value/exact type,GC counts/thresholds/events ve outcome içerir.

Dış `diagnostic.py`, Runner'ın Result constructor referansını özgün validation'a
iletmeden hemen önce payload'u bellekte kaydeder; `_cuda_memory_probe` wrapper'ı
GC olaylarını probe aralığına bağlar. Model/forward davranışı değiştirilmez,
validation bypass yoktur. I/O testlerin sonunda yapılır; event timing bölgesine
dosya yazılmaz. Python wrapper/callback/record allocation'ları GC zamanlamasını
etkileyebilir; doğal tekrarlar tam perturbasyonsuz koşu diye sunulmaz.

## 5. Kök neden kanıtı ve kontrollü yeniden üretim

`_profile()` içindeki recursive `visit` şu strong cycle'ı bırakıyordu:
`visit function → closure cell(visit) → visit function`; diğer closure cell'leri
`modules/entries` üzerinden caller modelini,parameter/buffer'ları tutuyordu.
`retention-graph-before.json`: self_cycle=True,6 freevar,module1,entry2;
caller referansları kalktıktan sonra weakref canlı,explicit GC sonrası ölü.
Bu CPU graph deneyi CUDA sayaçlarından bağımsız olarak HYSo retention bug'ını
kanıtlar. Constructor ve run da `_profile()` çağırdığından eski modeller birikir.

Ardından **orijinal8 test ve gerçek modeller** kullanıldı. Kontrollü koşulda
automatic GC ertelendi; yalnız son readonly senaryosunun memory probe'unda,
baseline okumaları bittikten sonra/reset satırından hemen önce gerçek
`gc.collect()` bir Python line trace ile çalıştırıldı. Model,allocator API,
synchronize/reset sırası ve validation değişmedi; sayaçlar gerçek GPU'dandır.
Trace yalnız latency loop'u bittikten sonra probe girişinde etkinleştirildi.
Bu scheduled-GC deneyi doğal recurrence değildir; mevcut interpreter sınırında
olası bir GC interleaving'ini deterministik olarak sınar.

Eski kod:GC öncesi39936 byte,GC sonrası4096 byte; reset yeni,current düşük
seviyeden peak izlemeye başlar. Sonraki gerçek peak6144 byte:
**6144−39936=-33792 (`builtins.int`)**. Aynı readonly test,aynı Result validation
hatasını verdi: **7 PASS+1 ERROR,exit1**. Device cuda:0,stream0,
scope separate_single_forward,reset=True; reserved tüm aşamalarda2097152 byte.
`controlled-before/report.json` ham değer ve traceback'i; ilgili command/log'lar
exit1'i korur. İlk tarihsel payload'un bununla birebir aynı olduğu iddia edilmez.

| Memory alanı (hepsi builtins.int) | Eski kod/kontrollü | Fix/aynı kontrollü sıra |
|---|---:|---:|
| `baseline_allocated_bytes` | 39936 | 4096 |
| `baseline_reserved_bytes` | 2097152 | 2097152 |
| `previous_peak_allocated_bytes` | 41984 | 6144 |
| `previous_peak_reserved_bytes` | 2097152 | 2097152 |
| `peak_allocated_bytes` | 6144 | 6144 |
| `peak_reserved_bytes` | 2097152 | 2097152 |
| `current_allocated_after_probe_bytes` | 4608 | 4608 |
| `current_reserved_after_probe_bytes` | 2097152 | 2097152 |
| `current_allocated_after_output_release_bytes` | 4096 | 4096 |
| `current_reserved_after_output_release_bytes` | 2097152 | 2097152 |
| `incremental_peak_bytes` | -33792 | 2048 |

## 6. Hipotez değerlendirmesi

| Hipotez | Kanıt ve sonuç |
|---|---|
| A baseline/peak zamanlaması | Baseline/reset arasındaki gerçek release doğrulandı; ölçüm dönemleri ayrışıyor. |
| B reset counter davranışı | Daha düşük canlı allocation sonrası peak6144 gözlendi; API/reset sözleşmesiyle tutarlı. |
| C async/eksik sync | Probe sync'leri mevcut; controlled hata host GC ile üretildi. Eksik sync kanıtı yok. |
| D tensor/output yaşam döngüsü | Eski modellerin closure içinde tutulması weakref ile kanıtlandı; output-release sırası değişmedi. |
| E Python GC | Self-cycle +scheduled collection+aynı hata +fix sonrası farkın kalkmasıyla doğrulandı. |
| F stream/device | Controlled ölçümler cuda:0/stream0; native non-default stream/restoration testleri PASS. Mismatch bulgusu yok. |
| G incremental türetme | Çıkarma aritmetiği doğru; negatif int,baseline/reset arası release'in sonucudur. |
| H API return tipi | Doğrudan native10 API return'ü int≥0; negative derived alanda. Type gevşetilmedi. |
| I sıralama/runtime state | Önceki profile traversal'larının GC'ye kadar birikmesi doğrulandı; başka state nedeni ileri sürülmez. |
| J hatalı test beklentisi | Validation haklı olarak negatif değeri reddeder; assertion değişmedi. |
| K Windows/allocator sınırı | Windows/RTX3060'da ölçüldü; closure cycle CPU'da da var. Windows/PyTorch'a özgü bug kanıtı yok. |

## 7. Minimal fix ve regression

Yalnız `runner.py:_profile`: `visit(...)` çağrısı try/finally içine alındı;
finally'da `visit=None` self-referential closure cell'ini kırar. Success ve rejected
profile yolları kapsanır; caller'ın canlı modeline veya dönen modules/entries
listelerine müdahale edilmez. Global GC policy/threshold/collect,probe sırası,
peak reset,event timing,API ve schema değiştirilmedi. AST karşılaştırmasında
Runner dosyasında değişen tek function `_profile`; Result dosyası aynı byte'larda.

`tests/test_benchmark_state_safety.py` içine4 parametrik regression assertion:
profile,constructor,run,rejected_profile. Automatic GC kapalıyken caller model ve
parameter/buffer weakref'lerinin operation scope'u biter bitmez ölmesini ister.
Eski production kodunda **4/4 FAIL** (`regression-before.xml`,exit1);
fix sonrası CPU koşusunda **4/4 PASS**. `retention-graph-after.json`:
retained_without_gc=False,self-cycle graph boş.

Fix sonrası aynı scheduled-GC sekizli sıra **8 PASS,exit0**;
GC öncesi/sonrası4096→4096 byte,peak6144,incremental2048. Bu,başarılı rastgele
tekrarın ötesinde kanıtlanan lifetime mekanizmasına karşı before/after kontroldür.

## 8. Fix sonrası doğrulama

| Güncel koşu | Sonuç |
|---|---|
| Altı CPU Benchmark/protocol/state/result/export/comparison dosyası |310 PASS,0 FAIL/ERROR/SKIP/XFAIL/XPASS;8.61s,2 mevcut warning,exit0|
| Eski Benchmark identity korunumu |306/306 PASS;4 yeni regression identity|
| Native Benchmark |45/45 PASS|
| Native Export/comparison |8/8 PASS|
| Native Experiment integration |2/2 PASS|
| Final yeni wheel,installed target native tekrarları |45+8+2=55 PASS;distinct native55'e eklenmedi|

CPU JUnit `cpu-benchmark.xml`; native script JSON/stdout/stderr ve argv/env/exit
`native-commands.json`; installed kaynak/install/build/command kayıtları
`installed-summary.json` ve `installed-commands.json` altında.
Runtime Python3.10.18,Torch2.5.1+cu121,CUDA12.1,RTX3060 Laptop,driver591.74;
CUDA_VISIBLE_DEVICES=0,gerçek availability check,ayrı sıralı process'ler.
Native guards state/RNG/stream/device ve memory reset policy'yi doğruladı.

Fix değiştiği için final wheel bir kez,no-isolation/mevcut build dependencies ile
repo dışında build edildi; local/no-index/no-deps/no-compile ile ayrı target'a
kuruldu.21 modülün wheel ve installed source byte'ları checkout ile eşit;
0.2.0/Python>=3.10/Torch>=2.2.2 ve mevcut8 dependency korunur.
Wheel SHA256 `41523a224d64c0d5ea5c964c28555673d7f61cfef08ff536c54278b26ae1f810`.
Tüm actual import'lar installed target'ta; integration adapter yalnız import-target
argümanı/path ve provenance assertion ekler. Yeni dependency kurulmadı.
Full CPU/sdist/minimum Torch2.2.2 veya bütün162 senaryo **bu görevde çalıştırılmadı**;
önceki PASS değerleri güncel source doğrulaması gibi sunulmaz.

## 9. Değişen dosyalar ve legacy

Bu görevde: `hyso/benchmark/runner.py`, `tests/test_benchmark_state_safety.py` ve
yeni `docs/audit/m2-d-c1-cuda-allocator-investigation.md`.
Generic Trainer,checkpoint FORMAT_VERSION1,Result schema1,M1/attention,
ExperimentManager public API,root exports,roadmap/backlog/packaging metadata aynı.
Önceki D-C audit'i ve kanıtları değiştirilmedi. Kullanıcı dosyası başlangıç/son SHA:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
`git diff --check` başarılı; index boş,HEAD/branch aynı.

## 10. Kanıt sınırı ve final gate'e dönüş

**ROOT CAUSE CONFIRMED / FIX VERIFIED:** HYSo'nun kendi closure retention'ı,
baseline/reset arasındaki GC release'i ve negatif incremental alanının üretimi
controlled raw değerler,weakref ve source before/after ile doğrulandı.
20 doğal process'te hata çıkmadı; ilk tarihsel çalıştırmanın kayıp counter'ları
ve tam GC zamanı kesin olarak geri kazanılamaz. Bu nedenle onun raw değerlerinin
-33792 olduğu veya tüm dış allocator etkilerinin giderildiği ileri sürülmez.

Fix caller'ın keyfi cyclic garbage'ını/concurrent allocator işini kontrol etmez;
process/device allocator scope ve caller isolation sınırı sürer. Peak reset geri
alınamaz; allocator peak model-only fiziksel VRAM veya memory winner değildir.
Bir Windows/GPU/runtime profili kanıtıdır; bütünOS/driver/multi-GPU garantisi yok.

**M2-D-C final kalite kapısına güncel source ile yeniden dönülebilir.**
Bu investigation tam final gate yerine geçmez; FINAL GO/release veya gelişmiş
performans implementation onayı verilmedi. Ayrı final gate güncel source için
full CPU,tüm ilgili native scripts,min runtime ve packaging/freeze'i doğrulamalıdır.
