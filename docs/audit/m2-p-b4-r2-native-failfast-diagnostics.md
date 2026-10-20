# M2-P-B4-R2 — Native fail-fast diagnostic capture

2026-10-20, Europe/Istanbul. **B4 NO-GO; B5'e geçilmedi.** Kullanılabilir native
debugger bulunmadı; R2'de yeni replay/capture denemesi yapılmadı. Buna karşılık
**orijinal çöküşün dump'ı bulundu ve exception context metadata'sı doğrulandı**.
R1'in dump bulunamadığı değerlendirmesi bu yeni bulguyla düzeltilir; geçmiş
[B4 FAIL](m2-p-b4-sweeps-precision-final-gate.md) ve
[R1](m2-p-b4-r1-native-replay-recovery.md) audit'leri değiştirilmedi.

## 1. Initial HEAD ve çalışma ağacı

Branch `feature/hyso-models-next`, HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`, index boş. B4 production/test/docs,
backlog ve R1 audit mevcut uncommitted durumlarıyla korundu. Kullanıcı
`tests/test_gradient_accumulation.py` SHA256 başlangıç/final:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
R2'nin repository'deki tek yeni dosyası bu rapordur. Production/harness/test,
version/dependency, driver/CUDA/Python/Torch, registry veya sistem ayarı değişmedi.
Commit/push/checkout/reset/rebase yoktur.

Evidence **D**: `C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb4-r2-20261020`.
initial-files.json tracked+B4/R1 untracked dosyaların başlangıç SHA envanteridir.
Önceki evidence köklerinin üst düzey dosya hash'leri ayrıca kaydedildi.

## 2. Diagnostic araç envanteri

Salt okunur Get-Command ile cdb.exe, windbg.exe, WinDbgX.exe ve ProcDump variants;
current-user Get-AppxPackage *WinDbg*; mevcut SDK Debuggers, Downloads, Local
Programs/WindowsApps aliases, Scoop/Chocolatey ve bilinen Tools/Sysinternals
konumları araştırıldı. **Çağrılabilir debugger/dump executable veya current-user
WinDbg paketi bulunmadı.** SDK Debuggers dizinleri yoktur; Windows Kits içindeki
ek bounded file taraması da sonuç vermedi. Program Files/WindowsApps fiziksel
listelemesi access denied verdi; başka kullanıcıya ait olası kurulumların yokluğu
iddia edilmez. Bu oturumda kullanılabilir tool bulunamadığı sonucu geçerlidir.
tools-inventory.json, additional-tools-check.json ve windowsapps-check.json
kapsam/erişim sınırlarını kaydeder. Hiçbir araç indirilmedi veya kurulmadı.

## 3. Dump configuration ve mevcut dump keşfi

HKLM native LocalDumps anahtarı mevcut; yalnız NVIDIA uygulamalarına ait8
application subkey gözlendi, DumpCount15/DumpType1. Global dump değerleri veya
python.exe application subkey yoktur. HKCU ve WOW6432Node LocalDumps anahtarları
yoktur. Bu current configuration geçmiş dump'ın hangi mekanizmayla yazıldığını
tek başına kanıtlamaz. NVIDIA entries değiştirilmedi; python/global LocalDumps
oluşturulmadı. localdumps-inventory.json yalnız gerekli ayar metadata'sını saklar.

Default CrashDumps konumunda iki Python dump metadata'sı bulundu. Eski10576 PID'li
dosya ilgisiz tarihli olduğundan açılmadı. **python.exe.15644.dmp** ise orijinal
PID0x3D1C=15644 ile eşleşir. WER temporary mdmp hâlâ yoktur; ancak ayrı CrashDumps
dosyası vardır. R1'deki temporary path yokluğu, bu dosyanın yokluğu anlamına gelmez.

Original dump yerinde tutuldu:
`C:\Users\hdgn5\AppData\Local\CrashDumps\python.exe.15644.dmp`.
Repository'ye veya evidence dizinine dump kopyalanmadı; dış servise yüklenmedi.

| Identity | Value |
|---|---|
| Size |295261910 byte|
| SHA256 |`c68a154e30fd66b6408f68aa88afc7d41db550bec0b6ee756ad1c14292da4ef1`|
| Header timestamp |2026-10-20 14:22:26 UTC /17:22:26 Istanbul|
| Process ID |15644|
| Exception thread ID |13196|

Dosya read-only mmap ile açıldı. Python stdlib struct kullanılarak Microsoft'un
belgelenmiş minidump header/directory/exception/module/thread alanları okundu;
yeni debugger, unwind engine veya production kodu oluşturulmadı. Parser bounds
kontrolleri ve PID/code/reason/module-offset çapraz doğrulamaları uygulandı.
Bellek payload'ı, stack bytes, command-line/environment strings veya secrets
çıkarılmadı. D'de yalnız sanitize edilmiş gerekli metadata/hash bulunur.

## 4. Gerçek native exception/context bulguları

| Dump field | Observed |
|---|---|
| Exception code / flags |0xC0000409 /0x1|
| Exception information[0] |2|
| Exception address mapping |nvcuda64.dll+0x18fd6e5|
| Faulting module version |32.0.15.9174|
| AMD64 context bytes |1232|
| Saved RIP |0x7ffed8efd6e5; aynı module+offset|
| Thread records / captured stacks |36 /36|
| Captured thread-stack bytes |105576 toplam|
| Loaded module records |334|

Exception code/reason/module offset dump içinden okunmuştur; yalnız WER metninden
tahmin edilmemiştir. WER BEX64/record159956–159957, PID/timestamp/module version ve
offset ile tutarlıdır. Reason2 stack-cookie-check-failure alt kodunu belirtir;
bu, memory corruption'ın kim tarafından başlatıldığını veya driver root cause'u
kanıtlamaz. Önceki R1 yorum sınırı korunur.

Relevant loaded-module metadata: python.exe/python310.dll3.10.18150.1013,
nvcuda.dll/nvcuda64.dll32.0.15.9174, cudart64_12.dll6.14.11.12010,
cudnn64_9.dll9.1.0.70. torch_cuda.dll/c10_cuda.dll kayıtlarında file-version0
olduğundan bunlardan Torch release sürümü uydurulmadı. Runtime version evidence
R1'de Torch2.5.1+cu121/CUDA runtime12010/driver API13010/RTX3060 olarak ayrıdır.
Native instruction adresi belirlendi; instruction disassembly veya symbolized
call stack elde edilmedi.

**36 stack memory region'ın mevcut olması,36 unwound stack trace elde edildiği
anlamına gelmez.** D/dump-metadata-sanitized.json bu ayrımı explicit tutar:
native_stack_trace_status=not_unwound_debugger_unavailable. Yeni GPU state veya
phase trace üretilmedi. İlk fp32/EMA=False directory/pre-checkpoint sınırı R1'deki
kanıttır; dump metadata'sı belirli bir Python forward/backward/load satırını
kanıtlamaz. Venv launcher ile real conda interpreter image farkı R1 runtime
probe'u ve dump process/module identity ile tutarlıdır.

Microsoft structure references:
[exception stream](https://learn.microsoft.com/en-us/windows/win32/api/minidumpapiset/ns-minidumpapiset-minidump_exception_stream),
[module record](https://learn.microsoft.com/en-us/windows/win32/api/minidumpapiset/ns-minidumpapiset-minidump_module),
[header](https://learn.microsoft.com/en-us/windows/win32/api/minidumpapiset/ns-minidumpapiset-minidump_header).

## 5. Reproduction budget ve kök neden

R2 debugger-controlled isolated replay **0 attempts — NOT RUN/tool unavailable**;
original-order debugger replay **0 attempts**. Yeni normal replay, full CPU,
full native veya capture work-around başlatılmadı. Dolayısıyla R2 için yeni PASS
veya yeni NOT REPRODUCED test sonucu üretilmez. R1'in iki başarılı replay gözlemi
geçmiş kanıt olarak kalır ve kök neden düzeltmesi sayılmaz.

**D — root cause unresolved.** A production regression, B harness ownership bug
veya C PyTorch/CUDA/driver interaction kanıtlanmadı. Dump metadata'sı daha güçlü
exception kanıtı sağlar; native call chain/unwind incelemesinin yerini tutmaz.
Production/harness düzeltmesi veya regression test eklenmesi gerekçelendirilmedi.
Tahmini sync/GC/exception suppression patch'i yazılmadı.

## 6. Kullanıcının yapacağı somut hazırlık

İlk tercih, **Microsoft WinDbg'yi kullanıcı tarafından kurup mevcut dump'ı offline
analiz etmektir**. Yeni replay'den önce bu mevcut orijinal kanıt kullanılabilir.
Resmî [WinDbg kurulum kaynağı](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/)
Microsoft Store/installer veya `winget install Microsoft.WinDbg` yolunu açıklar.
Bu kurulum komutu R2'de çalıştırılmadı; kullanıcı adımıdır.

Kurulum sonrası WinDbg'de Open dump ile yukarıdaki original dosyayı açın. Başlangıç
inceleme komutları, yalnız yerel analiz için:

```text
.exr -1
.ecxr
r
kv
~* kv
lmv m nvcuda64
!analyze -v
```

İlk hedef faulting thread'in exception context'i ve36 thread'in gerçek unwind
zinciridir. Module+offset sonuçları da değerlidir; NVIDIA private symbols mevcut
olmayabilir. Symbol path/cache ve olası Microsoft symbol-server indirmeleri ayrı
kullanıcı seçimi olmalıdır; R2'de symbols indirilmedi. Ham dump veya ham debugger
logunu repo/audit/dış servise koymayın; yalnız incelenip sanitize edilmiş gerekli
module/function/offset ve teşhis özetini aktarın.

Yeni live reproduction ancak tool hazırlandıktan sonra gerekçelendirilmelidir.
Orijinal Python/cwd/flags/env korunmalı, fresh output root kullanılmalı, gerçek
Python child PID ve child-debugging policy doğrulanmalıdır. Yalnız launcher'ın
izlenmesi yeterli değildir. Budget en fazla1 isolated ve gerekçeli1 original-order;
yeterli kanıt alınırsa durulur. Debugger overhead/phase logging değişiklikleri ayrı
kaydedilir. Bu live plan R2'de uygulanmadı.

Alternatif araç resmî [Microsoft Sysinternals ProcDump](https://learn.microsoft.com/en-us/sysinternals/downloads/procdump)'tır;
kullanıcı portable aracı temin ederse target **real PID** üzerinde unhandled
exception/full dump/one-dump sınırı için `-e -ma -n 1` parametreleri kullanılabilir.
Bu, baştan debugger altında çalıştırmanın otomatik eşdeğeri değildir; geç attach
ilk crash'i kaçırabilir. Registry-wide postmortem `-i` veya WER upload `-wer`
önerilmez ve çalıştırılmadı.

Riskler: dump process memory/secrets içerir; local private erişim ve yeterli disk
alanı gerekir, debugger execution'ı durdurabilir/timing'i değiştirebilir. Yalnız
owned Python hedeflenmeli. Geri alma: WinDbg'yi Settings/Installed apps üzerinden
kaldırın veya kullanıcı `winget uninstall Microsoft.WinDbg` kullanabilir; portable
ProcDump için yalnız kendi kurulum klasörünü kaldırın. Yeni capture registry ayarı
önerilmediği için registry rollback gerekmiyor; mevcut NVIDIA entries korunmalı.
Kullanıcı ayrıca WER LocalDumps isterse bunun scope/retention/ACL ve önceki değer
backup/restore planı ayrı onay gerektirir; python.exe kuralı başka Python süreçlerini
de etkiler. [Resmî WER yapılandırması](https://learn.microsoft.com/en-us/windows/win32/wer/collecting-user-mode-dumps).

## 7. Test/package validity ve final karar

R2 production/test/packaging kaynaklarını değiştirmedi. Önceki feature P06/P07/P08
PASS, full CPU2313 PASS/1 expected SKIP, R1 complete native198 executions/196
distinct PASS/2 duplicates ve29-module package/minimum Torch2.2.2 CPU kanıtları
korunur; R2'de yeniden çalıştırılmış gibi sunulmaz. Başlangıç source/evidence SHA
envanterleriyle final karşılaştırması verification-summary.json'dadır. Dump hash
önce/sonra aynıdır; dump dosyası taşınmadı/kopyalanmadı/değiştirilmedi.

**B4 NO-GO.** Somut engel artık dump yokluğu değil: mevcut native exception/stack
kanıtını unwind/yorumlayacak kullanılabilir debugger yokluğu ve kök nedenin hâlâ
belirsiz olmasıdır. Kullanıcının WinDbg hazırlamasıyla önce **mevcut dump analizi**
yapılabilir. Katı kabul kriteri gevşetilmedi, CONDITIONAL GO kabul edilmedi veya
B5 başlatılmadı.
