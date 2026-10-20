# M2-P-B4-R3 — Native CUDA crash call-path investigation

2026-10-20. **Root cause: UNRESOLVED. B4 NO-GO; B5'e geçilmedi.** En güçlü yeni
daraltma, crash anındaki host çağrı zincirinin **SiLU forward dispatcher üzerinden
CUDA library-loading yoluna** girmesiyle uyumlu olmasıdır. Bu, SiLU'nun, HYSo'nun,
PyTorch'un veya NVIDIA'nın hatalı olduğunu tek başına kanıtlamaz.

## 1. Kaynak, kanıt ve çalışma sınırı

Branch `feature/hyso-models-next`, HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`; index boş. Mevcut B4/R1/R2 dosyaları,
kullanıcı gradient farkı ve önceki audit'ler korundu. Başlangıç/final kullanıcı
dosyası SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
R3 yalnız bu raporu ekler. Yeni replay/test/gate, production/harness patch,
driver/dependency güncellemesi, GC/synchronize workaround, commit/push/checkout/
reset/rebase yoktur.

[B4 FAIL](m2-p-b4-sweeps-precision-final-gate.md),
[R1](m2-p-b4-r1-native-replay-recovery.md) ve
[R2](m2-p-b4-r2-native-failfast-diagnostics.md) incelendi. R1'deki196 distinct
native PASS/198 execution ve B4 full CPU2313 PASS/1 expected SKIP korunmuş
kanıtlardır; R3'te yeniden çalıştırılmış gibi sunulmaz. Package29-modül/minimum
Torch2.2.2 CPU kanıtları da değişmeyen kaynaklar için önceki kanıttır.

Original dump yerinde, salt okunur kullanıldı:
`C:\Users\hdgn5\AppData\Local\CrashDumps\python.exe.15644.dmp`.
295261910 byte, SHA256
`c68a154e30fd66b6408f68aa88afc7d41db550bec0b6ee756ad1c14292da4ef1`.
Dump kopyalanmadı/yüklenmedi; rapora ham bellek veya debugger register/stack
bellek dökümü eklenmedi.

Yerel R3 evidence **E3**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb4-r3-20261020`.
Mevcut Microsoft.WinDbg1.2610.1001.0 paketindeki amd64 CDB10.0.29661.1004
kullanıldı; R3 kurulum yapmadı. Sadece dump sorguları: `.exr -1`, `.ecxr`, `kn`,
`~*kn 12`, `lmvm`, `ln` ve hedefli CPython frame/type sorguları. `-sins` ve yalnız
yerel symbol-search dizini seçildi; symbol-server yolu/indirme yapılandırılmadı.
Live process başlatma, `.call`, `.restart` veya hedef kod yürütme yapılmadı.

## 2. Exception ve native stack özeti

Dump PID15644, exception thread13196; exception0xC0000409, reason2
STACK_COOKIE_CHECK_FAILURE; nvcuda64.dll32.0.15.9174/driver591.74,
fault offset**0x18fd6e5**, instruction `int 29h`. Bu fail-fast mekanizmasıdır;
memory corruption'ın başladığı yer değildir. Kullanıcının verdiği WinDbg zinciri
R3 yerel sorgularıyla yeniden kontrol edildi.

Çağırandan fault'a doğru daraltılmış zincir:

```text
Python interpreter
  -> torch_python binding frames (private işlev kimliği belirsiz)
  -> torch_cpu at::_ops::silu::call +0x158
  -> torch_cpu at::_ops::silu::redispatch +0x77
  -> torch_cpu at::native::structured_silu_out::impl +0x58
  -> torch_cuda internal frames (private işlev kimliği belirsiz)
  -> nvcuda64 cuLibraryLoadData +0x4f
  -> NVIDIA internal frames (private işlev kimliği belirsiz)
  -> nvcuda64 +0x18fd6e5, fail-fast
```

Aradaki private frame'ler atlanarak özetlenmiştir; bu bütün native zincirin birebir
tam listesi veya Python kaynak stack'i değildir. Native fault context ve36 thread
için bounded stack sorguları mevcut dump üzerinde yapıldı.

`cuLibraryLoadData` code/options ile CUDA library yükleyen Driver API'dir:
[NVIDIA Library Management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__LIBRARY.html).
Bu frame CUDA code/library-loading yolunu destekler; yüklenen image/PTX/cubin
kimliği, JIT/NVRTC kullanımı veya başarılı kernel launch'ı kanıtlanmaz. Model
state_dict/checkpoint deserialization ile aynı işlem değildir.

## 3. Sembol güvenilirliği ve yeni daraltmanın sınırı

NVIDIA ve Torch DLL'leri export-symbol düzeyindedir; private PDB/source line
eşlemesi yoktur. `unique_dim_cuda+0x15b83`, `CompileAndLaunchKernel+0xe94a2`,
`cuProfilerStop+0x1679ea5` gibi büyük offset'ler en yakın export etiketi olabilir.
**unique/dedup, gerçek compile/launch veya profiler-stop işlemi yapıldı** sonucu
çıkarılmadı. Aynı şekilde büyük-offset `GraphRoot::apply` etiketi backward kanıtı
olarak kullanılmadı.

SiLU için daha dar ve birbirini destekleyen üç export görüldü. `ln` sorguları:

| Frame | Export başlangıcı | Sonraki export | Değerlendirme |
|---|---|---|---|
| structured_silu_out::impl+0x58 |0x7ffef11dd820|0x7ffef11dd890|Kısa offset, komşu export sınırı içinde|
| silu::call+0x158 |0x7ffef1d4cf00|0x7ffef1d4d130|Kısa offset, sonraki silu_::call öncesi|
| silu::redispatch+0x77 |0x7ffef1db4640|0x7ffef1db4760|Kısa offset, sonraki silu_::redispatch öncesi|
| cuLibraryLoadData+0x4f |0x7ffed780ae90|0x7ffed780b1f0|Kısa offset, cuLibraryLoadFromFile öncesi|

Komşu export aralığı tam private function-boundary kanıtı değildir; ancak büyük
offset'li tek etikete göre daha güçlüdür. Üç tutarlı SiLU dispatcher frame'i ve
gerçek kaynak SiLU çağrıları **forward activation yolunu güçlü aday** yapar.
Exact HYSo satırı ve corruption origin bu tabloda kanıtlanmaz.

CDB'nin Torch checksum verification uyarısı checksum mismatch kanıtı sayılmadı.
Dump module TimeDateStamp/SizeOfImage ile yerel python310/torch_python/torch_cpu/
torch_cuda/nvcuda64 PE header değerleri eşleşti (image-header-validation.json).
Bu bir full-binary hash/PDB identity doğrulaması değildir.

CPython type bilgisi yerel olarak çözülebildi. `_PyEval_EvalFrameDefault` frame28
için `f` pointer'ı alınabildi; fakat ilgili `_frame` heap sayfası dump'ta yoktur:
f_back/f_code/f_lasti/f_lineno sorguları memory-read error verdi. Salt okunur
minidump MemoryList kontrolü de adresin capture edilmediğini doğruladı. Bu nedenle
Python filename/line recovery yapılamadı; eksik heap için kaynak satırı uydurulmadı.

## 4. İlk FP32/EMA=False senaryosunun checkpoint öncesi akışı

Kaynak konumları aşağıda mevcut sabit kaynak satırlarıdır; crash-PC'yi belirli bir
Python satırına çevirdiği iddia edilmez.

1. [Replay main](../../tests/cuda_trainer_replay_conformance.py#L47): amp=False,
   ema=False ilk döngü noktasıdır; CRITICAL içindeki ema=True `dict(..., ema=ema)`
   ile override edilir. FP32,7 örnek, batch3,accum2, AdamW, OneCycleLR, ECA,
   validation=True, EarlyStopping continue profili kullanılır.
2. [seed/build](../../tests/trainer_replay_support.py#L39): CPU model oluşturma,
   `.to(device='cuda', dtype=float32)` ile parameter/buffer movement, optimizer/
   scheduler ve Trainer kurulumu. `torch.manual_seed` CUDA lazy seed callback'leri
   de planlayabilir; model movement CUDA initialization gerektirebilir. CPU
   TensorDataset/DataLoader kurulur; worker process sayısı default0'dır.
3. [Initial snapshot](../../tests/trainer_replay_support.py#L97): CUDA model state/
   parameters deepcopy/clone ve [checkpoint._rng_state](../../hyso/checkpoint.py#L119)
   ile initialized CUDA RNG okuma. Henüz checkpoint restore yoktur.
4. [Baseline execute](../../tests/trainer_replay_support.py#L180) →
   [Trainer.fit](../../hyso/training/trainer.py#L407), resume_from=None. İlk
   [prepare_save](../../hyso/training/resume.py#L238) yapı/loader/optimizer planını
   doğrular. `_model_plan` structural reference'ı CPU'da kurar; forward yapmaz.
   Data plan tensor kontrolleri CPU dataset üzerindedir. Bunların hepsi GPU
   forward diye etiketlenmedi.
5. [Observed iterator](../../tests/trainer_replay_support.py#L137) her train/val
   batch'inden önce Python/NumPy/CPU RNG ve `torch.rand(1, device='cuda')` tüketir.
   [_batch](../../hyso/training/trainer.py#L353) CPU input/target'ı GPU'ya taşır.
6. [Training forward](../../hyso/training/trainer.py#L475) → ModularCNN → stem,
   residual block/ECA, pool/flatten/classifier → finite checks, loss ve metrics.
   AMP autocast enabled=False. [Backward](../../hyso/training/trainer.py#L477)
   ve iki update window: batch cardinalities3/3/1, gradients finite/reduction/
   division, clipping, AdamW, OneCycleLR, zero_grad. Sonraki validation de aynı
   model forward'ını no_grad altında çalıştırır; ilk publication'dan önce3 train
   ve3 val batch tamamlanması gerekir.
7. [Validation/ES](../../hyso/training/trainer.py#L491) → first-best state deepcopy
   → [save_last](../../hyso/training/trainer.py#L506). Save ön kontrolleri model/
   optimizer GPU finite reductions, state cloning ve RNG capture yapabilir.
   [Serializer](../../hyso/checkpoint.py#L257) ancak payload validation sonrasında
   temp file → torch.save → fsync → os.replace ile publication yapar.

Orijinal first-case directory'de published checkpoint ve temp file olmaması,
**ilk baseline epoch publication tamamlanmadı** sınırını destekler. Serializer
ön kontrollerine hiç girilmediğini tek başına kanıtlamaz. Baseline tamamlanmadan
source/prefix ve target/suffix restore yollarına sıra gelmez.

## 5. HYSo → PyTorch → CUDA ilişkisi ve aday matrisi

Kaynakta SiLU bağlantısı somuttur:
[ModularCNN.forward_features/forward](../../hyso/models/modular.py#L86) →
[ConvNormAct.forward](../../hyso/models/blocks.py#L67) veya
[ResidualBlock.forward](../../hyso/models/blocks.py#L113). ConvNormAct default
activation ve residual final activation `nn.SiLU()`'dur; inplace=False. Bu fixture'da
stem activation, block conv1 activation ve residual final activation üç olası
SiLU sitesidir. ECA doğrudan pooling/Conv1d/sigmoid/multiply kullanır; kendi içinde
SiLU yoktur. İlk mikro-batch için bu aktivasyonların şekli(3,3,5,5), tail için
(1,3,5,5) olabilir; dump'tan active tensor shape okunmuş değildir.

Yerel Torch2.5.1 Python kaynakları: nn/modules/activation.py:431 → F.silu;
nn/functional.py:2380 → torch._C._nn.silu → native dispatcher. Bu çağrı,
native `_ops::silu::call/redispatch` frame'leriyle uyuşur. `torch_cpu.dll` içindeki
dispatcher CUDA tensor çağrısı yönlendirebilir; DLL adından CPU fallback sonucu
çıkarılmaz.

Aşağıdaki tabloda “uyum”, olası yolun kaynağa/dump'a uygunluğudur; root cause
kanıtı veya adaylar arasında istatistiksel olasılık sıralaması değildir.

| Aday/yol | Kaynak ve gerçek çağrı | Dump ile uyum | Kanıtlanmayan varsayım | En küçük sonraki doğrulama |
|---|---|---|---|---|
| SiLU forward sırasında CUDA library load | trainer.py:476 veya177 → modular.py:86/97 → blocks.py:67/113 → nn.SiLU/F.silu/native | **En güçlü immediate-path uyumu:** üç kısa SiLU export ve cuLibraryLoadData | Hangi3 SiLU site'ı, train mi validation mı; loaded code kimliği; yükleyici mi caller mı kusurlu | Mevcut dump adreslerini tam eşleşen Torch build PDB/map ile eşlemek; mümkünse Python frame/line recovery |
| Build/initial snapshot/RNG/H2D ön işlemleri | tests/trainer_replay_support.py:39/50/97/147 → Module.to/deepcopy/get_rng_state_all/torch.rand; trainer.py:380 | CUDA/module-load üretebilir; ancak mevcut SiLU dispatcher'ını doğrudan açıklamaz | Önceden oluşmuş corruption'ın burada başlayıp SiLU'da fark edilmesi | Exact source/native frame veya zaman damgalı phase evidence gerekir; şu anda yok |
| Backward/gradient checks/clipping/AdamW | trainer.py:477 →315–351; isfinite/all/div_, clip_grad_norm_, optimizer.step | Pre-publication scope içinde; fakat current SiLU-forward stack doğrudan update/optimizer kanıtı değil | İlgili aşamaya ulaşılması veya daha önceki update'in corruption üretmesi | Eşleşen operator/phase stack veya exact-build symbol eşlemesi; GraphRoot büyük etiketi yeterli değil |
| Checkpoint pre-publication validation/cloning | resume.py:274–310 → checkpoint.py:257–292 | Published file yokluğu bunları tümüyle dışlamaz; kendileri model forward/SiLU çağırmaz | Crash'in ilk save validation içinde olduğu | Python caller/source identity; şu an SiLU yolu daha uyumlu, file yokluğundan restore bug çıkarılmaz |
| AMP/GradScaler/EMA/RNG restore | trainer.py:116/117/475/477; checkpoint.py:127 | **İlk case için active AMP/EMA/restore yolu desteklenmez** | Disabled wrapper çağrılarının active AMP kernel veya EMA update olduğu | Flags/order + yerel GradScaler disabled branches zaten ayrımı doğrular; ayrı AMP denemesi bu crash'i açıklamaz |
| Harness invocation/lifetime/global state | replay support execute patches; combined subprocess.run wait; venv→real child | Python→Torch→driver chain ile genel olarak bağdaşır; belirli ownership kusuru göstermez | Parent/PID/order, önceki process veya dış GPU contention'ın nedeni olduğu | Original launcher/child/phase evidence veya causal reproduction gerekir; fresh process global driver quiescence garantisi değildir |

GradScaler disabled branches yerel torch/amp/grad_scaler.py:202,319,379,488:
scale identity; unscale/update no-op; step doğrudan optimizer.step. Deprecation
warning aktif AMP kanıtı değildir. self.ema=None, EMA update guard'ı false'dur.
Initial seed/capture gerçek olabilir; checkpoint `_restore_rng` yalnız sonraki
target restore yolundadır. “Replay testi” adı, crash'in restore sırasında olduğunu
kanıtlamaz.

## 6. A/B/C/D ayrımı ve B4 sweep erişimi

- **A HYSo native kullanım hatası:** Kanıtlanmadı. İncelenen yol ordinary Torch
  modules/tensors/autograd kullanır; raw CUDA pointer, custom native extension,
  explicit stream/event ownership veya private driver API çağrısı bulunmadı.
  Bu, tüm HYSo kaynaklı dolaylı hataları matematiksel olarak dışlamaz.
- **B harness/process lifecycle:** Kanıtlanmadı. DataLoader wrapper ve update hook
  Python seviyesindedir; patch scope/finally restore vardır, worker0 kullanılır.
  Parent önceki process sonucunu bekler ve nonzero exit'i kaydeder. İlk published
  checkpoint öncesi crash'i, sonraki case cleanup'ına bağlayan kanıt yoktur.
- **C PyTorch/NVIDIA etkileşimi:** Çöküşün **görüldüğü sınır** bu katmanlardır.
  Geçersiz input/library blob, runtime internal corruption veya driver defect
  arasında ayrım yapacak private state/symbol/causal evidence yoktur. Faulting DLL
  tek başına NVIDIA kusuru değildir; SiLU dispatcher tek başına PyTorch kusuru
  değildir.
- **D UNRESOLVED:** Desteklenen root-cause sınıflandırmasıdır. Immediate call-path
  daraltıldı; corruption origin, tetikleyici ve fix kanıtlanmadı.

B4 sweep modülleri incelenen replay import/çağrı zincirinde kullanılmıyor:
`hyso/__init__.py` models/callbacks/augmentation import eder; replay
Trainer/checkpoint/modular/support import eder. Bu dosyalarda performance/sweeps
ve run_sweep çağrısı yoktur; legacy model imports da sweep modülüne gitmez.
Karşılaştırma runtime import probe/replay yürütmeden source üzerinden yapıldı.
Dolayısıyla bu trace'i B4 precision/sweep ölçümü diye tanımlayan kanıt yoktur.
Başka script'in aynı GPU'da daha önce çalışmış olmasını replay içinde doğrudan
sweep çağrısıyla karıştırmıyoruz.

## 7. Doğrulanmış ve belirsiz kalan sonuçlar

**Doğrulandı:** original dump identity/fail-fast reason/module offset; local native
unwind'daki library-loading sınırı; kısa-offset SiLU dispatcher zinciri; kaynakta
üç SiLU site'ı; AMP=False/EMA=None; baseline/prefix/restore sırası; current dump'ta
istenen Python frame heap alanının yokluğu. Dump/PE header sanity ve başlangıç
repository hash kontrolleri metadata evidence'ı destekler.

**Belirsiz:** exact HYSo call site/train-versus-validation/microbatch; ilk SiLU
invocation olup olmadığı; loaded CUDA library veya kernel kimliği; JIT/NVRTC
gerçekleşip gerçekleşmediği; stack-cookie bozulmasının kaynağı; library input
validity; original env/module-loading policy, contention ve önceki driver state.
Native stack reporting point'tir, yürütme geçmişi değildir. Async CUDA işleri
nedeniyle earlier operation ihtimali dışlanmaz; kör sync eklenerek çözülmüş sayılmaz.

## 8. B4 kararı ve en küçük sonraki adım

**B4 NO-GO önerisi korunur; B5'e geçilemez.** Feature-level P06/P07/P08 PASS ve
2313 CPU/196 distinct native korunmuş başarıları, açıklanmamış fail-fast'ı gidermez.
Yeni kod/fix/yeniden-test başarısı üretilmedi; eski audit veya strict acceptance
kriteri değiştirilmedi.

Şimdi full CPU/native gate veya normal replay tekrarı gerekli değildir. En küçük
yararlı adım, **mevcut dump'taki SiLU/driver return-address zincirinin tam eşleşen
Torch build symbol/map verisiyle offline eşlenmesidir**; private symbols bulunamazsa
bu sınır açıkça kalmalıdır. Mevcut partial dump'ın dışarıda bıraktığı Python frame
heap'i sonradan okunamaz.

Bu offline yol yetmezse ancak ayrı yetkilendirmeyle sınırlı bir yeni diagnostic
çalışma anlamlı olabilir: exact replay source/case/phase ve üç SiLU site'ını
ayırt eden metadata, actual child PID, full user-mode dump ve native failure
context. Gerekirse aynı FP32 SiLU shapes için tek küçük library-loading probe'u
planlanabilir; onun başarılı olması original replay'in düzeldiğini kanıtlamaz.
Bu R3 görevinde bu çalışmaların hiçbiri başlatılmadı. Driver/dependency değişimi,
SiLU replacement veya synchronization patch'i için kanıt yoktur.
