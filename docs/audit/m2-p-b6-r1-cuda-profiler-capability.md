# M2-P-B6-R1 — CUDA profiler capability investigation

2026-10-20, Europe/Istanbul. Branch `feature/hyso-models-next`, HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`. İnceleme başlangıcında index boştu;
mevcut kullanıcı/B4/B5/B6 çalışma ağacı korundu.

**Sonuç B:** Kurulu Windows/MSVC PyTorch build'i gerçek CUDA kernel activity
sağlamıyor. Eksikliğin belirleyici nedeni **CUPTI desteğinin Kineto'da derleme
zamanında kapatılmış olmasıdır**. Bu sonuç runtime capability, kurulu build
bayrakları ve sürümle eşleşen resmi CMake/C++ kaynaklarıyla doğrulanmıştır.
Bu, NVIDIA sürücü kusuru veya B4 crash kök nedeni bulgusu değildir.

Yerel kanıt dizini **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb6-r1-20261020`.
`environment.json`, `probe-receipt.json`, `probe-result.json`,
`minimal-trace.json`, `probe.stdout.log`, `probe.stderr.log`,
`historical-trace-review.json`, `initial.json`, `protection-check.json`.
Probe ve supervisor kaynakları `minimal_probe.py` / `run_probe.py` aynı dizindedir;
repository kaynaklarına eklenmedi.

## 1. Mevcut ortam ve backend

| Alan | Salt okunur doğrulama |
| --- | --- |
| OS | Microsoft Windows 11 Pro, 10.0.26200, x64; Python platform etiketi Windows-10 yazıyor |
| Python | Anaconda 3.10.18, MSC v.1929, AMD64 |
| Gerçek interpreter | `C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe` |
| B6 venv launcher | `%TEMP%\hyso-p01b-tests-40awqikl\Scripts\python.exe`; aynı base interpreter/Torch |
| Torch | `2.5.1+cu121`, git `a8d6afb511a69687bbb2b7e88a3cf67917e1697e` |
| CUDA build | 12.1; `USE_CUDA=ON`, `USE_CUDNN=ON` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6 |
| Bellek | nvidia-smi 6144 MiB; Torch total 6441926656 byte |
| NVIDIA driver | 591.74 (`nvidia-smi.txt`) |
| CUDA runtime kullanılabilirliği | `torch.cuda.is_available() == True`; gerçek forward da başarılı |
| Kineto | `torch.autograd.kineto_available() == True` |
| Activity capability | `torch.profiler.supported_activities() == {ProfilerActivity.CPU}` |
| Derleme | MSVC 192930154; `USE_KINETO`, `LIBKINETO_NOCUPTI`, `LIBKINETO_NOROCTRACER`, `LIBKINETO_NOXPUPTI=ON` |

Torch'un `lib` dizininde `cupti64_2023.1.1.dll` **mevcut**: 4327424 byte,
FileVersion 2023.1.1.0, SHA256
`26fcbaa723970aaa75a0a48b1f4707af405c2558015ea29009ee96bd54661e2f`.
`kineto.lib` de mevcut. DLL'in varlığı CUPTI Activity API'nin bu process'te
işlevsel olduğunu kanıtlamaz; ayrıca DLL yükleme/API deneyi yapılmadı.
Buradaki somut engel DLL arama yolu değil, backend'in CUPTI'siz derlenmesidir.
Makine genelinde her olası toolkit kurulumunun eksik olduğu iddia edilmez.

CUDA hesaplama, Kineto'nun varlığı ve Kineto CUDA activity desteği üç ayrı
capability'dir. İlk ikisinin True olması üçüncüsünü sağlamaz.

## 2. CUPTI / build / platform neden zinciri

1. Kurulu `torch.__config__.show()` doğrudan `LIBKINETO_NOCUPTI` bildiriyor.
2. [PyTorch v2.5.1 Dependencies.cmake, L1434–1455](https://github.com/pytorch/pytorch/blob/v2.5.1/cmake/Dependencies.cmake#L1434)
   MSVC koşulunda `LIBKINETO_NOCUPTI` değerini ON'a zorlar. GPU hesaplama açık
   kalırken Kineto CPU backend'i derlenebilir. Bu bir runtime environment
   variable seçimi değildir.
3. [v2.5.1 autograd init.cpp, L342–364](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/autograd/init.cpp#L342)
   CUDA activity eklenmesini Kineto ve CUPTI/Roctracer derleme koşullarıyla
   sınırlar. Yerel build'in her iki backend'i de kapalıdır; gözlenen CPU-only
   capability bu koşulla örtüşür.
4. [v2.5.1 profiler.py, L36–46](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/profiler/profiler.py#L36)
   CUPTI olmadan CUDA talebinin legacy CUDA timing'e düşebileceğini ve bu
   sürelerin JSON CUDA trace'i sağlamadığını açıklar.

Bu sınır **incelenen 2.5.1 Windows/MSVC build'i** içindir; tüm sürümlere,
tüm Windows profiler araçlarına veya NVIDIA CUPTI'nin kendisine genellenmez.
[NVIDIA CUPTI 12.1 resmi destek bilgisi](https://developer.nvidia.com/cupti-ctk12_1)
Windows x86_64 ve Linux x86_64 desteğini, Activity/Callback API'leri için CUDA
Toolkit'in desteklediği GPU mimarilerini listeler. Windows için belirtilen
531.14 veya daha yeni driver koşuluna 591.74 sayısal olarak uygundur; bu,
bu makinede CUPTI'nin bütün API'lerinin test edildiği anlamına gelmez.
Activity tracing ile hardware counter/metric profiling aynı destek matrisi
olarak değerlendirilmedi. Gerekli GPU activity bileşeni NVIDIA tarafından
destekleniyor; **mevcut Torch Kineto derlemesinde etkin değil**.

Yalnız PATH, CUDA activity listesi, profiler schedule, `with_stack`,
`with_modules`, CUDA synchronize veya CUPTI DLL kopyalama ile bu compile-time
kapı açılamaz. Bunların hiçbiri çözüm diye uygulanmadı.

## 3. Tek minimal probe

Yeni kaynak değişikliği olmadan, HYSo import etmeden, ayrı process'te
**bir profiler context + bir Conv2d forward**, warmup/retry olmadan çalıştırıldı.
Supervisor 60 saniye timeout kullandı; process 3.23 saniyede exit **0** verdi,
stderr boştu. UTC başlangıç `2026-10-20T17:23:22.528606+00:00`.
Child'da bir CPU thread ve CUDA allocator fraction 0.1 sınırı uygulandı.
Bu sınırlı probe full gate, replay veya yük/batch araması değildir.

Model `Conv2d(1,2,kernel_size=1,bias=False).eval()`, weight=1;
FP32 ones input `[1,1,5,7]`, inference mode. Model ve input `cuda:0` üzerinde.
Capability yalnız CPU olduğu için **yalnız CPU activity istendi**;
`ProfilerActivity.CUDA` zorlanmadı. `record_shapes=True`, memory/stack/FLOPs
profiler seçenekleri kapalıydı.

| Kanıt | Yeni probe sonucu |
| --- | --- |
| Gerçek GPU forward | Output `[1,2,5,7]`, FP32, `cuda:0`, finite ve beklenen ones değerleri doğrulandı |
| CPU operator tracing | 4 olay: `aten::conv2d`, `aten::convolution`, `aten::_convolution`, `aten::cudnn_convolution` |
| CPU annotation | `r1.single_gpu_conv2d`; toplam FunctionEvent sayısı 5, hepsi `DeviceType.CPU` |
| Trace | 13 olay, 4180 byte; `cpu_op=4`, `user_annotation=1`, metadata/Trace olayları |
| Gerçek CUDA kernel activity | `cat=kernel, ph=X` sayısı **0**; CUDA/GPU runtime category de gözlenmedi |
| GPU kernel adları ve süreleri | **UNAVAILABLE / UNSUPPORTED**, sıfır süre anlamına gelmez |
| Operator→kernel correlation | **NOT VERIFIED**, eşlenecek gerçek GPU olayı yok |
| Layer→kernel attribution | **NOT VERIFIED**, CPU annotation kernel eşlemesi sağlamaz |
| Torch allocator peak | 3584 byte; yalnız Torch allocated peak, toplam process/driver VRAM ölçümü değil |

Trace SHA256:
`655b83b3f9711d31360047687b3c9f0bd48196ca1761549f71a48523660df0cd`.

Aynı forward çevresindeki iki CUDA event arasında **97.8554916381836 ms**
ölçüldü. Bu, instrumentation içeren ilk forward'ın stream interval'idir;
ilk kullanım/host launch gecikmelerini de içerebilir. Tek kernel süresi,
steady-state latency, kernel listesi veya profiler attribution değildir.
CUDA event timing burada sadece ayrı bir zamanlama mekanizmasıdır.
Başlangıç ve bitişte stream drain ölçüm sınırı içindir; production workaround
eklenmedi. CPU `aten::cudnn_convolution` etiketi host dispatch kanıtıdır;
aynı isimde bir GPU kernel kanıtı değildir.

Ek olarak **yalnız dosyalar okunarak**, eski B6 standalone CUDA probe trace'i,
dört source capture ve dört installed capture incelendi. Dokuz tarihsel trace'in
her birinde gerçek kernel event sayısı 0; CUDA/GPU category yok.
SHA/event envanteri E/historical-trace-review.json içindedir.
Bu inceleme yeni dokuz çalıştırma veya yeni test PASS sayısı oluşturmaz.
Yeni CPU/native gate, packaging testi veya replay çalıştırılmadı.

## 4. HYSo kaynak incelemesi

- `_profile_worker.py:167–181`: GPU recipe için CPU+CUDA talebi, capability
  kaydı, gerçek forward ve capture sonu stream drain mevcut. Açık context
  kullanılıyor; eksik `prof.step()`/schedule nedeniyle hiç aktif pencere
  oluşmaması durumu yok. `record_shapes`, annotations veya Observer CUPTI'yi
  açmaz. HYSo'suz probe aynı capability engelini gösterdi.
- Bu worker'ın CUDA talebi mevcut Torch'ta **legacy fallback** yoluna gider.
  [v2.5.1 autograd/profiler.py, L249–258](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/profiler.py#L249)
  desteklenmeyen CUDA activity durumunda `KINETO_GPU_FALLBACK` seçer.
  Aynı dosyanın L546–557 yolu CPU olayı adına legacy CUDA süresi ekleyebilir.
  Önceki `FunctionEvent.kernels` içindeki operator isimlerinin açıklaması
  budur; gerçek GPU kernel kimliği olarak alınmaları doğru olmaz.
- `_profile_worker.py:209–217` yalnız gerçek exported `kernel/X` olaylarını
  alır. `_cuda_metrics` (`profiling.py:219`) bunlar yoksa `unsupported`,
  kernel_count/total_kernel_us için null üretir. Raw correlation ID varlığı
  doğrulanmış eşleme ilan edilmez. Bu davranış kernel PASS uydurmuyor.
- `_profile_worker.py` module annotations ile CPU sürelerini kaydeder;
  `cuda_activity_us` null başlatılır. `profiling.py:137–154` CUDA layer
  durumunu açıkça `no_verified_kernel_to_module_attribution` ile unsupported
  tutar. **P13'ün eksiği yalnız build değildir:** kernel görünürlüğü sağlansa
  bile mevcut kaynak doğrulanmış kernel→module eşlemesi sunmaz.

Eksik P14 kernel visibility'sinin HYSo'da activity seçimini düzelterek
giderilebilecek bir hata olduğu **bulunmadı**. Worker'ın unsupported CUDA
talebini önceden filtrelemek legacy fallback davranışını değiştirebilir,
fakat CUPTI kernel trace'i kazandırmaz; bu görevde kod değiştirilmedi.
Genel bir HYSo kusursuzluk iddiası yapılmıyor. P13'teki açık kapsam sınırı da
backend değiştirilince otomatik kapanmış sayılmayacak.

## 5. A / B / C kararı ve en küçük uygulanabilir yol

| Sonuç | Değerlendirme |
| --- | --- |
| A — mevcut build'de güvenli yapılandırmayla çözülen hata | **Bulunmadı.** Compile-time NOCUPTI kapısı runtime ayarla açılamıyor. |
| B — mevcut backend gerçek CUDA kernel activity sağlamıyor | **DOĞRULANDI.** Build bayrakları + sürüm kaynakları + CPU-only capability + eski/yeni trace sonuçları örtüşüyor. |
| C — capability nedeni belirsiz, ek native debugger gerekiyor | Bu eksikliğin açıklanması için gerekli değil. Bağımsız CUPTI DLL API işlevselliği ve başka backend'in bu cihazdaki sonucu test edilmedi. |

Mevcut bağımlılıkları koruyarak uygulanabilecek en küçük karar, desteklenen
CPU/analitik kapsamı kullanmak ve CUDA kernel/layer sonuçlarını unsupported
tutmaktır. Aynı build'de yeniden denemek, ek synchronize/GC veya full gate
bu derleme engeline yeni kanıt sağlamaz. **Bu ortamda ek native profiler
incelemesi gerekli değil.**

Tam P14 için sonraki en küçük bağımsız teşhis, önceden hazırlanmış ve CUPTI
etkin Kineto içeren destekli bir Linux CUDA PyTorch ortamında aynı küçük
Conv2d'nin tek trace'idir. Kabul şartı: capability'de CUDA, exported trace'te
gerçek kernel adı ve finite süre, ardından gerçek correlation ilişkisinin
ayrı doğrulanması. Bu bir öneridir; alternatif ortam çalıştırılmadı ve PASS
verilmedi. HYSo worker'ının mevcut Windows Job Object guard'ı Linux'ta
doğrulanmış değildir; başka ortamda standalone probe başarısı bütün HYSo
worker/güvenlik kapsamının taşındığı anlamına gelmez.

Bu yol farklı runtime/build veya başka destekli profiler backend'i gerektirir;
**mevcut kurulumda configuration-only çözüm yoktur**. Hazır alternatif ortam
kullanılırsa bu Windows kurulumunun bağımlılıklarını değiştirmek gerekmez.
Yeni ortam kurulacaksa dependency/tool kurulumu ayrı iştir. Driver güncelleme
gerekliliği kanıtlanmadı; tek başına driver güncellemesi NOCUPTI'yi gidermez.
Bu görev paket, CUDA, Torch veya driver kurmadı/değiştirmedi.

## 6. P13/P14 ve B6 kapısı

- **P13:** Önceki CPU layer hierarchy/shared/nested/inclusive/exclusive/shapes
  kanıtları korunur; yeni layer gate çalıştırılmadı. CUDA layer süreleri ve
  kernel attribution **UNSUPPORTED / NOT VERIFIED**, genel durum PARTIAL.
- **P14:** Önceki CPU operator kapsamı korunur; yeni minimal probe CPU operator
  tracing'i ayrıca gözledi. Gerçek GPU forward başarılıdır; CUDA kernel
  tracing, kernel süre/adları ve operator/kernel correlation **PASS değildir**.
- **B6 önerisi: scoped CONDITIONAL GO**, yalnız daha önce doğrulanmış
  CPU/analitik alt kapsamın kullanımı için. **Full B6 GO verilmez; tam GPU
  kapsam kapısı NO-GO kalır.** Bu, önceki B6 audit'indeki PARTIAL/full-scope
  NO-GO kararını değiştirmez; backlog veya milestone statüsü güncellenmedi.
- B7'ye başlanmadı; bu inceleme B7'ye geçiş veya stable release onayı sağlamaz.

[B4 koşullu kabul](m2-p-b4-conditional-acceptance.md) yalnız development
continuation içindir. B4 native crash riski **OPEN**, crash kök nedeni
**UNRESOLVED** kalır. Bu capability bulgusu crash'i açıklamaz veya düzeltmez.
B5/P17 sonuçları korunmuştur. Tarihsel 196 distinct native senaryo ve
2313 CPU testi bu görevde yeniden çalıştırılmış gibi sayılmadı; önceki B6
gate kanıtları da tarihsel olarak korunur.

## 7. Değişiklik ve koruma kaydı

Repository'ye yalnız bu audit eklendi. Başlangıç snapshot'ındaki mevcut
`hyso/`, `tests/`, `docs/` altındaki bütün `.py`/`.md` dosyaları SHA256 ile
karşılaştırıldı; değişen/silinen yok. B6 audit, source, tests ve rehber,
B4/B5 belgeleri/backlog başlangıç byte'larıyla aynı kaldı. Index boş,
branch/HEAD aynı. Stage/commit/push/checkout/reset/rebase yapılmadı.
Koruma ayrıntısı E/protection-check.json içindedir.

Kullanıcının `tests/test_gradient_accumulation.py` SHA256 değeri aynen:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Dump açılmadı, kopyalanmadı veya yüklenmedi; rapora ham bellek eklenmedi.
