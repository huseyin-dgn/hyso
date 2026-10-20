# M2-P-B8 / P18 — Roofline capability gate

2026-10-21, Europe/Istanbul. Başlangıç branch `feature/hyso-models-next`, HEAD
`ef64fd1beec1cde040d503ee8aa03a9d87123e37`. Bu görevde yalnız capability audit
yapıldı; Roofline production API, counter adapter veya RooflineResult eklenmedi.
**P18 planned / BLOCKED.** B7'nin tarihsel full-scope NO-GO kaydı değiştirilmedi.
Kullanıcının bu görevdeki açık talebi bağımsız P19 geliştirmesini yetkilendirir.

Yerel yeni kanıt dizini **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb9-20261021`.
`capability.py` ve `capability.json` yeni salt okunur probe kaydıdır.
Önceki B6-R1 trace araştırması tarihsel kanıttır; yeni trace testi diye sayılmaz.

## Gereken kanıt ve erişilebilen veri

| P18 gereksinimi | Bu ortamda bulgu / kabul eksiği |
|---|---|
| Gerçek operator/kernel FLOP coverage | P12 yalnız desteklenen ATen işlemlerinin analitik modelleridir; donanım FLOP counter değildir. Fused/nonlinear/custom coverage ve kernel FLOPs doğrulanmadı. |
| Gerçek device/kernel time | CUDA event stream interval P19'da ölçülebilir; tek kernel süresi/attribution değildir. Kineto CUDA activity erişilemiyor. |
| DRAM/cache traffic counters | Erişilen backend'lerde doğrulanmış byte/transaction counter yok; null/unavailable. Parametre byte veya allocator byte bunun yerine konulmadı. |
| Aynı kernel/time/hierarchy scope | Kernel kimliği, counter export ve replay correlation birlikte yok; eşleşme kanıtlanamıyor. L1/L2/DRAM birbirine karıştırılmadı. |
| Measured compute throughput | Aynı scope gerçek FLOP/time yok; FLOP/s üretilmedi. GPU busy yüzdesi FLOP/s değildir. |
| Memory bandwidth calibration | Aynı device/hierarchy için kalibrasyon yapılmadı; P15 H2D/D2H transfer rate, GPU DRAM bandwidth tavanı olarak kullanılmadı. |
| Theoretical / empirical ceilings | İkisi ayrı gereklilik; matching precision/clock/power scope ile doğrulanmış ampirik ceiling yok. Donanım etiketi ölçülmüş ceiling sayılmadı. |
| Arithmetic intensity FLOP/byte | Eşleşmiş gerçek pay/payda yok; measured intensity/roofline teşhisi üretilmedi. |
| Profiler permissions | Counter aracı çalıştırılmadığından izinler **NOT VERIFIED**; permission denied iddiası da yok. Sistem/driver ayarı değiştirilmedi. |
| Counter replay/correlation | Replay modu, deterministik kernel eşleşmesi ve counter pass aggregation doğrulanmadı. |
| Precision/backend matching | FP32/AMP training mümkündür; bu, precision-specific counter/ceiling eşleşmesi kanıtı değildir. |

Yeni yerel probe: Python3.10.18, Torch2.5.1+cu121, CUDA build12.1,
`cuda.is_available=True`, `supported_activities={ProfilerActivity.CPU}`.
Build çıktısı `USE_KINETO`, `LIBKINETO_NOCUPTI`, `LIBKINETO_NOROCTRACER`
ve `LIBKINETO_NOXPUPTI=ON` içeriyor. RTX3060 Laptop, compute capability8.6,
driver591.74, 6144MiB bilgileri read-only nvidia-smi sorgusundan geldi (exit0).
`shutil.which('ncu')` ve `shutil.which('nsys')` null; bu yalnız PATH keşfidir,
makinedeki her kurulumun eksik olduğu iddia edilmez. nvidia-smi sensor erişimi
DRAM transaction/FLOP counter backend'i sağlamaz.

## Windows / Torch engeli

[Torch v2.5.1 Dependencies.cmake](https://github.com/pytorch/pytorch/blob/v2.5.1/cmake/Dependencies.cmake#L1434)
MSVC yolunda Kineto CUPTI desteğini kapatır. Yerel config ve CPU-only activity
bu incelenen build sınırıyla uyumludur. B6-R1'in gerçek kernel trace eksikliği
bulgusu korunur. CUDA hesaplama/event desteği, CUPTI kernel trace/counter desteği
ile aynı capability değildir. PATH, schedule veya synchronize eklemek bu
compile-time eksikliği gidermez. Torch/CUDA/driver kurulumu değiştirilmedi.

## Alternatif ortam ve minimum native acceptance

Öneri: önceden hazırlanmış CUPTI etkin Linux CUDA PyTorch ortamı ve donanım/izin
uyumlu Nsight Compute veya eşdeğer doğrulanabilir counter backend'i. Bu ortam
bu görevde denenmedi; HYSo desteği/PASS ilan edilmez. Windows'taki başka native
araçların teorik varlığı da bu build'in Kineto sınırını kaldırmış sayılmaz.
[NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
metric collection, replay ve Roofline scope gereksinimlerinin resmi referansıdır.

Minimum yeni kabul kapısı:

1. Native küçük kernel için gerçek isim, finite device süresi, izin ve tool/build
   provenance; seçilen precision ve backend'in açık kanıtı.
2. Aynı kernel ve pass/replay kimliğine bağlı gerçek FLOP coverage ve belirli
   memory hierarchy traffic byte/counter export; ham dosya/SHA/correlation.
3. Bağımsız compute ve aynı hierarchy memory calibration; theoretical ve
   empirical ceilings ayrı, birimler ve clock/power/precision scope açık.
4. Bilinen workload ile FLOP/byte, FLOP/s ve GB/s arithmetic doğrulaması;
   eksik counter/izin/coverage halinde null + açık unsupported sebebi.
5. Bounded isolated native çalıştırma, replay/state güvenliği, immutable export
   ve artifact integrity. Başarısızlıkta tahminin measured etiketiyle sunulmaması.

Bu kapı şu anda geçilemedi. B8 ertelendi. P19 yalnız gerçek Trainer sınırlarını,
host monotonic clock ve bağımsız CUDA event interval'lerini gerektirir; DRAM
counter veya kernel attribution'a bağımlı değildir. P19'un ilerlemesi B4 riskini,
B6 profiling eksiklerini veya B7 overlap eksikliğini kapatmaz.
