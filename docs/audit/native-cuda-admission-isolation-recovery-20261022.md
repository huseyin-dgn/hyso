# HYSo Native CUDA Admission & Isolation Recovery

2026-10-22, Europe/Istanbul. **NO-GO. CUDA workload executions: 0. B4 OPEN / UNRESOLVED.**
READY verilmedi; bu rapor 17 performans ozelliginin native dogrulamasi degildir.

## 1. Admission ve her GPU PID

Tek read-only capture: `C:\Users\hdgn5\AppData\Local\Temp\hyso-native-admission-recovery-20261022/live-capture.json`.
Tam CIM process/command-line/parent/creation time ve raw NVML kimligi, explicit kararlar
ve identity digestleri `live-review.json` icindedir. Siniflandirma isim allowlist'iyle
uretilmedi; onceki PID uyeligi kullanilmadi. 13 PID'nin her biri bir kez degerlendirildi:
11 desktop, 2 unknown. Unknown karar siniflandirilmis olsa da admission'i engeller.

| PID | Process | Karar | Gerekce |
|---|---|---|---|
| 1212 | dwm.exe | unknown | CIM provides only dwm.exe and winlogon parent; executable and command lines restricted, raw NVML insufficient permissions. Insufficient identity to admit. |
| 10340 | CrossDeviceResume.exe | desktop | CIM/NVML paths agree; tileid CrossDeviceResumeApp and sihost parent support desktop resume UI. |
| 10640 | StartMenuExperienceHost.exe | desktop | CIM/NVML paths agree; FullTrustApp server activation and service-host parent support Start menu UI. Parent path/command restricted. |
| 11056 | SearchHost.exe | desktop | CIM/NVML paths agree; CortanaUI server activation and service-host parent support search UI. Parent path/command restricted. |
| 12964 | msedgewebview2.exe | desktop | CIM/NVML paths agree; gpu-process, SearchHost webview arguments and browser parent tied to SearchHost support UI rendering. |
| 13592 | LockApp.exe | desktop | CIM/NVML paths agree; WindowsDefaultLockScreen server activation and service-host parent support lock UI. Parent path/command restricted. |
| 12952 | TextInputHost.exe | desktop | CIM/NVML paths agree; InputApp server activation and service-host parent support text input UI. Parent path/command restricted. |
| 17352 | PhoneExperienceHost.exe | desktop | CIM/NVML paths agree; COM background embedding arguments and service-host parent support Phone Link UI. Parent path/command restricted. |
| 19256 | ChatGPT.exe | desktop | CIM/NVML paths agree; gpu-process arguments and matching ChatGPT main parent support Codex UI rendering. |
| 20440 | codex-computer-use-swift.exe | desktop | CIM/NVML paths agree; --parent-pid 18748 matches observed Codex main parent, supporting UI companion identity. |
| 21272 | msedgewebview2.exe | desktop | CIM/NVML paths agree; gpu-process with Widgets webview arguments and matching browser parent support Widgets rendering. |
| 9880 | Code.exe | desktop | CIM/NVML paths agree; gpu-process arguments, Code user-data-dir and matching Code main parent support editor rendering. |
| 4296 | ePowerButton_NB.exe | unknown | CIM/NVML vendor path agrees but -s argument and restricted service-host parent do not establish GPU workload role. Vendor name is not authorization. |

PID1212 executable/command line null; parent winlogon alanlari da restricted;
raw NVML `[Insufficient Permissions]`, GPU memory `[N/A]`. Diger service-host
parent executable/command line null alanlari `parent_restricted_fields` ile acik
kayitlidir. Her raw `[N/A]` korunur; sifir bellek veya guvenli process sayilmaz.
Riot/Vanguard isimli process bu capture'da bulunmadi; bunlar icin otomatik desktop
muafiyeti veya guvenli kabul eklenmedi.

Uc GPU ornegi: free **5425 / 5425 / 5425 MiB**, utilization **24 / 24 / 24%**;
UUID `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`. RAM **4.333 GiB**,
disk **46.871 GiB**. Bellek esikleri capture'da gecti.
Utilization admission esigi degildir. Capture baslangicindan itibaren 30 saniye
kurali aynen korunur. Inceleme sonundaki gercek saatle karar:
`compute_or_unknown_GPU_process` ve `stale_or_future_or_reversed_capture`.
Clock geri alinmadi, eski capture taze diye sunulmadi, uygun sonuc icin polling/resampling yok.
Son karar `final-admission-decision.json`, yeni aday run/manifest kimligine baglidir.
Gelecekte her native execution oncesi yeni capture ve explicit review gerekir;
controller ve worker, CUDA importu ve initialization oncesi expiry kontrolu yapar.

## 2. Runner isolation

Yeni aday: `C:\Users\hdgn5\AppData\Local\Temp\hyso-native-admission-recovery-20261022/candidate-reviewed`.
Run ID `c02fa01c-3d3b-4166-897d-136b9f16f1c9`.
Manifest SHA256 `9c73b760e1bc3b22b12d8c5a54ebc7faf519b9b3ea9017395205b4fa90f71451`.
50 kaynak ve degismemis r3 envanterinden 26 runtime dosyasi verify edilerek
ayri read-only source/manifest/seal olusturuldu. Eski r3 dizinine yazilmadi.
Onceki iki gelistirme snapshot'i evidence'da saklidir; nihai aday bu dizindir.

Tek senaryo: FP32 `Conv2d(1,3,3,padding=1)`, input `(2,1,7,9)`, eval inference,
seed71, default stream. Mevcut `BenchmarkRunner` ile warmup1/measure3/repeats1;
P01 latency, P02 throughput, opt-in P09 peak reset; mevcut `_telemetry.gpu_observe`
ile once/sonra fiziksel cihaz genelinde P16 samples. Mevcut CUDA validation
boyutlari ve protocol'u yeniden kullanilir. Sensorlar process attribution veya
kontrollu performans karsilastirmasi kaniti diye sunulmaz. Baseline0/retry0;
exclusive execution-consumed marker tek native hakki saglar. Approval dosyasi
bu gorevde uretilmedi ve execution marker yoktur.

Controller/parent/worker interpreter/source/runtime kimligi; calisan bootstrap
hashleri; parent PID, token, exact assigned Job membership, query-only Job handle
ve bagimsiz limit readback receipt'i CUDA importundan once dogrulanir. Worker gate
bekleme limiti10s'dir. Parent/worker Job receipt'i gelmeden Torch import etmez.
Mevcut stdlib `_training_process._Job` sinifi yeniden kullanilir; production degismez.
Parent outer Job'a, worker hem miras alinan outer hem child Job'a baglidir.
Controller stdlib supervisor'dur ve workload calistirmaz; outer limitler sahip
oldugu parent/worker tree'sine uygulanir.

Hard readback: process4 GiB private commit; child8 GiB/4 process;
outer12 GiB/8 process; flags `0x2308` (kill-on-close dahil).
Worker60s / outer90s; allocator fraction.15. Stdout/stderr65536 byte/channel,
result2 MiB/file; en fazla32 work JSON artefact; bu minimal runner ayrica event
streami uretmez. Metadata receipt'leri dry-run'da toplam2 MiB altindadir
(ayni zamanda 1 MiB/process altinda ve 2000 row sinirinin altinda).
Overflow/timeout/eksik receipt/hash veya ownership hatasi STOP'tur.

Her monitor kendi olusturdugu Job tree'sini temizler; assignment failure'da
fallback yalniz kendi Popen handle'ini kullanir. Inventory PID'leri hicbir
termination yoluna verilmez. Job accounting active0, owned PID envanteri ve
process exit sorgulari tutulur. Kullanici process'i sonlandirilmadi.

## 3. Offline ve dry-run dogrulamasi

Degismemis evaluator suite'i **14 PASS**. Yeni wrapper/identity suite'i
**11 PASS**; ilk10 case'lik ara kosum bu toplama eklenmedi. Torch/conftest yok:
`python -I -S -B -m unittest discover -s tests/forensics -p test_native_admission_recovery.py -v`.
Log `recovery-offline-tests.log`; Windows PowerShell stderr'i NativeCommandError
biciminde render etmistir, interpreter exit0 ve unittest11/11 OK'dir.

Kapsam: stale capture, unknown/new PID (Riot isimli sentetik PID dahil), duplicate
classification, changed command/parent, restricted identity, missing/wrong Job
receipt, limits mismatch, manifest/source/runtime/bootstrap farki, oversized
artifact, duplicate JSON keys ve no-input-mutation. Sentetik pozitif fixture
canli process onayi sayilmaz.

Nihai adayda stdlib-only gercek Windows Job negative/cleanup kontrolleri:

| Kontrol | Sonuc | Outer elapsed s | Outer active | Kanit |
|---|---|---|---|---|
| success | PASS | 8.703 | 0 | child active=0; None |
| timeout | PASS | 17.156 | 0 | child active=0; timeout |
| outer-timeout | PASS | 15.0 | 0 | outer Job killed entire owned tree; child receipt absent as expected |
| log-overflow | PASS | 6.328 | 0 | child active=0; bounded_log_exceeded |

Normal dry-run60/90 deadline configuration'ini readback receipt'leriyle dogruladi.
Worker ve outer timeout negative testleri, maksimumlari artirmadan15s kullandi.
Simule worker120s uyuyan stdlib descendant olusturdu; tum gorulen owned PID'ler
sonda cikmis, Job active0. Log negative testinde producer70000 byte yazdi;
monitor STOP urettti ve yalniz65536 byte tuttu. Bu beklenen reddetme kanitidir.
Outer timeout'ta parent zorla sonlandigi icin child final receipt'i beklenmez;
outer Job accounting ve tum owned PID exit kaniti butun agacin temizligini gosterir.
Native yolda boyle bir durum basari degil NO-GO olur.

Native approval-absent refusal PASS; hicbir process/attempt marker veya CUDA
baslatmadi. Nihai pre/post source+runtime rehash PASS. Detayli makine kaniti:
`candidate-reviewed/dryrun-summary.json` ve her dry directory'nin assignment,
identity, Job cleanup, owned PID, bounded log/result receipt'leri.

## 4. Dosya degisiklikleri

Yalniz yeni dosyalar:
- `tools/forensics/native_admission.py`: r3 evaluator'a identity-bound wrapper ve read-only capture.
- `tools/forensics/native_scenario_runner.py`: sealed tek senaryo runner ve stdlib dry modes.
- `tests/forensics/test_native_admission_recovery.py`: offline rejection tests.
- `tests/forensics/native_recovery_dryrun.py`: Windows Job integration checks.
- Bu yeni audit raporu.

Production performans algoritmalari, mevcut13 dirty dosya, P23 guard,
sealed r3 ve gecmis audit kayitlari degismedi. Index bos; commit/push/reset/
clean/stash/checkout yapilmadi. Dependency/driver degisikligi yok.

## 5. Korunan hashler

Branch `feature/hyso-models-next`, HEAD `a1864094cf41d242d75f634dd7fc134436fae6d6`.
263 baslangic tracked dosyasi byte-identical; 282 r3 dosyasi byte/mode-identical;
13 dirty dosyanin hash/status degerleri ayni. Tam envanterler evidence'da
`start-protection.json` ve `final-protection.json`.

| Dirty dosya | Baslangic = son SHA256 | Git status |
|---|---|---|
| `docs/guides/performance-max-batch.md` | `97f21d3c2ebe9ceb040c40699363be58c21cc441e21d1e02ed15c02a012aba65` | ` M` |
| `docs/guides/performance-pipeline.md` | `173fd3305c54151bbe4415fc75af9c81dd356bfc766bb1d495770e44b90e6bc0` | ` M` |
| `docs/guides/performance-transfer-utilization.md` | `5821ec853ead755d17766582e082f1348eaa841db94aa26705af71f04a587fca` | ` M` |
| `hyso/benchmark/runner.py` | `3dc655960de696efc39c57a395ce2e59263303a4da8da100480187c3c7d18cfe` | ` M` |
| `hyso/performance/_telemetry.py` | `99c5da75a29298dff2d7ad66907d095d218d9bd061d99c6964604f9d1ccb335e` | ` M` |
| `hyso/performance/capacity.py` | `8093b21ed8e23ccb9d39b62464ce495da2e2f396d5f2b41b6c7e267fe440be01` | ` M` |
| `hyso/performance/cold.py` | `9484155ed2f1ee6e46ad86f677957d12cad89ed7ffd2fd353c06dc310a253b83` | ` M` |
| `hyso/performance/utilization.py` | `e39724a066833057e41a420695a9eda221fc2c1ea758f7023cfa591bd3ed3aac` | ` M` |
| `tests/test_benchmark_cuda_protocol.py` | `056f49fb6e5163e78467ebe9580422c3557f9157724ba5664a45381fc964ddab` | ` M` |
| `tests/test_performance_max_batch.py` | `f1ef2b97b0d904e62434bd5bfb9200f05bc9b1be420eca6110207593ac80a662` | ` M` |
| `tests/test_performance_pipeline.py` | `73eb3ac5c8504b3dd6dfe3f14301b19b7611cf3a77fc160f19700701426d544a` | ` M` |
| `tests/test_performance_transfer_utilization.py` | `6dd7e0f53c6150ec019fefd0cfa4cf82e7ac8ea3fa521b3692061df6545181d3` | ` M` |
| `tests/test_training_details.py` | `06334ca669df6950c6b452e3f19a907c9f89df5a6ca8bdd92948ee83212ec32f` | ` M` |

P23 guard SHA256 `9bacc67691ab032dfa9b6202f5ea3c4bd8b78e0653aca40d53b2fa6e86f85a0d`.
Korunan r3 manifest SHA256 `660a53dbdde6a21901bf6ad2743a3fad9bb51bad57ebd25a5a5dee46ad2444b8`.
Yeni kod ve evidence hashleri `evidence-manifest.json` icindedir.

## 6. Acik riskler ve karar

**NO-GO:** iki unknown GPU PID ve expired capture. Kullanicidan uygulama kapatmasi
veya kimliksiz process'i guvenli kabul etmesi istenmedi. Native workload yapilmadi.
Job limitleri private commit limitidir; RSS veya fiziksel VRAM sandbox'i degildir.
Allocator fraction tum CUDA/native allocation'lari kapsamaz. Dosya hashleri local
source/runtime disk identity'sidir; adversarial binary/memory attestation degildir.
Process inventory sonrasindaki degisiklikler ve kaynak yarisi tamamen engellenemez;
her yeni execution taze admission ister. Native import/loading, gercek Conv2d ve
sensor/result davranisi henuz calistirilmadi; offline PASS bunlarin kabul kaniti degildir.
Tarihsel CPU/native audit sonuclari degistirilmedi veya yeni PASS diye toplanmadi.
B4 root cause ve 17 ozelligin native acceptance'i aciktir.

Ayrica acik yetkilendirme olmadan CUDA execution yok. Gelecekte READY verilmesi bile
CUDA'nin calistigi veya B4'un cozuldugu anlamina gelmez. Bu gorev burada durur.
