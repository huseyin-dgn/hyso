# HYSo GPU Admission Policy Redesign

2026-10-22, Europe/Istanbul. **Policy/offline READY; native CUDA NOT EXECUTED.**
GPU kaynak rekabeti g?venlik ihlali de?ildir. B4 native crash riski OPEN / UNRESOLVED.
Bu g?revde CUDA workload, ger?ek OOM veya GPU sens?r query'si ?al??t?r?lmad?.

## 1. Eski ve yeni politika

?ncelenen eski kaynaklar: `native_admission.py`, `native_scenario_runner.py`,
`p23_preflight.py`. Bunlar?n byte'lar? ve tarihsel davran??lar? korundu.
Eski wrapper'da isim/PID allowlist'i yoktu; her GPU PID i?in tekil explicit review,
CIM/parent/raw NVML digest binding zorunluydu. Unknown/compute PID hard red idi.
GPU utilization zaten admission e?i?i de?ildi. r3 her ?? ?rnekte free VRAM>=2048MiB,
RAM>=4GiB, disk>=12GiB ve capture ba?lang?c?ndan itibaren<=30s zorunlu k?l?yordu.
Runner import ?ncesi/sonras? expiry kontrol ediyor ve eksik P16 sensor'u reddediyordu.

Yeni politika `gpu_admission_policy.py`; yeni giri? noktas?
`gpu_scenario_runner.py`. Eski evaluator yaln?zca de?i?meyen LIMITS, JSON yard?mc?lar?
ve stdlib module loader i?in import edilir; karar i?in eski evaluate ?a?r?lmaz.
Yeni runner eski runner'dan ayr? s?r?md?r; eski forensic kan?t?n? yerinde tutmak
amac?yla kaynak dosyas? ?o?alt?ld?. ??levsel fark policy wiring, sealed identity
inventory, kontroll? failure ve forensic result metadata's?yla s?n?rl?d?r.

| Ko?ul | Eski davran?? | Yeni davran?? |
|---|---|---|
| Ba?ka GPU PID / compute i?i | Desktop d??? review red | Shared/contention warning; execution allowed |
| Unknown veya k?s?tl? PID kimli?i | NO-GO | Warning; process onay? gerekmez |
| Eksik/duplicate PID review | Red | Informational inventory-quality warning |
| Y?ksek utilization | Telemetry, hard e?ik yok | Telemetry; >=80% warning etiketi, hard e?ik yok |
| Free VRAM<2048MiB, s?f?r dahil | Red | Low-headroom warning; zorunlu minimum yok |
| WDDM N/A | Kimlik/telemetry eksikli?i red ?retebilir | Null / unsupported / reason |
| Capture>30s, eksik veya hatal? telemetry | Red | Unavailable/warning; i? ba?lamas? engellenmez |
| RAM/disk headroom<4/12GiB | Red | Warning; actual allocation/artifact failure a??k raporlan?r |
| Job/owned worker/hash/limit hatas? | Red | Hard block korunur |
| Bozuk execution s?zle?mesi veya retry | Red | Hard block; tek execution, retry0, baseline0 |
| OOM / deadline | Hata / timeout receipt | Controlled failure; ?l??m null, retry0, owned cleanup zorunlu |

30s yeni politikada yaln?zca veri kalitesi s?n?r?d?r. Eski telemetry yeniden g?ncelmi?
gibi kullan?lmaz; bu minimal uygulama onu unavailable olarak raporlar. Otomatik polling,
uygulama kapatma veya process ?ld?rme yoktur. GPU PID listesi termination girdisi de?ildir.

## 2. Warning ve hard-block ayr?m?

Warning kodlar?: shared GPU processes, unknown/compute GPU process, incomplete PID
identity, hardware contention, high GPU usage, low free VRAM, WDDM unsupported,
stale/unavailable/invalid telemetry ve d???k/eksik RAM/disk headroom.
Bunlar yaln?zca ?l??m g?venilirli?ini `unqualified` yapar. Kontroll? performans
kar??la?t?rmas? kan?t? yoksa de?er null / unsupported / reason kal?r.

Hard gates: Windows Job assignment, exact membership, worker/parent/token identity,
limit readback, full source/runtime ve ?al??an bootstrap hashleri, timeout ve
cleanup kan?t?, bounded logs/artifacts, manifest/senaryo/tek-execution s?zle?mesi,
no-retry ve result integrity. Native beklenmeyen exception/crash ba?ar?s?zl?kt?r;
OOM s?n?fland?rmas? yaln?zca ger?ek typed `torch.cuda.OutOfMemoryError` ?zerinden yap?l?r.
Exception metniyle OOM tahmini veya otomatik retry yoktur.

Policy `eligible/execution_allowed` yaln?zca bu politikan?n red ?retmedi?ini s?yler.
`hard_safety_status=pending_runner_checks` tek ba??na Job kabul? veya kullan?c?
execution yetkisi de?ildir. Runner ger?ek hard gate'leri ayr?ca ve fail-closed uygular.
Synthetic safety receipt testleri bu ayr?m? do?rular.

Korunan s?n?rlar: worker60s / outer90s; process4GiB private commit;
child Job8GiB/4 process; outer Job12GiB/8 process; allocator fraction0.15;
kill-on-close; stdout/stderr65536byte/channel ve result2MiB.
Job/assignment/membership/readback/cleanup i?levleri AST bak?m?ndan ayn?d?r;
supervisor'da yaln?zca yeni sealed bootstrap filename de?i?mi?tir.
Kan?t: `isolation-preservation.json`.

Public BenchmarkRunner varsay?lanlar?nda GPU utilization/PID/free-VRAM admission
gate'i bulunmad?. Public API/result schema veya production kod de?i?tirilmedi.
Yeni warning'ler d?? forensic envelope'a yaz?l?r; i? benchmark result aynen kal?r.
Allocator fraction yaln?zca sahip olunan opt-in worker'da uygulan?r; kullan?c?n?n
in-process benchmark allocator'?na veya di?er uygulamalara uygulanmaz.
P23 hooks-enabled CUDA guard ve production performans algoritmalar? de?i?medi.

## 3. Offline ve dry-run sonu?lar?

`python -I -S -B -m unittest discover -s tests/forensics -p test_*.py -v`
**42/42 PASS, 0 failure/error**: yeni17 policy testi + ?nceki25 regresyon testi.
Eski unknown/stale rejection testleri tarihsel evaluator i?indir; yeni politikayla
kar??t?r?lmaz. Log: `C:\Users\hdgn5\AppData\Local\Temp\hyso-gpu-policy-redesign-20261022/offline-tests.log`.
Typed OOM testi ger?ek Torch yerine yaln?zca stdlib fake exception s?n?f? kullan?r.

Yeni policy testleri unknown PID?warning/allowed,100% kullan?m?warning/allowed,
0MiB free?warning/allowed, stale?unavailable, N/A?unsupported, her hard receipt'in
tek tek eksik/yanl?? olmas??blocked, ger?ek gate receipt ownership/limit mismatch,
source/runtime/manifest fark? ve retry/senaryo s?zle?mesi, typed OOM no-retry,
timeout i?in cleanup zorunlulu?u, sonu? b?t?nl??? ve izolasyon AST e?li?ini kapsar.

Yeni sealed adayda ger?ek Windows Job + yaln?zca stdlib sim?le worker:

| Kontrol | Sonu? | Son active process | T?m owned PID ??kt? |
|---|---|---|---|
| success | PASS | 0 | yes |
| timeout | PASS | 0 | yes |
| outer-timeout | PASS | 0 | yes |
| log-overflow | PASS | 0 | yes |
| simulated-oom | PASS | 0 | yes |

Normal dry-run60/90 deadline config'i, hard limit readback ve assignment receipt'lerini
do?rular. Negative timeout'lar limitleri art?rmadan15s kullan?r. OOM testi ger?ek
allocation/OOM ?retmez; controlled failure envelope ve owned tree cleanup'?n? s?nar.
Log overflow producer70000byte ?retir, monitor STOP verir ve65536byte tutar.
Sim?le descendant'lar yaln?zca owned Job'lar ?zerinden temizlenir. Outer timeout'ta
child final receipt'i beklenmez; outer Job active0 ve t?m Job PID exit kan?t? tutulur.
Kaynak/runtime pre/post hash PASS. Approval-absent refusal PASS; native execution
marker yok; Torch/CUDA importu ve ger?ek GPU execution0.
Makine kan?t?: `candidate/dryrun-summary.json` ve her `dry-*` directory receipt'leri.

?nceki retained capture yeni politikada yeniden de?erlendirildi: `eligible=true`,
`blockers=[]`; shared GPU, unknown PID, WDDM ve stale warning'leri var. T?m eski
GPU sample de?erleri null/unavailable'a ?evrildi. Bu tarihsel veri bug?nk? canl?
envanter veya yeni donan?m ?l??m? de?ildir. Kan?t:
`retained-capture-new-policy-decision.json`. Yeni GPU capture al?nmad?.

## 4. De?i?en dosyalar

Yaln?zca be? yeni dosya:

- `tools/forensics/gpu_admission_policy.py`
- `tools/forensics/gpu_scenario_runner.py`
- `tests/forensics/test_gpu_admission_policy.py`
- `tests/forensics/gpu_policy_dryrun.py`
- Bu yeni audit raporu.

Yeni policy i?in `gpu_scenario_runner.py` giri? noktas? kullan?lmal?d?r.
Eski `native_scenario_runner.py` ve r3 evaluator tarihsel strict politikay? korur;
?nceki snapshot/onay yeni aday i?in kullan?lmaz. Yeni politika i?in yeni seal haz?rd?r.

## 5. Korunan hash ve ?al??ma a?ac?

Branch `feature/hyso-models-next`, HEAD `a1864094cf41d242d75f634dd7fc134436fae6d6`.
Ba?lang??13 tracked dirty + ?nceki5 forensic dosya hash/status/mode bak?m?ndan ayn?;
263 tracked dosyan?n t?m? byte-identical. Eski sealed r3'?n282 dosyas? ve ?nceki
recovery evidence'?n248 dosyas? byte/mode/envanter olarak ayn?d?r. Index bo?tur.
Commit/push/reset/clean/stash/checkout/dependency veya driver de?i?ikli?i yoktur.

| ?nceki forensic dosya | Ba?lang?? = son SHA256 |
|---|---|
| `docs/audit/native-cuda-admission-isolation-recovery-20261022.md` | `f2f4f5811514f7b35581b33922c32d7aebaa746861c39a905504ada09282675b` |
| `tests/forensics/native_recovery_dryrun.py` | `134631a82107db61df63bde3d9b827a9239414cf43465d69cce72bace0f162cb` |
| `tests/forensics/test_native_admission_recovery.py` | `f38196c558eae41582fb390653ce77965fcd14f781aa151ea9a4ad4887db47f8` |
| `tools/forensics/native_admission.py` | `33ff5cf357339f2492a5182c32fae6d3d8e8cf26f7e43ad8d5b46f935e8264a0` |
| `tools/forensics/native_scenario_runner.py` | `74f9aa04e4f6acbf2924d15ebe50b6357c3c6c56aa15fd4190b845495b4ba62d` |

| Korunan dirty dosya | Ba?lang?? = son SHA256 |
|---|---|
| `docs/guides/performance-max-batch.md` | `97f21d3c2ebe9ceb040c40699363be58c21cc441e21d1e02ed15c02a012aba65` |
| `docs/guides/performance-pipeline.md` | `173fd3305c54151bbe4415fc75af9c81dd356bfc766bb1d495770e44b90e6bc0` |
| `docs/guides/performance-transfer-utilization.md` | `5821ec853ead755d17766582e082f1348eaa841db94aa26705af71f04a587fca` |
| `hyso/benchmark/runner.py` | `3dc655960de696efc39c57a395ce2e59263303a4da8da100480187c3c7d18cfe` |
| `hyso/performance/_telemetry.py` | `99c5da75a29298dff2d7ad66907d095d218d9bd061d99c6964604f9d1ccb335e` |
| `hyso/performance/capacity.py` | `8093b21ed8e23ccb9d39b62464ce495da2e2f396d5f2b41b6c7e267fe440be01` |
| `hyso/performance/cold.py` | `9484155ed2f1ee6e46ad86f677957d12cad89ed7ffd2fd353c06dc310a253b83` |
| `hyso/performance/utilization.py` | `e39724a066833057e41a420695a9eda221fc2c1ea758f7023cfa591bd3ed3aac` |
| `tests/test_benchmark_cuda_protocol.py` | `056f49fb6e5163e78467ebe9580422c3557f9157724ba5664a45381fc964ddab` |
| `tests/test_performance_max_batch.py` | `f1ef2b97b0d904e62434bd5bfb9200f05bc9b1be420eca6110207593ac80a662` |
| `tests/test_performance_pipeline.py` | `73eb3ac5c8504b3dd6dfe3f14301b19b7611cf3a77fc160f19700701426d544a` |
| `tests/test_performance_transfer_utilization.py` | `6dd7e0f53c6150ec019fefd0cfa4cf82e7ac8ea3fa521b3692061df6545181d3` |
| `tests/test_training_details.py` | `06334ca669df6950c6b452e3f19a907c9f89df5a6ca8bdd92948ee83212ec32f` |

P23 guard SHA256:
`9bacc67691ab032dfa9b6202f5ea3c4bd8b78e0653aca40d53b2fa6e86f85a0d`.
Tam ba?lang??/son kay?tlar? `start-protection.json`, `final-protection.json`;
yeni dosya ve evidence hashleri `evidence-manifest.json` i?indedir.

Yeni aday: `C:\Users\hdgn5\AppData\Local\Temp\hyso-gpu-policy-redesign-20261022/candidate`.
Run ID `8fca9c14-469e-4413-8f5e-7e34c721bef4`.
Manifest SHA256 `893da6512f5d3fa41c843ebc6e7c548ba5365f3c90d3214e7387df55897cf7b0`.
51 read-only source +26 verified runtime file. Approval ve execution marker yoktur.

## 6. Native CUDA i?in kalan ger?ek engeller

GPU PID belirsizli?i, d???k VRAM, y?ksek utilization ve capture expiry art?k engel
de?ildir. Ayr? a??k native execution yetkisi hen?z yoktur. Y?r?tmede actual Job,
identity, source/runtime, readback ve cleanup gate'leri yeniden sa?lanmal?d?r;
birinin ba?ar?s?zl??? h?l? hard block'tur. Ger?ek OOM veya timeout, kontroll? failure
ve yaln?zca owned cleanup ?retir; ba?ar?l? ?l??m diye sunulmaz. Ger?ek CUDA
import/kernel/allocator/sensor davran??? bu g?revde denenmedi. Tarihsel native crash
root cause ??z?lmedi; B4 OPEN kal?r. Disk hash do?rulamas? native loaded-memory
attestation veya t?m fiziksel VRAM i?in sandbox garantisi de?ildir.

## 7. ?lk GPU testinin ?al??t?r?labilirlik karar?

**Politika/offline haz?rl??? READY: GPU kaynak rekabeti nedeniyle NO-GO yok.**
?lk k???k Conv2d FP32 senaryosu, ayr? a??k execution yetkisi ve y?r?tme an?ndaki
hard gate'ler sa?land???nda ?al??t?r?labilir. Bu g?rev yaln?zca politika ve offline
kan?tlar? tamamlad?; GPU testi ?al??t?r?lmad? ve B4 ??z?lm?? say?lmaz.
