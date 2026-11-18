> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio test selection / Test seçimi

Open Testing & Validation with the current model selected. Choose one or more check cards, inspect the selected steps, Check setup, then Start test. Changing any selection invalidates the previous setup check. Runtime checks run serially in one durable CPU job; native P01–P35 acceptance and CUDA workloads are never started by this flow.

Reborn starts with no implicit selection. A structure-only run creates a `validate` job in a separate collapsed Structure checks history, with an explicit statement that no runtime test ran. It creates no retained forward/backward result. Selecting runtime checks creates a `test` job. Runtime history shows the actual job ID, model, start time and observed phase; completed results show individual PASS/FAIL/SKIP cases and measured durations. Pending checks say Waiting for execution. Setup checking alone creates no job. The two data-dependent cards link to the real Training setup instead of pretending to run without data.

| Check | Actual scope and requirements |
| --- | --- |
| Structure | Allocation-free Core graph/shape/parameter preflight; schema1/2/3. Reports actual validity, including invalid models. |
| Forward | Fresh fp32 CPU model, synthetic zero inputs, finite output check. Concrete geometry and resource admission required. |
| Backward | Forward prerequisite, derivative of sum of eval outputs, finite observed gradients. No claim of loss suitability or gradient coverage for every parameter. No observed gradient is SKIP. |
| Serialization | Forward prerequisite, second fresh CPU module, strict real state_dict load and identical eval output on that fixture. No user checkpoint is written. |
| Data compatibility | Separate Training/Performance setup check against the user's actual JSON feature data. Training also checks targets and classification mapping. |
| Training readiness | Separate Training setup workflow for the supported single-input classification output/loss and bounded TrainingSession options. |
| Existing records | Retained JSON/JUnit cases preserve PASS/FAIL/XFAIL/SKIP, source and case messages. Import remains an explicit action. |

CPU execution admits a Core parameter upper bound <=1M, activation estimate <=1M elements, and synthetic inputs <=65536 elements. Training/benchmark use 1–32 real supplied samples; training allows 1–5 epochs. These are admission limits, not process RSS guarantees. Unsupported runtime geometry disables runtime cards; structure checks remain available. Backward and serialization perform a forward prerequisite even if the separate Forward card is unchecked, and the plan explains it.

Only selected runtime tests produce cases. Case durations are measured separately; overall duration includes setup/prerequisites. Actual runtime failures retain the job error instead of producing a PASS. The fixture is labelled `SYNTHETIC_ZERO_INPUT_CPU_EVAL` and does not establish dataset accuracy, trained quality, latency or training-loss correctness.

Türkçe: Test kartlarını seçin, planı inceleyin, yapılandırmayı kontrol edin ve testi açıkça başlatın. Yapı kontrolü tensor oluşturmaz. Gradient ve kaydet/yükle seçimi sentetik ileri hesaplamayı ön koşul olarak çalıştırır. Veri uyumluluğu ve eğitim uygunluğu gerçek dosyayla ayrı kurulum akışındadır. Sonuçlar seçili model/sürüm ve revision bağlamını korur; eski koşu başka modele bağlanmaz.
