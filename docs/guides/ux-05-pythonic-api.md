# Project-wide Pythonic API (UX-05)

Closed HYSo choices appear as real methods, existing string Enums, or `Literal`
annotations. Existing strings, configuration schemas, JSON, checkpoints and
native validators retain their behavior. Python 3.10 is supported.

```python
from hyso import Device, ModelBuilder, TrainingConfig

builder = ModelBuilder.classification((1, 8, 8), num_classes=2)
config = (TrainingConfig(epochs=2, batch_size=2, device=Device.cpu())
          .with_adamw(learning_rate=0.001, weight_decay=0.01)
          .with_cosine_scheduler(t_max=2))
# config.prepare_datasets(model, dataset) prepares a session.
# session.fit() is the separate, explicit training operation.
```

`with_sgd`, `with_adam`, and `with_adamw` expose optimizer options as named
parameters. `with_step_scheduler`, `with_cosine_scheduler`,
`with_onecycle_scheduler`, and `with_plateau_scheduler` expose the existing
schedulers. `with_early_stopping` exposes the existing early-stopping settings.
Each returns a new immutable `TrainingConfig` through the original validator.
Omitting an optimizer method's learning rate retains the configured rate.
Plateau scheduling and early stopping require `validation=True`, as before.

`ModelBuilder.classification`, `.regression`, and `.embedding` expose schema2
head options with named parameters. Existing `.sequential`, `.named`, `.modular`,
`.add` and Graph `.add` calls remain available for data-driven configurations.
The explicit component/factory methods remain profile-aware: Development 85/85,
Commercial 57/57, and Graph fusion 4/4. `ModelRegistry.get/validate/create` now
also advertise the current profile's component names; `.list` advertises its
categories. Installed Commercial sources omit Development-only selector types.

Direct advanced residual, self-attention and prediction-head constructors expose
their existing registry parameters. They use the same validation and native
engines. The legacy `CNNSENormalWithFit` constructor exposes CEMO's parameters.
Graph `.preview` and `performance.measure_stages` expose their delegated keyword
parameters directly.

Analysis planning exposes all P01–P35 identifiers, profiles, execution modes,
capabilities, goals, tasks and backends through typed options. Planning,
completion and help do not execute an analysis. Continue to use
`plan_analysis`, `preflight_analysis`, and the separate `execute_analysis` call
with its existing authorization requirements. The existing explicit functions
in `hyso.performance`, evaluation and quality remain the native operation APIs.

Comparison policies, allowed comparison axes, Advisor constraints and units,
experiment stages/status filters/artifact roles, performance policies, quality
metrics and evaluation tasks also have typed options. Existing Advisor Enums
such as `Metric`, `Priority` and `NativeMetric` remain unchanged.
`hyso.analysis.GraphHeadSpec` is a typed **plain dictionary** for existing P26
named-head specifications; the evaluator performs the original validation.

Reporting already offers `.to_json`, `.to_markdown`, and `.to_html` methods;
there is no format dispatcher to memorize. Report language remains TR/EN and
report level is beginner/technical/expert. Offline help language is independently
typed for TR/EN/DE/ES. New APIs link to the existing contextual help topics.

Paths, model/experiment names, graph node/output identities, dataset/split labels,
external provider identities and user metadata remain strings. Dynamic CUDA
indices remain strings too: `Device.cuda(index)` formats one without querying
hardware, allocating memory or granting execution admission. Open native source
option dictionaries and generic dispatchers retain their data-driven contracts;
their existing explicit native functions/builders provide named alternatives.
Legacy case-insensitive spellings and fallback policies are not tightened by
annotations.

Run the complete small example from the repository:

```console
python examples/pythonic_api.py
python examples/pythonic_api.py --execute --output outputs/my-pythonic-example
```

The first command prepares models, training and a plan. The second explicitly
fits one tiny CPU epoch, collects two P11 logical-footprint observations,
compares retained evidence, obtains evidence-based Advisor recommendations and
writes TR technical JSON/Markdown/HTML reports. Missing or unsupported evidence
retains its native status. This is a CPU example, not an acceptance campaign.

The source/alias and closed-option inventories can be regenerated without
importing HYSo, torch, models or measurement code:

```console
python tools/docs/audit_pythonic_api.py
```

Jedi, runtime signatures, source-level Python 3.10 parsing, native validation,
CPU equivalence and wheel/sdist tests are used for verification. Pylance and
PyCharm GUI completion have not been tested.
