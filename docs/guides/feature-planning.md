# HYSo G1 — P01–P35 feature selection plan (planning-only)

**Status:** additive development candidate, not production-ready. This file describes the new `hyso.analysis.feature_plan` module, which is not automatically imported by the existing `hyso.analysis` or `hyso.advisor` public APIs.

## Goal

One pure selection contract for all 35 performance features. Supports:

- `single`: explicitly choose one P feature.
- `selected`: explicitly choose a subset.
- `automatic`: propose features based on one or more user goals.
- `full`: enumerate all 35 and show prerequisites / blocking reasons.
- One model or two models, with per-model applicability and a comparison evidence gate.

All modes **plan only**. They never measure, train, profile, open devices, load external model code, change weights, grant permissions or automatically run an acceptance stage. `PROPOSED` is not `READY` or `PASS`. `BLOCKED` can mean missing prerequisites or mandatory native verification. The P31 7200-second run always requires independent approval. The comparisons never invent a winner or silently substitute a different workload.

## Example: selected features on two models

```python
from hyso.analysis.feature_plan import (
    ModelContext, FeaturePlanRequest, plan_features,
)

baseline = ModelContext('old', task='classification', input_contract=True)
candidate = ModelContext('new', task='classification', input_contract=True)
plan = plan_features(FeaturePlanRequest(
    'selected', (baseline, candidate), features=('P01', 'P02', 'P10'),
))
for feature in plan.features:
    print(feature.feature_id, feature.selection, feature.comparison)
    for per_model in feature.models:
        print(' ', per_model.model_id, per_model.status, per_model.reasons)

assert plan.execution_authorized is False
assert all(x.execution == 'NOT_RUN' for x in plan.features)
```

`input_contract=True` is a **caller declaration**. It does not certify input dimensions, preprocessing, trained status, model identity or code safety. The real future executor must bind these to native source evidence before running.

## Example: automatic selection

```python
request = FeaturePlanRequest(
    'automatic', (baseline,), goals=('latency', 'memory'),
)
plan = plan_features(request)
```

`automatic` means **propose an evidence/review plan**, not silently execute experiments. Dependencies such as P29->P26 are included as `selection='prerequisite'`, not as authorized native runs. In `full` mode the plan always includes all 35 features, but individual entries can be blocked. No feature is dropped solely to create an attractive coverage percentage.

## Intentional fail-closed decisions

- P18: NCU availability and a reference-kernel result do **not** establish target-model kernel FLOP/DRAM attribution.
- P29: no verified comparison until matching source/metric/split/precision/device/batch evidence is independently checked.
- P30: generic MNIST checkpoint is not a task-appropriate trained camera adapter; a real video/camera source is needed.
- P31: a single uninterrupted 7200-second run requires separate explicit authorization. No partial run can be upgraded.
- P32: even a caller-declared retained 7200-second series must have identity, continuity and completion checked.
- P33: NVIDIA power sensor presence is not full matching idle/active energy evidence.
- P35: ONNX Runtime provider availability does not prove real executed node placement.
- A second model never produces a winning model without a matched native comparison contract.
- ONNX models are not silently routed through PyTorch-only measurement producers.
- Detection/segmentation are not assigned classification-specific accuracy/F1 by default. Additional adapters and task-specific metrics remain future work.

## Next integration gates (not implemented here)

1. **G2 model adapter:** immutable model/config/input/output/training provenance, task-specific adapters, protected originals, permissions/trust boundary.
2. **G3 native capability resolver:** signed/hashed evidence of actual CPU/CUDA/ORT/camera/NCU capabilities and per-feature runtime scope, overriding declarations only through verified evidence.
3. **G4 safe executor:** bounded owned worker and cancellation with explicit approval, quota, cleanup and source snapshot validation; never call model code directly from planning.
4. **G5 evidence/Advisor integration:** validate native results, compare compatible pairs and render human-readable metrics for all selected features; missing remains missing.
5. **G6 real model acceptance:** classification/detection/segmentation, negative/timeout/OOM cases and final uninterrupted P31 last.

This is a separate selection contract from the existing `hyso.advisor.planning` retained-evidence experiment recommendation rules. Existing `hyso.performance`, `hyso.analysis`, `hyso.advisor` public exports and on-disk schemas are unchanged.

## Offline verification

```powershell
python -B -m unittest discover -s tests -p test_feature_plan.py -v
```

Do not confuse this pure planning test with native measurement, security isolation, stability, numerical correctness or production acceptance tests.
