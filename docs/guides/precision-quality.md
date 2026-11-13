# P28 precision quality

`evaluate_precision_quality(model, cohort, *, candidate='int8_dynamic', device='cpu', batch_size=16)` compares a fresh FP32 reference with an independently prepared copy of the same master. Each device has its own actual FP32 evaluation. The caller model remains unchanged; RNG states are restored. Explicit CUDA use initializes a context if necessary. CUDA work still requires the applicable operational approval and isolation policy.

```python
from hyso.quality import EvaluationCohort, evaluate_precision_quality

cohort = EvaluationCohort(
    test_inputs, test_labels, sample_ids=test_ids,
    class_names=[str(i) for i in range(10)], split='test',
    preprocessing_id='fixed uint8/255 -> float32 NCHW',
)
result = evaluate_precision_quality(
    trained_fp32_model, cohort, candidate='bf16', device='cpu', batch_size=256,
)
result.to_json('precision-quality.json')
result.to_csv('precision-quality.csv')
```

Use real ordered labeled inputs, unique sample IDs, unchanged class mapping and preprocessing. No sample is silently dropped. CPU supports FP32, FP64 and BF16; the existing dynamic qint8 recipe requires a supported Sequential Linear graph. A CNN is unsupported for that recipe. CUDA supports FP32, AMP FP16 and BF16 when the native backend supports them; unavailable/unsupported execution raises with a reason, without CPU/FP32 fallback. FP64 converts the FP32 checkpoint and does not imply FP64 training. Dynamic INT8 does not use a calibration dataset; general/static graph quantization is not provided.

The former 4096-sample/1,000,000-element cohort rejection is removed. Small cohort dictionaries retain their original nested-list representation. Large ones use lossless `zlib_base64_tensor_chunks_v1` inputs with shape, sample coverage and SHA256 for each slab. One raw slab targets 1 MiB (or one larger sample); this limits temporary codec memory, not the evaluation sample count. The evaluator decodes sequentially and does not materialize all images as Python floats. `tensors()` remains an explicit full materialization method. Canonical JSON contains the complete encoded cohort, labels, IDs, logits and execution evidence; CSV is a summary and identity/provenance view, not a standalone raw-data replacement. Overall result metadata/logit storage grows with the sample count.

New P28 execution evidence records every batch's actual input bytes hash, native output shape/dtype/device, finite values, autocast state/dtype and observed Conv2d/Linear input/weight/output dtypes. These hooks describe eligible operators, not every ATen instruction or CPU hardware acceleration. Native low-precision output is widened to FP32 only for metric capture. CPU and CUDA output materialization completes before a batch is recorded. The importer rechecks coverage, source/master/device/input hashes, batch/operator contracts and derived metrics. Old schema1 P28 results retain their original analysis and remain readable; new observations are additive.

Accuracy, macro precision/recall/F1, confusion matrices and raw predictions live in the embedded evaluation results. New P28 analysis also identifies changed predictions by sample ID/label and reports their count. Signed accuracy and macro-F1 deltas are independent:

- Percentage points: `100 * (candidate - baseline)`.
- Relative percent: `100 * (candidate - baseline) / baseline`; a zero baseline returns null with `undefined_zero_baseline`.

Positive observed differences are retained. One test split does not establish a general improvement or statistical significance. Timings in an operational audit are evaluation/serialization wall time, not inference latency benchmarks.
