# Build a unified analysis from existing evidence

Use `hyso.analysis` explicitly; importing it does not start a benchmark or fit.
Native result formats and existing import paths remain unchanged.

```python
import hashlib
from pathlib import Path
from hyso.analysis import (
    AnalysisContext, EvidenceScope, EvidenceRecord, EvidenceIndex,
    UnifiedAnalysisReport, compare_evidence,
)

path = Path("throughput.json")  # Existing native HYSo result.
expected = hashlib.sha256(path.read_bytes()).hexdigest()
# In a real import, pin expected to your previously trusted artifact catalog.
scope = EvidenceScope(workload_id="my-resident-inference")
record = EvidenceRecord.from_json(
    path, feature_id="P02", expected_sha256=expected,
    scope=scope, origin="reused",
)
missing = EvidenceRecord.unavailable(
    "P33", status="unsupported", reason="No compatible energy sensor",
)
index = EvidenceIndex([record, missing])
context = AnalysisContext(
    workload_id="my-analysis", task="image_classification",
    model={"name": "caller-declared model"}, dataset={"name": "caller-declared set"},
)
report = index.analyze(context=context, title="My model analysis")
index.to_json("evidence-index.json")
report.to_json("analysis.json")
report.to_markdown("analysis.md")
reopened = UnifiedAnalysisReport.from_json("analysis.json")
assert reopened.to_dict() == report.to_dict()
```

`index.add(record)` returns a new immutable index. For many records, construct
one index from the list. Multiple sources (for example optimized and unoptimized
provider captures) can occupy one feature slot without discarding either.

```python
decision = compare_evidence(record, record, domain="throughput")
print(decision)
# Often insufficient_evidence: the legacy result may not prove checkpoint,
# dataset/split, preprocessing or a clock-domain identity, even against itself.
```

A declared identity remains declared. The readable report labels that gap and
keeps raw native evidence separate. Missing features have no metric and are not
zero. Metadata summaries do not silently become measured training/evaluation.

The report is a library-generated evidence view, not an Advisor. It does not
run model/factory code, recover absent metadata, recalculate historical epoch
durations or infer stable roofs/causal memory leaks/verified speedups. Read the
[typed evidence contract](analysis-evidence-contract.md) for trust, comparison,
versioning, CSV limits and the future recommendation-engine input boundary.
