# HYSo UX-06 ? Smart Builder ergonomics

Verification: **PASS**. Branch: `feature/hyso-models-next`.
Starting HEAD: `fe728fb6f630aaf56346050b4e8ac06e2ccf0cdf`.
No AGENTS.md was found in the repository or its ancestors.

## Changed API

| Method | Optional inferred field |
| --- | --- |
| ModelBuilder.conv2d | in_channels: int or None = None |
| ModelBuilder.batch_norm | num_features: int or None = None |
| GraphModelBuilder.conv2d | in_channels: int or None = None |
| GraphModelBuilder.batch_norm | num_features: int or None = None |
| GraphModelBuilder.classification_head | in_features: int or None = None |

Omission or explicit None requests inference. Explicit numeric values are never
corrected and use the original scalar/group/channel/geometry validators. A failed
smart append leaves configuration unchanged. Node IDs, connections, output names,
class counts and all other component parameters stay explicit.

Sequential inference reads the existing validated current CHW feature shape.
Graph inference analyzes only the named source and its ancestors with the existing
_analyze/_describe code. A temporary metadata output is never stored in the graph.
There is no last-node heuristic, inserted connection, weight allocation, forward
probe, checkpoint load, training or GPU query during inference.

Known graph sources validate explicit dimensions immediately. Explicit sizes retain
legacy deferred validation for unresolved forward references; final preflight/build
must still validate the complete graph. Missing references, cycles, invalid upstream
geometry, multiple direct inputs to a unary node and unsupported NCHW ranks do not
supply inferred sizes. SHAPE_INFERENCE_UNAVAILABLE is registered in the existing
TR/EN diagnostic catalog; no new exception class or model algorithm was introduced.

## Classification and compatibility

classification() retains its existing schema2 meaning: it starts a sequential
classifier. Its native head derives channels for avg/max pooling or the full CHW
product for pooling=none. It does not invent class counts, labels, second heads or
reshapes. Graph vector heads require explicit pooling=none under the native rules.

Resolved sizes are ordinary integers in the existing component parameters. Schema2
head in_features remains derived native metadata rather than a new serialized field.
Schema1/2/3 JSON, reload-and-continue behavior, state_dict keys/values and CPU outputs
were compared with explicit-size calls. Generic add/Registry/config import behavior
and required native fields are retained. Other components and fixed complete-model
profiles keep their original explicit contracts. Unknown shapes are never guessed.

Development **85/85**, Commercial **57/57**, and Graph fusion **4/4** are preserved,
including the real add_inputs/concat/multiply/weighted_sum methods. Literal choices,
parameter names, other defaults, return types and autocomplete remain intact.
Ten native geometry functions are AST-identical to the expected UX-05 HEAD.
Analyzer, Advisor, training and measurement implementation files were not modified.

api-topics.json advertises inferred_parameters for the five convenience routes;
Registry constructor requirements remain unchanged. Existing topics/aliases are kept.
Docstrings, docs/guides/ux-06-smart-builder.md and the examples index explain the rules.

## Working short CNN

```console
python -m examples.models.smart_builder
```

Observed output: `{"schema_version": 2, "output_shape": [2, 3]}`.
The example is examples/models/smart_builder.py. cnn_builder() creates metadata only;
the script explicitly builds fresh CPU weights and performs one inference, with no
training. Existing examples were not rewritten.

## Test evidence

- Initial existing Jedi checks: **2 PASS** before implementation.
- Selected API/packaging group: **523 PASS** across six specified test files.
- Builder core: **935 final PASS** across eight specified test files; 933 initially
  passed and two updated expectations passed on targeted rerun.
- Merged per-case ledger: **1,458 distinct PASS**, including **61 new UX-06 cases**.
- Real signatures, Jedi parameter discovery, installed profile completion,
  wheel/sdist projection and installed package checks passed.
- Python 3.10 source parsing and diff whitespace checks passed.
- No configured Pylance check or installed Pyright CLI/module was found.
  Pylance/PyCharm GUI completion was **not verified**.

Tests cover multi-layer/channel-rearrangement inference, named sources and fusions,
explicit values and groups, spatial/rank mismatches, unresolved sources/cycles,
atomic failures, forward references, three pooling modes and nonsquare head inputs,
JSON/state_dict/schema regressions, metadata-only execution and the actual CPU script.

A core catalog fixture had a pre-existing documentation-only mismatch. An isolated
expected-HEAD copy and the current Registry catalogs are exactly equal; both differ
from the immutable fixture in 45 description strings. catalog-baseline-proof.json
records this. The test now compares every structural field, including nested
parameter descriptions, while separately requiring nonempty public descriptions.
The golden fixture was not regenerated and has no Git-normalized content changes;
its working-file hash is retained. The other expectation was updated from 18 to 19
diagnostic entries for the new inference error.

Raw logs/XML, commands, baseline proof, the merged per-case ledger and preservation
records are in outputs/ux06-smart-builder/. Failed intermediate attempts are retained.
The two corrected cases are explicitly marked PASS by the latest rerun in the ledger.

No GPU, P01?P35 acceptance campaign, P31 long run or full suite was run. Every test
command names selected files; all temporary roots are fresh UUID paths in local Temp,
outside OneDrive. Native warnings are retained in raw logs.

## Preservation and commit

The original **47/47** user files match their initial SHA-256 snapshots. A further
**60** previous QA records/package artifacts match their snapshots. The prior 19
artifacts and prior QA locations were not output/delete targets. New QA is isolated;
prior temporary roots are not cleaned. There is no reset/restore/clean/rebase/push,
new dependency, dashboard or new model algorithm.

Only the following 14 UX-06 files are committed. This report belongs to the
implementation commit; its concrete SHA and the post-commit check are recorded in
outputs/ux06-smart-builder/final-summary.json and the final response.

## Task files

- `docs/guides/ux-06-smart-builder.md`
- `docs/reports/ux06-smart-builder.md`
- `examples/models/README.md`
- `examples/models/smart_builder.py`
- `hyso/data/api-topics.json`
- `hyso/models/builder.py`
- `hyso/models/graph.py`
- `hyso/models/registry.py`
- `tests/test_builder_components.py`
- `tests/test_builder_graph_compatibility.py`
- `tests/test_project_wide_pythonic_api.py`
- `tests/test_pythonic_builders.py`
- `tests/test_registry_completion.py`
- `tests/test_smart_builder.py`
