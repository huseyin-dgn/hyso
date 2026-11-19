# Smart Builder ergonomics (UX-06)

`ModelBuilder.conv2d(out_channels=...)` infers an omitted `in_channels` from the
current validated NCHW feature shape. `batch_norm()` infers `num_features` from
the same channel dimension. Every append uses the existing shape and Registry
validators, so inference continues through convolution, pooling and other
supported channel/shape transformations. No tensors, weights or forward probes
are used to infer a shape.

```python
from hyso import ModelBuilder

builder = (ModelBuilder.classification((1, 8, 8), num_classes=3)
           .conv2d(out_channels=8)
           .batch_norm()
           .relu()
           .max_pool2d(kernel_size=2)
           .conv2d(out_channels=16)
           .batch_norm()
           .relu())
model = builder.build()  # Explicit weight construction; no training.
```

`classification()` starts the existing schema2 sequential classifier. Its head
continues to infer `in_features` from the final feature chain: the channel count
for `pooling='avg'/'max'`, or the full CHW product for `pooling='none'`. It does
not add a second head, infer class labels/counts, load weights, or start training.

The optional size parameters have real `int | None` annotations and default to
`None`. Omission or explicit `None` requests inference. Explicit numeric values
are validated and never corrected. Invalid values, group divisibility, channel
mismatches and unsupported ranks retain native validation. A failed append does
not mutate the builder. Flattened/vector features cannot be treated as NCHW;
restore a supported shape before appending a convolution or BatchNorm2d. Fixed
complete-model profiles remain non-appendable.

Graph nodes and connections stay explicit:

```python
from hyso import GraphModelBuilder

graph = (GraphModelBuilder({'image': [1, 8, 8]})
         .conv2d('stem', ['image'], out_channels=8)
         .batch_norm('norm', ['stem'])
         .classification_head('head', ['norm'], num_classes=3)
         .outputs(logits='head'))
```

Graph `conv2d`, `batch_norm`, and `classification_head` resolve their one optional
size only when the named source and its ancestor subgraph have a verified shape.
The existing graph analyzer handles fusion, ranks and geometry. No "last node"
heuristic or automatic connection is used. A verified concatenation contributes
the concatenated channel count; `add_inputs`/`multiply`/`weighted_sum` keep their original
shape requirements.

Missing references, cycles, incompatible upstream geometry or multiple direct
inputs to a unary layer cannot supply an inferred size and raise an actionable
`SHAPE_INFERENCE_UNAVAILABLE` error. A graph classification head uses the same
native pooling rules: a vector source requires explicit `pooling='none'`.
Known sources also validate explicit dimensions immediately. Explicit dimensions
retain the existing deferred validation behavior when a source is still a forward
reference; final preflight/build must validate the complete graph.

Inferred integers are materialized in the ordinary component parameters before
serialization. There is no auto flag, symbolic size, schema change, state_dict
change or new model algorithm. Generic `.add(...)` calls and other components
retain their existing contracts. Development 85/85, Commercial 57/57 and all four
explicit Graph fusion methods remain available according to the installed profile.

Run the short CPU example from the repository:

```console
python -m examples.models.smart_builder
```

Expected output: `{"schema_version": 2, "output_shape": [2, 3]}`. Existing examples
continue to work with explicit sizes. IDE verification uses existing and new Jedi
checks; no Pylance/Pyright CLI check is configured/installed in this environment,
and Pylance/PyCharm GUI completion has not been verified.
