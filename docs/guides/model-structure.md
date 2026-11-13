# Native model structure capture — 3B.4D.1

`capture_model_structure(model)` reads registered metadata from an **already
constructed, trusted** `torch.nn.Module`. It does not import a model factory,
load a checkpoint, run forward/backward, train, change modes, copy the model,
register hooks, call `state_dict`, read tensor values or move tensors/devices.
This is structural evidence collection; no architecture audit, optimization or
two-snapshot comparison engine is implemented.

```python
from hyso import capture_model_structure, ModelStructureSnapshot

# model is an existing trusted, quiescent nn.Module supplied by the caller.
snapshot = capture_model_structure(model)
print(snapshot.summary.parameter_elements.value)
print(snapshot.summary.trainable_elements.value)
print(snapshot.summary.unique_modules)
for module in snapshot.modules:
    print(module.primary_path, module.qualified_type, module.category,
          module.direct_parameter_tensor_count)
snapshot.to_json("model-structure.json")  # Exclusive creation, no overwrite.
reopened = ModelStructureSnapshot.from_json("model-structure.json")
assert reopened.content_sha256 == snapshot.content_sha256
```

The public API is also available from `hyso.structure`, including frozen typed
views for modules, tensors, paths, quantities, configuration fields, dtype/type
counts and storage groups. Returned collections are tuples; nested configuration
sequences are immutable. `to_dict()` returns a detached plain JSON projection.
`to_json()` without a path returns deterministic canonical UTF-8 JSON plus one
newline. No timestamp, UUID, pointer or tensor content enters the snapshot.

## Format and provenance

The standalone format is `hyso.model_structure`, schema **1**. Producer method
`registered_nn_module_metadata`, producer version **1**, classification version
**1**, and actual `torch_version` are recorded. `source.kind` is
`existing_trusted_torch_nn_module`. The `source.fields` table identifies the
sources of registration, configuration, tensor metadata, persistence, alias and
summary facts. This API accepts no caller parameter/layer counts or metadata
claims to fill those facts.

`content_sha256` is SHA-256 of canonical JSON excluding that field, using the
existing HYSo evidence codec. It covers provenance, modules, tensors, paths,
derived summary and limitations. It is a **structure snapshot hash**: it is
neither the historical P26 model-state hash nor a checkpoint artifact SHA.
Weight/buffer values and training/eval flags are deliberately absent. Different
weights or behavior can share a structure hash. Different Torch versions can
produce different hashes because provenance is included.

The reader checks schema/version, exact fields, native tensor shape/dtype/byte
consistency, references, expanded paths, derived counts and the canonical hash.
Unknown versions and incoherent records fail. This is internal consistency
validation, not a signature, live-object certification or authenticity proof for
JSON obtained from an untrusted producer.

## Counting and aliases

| Field | Definition |
|---|---|
| `unique_modules` | Unique registered Module objects, including the root |
| `registered_module_paths` | Expanded registered paths to actual modules, including root `""`; shared subtrees can appear at multiple paths |
| `unique_parameter_tensors` | Unique parameter-registry tensor objects; tied references do not increase this count |
| `parameter_elements` | Sum of elements of unique supported Parameter objects |
| `trainable_elements` / `frozen_elements` | The same object accounting partitioned by observed `requires_grad`; no training behavior is inferred |
| `unique_buffer_tensors` | Unique buffer-registry tensor objects; repeated references count once |
| `parameter_logical_bytes` / `buffer_logical_bytes` | Per-category unique-object element count times dtype element size |
| `tensor_union_logical_bytes` | Union of parameter and buffer objects, deduplicating cross-category object aliases |

Module IDs, tensor IDs and storage IDs are deterministic local ordinal labels,
assigned on first registration traversal. Module discovery/path traversal is
breadth-first and preserves each registry's insertion order. Registration order
is part of static identity; it is not a claim about runtime execution order.
`ModuleStructure.children` retains named edges, including registered `None`
slots. `paths` records parent, depth, target module ID, shared occurrence and
cycle closure. Cyclic edges are retained once per expanded path and are not
recursively followed into infinity. Registered `None` slots do not count as
actual modules/tensors.

Each unique module exposes its direct parameter/buffer slots and deduplicated
local counts. Those local counts **must not be summed** to infer global counts
when different modules tie parameters; the summary deduplicates globally.
Buffer persistence belongs to the owner's registration slot: the same buffer
can be persistent at one owner and nonpersistent at another.

Distinct Parameter objects that share underlying storage remain distinct
Parameter objects. Dense storage handle equality is separately mapped to local
storage groups, with tensor stride and storage offset. Raw handles, memory
addresses and tensor values are never exported. These groups do not certify
overlap/disjointness of byte ranges, storage allocation size or memory savings.
Independent empty storages are distinguished even when both have data pointer
zero. Unavailable storage identity has explicit missing/unsupported status.

Logical tensor bytes are not GPU allocation, RSS, physical unique-storage
bytes, gradients/optimizer/activation memory or serialized checkpoint size.

## Static categories and configuration

Classification v1 uses **exact class identity**, not class-name prefixes,
`isinstance` inheritance assumptions, model `repr()` or caller strings. It covers:

- Containers: Module, Sequential, ModuleList/Dict, ParameterList/Dict.
- Separate Conv1d/2d/3d and ConvTranspose1d/2d/3d categories; Linear.
- Batch/Instance/Layer/Group/SyncBatch and local-response normalization.
- RNN/LSTM/GRU and recurrent cells; MultiheadAttention; registered Transformer
  encoder/decoder/layer types; Embedding/EmbeddingBag.
- Common registered activation, max/average/adaptive pooling, dropout, reshape
  and Identity types.

Type and category distributions expose both unique-module and expanded-path
counts. Conv/Linear static fields include channels/features, kernels, strides,
padding, dilation/groups and registered bias presence as appropriate. Other
allowlisted fields include normalization configuration, recurrent dimensions,
attention dimensions/head counts, embedding and pooling configuration.

Only stored, bounded, plain fields are admitted. A missing stored field is
`missing`; a callable, object or unsupported value is `unsupported` with no
`repr()`/callable-name replacement. Transformer activation callables therefore
remain unsupported as configuration definitions. Registered activation child
modules, when present, are still visible normally.

Unknown/custom classes and subclasses of supported standard classes retain
their actual qualified Python type and all registered children/tensors, while
their internal configuration/semantics remain unsupported. A YOLO-like custom
network can expose registered Conv/normalization/activation components without
claiming to audit its detector topology. Functional calls, dynamic control flow
and unregistered Python objects may be absent. Registered module counts are
never labeled as an executed computational-layer count or a forward graph.

## Tensor support and partial evidence

| Kind | Metadata and counting |
|---|---|
| Exact ordinary initialized Tensor/Parameter, dense strided | Shape, dtype, device, requires_grad, logical elements/bytes; observed storage grouping when available |
| Meta Tensor/Parameter | Shape/dtype-based logical counts native; materialized storage identity missing |
| Lazy/uninitialized Parameter/Buffer | Object identity and safe dtype/device metadata retained; shape/counts unsupported, no materialization |
| Sparse/non-strided, quantized, nested | Type and safely available metadata retained; element/byte totals unsupported |
| Custom Tensor/Parameter subclass | Qualified type and registration retained; metadata/counts unsupported; subclass dispatch is not invoked |
| Unknown dtype encoding | Unsupported rather than guessed element size |

Native dtype byte sizes are explicitly allowlisted for ordinary boolean,
integer, floating-point and complex encodings. `StructureQuantity` exposes
`status`, `value`, `known_subtotal`, and `unsupported_tensor_ids`. If a required
tensor is unsupported, the total `value` is null, not a partial total masquerading
as complete. Known supported elements remain separately available as a subtotal.
Unique registered object counts are still known. Storage alias coverage is
separate from logical-count coverage.

Implementation relies on ordinary PyTorch registration dictionaries and native
Tensor metadata access. Storage grouping uses `untyped_storage()._cdata` equality
as a private, version-sensitive metadata feature, never `data_ptr()` or content
reads. If grouping is unavailable, it remains missing without erasing reliable
logical counts. Actual validation here uses **PyTorch 2.5.1+cu121 on CPU**; no
GPU/native CUDA acceptance or cross-version equivalence is claimed. Runtime
Torch version is always included in the snapshot.

## Safety, limits and future evidence links

The producer bypasses custom `__dict__` and normal attribute lookup for module
registries, and never calls model iteration overrides, state/forward hooks or
tensor-subclass methods. Only exact supported tensor types have their native
metadata read. It performs a second metadata check to reject observed registry,
configuration or tensor-metadata changes during capture. This is not an atomic
concurrency protocol or a sandbox: callers must supply trusted, quiescent
objects and prevent concurrent mutation. No parameter/buffer values are read,
cloned, detached, hashed or serialized.

Default hard ceilings are 10,000 unique modules, 50,000 expanded module paths,
100,000 unique tensors, depth 128, 500,000 registration slots and 16 MiB canonical
JSON. Matching keyword arguments may lower these ceilings. Invalid limits,
malformed registries, detected mutations and exceeded limits raise
`ModelStructureError`; the producer never returns a silently truncated profile.
Shared-path expansion is bounded independently of unique module count.

P26 schemas 1/2, P11, Advisor contracts and existing canonical reports are
unchanged. A standalone structure snapshot is not automatically imported as a
P11/P26 measurement or joined by a matching caller model name.
[3B.4E.1](p26-structure-binding.md) explicitly captures structure on the actual
isolated P26 forward receiver at pre/post evaluation boundaries and binds it
inside new P26 schema 3. This does not certify original/copy architecture or
behavior equivalence. Other measurement bridges still require their own actual
producer observations and hashed references; matching caller strings are
insufficient. This standalone capture API creates no automatic bridge. The separate
[3B.4D.2 snapshot comparison](model-structure-comparison.md) compares only the
recorded structures and does not add that bridge or runtime/architecture audits.

Focused tests: `tests/test_model_structure.py`. They prohibit forward/backward,
mode changes, value reads/copies, hooks, state_dict, deepcopy, device moves and
GPU/benchmark/training APIs during capture, while exercising CNN, recurrent,
attention/Transformer, custom detector-like, shared/tied and special-tensor
fixtures, canonical roundtrips and failure limits.
