# Graph checkpoint contract evidence v2

Graph checkpoint evidence v2 is a separate, opt-in process-local proof. It does
not upgrade historical P26 v1 evaluation records or change Graph state_dict keys.

```python
from hyso.checkpoint import (
    save_training_checkpoint, load_training_checkpoint, graph_checkpoint_evidence,
)

# source and target are caller-created matching GraphModel/GraphTensorOutput objects.
save_training_checkpoint(path, source, graph_contract=True, include_rng=False)
load_training_checkpoint(path, target, capture_evidence=True)
receipt = graph_checkpoint_evidence(target)
assert receipt["status"] == "native"
```

Default saves remain checkpoint format_version 1. `graph_contract=True` explicitly
creates format_version 2; old readers reject it. No existing file is converted.
Load without evidence capture restores supported v2 state but supplies no receipt.
Legacy Graph checkpoints remain loadable and have no new contract commitment.

An optional `expected_sha256` argument checks the same private snapshot before
weights-only deserialization. Obtain that expected value through a separately
trusted channel when authenticity against a particular artifact matters. A hash
provided alongside an attacker-controlled artifact does not prove its authorship.
The proof binds the loaded snapshot bytes to current model state and validated
runtime metadata; it is not a signature or a training-provenance certificate.

## Supported profile

Only exact GraphModel or exact GraphTensorOutput wrapping exact GraphModel is
supported, on dense little-endian CPU fp32/fp64. All graph inputs/outputs,
connections, node parameters and adapter output selection must match. Reference
reconstruction preserves CPU RNG and is admitted only with a known conservative
parameter estimate of at most 1,000,000. No forward or data loading occurs while
capturing/querying the proof. Parameterless graphs are supported.

The initial closed component profile is: `conv_norm_act`, `conv2d`,
`pointwise_conv2d`, `identity`, `relu`, `silu`, `batch_norm`,
`global_avg_pool2d`, `flatten`, `classification_head`, `add`, `multiply`,
`concat`, `weighted_sum`. Other components are unsupported for v2; their ordinary
v1 save/load behavior is unchanged. Unsupported opt-in saves/loads raise
TrainingCheckpointError rather than producing a partial trusted checkpoint.

Fresh closed constructors are compared recursively with the live model. Exact
module types, registered order/topology, plain behavioral attributes, parameter
and buffer schemas, dtype/device and aliasing must agree. Subclasses, extra
instance state/callables, local/global hooks, unknown nonpersistent buffers and
modified runtime structures are rejected. The only reconstructed nonpersistent
state is the known empty Graph `_contract` dtype/device anchor.

The serialized contract commits the exact state_dict fingerprints separately
from configuration, selected adapter output and the reconstructed runtime
signature. Parameters and persistent buffers must match the frozen checkpoint
fingerprints after strict load. Their values are not substituted by fresh random
reference weights. The contract records the Torch version. The runtime signature
is a typed profile check under the trusted installed implementation, not a proof
against malicious interpreter/global class-code modification.

## Receipt lifecycle and legacy scope

The receipt is owned by a weak reference to the exact loaded object and is bound
to the actual snapshot SHA, current full P26 named-tensor state hash and runtime
contract. Queries revalidate state and contract. Mutation of tensors, topology,
activation, output selection or hooks yields missing. A deepcopy/other model has
no receipt. Reload without capture clears it; failed mutation/rollback never
mints one. Validation failures before mutation retain an earlier valid receipt
only for its unchanged earlier snapshot/state, as in existing checkpoint policy.
Returning a receipt dict cannot mutate the internal frozen receipt.

Train/eval flags, gradients, optimizer behavior, dataset origin, checkpoint
authorship and arbitrary Python execution are outside this receipt's scope.
Ordinary mode changes alone do not change the architecture/state commitment.
The existing explicit loader remains responsible for optimizer/scheduler/RNG
restore; there is no second checkpoint engine.

`evaluate_classification(..., capture_evidence=True)` still emits the unchanged
schema2/3 P26 v1 producer evidence. Its Graph checkpoint field remains missing.
The separate `hyso.graph_checkpoint_evidence` schema_version 2 record must be
retained and interpreted with its explicit scope. It is not silently projected
into old EvidenceRecord checkpoint bindings; consumer projection is not added.
Imported JSON is a statement about a prior capture, not a live trusted receipt.
No CUDA proof or cross-process receipt is provided.

Run a fresh short CPU acceptance from the checkout with:

```powershell
python -B -m examples.models.graph_checkpoint_evidence --output C:/local-qa/fresh-graph-v2
```

That example uses explicit synthetic inputs for one training epoch, strict
snapshot load, equal outputs, a native v2 receipt, separate legacy P26 missing
evidence and mutation invalidation. It makes no real-dataset quality claim.
