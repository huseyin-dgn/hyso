# P12: versioned logical FLOPs/MACs acceptance

P12 is IMPLEMENTED within the observed dense inference reference scope below.
This is not universal operator coverage, a GPU instruction count, timing estimate,
or a claim that every model result is complete. Unknown calls always make total
FLOPs/MACs null. This decision concerns the original roadmap's analytic references,
explicit counting convention and visible unsupported coverage.

## Existing implementation and observed gaps

The base is `1ab52822c39c073eebca1d8c1d773e56c47cb728`. It already contains
`bc269eb`, including the v2 validated operand/scalar models, inference BatchNorm
and reductions. The earlier B6 audit predates that implementation; its description
of all normalization as unmodeled is historical. No old result is rewritten.

Five distinct saved dispatch ledgers (deduplicated by their full call metadata)
contained 31 calls and no v2 unknowns. They are a narrow sample, not proof of model
coverage. A short CPU profiler/dispatch inventory therefore examined sixteen tiny
existing/representative fixtures: Norm, Shared, Matrix, LayerNorm, GroupNorm,
Softmax, LogSoftmax, GELU, grouped/depthwise convolution, transposed convolution
and ModularCNN with none/SE/CBAM/ECA/coordinate attention. Each was observed once
in FP32 and once in FP64, separately from construction and state preparation.

Within each dtype's 134-call inventory, v2 unknown frequencies were:

| Observed ATen call | Calls |
| --- | ---: |
| `aten.silu.default` | 19 |
| `aten.sigmoid.default` | 6 |
| `aten.layer_norm.default` | 1 |
| `aten.group_norm.default` | 1 |
| `aten.softmax.int` | 1 |
| `aten.log_softmax.int` | 1 |
| `aten.gelu.default` | 1 |
| `aten.conv_transpose2d.input` | 1 |
| `aten.adaptive_max_pool2d.default` | 1 |
| `aten.max.dim` | 1 |
| `aten.hardswish.default` | 1 |

These are fixture frequencies, not estimates of application-wide frequency or
fractions of compute work. All 134 observed calls per dtype have modeled v3
operation vectors. This includes comparison-only and named special operations;
it does not equate special operations to zero work.

## Counting domains and compatibility

The public `analyze_flops` signature and result schema_version=1 are unchanged.
New results explicitly select operation_model_version=3. Previously exported v1
(including an absent model version) and v2 results retain their original convention
and derivation. JSON reopening validates each version; CSV columns stay unchanged,
with additional named-operation fields inside the existing JSON payload column.
Existing saved files must not be relabeled or migrated silently.

In v3:

- `flops`: ordinary additions, subtractions, multiplications and divisions;
  dense contraction FMA remains one MAC / two FLOPs. Bias arithmetic is separate.
- `comparisons`: separate counts for ReLU/max/hardswish. MACs do not absorb
  normalization affine multiplication/addition pairs.
- `root_ops`: named sqrt/rsqrt counts, never assigned a FLOP weight.
- `transcendental_ops`: named exp/log/tanh/erf counts, never assigned a FLOP weight.

Thus a complete arithmetic total is not a weighted total of all numerical work.
`status=complete` means every observed call's declared reference vector is modeled;
it does not establish complete hardware activity or support for unexecuted paths.
Special-operation totals become null if any observed call is unknown; supported
subtotals remain available. `comparisons_per_batch` retains its existing supported
subtotal semantics; the new `total_comparisons_per_batch` requires complete coverage.

BatchNorm v2 deliberately counted sqrt as one logical scalar operation. V3 moves
those operations into `root_ops`, so raw FLOP totals across model versions are not
directly comparable. This is a versioned domain clarification, not a silent rewrite
of historical measurements.

## Reference formulas

All formulas describe a declared unfused algebra, not the implementation's fusion,
instruction mix, precision promotion, branch optimization or latency.

Let N be the output element count, C channels, G normalization groups and K the
elements per reduction group. Affine flags w/b are 0 or 1.

| Operation | Ordinary FLOPs | Other domain |
| --- | --- | --- |
| Dense Conv/Linear/matrix contraction | 2 MAC + explicit bias/scalar work | MAC convention retained |
| Inference BatchNorm | C + N(2+w+b) | C sqrt |
| LayerNorm/GroupNorm population reference | N(6+w+b)+G | G rsqrt |
| Stable softmax | 2N+G(K−1) | N exp, G(K−1) comparisons |
| Stable log-softmax | 2N+G(K−1) | N exp, G log, G(K−1) comparisons |
| Sigmoid | 3N | N exp |
| SiLU | 4N | N exp |
| GELU erf / tanh approximation | 4N / 8N | N erf / N tanh |
| Hardswish | 3N | 2N comparisons |
| Max reduction/adaptive max pool | 0 | Sum of exact window sizes minus one per window |

Layer/GroupNorm use mean K, population variance 3K, one epsilon addition,
separate reciprocal square root and output centering/scaling 2K, before affine.
This explicitly recomputes centering in the reference expression. GELU constants
are precomputed; tanh approximation uses two multiplies for the cubic. Standalone
exp/log/tanh have zero *ordinary arithmetic* FLOPs and nonzero named counts.

Transposed convolution supports grouped 1D/2D/3D references. For each spatial axis,
count input/kernel pairs whose `input*stride-padding+kernel*dilation` lies inside
the recorded output. Multiply the axis counts by batch, input channels and output
channels per group. This counts clipped scatter contributions without charging
stride-inserted zeros; output padding and exact output shape are validated. It is
not the ordinary convolution's dense padded-output convention or a kernel counter.

V2 operand/scalar checks remain in use for existing operations. In-place output
expansion is rejected in v3. Forward and batch normalization must be positive
integers (not bools), including when reopening a supported historical version.
Alpha/beta counts describe the dense reference, including alpha=0; they do not
claim the backend executed or skipped a specific instruction.

## Unsupported and safety boundaries

Training BatchNorm statistics/state updates, arbitrary custom/fused operators,
unmodeled transcendental functions (for example sin), sparse/complex arithmetic,
mixed-dtype normalization, unsupported cast/softmax modes, padded/ceil fixed-pool
policies, ordinary Conv3D and backward graphs remain unknown where no validated
model applies. Transposed 1D/3D are independently checked arithmetic references;
the native transposed-convolution fixture in this review is 2D.
Unknowns are never replaced by zero and cannot produce a complete model total.
No all-activation, all-backend, training or physical GPU instruction claim is made.

Only outer observed ATen dispatch calls contribute counts; nested CPU profiler
events are not added again. The existing shared-module test checks three actual
invocations plus residual arithmetic, rather than multiplying by aliases.

## Acceptance and verification

- Final targeted arithmetic/scope suite: **116 passed**, 15.00 seconds.
- Selected real CPU profiler/regression suite: **16 passed, 37 deselected**,
  131.16 seconds. These are separate test sets, **132 passing tests** in total;
  this is not a full-suite or GPU acceptance claim.
- Independent small-shape references cover grouped/depthwise Conv, scalar and
  broadcast matrix operations, reductions, normalization and activation vectors.
- Transposed convolution uses an independent brute-force input/kernel-pair
  enumerator with clipping, groups, stride, dilation and output padding.
- Native CPU profiler/dispatch checks cover FP32/FP64 and actual model paths.
- Unknown/custom calls, invalid normalization, metadata, v1/v2/v3 reopening,
  symbolic-count tampering and unchanged CSV columns are verified.
- A pre-existing Matrix fixture created FP32 bias even for FP64 input; the test
  fixture now creates bias in the input dtype. Production model behavior is unchanged.

Raw inventories, test XML and temporary results remain outside the source tree.
No GPU workload, dependency installation, commit or push is part of this review.
P31's two-hour acceptance remains deferred. No P29 work is included.
