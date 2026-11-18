# Stage 13.5.16 performance usability backlog

These observations come from the fixed CPU acceptance and source inspection.
They propose follow-up work. Runtime tuning and model changes have zero scope
in this stage. [Acceptance report](stage13-5-16-cpu-cuda-performance-acceptance.md)

| Area | Observed evidence | Proposed follow-up | Current boundary |
| --- | --- | --- | --- |
| Reporting cost | 16-record expert creation plus JSON/MD/HTML export p50 3500.2627 and 3693.9038 ms, n=3 per child | Profile validation, canonical JSON and renderer stages under a new bounded protocol | Component-level cause remains a hypothesis |
| Protocol identity | External QA protocol SHA conflicts with native protocol_sha256 declaration in the initial setup | Document the external-run association recipe alongside native protocol identity | Runtime correctly rejects the conflicting native scope |
| Evaluation loader | A list of tensor batches fails the standard DataLoader contract | Add a short synthetic TensorDataset/DataLoader example near the signature | Final QA uses standard DataLoader, same sample geometry |
| Timing interpretation | Benchmark serial service rate and completion-window throughput have different inclusion scopes | Put timing-scope descriptions next to both public examples | Both measured series remain available |
| Tail statistics | Built-in BenchmarkResult exposes descriptive p95 for n=120, strict TailPolicy returns insufficient_samples | Show descriptive quantiles and tail eligibility together | Effective sample size remains unestablished |
| Memory interpretation | Sampled RSS maximum, Windows lifetime HWM, logical bytes and checkpoint length differ in scope | Use an explicit memory-domain table in user guidance | Model-exclusive interval peak remains NOT_MEASURED |
| Same-model regression | General compare_results permits architecture/state differences | Document the extra state-identity constraint for same-model use | EvidenceScope state mismatch guard was tested |
| CUDA admission | PID 2000 has restricted identity and differs between captured compute and C/G inventories | Resolve current workload identity in a separately reviewed fresh admission | H BLOCKED, I NOT_TESTED, zero CUDA worker launches |
| Historical diagnostics | The 2026-10-22 raw diagnostic directory is unavailable at its retained locator | Repair the evidence locator through an authorized evidence-maintenance task | Versioned diagnostic report remains a historical observation |
| Workflow metrics | Custom train-step and IO measurements use QA supplements | Document which result types have native typed adapters and which use generic artifacts | Existing Reporting/ExperimentManager engines are retained |
| CLI discoverability | Performance features are reached through analysis list/show/plan/run | Add a cross-reference from performance documentation to actual CLI help | Both package installations pass existing help routes |

## Reproduction scope

The committed example supplies prepare/run entry points over existing HYSo APIs.
Use a fresh external QA root, retain protocol.json and its source lock, and run
each CPU child under a bounded external controller. The original controller,
environment, commands and receipts are archived in this stage's QA directory.
The selected regression is recorded in QA/regression-selection.json.

The architecture shapes are QA fixture choices derived from the sequential and
branching CNN types accepted in 13.5.13. Seed 13516, channels 1→4→8 and the
3×40 repetition schedule were selected before measurement. They supply a small
finite acceptance workload. User-model parameter changes, performance winner
selection, driver changes and optimization commits have zero execution.

## Open acceptance scope

CPU has verified measurements and preserved source/data hashes. CUDA remains
BLOCKED_SAFETY. The B4 native root cause remains unresolved. The next stage has
zero execution. This backlog adds no authorization to start historical programs
or native diagnostics.
