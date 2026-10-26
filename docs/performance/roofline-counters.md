# Saved native counters and empirical Roofline references

HYSo can compare a saved FP32 kernel measurement with separate empirical compute
sessions and a saved device-DRAM bandwidth session. This is optional, offline
analysis. It never launches a profiler, benchmark, CUDA kernel, driver query or
calibration. Normal HYSo installation and the existing analytic Roofline API do
not require Nsight Compute.

## Inputs

`read_ncu_counters` reads an existing Nsight Compute **raw CSV in base units**
and the linked workload receipt. For an already saved `.ncu-rep` / `.ncu-repz`,
the report owner can export CSV with `ncu --import ... --page raw --csv
--print-units base`. That command reads a saved report; it is not a new capture.
The HYSo adapter itself does not run this command or import NVIDIA's Python API.

Required counter columns:

- `gpu__time_duration.sum` in `ns`;
- `dram__bytes_read.sum`, `dram__bytes_write.sum` in `byte`;
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`, `...fmul...`,
  `...ffma...` in `inst`;
- `profiler__replayer_passes` in `pass`.

The adapter also reads saved SM/DRAM frequency columns in `hz` when available.
CSV must retain kernel ID/name, PID, context/stream, device/name/CC/PCI fields
and NVTX push/pop attribution. Supply the **exact** kernel name, not a fuzzy
match. Missing counters, wrong units, duplicates, incomplete receipts and
attribution mismatches raise `ValueError`; payload or allocator bytes are never
substituted.

The receipt must contain `pid` (or `worker_pid`), `device`, `gpu`, `nvtx_range`,
explicit `dtype="float32"` (or `precision="FP32_SIMT_no_Tensor_Core_no_TF32"`)
and correctness/completion evidence. Multiple samples use `targeted_launches`
and `samples` with sequential `index`, `output_equal=true`,
`terminal_completed=true`. A single target can instead use top-level
`output_equal=true`, `terminal_completed=true` and `targeted_additions=1`.
Recorded capability, PCI and kernel fields must agree when supplied.

The producer must actually use FP32 SIMT; a dtype label alone cannot exclude
Tensor Core/TF32 execution. Inspect the original native report/SASS and the
reference workload when qualifying external evidence. The adapter checks the
recorded protocol; it cannot authenticate arbitrary caller-created CSV/receipts.

Choose `number_format="decimal_point"` or `"decimal_comma"` explicitly. For
example, `3.936` in a decimal-comma export means 3936; it is not 3.936 ns.
Units are never inferred from display magnitudes.

## Example

Paths below are local saved exports/receipts, not bundled recordings:

```python
from hyso.performance import (
    read_ncu_counters, analyze_counter_roofline, plot_counter_roofline,
    RooflineResult,
)

target = read_ncu_counters(
    "private/target-raw.csv", receipt_path="private/target-receipt.json",
    session_id="target", role="target", kernel_name="EXACT RECORDED KERNEL",
    number_format="decimal_comma",
)
compute_a = read_ncu_counters(
    "private/compute-a-raw.csv", receipt_path="private/compute-a-receipt.json",
    session_id="compute-a", role="compute", kernel_name="my_fp32_fma_kernel",
    number_format="decimal_comma",
)
compute_b = read_ncu_counters(
    "private/compute-b-raw.csv", receipt_path="private/compute-b-receipt.json",
    session_id="compute-b", role="compute", kernel_name="my_fp32_fma_kernel",
    number_format="decimal_comma",
)
memory = read_ncu_counters(
    "private/dram-raw.csv", receipt_path="private/dram-receipt.json",
    session_id="dram", role="memory", kernel_name="my_streaming_copy_kernel",
    number_format="decimal_comma",
)
result = analyze_counter_roofline(
    target, compute_sessions=[compute_a, compute_b], memory_session=memory,
)
result.to_json("private/roofline.json")
result.to_csv("private/roofline.csv")
assert RooflineResult.from_csv("private/roofline.csv").to_dict() == result.to_dict()
plot_counter_roofline(result, "private/roofline.png")
```

Optional `source_paths={"worker": ..., "cuda": ..., "ptx": ...}` verifies files
against receipt checksums. Optional `sensor_paths={"before": ..., "after": ...}`
reads saved `nvidia-smi -q -x` XML, checking GPU/PCI and retaining clock,
temperature, power, P-state and reason snapshots. These are not live queries or
a continuous timeline. A `report_path` records the binary report's name/SHA256
only; it does not independently prove the CSV was exported from that binary.
Keep the original report and its external native verification.

All supplied files are read-only. Exports and plots refuse to overwrite by
default. JSON/CSV may contain private paths, receipts and environment details;
keep them and the profiler artifacts outside version control.

## Interpretation

Hardware-equivalent FP32 work is `FADD + FMUL + 2*FFMA` using **pred_on thread
instructions**. FMA counts twice. A useful analytic add can still execute as an
FFMA with a runtime multiplier of one. Analytic useful work remains separately
labelled and is not substituted in the native arithmetic intensity.

The target uses work, duration and DRAM read+write from the **same saved kernel
interval**. Compute and memory references must use the same device, FP32 work
definition and DRAM hierarchy. PCIe bandwidth, parameter size, RSS and allocator
bytes cannot stand in for DRAM traffic. Large working-set memory references and
compute-dominated SIMT reference workloads still require external qualification;
the reader cannot prove saturation from a kernel name or declared role.

Every compute session retains every sample, its own min/median/max, descriptive
range/median, clocks and sensor conditions. Sessions are never pooled or reduced
to a device-wide maximum. The DRAM reference retains all samples and labels its
maximum as **maximum observed in that session**. These are workload- and
condition-dependent empirical references, not exact theoretical ceilings,
confidence intervals or guaranteed sustainable capacities.

The graph shows one model per compute session. Its line uses that session's
compute median and the DRAM session's observed maximum. The band spans the
observed compute and memory ranges. A `DRAM_reference` branch means the DRAM
term is the smaller term in this **reference model**; it does not establish that
the real kernel is memory-bound. Model geometry is not speedup evidence.

`bottleneck`, `hardware_headroom`, `theoretical_maximum` and `speedup` remain
`unsupported`/null. Unknown target clocks, replay differences, shared DRAM
traffic and incomplete output writeback prevent stronger causal claims. A
snapshot or short run does not establish all-time thermal/power steady state.

There are no mandatory 10% spread/clock thresholds, 90–110% traffic ratios,
20 microsecond minimum durations, RAM/VRAM quotas or additional benchmark runs.
Such checks can be useful experiment-specific quality indicators. They are not
hardware limits or automatic product acceptance rules. Short/variable samples
retain their uncertainty; missing or invalid evidence is never turned into a
measurement.

## Compatibility

`analyze_roofline` and `measure_roofline` keep their analytic work/payload
semantics. They do not silently ingest native counters or promote theoretical
references to measured roofs. The optional native path returns the same
immutable `RooflineResult`, schema version 1 and canonical JSON/CSV envelope,
with protocol ID `ncu_empirical_roofline`. Both old and new protocols validate
their derived fields from retained evidence when reopening an export.
