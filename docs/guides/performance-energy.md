# P33 sampled power and energy

`analyze_energy(P31_result, max_gap_seconds=..., idle_max_age_seconds=...)`
analyzes recorded evidence. It does not start inference, call a sensor, initialize
CUDA or change clocks/power limits. Live acquisition remains the existing
explicit P31 NVIDIA adapter. CPU/RAPL, external meters and cumulative energy
counters are unsupported here; their Joules, units, wrap/reset corrections and
per-inference energy are never guessed.

The NVIDIA adapter explicitly requests `power.draw.instant`, in watts, for one
physical UUID. NVIDIA describes it as the last measured entire-board reading;
it is not the laptop/system input power or model-only power. See the
[NVIDIA SMI documentation](https://docs.nvidia.com/deploy/nvidia-smi/).
Host query start/end and midpoint use the worker's monotonic clock. The actual
sensor refresh time and delay are unknown; host query times are associations,
not sensor hardware timestamps. No averaged-power fallback is silently selected.

New results carry `evidence.energy_version=2`; the outer runtime envelope remains
v1. Legacy P33 analysis and flattened CSV export remain unchanged for existing
envelopes. New P33 CSV uses the existing lossless canonical envelope and supports
`RuntimeResult.from_csv`, including nulls, empty collections and raw source data.

The v2 analysis validates source/clock/unit/UUID and available PCI/device/sensor
identity, successful call/image counters and completion markers. One forward
completes the recorded batch; `J/forward` and `J/image` have different denominators.
The source must be physical-GPU `model_inference` with image accounting. A trained
CPU model's latency/utilization record does not supply Joules.

Power samples are integrated with a piecewise-linear trapezoid, clipped to the
exact active window. Full-window gross energy needs bracketing endpoints, valid
samples and gaps no larger than the declared limit. Missing samples and long gaps
create explicit uncovered intervals; they are not bridged. `covered_energy`
can report Joules over the supported segments, but full gross and per-work ratios
stay null when the entire selected energy window is not covered.

A completed source uses its exact terminal work window. An interrupted source
can qualify only a verified committed prefix: the first work observation must
attest zero completed calls/images; the end is the last successful completion
marker, and every counter transition must agree with the batch and markers.
Its ratios cover that prefix only, never an unknown tail, controller wall time or
the requested but unfinished duration. Successful-prefix energy does not complete
P31 acceptance or establish the energy of an entire aborted run.

Gross energy is independent of idle availability. Net-above-idle energy requires
at least two comparable idle-phase samples from the same sensor/domain/clock,
time-weighted integration, a recent baseline and no overlap with active work.
Missing/stale idle data leaves net energy null while gross can remain available.
Negative subtraction is preserved. Even with a baseline, the board's background
and other applications are uncontrolled; the difference is not model-exclusive
energy or an energy-saving proof.

`measurements` provides each metric's name, unit, value, measured/derived kind,
status/reason, device/domain, clock/window, sensor and raw-evidence references,
completed-work denominator usage, method and uncertainty reference.
`measurement_context`, `coverage`, `idle_baseline`, `sampling` and
`uncertainty_details` retain the shared provenance and limitations.

Observed sample min/max/arithmetic mean power are summaries of the recorded
trace, explicitly including idle and bracketing readings. The active
time-weighted mean is separately derived as full gross Joules / active seconds.
Recorded extrema are not continuous extrema. `direct_counter_energy` remains
null/unsupported. No TDP, utilization or nominal power is substituted for a sensor.

The old adapter's constant `sensor_accuracy_watts=5` was not returned by the
query or validated for the recorded device. New acquisition records accuracy as
unknown with a reason. V2 keeps historical declared values as unverified metadata,
not a sensor-spec energy bound. NVIDIA's documented accuracy statements vary by
API/device generation; the old constant is not a calibrated universal 5 W bound.
See [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
No total uncertainty bound or confidence interval is produced: refresh latency,
between-sample variation and uncalibrated measurement overhead remain unknown.

The [P33 audit](../../outputs/p33-energy-analysis-20261030/final-report.md)
separates constructed arithmetic traces, archived native GPU fixture samples and
the trained Model B CPU record. Native energy for trained A/B and real idle/net
validation remain deferred; no new GPU or long experiment is authorized by replay.
