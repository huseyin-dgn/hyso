"""Optional, offline Nsight raw-CSV evidence; no profiler or CUDA execution.

Empirical session observations are not hardware limits or confidence intervals.
The existing analytic Roofline API and its schema remain unchanged.
"""
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

from .inference import _base
from .result import RooflineResult, _integer, _positive
from .roofline import _hash, _math

_PROTOCOL = dict(id='ncu_empirical_roofline', version=1, scope='saved_native_kernel_intervals', configuration={})
_UNITS = {
    'gpu__time_duration.sum': 'ns', 'dram__bytes_read.sum': 'byte', 'dram__bytes_write.sum': 'byte',
    'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 'inst',
    'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 'inst',
    'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 'inst', 'profiler__replayer_passes': 'pass',
}
_CLOCKS = ('sm__cycles_elapsed.avg.per_second', 'dram__cycles_elapsed.avg.per_second')
_PCI = ('device__attribute_pci_domain_id', 'device__attribute_pci_bus_id', 'device__attribute_pci_device_id')
_LIMITATIONS = [
    'Saved counter exports and linked receipts only; no GPU execution or profiler dependency.',
    'FP32 SIMT work = FADD + FMUL + 2*FFMA pred_on thread instructions, not analytic useful work.',
    'DRAM counters describe the profiled interval, not tensor payload, PCIe or all eventual output writeback.',
    'Shared DRAM may include other GPU engines; NVTX does not prove exclusive traffic attribution.',
    'Replay passes may collect metrics at different times; profiler/cache/clock conditions remain attached.',
    'Each compute session remains separate. Observed ranges are not confidence intervals or capacity bounds.',
    'Empirical maxima are observations in these conditions, not theoretical or guaranteed sustainable limits.',
    'Model branch is a reference-model property, not a diagnosis of the actual kernel.',
    'No proven hardware headroom, speedup or causal bottleneck follows from these observations.',
    'Spread, duration and payload ratios are descriptive quality indicators, not hardware admission limits.',
]


def _file(path):
    path = Path(path)
    return dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _number(text, comma):
    # The caller chooses the export locale; never guess whether 3.936 means 3936.
    group, decimal = ('.', ',') if comma else (',', '.')
    pattern = rf'[+]?(?:\d{{1,3}}(?:{re.escape(group)}\d{{3}})+|\d+)(?:{re.escape(decimal)}\d+)?(?:[eE][+-]?\d+)?'
    if not re.fullmatch(pattern, text.strip()):
        raise ValueError('invalid counter number or wrong explicit number_format')
    try:
        value = Decimal(text.strip().replace(group, '').replace(decimal, '.'))
        if not value.is_finite() or value < 0:
            raise ValueError('finite nonnegative counter required')
        return value
    except InvalidOperation as error:
        raise ValueError('invalid counter value') from error


def _identity(receipt):
    if not {'device', 'gpu', 'nvtx_range'} <= set(receipt):
        raise ValueError('receipt device/GPU/NVTX fields required')
    pid = receipt.get('pid', receipt.get('worker_pid'))
    _integer(pid, 1)
    if 'pid' in receipt and 'worker_pid' in receipt and receipt['pid'] != receipt['worker_pid']:
        raise ValueError('receipt PID mismatch')
    _integer(receipt['device'])
    if receipt.get('dtype', receipt.get('precision')) not in ('float32', 'FP32_SIMT_no_Tensor_Core_no_TF32'):
        raise ValueError('explicit FP32 SIMT precision evidence required')
    if any(type(receipt.get(k)) is not str or not receipt[k].strip() for k in ('nvtx_range', 'gpu')):
        raise ValueError('receipt GPU and NVTX required')
    if 'samples' in receipt:
        samples = receipt['samples']
        if not samples or receipt.get('targeted_launches') != len(samples):
            raise ValueError('receipt sample count mismatch')
        for i, sample in enumerate(samples):
            _integer(sample['index'])
            if sample['index'] != i or sample['output_equal'] is not True or sample['terminal_completed'] is not True:
                raise ValueError('incomplete or incorrect receipt sample')
        count = len(samples)
    else:
        if receipt.get('output_equal') is not True or receipt.get('terminal_completed') is not True:
            raise ValueError('correctness and terminal completion required')
        count = receipt.get('targeted_additions', 1)
        _integer(count, 1)
    return pid, count


def _source_expected(receipt):
    origin = receipt.get('original_compilation', receipt.get('compilation', {}))
    return dict(worker=receipt.get('script_sha256', receipt.get('worker_sha256', origin.get('worker_sha256'))),
                cuda=origin.get('source_sha256'), ptx=origin.get('ptx_sha256'))


def read_ncu_counters(csv_path, *, receipt_path, session_id, role, kernel_name,
                      number_format='decimal_point', source_paths=None, sensor_paths=None, report_path=None):
    """Read an existing ``--page raw --csv --print-units base`` export + receipt.

    No binary report decoding, Nsight import, subprocess, driver or GPU query.
    ``report_path`` retains a fingerprint only, not proof of CSV/binary equivalence.
    Optional source_paths maps worker/cuda/ptx to files checked against the receipt.
    sensor_paths maps phase names to saved nvidia-smi -q -x XML snapshots.
    """
    if number_format not in ('decimal_point', 'decimal_comma'):
        raise ValueError('choose decimal_point or decimal_comma explicitly')
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8-sig'))
    pid, count = _identity(receipt)
    lines = Path(csv_path).read_text(encoding='utf-8-sig').splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.startswith(('"ID",', 'ID,')) and next(csv.reader([line]))[:2] == ['ID', 'Process ID'])
    except StopIteration as error:
        raise ValueError('Nsight raw CSV header missing') from error
    rows = [row for row in csv.reader(io.StringIO('\n'.join(lines[start:]))) if row]
    if len(rows) != count + 2 or len(set(rows[0])) != len(rows[0]) or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('ambiguous, missing or extra raw CSV rows/columns')
    header, unit_row = rows[:2]
    required = {'ID', 'Process ID', 'Kernel Name', 'Context', 'Stream', 'Device', 'CC',
                'device__attribute_display_name', *_PCI, *_UNITS}
    if not required <= set(header):
        raise ValueError('missing raw identity/counter columns')
    units = dict(zip(header, unit_row))
    nvtx_columns = [name for name in header if name.startswith('thread Domain:Push/Pop_Range:')]
    if len(nvtx_columns) != 1:
        raise ValueError('recorded NVTX push/pop attribution required')
    samples, devices = [], []
    for values in rows[2:]:
        row = dict(zip(header, values))
        markers = [token.split(':')[:2] for token in re.findall(r'"([^"]+)"', row[nvtx_columns[0]])]
        if markers.count(['<default domain>', receipt['nvtx_range']]) != 1:
            raise ValueError('NVTX range mismatch or ambiguous attribution')
        if row['Kernel Name'] != kernel_name or int(row['Process ID']) != pid:
            raise ValueError('kernel/PID attribution mismatch')
        pci_values = [_number(row[k], number_format == 'decimal_comma') for k in _PCI]
        if any(v != v.to_integral_value() for v in pci_values):
            raise ValueError('integer PCI identifiers required')
        device = dict(index=int(row['Device']), name=row['device__attribute_display_name'],
                      cc=row['CC'], pci=[int(v) for v in pci_values])
        if device['index'] != receipt['device'] or device['name'] != receipt['gpu']:
            raise ValueError('device attribution mismatch')
        if 'capability' in receipt and device['cc'] != '.'.join(map(str, receipt['capability'])):
            raise ValueError('compute capability mismatch')
        if 'pci_bus' in receipt:
            match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])', receipt['pci_bus'])
            if not match or device['pci'] != [int(v, 16) for v in match.groups()[:3]]:
                raise ValueError('PCI identity mismatch')
        devices.append(device)
        counters = {}
        for name, unit in dict(_UNITS, **{k: 'hz' for k in _CLOCKS if k in header}).items():
            if name not in row or units[name] != unit:
                raise ValueError('missing counter or wrong base unit: ' + name)
            number = _number(row[name], number_format == 'decimal_comma')
            if name in _UNITS and number != number.to_integral_value():
                raise ValueError('integer native counter required')
            counters[name] = dict(value=int(number) if name in _UNITS else float(number), unit=unit)
        samples.append(dict(action_id=int(row['ID']), pid=pid, context=int(row['Context']), stream=int(row['Stream']),
                            kernel=kernel_name, nvtx=receipt['nvtx_range'], counters=counters))
    if any(d != devices[0] for d in devices):
        raise ValueError('multiple devices in one session')
    files = dict(csv=_file(csv_path), receipt=_file(receipt_path))
    expected_sources = _source_expected(receipt)
    for name, path in (source_paths or {}).items():
        if name not in expected_sources or not expected_sources[name]:
            raise ValueError('source has no linked receipt checksum: ' + name)
        files[name] = _file(path)
        if files[name]['sha256'].lower() != expected_sources[name].lower():
            raise ValueError('source checksum mismatch: ' + name)
    if report_path is not None:
        files['native_report'] = _file(report_path)
    snapshots = []
    for phase, path in (sensor_paths or {}).items():
        gpu = ET.fromstring(Path(path).read_text(encoding='utf-8-sig')).find('gpu')
        if gpu is None or gpu.findtext('product_name') != devices[0]['name']:
            raise ValueError('sensor GPU identity mismatch')
        pci = gpu.findtext('pci/pci_bus_id', '')
        if [int(v, 16) for v in pci.split('.')[0].split(':')] != devices[0]['pci']:
            raise ValueError('sensor PCI identity mismatch')
        snapshots.append(dict(phase=phase, file=_file(path), uuid=gpu.findtext('uuid'),
            temperature=gpu.findtext('temperature/gpu_temp'), pstate=gpu.findtext('performance_state'),
            sm_clock=gpu.findtext('clocks/sm_clock'), memory_clock=gpu.findtext('clocks/mem_clock'),
            power=gpu.findtext('gpu_power_readings/instant_power_draw', gpu.findtext('power_readings/power_draw')),
            clock_reasons={x.tag: x.text for x in gpu.iter() if x.tag.startswith('clocks_event_reason_') and x.text and x.text.strip()}))
    session = dict(format='hyso.performance.ncu_session', schema_version=1, session_id=session_id, role=role,
        device=devices[0], precision='FP32_SIMT_FMA2', hierarchy='device_DRAM', kernel=kernel_name,
        receipt=receipt, samples=samples, files=files, snapshots=snapshots,
        export_scope='native_raw_counter_export; report fingerprint is not binary authentication')
    _validate_session(session)
    return session


def _validate_session(s):
    if s['format'] != 'hyso.performance.ncu_session' or type(s['schema_version']) is not int or s['schema_version'] != 1:
        raise ValueError('unsupported native session schema')
    if not isinstance(s['session_id'], str) or not s['session_id'].strip() or s['role'] not in ('target', 'compute', 'memory'):
        raise ValueError('explicit session identity and role required')
    if s['precision'] != 'FP32_SIMT_FMA2' or s['hierarchy'] != 'device_DRAM':
        raise ValueError('incompatible work/traffic definition')
    pid, count = _identity(s['receipt'])
    _integer(s['device']['index'])
    if len(s['device']['pci']) != 3:
        raise ValueError('PCI tuple required')
    for v in s['device']['pci']:
        _integer(v)
    if len(s['samples']) != count or s['device']['index'] != s['receipt']['device'] or s['device']['name'] != s['receipt']['gpu']:
        raise ValueError('session/receipt identity mismatch')
    if s['receipt'].get('kernel', s['kernel']) != s['kernel']:
        raise ValueError('receipt kernel mismatch')
    if 'capability' in s['receipt'] and s['device']['cc'] != '.'.join(map(str, s['receipt']['capability'])):
        raise ValueError('receipt capability mismatch')
    if 'pci_bus' in s['receipt']:
        parts = s['receipt']['pci_bus'].split('.')[0].split(':')
        if len(parts) != 3 or s['device']['pci'] != [int(v, 16) for v in parts]:
            raise ValueError('receipt PCI mismatch')
    for name, record in s['files'].items():
        if not re.fullmatch('[0-9a-fA-F]{64}', record['sha256']):
            raise ValueError('invalid evidence fingerprint')
        if name in ('worker', 'cuda', 'ptx') and record['sha256'].lower() != (_source_expected(s['receipt'])[name] or '').lower():
            raise ValueError('source fingerprint mismatch')
    if s['role'] == 'target' and count != 1:
        raise ValueError('one target kernel interval required')
    ids = []
    for sample in s['samples']:
        for name in ('action_id', 'context', 'stream'):
            _integer(sample[name])
        ids.append(sample['action_id'])
        if sample['pid'] != pid or sample['kernel'] != s['kernel'] or sample['nvtx'] != s['receipt']['nvtx_range']:
            raise ValueError('sample attribution mismatch')
        counters = sample['counters']
        if not set(_UNITS) <= set(counters) or set(counters) - set(_UNITS) - set(_CLOCKS):
            raise ValueError('missing or unknown counter')
        for name, unit in _UNITS.items():
            if counters[name]['unit'] != unit:
                raise ValueError('counter unit mismatch')
            _integer(counters[name]['value'])
        _positive(counters['gpu__time_duration.sum']['value'])
        _integer(counters['profiler__replayer_passes']['value'], 1)
        for name in _CLOCKS:
            if name in counters:
                if counters[name]['unit'] != 'hz':
                    raise ValueError('clock unit mismatch')
                _positive(counters[name]['value'])
        rates = _sample(sample)
        if s['role'] in ('target', 'compute'):
            _positive(rates['hardware_flops'])
        if s['role'] in ('target', 'memory'):
            _positive(rates['dram_bytes'])
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate kernel action')
    if len({x['uuid'] for x in s['snapshots'] if x['uuid']}) > 1:
        raise ValueError('multiple sensor devices')


def _sample(sample):
    c = {k: v['value'] for k, v in sample['counters'].items()}
    flops = sum(c[f'smsp__sass_thread_inst_executed_op_{name}_pred_on.sum'] * weight
                for name, weight in [('fadd', 1), ('fmul', 1), ('ffma', 2)])
    traffic = c['dram__bytes_read.sum'] + c['dram__bytes_write.sum']
    seconds = c['gpu__time_duration.sum'] / 1e9
    math_data = _math(flops, seconds, traffic if traffic else None)
    return dict(hardware_flops=flops, dram_bytes=traffic, duration_seconds=seconds,
        gflops_per_second=math_data['achieved_gflops_per_second'], gbytes_per_second=traffic / seconds / 1e9,
        flops_per_byte=math_data['arithmetic_intensity_flops_per_byte'])


def _stats(values):
    median = statistics.median(values)
    return dict(samples=list(values), count=len(values), observed_min=min(values), median=median,
        observed_max=max(values), range_over_median=(max(values)-min(values))/median if median else None)


def _summary(session):
    values = [_sample(s) for s in session['samples']]
    key = 'gflops_per_second' if session['role'] == 'compute' else 'gbytes_per_second'
    return dict(session_id=session['session_id'], classification='empirical_session_observations',
        unit='GFLOP/s' if session['role'] == 'compute' else 'GB/s', **_stats([v[key] for v in values]),
        clocks={k: _stats([s['counters'][k]['value'] for s in session['samples']]) if all(k in s['counters'] for s in session['samples']) else None for k in _CLOCKS},
        snapshots=session['snapshots'], uncertainty='observed range only; not a confidence interval or hardware capacity bound')


def _derive(raw):
    target, computes, memory = raw['target'], raw['compute_sessions'], raw['memory_session']
    sessions = [target, *computes, *([memory] if memory is not None else [])]
    if (target['role'] != 'target' or (memory is not None and memory['role'] != 'memory')
            or any(s['role'] != 'compute' for s in computes)):
        raise ValueError('target, separate compute sessions and memory reference required')
    for s in sessions:
        _validate_session(s)
        if s['device'] != target['device']:
            raise ValueError('cross-device reference cannot substitute')
    ids = [s['session_id'] for s in sessions]
    if (len(ids) != len(set(ids)) or len({_hash(s['samples']) for s in sessions}) != len(sessions)
            or len({s['files']['csv']['sha256'] for s in sessions}) != len(sessions)):
        raise ValueError('duplicate or relabeled session evidence')
    if len({x['uuid'] for s in sessions for x in s['snapshots'] if x['uuid']}) > 1:
        raise ValueError('reference sensor device mismatch')
    point = _sample(target['samples'][0])
    partial = not computes or memory is None
    mem = _summary(memory) if memory is not None else dict(
        status='unsupported', value=None, reason='saved_device_DRAM_reference_not_supplied')
    refs = [_summary(s) for s in computes]
    models = []
    for ref in refs:
        if memory is None:
            break
        intensity = point['flops_per_byte']
        branch = ('DRAM_reference' if mem['observed_max'] * intensity < ref['observed_min'] else
                  'compute_reference' if mem['observed_min'] * intensity > ref['observed_max'] else 'overlapping_reference_bands')
        models.append(dict(session_id=ref['session_id'], branch_at_target=branch,
            at_target_observed_model_band_gflops=[min(ref['observed_min'], mem['observed_min'] * intensity),
                                                 min(ref['observed_max'], mem['observed_max'] * intensity)]))
    unsupported = dict(status='unsupported', value=None, reason='reference geometry is not causal performance or counterfactual speedup evidence')
    measurements = dict(status='partial' if partial else 'derived_with_observed_session_variability', target=dict(
        **point, status='derived_from_native_counters', flop_definition='FADD+FMUL+2*FFMA',
        analytic_work_reference=dict(status='analytic_reference_only', value=target['receipt'].get('logical_reference_flops'))),
        compute_references=refs, memory_reference=mem, reference_models=models,
        empirical_memory_max_gbytes_per_second=mem['observed_max'] if memory is not None else None,
        theoretical_maximum=unsupported, bottleneck=unsupported, hardware_headroom=unsupported, speedup=unsupported)
    return dict(measurements=measurements,
        provenance=dict(model=dict(kernel=target['kernel']), input=dict(dtype='float32', scope='single_saved_kernel_interval'),
                        environment=dict(device=target['device'], source='Nsight Compute raw CSV exports')),
        requested=dict(target=target['session_id'], compute_sessions=[s['session_id'] for s in computes],
                       memory_session=memory['session_id'] if memory is not None else None),
        actual=dict(target_kernels=1, compute_samples=[len(s['samples']) for s in computes],
                    memory_samples=len(memory['samples']) if memory is not None else 0),
        diagnostics=dict(warnings=['Target conditions are not assumed equal to reference sessions; inspect saved clocks/snapshots.']
            + (['Independent compute and device-DRAM references are required for a two-roof comparison.'] if partial else []),
            limitations=_LIMITATIONS))


def analyze_counter_roofline(target, *, compute_sessions=(), memory_session=None):
    """Offline empirical comparison, preserving every sample and session.

    No spread, traffic-ratio, minimum-duration policy or resource quota. Missing
    counters/units/attribution fail closed; variation is reported, never hidden.
    Omit unqualified references to retain measured target values in a partial
    result without manufacturing a two-roof comparison. Invalid supplied
    sessions still raise rather than silently disappearing from the evidence.
    """
    raw = dict(target=target, compute_sessions=list(compute_sessions), memory_session=memory_session)
    derived = _derive(raw)
    result = _base('roofline', 'P18', _PROTOCOL, derived['provenance'])
    result.update(derived, raw_evidence=raw)
    return RooflineResult(result)


def _validate_counter_result(data):
    if data['protocol'] != _PROTOCOL or set(data['raw_evidence']) != {'target', 'compute_sessions', 'memory_session'}:
        raise ValueError('invalid native Roofline protocol/evidence')
    if data['identity'] != dict(experiment_run_id=None, source_result_id=None):
        raise ValueError('native Roofline identity mismatch')
    expected = _derive(data['raw_evidence'])
    if any(data[key] != value for key, value in expected.items()):
        raise ValueError('native Roofline derived evidence mismatch')


def plot_counter_roofline(result, path, *, overwrite=False):
    """Save a CPU-only PNG/SVG; observed bands, separate sessions, no speedup claim."""
    if type(result) is not RooflineResult:
        raise TypeError('RooflineResult required')
    data = RooflineResult(result.to_dict()).to_dict()
    if data['protocol']['id'] != _PROTOCOL['id']:
        raise ValueError('native counter result required')
    if data['measurements']['status'] == 'partial':
        raise ValueError('independent compute and device-DRAM references required for a Roofline plot')
    path = Path(path)
    if path.suffix.lower() not in ('.png', '.svg'):
        raise ValueError('choose .png or .svg')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    import matplotlib.pyplot as plt
    m = data['measurements']
    point, mem = m['target'], m['memory_reference']
    refs = m['compute_references']
    lo = min(point['flops_per_byte'] / 8, min(r['observed_min'] / mem['observed_max'] for r in refs) / 8)
    hi = max(point['flops_per_byte'] * 8, max(r['observed_max'] / mem['observed_min'] for r in refs) * 8)
    xs = [math.exp(math.log(lo) + i * (math.log(hi)-math.log(lo)) / 199) for i in range(200)]
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        for i, ref in enumerate(refs):
            color = f'C{i % 10}'
            low = [min(ref['observed_min'], mem['observed_min'] * x) for x in xs]
            high = [min(ref['observed_max'], mem['observed_max'] * x) for x in xs]
            ax.fill_between(xs, low, high, color=color, alpha=.15)
            ax.loglog(xs, [min(ref['median'], mem['observed_max'] * x) for x in xs], color=color,
                label=f"{ref['session_id']} (n={ref['count']}): {ref['observed_min']:.1f}–{ref['observed_max']:.1f} GFLOP/s")
        ax.loglog(xs, [mem['observed_max'] * x for x in xs], '--', color='0.5', alpha=.7,
            label=f"DRAM {mem['session_id']} (n={mem['count']}): {mem['observed_min']:.3f}–{mem['observed_max']:.3f} GB/s; line=max")
        ax.scatter([point['flops_per_byte']], [point['gflops_per_second']], s=85, color='#b64721', zorder=5,
                   label=f"Target: {point['flops_per_byte']:.6f} FLOP/byte, {point['gflops_per_second']:.6f} GFLOP/s")
        ax.set(xlim=(lo, hi), ylim=(point['gflops_per_second']/3, max(r['observed_max'] for r in refs)*2),
            xlabel='Arithmetic intensity (FP32 FMA=2 FLOP / observed DRAM byte)', ylabel='Performance (GFLOP/s)',
            title='Empirical Roofline references — separate measurement sessions')
        ax.grid(which='major', alpha=.2)
        ax.legend(loc='lower right', fontsize=8)
        fig.subplots_adjust(bottom=.24, left=.10, right=.96, top=.90)
        fig.text(.10, .14, 'Bands = observed sample ranges, not confidence intervals or hardware capacity bounds.', fontsize=10)
        fig.text(.10, .10, 'Lines use session compute medians and the observed DRAM maximum; all samples remain in the result.', fontsize=9)
        branches = {x['branch_at_target'].replace('_', ' ') for x in m['reference_models']}
        branch = next(iter(branches)) if len(branches) == 1 else 'varies by session'
        fig.text(.10, .06, f'Target model term: {branch}. Actual bottleneck, headroom and speedup: unsupported.', fontsize=10)
        buffer = io.BytesIO()
        fig.savefig(buffer, format=path.suffix[1:].lower(), dpi=170)
        with path.open('wb' if overwrite else 'xb') as stream:
            stream.write(buffer.getvalue())
    finally:
        plt.close(fig)
    return path
