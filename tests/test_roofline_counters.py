"""Synthetic counter protocol and CPU plotting only; no captured private data."""
import copy
import csv
import hashlib
import json
import subprocess

import pytest
import torch

from hyso.performance import (RooflineResult, read_ncu_counters, analyze_counter_roofline,
                              plot_counter_roofline)

NVTX = 'thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'
METRICS = {
    'gpu__time_duration.sum': 'ns', 'dram__bytes_read.sum': 'byte', 'dram__bytes_write.sum': 'byte',
    'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 'inst',
    'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 'inst',
    'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 'inst', 'profiler__replayer_passes': 'pass',
    'sm__cycles_elapsed.avg.per_second': 'hz',
}


def evidence(tmp_path, role, sid, durations, *, comma=False, mutate=None, wrong_unit=False, sensor=None):
    pid = {'target': 11, 'compute': 12, 'memory': 13}[role]
    kernel, nvtx = 'synthetic_' + role, 'range_' + role
    worker = tmp_path / (sid + '.py')
    worker.write_text('# synthetic receipt producer\n')
    receipt = dict(pid=pid, device=0, gpu='Synthetic GPU', dtype='float32', capability=[8, 0],
        pci_bus='0000:01:00.0', kernel=kernel, nvtx_range=nvtx, targeted_launches=len(durations),
        samples=[dict(index=i, output_equal=True, terminal_completed=True) for i in range(len(durations))],
        worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(), logical_reference_flops=4)
    header = ['ID', 'Process ID', NVTX, 'Kernel Name', 'Context', 'Stream', 'Device', 'CC',
        'device__attribute_display_name', 'device__attribute_pci_domain_id',
        'device__attribute_pci_bus_id', 'device__attribute_pci_device_id', *METRICS]
    rows = []
    for i, duration in enumerate(durations):
        row = dict(zip(header[:12], [str(i), str(pid), f'7 "<default domain>:{nvtx}:none:none"',
            kernel, '1', '2', '0', '8.0', 'Synthetic GPU', '0', '1', '0']))
        vals = [duration, 1600 if role == 'memory' else 64, 800 if role == 'memory' else 16,
                10 if role == 'compute' else 0, 0, 1000 if role == 'compute' else 4 if role == 'target' else 0,
                3, 1_000_000_000]
        row.update({k: f'{v:,}'.replace(',', '.') if comma else str(v) for k, v in zip(METRICS, vals)})
        rows.append(row)
    if mutate:
        mutate(receipt, rows)
    csv_path, receipt_path = tmp_path / (sid + '.csv'), tmp_path / (sid + '.json')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        units = {k: METRICS.get(k, '') for k in header}
        if wrong_unit:
            units['gpu__time_duration.sum'] = 'us'
        writer.writerow(units)
        writer.writerows(rows)
    receipt_path.write_text(json.dumps(receipt))
    return read_ncu_counters(csv_path, receipt_path=receipt_path, session_id=sid, role=role,
        kernel_name=kernel, number_format='decimal_comma' if comma else 'decimal_point', source_paths={'worker': worker},
        sensor_paths={'snapshot': sensor} if sensor else None)


def result(tmp_path):
    target = evidence(tmp_path, 'target', 'target', [4])
    c1 = evidence(tmp_path, 'compute', 'three', [10, 20, 30], comma=True)
    c2 = evidence(tmp_path, 'compute', 'five', [5, 8, 10, 12, 15])
    memory = evidence(tmp_path, 'memory', 'memory', [10, 12, 9])
    return analyze_counter_roofline(target, compute_sessions=[c1, c2], memory_session=memory)


def test_native_math_session_variability_and_no_hardware_gates(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('offline analysis must not execute GPU or launch processes')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    r = result(tmp_path).to_dict()
    m = r['measurements']
    # Four FFMA thread instructions = 8 hardware FLOPs; analytic work stays 4.
    assert m['target']['hardware_flops'] == 8
    assert m['target']['analytic_work_reference']['value'] == 4
    assert m['target']['flops_per_byte'] == pytest.approx(.1)
    assert m['target']['gflops_per_second'] == pytest.approx(2)
    assert m['memory_reference']['samples'] == pytest.approx([240, 200, 2400/9])
    c1, c2 = m['compute_references']
    assert c1['count'] == 3 and c2['count'] == 5
    assert c1['samples'] == pytest.approx([201, 100.5, 67])
    assert c1['range_over_median'] > .10  # descriptive, not rejection
    assert c2['observed_max'] == pytest.approx(402)
    assert r['actual']['compute_samples'] == [3, 5]  # never pooled
    assert all(x['branch_at_target'] == 'DRAM_reference' for x in m['reference_models'])
    assert all(m[k]['status'] == 'unsupported' and m[k]['value'] is None
               for k in ['theoretical_maximum', 'bottleneck', 'hardware_headroom', 'speedup'])
    # 4–30 ns samples intentionally below former 20 us policy: protocol validity is independent.
    assert m['target']['duration_seconds'] == pytest.approx(4e-9)


@pytest.mark.parametrize('fault', ['pid', 'kernel', 'nvtx', 'device', 'fractional_pci', 'completion', 'sample_count', 'counter', 'source'])
def test_wrong_attribution_missing_and_incorrect_evidence_rejected(tmp_path, fault):
    def mutate(receipt, rows):
        if fault == 'pid': rows[0]['Process ID'] = '99'
        elif fault == 'kernel': rows[0]['Kernel Name'] = 'unrelated'
        elif fault == 'nvtx': rows[0][NVTX] = '7 "<default domain>:other:none:none"'
        elif fault == 'device': rows[0]['device__attribute_pci_bus_id'] = '2'
        elif fault == 'fractional_pci': rows[0]['device__attribute_pci_bus_id'] = '1.5'
        elif fault == 'completion': receipt['samples'][0]['terminal_completed'] = False
        elif fault == 'sample_count': receipt['targeted_launches'] = 2
        elif fault == 'counter': rows[0]['dram__bytes_read.sum'] = ''
        else: receipt['worker_sha256'] = '0' * 64
    messages = dict(pid='attribution', kernel='attribution', nvtx='NVTX', device='PCI', fractional_pci='PCI',
                    completion='incomplete', sample_count='count', counter='number', source='checksum')
    with pytest.raises(ValueError, match=messages[fault]):
        evidence(tmp_path, 'target', fault, [5], mutate=mutate)


def test_base_units_and_explicit_locale(tmp_path):
    with pytest.raises(ValueError, match='unit'):
        evidence(tmp_path, 'target', 'units', [5], wrong_unit=True)
    s = evidence(tmp_path, 'compute', 'locale', [1234], comma=True)
    assert s['samples'][0]['counters']['gpu__time_duration.sum']['value'] == 1234
    with pytest.raises(ValueError):
        read_ncu_counters(tmp_path/'locale.csv', receipt_path=tmp_path/'locale.json', session_id='bad',
                          role='compute', kernel_name='synthetic_compute', number_format='decimal_point')


def test_cross_device_and_duplicate_session_rejected(tmp_path):
    r = result(tmp_path).to_dict()['raw_evidence']
    c1, c2 = r['compute_sessions']
    with pytest.raises(ValueError, match='duplicate'):
        analyze_counter_roofline(r['target'], compute_sessions=[c1, c1], memory_session=r['memory_session'])
    c2 = copy.deepcopy(c2)
    c2['device']['name'] = c2['receipt']['gpu'] = 'Another GPU'
    with pytest.raises(ValueError, match='cross-device'):
        analyze_counter_roofline(r['target'], compute_sessions=[c1, c2], memory_session=r['memory_session'])


@pytest.mark.parametrize('fault', ['rate', 'bottleneck', 'identity', 'units', 'pci', 'source'])
def test_result_tampering_rejected(tmp_path, fault):
    r = result(tmp_path).to_dict()
    if fault == 'rate': r['measurements']['target']['gflops_per_second'] += 1
    elif fault == 'bottleneck': r['measurements']['bottleneck'].update(status='measured', value='memory_bound')
    elif fault == 'identity': r['raw_evidence']['target']['samples'][0]['pid'] += 1
    elif fault == 'units': r['raw_evidence']['target']['samples'][0]['counters']['gpu__time_duration.sum']['unit'] = 'ms'
    elif fault == 'pci': r['raw_evidence']['target']['device']['pci'][1] = 9
    else: r['raw_evidence']['target']['files']['worker']['sha256'] = '0'*64
    with pytest.raises(ValueError):
        RooflineResult(r)


def test_canonical_json_csv_and_plot_no_clobber(tmp_path):
    r = result(tmp_path)
    before = r.to_dict()
    r.to_json(tmp_path/'result.json')
    r.to_csv(tmp_path/'result.csv')
    assert RooflineResult.from_json(tmp_path/'result.json').to_dict() == before
    assert RooflineResult.from_csv(tmp_path/'result.csv').to_dict() == before
    with pytest.raises(FileExistsError): r.to_csv(tmp_path/'result.csv')
    r.to_dict()['measurements'].clear()
    assert r.to_dict() == before
    plot_counter_roofline(r, tmp_path/'plot.svg')
    text = (tmp_path/'plot.svg').read_text()
    assert 'three (n=3)' in text and 'five (n=5)' in text
    assert 'not confidence intervals' in text and 'unsupported' in text
    with pytest.raises(FileExistsError): plot_counter_roofline(r, tmp_path/'plot.svg')


def test_saved_sensor_scope_and_device_mismatch(tmp_path):
    sensor = tmp_path/'sensor.xml'
    text = ('<nvidia_smi_log><gpu><product_name>Synthetic GPU</product_name><uuid>fake-A</uuid>'
            '<pci><pci_bus_id>00000000:01:00.0</pci_bus_id></pci>'
            '<temperature><gpu_temp>40 C</gpu_temp></temperature><performance_state>P0</performance_state>'
            '</gpu></nvidia_smi_log>')
    sensor.write_text(text)
    s = evidence(tmp_path, 'memory', 'sensor', [10, 12], sensor=sensor)
    assert s['snapshots'][0]['temperature'] == '40 C'
    assert s['snapshots'][0]['uuid'] == 'fake-A'
    assert s['snapshots'][0]['power'] is None  # missing is not fabricated
    sensor.write_text(text.replace('01:00.0', '02:00.0'))
    with pytest.raises(ValueError, match='sensor PCI'):
        evidence(tmp_path, 'memory', 'wrong-sensor', [10, 12], sensor=sensor)


def test_reference_sensor_uuid_mismatch_rejected(tmp_path):
    raw = result(tmp_path).to_dict()['raw_evidence']
    raw['compute_sessions'][0]['snapshots'] = [dict(uuid='fake-A')]
    raw['memory_session']['snapshots'] = [dict(uuid='fake-B')]
    with pytest.raises(ValueError, match='sensor device'):
        analyze_counter_roofline(raw['target'], compute_sessions=raw['compute_sessions'], memory_session=raw['memory_session'])
