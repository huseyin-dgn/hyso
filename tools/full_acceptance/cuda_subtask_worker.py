"""Isolated CUDA submeasurement executor for the 23 declared 04.cuda tasks.

Never calls the legacy full inference() path. The supervisor, not this module,
owns Job admission, device visibility, output directory and deadlines.
"""
from pathlib import Path
import copy
import hashlib
import math
import threading

from .cuda_tasks import TASK_BY_ID


def _check_route(route):
    task = TASK_BY_ID.get(route.task_id)
    if task is None or route.arm != task.arm or route.measurement != task.measurement:
        raise ValueError('invalid CUDA submeasurement route')
    if getattr(route, 'operation', None) != 'inference_cuda_subtask':
        raise PermissionError('unexpected CUDA operation')
    return task


def _transfer_flags(task_id):
    if task_id == '04.cuda.shared.transfer_pageable_blocking':
        return False, False
    if task_id == '04.cuda.shared.transfer_pinned_blocking':
        return True, False
    if task_id == '04.cuda.shared.transfer_pinned_nonblocking':
        return True, True
    raise ValueError('unknown transfer variant')


def _run_measurement(task, model, x, y, sample, p, manifest):
    """Dispatch exactly one native measurement, returning named evidence objects.

    Import dependencies lazily so module loading and offline routing tests use no GPU.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from hyso.performance import (
        analyze_latency, measure_throughput, ThroughputProtocol,
        measure_utilization, UtilizationProtocol, measure_footprint,
        run_sweep, SweepProtocol, PipelineSample, PipelineStage,
        PipelineProtocol, measure_pipeline, analyze_stages,
    )
    from hyso import evaluate_classification, EvaluationCohort, evaluate_precision_quality
    from .handlers import inference_benchmark

    arm = task.arm
    operation = task.measurement
    device = 'cuda:0'
    if operation == 'latency':
        bench = inference_benchmark(model, sample, p, device)
        return {'benchmark': bench, 'latency': analyze_latency(bench, quantiles=(.5, .95, .99))}
    if operation == 'throughput':
        observed = copy.deepcopy(model).to(device)
        return {'throughput': measure_throughput(
            observed, sample.to(device),
            protocol=ThroughputProtocol(device=device, warmup_iterations=p['warmup'],
                                        batch_count=p['completed_batches'], repeats=p['repeats']))}
    if operation == 'utilization':
        observed = copy.deepcopy(model).to(device)
        observed_x = sample.to(device)
        running = threading.Event()
        errors = []
        completed = [0]
        def drive():
            try:
                while not running.is_set():
                    with torch.inference_mode():
                        output = observed(observed_x)
                    torch.cuda.synchronize(0)
                    if not torch.isfinite(output).all().item():
                        raise ValueError('utilization workload output invalid')
                    completed[0] += 1
            except BaseException as exc:
                errors.append(exc)
                running.set()
        thread = threading.Thread(target=drive)
        thread.start()
        try:
            util = measure_utilization(protocol=UtilizationProtocol(
                duration_seconds=2., interval_seconds=.1, gpu=True,
                gpu_selector=manifest['hardware']['gpu_uuid'],
                workload_label=manifest['assets']['models'][arm]['model_id']))
        finally:
            running.set()
            thread.join(60)
        if thread.is_alive():
            raise RuntimeError('utilization worker thread did not stop')
        if errors:
            raise errors[0]
        return {'utilization': util, 'utilization-work': dict(
            completed_calls=completed[0],
            scope='same_process_independent_driver_thread;rate_not_substituted_for_P02;device_busy_not_model_exclusive')}
    if operation == 'quality':
        value = evaluate_classification(
            model, DataLoader(TensorDataset(x, y), batch_size=p['quality_batch'], shuffle=False),
            class_names=[str(i) for i in range(10)], split='test', device=device,
            source_id='full_acceptance:' + manifest['run_id'] + ':' + arm,
            capture_evidence=False)
        return {'quality': value}
    if operation == 'precision_quality':
        cohort = EvaluationCohort(
            x, y, sample_ids=[f'mnist:test:{i:05}' for i in range(len(x))],
            class_names=[str(i) for i in range(10)], split='test',
            preprocessing_id=manifest['assets']['dataset']['preprocessing'])
        return {'quality-' + precision: evaluate_precision_quality(
            model, cohort, candidate=precision, device=device,
            batch_size=p['quality_batch'])
            for precision in p['precisions_cuda'] if precision != 'fp32'}
    if operation == 'footprint':
        asset = manifest['assets']['models'][arm]
        return {'footprint': measure_footprint(
            model, artifact_path=asset['path'],
            artifact_format='full_training_checkpoint_not_weight_only',
            expected_sha256=asset['sha256'])}
    if operation in ('batch_sweep', 'resolution_sweep', 'precision_sweep'):
        def factory(shape, dtype, dev):
            images = (sample[:shape[0]] if shape[0] <= len(sample) else
                      sample.repeat((math.ceil(shape[0] / len(sample)), 1, 1, 1))[:shape[0]])
            if tuple(images.shape[2:]) != tuple(shape[2:]):
                images = torch.nn.functional.interpolate(
                    images, size=tuple(shape[2:]), mode='bilinear', align_corners=False)
            return images.to(dtype=dtype, device=dev)
        protocol = SweepProtocol(device=device, warmup_iterations=p['warmup'],
                                 measure_iterations=20, repeats=1, sustained_batches=10)
        if operation == 'batch_sweep':
            axis, points, shape = 'batch', p['sweep_batches'], tuple(sample.shape)
        elif operation == 'resolution_sweep':
            axis, points, shape = 'resolution', [tuple(v) for v in p['resolution_points']], (1, 1, 28, 28)
        else:
            axis, points, shape = 'precision', p['precisions_cuda'], tuple(sample.shape)
        return {axis + '-sweep': run_sweep(model, factory, axis=axis, points=points,
                                           protocol=protocol, input_shape=shape)}
    if operation == 'pipeline':
        ids = iter(range(1000))
        def source():
            i = next(ids)
            return PipelineSample(str(i), x[i:i+1].clone())
        def post(v):
            return PipelineSample(v.sample_id, v.payload.argmax(1).float().reshape(1,1,1,1))
        transfer = PipelineStage('transfer', lambda v: PipelineSample(
            v.sample_id, v.payload.to('cuda:0')), 'cuda')
        observed = copy.deepcopy(model).to(device)
        pipeline = measure_pipeline(
            observed, source, transfer=transfer,
            postprocess=PipelineStage('postprocess', post, 'cuda'),
            protocol=PipelineProtocol(device=device, samples=40, repeats=1, warmup_samples=1))
        return {'pipeline': pipeline, 'stages': analyze_stages(pipeline)}
    raise ValueError('unknown CUDA measurement')


def run(manifest, stage_output, route):
    """One task only. Called after parent+child Windows Job admission.

    This alone does not establish native acceptance or authorize a run.
    """
    task = _check_route(route)
    root = Path(stage_output)
    if root.resolve() != (Path(manifest['output']) / 'cuda-subtasks' / task.task_id).resolve():
        raise ValueError('subtask evidence path mismatch')

    import torch
    from hyso.performance.devices import _cuda_hardware
    from hyso._evaluation_evidence import model_hash
    from .handlers import save, result
    from .workloads import configure, load_model, data
    configure(manifest)
    if not manifest['hardware']['gpu_uuid'] or not torch.cuda.is_available():
        return result('UNSUPPORTED', reason='native_CUDA_device_unavailable',
                      criteria=dict(single_measurement_executed=False))
    torch.cuda.set_device(0)
    p = manifest['protocols']['inference']
    evidence = [save(root, 'physical-GPU-runtime-identity',
                     _cuda_hardware(torch, manifest['hardware']['gpu_uuid']))]
    if task.arm == 'shared':
        from hyso.performance import measure_transfers, TransferProtocol
        pinned, nonblocking = _transfer_flags(task.task_id)
        value = measure_transfers(protocol=TransferProtocol(
            shape=(p['batch'], 1, 28, 28), warmup_iterations=p['warmup'],
            measure_iterations=20, repeats=1, pinned=pinned,
            non_blocking=nonblocking, timeout_seconds=120,
            cpu_memory_limit_bytes=None, input_limit_bytes=None,
            cuda_memory_fraction=None))
        evidence.append(save(root, 'transfer-' + str(pinned) + '-' + str(nonblocking), value))
        return result(criteria=dict(single_measurement_executed=True,
                                    isolated_transfer_variant=True), evidence=evidence,
                      task_id=task.task_id, device='cuda:0')

    # A fresh subprocess owns a fresh CPU master. Never mutate its checkpoint.
    model = load_model(manifest, task.arm)
    state = model_hash(model)
    x, y = data(manifest)
    sample = x[:p['batch']].clone()
    identity = dict(checkpoint=manifest['assets']['models'][task.arm],
                    device='cuda:0', gpu_uuid=manifest['hardware']['gpu_uuid'],
                    precision='fp32', input_shape=list(sample.shape),
                    input_sha256=hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
                    preprocessing=manifest['assets']['dataset']['preprocessing'],
                    state_sha256=state, quality_samples=len(x),
                    quality_batch=p['quality_batch'], protocol=p,
                    arm=task.arm, measurement=task.measurement)
    evidence.append(save(root, 'identity', identity))
    measured = _run_measurement(task, model, x, y, sample, p, manifest)
    if not measured:
        return result('UNSUPPORTED', reason='no_declared_precision_quality_variant',
                      evidence=evidence, task_id=task.task_id,
                      criteria=dict(single_measurement_executed=False))
    for label, value in measured.items():
        evidence.append(save(root, label, value))
    if model_hash(model) != state:
        raise ValueError('master model mutated')
    from .core import sha
    asset = manifest['assets']['models'][task.arm]
    if sha(asset['path']) != asset['sha256']:
        raise ValueError('checkpoint changed during submeasurement')
    if any(t.device.type != 'cpu' for t in [*model.parameters(), *model.buffers()]):
        raise ValueError('CPU master moved')
    return result(criteria=dict(single_measurement_executed=True,
                                source_model_state_preserved=True,
                                trained_model=True), evidence=evidence,
                  task_id=task.task_id, device='cuda:0')
