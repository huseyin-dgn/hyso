"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('from pathlib import Path\n'
 'import tempfile\n'
 'import torch\n'
 'from hyso.performance import SweepProtocol, SweepResult, PrecisionResult, run_sweep\n'
 '\n'
 'torch.set_num_threads(1)\n'
 "output_dir = Path(tempfile.mkdtemp(prefix='hyso-sweeps-'))\n"
 'model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(35, 8),\n'
 '                           torch.nn.ReLU(), torch.nn.Linear(8, 2))\n'
 'def inputs(shape, dtype, device):\n'
 '    return torch.randn(shape, dtype=dtype, device=device)\n'
 '\n'
 'plan = SweepProtocol(warmup_iterations=1, measure_iterations=3, repeats=1,\n'
 '                     sustained_batches=4)\n'
 "batches = run_sweep(model, inputs, axis='batch', points=[1, 2, 4, 8], protocol=plan)\n"
 'spatial_model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1),\n'
 '                                   torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())\n'
 "resolutions = run_sweep(spatial_model, inputs, axis='resolution',\n"
 '                       points=[(5, 7), (8, 8), (9, 13)], protocol=plan)\n'
 "precisions = run_sweep(model, inputs, axis='precision',\n"
 "                      points=['fp32', 'fp64', 'bf16', 'int8_dynamic'], protocol=plan)\n"
 "assert batches.to_dict()['actual']['measured'] == 4\n"
 "assert resolutions.to_dict()['actual']['measured'] == 3\n"
 "for name, result in [('batch', batches), ('resolution', resolutions), ('precision', "
 'precisions)]:\n'
 "    result.to_json(output_dir / (name + '.json'))\n"
 "    result.to_csv(output_dir / (name + '.csv'))\n"
 '    assert isinstance(SweepResult(result.to_dict()), SweepResult)\n'
 "    for point in result.to_dict()['raw_evidence']['points']:\n"
 "        if point['status'] == 'measured':\n"
 "            measurement = PrecisionResult(point['result'])\n"
 "            measurement.to_json(output_dir / (point['point_id'] + '.json'))\n"
 "            measurement.to_csv(output_dir / (point['point_id'] + '.csv'))\n",
 'def cuda_example():\n'
 '    if not torch.cuda.is_available():\n'
 "        raise RuntimeError('Native CUDA required')\n"
 '    stream = torch.cuda.Stream(device=0)\n'
 '    with torch.cuda.stream(stream):\n'
 "        return run_sweep(model, inputs, axis='precision',\n"
 "            points=['fp32', 'amp_fp16', 'bf16', 'int8_dynamic'],\n"
 "            protocol=SweepProtocol(device='cuda:0', measure_iterations=5))\n",
 'from hyso.experiments import ExperimentManager\n'
 "root = output_dir / 'runs'\n"
 'manager = ExperimentManager(root)\n'
 "experiment = manager.create_experiment('sweeps')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(root)\n'
 'manager = ExperimentManager(root)\n'
 "run = manager.create_run(experiment['id'], 'benchmark', config={'axis': 'batch'},\n"
 "                        provenance={'sweep_result_id': batches.to_dict()['result_id']})\n"
 'run.start()\n'
 'artifact_ids = []\n'
 "for filename in ['batch.json', 'batch.csv']:\n"
 "    record = run.add_artifact(output_dir / filename, kind='performance_result', "
 "ownership='copy')\n"
 "    artifact_ids.append(record['id'])\n"
 "for point in batches.to_dict()['raw_evidence']['points']:\n"
 "    if point['status'] != 'measured':\n"
 '        continue\n'
 "    record = run.add_artifact(output_dir / (point['point_id'] + '.json'),\n"
 "                             kind='performance_result', ownership='copy')\n"
 "    artifact_ids.append(record['id'])\n"
 "    value = point['result']['measurements']['latency_mean_ms']['value']\n"
 '    if value is not None:\n'
 "        run.log_metric(name='perf.point.latency', value=value, unit='ms/batch',\n"
 "            step=point['observation_index'], step_kind='observation', "
 "source='performance.sweep_v1')\n"
 'run.finish()\n'
 'run_id = run.id\n'
 'manager.close()\n'
 'manager = ExperimentManager(root)\n'
 "assert manager.get_run(run_id).to_dict()['status'] == 'completed'\n"
 'assert len(manager.get_run(run_id).read_metrics()) == 4\n'
 "assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in "
 'artifact_ids)\n'
 'manager.close()\n')
