"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('from pathlib import Path\n'
 'import tempfile\n'
 'from hyso.performance import ColdStartRecipe, BatchSearchProtocol, find_max_batch_size\n'
 '\n'
 "output_dir = Path(tempfile.mkdtemp(prefix='hyso-capacity-guide-'))\n"
 "recipe = ColdStartRecipe('torch.nn:Conv2d',\n"
 "    model_kwargs={'in_channels': 1, 'out_channels': 2, 'kernel_size': 1},\n"
 "    input_shape=(1, 1, 5, 7), device='cpu', precision='fp32', seed=3)\n"
 'protocol = BatchSearchProtocol(candidates=(1, 2, 3), maximum_batch=3,\n'
 '    cpu_memory_limit_bytes=3 * 1024**3, input_limit_bytes=8 * 1024**2,\n'
 '    timeout_seconds=45, warmup_iterations=1, measure_iterations=3, repeats=1,\n'
 "    stop_policy='exhaust_candidates')\n"
 'capacity = find_max_batch_size(recipe, protocol=protocol)\n'
 'data = capacity.to_dict()\n'
 "assert data['measurements']['status'] == 'complete', data\n"
 "assert data['measurements']['maximum_proven_within_range'] == 3\n"
 "capacity.to_json(output_dir / 'capacity.json')\n"
 "capacity.to_csv(output_dir / 'capacity.csv')\n",
 'from hyso.experiments import ExperimentManager\n'
 "root = output_dir / 'runs'\n"
 'manager = ExperimentManager(root)\n'
 "experiment = manager.create_experiment('capacity')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(root)\n'
 'manager = ExperimentManager(root)\n'
 "run = manager.create_run(experiment['id'], 'benchmark', config={'feature': 'P17'},\n"
 "    provenance={'capacity_result_id': data['result_id']})\n"
 'run.start()\n'
 'artifact_ids = []\n'
 "for filename in ['capacity.json', 'capacity.csv']:\n"
 "    artifact = run.add_artifact(output_dir / filename, kind='performance_result', "
 "ownership='copy')\n"
 "    artifact_ids.append(artifact['id'])\n"
 "for index, row in enumerate(data['raw_evidence']['candidates']):\n"
 "    if row['status'] != 'feasible':\n"
 '        continue\n'
 "    for name, value, unit in [('latency', row['latency_mean_ms'], 'ms/batch'),\n"
 "        ('rss_after', row['memory']['rss_after']['rss_bytes'], 'byte')]:\n"
 "        run.log_metric(name='capacity.' + name, value=value, unit=unit, step=index,\n"
 "            step_kind='observation', source='performance.capacity_v1')\n"
 'run.finish()\n'
 'run_id = run.id\n'
 'manager.close()\n'
 'manager = ExperimentManager(root)\n'
 "assert manager.get_run(run_id).to_dict()['status'] == 'completed'\n"
 'assert len(manager.get_run(run_id).read_metrics()) == 6\n'
 "assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in "
 'artifact_ids)\n'
 'manager.close()\n')
