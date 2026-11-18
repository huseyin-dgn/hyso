"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('from pathlib import Path\n'
 'import tempfile\n'
 'import time\n'
 'import torch\n'
 'from hyso.performance import MemoryProtocol, measure_memory, measure_footprint\n'
 '\n'
 'torch.set_num_threads(1)\n'
 "output_dir = Path(tempfile.mkdtemp(prefix='hyso-memory-'))\n"
 'current = measure_memory()  # tek native OS observation; workload yok\n'
 '\n'
 'def workload():\n'
 '    tensor = torch.empty(2 * 1024 * 1024, dtype=torch.float32)\n'
 "    tensor.fill_(1.25)  # native CPU storage'u gerçekten touch et\n"
 "    time.sleep(.03)    # bu örnekte sampler'a gözlem fırsatı verir\n"
 '    del tensor\n'
 '\n'
 'memory = measure_memory(workload, protocol=MemoryProtocol(interval_seconds=.005, '
 'max_samples=1000))\n'
 'memory_data = memory.to_dict()\n'
 "assert memory_data['measurements']['current_rss_bytes']['status'] == 'measured'\n"
 "assert memory_data['actual']['observations'] >= 2\n"
 "assert memory_data['protocol']['lifetime_peak_reset'] is False\n"
 "memory.to_json(output_dir / 'memory.json')\n"
 "memory.to_csv(output_dir / 'memory.csv')\n",
 'model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))\n'
 'logical_only = measure_footprint(model)\n'
 "model_path = output_dir / 'model-state.pt'\n"
 'footprint = measure_footprint(model, serialize_to=model_path, verify_roundtrip=True)\n'
 'footprint_data = footprint.to_dict()\n'
 "artifact_info = footprint_data['raw_evidence']['artifact']\n"
 "assert artifact_info['size_bytes'] == model_path.stat().st_size\n"
 "assert artifact_info['format_validation'] == 'roundtrip_verified'\n"
 "assert footprint_data['measurements']['physical_memory_bytes']['value'] is None\n"
 "footprint.to_json(output_dir / 'footprint.json')\n"
 "footprint.to_csv(output_dir / 'footprint.csv')\n",
 'existing = measure_footprint(model, artifact_path=model_path,\n'
 "    artifact_format='pytorch.state_dict', expected_sha256=artifact_info['sha256'])\n"
 "assert existing.to_dict()['raw_evidence']['artifact']['model_match'] == 'not_verified'\n",
 'from hyso.experiments import ExperimentManager\n'
 '\n'
 "root = output_dir / 'runs'\n"
 'manager = ExperimentManager(root)\n'
 "experiment = manager.create_experiment('P10-P11 CPU memory')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(root)  # explicit offline schema2 artifact capability\n'
 'manager = ExperimentManager(root)\n'
 "run = manager.create_run(experiment['id'], 'benchmark',\n"
 "    config={'memory_protocol': memory_data['protocol']},\n"
 "    provenance={'memory_result_id': memory_data['result_id'],\n"
 "                'footprint_result_id': footprint_data['result_id'],\n"
 "                'model': footprint_data['provenance']['model']})\n"
 'run.start()\n'
 'artifact_ids = []\n'
 "for path, kind in [(output_dir / 'memory.json', 'performance_result'),\n"
 "                   (output_dir / 'footprint.json', 'performance_result'),\n"
 "                   (model_path, 'model_state_dict')]:\n"
 "    artifact = run.add_artifact(path, kind=kind, ownership='copy', role='output')\n"
 "    artifact_ids.append(artifact['id'])\n"
 "    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'\n"
 "for name, metric in [('perf.rss.sampled_peak', "
 "memory_data['measurements']['sampled_peak_rss_bytes']),\n"
 "                     ('perf.model.logical_bytes', "
 "footprint_data['measurements']['parameter_buffer_bytes']),\n"
 "                     ('perf.model.serialized_bytes', "
 "footprint_data['measurements']['serialized_artifact_bytes'])]:\n"
 "    if metric['value'] is not None:\n"
 "        run.log_metric(name=name, value=metric['value'], unit=metric['unit'],\n"
 "                       source='performance.memory_v1', step=0, step_kind='observation')\n"
 'run.finish()\n'
 'run_id = run.id\n'
 'manager.close()\n'
 'manager = ExperimentManager(root)\n'
 "assert manager.get_run(run_id).to_dict()['status'] == 'completed'\n"
 "assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in "
 'artifact_ids)\n'
 'manager.close()\n')
